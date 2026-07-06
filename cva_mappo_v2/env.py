from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from data.mission_generator import Mission
from envs.multi_satellite_env import MultiSatelliteEnv
from .allocator import CapacityAwareTaskAllocator
from .config import CVAMAPPOV2Config
from .scorer import CandidateScore, CandidateValueScorer


class CVAMAPPOV2Env(MultiSatelliteEnv):
    """Clean CVA-MAPPO v2 environment.

    It keeps the old low-level scheduling environment but replaces the legacy
    high-level owner/top-k logic with:

    1. state-aware satellite-task pair scoring;
    2. task-centered multi-candidate assignment under slot capacity;
    3. typed fixed-size local action slots;
    4. periodic and event-triggered candidate repair.
    """

    def __init__(
        self,
        *args,
        cva_config: Optional[CVAMAPPOV2Config] = None,
        **kwargs,
    ):
        self.v2_cfg = cva_config or CVAMAPPOV2Config()
        self.v2_cfg.validate()
        self.task_candidate_owners: Dict[int, List[str]] = {}
        self._slot_types: Dict[str, List[str]] = {}
        self._slot_scores: Dict[str, List[float]] = {}
        self._candidate_score_table: Dict[int, Dict[str, CandidateScore]] = {}
        self._v2_step_cache = {}
        self._slot_valid_sum = 0
        self._slot_filled_sum = 0
        self._slot_exposure_count = 0
        kwargs["candidate_action_top_k"] = self.v2_cfg.slots.total_slots
        kwargs["episode_assignment"] = True
        kwargs["assignment_replan_interval_s"] = self.v2_cfg.replan_interval_s
        kwargs["assignment_replan_horizon_s"] = self.v2_cfg.replan_horizon_s
        kwargs["assignment_replan_trigger"] = ",".join(self.v2_cfg.triggers)
        kwargs["release_before_deadline_s"] = self.v2_cfg.release_before_deadline_s
        kwargs["assignment_lock_window_s"] = self.v2_cfg.lock_window_s
        kwargs["assignment_max_switches_per_task"] = self.v2_cfg.max_switches_per_task
        super().__init__(*args, **kwargs)
        self.scorer = CandidateValueScorer(self.v2_cfg)
        self.allocator = CapacityAwareTaskAllocator(self.v2_cfg, self.agent_ids)

    def reset(self, *args, **kwargs):
        self.task_candidate_owners = {}
        self._slot_types = {}
        self._slot_scores = {}
        self._candidate_score_table = {}
        self._slot_valid_sum = 0
        self._slot_filled_sum = 0
        self._slot_exposure_count = 0
        self._clear_v2_step_cache()
        result = super().reset(*args, **kwargs)
        self._clear_v2_step_cache()
        return result

    def step(self, actions):
        self._clear_v2_step_cache()
        result = super().step(actions)
        self._clear_v2_step_cache()
        return result

    def _clear_v2_step_cache(self) -> None:
        self._v2_step_cache = {}

    # ------------------------------------------------------------------
    # Task-centered candidate assignment
    # ------------------------------------------------------------------
    def _assign_tasks(self, missions: list):
        pending = [
            m for m in missions
            if m is not None and not m.is_observed
        ]
        if not pending:
            return
        current_time = self._current_time_s()
        self._candidate_score_table = self._build_score_table(pending, current_time)
        stale = self._stale_task_ids()
        assignment = self.allocator.allocate(
            missions=pending,
            score_table=self._candidate_score_table,
            current_candidates=self.task_candidate_owners,
            current_primary=self.task_owner,
            current_load=self._candidate_load(),
            current_time_s=current_time,
            horizon_s=self.horizon_s,
            stale_tasks=stale,
        )
        self.task_candidate_owners.update(assignment.task_candidates)
        self.task_owner.update(assignment.primary_owner)
        self._refresh_assignment_load()
        self._clear_v2_step_cache()

    def _reassign_tasks(self, missions: list) -> int:
        current_time = self._current_time_s()
        self._candidate_score_table = self._build_score_table(missions, current_time)
        stale = self._stale_task_ids()
        assignment = self.allocator.allocate(
            missions=missions,
            score_table=self._candidate_score_table,
            current_candidates=self.task_candidate_owners,
            current_primary=self.task_owner,
            current_load=self._candidate_load(exclude_missions={m.id for m in missions}),
            current_time_s=current_time,
            horizon_s=self.horizon_s,
            stale_tasks=stale,
        )
        old_primary = dict(self.task_owner)
        self.task_candidate_owners.update(assignment.task_candidates)
        self.task_owner.update(assignment.primary_owner)
        switched = 0
        for mission in missions:
            old = old_primary.get(mission.id)
            new = self.task_owner.get(mission.id)
            if old is not None and new is not None and old != new:
                self._owner_switch_counts[mission.id] = self._owner_switch_counts.get(mission.id, 0) + 1
                switched += 1
        self._refresh_assignment_load()
        self._clear_v2_step_cache()
        return switched

    def set_task_owner(self, mission_id: int, agent_id: str, count_switch: bool = True) -> bool:
        changed = super().set_task_owner(mission_id, agent_id, count_switch=count_switch)
        if changed:
            self.task_candidate_owners[mission_id] = [agent_id]
            self._clear_v2_step_cache()
        return changed

    def _build_score_table(self, missions: List[Mission], current_time_s: float):
        raw_quality = {}
        for mission in missions:
            visible_agents = []
            for aid in self.agent_ids:
                q = self._task_quality_window(aid, mission)
                if q is not None:
                    visible_agents.append((aid, q))
            raw_quality[mission.id] = visible_agents

        score_table: Dict[int, Dict[str, CandidateScore]] = {}
        loads = self._candidate_load()
        max_load = max(max(loads.values(), default=0), 1)
        stale = self._stale_task_ids()
        for mission in missions:
            agent_scores = {}
            visible_agents = raw_quality.get(mission.id, [])
            for aid, _ in visible_agents:
                env = self.envs[aid]
                load_pressure = loads.get(aid, 0) / max_load
                score = self.scorer.score_pair(
                    env=env,
                    agent_id=aid,
                    mission=mission,
                    current_time_s=current_time_s,
                    load_pressure=load_pressure,
                    n_visible_agents=len(visible_agents),
                    n_agents=self.n_agents,
                    current_owner=self.task_owner.get(mission.id),
                    owner_stale=mission.id in stale,
                    allow_future=True,
                )
                if score is not None:
                    agent_scores[aid] = score
            if agent_scores:
                score_table[mission.id] = agent_scores
        return score_table

    def _candidate_load(self, exclude_missions: Optional[set] = None) -> Dict[str, int]:
        if not exclude_missions and "candidate_load" in self._v2_step_cache:
            return self._v2_step_cache["candidate_load"]
        exclude_missions = exclude_missions or set()
        load = {aid: len(self.envs[aid].schedule_log) for aid in self.agent_ids}
        for mission_id, owners in self.task_candidate_owners.items():
            if mission_id in exclude_missions:
                continue
            for aid in owners:
                if aid in load:
                    load[aid] += 1
        if not exclude_missions:
            self._v2_step_cache["candidate_load"] = load
        return load

    def _stale_task_ids(self) -> set:
        if "stale_task_ids" in self._v2_step_cache:
            return self._v2_step_cache["stale_task_ids"]
        stale = set()
        for mission_id, owner in self.task_owner.items():
            if owner is not None and not self._has_future_feasible_window(owner, mission_id):
                stale.add(mission_id)
        self._v2_step_cache["stale_task_ids"] = stale
        return stale

    def _mission_by_id(self, agent_id: str) -> Dict[int, Mission]:
        key = ("mission_by_id", agent_id)
        cached = self._v2_step_cache.get(key)
        if cached is not None:
            return cached
        lookup = {
            int(m.id): m
            for m in self.envs[agent_id].missions
            if m is not None
        }
        self._v2_step_cache[key] = lookup
        return lookup

    def _action_by_mission_id(self, agent_id: str) -> Dict[int, int]:
        key = ("action_by_mission_id", agent_id)
        cached = self._v2_step_cache.get(key)
        if cached is not None:
            return cached
        lookup = {
            int(m.id): int(action)
            for action, m in enumerate(self.envs[agent_id].missions[:self.max_action_dim])
            if m is not None
        }
        self._v2_step_cache[key] = lookup
        return lookup

    def _has_future_feasible_window(self, agent_id: str, mission_id: int) -> bool:
        env = self.envs[agent_id]
        mission = self._mission_by_id(agent_id).get(int(mission_id))
        if mission is None or mission.is_observed:
            return False
        from_t = max(env.current_time_s, mission.earliest_time_s)
        for vtw in env.mission_vtw.get(mission.id, []):
            obs_start = max(vtw.start_time, from_t)
            obs_end = obs_start + mission.duration_s
            if obs_end <= min(vtw.end_time, mission.deadline_s):
                return True
        return False

    def _current_time_s(self) -> float:
        first_env = list(self.envs.values())[0]
        return float(first_env.current_time_s)

    # ------------------------------------------------------------------
    # Candidate ownership masks and typed slots
    # ------------------------------------------------------------------
    def _apply_ownership_mask(self, agent_id: str, mask: np.ndarray) -> np.ndarray:
        env = self.envs[agent_id]
        current_time = float(env.current_time_s)
        stale = self._stale_task_ids()
        for i in range(self.max_action_dim):
            mission = env.missions[i]
            if mission is None:
                continue
            candidates = self.task_candidate_owners.get(mission.id)
            if (
                candidates
                and agent_id not in candidates
                and not self._ownership_released_with_context(agent_id, mission, current_time, stale)
            ):
                mask[i] = 0.0
        return mask

    def _ownership_released(self, agent_id: str, mission) -> bool:
        if mission.is_observed:
            return True
        candidates = self.task_candidate_owners.get(mission.id, [])
        if agent_id in candidates:
            return True
        owner = self.task_owner.get(mission.id)
        if owner is None:
            return False
        env = self.envs[agent_id]
        near_deadline = env.current_time_s >= mission.deadline_s - self.v2_cfg.release_before_deadline_s
        owner_stale = not self._has_future_feasible_window(owner, mission.id)
        released = near_deadline or owner_stale
        if released:
            self._released_mission_ids.add(mission.id)
            if near_deadline:
                self._deadline_release_mission_ids.add(mission.id)
        return released

    def _ownership_released_with_context(
        self,
        agent_id: str,
        mission,
        current_time: float,
        stale: set,
    ) -> bool:
        if mission.is_observed:
            return True
        candidates = self.task_candidate_owners.get(mission.id, [])
        if agent_id in candidates:
            return True
        owner = self.task_owner.get(mission.id)
        if owner is None:
            return False
        near_deadline = current_time >= mission.deadline_s - self.v2_cfg.release_before_deadline_s
        released = near_deadline or mission.id in stale
        if released:
            self._released_mission_ids.add(mission.id)
            if near_deadline:
                self._deadline_release_mission_ids.add(mission.id)
        return released

    def _select_candidate_actions(self, agent_id: str, full_mask: np.ndarray) -> List[Optional[int]]:
        routine, dynamic, flex = self._rank_all_slot_groups(agent_id, full_mask)

        selected: List[Optional[int]] = []
        slot_types: List[str] = []
        slot_scores: List[float] = []
        used = set()
        slot_type_counts = {"routine": 0, "dynamic": 0, "flex": 0}

        def take(group_items, n, slot_type):
            for _, score, action in group_items:
                if slot_type_counts[slot_type] >= n:
                    break
                if action in used:
                    continue
                selected.append(action)
                slot_types.append(slot_type)
                slot_scores.append(float(score))
                used.add(action)
                slot_type_counts[slot_type] += 1

        take(routine, self.v2_cfg.slots.routine_slots, "routine")
        take(dynamic, self.v2_cfg.slots.dynamic_slots, "dynamic")
        # Flex slots can use urgent dynamic, stale owner, or high-value routine tasks.
        flex_count = self.v2_cfg.slots.flex_slots
        for _, score, action in flex:
            if slot_type_counts["flex"] >= flex_count:
                break
            if action in used:
                continue
            selected.append(action)
            slot_types.append("flex")
            slot_scores.append(float(score))
            used.add(action)
            slot_type_counts["flex"] += 1

        while len(selected) < self.v2_cfg.slots.total_slots:
            selected.append(None)
            slot_types.append("empty")
            slot_scores.append(0.0)

        self._slot_types[agent_id] = slot_types
        self._slot_scores[agent_id] = slot_scores
        self._slot_valid_sum += sum(
            1
            for action in selected
            if action is not None and full_mask[action] > 0
        )
        self._slot_filled_sum += sum(1 for action in selected if action is not None)
        self._slot_exposure_count += 1
        return selected[:self.v2_cfg.slots.total_slots]

    def _rank_all_slot_groups(self, agent_id: str, full_mask: np.ndarray):
        """Rank all typed slot groups with a single pass over the task list.

        The first v2 implementation scanned all missions separately for
        routine, dynamic and flex slots.  At stress scale this multiplied the
        expensive pair scoring work by roughly three for every satellite and
        every environment step.
        """
        env = self.envs[agent_id]
        current_time = self._current_time_s()
        stale = self._stale_task_ids()
        loads = self._candidate_load()
        load_pressure = loads.get(agent_id, 0) / max(self.v2_cfg.slots.total_slots, 1)
        action_lookup = self._action_by_mission_id(agent_id)
        candidate_ids = {
            int(mid)
            for mid, owners in self.task_candidate_owners.items()
            if agent_id in owners
        }
        if candidate_ids:
            # Released urgent/stale tasks may be taken by non-candidate agents.
            for mid, owner in self.task_owner.items():
                if mid in candidate_ids:
                    continue
                mission = self._mission_by_id(agent_id).get(int(mid))
                if mission is None or mission.is_observed:
                    continue
                near_deadline = current_time >= mission.deadline_s - self.v2_cfg.release_before_deadline_s
                if near_deadline or mid in stale:
                    candidate_ids.add(int(mid))
            action_iter = [
                action_lookup[mid]
                for mid in candidate_ids
                if mid in action_lookup
            ]
        else:
            action_iter = range(self.max_action_dim)

        routine = []
        dynamic = []
        flex = []
        for action in action_iter:
            mission = env.missions[action]
            if mission is None or mission.is_observed:
                continue
            candidates = self.task_candidate_owners.get(mission.id)
            if (
                candidates
                and agent_id not in candidates
                and not self._ownership_released_with_context(agent_id, mission, current_time, stale)
            ):
                continue

            score = self._candidate_action_score_from_context(
                agent_id=agent_id,
                mission=mission,
                current_time=current_time,
                load_pressure=load_pressure,
                stale=stale,
                allow_future=True,
            )
            if score is None:
                continue
            is_available = 1 if full_mask[action] > 0 else 0
            item = (is_available, float(score), int(action))
            if not mission.is_dynamic:
                routine.append(item)
            else:
                dynamic.append(item)
            if self._belongs_to_flex_group(mission, current_time, stale):
                flex.append(item)

        # Valid actions must be exposed before future-only tasks.  Otherwise
        # high-value future tasks can fill all slots while the policy sees only
        # idle as executable, which severely depresses completion rate.
        routine.sort(key=lambda x: (x[0], x[1]), reverse=True)
        dynamic.sort(key=lambda x: (x[0], x[1]), reverse=True)
        flex.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return routine, dynamic, flex

    def _rank_slot_group(self, agent_id: str, full_mask: np.ndarray, group: str):
        env = self.envs[agent_id]
        items = []
        for action, mission in enumerate(env.missions[:self.max_action_dim]):
            if mission is None or mission.is_observed:
                continue
            candidates = self.task_candidate_owners.get(mission.id)
            if candidates and agent_id not in candidates and not self._ownership_released(agent_id, mission):
                continue
            is_currently_available = full_mask[action] > 0
            score = self._candidate_action_score(agent_id, action, allow_future=True)
            if score is None:
                continue
            if not self._belongs_to_group(mission, group):
                continue
            if is_currently_available:
                score += 0.25
            items.append((float(score), int(action)))
        items.sort(key=lambda x: x[0], reverse=True)
        return items

    def _belongs_to_group(self, mission: Mission, group: str) -> bool:
        if group == "routine":
            return not mission.is_dynamic
        if group == "dynamic":
            return mission.is_dynamic
        if group == "flex":
            current_time = self._current_time_s()
            return self._belongs_to_flex_group(mission, current_time, self._stale_task_ids())
        return False

    def _belongs_to_flex_group(self, mission: Mission, current_time: float, stale: set) -> bool:
        near_deadline = current_time >= mission.deadline_s - self.v2_cfg.release_before_deadline_s
        return mission.is_dynamic or near_deadline or mission.id in stale or mission.priority >= 7.5

    def _candidate_action_score(
        self,
        agent_id: str,
        action: int,
        allow_future: bool = False,
    ) -> Optional[float]:
        env = self.envs[agent_id]
        mission = env.missions[action]
        if mission is None or mission.is_observed:
            return None
        load_pressure = self._candidate_load().get(agent_id, 0) / max(self.v2_cfg.slots.total_slots, 1)
        return self._candidate_action_score_from_context(
            agent_id=agent_id,
            mission=mission,
            current_time=env.current_time_s,
            load_pressure=load_pressure,
            stale=self._stale_task_ids(),
            allow_future=allow_future,
        )

    def _candidate_action_score_from_context(
        self,
        agent_id: str,
        mission: Mission,
        current_time: float,
        load_pressure: float,
        stale: set,
        allow_future: bool = False,
    ) -> Optional[float]:
        env = self.envs[agent_id]
        score = self.scorer.score_pair(
            env=env,
            agent_id=agent_id,
            mission=mission,
            current_time_s=current_time,
            load_pressure=load_pressure,
            n_visible_agents=len(self.task_candidate_owners.get(mission.id, [])) or 1,
            n_agents=self.n_agents,
            current_owner=self.task_owner.get(mission.id),
            owner_stale=mission.id in stale,
            allow_future=allow_future,
        )
        return None if score is None else score.score

    def _expose_obs_info(self, agent_id: str, full_mask: Optional[np.ndarray] = None):
        obs, info = super()._expose_obs_info(agent_id, full_mask=full_mask)
        if self._candidate_actions_enabled():
            info = {
                **info,
                "slot_types": list(self._slot_types.get(agent_id, [])),
                "slot_scores": list(self._slot_scores.get(agent_id, [])),
                "n_task_candidate_owners": len(self.task_candidate_owners),
            }
        return obs, info

    def get_metrics(self):
        metrics = super().get_metrics()
        denom = max(self._slot_exposure_count * self.v2_cfg.slots.total_slots, 1)
        metrics.update({
            "slot_valid_ratio": self._slot_valid_sum / denom,
            "slot_filled_ratio": self._slot_filled_sum / denom,
            "avg_valid_slots": self._slot_valid_sum / max(self._slot_exposure_count, 1),
            "avg_filled_slots": self._slot_filled_sum / max(self._slot_exposure_count, 1),
        })
        return metrics
