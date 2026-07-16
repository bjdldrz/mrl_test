from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from data.mission_generator import Mission
from envs.multi_satellite_env import MultiSatelliteEnv
from .allocator import CapacityAwareTaskAllocator
from .config import CVAMAPPOV2Config
from .scorer import CandidateScore, CandidateValueScorer


SLOT_INVALID_REASONS = (
    "empty",
    "observed",
    "not_arrived",
    "deadline_missed",
    "storage_full_current",
    "future_vtw",
    "no_feasible_vtw",
    "transition_infeasible",
    "schedule_conflict",
    "storage_full",
    "ownership_masked",
    "other",
)


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
        self._slot_type_valid_sum = {"routine": 0, "dynamic": 0, "flex": 0}
        self._slot_type_filled_sum = {"routine": 0, "dynamic": 0, "flex": 0}
        self._slot_type_capacity_sum = {"routine": 0, "dynamic": 0, "flex": 0}
        self._slot_invalid_reason_sum = {reason: 0 for reason in SLOT_INVALID_REASONS}
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
        self._slot_type_valid_sum = {"routine": 0, "dynamic": 0, "flex": 0}
        self._slot_type_filled_sum = {"routine": 0, "dynamic": 0, "flex": 0}
        self._slot_type_capacity_sum = {"routine": 0, "dynamic": 0, "flex": 0}
        self._slot_invalid_reason_sum = {reason: 0 for reason in SLOT_INVALID_REASONS}
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
                    current_time_s=float(env.current_time_s),
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
        self._released_mission_ids.update(stale)
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
        mission = self._mission_for_agent(agent_id, mission_id)
        if mission is None or self._mission_observed_anywhere(mission_id):
            return False
        self._ensure_mission_vtw(agent_id, mission)
        from_t = max(env.current_time_s, mission.earliest_time_s)
        for vtw in env.mission_vtw.get(mission.id, []):
            obs_start = max(vtw.start_time, from_t)
            obs_end = obs_start + mission.duration_s
            if obs_end <= min(vtw.end_time, mission.deadline_s):
                return True
        return False

    def _current_time_s(self) -> float:
        return self._team_current_time_s()

    # ------------------------------------------------------------------
    # Candidate ownership masks and typed slots
    # ------------------------------------------------------------------
    def _hard_ownership_mask_enabled(self) -> bool:
        return self.v2_cfg.ownership_mask_mode == "hard"

    def _apply_ownership_mask(self, agent_id: str, mask: np.ndarray) -> np.ndarray:
        if not self._hard_ownership_mask_enabled():
            return mask
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
                and not (mask[i] > 0 and self._dynamic_broadcast_open(mission, current_time))
            ):
                mask[i] = 0.0
        return mask

    def _ownership_released(self, agent_id: str, mission) -> bool:
        if mission.is_observed:
            return True
        candidates = self.task_candidate_owners.get(mission.id, [])
        if agent_id in candidates:
            return True
        env = self.envs[agent_id]
        owner = self.task_owner.get(mission.id)
        if self._dynamic_broadcast_open(mission, env.current_time_s):
            return True
        if self._dynamic_takeover_release_open(agent_id, mission, env.current_time_s):
            return True
        if owner is None:
            return False
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
        if self._dynamic_broadcast_open(mission, current_time):
            return True
        owner = self.task_owner.get(mission.id)
        if owner is None:
            return False
        if self._dynamic_takeover_release_open(agent_id, mission, current_time):
            return True
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

        def append_item(slot_type, score, action):
            selected.append(action)
            slot_types.append(slot_type)
            slot_scores.append(float(score))
            used.add(action)

        if self.v2_cfg.slot_selection_mode == "mixed":
            mixed_items = []
            for item in routine:
                mixed_items.append((*item, "routine"))
            for item in dynamic:
                mixed_items.append((*item, "dynamic"))
            mixed_items.sort(key=lambda x: (x[0], x[1]), reverse=True)
            for _, score, action, slot_type in mixed_items:
                if len(selected) >= self.v2_cfg.slots.total_slots:
                    break
                if action in used:
                    continue
                append_item(slot_type, score, action)
        else:
            def take(group_items, n, slot_type):
                for _, score, action in group_items:
                    if slot_type_counts[slot_type] >= n:
                        break
                    if action in used:
                        continue
                    append_item(slot_type, score, action)
                    slot_type_counts[slot_type] += 1

            # Put dynamic candidates first, then give currently executable
            # flex tasks an early chance before routine future-only context
            # fills the middle of the action set. Remaining flex slots are
            # filled after routine quota so routine throughput is still visible.
            take(dynamic, self.v2_cfg.slots.dynamic_slots, "dynamic")
            flex_count = self.v2_cfg.slots.flex_slots
            flex_items = self._flex_slot_items(flex, dynamic, routine)
            early_flex_target = min(flex_count, max(1, flex_count // 2)) if flex_count > 0 else 0

            def take_flex(group_items, n, require_available=False):
                for is_available, score, action in group_items:
                    if slot_type_counts["flex"] >= n:
                        break
                    if require_available and not is_available:
                        continue
                    if action in used:
                        continue
                    append_item("flex", score, action)
                    slot_type_counts["flex"] += 1

            take_flex(flex_items, early_flex_target, require_available=True)
            take(routine, self.v2_cfg.slots.routine_slots, "routine")
            # If the urgent/current flex pool is small, backfill flex with the
            # best remaining dynamic/routine candidates as context.
            for _, score, action in flex_items:
                if slot_type_counts["flex"] >= flex_count:
                    break
                if action in used:
                    continue
                append_item("flex", score, action)
                slot_type_counts["flex"] += 1

        while len(selected) < self.v2_cfg.slots.total_slots:
            selected.append(None)
            slot_types.append("empty")
            slot_scores.append(0.0)

        self._slot_types[agent_id] = slot_types
        self._slot_scores[agent_id] = slot_scores
        for slot_type in ("routine", "dynamic", "flex"):
            if self.v2_cfg.slot_selection_mode == "mixed":
                capacity = self.v2_cfg.slots.total_slots
            else:
                capacity = int(getattr(self.v2_cfg.slots, f"{slot_type}_slots"))
            self._slot_type_capacity_sum[slot_type] += capacity
        self._record_slot_diagnostics(agent_id, selected, full_mask, slot_types)
        return selected[:self.v2_cfg.slots.total_slots]

    def _record_slot_diagnostics(
        self,
        agent_id: str,
        selected: List[Optional[int]],
        full_mask: np.ndarray,
        slot_types: List[str],
    ) -> None:
        self._slot_exposure_count += 1
        local_mask = None
        for slot_type, action in zip(slot_types, selected):
            if action is not None:
                self._slot_filled_sum += 1
            valid = (
                action is not None
                and 0 <= action < len(full_mask)
                and full_mask[action] > 0
            )
            if valid:
                self._slot_valid_sum += 1
            if slot_type in self._slot_type_valid_sum and action is not None:
                self._slot_type_filled_sum[slot_type] += 1
                if valid:
                    self._slot_type_valid_sum[slot_type] += 1
            if not valid:
                reason, local_mask = self._slot_invalid_reason(
                    agent_id=agent_id,
                    action=action,
                    full_mask=full_mask,
                    local_mask=local_mask,
                )
                self._slot_invalid_reason_sum[reason] = (
                    self._slot_invalid_reason_sum.get(reason, 0) + 1
                )

    def _slot_invalid_reason(
        self,
        agent_id: str,
        action: Optional[int],
        full_mask: np.ndarray,
        local_mask: Optional[np.ndarray],
    ):
        if action is None:
            return "empty", local_mask
        if action < 0 or action >= self.max_action_dim:
            return "other", local_mask
        env = self.envs[agent_id]
        mission = env.missions[action]
        if mission is None:
            return "empty", local_mask
        if mission.is_observed or self._mission_observed_anywhere(mission.id):
            return "observed", local_mask

        if local_mask is None:
            local_mask = env._build_action_mask()
        if action < len(local_mask) and local_mask[action] > 0 and full_mask[action] <= 0:
            return "ownership_masked", local_mask

        current_time = float(env.current_time_s)
        if getattr(env, "storage_limited", False) and not env._has_storage_capacity(current_time):
            return "storage_full_current", local_mask
        if mission.earliest_time_s > current_time:
            return "not_arrived", local_mask
        if current_time > mission.deadline_s:
            return "deadline_missed", local_mask

        usable_vtw = None
        for vtw in env.mission_vtw.get(mission.id, []):
            if vtw.start_time <= current_time <= vtw.end_time - mission.duration_s:
                usable_vtw = vtw
                break
        if usable_vtw is None:
            next_start = env._earliest_feasible_observation_start(mission, current_time)
            return ("future_vtw" if next_start is not None else "no_feasible_vtw"), local_mask

        obs_start = current_time
        last_obs_end = env.schedule_log[-1].obs_end_s if env.schedule_log else None
        if last_obs_end is not None:
            transition = env.propagator.compute_transition_time(
                env._last_off_nadir_deg,
                usable_vtw.off_nadir_deg,
            )
            obs_start = max(obs_start, float(last_obs_end) + float(transition))
            if obs_start > usable_vtw.end_time - mission.duration_s:
                return "transition_infeasible", local_mask
            if obs_start + mission.duration_s > mission.deadline_s:
                return "transition_infeasible", local_mask
        obs_end = obs_start + mission.duration_s

        if env._conflicts_with_schedule(obs_start, obs_end):
            return "schedule_conflict", local_mask
        if not env._has_storage_capacity(obs_start):
            return "storage_full", local_mask
        return "other", local_mask

    @staticmethod
    def _flex_slot_items(flex, dynamic, routine):
        items = []
        seen = set()

        def add(group_items, bonus=0.0):
            for is_available, score, action in group_items:
                if action in seen:
                    continue
                seen.add(action)
                items.append((is_available, float(score) + float(bonus), action))

        add(flex, bonus=0.08)
        add(dynamic, bonus=0.04)
        add(routine, bonus=0.0)
        items.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return items

    def _rank_all_slot_groups(self, agent_id: str, full_mask: np.ndarray):
        """Rank all typed slot groups with a single pass over the task list.

        The first v2 implementation scanned all missions separately for
        routine, dynamic and flex slots.  At stress scale this multiplied the
        expensive pair scoring work by roughly three for every satellite and
        every environment step.
        """
        env = self.envs[agent_id]
        current_time = float(env.current_time_s)
        stale = self._stale_task_ids()
        loads = self._candidate_load()
        load_pressure = loads.get(agent_id, 0) / max(self.v2_cfg.slots.total_slots, 1)
        action_lookup = self._action_by_mission_id(agent_id)
        use_shared_mixed_pool = (
            self.v2_cfg.slot_selection_mode == "mixed"
            and not self._hard_ownership_mask_enabled()
        )
        candidate_ids = {
            int(mid)
            for mid, owners in self.task_candidate_owners.items()
            if agent_id in owners
        }
        if use_shared_mixed_pool:
            # Mixed-TopK keeps every currently executable task visible, but it
            # should not rescore all future non-owner tasks at stress scale.
            # The bounded pool preserves the intended soft-owner behavior while
            # avoiding O(n_agents * max_action_dim) full scans every env step.
            for action in np.nonzero(full_mask[:self.max_action_dim])[0].tolist():
                mission = env.missions[action]
                if mission is not None and not mission.is_observed:
                    candidate_ids.add(int(mission.id))

            for mid in stale:
                mission = self._mission_by_id(agent_id).get(int(mid))
                if mission is not None and not mission.is_observed:
                    candidate_ids.add(int(mid))

            # Rare fallback for scenarios where assignment produced no owner
            # candidates for this satellite.  Expose the actual executable set
            # rather than falling back to the full task pool.
            if not candidate_ids:
                action_iter = np.nonzero(full_mask[:self.max_action_dim])[0].tolist()
            else:
                action_iter = [
                    action_lookup[mid]
                    for mid in candidate_ids
                    if mid in action_lookup
                ]
        else:
            if not candidate_ids:
                action_iter = range(self.max_action_dim)
            else:
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
                # Always expose currently executable tasks allowed by the mask.
                # This prevents future-only high-score candidates from hiding the
                # few actions the satellite can actually execute at this step.
                for action in np.nonzero(full_mask[:self.max_action_dim])[0].tolist():
                    mission = env.missions[action]
                    if mission is not None and not mission.is_observed:
                        candidate_ids.add(int(mission.id))
                for action in self._dynamic_rescue_actions(agent_id, full_mask, current_time, stale):
                    mission = env.missions[action]
                    if mission is not None and not mission.is_observed:
                        candidate_ids.add(int(mission.id))
                action_iter = [
                    action_lookup[mid]
                    for mid in candidate_ids
                    if mid in action_lookup
                ]

        routine = []
        dynamic = []
        flex = []
        for action in action_iter:
            mission = env.missions[action]
            if mission is None or mission.is_observed:
                continue
            candidates = self.task_candidate_owners.get(mission.id)
            if (
                self._hard_ownership_mask_enabled()
                and candidates
                and agent_id not in candidates
                and not self._ownership_released_with_context(agent_id, mission, current_time, stale)
            ):
                continue
            if (
                not self._hard_ownership_mask_enabled()
                and not use_shared_mixed_pool
                and candidates
                and agent_id not in candidates
                and not self._ownership_released_with_context(agent_id, mission, current_time, stale)
                and full_mask[action] <= 0
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
            score += self._soft_owner_score_bonus(agent_id, mission)
            if self._dynamic_takeover_release_open(agent_id, mission, current_time):
                score += 0.30
            is_available = 1 if full_mask[action] > 0 else 0
            if is_available:
                score += 0.15
                if mission.is_dynamic:
                    score += 0.35
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

    def _dynamic_rescue_actions(
        self,
        agent_id: str,
        full_mask: np.ndarray,
        current_time: float,
        stale: set,
    ) -> List[int]:
        """Expose arrived dynamic tasks to rescue agents for typed slots."""
        env = self.envs[agent_id]
        actions: List[int] = []
        release_window = max(
            float(self.v2_cfg.release_before_deadline_s or 0.0),
            float(self.v2_cfg.dynamic_broadcast_window_s or 0.0),
        )
        for action, mission in enumerate(env.missions[:self.max_action_dim]):
            if mission is None or mission.is_observed or not mission.is_dynamic:
                continue
            arrival_s = float(getattr(mission, "arrival_time_s", mission.earliest_time_s))
            if arrival_s > current_time or current_time > mission.deadline_s:
                continue
            near_release = current_time >= mission.deadline_s - release_window
            if (
                full_mask[action] > 0
                or self._dynamic_broadcast_open(mission, current_time)
                or mission.id in stale
                or near_release
            ):
                actions.append(int(action))
        return actions

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
            if not self._belongs_to_group(mission, group, env.current_time_s):
                continue
            if is_currently_available:
                score += 0.25
            items.append((float(score), int(action)))
        items.sort(key=lambda x: x[0], reverse=True)
        return items

    def _belongs_to_group(self, mission: Mission, group: str, current_time: Optional[float] = None) -> bool:
        if group == "routine":
            return not mission.is_dynamic
        if group == "dynamic":
            return mission.is_dynamic
        if group == "flex":
            if current_time is None:
                current_time = self._current_time_s()
            return self._belongs_to_flex_group(mission, current_time, self._stale_task_ids())
        return False

    def _belongs_to_flex_group(self, mission: Mission, current_time: float, stale: set) -> bool:
        near_deadline = current_time >= mission.deadline_s - self.v2_cfg.release_before_deadline_s
        return mission.is_dynamic or near_deadline or mission.id in stale or mission.priority >= 7.5

    def _dynamic_broadcast_open(self, mission: Mission, current_time: float) -> bool:
        window_s = float(self.v2_cfg.dynamic_broadcast_window_s or 0.0)
        if window_s <= 0 or not mission.is_dynamic or mission.is_observed:
            return False
        arrival_s = float(getattr(mission, "arrival_time_s", mission.earliest_time_s))
        return arrival_s <= current_time <= min(arrival_s + window_s, mission.deadline_s)

    def _dynamic_takeover_release_open(self, agent_id: str, mission: Mission, current_time: float) -> bool:
        """Release dynamic tasks to a non-owner with an earlier near-term window."""
        if mission is None or mission.is_observed or not getattr(mission, "is_dynamic", False):
            return False
        owner = self.task_owner.get(mission.id)
        if owner is None or owner == agent_id:
            return False
        arrival_s = float(getattr(mission, "arrival_time_s", mission.earliest_time_s))
        if current_time < arrival_s or current_time > mission.deadline_s:
            return False

        window_s = max(
            float(self.v2_cfg.dynamic_broadcast_window_s or 0.0),
            float(self.v2_cfg.release_before_deadline_s or 0.0),
        )
        if window_s <= 0:
            return False

        agent_next = self._next_feasible_window_start(agent_id, mission.id, current_time)
        if agent_next is None or agent_next - current_time > window_s:
            return False

        owner_env = self.envs.get(owner)
        owner_time = float(owner_env.current_time_s) if owner_env is not None else current_time
        owner_next = self._next_feasible_window_start(owner, mission.id, owner_time)
        configured_margin = float(getattr(self.v2_cfg, "dynamic_takeover_margin_s", 0.0) or 0.0)
        if configured_margin > 0:
            handoff_margin_s = configured_margin
        else:
            handoff_margin_s = max(60.0, min(float(self.assignment_lock_window_s), 600.0))
        if owner_next is not None and owner_next <= agent_next + handoff_margin_s:
            return False

        key = (int(mission.id), agent_id)
        if key not in self._dynamic_takeover_release_keys:
            self._dynamic_takeover_release_keys.add(key)
            self._n_dynamic_takeover_release_events += 1
            self._released_mission_ids.add(mission.id)
        return True

    def _soft_owner_score_bonus(self, agent_id: str, mission: Mission) -> float:
        bonus = float(self.v2_cfg.candidate_owner_bonus or 0.0)
        if bonus <= 0:
            return 0.0
        owners = self.task_candidate_owners.get(mission.id, [])
        return bonus if agent_id in owners else 0.0

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
        exposures = max(self._slot_exposure_count, 1)
        invalid_total = sum(self._slot_invalid_reason_sum.values())
        empty_invalid = self._slot_invalid_reason_sum.get("empty", 0)
        filled_invalid = max(invalid_total - empty_invalid, 0)
        metrics.update({
            "slot_valid_ratio": self._slot_valid_sum / denom,
            "slot_filled_ratio": self._slot_filled_sum / denom,
            "slot_invalid_ratio": invalid_total / denom,
            "slot_filled_invalid_ratio": filled_invalid / denom,
            "avg_valid_slots": self._slot_valid_sum / exposures,
            "avg_filled_slots": self._slot_filled_sum / exposures,
            "avg_invalid_slots": invalid_total / exposures,
            "avg_filled_invalid_slots": filled_invalid / exposures,
            "slot_selection_mixed": 1.0 if self.v2_cfg.slot_selection_mode == "mixed" else 0.0,
        })
        for reason in SLOT_INVALID_REASONS:
            count = self._slot_invalid_reason_sum.get(reason, 0)
            metrics[f"slot_invalid_{reason}_ratio"] = count / denom
            metrics[f"avg_invalid_{reason}_slots"] = count / exposures
        for slot_type in ("routine", "dynamic", "flex"):
            capacity = max(self._slot_type_capacity_sum.get(slot_type, 0), 1)
            metrics[f"{slot_type}_slot_valid_ratio"] = (
                self._slot_type_valid_sum.get(slot_type, 0) / capacity
            )
            metrics[f"{slot_type}_slot_filled_ratio"] = (
                self._slot_type_filled_sum.get(slot_type, 0) / capacity
            )
            metrics[f"avg_valid_{slot_type}_slots"] = (
                self._slot_type_valid_sum.get(slot_type, 0) / exposures
            )
            metrics[f"avg_filled_{slot_type}_slots"] = (
                self._slot_type_filled_sum.get(slot_type, 0) / exposures
            )
        return metrics
