from __future__ import annotations

from typing import Any, Dict, List, Optional

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
        self._slot_timing: Dict[str, List[dict]] = {}
        self._candidate_score_table: Dict[int, Dict[str, CandidateScore]] = {}
        self._v2_step_cache = {}
        self._slot_valid_sum = 0
        self._slot_filled_sum = 0
        self._slot_exposure_count = 0
        self._slot_current_valid_sum = 0
        self._slot_future_valid_sum = 0
        self._slot_type_valid_sum = {"routine": 0, "dynamic": 0, "flex": 0}
        self._slot_type_filled_sum = {"routine": 0, "dynamic": 0, "flex": 0}
        self._slot_type_capacity_sum = {"routine": 0, "dynamic": 0, "flex": 0}
        self._slot_invalid_reason_sum = {reason: 0 for reason in SLOT_INVALID_REASONS}
        self._dynamic_current_slot_candidate_sum = 0
        self._dynamic_current_slot_selected_sum = 0
        self._dynamic_future_slot_candidate_sum = 0
        self._dynamic_future_slot_selected_sum = 0
        self._dynamic_task_diagnostics: Dict[int, Dict[str, Any]] = {}
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
        self._slot_timing = {}
        self._candidate_score_table = {}
        self._slot_valid_sum = 0
        self._slot_filled_sum = 0
        self._slot_exposure_count = 0
        self._slot_current_valid_sum = 0
        self._slot_future_valid_sum = 0
        self._slot_type_valid_sum = {"routine": 0, "dynamic": 0, "flex": 0}
        self._slot_type_filled_sum = {"routine": 0, "dynamic": 0, "flex": 0}
        self._slot_type_capacity_sum = {"routine": 0, "dynamic": 0, "flex": 0}
        self._slot_invalid_reason_sum = {reason: 0 for reason in SLOT_INVALID_REASONS}
        self._dynamic_current_slot_candidate_sum = 0
        self._dynamic_current_slot_selected_sum = 0
        self._dynamic_future_slot_candidate_sum = 0
        self._dynamic_future_slot_selected_sum = 0
        self._dynamic_task_diagnostics = {}
        self._clear_v2_step_cache()
        result = super().reset(*args, **kwargs)
        self._clear_v2_step_cache()
        return result

    def step(self, actions):
        prev_schedule_lens = {
            aid: len(env.schedule_log)
            for aid, env in self.envs.items()
        }
        self._record_dynamic_policy_selection(actions)
        self._clear_v2_step_cache()
        result = super().step(actions)
        self._record_dynamic_delivery_diagnostics(prev_schedule_lens)
        self._clear_v2_step_cache()
        return result

    def _clear_v2_step_cache(self) -> None:
        self._v2_step_cache = {}

    def _dynamic_diag_entry(self, mission: Mission) -> Dict[str, Any]:
        mission_id = int(mission.id)
        entry = self._dynamic_task_diagnostics.get(mission_id)
        if entry is None:
            entry = {
                "mission_id": mission_id,
                "arrival_time_s": float(getattr(mission, "arrival_time_s", mission.earliest_time_s)),
                "earliest_time_s": float(getattr(mission, "earliest_time_s", 0.0)),
                "deadline_s": float(getattr(mission, "deadline_s", 0.0)),
                "priority": float(getattr(mission, "priority", 0.0)),
                "arrived": False,
                "candidate_seen": False,
                "candidate_seen_count": 0,
                "candidate_current_executable_seen": False,
                "candidate_current_executable_seen_count": 0,
                "candidate_future_executable_seen": False,
                "candidate_future_executable_seen_count": 0,
                "selected": False,
                "selected_count": 0,
                "selected_current_executable": False,
                "observed": False,
                "downlinked": False,
                "downlink_queued": False,
                "downlink_queue_blocked": False,
                "downlink_failed": False,
                "first_candidate_seen_s": None,
                "first_current_executable_seen_s": None,
                "first_selected_s": None,
                "obs_end_s": None,
                "downlink_start_s": None,
                "downlink_end_s": None,
                "final_downlink_queue_s": 0.0,
                "max_downlink_queue_s": 0.0,
            }
            self._dynamic_task_diagnostics[mission_id] = entry
        else:
            entry["arrival_time_s"] = float(getattr(mission, "arrival_time_s", mission.earliest_time_s))
            entry["earliest_time_s"] = float(getattr(mission, "earliest_time_s", 0.0))
            entry["deadline_s"] = float(getattr(mission, "deadline_s", 0.0))
            entry["priority"] = float(getattr(mission, "priority", 0.0))
        return entry

    def _record_dynamic_candidate_seen(
        self,
        agent_id: str,
        action: int,
        full_mask: np.ndarray,
    ) -> None:
        env = self.envs[agent_id]
        if action < 0 or action >= self.max_action_dim:
            return
        mission = env.missions[int(action)]
        if mission is None or not getattr(mission, "is_dynamic", False):
            return
        current_time = float(env.current_time_s)
        arrival_s = float(getattr(mission, "arrival_time_s", mission.earliest_time_s))
        if arrival_s > current_time:
            return
        entry = self._dynamic_diag_entry(mission)
        entry["arrived"] = True
        entry["candidate_seen"] = True
        entry["candidate_seen_count"] = int(entry["candidate_seen_count"]) + 1
        if entry["first_candidate_seen_s"] is None:
            entry["first_candidate_seen_s"] = current_time
        current_valid = action < len(full_mask) and full_mask[action] > 0
        future_valid = (not current_valid) and self._future_task_action_valid(
            agent_id,
            int(action),
            full_mask,
        )
        if current_valid:
            entry["candidate_current_executable_seen"] = True
            entry["candidate_current_executable_seen_count"] = (
                int(entry["candidate_current_executable_seen_count"]) + 1
            )
            if entry["first_current_executable_seen_s"] is None:
                entry["first_current_executable_seen_s"] = current_time
        if future_valid:
            entry["candidate_future_executable_seen"] = True
            entry["candidate_future_executable_seen_count"] = (
                int(entry["candidate_future_executable_seen_count"]) + 1
            )

    def _record_dynamic_policy_selection(self, actions: Dict[str, int]) -> None:
        if not self._candidate_actions_enabled():
            return
        raw_actions = self._decode_actions(actions)
        for agent_id, raw_action in raw_actions.items():
            if raw_action is None or raw_action < 0 or raw_action >= self.max_action_dim:
                continue
            env = self.envs[agent_id]
            mission = env.missions[int(raw_action)]
            if mission is None or not getattr(mission, "is_dynamic", False):
                continue
            current_time = float(env.current_time_s)
            arrival_s = float(getattr(mission, "arrival_time_s", mission.earliest_time_s))
            if arrival_s > current_time:
                continue
            entry = self._dynamic_diag_entry(mission)
            entry["arrived"] = True
            entry["selected"] = True
            entry["selected_count"] = int(entry["selected_count"]) + 1
            if entry["first_selected_s"] is None:
                entry["first_selected_s"] = current_time
            if bool(entry.get("candidate_current_executable_seen", False)):
                entry["selected_current_executable"] = True

    def _record_dynamic_delivery_diagnostics(
        self,
        prev_schedule_lens: Dict[str, int],
    ) -> None:
        queue_target_s = max(
            float(getattr(self.v2_cfg, "downlink_queue_target_s", 3600.0) or 3600.0),
            1.0,
        )
        for agent_id, env in self.envs.items():
            start = int(prev_schedule_lens.get(agent_id, len(env.schedule_log)))
            for record in env.schedule_log[start:]:
                mission = self._mission_for_agent(agent_id, record.mission_id)
                is_dynamic = bool(getattr(record, "is_dynamic", False)) or bool(
                    getattr(mission, "is_dynamic", False)
                )
                if mission is None or not is_dynamic:
                    continue
                entry = self._dynamic_diag_entry(mission)
                queue_s = (
                    max(0.0, float(record.downlink_start_s) - float(record.obs_end_s))
                    if int(getattr(record, "ground_station_id", -1)) >= 0
                    else 0.0
                )
                entry["arrived"] = True
                entry["observed"] = True
                entry["downlinked"] = bool(getattr(mission, "is_downlinked", False))
                entry["obs_end_s"] = float(record.obs_end_s)
                entry["downlink_start_s"] = float(record.downlink_start_s)
                entry["downlink_end_s"] = float(record.downlink_end_s)
                entry["final_downlink_queue_s"] = float(queue_s)
                entry["max_downlink_queue_s"] = max(
                    float(entry.get("max_downlink_queue_s", 0.0) or 0.0),
                    float(queue_s),
                )
                if queue_s > 1e-6:
                    entry["downlink_queued"] = True
                if queue_s >= queue_target_s:
                    entry["downlink_queue_blocked"] = True
                if getattr(env, "downlink_required", False) and int(getattr(record, "ground_station_id", -1)) < 0:
                    entry["downlink_failed"] = True

    def get_dynamic_task_diagnostics(self) -> Dict[str, Dict[str, Any]]:
        current_time = float(self._current_time_s())
        missions: Dict[int, Mission] = {}
        for env in self.envs.values():
            for mission in env.missions[:self.max_action_dim]:
                if mission is not None and getattr(mission, "is_dynamic", False):
                    missions[int(mission.id)] = mission
        for mission in missions.values():
            arrival_s = float(getattr(mission, "arrival_time_s", mission.earliest_time_s))
            if arrival_s <= current_time or int(mission.id) in self._dynamic_task_diagnostics:
                entry = self._dynamic_diag_entry(mission)
                entry["arrived"] = bool(entry["arrived"] or arrival_s <= current_time)
                entry["observed"] = bool(entry["observed"] or self._mission_observed_anywhere(mission.id))
                entry["downlinked"] = bool(entry["downlinked"] or getattr(mission, "is_downlinked", False))
        queue_target_s = max(
            float(getattr(self.v2_cfg, "downlink_queue_target_s", 3600.0) or 3600.0),
            1.0,
        )
        for agent_id, env in self.envs.items():
            for record in env.schedule_log:
                mission = self._mission_for_agent(agent_id, record.mission_id)
                is_dynamic = bool(getattr(record, "is_dynamic", False)) or bool(
                    getattr(mission, "is_dynamic", False)
                )
                if mission is None or not is_dynamic:
                    continue
                entry = self._dynamic_diag_entry(mission)
                queue_s = (
                    max(0.0, float(record.downlink_start_s) - float(record.obs_end_s))
                    if int(getattr(record, "ground_station_id", -1)) >= 0
                    else 0.0
                )
                entry["arrived"] = True
                entry["observed"] = True
                entry["downlinked"] = bool(entry["downlinked"] or getattr(mission, "is_downlinked", False))
                entry["obs_end_s"] = float(record.obs_end_s)
                entry["downlink_start_s"] = float(record.downlink_start_s)
                entry["downlink_end_s"] = float(record.downlink_end_s)
                entry["final_downlink_queue_s"] = float(queue_s)
                entry["max_downlink_queue_s"] = max(
                    float(entry.get("max_downlink_queue_s", 0.0) or 0.0),
                    float(queue_s),
                )
                entry["downlink_queued"] = bool(queue_s > 1e-6)
                entry["downlink_queue_blocked"] = bool(queue_s >= queue_target_s)
                entry["downlink_failed"] = bool(
                    getattr(env, "downlink_required", False)
                    and int(getattr(record, "ground_station_id", -1)) < 0
                )
        return {
            str(mission_id): dict(entry)
            for mission_id, entry in sorted(self._dynamic_task_diagnostics.items())
        }

    def _future_task_execution_enabled(self) -> bool:
        return bool(getattr(self.v2_cfg, "allow_future_task_execution", False))

    def _future_task_max_wait_s(self) -> float:
        return float(getattr(self.v2_cfg, "future_task_max_wait_s", 0.0) or 0.0)

    def _future_task_max_wait_for_mission(self, mission) -> float:
        default_wait = self._future_task_max_wait_s()
        if mission is not None and not getattr(mission, "is_dynamic", False):
            routine_wait = float(getattr(self.v2_cfg, "future_routine_max_wait_s", default_wait) or 0.0)
            if routine_wait > 0.0:
                return routine_wait
        return default_wait

    def _future_task_requires_no_current_valid(self) -> bool:
        return bool(getattr(self.v2_cfg, "future_task_requires_no_current_valid", False))

    def _routine_future_dynamic_guard_s(self) -> float:
        return float(getattr(self.v2_cfg, "routine_future_dynamic_guard_s", 0.0) or 0.0)

    def _dynamic_response_target_s(self) -> float:
        return float(getattr(self.v2_cfg, "dynamic_response_target_s", self.horizon_s) or self.horizon_s)

    def _dynamic_rescue_response_bonus(self) -> float:
        return float(getattr(self.v2_cfg, "dynamic_rescue_response_bonus", 0.0) or 0.0)

    def _dynamic_downlink_priority_enabled(self) -> bool:
        return bool(getattr(self.v2_cfg, "dynamic_downlink_priority", False))

    def _downlink_priority_key(self, agent_id: str, record, mission) -> tuple:
        if not self._dynamic_downlink_priority_enabled():
            return super()._downlink_priority_key(agent_id, record, mission)
        dynamic_rank = 0 if getattr(mission, "is_dynamic", False) else 1
        arrival_s = float(getattr(mission, "arrival_time_s", mission.earliest_time_s))
        return (
            dynamic_rank,
            float(mission.deadline_s),
            -float(getattr(mission, "priority", 0.0)),
            arrival_s if dynamic_rank == 0 else float(record.obs_end_s),
            float(record.obs_end_s),
            str(agent_id),
            int(record.mission_id),
        )

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
        self._record_dynamic_slot_candidates(agent_id, dynamic, full_mask)

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
            executable_reserve = int(
                round(self.v2_cfg.slots.total_slots * float(self.v2_cfg.executable_slot_reserve_ratio))
            )
            if executable_reserve > 0:
                current_items = []
                for item, slot_type in (
                    [(item, "dynamic") for item in dynamic]
                    + [(item, "flex") for item in flex]
                    + [(item, "routine") for item in routine]
                ):
                    is_available, score, action = item
                    if not is_available or action in used:
                        continue
                    current_items.append((score, action, slot_type))
                current_items.sort(key=lambda x: x[0], reverse=True)
                for score, action, slot_type in current_items:
                    if len(selected) >= executable_reserve:
                        break
                    if action in used:
                        continue
                    append_item(slot_type, score, action)
                    if slot_type in slot_type_counts:
                        slot_type_counts[slot_type] += 1

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
        self._slot_timing[agent_id] = self._slot_timing_features(agent_id, selected, full_mask)
        self._record_dynamic_slot_selection(agent_id, selected, full_mask)
        for slot_type in ("routine", "dynamic", "flex"):
            if self.v2_cfg.slot_selection_mode == "mixed":
                capacity = self.v2_cfg.slots.total_slots
            else:
                capacity = int(getattr(self.v2_cfg.slots, f"{slot_type}_slots"))
            self._slot_type_capacity_sum[slot_type] += capacity
        self._record_slot_diagnostics(agent_id, selected, full_mask, slot_types)
        return selected[:self.v2_cfg.slots.total_slots]

    def _slot_timing_features(
        self,
        agent_id: str,
        selected: List[Optional[int]],
        full_mask: np.ndarray,
    ) -> List[dict]:
        env = self.envs[agent_id]
        timing = []
        for action in selected[:self.v2_cfg.slots.total_slots]:
            row = {
                "currently_executable": 0.0,
                "future_executable": 0.0,
                "wait_norm": 1.0,
                "next_start_norm": 0.0,
                "time_to_deadline_norm": 0.0,
            }
            if action is None or action < 0 or action >= self.max_action_dim:
                timing.append(row)
                continue
            mission = env.missions[action]
            if mission is None or mission.is_observed:
                timing.append(row)
                continue
            current_time = float(env.current_time_s)
            row["currently_executable"] = 1.0 if full_mask[action] > 0 else 0.0
            next_start = (
                current_time
                if row["currently_executable"] > 0
                else self._future_task_ready_time(agent_id, int(action), full_mask=full_mask)
            )
            if next_start is not None:
                wait_s = max(float(next_start) - current_time, 0.0)
                row["future_executable"] = 1.0 if wait_s > 1e-6 else 0.0
                row["wait_norm"] = float(np.clip(wait_s / max(env.horizon_s, 1.0), 0.0, 1.0))
                row["next_start_norm"] = float(np.clip(float(next_start) / max(env.horizon_s, 1.0), 0.0, 1.0))
            slack_s = max(float(mission.deadline_s) - max(current_time, float(mission.earliest_time_s)), 0.0)
            row["time_to_deadline_norm"] = float(np.clip(slack_s / max(env.horizon_s, 1.0), 0.0, 1.0))
            timing.append(row)
        return timing

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
                and (
                    full_mask[action] > 0
                    or self._future_task_action_valid(agent_id, int(action), full_mask)
                )
            )
            current_valid = (
                action is not None
                and 0 <= action < len(full_mask)
                and full_mask[action] > 0
            )
            future_valid = (
                action is not None
                and 0 <= action < len(full_mask)
                and full_mask[action] <= 0
                and self._future_task_action_valid(agent_id, int(action), full_mask)
            )
            if valid:
                self._slot_valid_sum += 1
            if current_valid:
                self._slot_current_valid_sum += 1
            if future_valid:
                self._slot_future_valid_sum += 1
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

    def _record_dynamic_slot_candidates(
        self,
        agent_id: str,
        dynamic_items,
        full_mask: np.ndarray,
    ) -> None:
        for is_available, _, action in dynamic_items:
            if is_available:
                self._dynamic_current_slot_candidate_sum += 1
            elif self._future_task_action_valid(agent_id, int(action), full_mask):
                self._dynamic_future_slot_candidate_sum += 1

    def _record_dynamic_slot_selection(
        self,
        agent_id: str,
        selected: List[Optional[int]],
        full_mask: np.ndarray,
    ) -> None:
        env = self.envs[agent_id]
        for action in selected[:self.v2_cfg.slots.total_slots]:
            if action is None or action < 0 or action >= self.max_action_dim:
                continue
            mission = env.missions[int(action)]
            if mission is None or not getattr(mission, "is_dynamic", False):
                continue
            self._record_dynamic_candidate_seen(agent_id, int(action), full_mask)
            if action < len(full_mask) and full_mask[action] > 0:
                self._dynamic_current_slot_selected_sum += 1
            elif self._future_task_action_valid(agent_id, int(action), full_mask):
                self._dynamic_future_slot_selected_sum += 1

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

            is_available = 1 if full_mask[action] > 0 else 0
            future_ready = None
            if not is_available:
                future_ready = self._future_task_ready_time(agent_id, int(action), full_mask=full_mask)
                if (
                    future_ready is None
                    and bool(getattr(self.v2_cfg, "drop_ineligible_future_candidates", False))
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
            if is_available:
                score += 0.15
                if mission.is_dynamic:
                    score += float(getattr(self.v2_cfg, "dynamic_current_slot_bonus", 0.0) or 0.0)
            elif future_ready is not None:
                if mission.is_dynamic:
                    target_s = max(float(self.v2_cfg.dynamic_response_target_s or 0.0), 1.0)
                    wait_s = max(float(future_ready) - current_time, 0.0)
                    response_weight = 1.0 - float(np.clip(wait_s / target_s, 0.0, 1.0))
                    score += (
                        float(getattr(self.v2_cfg, "dynamic_future_bonus", 0.0) or 0.0)
                        * (0.5 + 0.5 * response_weight)
                    )
                elif self._near_dynamic_pressure(agent_id, full_mask, current_time):
                    score -= float(getattr(self.v2_cfg, "routine_future_dynamic_penalty", 0.0) or 0.0)
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
                "slot_timing": list(self._slot_timing.get(agent_id, [])),
                "n_task_candidate_owners": len(self.task_candidate_owners),
            }
        return obs, info

    def _dynamic_task_diagnostic_metrics(self) -> Dict[str, float]:
        rows = [
            entry for entry in self.get_dynamic_task_diagnostics().values()
            if bool(entry.get("arrived", False))
        ]
        n_arrived = len(rows)
        seen = [entry for entry in rows if bool(entry.get("candidate_seen", False))]
        current_seen = [
            entry for entry in rows
            if bool(entry.get("candidate_current_executable_seen", False))
        ]
        future_seen = [
            entry for entry in rows
            if bool(entry.get("candidate_future_executable_seen", False))
        ]
        selected = [entry for entry in rows if bool(entry.get("selected", False))]
        observed = [entry for entry in rows if bool(entry.get("observed", False))]
        downlinked = [entry for entry in rows if bool(entry.get("downlinked", False))]
        queued = [entry for entry in observed if bool(entry.get("downlink_queued", False))]
        blocked = [entry for entry in observed if bool(entry.get("downlink_queue_blocked", False))]
        failed = [entry for entry in observed if bool(entry.get("downlink_failed", False))]
        queue_values = [
            float(entry.get("final_downlink_queue_s", 0.0) or 0.0)
            for entry in observed
        ]
        return {
            "n_dynamic_tasks_arrived": float(n_arrived),
            "n_dynamic_tasks_candidate_seen": float(len(seen)),
            "dynamic_task_candidate_seen_rate": len(seen) / max(n_arrived, 1),
            "n_dynamic_tasks_current_executable_seen": float(len(current_seen)),
            "dynamic_task_current_executable_seen_rate": len(current_seen) / max(len(seen), 1),
            "n_dynamic_tasks_future_executable_seen": float(len(future_seen)),
            "dynamic_task_future_executable_seen_rate": len(future_seen) / max(len(seen), 1),
            "n_dynamic_tasks_policy_selected": float(len(selected)),
            "dynamic_task_policy_selected_rate": len(selected) / max(len(seen), 1),
            "n_dynamic_tasks_observed_diag": float(len(observed)),
            "dynamic_task_observed_after_selected_rate": len(observed) / max(len(selected), 1),
            "n_dynamic_tasks_downlinked_diag": float(len(downlinked)),
            "dynamic_task_downlinked_after_observed_rate": len(downlinked) / max(len(observed), 1),
            "n_dynamic_tasks_with_downlink_queue": float(len(queued)),
            "dynamic_task_downlink_queue_rate": len(queued) / max(len(observed), 1),
            "n_dynamic_tasks_downlink_queue_blocked": float(len(blocked)),
            "dynamic_task_downlink_queue_block_rate": len(blocked) / max(len(observed), 1),
            "n_dynamic_tasks_downlink_failed": float(len(failed)),
            "dynamic_task_downlink_failed_rate": len(failed) / max(len(observed), 1),
            "avg_dynamic_task_downlink_queue_s": (
                float(np.mean(queue_values)) if queue_values else 0.0
            ),
        }

    def get_metrics(self):
        metrics = super().get_metrics()
        denom = max(self._slot_exposure_count * self.v2_cfg.slots.total_slots, 1)
        exposures = max(self._slot_exposure_count, 1)
        invalid_total = sum(self._slot_invalid_reason_sum.values())
        empty_invalid = self._slot_invalid_reason_sum.get("empty", 0)
        filled_invalid = max(invalid_total - empty_invalid, 0)
        metrics.update({
            "slot_valid_ratio": self._slot_valid_sum / denom,
            "slot_current_valid_ratio": self._slot_current_valid_sum / denom,
            "slot_future_valid_ratio": self._slot_future_valid_sum / denom,
            "slot_filled_ratio": self._slot_filled_sum / denom,
            "slot_invalid_ratio": invalid_total / denom,
            "slot_filled_invalid_ratio": filled_invalid / denom,
            "avg_valid_slots": self._slot_valid_sum / exposures,
            "avg_current_valid_slots": self._slot_current_valid_sum / exposures,
            "avg_future_valid_slots": self._slot_future_valid_sum / exposures,
            "avg_filled_slots": self._slot_filled_sum / exposures,
            "avg_invalid_slots": invalid_total / exposures,
            "avg_filled_invalid_slots": filled_invalid / exposures,
            "dynamic_current_slot_exposure_rate": (
                self._dynamic_current_slot_selected_sum
                / max(self._dynamic_current_slot_candidate_sum, 1)
            ),
            "dynamic_future_slot_exposure_rate": (
                self._dynamic_future_slot_selected_sum
                / max(self._dynamic_future_slot_candidate_sum, 1)
            ),
            "avg_dynamic_current_slot_candidates": (
                self._dynamic_current_slot_candidate_sum / exposures
            ),
            "avg_dynamic_current_slots_selected": (
                self._dynamic_current_slot_selected_sum / exposures
            ),
            "avg_dynamic_future_slot_candidates": (
                self._dynamic_future_slot_candidate_sum / exposures
            ),
            "avg_dynamic_future_slots_selected": (
                self._dynamic_future_slot_selected_sum / exposures
            ),
            "slot_selection_mixed": 1.0 if self.v2_cfg.slot_selection_mode == "mixed" else 0.0,
        })
        metrics.update(self._dynamic_task_diagnostic_metrics())
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
