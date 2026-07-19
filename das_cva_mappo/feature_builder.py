from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .candidate_scorer import EDGE_FEATURE_DIM
from .env_adapter import V2CandidateAdapter
from .temporal_features import TEMPORAL_WINDOW_FEATURE_DIM, temporal_window_features


SLOT_TYPES = {"routine": 0, "dynamic": 1, "flex": 2, "empty": 3}


@dataclass
class ActionSetBatch:
    state: np.ndarray
    action_features: np.ndarray
    candidate_edge_features: np.ndarray
    candidate_task_ids: np.ndarray
    action_mask: np.ndarray


class ActionSetFeatureBuilder:
    """Build state and action-entity features from the v2 candidate interface."""

    def __init__(
        self,
        state_dim: int = 20,
        action_feature_dim: int = 28 + TEMPORAL_WINDOW_FEATURE_DIM,
        state_base_dim: int = 20,
        mode: str = "full",
        use_candidate_score: bool = True,
        candidate_scorer=None,
        candidate_adapter=None,
        use_response_budget_features: bool = True,
        use_temporal_window_features: bool = True,
        use_early_delivery_temporal_features: bool = True,
        temporal_window_top_k: int = 3,
        temporal_early_delivery_weight: float = 0.35,
        temporal_state_history_len: int = 1,
    ):
        self.state_dim = int(state_dim)
        self.action_feature_dim = int(action_feature_dim)
        self.state_base_dim = int(state_base_dim)
        self.mode = str(mode)
        self.use_candidate_score = bool(use_candidate_score and self.mode != "no_score")
        self.candidate_scorer = candidate_scorer
        self.candidate_adapter = candidate_adapter or V2CandidateAdapter()
        self.use_response_budget_features = bool(use_response_budget_features)
        self.use_temporal_window_features = bool(use_temporal_window_features)
        self.use_early_delivery_temporal_features = bool(use_early_delivery_temporal_features)
        self.temporal_window_top_k = max(int(temporal_window_top_k), 1)
        self.temporal_early_delivery_weight = max(float(temporal_early_delivery_weight), 0.0)
        self.temporal_state_history_len = max(int(temporal_state_history_len), 1)
        self._state_history: Dict[Tuple[int, str], List[np.ndarray]] = {}
        self._state_history_last_time: Dict[Tuple[int, str], float] = {}

    def build_agent(self, env, agent_id: str, info: Dict) -> ActionSetBatch:
        mask = np.asarray(info.get("action_mask", np.ones(env.action_dim)), dtype=np.float32)
        features = np.zeros((env.action_dim, self.action_feature_dim), dtype=np.float32)
        edge_features = np.zeros((env.action_dim, EDGE_FEATURE_DIM), dtype=np.float32)
        task_ids = np.full(env.action_dim, -1, dtype=np.int64)
        self._fill_task_features(env, agent_id, info, mask, features, edge_features, task_ids)
        self._fill_transfer_features(env, agent_id, info, mask, features)
        self._fill_idle_features(env, agent_id, info, mask, features)
        state = self._build_state_features(env, agent_id, info, mask)
        if self.mode == "minimal":
            keep = np.zeros_like(features)
            keep[:, :5] = features[:, :5]
            features = keep
        return ActionSetBatch(
            state=state,
            action_features=features,
            candidate_edge_features=edge_features,
            candidate_task_ids=task_ids,
            action_mask=mask,
        )

    def build_many(self, env, infos: Dict[str, Dict]) -> Dict[str, ActionSetBatch]:
        return {
            aid: self.build_agent(env, aid, infos[aid])
            for aid in env.agent_ids
        }

    def _build_state_features(self, env, agent_id: str, info: Dict, mask: np.ndarray) -> np.ndarray:
        sub_env = env.envs[agent_id]
        sat_state = sub_env.propagator.propagate(sub_env.current_time_s)
        storage_capacity = max(int(getattr(sub_env, "satellite_storage_capacity", 0) or 0), 1)
        onboard = 0.0
        if hasattr(sub_env, "_onboard_image_count"):
            onboard = float(sub_env._onboard_image_count(sub_env.current_time_s))
        valid_no_idle = max(float(np.sum(mask)) - 1.0, 0.0)
        raw_valid = float(info.get("raw_valid_action_count", valid_no_idle))
        pending = sum(
            1 for mission in sub_env.missions[: env.max_action_dim]
            if mission is not None and not mission.is_observed
        )
        state = np.zeros(self.state_base_dim, dtype=np.float32)
        values = [
            sub_env.current_time_s / max(sub_env.horizon_s, 1.0),
            sat_state.latitude_deg / 90.0,
            sat_state.longitude_deg / 180.0,
            len(sub_env.schedule_log) / max(env.max_action_dim, 1),
            onboard / storage_capacity,
            valid_no_idle / max(env.action_dim - 1, 1),
            raw_valid / max(getattr(env, "max_action_dim", env.action_dim), 1),
            pending / max(env.max_action_dim, 1),
            float(getattr(env, "n_ground_stations", 0)) / 16.0,
            1.0 if getattr(env, "enable_inter_satellite_transfer", False) else 0.0,
            float(self.candidate_adapter.n_task_candidate_owners(env, info)) / max(env.max_action_dim, 1),
            float(getattr(env, "n_agents", 1)) / 32.0,
        ]
        values.extend(
            self._dynamic_response_state(env, sub_env)
            if self.use_response_budget_features
            else [0.0, 0.0, 0.0, 0.0]
        )
        values.extend(
            self._temporal_window_state(env, sub_env, info)
            if self.use_temporal_window_features
            else [0.0, 0.0, 0.0, 0.0]
        )
        state[: min(len(values), self.state_base_dim)] = values[: self.state_base_dim]
        return self._state_with_history(env, agent_id, state, float(sub_env.current_time_s))

    def _fill_task_features(
        self,
        env,
        agent_id: str,
        info: Dict,
        mask: np.ndarray,
        out: np.ndarray,
        edge_out: np.ndarray,
        task_id_out: np.ndarray,
    ) -> None:
        sub_env = env.envs[agent_id]
        slots = self.candidate_adapter.candidate_slots(info)
        slot_types = self.candidate_adapter.slot_types(info)
        slot_scores = self.candidate_adapter.slot_scores(info)
        slot_timing = self.candidate_adapter.slot_timing(info)
        task_limit = min(len(slots), getattr(env, "candidate_action_top_k", len(slots)), out.shape[0])
        for exposed_action in range(task_limit):
            raw_action = slots[exposed_action]
            self._set_type(out[exposed_action], "task")
            out[exposed_action, 3] = float(mask[exposed_action] > 0)
            if self.use_candidate_score and exposed_action < len(slot_scores):
                out[exposed_action, 4] = float(np.tanh(float(slot_scores[exposed_action])))
            slot_type = slot_types[exposed_action] if exposed_action < len(slot_types) else "empty"
            self._set_slot_type(out[exposed_action], slot_type)
            if raw_action is None:
                continue
            mission = sub_env.missions[int(raw_action)]
            if mission is None:
                continue
            timing = slot_timing[exposed_action] if exposed_action < len(slot_timing) else {}
            task_id_out[exposed_action] = int(mission.id)
            mission_feats, wait_norm, slack_norm = self._mission_features(sub_env, mission)
            out[exposed_action, 9:16] = mission_feats
            out[exposed_action, 16] = slack_norm
            out[exposed_action, 17] = float(timing.get("wait_norm", wait_norm))
            owners = self.candidate_adapter.candidate_owners(env, mission.id)
            out[exposed_action, 18] = 1.0 if agent_id in owners else 0.0
            out[exposed_action, 19] = 1.0 if getattr(mission, "is_dynamic", False) else 0.0
            out[exposed_action, 20] = exposed_action / max(task_limit - 1, 1)
            out[exposed_action, 21] = float(getattr(mission, "priority", 0.0)) / 10.0
            out[exposed_action, 22] = float(timing.get("currently_executable", mask[exposed_action] > 0))
            out[exposed_action, 23] = float(timing.get("future_executable", 0.0))
            if self.use_response_budget_features and out.shape[1] > 24:
                wait_s = float(timing.get("wait_norm", wait_norm)) * max(float(sub_env.horizon_s), 1.0)
                response_features = self._dynamic_response_features(env, sub_env, mission, wait_s)
                limit = min(out.shape[1] - 24, len(response_features))
                out[exposed_action, 24:24 + limit] = response_features[:limit]
            if self.use_temporal_window_features and out.shape[1] > 28:
                temporal_features = self._temporal_window_features(env, sub_env, mission)
                limit = min(out.shape[1] - 28, len(temporal_features))
                out[exposed_action, 28:28 + limit] = temporal_features[:limit]
            edge_feature = self._candidate_edge_feature(env, agent_id, sub_env, mission)
            if edge_feature is not None:
                edge_out[exposed_action] = edge_feature

    def _candidate_edge_feature(self, env, agent_id: str, sub_env, mission) -> Optional[np.ndarray]:
        scorer = self.candidate_scorer
        if scorer is None:
            return None
        return self.candidate_adapter.edge_features(
            env=env,
            agent_id=agent_id,
            sub_env=sub_env,
            mission=mission,
            scorer=scorer,
        )

    def _fill_transfer_features(self, env, agent_id: str, info: Dict, mask: np.ndarray, out: np.ndarray) -> None:
        targets = info.get("transfer_action_targets") or []
        start = int(getattr(env, "candidate_action_top_k", 0))
        source_env = env.envs[agent_id]
        storage_capacity = max(int(getattr(source_env, "satellite_storage_capacity", 0) or 0), 1)
        source_onboard = 0.0
        if hasattr(source_env, "_onboard_image_count"):
            source_onboard = float(source_env._onboard_image_count(source_env.current_time_s))
        for offset, target_id in enumerate(targets):
            exposed_action = start + offset
            if exposed_action >= out.shape[0]:
                break
            self._set_type(out[exposed_action], "transfer")
            out[exposed_action, 3] = float(mask[exposed_action] > 0)
            out[exposed_action, 16] = source_onboard / storage_capacity
            target_env = env.envs.get(target_id)
            if target_env is not None and hasattr(target_env, "_onboard_image_count"):
                target_capacity = max(int(getattr(target_env, "satellite_storage_capacity", 0) or 0), 1)
                out[exposed_action, 17] = float(target_env._onboard_image_count(target_env.current_time_s)) / target_capacity
            out[exposed_action, 20] = offset / max(len(targets) - 1, 1)

    def _fill_idle_features(self, env, agent_id: str, info: Dict, mask: np.ndarray, out: np.ndarray) -> None:
        idle = int(getattr(env, "idle_action", out.shape[0] - 1))
        if not 0 <= idle < out.shape[0]:
            return
        self._set_type(out[idle], "idle")
        out[idle, 3] = float(mask[idle] > 0)
        out[idle, 16] = max(float(np.sum(mask)) - 1.0, 0.0) / max(out.shape[0] - 1, 1)
        out[idle, 20] = 1.0

    def _mission_features(self, sub_env, mission) -> Tuple[np.ndarray, float, float]:
        if mission.is_observed:
            obs_status = 1.0
        elif mission.obs_start_s > 0:
            obs_status = 0.5
        else:
            obs_status = 0.0
        w_start, w_end = sub_env._get_next_vtw_times(mission.id)
        wait_s = max(w_start - sub_env.current_time_s, 0.0) if w_start > 0 else sub_env.horizon_s
        slack_s = max(mission.deadline_s - max(sub_env.current_time_s, mission.earliest_time_s), 0.0)
        mission_feats = np.array([
            obs_status,
            w_start / max(sub_env.horizon_s, 1.0),
            w_end / max(sub_env.horizon_s, 1.0),
            mission.obs_start_s / sub_env.horizon_s if mission.obs_start_s > 0 else 0.0,
            mission.obs_end_s / sub_env.horizon_s if mission.obs_end_s > 0 else 0.0,
            mission.priority / 10.0,
            1.0 if mission.is_dynamic else 0.0,
        ], dtype=np.float32)
        wait_norm = float(np.clip(wait_s / max(sub_env.horizon_s, 1.0), 0.0, 1.0))
        slack_norm = float(np.clip(slack_s / max(sub_env.horizon_s, 1.0), 0.0, 1.0))
        return mission_feats, wait_norm, slack_norm

    def _dynamic_response_state(self, env, sub_env) -> List[float]:
        current_time = float(sub_env.current_time_s)
        target_s = self._dynamic_response_target_s(env, sub_env)
        known_dynamic = []
        arrived_pending = []
        for mission in sub_env.missions[: getattr(env, "max_action_dim", len(sub_env.missions))]:
            if mission is None or not getattr(mission, "is_dynamic", False):
                continue
            known_dynamic.append(mission)
            if mission.is_observed:
                continue
            arrival_s = float(getattr(mission, "arrival_time_s", mission.earliest_time_s))
            if arrival_s <= current_time <= float(mission.deadline_s):
                arrived_pending.append(mission)

        age_pressures = []
        deadline_pressures = []
        over_budget = 0
        for mission in arrived_pending:
            arrival_s = float(getattr(mission, "arrival_time_s", mission.earliest_time_s))
            age_s = max(current_time - arrival_s, 0.0)
            age_pressure = float(np.clip(age_s / target_s, 0.0, 1.0))
            age_pressures.append(age_pressure)
            over_budget += int(age_s >= target_s)
            slack_s = max(float(mission.deadline_s) - current_time, 0.0)
            horizon_s = max(float(sub_env.horizon_s), 1.0)
            deadline_pressures.append(1.0 - float(np.clip(slack_s / horizon_s, 0.0, 1.0)))

        pending_frac = len(arrived_pending) / max(len(known_dynamic), 1)
        over_budget_frac = over_budget / max(len(arrived_pending), 1)
        return [
            float(np.clip(pending_frac, 0.0, 1.0)),
            float(np.clip(max(age_pressures, default=0.0), 0.0, 1.0)),
            float(np.clip(max(deadline_pressures, default=0.0), 0.0, 1.0)),
            float(np.clip(over_budget_frac, 0.0, 1.0)),
        ]

    def _dynamic_response_features(self, env, sub_env, mission, wait_s: float) -> np.ndarray:
        if not getattr(mission, "is_dynamic", False):
            return np.zeros(4, dtype=np.float32)
        target_s = self._dynamic_response_target_s(env, sub_env)
        current_time = float(sub_env.current_time_s)
        arrival_s = float(getattr(mission, "arrival_time_s", mission.earliest_time_s))
        age_s = max(current_time - arrival_s, 0.0)
        age_pressure = float(np.clip(age_s / target_s, 0.0, 1.0))
        budget_remaining = float(np.clip((target_s - age_s) / target_s, 0.0, 1.0))
        budget_overrun = 1.0 if age_s >= target_s else 0.0
        post_wait_pressure = float(
            np.clip((age_s + max(float(wait_s), 0.0)) / target_s, 0.0, 1.0)
        )
        return np.array([
            age_pressure,
            budget_remaining,
            budget_overrun,
            post_wait_pressure,
        ], dtype=np.float32)

    def _temporal_window_state(self, env, sub_env, info: Dict) -> List[float]:
        slot_timing = self.candidate_adapter.slot_timing(info)
        slot_types = self.candidate_adapter.slot_types(info)
        task_limit = min(
            len(self.candidate_adapter.candidate_slots(info)),
            getattr(env, "candidate_action_top_k", len(slot_timing)),
        )
        if task_limit <= 0:
            return [0.0, 0.0, 0.0, 0.0]

        future_waits = []
        dynamic_future_waits = []
        future_count = 0
        for idx in range(task_limit):
            timing = slot_timing[idx] if idx < len(slot_timing) else {}
            slot_type = slot_types[idx] if idx < len(slot_types) else ""
            is_future = bool(float(timing.get("future_executable", 0.0)) > 0.0)
            future_count += int(is_future)
            if is_future:
                wait_s = float(timing.get("wait_norm", 0.0)) * max(float(sub_env.horizon_s), 1.0)
                future_waits.append(wait_s)
                if str(slot_type) == "dynamic":
                    dynamic_future_waits.append(wait_s)

        target_s = self._dynamic_response_target_s(env, sub_env)
        min_future_wait = min(future_waits, default=float(sub_env.horizon_s))
        min_dynamic_future_wait = min(dynamic_future_waits, default=float(sub_env.horizon_s))
        return [
            float(np.clip(future_count / max(task_limit, 1), 0.0, 1.0)),
            float(np.clip(len(dynamic_future_waits) / max(future_count, 1), 0.0, 1.0)),
            float(np.clip(min_future_wait / max(float(sub_env.horizon_s), 1.0), 0.0, 1.0)),
            float(np.clip(min_dynamic_future_wait / target_s, 0.0, 1.0)),
        ]

    def _temporal_window_features(self, env, sub_env, mission) -> np.ndarray:
        v2_cfg = getattr(env, "v2_cfg", None)
        response_target_s = float(getattr(v2_cfg, "dynamic_response_target_s", 3600.0) or 3600.0)
        downlink_queue_target_s = float(getattr(v2_cfg, "downlink_queue_target_s", 3600.0) or 3600.0)
        downlink_fn = None
        if self.candidate_scorer is not None:
            downlink_fn = self.candidate_scorer.heuristic._downlink_features
        elif v2_cfg is not None and not bool(getattr(v2_cfg, "downlink_aware_candidate_score", True)):
            downlink_fn = self._no_downlink_temporal_features
        return temporal_window_features(
            env=sub_env,
            mission=mission,
            current_time_s=float(sub_env.current_time_s),
            top_k=self.temporal_window_top_k,
            response_target_s=response_target_s,
            downlink_queue_target_s=downlink_queue_target_s,
            downlink_feature_fn=downlink_fn,
            use_early_delivery_features=self.use_early_delivery_temporal_features,
            early_delivery_weight=self.temporal_early_delivery_weight,
        )

    @staticmethod
    def _no_downlink_temporal_features(env, mission, obs_end_s: float) -> Tuple[float, float, float]:
        origin_s = (
            float(getattr(mission, "arrival_time_s", mission.earliest_time_s))
            if getattr(mission, "is_dynamic", False)
            else float(mission.earliest_time_s)
        )
        return 0.0, max(float(obs_end_s) - origin_s, 0.0), 1.0

    def _state_with_history(
        self,
        env,
        agent_id: str,
        state: np.ndarray,
        current_time_s: float,
    ) -> np.ndarray:
        if self.temporal_state_history_len <= 1:
            return self._fit_state_dim(state)

        key = (id(env), str(agent_id))
        last_time = self._state_history_last_time.get(key)
        if last_time is None or current_time_s < last_time - 1e-6:
            self._state_history[key] = []
        history = self._state_history.setdefault(key, [])
        history.append(np.asarray(state, dtype=np.float32))
        if len(history) > self.temporal_state_history_len:
            del history[:-self.temporal_state_history_len]
        self._state_history_last_time[key] = float(current_time_s)

        pad_count = self.temporal_state_history_len - len(history)
        padded = [np.zeros(self.state_base_dim, dtype=np.float32) for _ in range(max(pad_count, 0))]
        flat = np.concatenate([*padded, *history], axis=0).astype(np.float32)
        return self._fit_state_dim(flat)

    def _fit_state_dim(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if len(state) == self.state_dim:
            return state
        out = np.zeros(self.state_dim, dtype=np.float32)
        out[: min(len(state), self.state_dim)] = state[: self.state_dim]
        return out

    @staticmethod
    def _dynamic_response_target_s(env, sub_env) -> float:
        v2_cfg = getattr(env, "v2_cfg", None)
        target_s = float(getattr(v2_cfg, "dynamic_response_target_s", 0.0) or 0.0)
        if target_s <= 0.0:
            target_s = float(getattr(sub_env, "horizon_s", 1.0))
        return max(target_s, 1.0)

    @staticmethod
    def _set_type(row: np.ndarray, action_type: str) -> None:
        if action_type == "task":
            row[0] = 1.0
        elif action_type == "transfer":
            row[1] = 1.0
        elif action_type == "idle":
            row[2] = 1.0

    @staticmethod
    def _set_slot_type(row: np.ndarray, slot_type: str) -> None:
        idx = SLOT_TYPES.get(str(slot_type), SLOT_TYPES["empty"])
        row[5 + idx] = 1.0
