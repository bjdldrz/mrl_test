from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .candidate_scorer import EDGE_FEATURE_DIM
from .env_adapter import V2CandidateAdapter


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
        state_dim: int = 12,
        action_feature_dim: int = 24,
        mode: str = "full",
        use_candidate_score: bool = True,
        candidate_scorer=None,
        candidate_adapter=None,
    ):
        self.state_dim = int(state_dim)
        self.action_feature_dim = int(action_feature_dim)
        self.mode = str(mode)
        self.use_candidate_score = bool(use_candidate_score and self.mode != "no_score")
        self.candidate_scorer = candidate_scorer
        self.candidate_adapter = candidate_adapter or V2CandidateAdapter()

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
        state = np.zeros(self.state_dim, dtype=np.float32)
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
        state[: min(len(values), self.state_dim)] = values[: self.state_dim]
        return state

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
            task_id_out[exposed_action] = int(mission.id)
            mission_feats, wait_norm, slack_norm = self._mission_features(sub_env, mission)
            out[exposed_action, 9:16] = mission_feats
            out[exposed_action, 16] = slack_norm
            out[exposed_action, 17] = wait_norm
            owners = self.candidate_adapter.candidate_owners(env, mission.id)
            out[exposed_action, 18] = 1.0 if agent_id in owners else 0.0
            out[exposed_action, 19] = 1.0 if getattr(mission, "is_dynamic", False) else 0.0
            out[exposed_action, 20] = exposed_action / max(task_limit - 1, 1)
            out[exposed_action, 21] = float(getattr(mission, "priority", 0.0)) / 10.0
            out[exposed_action, 22] = 1.0 if getattr(mission, "is_observed", False) else 0.0
            out[exposed_action, 23] = float(raw_action) / max(env.max_action_dim - 1, 1)
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
