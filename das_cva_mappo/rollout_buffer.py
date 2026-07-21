from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class ActionSetRolloutBuffer:
    """MAPPO buffer with action-set snapshots captured at sampling time."""

    state_obs: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    action_features: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    candidate_edge_features: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    candidate_task_ids: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    action_masks: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    decision_times: Dict[str, List[float]] = field(default_factory=dict)
    actions: Dict[str, List[int]] = field(default_factory=dict)
    log_probs: Dict[str, List[float]] = field(default_factory=dict)
    rewards: Dict[str, List[float]] = field(default_factory=dict)
    dones: Dict[str, List[bool]] = field(default_factory=dict)
    global_states: List[np.ndarray] = field(default_factory=list)
    values: List[float] = field(default_factory=list)

    def init_agents(self, agent_ids: List[str]) -> None:
        for aid in agent_ids:
            self.state_obs[aid] = []
            self.action_features[aid] = []
            self.candidate_edge_features[aid] = []
            self.candidate_task_ids[aid] = []
            self.action_masks[aid] = []
            self.decision_times[aid] = []
            self.actions[aid] = []
            self.log_probs[aid] = []
            self.rewards[aid] = []
            self.dones[aid] = []

    def add_step_value(self, global_state: np.ndarray, value: float) -> None:
        self.global_states.append(np.asarray(global_state, dtype=np.float32))
        self.values.append(float(value))

    def add_agent(
        self,
        agent_id: str,
        state_obs: np.ndarray,
        action_features: np.ndarray,
        candidate_edge_features: np.ndarray,
        candidate_task_ids: np.ndarray,
        action_mask: np.ndarray,
        action: int,
        log_prob: float,
        decision_time_s: float = 0.0,
        reward: float = 0.0,
        done: bool = False,
    ) -> None:
        self.state_obs[agent_id].append(np.asarray(state_obs, dtype=np.float32))
        self.action_features[agent_id].append(np.asarray(action_features, dtype=np.float32))
        self.candidate_edge_features[agent_id].append(np.asarray(candidate_edge_features, dtype=np.float32))
        self.candidate_task_ids[agent_id].append(np.asarray(candidate_task_ids, dtype=np.int64))
        self.action_masks[agent_id].append(np.asarray(action_mask, dtype=np.float32))
        self.decision_times[agent_id].append(float(decision_time_s))
        self.actions[agent_id].append(int(action))
        self.log_probs[agent_id].append(float(log_prob))
        self.rewards[agent_id].append(float(reward))
        self.dones[agent_id].append(bool(done))

    def __len__(self) -> int:
        return len(self.global_states)
