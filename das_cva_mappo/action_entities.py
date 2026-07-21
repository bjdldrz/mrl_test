from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


ACTION_OBSERVE = 0
ACTION_RELAY = 1
ACTION_WAIT = 2

TARGET_TASK = 0
TARGET_SATELLITE = 1
TARGET_NONE = 2


@dataclass
class ActionEntity:
    """Semantic action entity used before packing into fixed-size tensors."""

    action_type: int
    target_type: int
    target_id: int
    feasible: bool
    features: np.ndarray
    edge_features: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.action_type = int(self.action_type)
        self.target_type = int(self.target_type)
        self.target_id = int(self.target_id)
        self.feasible = bool(self.feasible)
        self.features = np.asarray(self.features, dtype=np.float32)
        if self.edge_features is not None:
            self.edge_features = np.asarray(self.edge_features, dtype=np.float32)


@dataclass
class EdgeDecisionRecord:
    """Selected observe-edge snapshot for candidate value supervision."""

    decision_id: int
    time_s: float
    agent_id: str
    action_idx: int
    action_type: int
    target_id: int
    edge_features: np.ndarray
    target: Optional[float] = None
    predicted_value: Optional[float] = None

    def __post_init__(self) -> None:
        self.decision_id = int(self.decision_id)
        self.time_s = float(self.time_s)
        self.agent_id = str(self.agent_id)
        self.action_idx = int(self.action_idx)
        self.action_type = int(self.action_type)
        self.target_id = int(self.target_id)
        self.edge_features = np.asarray(self.edge_features, dtype=np.float32)
        if self.target is not None:
            self.target = float(self.target)
        if self.predicted_value is not None:
            self.predicted_value = float(self.predicted_value)
