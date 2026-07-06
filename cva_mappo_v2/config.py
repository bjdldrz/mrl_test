from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class CandidateSlotConfig:
    """Typed fixed-size local action space.

    The actor sees K task slots plus one idle action.  Slots are grouped by
    role, but all agents share the same total dimension.
    """

    routine_slots: int = 64
    dynamic_slots: int = 32
    flex_slots: int = 32

    @property
    def total_slots(self) -> int:
        return int(self.routine_slots + self.dynamic_slots + self.flex_slots)


@dataclass
class CVAMAPPOV2Config:
    """High-level CVA-MAPPO v2 settings."""

    slots: CandidateSlotConfig = field(default_factory=CandidateSlotConfig)

    # Task-centered candidate ownership.
    routine_candidate_owners: int = 1
    dynamic_candidate_owners: int = 2
    urgent_candidate_owners: int = 3
    stale_candidate_owners: int = 3
    capacity_slack_ratio: float = 0.05
    load_penalty: float = 0.15
    switch_penalty: float = 0.05
    owner_switch_margin: float = 0.08
    ownership_mask_mode: str = "soft"
    candidate_owner_bonus: float = 0.06

    # Event-triggered candidate repair.
    replan_interval_s: float = 3600.0
    replan_horizon_s: float = 7200.0
    release_before_deadline_s: float = 1800.0
    dynamic_broadcast_window_s: float = 1800.0
    lock_window_s: float = 600.0
    max_switches_per_task: int = 2
    triggers: Tuple[str, ...] = ("periodic", "dynamic", "stale_owner", "deadline")

    # Candidate value score weights.
    w_quality: float = 0.42
    w_priority: float = 0.18
    w_deadline: float = 0.14
    w_dynamic: float = 0.10
    w_scarcity: float = 0.10
    w_future_opportunity_loss: float = 0.08
    w_load: float = 0.16
    w_owner_stability: float = 0.04

    def validate(self) -> None:
        if self.slots.total_slots <= 0:
            raise ValueError("候选槽位总数必须大于 0")
        for name in [
            "routine_candidate_owners",
            "dynamic_candidate_owners",
            "urgent_candidate_owners",
            "stale_candidate_owners",
        ]:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须大于 0")
        if self.owner_switch_margin < 0:
            raise ValueError("owner_switch_margin 必须大于等于 0")
        if self.dynamic_broadcast_window_s < 0:
            raise ValueError("dynamic_broadcast_window_s 必须大于等于 0")
        if self.ownership_mask_mode not in {"hard", "soft"}:
            raise ValueError("ownership_mask_mode 必须是 hard 或 soft")
        if self.candidate_owner_bonus < 0:
            raise ValueError("candidate_owner_bonus 必须大于等于 0")
