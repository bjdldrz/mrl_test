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
    dynamic_candidate_owners: int = 6
    urgent_candidate_owners: int = 6
    stale_candidate_owners: int = 6
    capacity_slack_ratio: float = 0.05
    load_penalty: float = 0.15
    switch_penalty: float = 0.05
    owner_switch_margin: float = 0.08
    ownership_mask_mode: str = "soft"
    candidate_owner_bonus: float = 0.06
    slot_selection_mode: str = "typed"
    executable_slot_reserve_ratio: float = 0.5
    allow_future_task_execution: bool = True
    future_task_requires_no_current_valid: bool = False
    future_task_max_wait_s: float = 600.0
    future_routine_max_wait_s: float = 180.0
    routine_future_dynamic_guard_s: float = 1800.0
    routine_future_dynamic_penalty: float = 0.35
    dynamic_future_bonus: float = 0.25
    drop_ineligible_future_candidates: bool = True

    # Event-triggered candidate repair.
    replan_interval_s: float = 3600.0
    replan_horizon_s: float = 21600.0
    release_before_deadline_s: float = 3600.0
    dynamic_broadcast_window_s: float = 3600.0
    lock_window_s: float = 600.0
    max_switches_per_task: int = 2
    triggers: Tuple[str, ...] = ("periodic", "dynamic", "stale_owner", "deadline")

    # Candidate value score weights.
    w_quality: float = 0.42
    w_priority: float = 0.18
    w_deadline: float = 0.14
    w_dynamic: float = 0.18
    w_scarcity: float = 0.10
    w_future_opportunity_loss: float = 0.08
    w_load: float = 0.16
    w_owner_stability: float = 0.04
    w_wait: float = 0.08
    w_storage_pressure: float = 0.08
    w_dynamic_urgency: float = 0.12
    w_dynamic_response: float = 0.24
    w_dynamic_wait: float = 0.20
    dynamic_response_target_s: float = 3600.0

    # Allocator repair weights. These bias ownership toward agents with earlier
    # feasible windows, especially for stale owners and urgent dynamic tasks.
    allocator_wait_penalty: float = 0.10
    allocator_stale_rescue_bonus: float = 0.25
    allocator_dynamic_urgency_bonus: float = 0.10
    allocator_dynamic_response_bonus: float = 0.24
    allocator_dynamic_wait_penalty: float = 0.20
    dynamic_rescue_response_bonus: float = 1.0
    dynamic_takeover_margin_s: float = 300.0

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
        if self.slot_selection_mode not in {"mixed", "typed"}:
            raise ValueError("slot_selection_mode 必须是 mixed 或 typed")
        if not 0.0 <= float(self.executable_slot_reserve_ratio) <= 1.0:
            raise ValueError("executable_slot_reserve_ratio 必须在 [0, 1]")
        if self.future_task_max_wait_s < 0:
            raise ValueError("future_task_max_wait_s 必须大于等于 0")
        if self.future_routine_max_wait_s < 0:
            raise ValueError("future_routine_max_wait_s 必须大于等于 0")
        for name in [
            "w_wait",
            "w_storage_pressure",
            "w_dynamic_urgency",
            "w_dynamic_response",
            "w_dynamic_wait",
            "dynamic_response_target_s",
            "allocator_wait_penalty",
            "allocator_stale_rescue_bonus",
            "allocator_dynamic_urgency_bonus",
            "allocator_dynamic_response_bonus",
            "allocator_dynamic_wait_penalty",
            "dynamic_rescue_response_bonus",
            "dynamic_takeover_margin_s",
            "routine_future_dynamic_guard_s",
            "routine_future_dynamic_penalty",
            "dynamic_future_bonus",
        ]:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} 必须大于等于 0")
