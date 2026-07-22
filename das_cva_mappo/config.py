from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

from .temporal_features import TEMPORAL_WINDOW_FEATURE_DIM


BASE_STATE_DIM = 20
BASE_ACTION_FEATURE_DIM = 28 + TEMPORAL_WINDOW_FEATURE_DIM


@dataclass
class DASConfig:
    """Runtime options for the DAS-CVA-MAPPO implementation.

    The current version uses the v2 environment as a compatibility layer, but
    keeps the main method logic in DAS-specific action and candidate modules.
    """

    version: str = "0.33.0"
    state_base_dim: int = BASE_STATE_DIM
    state_dim: int = BASE_STATE_DIM
    action_feature_dim: int = BASE_ACTION_FEATURE_DIM
    actor_hidden_dims: Tuple[int, ...] = (256, 256)
    action_hidden_dim: int = 128
    critic_hidden_dims: Tuple[int, ...] = (256, 256)
    matcher: str = "set_transformer"
    action_feature_mode: str = "full"
    use_candidate_score_feature: bool = True
    use_set_context: bool = True
    use_action_type_gate: bool = True
    use_response_budget_features: bool = True
    use_temporal_window_features: bool = True
    use_early_delivery_temporal_features: bool = True
    temporal_window_top_k: int = 3
    temporal_early_delivery_weight: float = 0.35
    temporal_state_encoder: str = "mlp"
    temporal_state_history_len: int = 1
    idle_valid_penalty: float = 0.0
    idle_aux_coeff: float = 0.05
    dynamic_select_aux_coeff: float = 0.0
    dynamic_task_logit_bonus: float = 0.0
    dynamic_current_logit_bonus: float = 0.0
    routine_task_logit_penalty: float = 0.0
    candidate_dropout_prob: float = 0.0

    candidate_scorer_mode: str = "hybrid"
    candidate_scorer_mix: float = 0.35
    candidate_scorer_mix_start: Optional[float] = None
    candidate_scorer_mix_end: Optional[float] = None
    candidate_scorer_mix_anneal_epochs: int = 0
    candidate_scorer_hidden_dim: int = 64
    candidate_scorer_lr: float = 1e-3
    candidate_warmup_edges: int = 4096
    candidate_warmup_epochs: int = 2
    candidate_warmup_batch_size: int = 256
    candidate_aux_update: bool = True
    candidate_aux_epochs: int = 1
    candidate_aux_batch_size: int = 256
    candidate_aux_rank_weight: float = 0.2
    candidate_aux_target_clip: float = 3.0
    candidate_aux_min_edges: int = 4
    candidate_hard_negative_samples: int = 2
    candidate_hard_negative_valid_only: bool = True
    candidate_hard_negative_margin: float = 0.25
    candidate_hard_negative_value_weight: float = 0.5
    candidate_aux_conflict_penalty: float = 0.5
    candidate_aux_load_penalty: float = 0.1
    candidate_adapter_mode: str = "v2_compat"

    supported_matchers: Tuple[str, ...] = field(default=("additive", "dot", "set_transformer"), init=False)
    supported_feature_modes: Tuple[str, ...] = field(default=("full", "minimal", "no_score"), init=False)
    supported_scorers: Tuple[str, ...] = field(default=("v2_heuristic", "learned", "hybrid"), init=False)
    supported_adapters: Tuple[str, ...] = field(default=("v2_compat",), init=False)
    supported_temporal_state_encoders: Tuple[str, ...] = field(default=("mlp", "gru"), init=False)

    def validate(self) -> None:
        self.temporal_window_top_k = max(int(self.temporal_window_top_k), 1)
        self.temporal_early_delivery_weight = max(float(self.temporal_early_delivery_weight), 0.0)
        self.temporal_state_history_len = max(int(self.temporal_state_history_len), 1)
        self.state_base_dim = max(int(self.state_base_dim), 1)
        self.state_dim = (
            self.state_base_dim * self.temporal_state_history_len
            if self.temporal_state_encoder == "gru"
            else self.state_base_dim
        )
        if self.matcher not in self.supported_matchers:
            raise ValueError(f"matcher must be one of {self.supported_matchers}")
        if self.action_feature_mode not in self.supported_feature_modes:
            raise ValueError(f"action_feature_mode must be one of {self.supported_feature_modes}")
        if self.candidate_scorer_mode not in self.supported_scorers:
            raise ValueError(f"candidate_scorer_mode must be one of {self.supported_scorers}")
        if self.candidate_adapter_mode not in self.supported_adapters:
            raise ValueError(f"candidate_adapter_mode must be one of {self.supported_adapters}")
        if self.temporal_state_encoder not in self.supported_temporal_state_encoders:
            raise ValueError(
                f"temporal_state_encoder must be one of {self.supported_temporal_state_encoders}"
            )
        if not 0.0 <= float(self.candidate_dropout_prob) < 1.0:
            raise ValueError("candidate_dropout_prob must be in [0, 1)")
        if float(self.idle_valid_penalty) < 0:
            raise ValueError("idle_valid_penalty must be non-negative")
        if float(self.idle_aux_coeff) < 0:
            raise ValueError("idle_aux_coeff must be non-negative")
        if float(self.dynamic_select_aux_coeff) < 0:
            raise ValueError("dynamic_select_aux_coeff must be non-negative")
        if float(self.dynamic_task_logit_bonus) < 0:
            raise ValueError("dynamic_task_logit_bonus must be non-negative")
        if float(self.dynamic_current_logit_bonus) < 0:
            raise ValueError("dynamic_current_logit_bonus must be non-negative")
        if float(self.routine_task_logit_penalty) < 0:
            raise ValueError("routine_task_logit_penalty must be non-negative")
        if not 0.0 <= float(self.candidate_scorer_mix) <= 1.0:
            raise ValueError("candidate_scorer_mix must be in [0, 1]")
        for name in ("candidate_scorer_mix_start", "candidate_scorer_mix_end"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.candidate_scorer_mix_anneal_epochs < 0:
            raise ValueError("candidate_scorer_mix_anneal_epochs must be non-negative")
        if self.candidate_scorer_hidden_dim <= 0:
            raise ValueError("candidate_scorer_hidden_dim must be positive")
        if self.candidate_scorer_lr <= 0:
            raise ValueError("candidate_scorer_lr must be positive")
        if self.candidate_warmup_edges < 0 or self.candidate_warmup_epochs < 0:
            raise ValueError("candidate warmup edges and epochs must be non-negative")
        if self.candidate_warmup_batch_size <= 0:
            raise ValueError("candidate_warmup_batch_size must be positive")
        if self.candidate_aux_epochs < 0:
            raise ValueError("candidate_aux_epochs must be non-negative")
        if self.candidate_aux_batch_size <= 0:
            raise ValueError("candidate_aux_batch_size must be positive")
        if self.candidate_aux_rank_weight < 0:
            raise ValueError("candidate_aux_rank_weight must be non-negative")
        if self.candidate_aux_target_clip <= 0:
            raise ValueError("candidate_aux_target_clip must be positive")
        if self.candidate_aux_min_edges < 0:
            raise ValueError("candidate_aux_min_edges must be non-negative")
        if self.candidate_hard_negative_samples < 0:
            raise ValueError("candidate_hard_negative_samples must be non-negative")
        if self.candidate_hard_negative_margin < 0:
            raise ValueError("candidate_hard_negative_margin must be non-negative")
        if self.candidate_hard_negative_value_weight < 0:
            raise ValueError("candidate_hard_negative_value_weight must be non-negative")
        if self.candidate_aux_conflict_penalty < 0:
            raise ValueError("candidate_aux_conflict_penalty must be non-negative")
        if self.candidate_aux_load_penalty < 0:
            raise ValueError("candidate_aux_load_penalty must be non-negative")
        if self.state_dim <= 0 or self.action_feature_dim <= 0:
            raise ValueError("state_dim and action_feature_dim must be positive")
