from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def _mlp(input_dim: int, hidden_dims: Iterable[int], output_dim: int) -> nn.Sequential:
    layers = []
    last = input_dim
    for hidden in hidden_dims:
        layers.append(nn.Linear(last, int(hidden)))
        layers.append(nn.ReLU())
        last = int(hidden)
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


class ActionSetActor(nn.Module):
    """Shared policy that scores each action entity in the current action set."""

    def __init__(
        self,
        state_dim: int,
        action_feature_dim: int,
        hidden_dims=(256, 256),
        action_hidden_dim: int = 128,
        matcher: str = "set_transformer",
        use_set_context: bool = True,
        use_action_type_gate: bool = True,
        idle_valid_penalty: float = 2.0,
    ):
        super().__init__()
        self.matcher = matcher
        self.use_set_context = bool(use_set_context)
        self.use_action_type_gate = bool(use_action_type_gate)
        self.idle_valid_penalty = max(0.0, float(idle_valid_penalty))
        self.state_encoder = _mlp(state_dim, hidden_dims, action_hidden_dim)
        self.action_encoder = _mlp(action_feature_dim, hidden_dims[:1], action_hidden_dim)
        if matcher == "additive":
            self.state_proj = nn.Linear(action_hidden_dim, action_hidden_dim)
            self.action_proj = nn.Linear(action_hidden_dim, action_hidden_dim)
            self.context_proj = nn.Linear(action_hidden_dim, action_hidden_dim)
            self.logit_head = nn.Linear(action_hidden_dim, 1)
        elif matcher == "dot":
            self.state_proj = nn.Linear(action_hidden_dim, action_hidden_dim)
            self.action_proj = nn.Linear(action_hidden_dim, action_hidden_dim)
        elif matcher == "set_transformer":
            nhead = self._attention_heads(action_hidden_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=action_hidden_dim,
                nhead=nhead,
                dim_feedforward=action_hidden_dim * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
            )
            self.state_token_proj = nn.Linear(action_hidden_dim, action_hidden_dim)
            self.set_encoder = nn.TransformerEncoder(layer, num_layers=2)
            self.context_proj = nn.Linear(action_hidden_dim, action_hidden_dim)
            self.logit_head = nn.Linear(action_hidden_dim, 1)
        else:
            raise ValueError("matcher must be additive, dot, or set_transformer")
        if self.use_action_type_gate:
            self.type_gate_head = nn.Linear(action_hidden_dim, 5)
        self._init_weights()

    @staticmethod
    def _attention_heads(hidden_dim: int) -> int:
        for nhead in (8, 4, 2):
            if hidden_dim % nhead == 0:
                return nhead
        return 1

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        if hasattr(self, "logit_head"):
            nn.init.orthogonal_(self.logit_head.weight, gain=0.01)
        if hasattr(self, "type_gate_head"):
            nn.init.constant_(self.type_gate_head.weight, 0.0)
            nn.init.constant_(self.type_gate_head.bias, 0.0)

    def forward(
        self,
        state: torch.Tensor,
        action_features: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Categorical:
        state_h = self.state_encoder(state)
        action_h = self.action_encoder(action_features)
        if action_mask is None:
            action_mask = torch.ones(action_features.shape[:2], device=action_features.device)

        if self.matcher == "dot":
            query = self.state_proj(state_h).unsqueeze(1)
            keys = self.action_proj(action_h)
            logits = torch.sum(query * keys, dim=-1) / np.sqrt(keys.shape[-1])
            gate_context = state_h
        elif self.matcher == "set_transformer":
            # Let filled future-task tokens contribute context even when they
            # are not currently selectable. The final action mask still gates
            # logits, so invalid/future slots cannot be sampled.
            if self.use_set_context:
                present = (torch.sum(torch.abs(action_features), dim=-1) > 0) | (action_mask > 0)
                state_token = self.state_token_proj(state_h).unsqueeze(1)
                tokens = torch.cat([state_token, action_h], dim=1)
                state_present = torch.ones(
                    (present.shape[0], 1),
                    dtype=torch.bool,
                    device=present.device,
                )
                token_present = torch.cat([state_present, present], dim=1)
                encoded = self.set_encoder(tokens, src_key_padding_mask=~token_present)
                global_context = self.context_proj(encoded[:, 0]).unsqueeze(1)
                action_context = encoded[:, 1:]
                gate_context = encoded[:, 0]
            else:
                global_context = self.context_proj(state_h).unsqueeze(1)
                action_context = action_h
                gate_context = state_h
            logits = self.logit_head(torch.tanh(action_context + global_context)).squeeze(-1)
        else:
            state_term = self.state_proj(state_h).unsqueeze(1)
            action_term = self.action_proj(action_h)
            if self.use_set_context:
                denom = torch.clamp(action_mask.sum(dim=1, keepdim=True), min=1.0)
                context = (action_h * action_mask.unsqueeze(-1)).sum(dim=1) / denom
                context_term = self.context_proj(context).unsqueeze(1)
                gate_context = state_h + context
            else:
                context_term = 0.0
                gate_context = state_h
            logits = self.logit_head(torch.tanh(state_term + action_term + context_term)).squeeze(-1)

        if self.use_action_type_gate:
            logits = logits + self._action_type_gate(gate_context, action_features)
        if self.idle_valid_penalty > 0:
            logits = logits - self._idle_valid_penalty(action_features, action_mask)
        logits = logits + (1.0 - action_mask) * (-1e8)
        return Categorical(logits=logits)

    def _action_type_gate(self, context: torch.Tensor, action_features: torch.Tensor) -> torch.Tensor:
        """Add learnable routine/dynamic/flex/transfer/idle mode logits."""
        gate_logits = self.type_gate_head(context)

        def feat(idx: int) -> torch.Tensor:
            if action_features.shape[-1] <= idx:
                return torch.zeros(action_features.shape[:2], device=action_features.device)
            return torch.clamp(action_features[..., idx], 0.0, 1.0)

        task = feat(0)
        transfer = feat(1)
        idle = feat(2)
        slot_routine = feat(5)
        slot_dynamic = feat(6)
        slot_flex = feat(7)
        mission_dynamic = feat(19)
        dynamic = task * torch.clamp(slot_dynamic + mission_dynamic, 0.0, 1.0)
        routine = task * slot_routine * (1.0 - mission_dynamic)
        flex = task * slot_flex
        weights = torch.stack([routine, dynamic, flex, transfer, idle], dim=-1)
        weights = weights / torch.clamp(weights.sum(dim=-1, keepdim=True), min=1.0)
        return torch.sum(weights * gate_logits.unsqueeze(1), dim=-1)

    def _idle_valid_penalty(self, action_features: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        """Penalize idle only when a valid non-idle action is available."""
        if action_features.shape[-1] <= 2:
            return torch.zeros(action_mask.shape, device=action_mask.device)
        idle = torch.clamp(action_features[..., 2], 0.0, 1.0)
        valid_non_idle = torch.sum(action_mask * (1.0 - idle), dim=1, keepdim=True) > 0
        return idle * valid_non_idle.to(action_mask.dtype) * self.idle_valid_penalty

    def get_action(
        self,
        state: torch.Tensor,
        action_features: torch.Tensor,
        action_mask: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.forward(state, action_features, action_mask)
        if action is None:
            action = torch.argmax(dist.logits, dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy()


class SetCritic(nn.Module):
    def __init__(self, global_state_dim: int, hidden_dims=(256, 256)):
        super().__init__()
        self.net = _mlp(global_state_dim, hidden_dims, 1)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.net(global_state).squeeze(-1)


class ActionSetActorCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_feature_dim: int,
        global_state_dim: int,
        actor_hidden_dims=(256, 256),
        action_hidden_dim: int = 128,
        critic_hidden_dims=(256, 256),
        matcher: str = "set_transformer",
        use_set_context: bool = True,
        use_action_type_gate: bool = True,
        idle_valid_penalty: float = 2.0,
    ):
        super().__init__()
        self.actor = ActionSetActor(
            state_dim=state_dim,
            action_feature_dim=action_feature_dim,
            hidden_dims=tuple(actor_hidden_dims),
            action_hidden_dim=action_hidden_dim,
            matcher=matcher,
            use_set_context=use_set_context,
            use_action_type_gate=use_action_type_gate,
            idle_valid_penalty=idle_valid_penalty,
        )
        self.critic = SetCritic(global_state_dim, tuple(critic_hidden_dims))

    def get_values(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.critic(global_state)
