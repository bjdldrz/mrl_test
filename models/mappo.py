"""
MAPPO Actor-Critic
==================
CTDE (Centralized Training Decentralized Execution) 架构:

  - Actor (分布式): 每颗卫星基于局部观测选择动作，所有卫星共享参数
  - Critic (集中式): 基于全局状态评估价值，仅训练时使用

参考:
  - Yu et al., "The Surprising Effectiveness of PPO in Cooperative MARL", 2022
  - Hady et al., "MARL for Autonomous Multi-Satellite EO", arXiv 2506.15207
"""

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from typing import Dict, List, Tuple, Optional


class MAPPOActor(nn.Module):
    """
    分布式 Actor (与单智能体版本结构一致)。

    所有卫星共享同一套参数。输入局部观测 + 动作掩码，输出动作分布。
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        hidden_dims = hidden_dims or [256, 256, 256]

        layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, action_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.head.weight, gain=0.01)

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Categorical:
        features = self.net(obs)
        logits = self.head(features)
        if action_mask is not None:
            logits = logits + (1.0 - action_mask) * (-1e8)
        return Categorical(logits=logits)

    def get_action(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回
        ----
        action, log_prob, entropy
        """
        dist = self.forward(obs, action_mask)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()


class MAPPOCritic(nn.Module):
    """
    集中式 Critic: 基于全局状态估计状态价值。

    全局状态 = 所有卫星局部观测的拼接。
    仅在训练阶段使用，部署时丢弃。
    """

    def __init__(self, global_state_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        hidden_dims = hidden_dims or [256, 256]

        layers = []
        in_dim = global_state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.head.weight, gain=1.0)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """
        参数: global_state [batch, global_state_dim]
        返回: value [batch]
        """
        features = self.net(global_state)
        return self.head(features).squeeze(-1)


class MAPPOActorCritic(nn.Module):
    """
    MAPPO 完整模型：共享 Actor + 集中式 Critic。
    """

    def __init__(
        self,
        local_obs_dim: int,
        action_dim: int,
        global_state_dim: int,
        actor_hidden_dims: List[int] = None,
        critic_hidden_dims: List[int] = None,
    ):
        super().__init__()
        self.actor = MAPPOActor(local_obs_dim, action_dim, actor_hidden_dims)
        self.critic = MAPPOCritic(global_state_dim, critic_hidden_dims)

    def get_actions_and_values(
        self,
        local_obs: torch.Tensor,
        action_masks: torch.Tensor,
        global_state: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        为一批智能体同时计算动作和价值。

        参数
        ----
        local_obs : Tensor [batch, obs_dim]        每个智能体的局部观测
        action_masks : Tensor [batch, action_dim]   动作掩码
        global_state : Tensor [batch, global_dim]   全局状态 (Critic 输入)
        actions : Tensor [batch] or None            已知动作 (PPO 更新时)

        返回
        ----
        actions, log_probs, entropy, values
        """
        actions_out, log_probs, entropy = self.actor.get_action(
            local_obs, action_masks, actions
        )
        values = self.critic(global_state)
        return actions_out, log_probs, entropy, values

    def get_values(self, global_state: torch.Tensor) -> torch.Tensor:
        """仅计算全局状态价值 (用于 GAE)"""
        return self.critic(global_state)
