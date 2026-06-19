"""
Actor-Critic 网络
=================
实现论文 Section 3.4 的策略网络:
  - Actor:  π_θ(a|h_t, s_t) = Softmax(f_θ(h_t, s_t)) ⊙ M_t  (Eq.14)
  - Critic: V_μ(s) 状态价值估计

网络结构: 3×256 MLP + ReLU (论文 Table 3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import List, Tuple, Optional
import numpy as np


class ActorCritic(nn.Module):
    """
    带动作掩码的 Actor-Critic 网络。

    Actor 和 Critic 共享底层特征提取, 各有独立的输出头。
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = None,
        activation: str = "relu",
    ):
        """
        参数
        ----
        obs_dim : int
            状态向量维度
        action_dim : int
            动作空间维度 (含 idle 动作)
        hidden_dims : List[int]
            隐藏层维度列表, 默认 [256, 256, 256]
        """
        super().__init__()
        hidden_dims = hidden_dims or [256, 256, 256]

        act_fn = {"relu": nn.ReLU, "tanh": nn.Tanh}[activation]

        # 共享特征提取层
        layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(act_fn())
            in_dim = h_dim
        self.shared = nn.Sequential(*layers)

        # Actor 头: 输出动作 logits
        self.actor_head = nn.Linear(in_dim, action_dim)

        # Critic 头: 输出状态价值
        self.critic_head = nn.Linear(in_dim, 1)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Actor 头用更小的初始化 (鼓励初期探索)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        # Critic 头
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Categorical, torch.Tensor]:
        """
        前向传播。

        参数
        ----
        obs : Tensor [batch, obs_dim]
        action_mask : Tensor [batch, action_dim], 1=可选 0=不可选

        返回
        ----
        dist : Categorical 动作分布 (已掩码)
        value : Tensor [batch, 1] 状态价值
        """
        features = self.shared(obs)

        # Actor: logits → 掩码 → softmax (论文 Eq.14)
        logits = self.actor_head(features)

        if action_mask is not None:
            # 将不可选动作的 logits 设为极小值 (-1e8)
            logits = logits + (1.0 - action_mask) * (-1e8)

        dist = Categorical(logits=logits)

        # Critic
        value = self.critic_head(features)

        return dist, value

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        采样动作并返回所有 PPO 所需的量。

        返回
        ----
        action : Tensor [batch]
        log_prob : Tensor [batch]
        entropy : Tensor [batch]
        value : Tensor [batch]
        """
        dist, value = self.forward(obs, action_mask)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value.squeeze(-1)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """仅计算状态价值 (用于 GAE 计算)"""
        features = self.shared(obs)
        return self.critic_head(features).squeeze(-1)
