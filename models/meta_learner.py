"""
LSTM 元学习器 (外循环)
=======================
实现论文 Section 3.5.3:
  - LSTM 接收任务反馈向量 x_t (Eq.24)
  - 嵌入网络 g_η 压缩高维反馈 (Eq.25)
  - 两个调制头 H_θ 和 H_ϕ 微调 Actor-Critic 参数 (Eq.26)
  - 元目标 L_meta 跨任务优化初始化参数 (Eq.27-29)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class FeedbackEncoder(nn.Module):
    """
    嵌入网络 g_η (论文 Eq.24-25)。
    将任务反馈向量 x_t 压缩到紧凑的隐空间。
    """
    def __init__(self, input_dim: int, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModulationHead(nn.Module):
    """
    参数调制头 H_θ 或 H_ϕ (论文 Eq.26)。
    将 LSTM 隐状态映射为 Actor/Critic 网络的参数偏移量。

    采用 FiLM (Feature-wise Linear Modulation) 风格:
      θ_t^(0) = θ_base ⊕ H_θ(h_t)
    其中 ⊕ 表示逐层组合 (scale + shift)。
    """
    def __init__(self, lstm_hidden_dim: int, target_param_shapes: List[Tuple[str, int]]):
        """
        参数
        ----
        lstm_hidden_dim : int
            LSTM 隐藏层维度
        target_param_shapes : List[Tuple[str, int]]
            要调制的参数列表: [(参数名, 参数元素总数), ...]
        """
        super().__init__()
        self.target_shapes = target_param_shapes

        # 为每个目标参数生成 scale 和 shift
        self.scale_heads = nn.ModuleDict()
        self.shift_heads = nn.ModuleDict()

        for name, n_params in target_param_shapes:
            safe_name = name.replace('.', '_')
            # 只调制偏置项, 大幅减少参数量
            self.scale_heads[safe_name] = nn.Sequential(
                nn.Linear(lstm_hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, n_params),
                nn.Sigmoid(),  # scale ∈ (0, 1), 中心化后 ∈ (0.5, 1.5)
            )
            self.shift_heads[safe_name] = nn.Sequential(
                nn.Linear(lstm_hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, n_params),
                nn.Tanh(),  # shift ∈ (-1, 1), 缩放后幅度可控
            )

    def forward(self, h: torch.Tensor) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        返回 {参数名: (scale, shift)} 字典
        """
        modulations = {}
        for name, _ in self.target_shapes:
            safe_name = name.replace('.', '_')
            scale = self.scale_heads[safe_name](h) + 0.5  # ∈ (0.5, 1.5)
            shift = self.shift_heads[safe_name](h) * 0.1  # 小幅偏移
            modulations[name] = (scale, shift)
        return modulations


class MetaLearner(nn.Module):
    """
    完整的 LSTM 元学习器 (论文 Section 3.5.3, Algorithm 1)。

    职责:
    1. 维护跨任务的递归记忆 (h_t, c_t)
    2. 根据任务反馈调制 Actor-Critic 的初始化参数
    3. 计算元损失 L_meta
    """

    def __init__(
        self,
        feedback_dim: int,
        lstm_hidden_dim: int = 128,
        feedback_embed_dim: int = 64,
        actor_critic_config: Optional[Dict] = None,
    ):
        """
        参数
        ----
        feedback_dim : int
            原始反馈向量 x_t 的维度
        lstm_hidden_dim : int
            LSTM 隐藏层维度
        feedback_embed_dim : int
            g_η 嵌入后的维度
        actor_critic_config : dict
            Actor-Critic 网络配置, 用于确定需要调制哪些参数
        """
        super().__init__()
        self.lstm_hidden_dim = lstm_hidden_dim

        # 嵌入网络 g_η
        self.feedback_encoder = FeedbackEncoder(feedback_dim, feedback_embed_dim)

        # LSTM (论文 Eq.25)
        self.lstm = nn.LSTM(
            input_size=feedback_embed_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # 调制头的参数规格将在 attach_to_actor_critic 中动态确定
        self.actor_modulation = None
        self.critic_modulation = None

        # LSTM 状态
        self._hidden = None
        self._cell = None

    def attach_to_actor_critic(self, actor_critic: nn.Module):
        """
        根据实际的 Actor-Critic 网络结构, 初始化调制头。
        只调制偏置项 (bias) 以减少参数量。
        """
        actor_params = []
        critic_params = []

        for name, param in actor_critic.named_parameters():
            if 'bias' not in name:
                continue  # 只调制 bias
            if 'actor' in name:
                actor_params.append((name, param.numel()))
            elif 'critic' in name:
                critic_params.append((name, param.numel()))
            else:
                # 共享层的 bias 归入 actor 调制
                actor_params.append((name, param.numel()))

        if actor_params:
            self.actor_modulation = ModulationHead(
                self.lstm_hidden_dim, actor_params
            )
        if critic_params:
            self.critic_modulation = ModulationHead(
                self.lstm_hidden_dim, critic_params
            )

        # 将新建的子模块移动到与已有参数相同的设备
        target_device = next(self.parameters()).device
        if self.actor_modulation is not None:
            self.actor_modulation.to(target_device)
        if self.critic_modulation is not None:
            self.critic_modulation.to(target_device)

    def reset_hidden(self, batch_size: int = 1, device: torch.device = None):
        """重置 LSTM 隐状态 (论文 Algorithm 1, line 4)"""
        device = device or next(self.parameters()).device
        self._hidden = torch.zeros(1, batch_size, self.lstm_hidden_dim, device=device)
        self._cell = torch.zeros(1, batch_size, self.lstm_hidden_dim, device=device)

    def forward(self, feedback: torch.Tensor) -> torch.Tensor:
        """
        处理一步任务反馈, 更新 LSTM 状态, 返回隐状态 h_t。

        参数
        ----
        feedback : Tensor [batch, feedback_dim]
            任务反馈向量 x_t (论文 Eq.24)

        返回
        ----
        h_t : Tensor [batch, lstm_hidden_dim]
        """
        # g_η(x_t)
        z = self.feedback_encoder(feedback)  # [batch, embed_dim]
        z = z.unsqueeze(1)  # [batch, 1, embed_dim] for LSTM

        # LSTM 更新 (Eq.25)
        if self._hidden is None:
            self.reset_hidden(feedback.size(0), feedback.device)

        # 截断跨任务的计算图, 只保留隐状态的值
        h_in = self._hidden.detach()
        c_in = self._cell.detach()

        lstm_out, (self._hidden, self._cell) = self.lstm(
            z, (h_in, c_in)
        )

        h_t = self._hidden.squeeze(0)  # [batch, lstm_hidden_dim]
        return h_t

    def get_modulations(
        self, h_t: torch.Tensor
    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        通过调制头生成参数偏移 (论文 Eq.26)。

        返回
        ----
        actor_mods : {param_name: (scale, shift)}
        critic_mods : {param_name: (scale, shift)}
        """
        actor_mods = None
        critic_mods = None
        if self.actor_modulation is not None:
            actor_mods = self.actor_modulation(h_t)
        if self.critic_modulation is not None:
            critic_mods = self.critic_modulation(h_t)
        return actor_mods, critic_mods

    def apply_modulations(
        self,
        actor_critic: nn.Module,
        actor_mods: Optional[Dict],
        critic_mods: Optional[Dict],
    ):
        """
        将调制应用到 Actor-Critic 网络参数上:
        θ_t^(0) = θ_base ⊕ H_θ(h_t) (论文 Eq.26)
        """
        all_mods = {}
        if actor_mods:
            all_mods.update(actor_mods)
        if critic_mods:
            all_mods.update(critic_mods)

        for name, param in actor_critic.named_parameters():
            if name in all_mods:
                scale, shift = all_mods[name]
                # 确保维度匹配
                scale = scale.view_as(param.data)
                shift = shift.view_as(param.data)
                param.data = param.data * scale + shift

    @staticmethod
    def build_feedback_vector(
        cumulative_reward: float,
        avg_advantage: float,
        policy_entropy: float,
        kl_divergence: float,
        dynamic_ratio: float,
        n_dynamic_completed: int,
        n_routine_completed: int,
    ) -> np.ndarray:
        """
        构建任务反馈向量 x_t (论文 Eq.24)。

        x_t = [R(T_t), ϕ(s_t, a_{t-1}, r_{t-1}, d_{t-1})]

        返回 numpy 数组, 外部需转为 tensor。
        """
        return np.array([
            cumulative_reward / 1000.0,   # 归一化
            avg_advantage,
            policy_entropy,
            kl_divergence,
            dynamic_ratio,
            n_dynamic_completed / 100.0,
            n_routine_completed / 100.0,
        ], dtype=np.float32)
