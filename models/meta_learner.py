"""
可切换元学习器 (外循环)
=======================
实现论文 Section 3.5.3:
  - 历史反馈编码器接收任务反馈向量 x_t (Eq.24)
  - 嵌入网络 g_η 压缩高维反馈 (Eq.25)
  - 两个调制头 H_θ 和 H_ϕ 微调 Actor-Critic 参数 (Eq.26)
  - 元目标 L_meta 跨任务优化初始化参数 (Eq.27-29)

默认 encoder_type="lstm" 严格保留原论文复现口径; GRU/MLP/Transformer/
Set Transformer 用于外循环结构消融。
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
    将外循环隐状态映射为 Actor/Critic 网络的参数偏移量。

    采用 FiLM (Feature-wise Linear Modulation) 风格:
      θ_t^(0) = θ_base ⊕ H_θ(h_t)
    其中 ⊕ 表示逐层组合 (scale + shift)。
    """
    def __init__(self, hidden_dim: int, target_param_shapes: List[Tuple[str, int]]):
        """
        参数
        ----
        hidden_dim : int
            外循环编码器输出维度
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
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, n_params),
                nn.Sigmoid(),  # scale ∈ (0, 1), 中心化后 ∈ (0.5, 1.5)
            )
            self.shift_heads[safe_name] = nn.Sequential(
                nn.Linear(hidden_dim, 64),
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
    完整的元学习器 (论文 Section 3.5.3, Algorithm 1)。

    职责:
    1. 维护跨任务反馈历史或递归记忆
    2. 根据任务反馈调制 Actor-Critic 的初始化参数
    3. 计算元损失 L_meta
    """

    def __init__(
        self,
        feedback_dim: int,
        lstm_hidden_dim: int = 128,
        feedback_embed_dim: int = 64,
        actor_critic_config: Optional[Dict] = None,
        encoder_type: str = "lstm",
        transformer_heads: int = 4,
        transformer_layers: int = 1,
        max_history_len: int = 32,
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
        encoder_type : str
            外循环历史反馈编码器: lstm/gru/mlp/transformer/set_transformer
        """
        super().__init__()
        self.lstm_hidden_dim = lstm_hidden_dim
        self.encoder_type = encoder_type.lower()
        self.max_history_len = max_history_len

        # 嵌入网络 g_η
        self.feedback_encoder = FeedbackEncoder(feedback_dim, feedback_embed_dim)

        if self.encoder_type == "lstm":
            # LSTM (论文 Eq.25)
            self.sequence_encoder = nn.LSTM(
                input_size=feedback_embed_dim,
                hidden_size=lstm_hidden_dim,
                num_layers=1,
                batch_first=True,
            )
        elif self.encoder_type == "gru":
            self.sequence_encoder = nn.GRU(
                input_size=feedback_embed_dim,
                hidden_size=lstm_hidden_dim,
                num_layers=1,
                batch_first=True,
            )
        elif self.encoder_type == "mlp":
            self.sequence_encoder = nn.Sequential(
                nn.Linear(feedback_embed_dim, lstm_hidden_dim),
                nn.ReLU(),
                nn.Linear(lstm_hidden_dim, lstm_hidden_dim),
                nn.ReLU(),
            )
        elif self.encoder_type in {"transformer", "set_transformer"}:
            n_heads = max(1, min(transformer_heads, feedback_embed_dim))
            while feedback_embed_dim % n_heads != 0 and n_heads > 1:
                n_heads -= 1
            layer = nn.TransformerEncoderLayer(
                d_model=feedback_embed_dim,
                nhead=n_heads,
                dim_feedforward=max(lstm_hidden_dim * 2, feedback_embed_dim * 2),
                dropout=0.0,
                batch_first=True,
                activation="relu",
            )
            self.sequence_encoder = nn.TransformerEncoder(
                layer,
                num_layers=max(1, transformer_layers),
            )
            self.sequence_pool = nn.Linear(feedback_embed_dim, lstm_hidden_dim)
            if self.encoder_type == "transformer":
                self.pos_embedding = nn.Parameter(
                    torch.zeros(1, max_history_len, feedback_embed_dim)
                )
                nn.init.normal_(self.pos_embedding, std=0.02)
            else:
                self.pos_embedding = None
        else:
            raise ValueError(
                f"未知 meta encoder_type={encoder_type!r}; "
                "可选: lstm/gru/mlp/transformer/set_transformer"
            )

        # 调制头的参数规格将在 attach_to_actor_critic 中动态确定
        self.actor_modulation = None
        self.critic_modulation = None

        # 外循环状态
        self._hidden = None
        self._cell = None
        self._history = None

    def attach_to_actor_critic(self, actor_critic: nn.Module):
        """
        根据实际的 Actor-Critic 网络结构, 初始化调制头。
        只调制偏置项 (bias) 以减少参数量。
        """
        self.actor_modulation = None
        self.critic_modulation = None
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
        """重置外循环状态 (论文 Algorithm 1, line 4)"""
        device = device or next(self.parameters()).device
        self._hidden = None
        self._cell = None
        self._history = None
        if self.encoder_type == "lstm":
            self._hidden = torch.zeros(1, batch_size, self.lstm_hidden_dim, device=device)
            self._cell = torch.zeros(1, batch_size, self.lstm_hidden_dim, device=device)
        elif self.encoder_type == "gru":
            self._hidden = torch.zeros(1, batch_size, self.lstm_hidden_dim, device=device)

    def forward(self, feedback: torch.Tensor) -> torch.Tensor:
        """
        处理一步任务反馈, 更新外循环状态, 返回隐状态 h_t。

        参数
        ----
        feedback : Tensor [batch, feedback_dim]
            任务反馈向量 x_t (论文 Eq.24)

        返回
        ----
        h_t : Tensor [batch, hidden_dim]
        """
        # g_η(x_t)
        z = self.feedback_encoder(feedback)  # [batch, embed_dim]
        if self.encoder_type == "mlp":
            return self.sequence_encoder(z)

        z_seq = z.unsqueeze(1)  # [batch, 1, embed_dim]
        if self.encoder_type == "lstm":
            if self._hidden is None:
                self.reset_hidden(feedback.size(0), feedback.device)
            _, (self._hidden, self._cell) = self.sequence_encoder(
                z_seq, (self._hidden, self._cell)
            )
            return self._hidden.squeeze(0)

        if self.encoder_type == "gru":
            if self._hidden is None:
                self.reset_hidden(feedback.size(0), feedback.device)
            _, self._hidden = self.sequence_encoder(z_seq, self._hidden)
            return self._hidden.squeeze(0)

        if self._history is None:
            self._history = z_seq
        else:
            self._history = torch.cat([self._history, z_seq], dim=1)
            if self._history.size(1) > self.max_history_len:
                self._history = self._history[:, -self.max_history_len:, :]

        hist = self._history
        if self.encoder_type == "transformer":
            hist = hist + self.pos_embedding[:, :hist.size(1), :]
            encoded = self.sequence_encoder(hist)
            pooled = encoded[:, -1, :]  # 带位置编码, 取最近反馈的上下文表示
        else:
            encoded = self.sequence_encoder(hist)
            pooled = encoded.mean(dim=1)  # Set Transformer 消融: 无位置编码, 集合池化
        return self.sequence_pool(pooled)

    def get_modulations(
        self, h_t: torch.Tensor
    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        通过调制头生成参数偏移 (论文 Eq.26)。确定性版本(评估用)。

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

    def get_modulations_stochastic(
        self, h_t: torch.Tensor, explore_std: float = 0.05,
    ) -> Tuple[Optional[Dict], Optional[Dict], torch.Tensor]:
        """
        随机调制 (论文 Eq.26-27 的 REINFORCE 实现)。

        把"LSTM 产生调制"视为一个随机策略动作: 以确定性调制头输出为均值,
        叠加固定标准差的高斯噪声采样得到实际调制量, 并记录其对数概率 log_prob。
        外循环用 L = -(R_normalized · log_prob) 做策略梯度更新, 使适应后累积奖励 R
        的信号通过 log_prob 真正反传到 LSTM + 调制头 (实现 Eq.27 元目标精神)。

        返回
        ----
        actor_mods  : {name: (scale, shift)}  采样后的调制 (用于应用到 base 网络)
        critic_mods : {name: (scale, shift)}
        log_prob    : 标量 tensor, 本次全部调制采样的对数概率之和 (连在计算图上)
        """
        det_actor, det_critic = self.get_modulations(h_t)
        log_prob_terms = []

        def _sample(mods):
            if mods is None:
                return None
            sampled = {}
            for name, (scale_mu, shift_mu) in mods.items():
                # 对 scale 和 shift 各自建高斯分布, 采样并累加 log_prob
                # 用 sample()(非 rsample): REINFORCE 需 log_prob 对均值 mu 保留梯度,
                # 而 rsample 的重参数化会使 (x-mu) 成为常数路径, 导致 log_prob 对 mu 梯度为 0
                scale_dist = torch.distributions.Normal(scale_mu, explore_std)
                shift_dist = torch.distributions.Normal(shift_mu, explore_std)
                scale_s = scale_dist.sample()
                shift_s = shift_dist.sample()
                log_prob_terms.append(scale_dist.log_prob(scale_s).sum())
                log_prob_terms.append(shift_dist.log_prob(shift_s).sum())
                sampled[name] = (scale_s, shift_s)
            return sampled

        actor_mods = _sample(det_actor)
        critic_mods = _sample(det_critic)

        if log_prob_terms:
            log_prob = torch.stack(log_prob_terms).sum()
        else:
            log_prob = torch.zeros(1, device=h_t.device).sum()

        return actor_mods, critic_mods, log_prob

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
                # 调制量应用到 base 参数仅用于给内循环提供初始化, 不需梯度
                # (REINFORCE 的梯度路径靠 log_prob, 不靠这步); detach 避免脱图警告
                scale = scale.detach().view_as(param.data)
                shift = shift.detach().view_as(param.data)
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
