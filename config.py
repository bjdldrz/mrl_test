"""
全局配置
========
所有超参数集中管理，对应论文 Table 2 (卫星参数)、Table 3 (算法参数)。
修改实验设置只需改这一个文件。
"""

from dataclasses import dataclass, field
from typing import List, Tuple


# -----------------------------------------------------------------------
# 卫星轨道参数 (论文 Table 2)
# -----------------------------------------------------------------------
@dataclass
class SatelliteConfig:
    """单颗卫星的轨道六根数 + 传感器参数"""
    name: str
    semi_major_axis_km: float    # 半长轴 (km)
    eccentricity: float          # 偏心率
    inclination_deg: float       # 轨道倾角 (°)
    raan_deg: float              # 升交点赤经 (°)
    arg_perigee_deg: float       # 近地点幅角 (°)
    mean_anomaly_deg: float      # 平近点角 (°)
    max_roll_deg: float = 25.0   # 最大滚动角 (±25°, 论文 Constraint 4)
    fov_deg: float = 45.0        # 视场角 (论文 Constraint 4)
    maneuver_speed_deg_s: float = 3.0  # 姿态机动速度 (°/s, 论文 4.1.2)


@dataclass
class GroundStationConfig:
    """地面基站位置与通信可见性约束"""
    name: str
    lat: float
    lon: float
    min_elevation_deg: float = 5.0


# 论文 Table 2: 6颗异构太阳同步轨道卫星
DEFAULT_SATELLITES: List[SatelliteConfig] = [
    SatelliteConfig("Sat1", 644 + 6371,  0.0019, 98.7, 271, 5,   355),
    SatelliteConfig("Sat2", 705 + 6371,  0.0,    98.0, 78,  296, 64),
    SatelliteConfig("Sat3", 705 + 6371,  0.0,    98.2, 269, 108, 252),
    SatelliteConfig("Sat4", 822 + 6371,  0.0,    98.7, 255, 288, 70),
    SatelliteConfig("Sat5", 694 + 6371,  0.0,    98.2, 266, 102, 258),
    SatelliteConfig("Sat6", 496 + 6371,  0.00042,97.2, 318, 185, 175),
]


DEFAULT_GROUND_STATIONS: List[GroundStationConfig] = [
    GroundStationConfig("GS_Svalbard", 78.2, 15.6, 5.0),
    GroundStationConfig("GS_Alaska", 64.8, -147.7, 5.0),
    GroundStationConfig("GS_Madrid", 40.4, -3.7, 5.0),
    GroundStationConfig("GS_Hawaii", 21.3, -157.8, 5.0),
    GroundStationConfig("GS_Singapore", 1.35, 103.8, 5.0),
    GroundStationConfig("GS_Pretoria", -25.7, 28.2, 5.0),
    GroundStationConfig("GS_Canberra", -35.3, 149.1, 5.0),
    GroundStationConfig("GS_Santiago", -33.4, -70.7, 5.0),
]


# -----------------------------------------------------------------------
# 任务参数
# -----------------------------------------------------------------------
@dataclass
class MissionConfig:
    """任务生成与调度参数"""
    # 常规任务
    routine_pool_sizes: List[int] = field(
        default_factory=lambda: [100, 200, 300, 400, 500]
    )
    routine_priority_range: Tuple[float, float] = (0.0, 10.0)
    routine_duration_range_s: Tuple[float, float] = (10.0, 60.0)

    # 动态任务
    dynamic_pool_sizes: List[int] = field(
        default_factory=lambda: [5, 10, 50, 100]
    )
    dynamic_insertions_per_day: int = 3      # 每天插入3次 (论文 4.1.3)
    dynamic_priority_range: Tuple[float, float] = (5.0, 10.0)  # 动态任务优先级更高

    # 调度周期
    schedule_horizon_hours: float = 24.0     # 24小时规划周期 (论文 4.1.2)

    # 基站/下传约束
    n_ground_stations: int = 0               # 0=关闭下传约束; >0 表示共享基站数量
    downlink_time_s: float = 0.0             # 每幅图像固定下传耗时(秒)

    # 动作空间
    max_action_dim: int = 800                # A_max, 常规+动态槽位总数
                                             # = routine_max(500) + insertions(3)×dynamic_per_insertion_max(100)
                                             # 必须 ≥800,否则动态任务会溢出槽位被静默丢弃，污染 dynamic_completion_rate


# -----------------------------------------------------------------------
# PPO 超参数 (论文 Table 3)
# -----------------------------------------------------------------------
@dataclass
class PPOConfig:
    learning_rate: float = 0.005            # 论文 Table 3: α=0.005
    discount_factor: float = 0.99            # γ
    clip_ratio: float = 0.2                  # ε
    gae_lambda: float = 0.95                 # λ
    entropy_coeff: float = 0.01             # 论文 Table 3: 0.01
    value_loss_coeff: float = 0.5
    batch_size: int = 128
    ppo_epochs: int = 4                      # 每次更新的梯度步数 K


# -----------------------------------------------------------------------
# 网络结构 (论文 Table 3)
# -----------------------------------------------------------------------
@dataclass
class NetworkConfig:
    hidden_layers: List[int] = field(
        default_factory=lambda: [256, 256, 256]  # 3×256
    )
    lstm_hidden_dim: int = 128               # LSTM 隐藏层维度
    meta_encoder_type: str = "lstm"          # 外循环编码器: lstm/gru/mlp/transformer/set_transformer
    meta_transformer_heads: int = 4          # Transformer/Set Transformer 注意力头数
    meta_transformer_layers: int = 1         # Transformer/Set Transformer 层数
    meta_history_len: int = 32               # Transformer 类编码器保留的跨任务反馈历史长度
    activation: str = "relu"


# -----------------------------------------------------------------------
# 元学习参数
# -----------------------------------------------------------------------
@dataclass
class MetaConfig:
    meta_lr: float = 0.005                    # 论文 Table 3: α=0.005（内外循环统一学习率）
    meta_batch_size: int = 16                # 每次元更新采样的任务数（=CPU核心数，充分并行）
    inner_steps: int = 4                     # 内循环 PPO 更新步数 K（调小加速，4 步足够内循环适应）
    rollout_steps: int = 512                 # T_rollout（调小加速，512 步轨迹）
    eval_steps: int = 1024                   # T_eval (评估轨迹长度)
    feedback_dim: int = 64                   # 反馈向量压缩后的维度 (g_η 输出)


# -----------------------------------------------------------------------
# 多智能体参数 (MAPPO)
# -----------------------------------------------------------------------
@dataclass
class MAPPOConfig:
    n_satellites: int = 1                    # 参与调度的卫星数量
    parameter_sharing: bool = True           # Actor 是否共享参数
    critic_hidden_dims: List[int] = field(
        default_factory=lambda: [256, 256]   # 集中式 Critic 隐藏层
    )
    use_global_state: bool = True            # Critic 是否使用全局状态


# -----------------------------------------------------------------------
# 奖励函数权重 (论文 Section 3.4)
# -----------------------------------------------------------------------
@dataclass
class RewardConfig:
    # --- 论文 Eq.15 奖励: R = Rp + Rt + Rd (严格复现) ---
    w_priority: float = 1.0                  # w_p (优先级权重, Eq.16)
    w_dynamic: float = 1.0                   # w_d (动态任务权重, Eq.18; 论文 Table 3 未给具体值, 取 1.0 中性)
    dynamic_decay_k: float = 2.0             # Rd 指数衰减系数 f=exp(-k·time_ratio)
                                             # 论文文字矛盾: 第639行"越早完成奖励越高"(与Eq.17/此实现一致),
                                             # 第642行"越接近截止越高"(相反). 此处取与 Eq.17 自洽的"越早越高"
    # --- 以下为论文外扩展(reward shaping), 严格复现时置 0 关闭; 作为优化方案可开启 ---
    w_quality: float = 0.0                   # w_q 观测质量(论文无此项, Eq.15 仅三项); >0 时启用
    penalty_idle: float = 0.0                # 空闲惩罚(论文无, 论文靠掩码而非惩罚)
    penalty_invalid: float = 0.0             # 无效动作惩罚(论文无)
    penalty_deadline_miss: float = 0.0       # 超截止惩罚(论文无)


# -----------------------------------------------------------------------
# 训练参数
# -----------------------------------------------------------------------
@dataclass
class TrainConfig:
    total_training_steps: int = 3_276_800    # 16 batch × 4 steps × 512 rollout × 100 iters
    seed: int = 42
    device: str = "auto"                     # "auto" / "cpu" / "cuda" / "mps"
    log_interval: int = 1          # 单位: 元迭代次数
    eval_interval: int = 10        # 每 10 次元迭代评估一次
    save_interval: int = 20        # 每 20 次元迭代保存一次
    log_dir: str = "runs/"
    checkpoint_dir: str = "checkpoints/"
    eval_n_routine: int = 200
    eval_n_dynamic_per_insertion: int = 50
    eval_workers: int = 1                    # 评估 episode 并行 worker 数; 1 为串行
    vtw_time_step_s: float = 60.0            # VTW 采样步长(秒); LEO 过境快, >60s 会漏采过境最接近点
                                             # 实测: 300s 严重漏窗→0, 60s 稳定且单次仅 ~4ms(有跨任务缓存)
    num_workers: int = 16                    # 并行 worker 数; 0 = 自动(等于 meta_batch_size)
    profile_timing: bool = True              # 记录 MRL-DMS 各阶段耗时,用于定位 CPU/IO/GPU 瓶颈


# -----------------------------------------------------------------------
# 汇总配置
# -----------------------------------------------------------------------
@dataclass
class Config:
    satellites: List[SatelliteConfig] = field(
        default_factory=lambda: DEFAULT_SATELLITES
    )
    ground_stations: List[GroundStationConfig] = field(
        default_factory=lambda: DEFAULT_GROUND_STATIONS
    )
    mission: MissionConfig = field(default_factory=MissionConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    meta: MetaConfig = field(default_factory=MetaConfig)
    mappo: MAPPOConfig = field(default_factory=MAPPOConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def get_default_config() -> Config:
    return Config()
