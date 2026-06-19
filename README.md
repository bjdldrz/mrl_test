# MRL-DMS: Meta Reinforcement Learning for Dynamic Mission Scheduling

论文复现代码框架：*Meta reinforcement learning method for dynamic mission scheduling of earth observation satellites* (Aerospace Science and Technology, 2026)

## 项目结构

```
mrl_dms/
├── config.py                   # 所有超参数 (对应论文 Table 2/3)
├── requirements.txt
├── train.py                    # 主训练入口
├── evaluate.py                 # 评估脚本 (复现论文 Table 5/6)
│
├── data/                       # 数据层
│   ├── orbit_utils.py          # 轨道传播 + VTW 计算 (Section 3.3)
│   └── mission_generator.py    # 任务生成: ACLED加载 + 采样 (Section 3.2, 4.1)
│
├── env/                        # 环境层
│   └── satellite_env.py        # Gymnasium MDP 环境 (Section 3.4)
│
├── models/                     # 网络层
│   ├── actor_critic.py         # Actor-Critic + 动作掩码 (Eq.14)
│   └── meta_learner.py         # LSTM 元学习器 (Section 3.5.3, Eq.24-29)
│
├── algo/                       # 算法层
│   ├── ppo.py                  # PPO 内循环 (Section 3.5.2, Eq.22)
│   └── mrl_dms.py              # MRL-DMS 完整训练器 (Algorithm 1)
│
└── utils/
    └── metrics.py              # 评估指标 (Table 4)
```

## 论文公式 → 代码映射

| 论文公式/章节 | 代码位置 | 说明 |
|---|---|---|
| Eq.1 M = M_r ∪ M_d | `mission_generator.py` | 混合任务集 |
| Eq.2-4 VTW, 观测状态 | `orbit_utils.py` | 可见时间窗口 |
| Eq.5-10 约束条件 | `satellite_env.py` | 环境的 step() 逻辑 |
| Eq.11-12 状态空间 | `satellite_env.py: _build_observation()` | s_I × s_Sat |
| Eq.13-14 动作掩码 | `satellite_env.py: _build_action_mask()` + `actor_critic.py` | Valid() + Softmax⊙M |
| Eq.15-20 奖励函数 | `satellite_env.py: compute_reward()` | R_p + R_t + R_d |
| Eq.22 PPO objective | `ppo.py: update()` | Clipped surrogate |
| Eq.24-26 LSTM meta-learner | `meta_learner.py` | 反馈编码 + 参数调制 |
| Eq.27-29 元目标 | `mrl_dms.py: _meta_update()` | L_meta 优化 |
| Algorithm 1 | `mrl_dms.py: train()` | 完整元训练流程 |
| Table 2 卫星参数 | `config.py: DEFAULT_SATELLITES` | 6颗 SSO 卫星 |
| Table 3 算法参数 | `config.py: PPOConfig, NetworkConfig` | 超参数 |
| Table 4 评估指标 | `satellite_env.py: get_metrics()` | 7项指标 |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 快速测试 (验证代码可运行)

```bash
python train.py --method mrl_dms --fast
```

### 3. 使用 ACLED 数据训练

```bash
# 下载数据: https://github.com/YYYauW/DynamicMission
python train.py --method mrl_dms --acled_path path/to/DynamicMission.shp

python train.py --method mrl_dms --acled_path ./DynamicMission/DynamicMission.shp

```

### 4. PPO Baseline 对比

```bash
python train.py --method ppo
```

### 5. 评估

```bash
python evaluate.py --checkpoint checkpoints/mrl_dms_best.pt --experiment all
```

## 扩展指南

### 添加 A2C / DQN Baseline

在 `algo/` 目录下参照 `ppo.py` 的结构实现:
- A2C: 去掉 clipping, 使用 n-step return
- DQN: 改为 Q-network + ε-greedy + replay buffer

### 多星并行

当前框架已预留多星环境列表 (`MRLDMSTrainer.envs`)。
扩展为多智能体时，可在 `_inner_loop_adapt()` 中并行运行多个环境，
共享经验池 (论文 Section 3.5.2 提到的 shared memory pool)。

### 提高 VTW 精度

替换 `orbit_utils.py` 中的简化模型:
- 安装 `sgp4`: 自动切换到 SGP4 精确传播
- 安装 `skyfield`: 启用太阳光照条件判断
- 或对接 Basilisk (`bsk_rl`) 实现高精度姿态动力学

## 数据集

- ACLED 动态任务: https://github.com/YYYauW/DynamicMission
- 卫星 TLE: https://celestrak.org/
- 论文 Zenodo: https://doi.org/10.5281/zenodo.17724850
