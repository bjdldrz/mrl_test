# MRL-DMS:面向对地观测卫星动态任务调度的元强化学习

> 论文复现 + 多星协同优化扩展
> 参考论文:*Meta reinforcement learning method for dynamic mission scheduling of earth observation satellites*,Wei Yao 等,武汉大学,*Aerospace Science and Technology* 176 (2026) 112094。

本项目分两层:

1. **单星 MRL-DMS(严格复现原论文)**:FOMAML 外循环 + PPO 内循环 + LSTM 元学习器,对应论文 Section 3–5。
2. **多星 MAPPO 协同(本项目的优化方案)**:在论文单星框架之上扩展到 N 颗卫星协同调度(CTDE,集中式 Critic)。论文 Section 5.2 仅做单星实验,多星协同是论文列为未来工作的方向,这里作为我们的优化贡献,并提供"无协同 baseline"做对照。

---

## 一、目录结构

数据集与实验场景说明见 [DATASET_DESCRIPTION.md](DATASET_DESCRIPTION.md)。

```
mrl_dms/
├── config.py                   # 所有超参数 (对应论文 Table 2/3) + 6 颗 SSO 卫星定义
├── train.py                    # 主训练入口 (mrl_dms / ppo / a2c / dqn)
├── evaluate.py                 # 评估脚本 (复现论文 Table 5/6 的泛化实验)
├── compare_methods.py          # ★ 方案对比实验: Single-PPO vs Indep-PPO vs MAPPO
├── visualize.py                # 训练曲线 + 评估指标 + 方案对比图绘制
│
├── data/                       # 数据层
│   ├── orbit_utils.py          # SGP4 轨道传播 + TEME→ECEF + VTW 计算 (Section 3.3)
│   └── mission_generator.py    # 任务生成: ACLED 加载 + 时空采样 (Section 3.2, 4.1)
│
├── envs/                       # 环境层
│   ├── satellite_env.py        # 单星 Gymnasium MDP 环境 (Section 3.4)
│   └── multi_satellite_env.py  # ★ 多星 CTDE 环境 (PettingZoo 风格, coordinate 开关)
│
├── models/                     # 网络层
│   ├── actor_critic.py         # 单星 Actor-Critic + 动作掩码 (Eq.14)
│   ├── meta_learner.py         # 可切换外循环元学习器: LSTM/GRU/MLP/Transformer/Set Transformer
│   └── mappo.py                # 多星 MAPPO: 分布式 Actor + 集中式 Critic
│
├── algo/                       # 算法层
│   ├── ppo.py                  # PPO 内循环 (Section 3.5.2, Eq.22)
│   ├── mappo_trainer.py        # MAPPO 训练器 + 多智能体 Rollout Buffer
│   ├── mrl_dms.py              # MRL-DMS 完整元训练器 (Algorithm 1, 含进程池并行)
│   └── task_worker.py          # 多进程任务 worker (spawn, 线程数隔离)
│
└── utils/
    └── metrics.py              # 评估指标工具 (Table 4)
```

---

## 二、核心方法

### 2.1 单星 MRL-DMS(论文复现)

- **MDP 建模**(Section 3.4):状态 = 任务信息编码 `s_I` × 卫星状态 `s_Sat`;动作 = 选择某个任务观测或 idle;奖励 = 优先级 `R_p` + 时效 `R_t` + 动态响应 `R_d`。
- **动作掩码**(Eq.13-14):用 `Valid()` 过滤不可行任务(无 VTW、机动时间不足、任务冲突等约束 5-10),Softmax 后逐元素相乘。
- **元学习**(Section 3.5):FOMAML 外循环采样一批任务分布,每个任务用 PPO 做内循环适应;默认 LSTM 元学习器读取内循环反馈,输出对 Actor/Critic 参数的**调制量**(scale/shift)。为研究外循环结构,也可切换为 GRU/MLP/Transformer/Set Transformer。元目标用 **REINFORCE** 优化(必须 `.sample()` 而非 `.rsample()`,否则 log_prob 对均值梯度为 0)。

### 2.2 多星 MAPPO 协同(优化扩展)

- **CTDE 架构**:每颗卫星一个分布式 Actor(只看局部观测),共享一个集中式 Critic(看全局状态 = 各星局部观测的 mean pooling,维度与卫星数无关,避免参数爆炸)。
- **`coordinate` 开关**(`envs/multi_satellite_env.py`)是对照实验的关键:
  - `coordinate=True`(**MAPPO,本方法**):① 全局 episode 级任务指派(见下);② 逐时刻冲突解决(负载感知贪心拍卖 + 败者改派);③ 观测状态同步——任一颗星完成任务后全体标记完成,避免重复观测。
  - `coordinate=False`(**Indep-PPO,无协同 baseline**):各星完全独立决策,不去冲突、不同步,会产生重复观测。
- 这样设计使"多星 vs 单星"的对比不会因"卫星数量多 = 可观测窗口多"而失真,**重复观测率/负载均衡/动态响应延迟**才是体现协同价值的真正指标。

#### 协同优化机制(`OPTIMIZATION_ROADMAP.md` 记录完整路线图)

- **全局 episode 级任务指派**(主力):`reset()` 时综合每颗星对每个任务在整个 24h 内的窗口质量(最小 off-nadir),用「最少候选优先 + 负载惩罚」贪心广义指派,为每个任务预分配归属卫星;**所有权掩码**让各星只在自己负责的任务上行动,从构造上消除重复、按设计均衡负载。动态任务到达时增量指派。
  - 效果:MAPPO 负载变异系数从 0.46 降到 0.15(优于无协同的 0.21)、重复率 0%、画质(off-nadir)最优。
  - **权衡旋钮 `assign_w_load`**:存在"负载均衡 vs 吞吐"的本质权衡(覆盖好的卫星被限额后其多余任务可能无人完成);该权重越大越均衡、吞吐越低,把权衡变成可调的帕累托曲线。
- **逐时刻冲突解决**(辅助):同一时刻多星争抢同一任务时按边际价值竞价(优先级 + 质量 − 负载),败者评估期改派次优任务、训练期保持 idle 以保信用分配。在当前 SSO 稀疏可行性下杠杆较低,主要作安全网。

---

## 三、论文公式 → 代码映射

| 论文公式/章节 | 代码位置 | 说明 |
|---|---|---|
| Eq.1 `M = M_r ∪ M_d` | `data/mission_generator.py` | 常规 + 动态混合任务集 |
| Eq.2-4 VTW、观测状态 | `data/orbit_utils.py` | SGP4 传播 + TEME→ECEF + ±25° roll + 45° FOV + 光照 |
| Eq.5-10 约束条件 | `envs/satellite_env.py: _build_action_mask()` | 含机动时间(约束 8)、任务冲突(约束 10) |
| Eq.11-12 状态空间 | `satellite_env.py: _build_observation()` | `s_I × s_Sat` |
| Eq.13-14 动作掩码 | `satellite_env.py` + `models/actor_critic.py` | `Valid()` + `Softmax ⊙ M` |
| Eq.15-20 奖励函数 | `satellite_env.py: compute_reward()` | `R_p + R_t + R_d`(动态指数衰减 `dynamic_decay_k`) |
| Eq.22 PPO 目标 | `algo/ppo.py: update()` | Clipped surrogate |
| Eq.24-26 LSTM 元学习器 | `models/meta_learner.py` | 反馈编码 + 参数调制 `get_modulations_stochastic()` |
| Eq.27-29 元目标 | `algo/mrl_dms.py: _meta_update()` | REINFORCE 优化 `L_meta` |
| Algorithm 1 | `algo/mrl_dms.py: train()` | 完整元训练流程(进程池并行) |
| Table 2 卫星参数 | `config.py: DEFAULT_SATELLITES` | 6 颗 SSO 卫星 |
| Table 3 算法参数 | `config.py: PPOConfig / MetaConfig` | lr=0.005, entropy=0.01, K=4 |
| Table 4 评估指标 | `satellite_env.py / multi_satellite_env.py: get_metrics()` | feasible 口径完成率 + 协同指标 |

---

## 四、环境准备

```bash
pip install -r requirements.txt   # numpy, torch, sgp4, geopandas, matplotlib 等
```

**运行环境说明(重要):**

- 本项目实测使用的解释器:`/Users/zhouzidie/miniconda3/envs/myenv/bin/python`(torch 2.5.1)。base conda 环境没有 torch。
- **本地 Mac 必须用 `--device cpu`**:Mac 的 MPS 后端有 LSTM backward 的 bug(`shape4.size() >= 3` 断言失败),会在 REINFORCE 真正反传 LSTM 时崩溃。CUDA 服务器不受此影响。
- **CPU 并行**:训练用 `multiprocessing` spawn 进程池,每个 worker 内 `OMP_NUM_THREADS=1` 防止线程超额订阅(此环境变量在所有 import 之前设置)。本设计下 GPU 利用率本身就低(MAML 风格:多个小策略在 CPU 上 rollout,GPU 只做极小的元更新),属正常现象。

---

## 五、运行流程

### 1. 快速冒烟测试(验证代码可跑通)

```bash
python train.py --method mrl_dms --fast
```

`--fast` 自动:单星模式、缩减训练步数、`vtw_time_step_s=60`。

### 2. 单星 MRL-DMS 训练(论文复现,推荐用真实 ACLED 数据)

```bash
python train.py --method mrl_dms --acled_path ./DynamicMission/DynamicMission.shp
# 本地 Mac 训练务必加 --device cpu (规避 MPS 的 LSTM backward 崩溃):
python train.py --method mrl_dms --acled_path ./DynamicMission/DynamicMission.shp --device cpu
# 每次训练默认会写入唯一目录 runs/<method>_<timestamp>/; 可加标签便于区分:
python train.py --method mrl_dms --acled_path ./DynamicMission/DynamicMission.shp --device cpu --run_tag paper_repro
```

> 用真实 ACLED 数据训练时,冲突热点聚集会**大幅提高任务可达率**,更贴近论文设定。
> `--device` 可选 `auto`(默认,CUDA>MPS>CPU)/ `cpu` / `cuda` / `mps`。

### 3. PPO Baseline(对照单星元学习)

```bash
python train.py --method ppo --acled_path ./DynamicMission/DynamicMission.shp
```

### 4. ★ 方案对比实验(单星 vs 多星无协同 vs 多星协同)

这是体现多星 MAPPO 协同优势的核心实验,三种方案在**完全相同的固定测试集**上评估:

```bash
python compare_methods.py \
    --acled_path ./DynamicMission/DynamicMission.shp \
    --n_satellites 6 --train_iters 30 --eval_episodes 5 \
    --n_routine 200 --n_dynamic 50 \
    --out_dir runs/compare
# 协同机制开关 (仅影响 MAPPO):
#   --no_episode_assignment   关闭全局指派, 退回逐时刻协同
#   --assign_w_load 0.1       负载均衡权重 (越大越均衡, 吞吐换均衡)
#   --assignment_capacity_mode proportional  按覆盖容量比例指派; equal 为等额指派
#   --release_before_deadline_s 1800         截止前释放所有权, 回收硬指派吞吐损失
```

默认情况下,`compare_methods.py` 会在 `--out_dir` 下自动创建唯一子目录,例如
`runs/compare/single_compare_sat6_iter30_assign_on_seed42_20260622_143000/`,
避免重复运行覆盖旧结果。若确实想直接写入指定目录,加 `--flat_out_dir`。

| 方案 | 含义 |
|---|---|
| `Single-PPO`  | 单星 PPO |
| `Indep-PPO`   | 多星独立 PPO(`coordinate=False`,无协同 baseline) |
| `MAPPO`       | 多星 MAPPO(`coordinate=True`,集中式 Critic,**本方法**) |

输出 `runs/compare/comparison_results.json`,并打印对比表。**重点看重复观测率**:无协同方案会出现大量重复观测(多星扑向同一目标),MAPPO 协同后重复观测率应趋近 0,同时动态响应更快、观测质量更高。

> 注意:`Indep-PPO` 的累积奖励可能虚高,因为重复观测在重复刷分——这恰好说明只看奖励/完成率会被误导,**必须看重复观测率**才能看出协同的真正价值。

### 4.1 批量消融实验(推荐用于优化对比)

`run_ablation.py` 会批量调用 `compare_methods.py`,每个子实验保存 `comparison_results.json` 和 `manifest.json`,并汇总成 `ablation_summary.csv/json`。

```bash
python run_ablation.py \
    --python /Users/zhouzidie/miniconda3/envs/myenv/bin/python \
    --preset assignment_v2 \
    --n_satellites 6 --train_iters 30 --eval_episodes 5 \
    --n_routine 200 --n_dynamic 50 \
    --out_root runs/ablation_assignment_v2 \
    --device cpu

# 快速检查命令组合,不真正运行:
python run_ablation.py --dry_run --train_iters 0 --eval_episodes 1
```

默认情况下,`run_ablation.py` 会在 `--out_root` 下创建唯一批次目录,例如
`runs/ablation_assignment_v2/assignment_v2_sat6_iter30_eval5_seed42_20260622_143000/`;
各子实验再写入该批次目录下的 `<tag>/`。若要复用指定目录,加 `--flat_out_root`。

默认 `assignment_v2` preset 会比较:
- `--no_episode_assignment` baseline
- `assignment_capacity_mode=equal/proportional`
- `assign_w_load=0.05/0.1/0.2`
- `release_before_deadline_s=0/1800`

每次单独运行 `compare_methods.py` 也会额外写入 `manifest.json`,记录参数、git commit、dirty 状态、运行环境和输出文件路径。

奖励塑形消融:

```bash
python run_ablation.py \
    --python /Users/zhouzidie/miniconda3/envs/myenv/bin/python \
    --preset reward_v1 \
    --n_satellites 6 --train_iters 30 --eval_episodes 5 \
    --n_routine 200 --n_dynamic 50 \
    --out_root runs/ablation_reward_v1 \
    --device cpu
```

`reward_v1` 只影响 MAPPO 训练奖励,默认单次对比保持关闭。可单独打开:
- `--team_reward_mix 0.25`: 个体奖励与团队平均奖励混合。
- `--load_balance_reward_coeff 0.1`: 相对空闲卫星完成任务时给额外奖励。
- `--team_completion_bonus 0.05`: 全队每新增完成一个任务时给全体小 bonus。
- `--normalize_agent_rewards`: MAPPO 更新前对每颗卫星 rollout 奖励归一化。

critic 全局状态消融:

```bash
python run_ablation.py \
    --python /Users/zhouzidie/miniconda3/envs/myenv/bin/python \
    --preset state_v1 \
    --n_satellites 6 --train_iters 30 --eval_episodes 5 \
    --n_routine 200 --n_dynamic 50 \
    --out_root runs/ablation_state_v1 \
    --device cpu
```

`state_v1` 比较 `mean` pooling、`mean + task_stats`、`concat`、`concat + task_stats`。也可在单次运行中使用:
- `--global_state_mode mean|concat`
- `--global_state_task_stats`

Greedy-Oracle 参考上界:

```bash
python run_ablation.py \
    --python /Users/zhouzidie/miniconda3/envs/myenv/bin/python \
    --preset oracle_v1 \
    --n_satellites 6 --train_iters 30 --eval_episodes 5 \
    --n_routine 200 --n_dynamic 50 \
    --out_root runs/ablation_oracle_v1 \
    --device cpu

# 或给任意 preset 加 oracle 参考:
python run_ablation.py --preset assignment_v2 --run_oracle
```

`--run_oracle` 会额外运行 `Greedy-Oracle`:每一步由集中式启发式在所有卫星可行动作中选择非冲突任务,用于估计当前场景下的可达参考上界。它不是严格 ILP 最优,但能给出 `oracle_relative_completion` 和 `mappo_oracle_gap_n_scheduled`,用于判断 MAPPO 离强启发式还有多远。

训练稳定性消融:

```bash
python run_ablation.py \
    --python /Users/zhouzidie/miniconda3/envs/myenv/bin/python \
    --preset train_stability_v1 \
    --n_satellites 6 --train_iters 30 --eval_episodes 5 \
    --n_routine 200 --n_dynamic 50 \
    --out_root runs/ablation_train_stability_v1 \
    --device cpu
```

`train_stability_v1` 比较默认训练、卫星数量课程、联合探索、课程+联合探索。单次运行可用:
- `--satellite_curriculum --curriculum_min_satellites 1 --curriculum_iters 10`
- `--joint_explore_prob 0.05`

执行期通信消融:

```bash
python run_ablation.py \
    --python /Users/zhouzidie/miniconda3/envs/myenv/bin/python \
    --preset communication_v1 \
    --n_satellites 6 --train_iters 30 --eval_episodes 5 \
    --n_routine 200 --n_dynamic 50 \
    --out_root runs/ablation_communication_v1 \
    --device cpu
```

`communication_v1` 比较默认训练、意图广播、意图广播+训练稳定性。单次运行可用:
- `--intent_broadcast --intent_replan_rounds 1`

外循环编码器消融:

```bash
python run_ablation.py \
    --python /Users/zhouzidie/miniconda3/envs/myenv/bin/python \
    --preset meta_encoder_v1 \
    --out_root runs/ablation_meta_encoder_v1 \
    --batch_name meta_encoder_v1 \
    --meta_iterations 2 \
    --meta_mappo_n_satellites 2 \
    --device cpu
```

`meta_encoder_v1` 比较单星 MRL-DMS 外循环的 `lstm/gru/mlp/transformer/set_transformer`,并额外运行 `MAPPO + LSTM 外循环`。正式实验可调大 `--meta_iterations`,以及用 `--meta_mappo_n_satellites 6` 做完整多星版本。单次训练可用:
- `python train.py --method mrl_dms --meta_encoder_type gru --device cpu`
- `python train.py --method mrl_dms --meta_encoder_type lstm --mappo_n_satellites 6 --device cpu`

### 4.2 优化步骤命令清单

建议按下面顺序跑,每一步都会自动生成唯一结果目录,便于横向对比。

```bash
# Step 1: 全局任务指派/所有权/截止释放消融
python run_ablation.py --python /Users/zhouzidie/miniconda3/envs/myenv/bin/python \
  --preset assignment_v2 --out_root runs/ablation_assignment_v2 --batch_name step1_assignment --device cpu

# Step 2: 协同奖励塑形消融
python run_ablation.py --python /Users/zhouzidie/miniconda3/envs/myenv/bin/python \
  --preset reward_v1 --out_root runs/ablation_reward_v1 --batch_name step2_reward --device cpu

# Step 3: 集中式 critic 全局状态消融
python run_ablation.py --python /Users/zhouzidie/miniconda3/envs/myenv/bin/python \
  --preset state_v1 --out_root runs/ablation_state_v1 --batch_name step3_state --device cpu

# Step 4: Greedy-Oracle 参考上界
python run_ablation.py --python /Users/zhouzidie/miniconda3/envs/myenv/bin/python \
  --preset oracle_v1 --out_root runs/ablation_oracle_v1 --batch_name step4_oracle --device cpu

# Step 5: 训练稳定性消融
python run_ablation.py --python /Users/zhouzidie/miniconda3/envs/myenv/bin/python \
  --preset train_stability_v1 --out_root runs/ablation_train_stability_v1 --batch_name step5_train_stability --device cpu

# Step 6: 规则式意图广播通信消融
python run_ablation.py --python /Users/zhouzidie/miniconda3/envs/myenv/bin/python \
  --preset communication_v1 --out_root runs/ablation_communication_v1 --batch_name step6_communication --device cpu

# Step 7: MRL-DMS 外循环编码器结构消融 + MAPPO-LSTM 外循环
python run_ablation.py --python /Users/zhouzidie/miniconda3/envs/myenv/bin/python \
  --preset meta_encoder_v1 --out_root runs/ablation_meta_encoder_v1 --batch_name step7_meta_encoder \
  --meta_iterations 2 --meta_mappo_n_satellites 2 --device cpu
```

每个结果目录下的 `comparison_results.json` 含完成率、可观测任务数、重复率、负载均衡等指标;`manifest.json` 记录参数和 git commit;`*_viz_data.json` 可用于画任务分布图和调度甘特图。
`meta_encoder_v1` 属于训练型消融,每个子目录输出 `summary.json/train_log.csv/eval_log.csv`,批次根目录输出 `ablation_summary.json/csv`。

可视化某次 compare/ablation 子实验:

```bash
python visualize.py --compare_json <结果目录>/comparison_results.json
```

该命令会额外生成:
- `method_comparison_task_counts.png`: 全部任务数 / 可观测任务数 / 已调度任务数
- `task_distribution_<method>.png`: 任务地理分布图
- `gantt_multi_<method>.png`: 任务调度甘特图

### 5. 评估(论文泛化实验 Table 5/6)

```bash
python evaluate.py --checkpoint checkpoints/mrl_dms_best.pt --experiment all
# --experiment 可选: all / cross_rl / scale / spatial
# evaluate.py 默认在 CPU 上运行, 无需额外指定设备
```

### 6. 可视化

```bash
# 单次训练曲线 + 评估指标
python visualize.py --run_dir runs/<exp_name>

# 多次运行对比 (例如 MRL-DMS vs PPO)
python visualize.py --run_dirs runs/exp_a runs/exp_b --labels MRL-DMS PPO

# 方案对比图 (配合 compare_methods.py 的输出)
python visualize.py --compare_json runs/compare/comparison_results.json
```

`visualize.py` 会自动检测系统中文字体(Mac 的 PingFang/Heiti,Linux 的 Noto/WenQuanYi/SimHei),避免 PNG 中文显示成方块。

---

## 六、评估指标(Table 4 口径)

完成率类指标的分母为 **feasible 任务**(有可用 VTW 的任务),而非全部任务——这与论文 Table 4 一致。SSO 近极轨道对中低纬度的 ACLED 冲突热点覆盖有限,若用"全部任务"做分母会得到极低且失真的完成率。

`get_metrics()` 返回:

- **完成率类**(feasible 口径):`observation_success_rate`、`dynamic_completion_rate`、`routine_completion_rate`;另附 `*_raw`(全部任务口径)、`feasible_ratio` 做诊断对照。
- **奖励**:`total_reward`。
- **协同质量类(多星独有,体现 MAPPO 优势)**:
  - `n_duplicates` / `duplicate_rate`——重复观测数/率(协同好 → 0)
  - `load_balance_cv`——负载变异系数(越小越均衡)
  - `avg_off_nadir_deg`——平均观测质量(越小越好)
  - `avg_dynamic_response_s`——动态任务响应延迟(越小越快)
  - `coordination_gain`——协同增益 = 多星完成数 /(N × 单星完成数)

---

## 七、输出文件

| 路径 | 内容 |
|---|---|
| `runs/<exp_name>/train_log.csv` | 训练曲线数据 |
| `checkpoints/<method>_best.pt`  | 最优模型 |
| `runs/compare/comparison_results.json` | 方案对比原始指标 |
| `runs/compare/comparison_summary.txt`  | 对比汇总表 |
| `runs/compare/method_comparison_performance.png` | 完成率 + 奖励对比图 |
| `runs/compare/method_comparison_coordination.png` | 5 项协同质量对比图 |

---

## 八、关键超参数(`config.py`)

| 参数 | 值 | 出处 |
|---|---|---|
| `max_action_dim` | 800 | 常规 500 + 3×100 动态 |
| `learning_rate` / `meta_lr` | 0.005 | Table 3 |
| `entropy_coeff` | 0.01 | Table 3 |
| `inner_steps` (K) | 4 | 内循环 PPO 更新步数 |
| `rollout_steps` | 512 | 单条轨迹长度 |
| `vtw_time_step_s` | 60.0 | LEO 过境快,>60s 会漏采最接近点 |
| `n_satellites` | 1(默认单星) | 多星实验用 `--n_satellites` 覆盖 |

---

## 九、数据集

- ACLED 动态任务:https://github.com/YYYauW/DynamicMission
- 卫星 TLE:https://celestrak.org/
- 论文 Zenodo:https://doi.org/10.5281/zenodo.17724850
