# CVA-MAPPO 主方案实验计划

本文档只围绕论文主方案 **CVA-MAPPO** 展开,用于执行训练、主对比和消融实验。

主方案定义:

```text
CVA-MAPPO = 上下文价值感知任务分配 + 滚动 owner 重分配 + 低层 MAPPO 调度
```

其中外循环思想不再作为单独的 PPO 结构消融,而是作为任务分配阶段的上下文价值编码器:

```text
score(satellite, task)
  = heuristic_value
  + context_value_encoder(task set / task sequence / satellite-task graph)
  + rolling ownership history terms
```

---

## 1. 推荐实验路线

| 阶段 | 实验 | 目的 | 输出目录 |
|---|---|---|---|
| A0 | dry-run 检查 | 确认命令、路径、子实验数量正确 | 不产生结果 |
| A1 | CVA-MAPPO 单次训练评估 | 确认主方案能完整训练和评估 | `runs/main_cva_mappo_train_eval` |
| A2 | 基础三方案压力对比 | 得到 Single-PPO、Indep-PPO、Vanilla MAPPO 基线 | `runs/compare_vanilla_mappo_stress` |
| A3 | CVA-MAPPO 完整三方案对比 | 得到最终方法与 Single/Indep 的同表对比 | `runs/compare_cva_mappo_stress` |
| A4 | CVA 主消融 | 验证 rolling、CVA scorer、上下文 encoder 的贡献 | `runs/ablation_cva_assignment_v1_stress` |
| A5 | CVA 参数消融 | 验证 mix/context weight 是否稳健 | `runs/ablation_cva_mix_v1_stress` |

建议先跑 A0、A1,确认无报错后再跑 A2-A5。

---

## 2. 统一压力场景

后续主对比和消融默认使用同一压力口径:

```text
n_satellites = 12
n_routine = 1200
n_dynamic = 300
eval_episodes = 8
train_iters = 30
vtw_time_step_s = 60
rollout_steps = 256
ppo_epochs = 2
ppo_batch_size = 256
seed = 42
```

原因:

- 小规模任务下各方法容易接近可观测任务上限,优化差异不明显。
- 多星压力场景更能暴露 Indep-PPO 的重复观测问题。
- CVA-MAPPO 的优势应该体现在重复观测控制、owner 失效修复、动态任务响应和负载均衡上。

数据路径统一写成显式路径:

```text
./DynamicMission/DynamicMission.shp
```

---

## 3. A0: Dry-run 检查

先确认 `cva_assignment_v1` 会生成预期的 8 个子实验。

```bash
python run_ablation.py \
  --python python \
  --preset cva_assignment_v1 \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --n_satellites 12 \
  --train_iters 30 \
  --eval_episodes 8 \
  --n_routine 1200 \
  --n_dynamic 300 \
  --methods mappo \
  --out_root runs/ablation_cva_assignment_v1_stress \
  --device cpu \
  --jobs 4 \
  --eval_workers 4 \
  --rollout_steps 256 \
  --ppo_epochs 2 \
  --ppo_batch_size 256 \
  --vtw_time_step_s 60 \
  --no_viz \
  --dry_run
```

应看到这些子实验:

- `heuristic_static`
- `heuristic_rolling`
- `cva_lstm_static_mix0p35`
- `cva_mlp_rolling_mix0p35`
- `cva_lstm_rolling_mix0p35`
- `cva_gru_rolling_mix0p35`
- `cva_transformer_rolling_mix0p35`
- `cva_set_transformer_rolling_mix0p35`

---

## 4. A1: CVA-MAPPO 单次训练评估

目的:先单独训练和评估最终候选方法,确认主方案链路稳定。

推荐最终候选:

```text
assignment_scorer = cva
assignment_context_encoder = lstm
assignment_scorer_mix = 0.35
assignment_context_weight = 0.25
rolling trigger = periodic,dynamic,stale_owner,deadline
rolling horizon = 7200s
```

命令:

```bash
python compare_methods.py \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --n_satellites 12 \
  --train_iters 30 \
  --eval_episodes 8 \
  --n_routine 1200 \
  --n_dynamic 300 \
  --methods mappo \
  --assignment_capacity_mode proportional \
  --assign_w_load 0.1 \
  --release_before_deadline_s 1800 \
  --assignment_scorer cva \
  --assignment_scorer_mix 0.35 \
  --assignment_context_encoder lstm \
  --assignment_context_weight 0.25 \
  --assignment_sequence_hidden_dim 16 \
  --assignment_replan_interval_s 3600 \
  --assignment_replan_horizon_s 7200 \
  --assignment_replan_trigger periodic,dynamic,stale_owner,deadline \
  --assignment_switch_penalty 0.05 \
  --assignment_lock_window_s 600 \
  --assignment_max_switches_per_task 2 \
  --rollout_steps 256 \
  --ppo_epochs 2 \
  --ppo_batch_size 256 \
  --eval_workers 4 \
  --torch_num_threads 4 \
  --vtw_time_step_s 60 \
  --out_dir runs/main_cva_mappo_train_eval \
  --run_name cva_mappo_lstm_rolling_stress \
  --no_viz \
  --device cpu
```

说明:

- A1 只运行 `--methods mappo`,不能使用 `--method_jobs` 并行多个顶层方法。
- `--eval_workers 4` 只并行训练后的评估 episode。
- `--torch_num_threads 4` 让单个 MAPPO 训练进程内部的 PyTorch 前向/更新使用更多 CPU 线程。
- `--no_viz` 会跳过 `*_viz_data.json`,避免并行评估后为了可视化额外串行重跑 1 个 episode;需要画任务分布图和甘特图时去掉这一项。

结果文件:

```text
runs/main_cva_mappo_train_eval/<run_name>/comparison_results.json
runs/main_cva_mappo_train_eval/<run_name>/manifest.json
runs/main_cva_mappo_train_eval/<run_name>/*_viz_data.json  # 未使用 --no_viz 时生成
```

---

## 5. A2: 基础三方案压力对比

目的:建立论文主表的基础参照,尤其看 Indep-PPO 在多星压力场景下的重复观测。

这里关闭 episode assignment,得到更接近 Vanilla MAPPO 的低层多星协同基线。

```bash
python compare_methods.py \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --n_satellites 12 \
  --train_iters 30 \
  --eval_episodes 8 \
  --n_routine 1200 \
  --n_dynamic 300 \
  --methods single,indep,mappo \
  --method_jobs 3 \
  --no_episode_assignment \
  --rollout_steps 256 \
  --ppo_epochs 2 \
  --ppo_batch_size 256 \
  --eval_workers 4 \
  --vtw_time_step_s 60 \
  --out_dir runs/compare_vanilla_mappo_stress \
  --run_name vanilla_mappo_stress \
  --device cpu
```

该结果用于提供:

- `Single-PPO`
- `Indep-PPO`
- `Vanilla MAPPO`

---

## 6. A3: CVA-MAPPO 完整三方案对比

目的:最终论文主表可直接使用这一组结果,展示最终方法相对 Single-PPO 和 Indep-PPO 的表现。

```bash
python compare_methods.py \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --n_satellites 12 \
  --train_iters 30 \
  --eval_episodes 8 \
  --n_routine 1200 \
  --n_dynamic 300 \
  --methods single,indep,mappo \
  --method_jobs 3 \
  --assignment_capacity_mode proportional \
  --assign_w_load 0.1 \
  --release_before_deadline_s 1800 \
  --assignment_scorer cva \
  --assignment_scorer_mix 0.35 \
  --assignment_context_encoder lstm \
  --assignment_context_weight 0.25 \
  --assignment_sequence_hidden_dim 16 \
  --assignment_replan_interval_s 3600 \
  --assignment_replan_horizon_s 7200 \
  --assignment_replan_trigger periodic,dynamic,stale_owner,deadline \
  --assignment_switch_penalty 0.05 \
  --assignment_lock_window_s 600 \
  --assignment_max_switches_per_task 2 \
  --rollout_steps 256 \
  --ppo_epochs 2 \
  --ppo_batch_size 256 \
  --eval_workers 4 \
  --vtw_time_step_s 60 \
  --out_dir runs/compare_cva_mappo_stress \
  --run_name cva_mappo_lstm_rolling_stress \
  --device cpu
```

注意:

- 这条命令会重新训练 `Single-PPO` 和 `Indep-PPO`,耗时更长。
- 如果时间紧,可以只用 A2 的 `Single/Indep` 结果,再结合 A1 或 A4 的 CVA-MAPPO 结果做主表。
- 如果要最严格同表对比,使用本节 A3。

---

## 7. A4: CVA 主消融

目的:回答论文里最重要的三个问题。

| 对比 | 回答的问题 |
|---|---|
| `heuristic_static` vs `heuristic_rolling` | 滚动重分配是否有用 |
| `heuristic_rolling` vs `cva_*_rolling` | 上下文价值感知分配是否有用 |
| `cva_lstm_static` vs `cva_lstm_rolling` | CVA 与 rolling 是否互补 |
| `cva_mlp_rolling` vs `cva_lstm/gru/transformer/set_transformer_rolling` | 外循环/上下文编码器是否有用 |

命令:

```bash
python run_ablation.py \
  --python python \
  --preset cva_assignment_v1 \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --n_satellites 12 \
  --train_iters 30 \
  --eval_episodes 8 \
  --n_routine 1200 \
  --n_dynamic 300 \
  --methods mappo \
  --out_root runs/ablation_cva_assignment_v1_stress \
  --device cpu \
  --jobs 4 \
  --eval_workers 4 \
  --rollout_steps 256 \
  --ppo_epochs 2 \
  --ppo_batch_size 256 \
  --vtw_time_step_s 60 \
  --resume_latest \
  --no_viz \
  --skip_existing
```

输出:

```text
runs/ablation_cva_assignment_v1_stress/<batch>/ablation_summary.csv
runs/ablation_cva_assignment_v1_stress/<batch>/ablation_summary.json
```

---

## 8. A5: CVA 参数消融

如果 A4 中 `cva_lstm_rolling_mix0p35` 表现较好,再跑一个轻量参数消融,验证主方案不是单个超参数偶然有效。

### 8.1 scorer mix 消融

```bash
python run_ablation.py \
  --python python \
  --preset cva_assignment_v1 \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --n_satellites 12 \
  --train_iters 30 \
  --eval_episodes 8 \
  --n_routine 1200 \
  --n_dynamic 300 \
  --methods mappo \
  --cva_context_encoders lstm \
  --cva_scorer_mixes 0.2,0.35,0.5 \
  --cva_context_weight 0.25 \
  --out_root runs/ablation_cva_mix_v1_stress \
  --device cpu \
  --jobs 3 \
  --eval_workers 4 \
  --rollout_steps 256 \
  --ppo_epochs 2 \
  --ppo_batch_size 256 \
  --vtw_time_step_s 60 \
  --resume_latest \
  --no_viz \
  --skip_existing
```

### 8.2 encoder 精简消融

如果时间不够,可以只跑三种最有解释力的编码器:

```bash
python run_ablation.py \
  --python python \
  --preset cva_assignment_v1 \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --n_satellites 12 \
  --train_iters 30 \
  --eval_episodes 8 \
  --n_routine 1200 \
  --n_dynamic 300 \
  --methods mappo \
  --cva_context_encoders mlp,lstm,transformer \
  --cva_scorer_mixes 0.35 \
  --cva_context_weight 0.25 \
  --out_root runs/ablation_cva_encoder_core_stress \
  --device cpu \
  --jobs 3 \
  --eval_workers 4 \
  --rollout_steps 256 \
  --ppo_epochs 2 \
  --ppo_batch_size 256 \
  --vtw_time_step_s 60 \
  --resume_latest \
  --no_viz \
  --skip_existing
```

---

## 9. 结果指标读取

优先看这些字段:

| 指标 | 含义 | 期望 |
|---|---|---|
| `mappo_n_scheduled` | 完成任务数 | 越高越好 |
| `mappo_observation_success_rate` | 可观测任务完成率 | 越高越好 |
| `mappo_dynamic_completion_rate` | 可观测动态任务完成率 | 越高越好 |
| `mappo_duplicate_rate` | 重复观测率 | 越低越好 |
| `mappo_load_balance_cv` | 负载变异系数 | 越低越好 |
| `mappo_avg_off_nadir_deg` | 平均观测角 | 越低越好 |
| `mappo_avg_dynamic_response_s` | 动态任务响应延迟 | 越低越好 |
| `mappo_n_replans` | 重分配次数 | 不能过高 |
| `mappo_owner_churn_rate` | owner 切换率 | 不能过高 |
| `mappo_stale_owner_rate` | 失效 owner 比例 | 越低越好 |
| `mappo_deadline_rescue_rate` | deadline 救援比例 | 作为动态修复证据 |

注意:

- `observation_success_rate` 是基于可观测任务数计算,不是全部任务数。
- 论文表中应同时报告 `n_total_tasks`、`n_feasible_tasks`、`n_scheduled`。
- `raw completion rate` 可作为诊断,但主完成率建议使用可观测任务口径。

---

## 10. 建议论文主表结构

主表方法:

| 方法 | 来源 |
|---|---|
| Single-PPO | A2 或 A3 |
| Indep-PPO | A2 或 A3 |
| Vanilla MAPPO | A2 |
| Heuristic Rolling MAPPO | A4: `heuristic_rolling` |
| CVA-MAPPO | A3 或 A4: `cva_lstm_rolling_mix0p35` |

主表指标:

```text
observation_success_rate
dynamic_completion_rate
routine_completion_rate
n_total_tasks
n_feasible_tasks
n_scheduled
duplicate_rate
load_balance_cv
avg_off_nadir_deg
avg_dynamic_response_s
stale_owner_rate
deadline_rescue_rate
```

消融表:

| 表 | 使用结果 | 目的 |
|---|---|---|
| Ablation-1 | A4 全部子实验 | 主方案组件贡献 |
| Ablation-2 | A5 scorer mix | 参数稳健性 |
| Ablation-3 | A5 encoder core | 外循环上下文编码器价值 |

---

## 11. 断点续跑与版本记录

长实验中断后继续:

```bash
python run_ablation.py \
  --python python \
  --preset cva_assignment_v1 \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --n_satellites 12 \
  --train_iters 30 \
  --eval_episodes 8 \
  --n_routine 1200 \
  --n_dynamic 300 \
  --methods mappo \
  --out_root runs/ablation_cva_assignment_v1_stress \
  --device cpu \
  --jobs 4 \
  --eval_workers 4 \
  --rollout_steps 256 \
  --ppo_epochs 2 \
  --ppo_batch_size 256 \
  --vtw_time_step_s 60 \
  --resume_latest \
  --skip_existing
```

每个子实验都会记录:

```text
manifest.json
comparison_results.json
*_viz_data.json
```

`manifest.json` 中包含:

- 运行命令
- 参数
- git commit
- dirty 状态
- 输出文件路径

---

## 12. 常见问题

### 为什么推荐 `device cpu`

MRL-DMS 当前瓶颈主要是环境 rollout、VTW 计算、任务分配和评估,不是神经网络矩阵计算。GPU 利用率低是正常现象。普通消融建议用:

```bash
--device cpu --jobs 4 --eval_workers 4
```

### `--jobs` 和 `--eval_workers` 区别

- `--jobs`:并行多个子实验,用于普通消融的训练阶段吞吐。
- `--eval_workers`:每个子实验内部并行多个 eval episode。
- 有效评估 worker 上限是 `eval_episodes`。

### `compare_methods.py` 如何多 CPU 训练

完整三方案对比可以用:

```bash
--methods single,indep,mappo --method_jobs 3
```

这会把 `Single-PPO`、`Indep-PPO`、`MAPPO` 三个顶层方法放到独立进程并行训练。它不会改变单个 MAPPO 内部的 rollout 结构;如果只运行 `--methods mappo`,仍然只有一个顶层训练进程。

16 CPU 推荐:

```bash
--method_jobs 3 --eval_workers 4
```

### 动态任务槽位是否需要手动调

当前 `compare_methods.py` 会根据任务规模自动扩容:

```text
max_action_dim >= n_routine + dynamic_insertions_per_day * n_dynamic
```

因此 `1200 + 3×300 = 2100` 会自动扩容到至少 2100。

### 如果只想先跑最小验证

```bash
python run_ablation.py \
  --python python \
  --preset cva_assignment_v1 \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --n_satellites 4 \
  --train_iters 1 \
  --eval_episodes 1 \
  --n_routine 40 \
  --n_dynamic 10 \
  --methods mappo \
  --cva_context_encoders lstm \
  --cva_scorer_mixes 0.35 \
  --out_root runs/smoke_cva_assignment_v1 \
  --device cpu
```
