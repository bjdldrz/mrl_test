# MRL-DMS 多 CPU 实验运行方案

## 1. 并行逻辑

MRL-DMS 不是 GPU 密集型任务。主要耗时来自环境 rollout、VTW 计算、任务调度和评估。

普通消融分两层并行:

- 训练阶段:用 `run_ablation.py --jobs N` 并行运行多个子实验。
- 评估阶段:用 `--eval_workers M` 在每个子实验内部并行多个 eval episode。

注意:

- `--eval_workers` 的有效上限是 `--eval_episodes`。
- `--num_workers` 只对 `meta_encoder_v1` 这种训练型消融有效,对普通消融无效。
- `--method_jobs` 只对 `compare_methods.py --methods single,indep,mappo` 这种多方法对比有效;A1 只跑 `mappo` 时不会带来训练并行收益。
- `--train_env_workers` 只并行 MAPPO 训练 rollout;A1 单方法推荐 4,批量消融时要和 `--jobs` 折中。
- `--torch_num_threads` 控制单个训练进程内 PyTorch CPU 线程数;A1 可设 4,多子实验/多方法并行时通常保持默认或设 1-2。
- `--vtw_cache_dir` 启用 VTW 磁盘缓存,可跨进程复用同一卫星-目标的可见窗口。
- `--candidate_action_top_k` 控制多星低层策略的候选动作空间;0 为 full action,128 是当前压力场景推荐值。
- `--no_viz` 会跳过可视化 JSON,避免并行评估完成后为了可视化额外串行重跑 1 个 episode。
- 总 CPU 压力大约是 `jobs × eval_workers`;如果机器是 16 核,建议从 `--jobs 4 --eval_workers 4` 开始。

## 2. 普通消融推荐命令

适用于:

- `assignment_v2`
- `assignment_rolling_v1`
- `hier_assignment_v1`
- `learned_assignment_v1`
- `cva_assignment_v1`
- `owner_effect_v1`
- `reward_v1`
- `state_v1`
- `communication_v1`
- `oracle_v1`

以滚动重分配压力消融为例:

```bash
python run_ablation.py \
  --python python \
  --preset assignment_rolling_v1 \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --n_satellites 12 \
  --train_iters 30 \
  --eval_episodes 8 \
  --n_routine 1200 \
  --n_dynamic 300 \
  --methods mappo \
  --out_root runs/ablation_assignment_rolling_v1_stress \
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

如果 CPU 核数充足,可以扩大为:

```bash
--jobs 4 --eval_episodes 16 --eval_workers 8
```

不要写成:

```bash
--device cpu
--eval_workers 4
```

缺少反斜杠时,Shell 会在 `--device cpu` 结束命令,后续参数不会传入。

## 3. 论文主方案 CVA-MAPPO

当前建议把论文主方法收敛为 `CVA-MAPPO`:上下文价值感知任务分配 + 滚动 owner 重分配 + 低层 MAPPO 调度。

正式压力消融命令:

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
  --candidate_action_top_k 128 \
  --vtw_time_step_s 60 \
  --resume_latest \
  --no_viz \
  --skip_existing
```

cva_assignment_v1 默认会运行:

- `heuristic_static`:静态规则 owner 分配。
- `heuristic_rolling`:规则 owner + 滚动重分配。
- `cva_lstm_static`:CVA-LSTM 但不滚动,用于隔离 CVA 本身。
- `cva_mlp/gru/lstm/transformer/set_transformer_rolling`:不同上下文编码器的 CVA rolling 主消融。

核心对比目的:

- `heuristic_static` vs `heuristic_rolling`:滚动重分配是否有用。
- `heuristic_rolling` vs `cva_*_rolling`:上下文价值分配是否有用。
- `cva_lstm_static` vs `cva_lstm_rolling`:CVA 与 rolling 是否互补。
- `cva_mlp` vs `cva_lstm/gru/transformer/set_transformer`:外循环上下文编码器是否优于无序/无记忆打分。

单独跑 A1 主方案时,推荐使用:

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
  --train_env_workers 4 \
  --eval_workers 4 \
  --torch_num_threads 4 \
  --vtw_cache_dir runs/vtw_cache \
  --candidate_action_top_k 128 \
  --vtw_time_step_s 60 \
  --out_dir runs/main_cva_mappo_train_eval \
  --run_name cva_mappo_lstm_rolling_stress \
  --no_viz \
  --device cpu
```

如果这次运行需要后续可视化,删除 `--no_viz`。

候选动作空间消融:

```bash
python run_ablation.py \
  --python python \
  --preset candidate_action_v1 \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --n_satellites 12 \
  --train_iters 30 \
  --eval_episodes 8 \
  --n_routine 1200 \
  --n_dynamic 300 \
  --methods mappo \
  --candidate_action_top_ks 0,64,128,256 \
  --out_root runs/ablation_candidate_action_v1_stress \
  --device cpu \
  --jobs 4 \
  --eval_workers 4 \
  --torch_num_threads 2 \
  --vtw_cache_dir runs/vtw_cache \
  --rollout_steps 256 \
  --ppo_epochs 2 \
  --ppo_batch_size 256 \
  --vtw_time_step_s 60 \
  --resume_latest \
  --no_viz \
  --skip_existing
```

owner 预分配效果消融:

```bash
python run_ablation.py \
  --python python \
  --preset owner_effect_v1 \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --n_satellites 12 \
  --train_iters 30 \
  --eval_episodes 8 \
  --n_routine 1200 \
  --n_dynamic 300 \
  --out_root runs/ablation_owner_effect_v1_stress \
  --device cpu \
  --jobs 2 \
  --eval_workers 4 \
  --train_env_workers 2 \
  --torch_num_threads 2 \
  --vtw_cache_dir runs/vtw_cache \
  --candidate_action_top_k 128 \
  --rollout_steps 256 \
  --ppo_epochs 2 \
  --ppo_batch_size 256 \
  --vtw_time_step_s 60 \
  --resume_latest \
  --no_viz \
  --skip_existing
```

owner_effect_v1 默认会运行:

- `no_owner_indep_ppo`:无 owner、各星独立执行、统一评估,用于观察重复观测和资源浪费。
- `no_owner_mappo`:无 owner,但使用 MAPPO 协同执行,用于区分低层协同与高层 owner 预分配。
- `owner_heuristic_static`:静态 owner 分配。
- `owner_cva_lstm_rolling`:最终 CVA-LSTM rolling owner 分配。

核心对比目的:

- `no_owner_indep_ppo` vs `owner_heuristic_static`:预先分配 owner 是否能降低重复观测。
- `no_owner_mappo` vs `owner_cva_lstm_rolling`:低层协同之外,高层 owner 责任边界是否仍然有价值。
- `owner_heuristic_static` vs `owner_cva_lstm_rolling`:价值感知和滚动修复是否进一步提升动态任务表现。

## 4. 训练型消融推荐命令

适用于 `meta_encoder_v1`。

```bash
python run_ablation.py \
  --python python \
  --preset meta_encoder_v1 \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --meta_encoder_types lstm,gru,mlp,transformer,set_transformer \
  --meta_iterations 12 \
  --n_routine 200 \
  --n_dynamic 50 \
  --out_root runs/ablation_meta_encoder_v1_eval \
  --device cpu \
  --num_workers 8 \
  --meta_batch_size 8 \
  --inner_steps 2 \
  --rollout_steps 256 \
  --eval_interval 20 \
  --eval_workers 4 \
  --ppo_epochs 2 \
  --ppo_batch_size 256 \
  --resume_latest \
  --skip_existing
```

这里:

- `--num_workers` 控制训练阶段 meta batch worker。
- `--meta_batch_size` 控制每轮采样多少任务。
- `--eval_workers` 控制评估 episode 并行。
- `train_log.csv` 中的 `worker_map_s`、`eval_s` 用于判断瓶颈。

## 5. 结果检查

确认参数是否生效:

```bash
grep -R '"eval_workers"' runs/ablation_assignment_rolling_v1_stress | head
grep -R '"command"' runs/ablation_assignment_rolling_v1_stress | head
grep -R '"assignment_context_encoder"' runs/ablation_cva_assignment_v1_stress | head
```

普通消融并行评估时,日志应出现:

```text
并行评估 MAPPO: episodes=8, eval_workers=4
```

训练型消融并行评估时,日志应出现:

```text
并行评估: episodes=3, eval_workers=3
```

## 6. 推荐默认值

16 核 CPU:

```bash
--jobs 4 --eval_workers 4
```

32 核 CPU:

```bash
--jobs 4 --eval_workers 8
```

如果出现系统负载过高、单个子实验变慢,先降低 `--jobs`,再降低 `--eval_workers`。
