# CVA-MAPPO v2 消融接口检查与执行命令

本文档根据 `CVA-MAPPO_消融与对比实验设计.docx` 检查当前代码接口，并给出可直接运行的命令。命令均使用显式路径，不使用 `$ACLED` 形式。

## 0. 评估设备说明

当前推荐使用高吞吐配置:

- 训练 rollout: `--train_env_workers 8 --torch_num_threads 1`, 多 CPU 进程并行采样环境。
- 策略更新: `--device cuda:0`, 主进程在 GPU 上进行 MAPPO update。
- 评估: `--eval_device cpu --eval_workers 8`, 多 CPU 进程并行评估 episode。
- 更新强度: `--rollout_steps 512 --ppo_epochs 4 --ppo_batch_size 512`, 增大每次 GPU update 的批量, 减少 GPU 只短暂闪一下的问题。
- 使用 `--scenario_cache_dir` 时,实际评估 episode 数以缓存中的 `eval_scenarios.pkl` 为准。v2 的 `manifest.json` 会记录 `requested_eval_episodes` 和 `actual_eval_episodes`。
- v2 与旧版 `compare_methods.py` 默认均为 stochastic eval;如需确定性评估,加 `--eval_deterministic`。
- 使用预热 VTW 缓存的正式命令应同时包含 `--vtw_cache_dir runs/scenario_cache/cva_stress_sat12_r1200_d300_gs4_seed42/vtw_cache` 与 `--vtw_time_step_s 60`。
- 最新联合约束口径默认追加 `--n_ground_stations 4 --downlink_time_s 300 --satellite_storage_capacity 8 --enable_inter_satellite_transfer --inter_satellite_transfer_time_s 300`。

注意: 单卡 GPU 不建议多个评估进程同时使用。若显式设置 `--eval_device same/cuda:0` 且 `eval_workers > 1`, 代码会自动降为 1。

## 1. 场景预生成

推荐先生成固定训练/评估场景，保证所有方法使用相同任务、到达时间和 VTW 缓存。

```bash
python precompute_scenarios.py \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --n_satellites 12 \
  --n_train_scenarios 800 \
  --n_eval_scenarios 20 \
  --n_routine 1200 \
  --n_dynamic 300 \
  --n_ground_stations 4 \
  --curriculum_stages 300:75,600:150,900:225,1200:300 \
  --vtw_time_step_s 60 \
  --vtw_workers 16 \
  --map_max_scenarios 8 \
  --out_dir runs/scenario_cache/cva_stress_sat12_r1200_d300_gs4_seed42
```

## 2. 接口支持情况

| 实验组 | docx 中的方案 | 当前支持情况 | 推荐入口 |
|---|---|---:|---|
| 主对比 | Greedy/Heuristic | 已支持 | `compare_methods.py --methods oracle` |
| 主对比 | Indep-PPO/IPPO | 已支持 | `compare_methods.py --methods indep` |
| 主对比 | Vanilla MAPPO | 已支持 | `compare_methods.py --methods mappo --no_episode_assignment --candidate_action_top_k 0` |
| 主对比 | MAPPO + Mixed Top-K | 已支持 | `compare_methods.py --methods mappo --no_episode_assignment --candidate_action_top_k 128` |
| 主对比 | CVA-MAPPO | 已支持 | `python -m cva_mappo_v2.run_experiment` |
| 候选动作空间 | Full Action | 已支持 | 旧版 `candidate_action_top_k=0` |
| 候选动作空间 | Random Feasible-K | 未完整实现 | 待实现随机可行动作候选采样器 |
| 候选动作空间 | Mixed Top-K | 已支持 | 旧版 `--candidate_action_top_k` |
| 候选动作空间 | Typed Slots | 已支持 | v2, `--flex_slots 0` |
| 候选动作空间 | Typed Slots + Flex | 已支持 | v2 默认 |
| 槽位比例 | 固定比例 | 已支持 | v2 `--routine_slots/--dynamic_slots/--flex_slots` |
| 槽位比例 | Adaptive | 未完整实现 | 待实现自适应槽位分配 |
| 归属机制 | No Ownership | 已支持 | 旧版 `--no_episode_assignment` |
| 归属机制 | Hard Single Owner | 已支持 | v2 owners 全部设为 1 |
| 归属机制 | Static Multi-Owner | 已支持 | v2 多 owner, 关闭重分配触发 |
| 归属机制 | Dynamic Multi-Owner | 已支持 | v2 多 owner + 事件触发 |
| 归属机制 | Capacity-aware Dynamic Ownership | 已支持 | v2 `capacity_slack/load_penalty` |
| 归属机制 | Full CVA Ownership | 已支持 | v2 默认 |
| 感知器 | Priority-only/Visibility-only/启发式 | 已支持权重消融 | v2 `--w_*` |
| 感知器 | MLP Perceiver | 未完整实现 | 待实现可训练 perceiver |
| 感知器 | History-aware Perceiver | 未完整实现 | 待实现历史编码/训练接口 |
| 分数项 | Base/+Urgency/+Scarcity/+Load/+Future/Full | 已支持权重消融 | v2 `--w_*` |
| 重分配触发 | None/Periodic/Dynamic/Deadline/Stale/Full | 已支持 | v2 `--assignment_replan_trigger` |
| 负载均衡 | w_load 多档 | 已支持 | v2 `--cva_load_penalty` 或 `--w_load` |
| 动态压力 | Low/Medium/High/Urgent | 已支持 | 预生成不同 `n_dynamic`/deadline 场景 |
| 规模扩展 | 3/6/12/18/24 星 | 已支持 | `--n_satellites` + 场景缓存 |

## 3. 主对比命令

### 3.1 Greedy Oracle

```bash
python compare_methods.py \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --scenario_cache_dir runs/scenario_cache/cva_stress_sat12_r1200_d300_gs4_seed42 \
  --vtw_cache_dir runs/scenario_cache/cva_stress_sat12_r1200_d300_gs4_seed42/vtw_cache \
  --n_satellites 12 \
  --train_iters 0 \
  --eval_episodes 20 \
  --n_routine 1200 \
  --n_dynamic 300 \
  --n_ground_stations 4 \
  --downlink_time_s 300 \
  --satellite_storage_capacity 8 \
  --enable_inter_satellite_transfer \
  --inter_satellite_transfer_time_s 300 \
  --methods oracle \
  --vtw_time_step_s 60 \
  --out_dir runs/main_compare_v2/oracle \
  --run_name greedy_oracle_stress \
  --no_viz \
  --device cpu \
  --eval_device cpu \
  --eval_workers 16
```

### 3.2 Indep-PPO

```bash
python compare_methods.py \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --scenario_cache_dir runs/scenario_cache/cva_stress_sat12_r1200_d300_gs4_seed42 \
  --vtw_cache_dir runs/scenario_cache/cva_stress_sat12_r1200_d300_gs4_seed42/vtw_cache \
  --n_satellites 12 \
  --train_iters 30 \
  --eval_episodes 20 \
  --n_routine 1200 \
  --n_dynamic 300 \
  --n_ground_stations 4 \
  --downlink_time_s 300 \
  --satellite_storage_capacity 8 \
  --enable_inter_satellite_transfer \
  --inter_satellite_transfer_time_s 300 \
  --methods indep \
  --rollout_steps 512 \
  --ppo_epochs 4 \
  --ppo_batch_size 512 \
  --train_env_workers 8 \
  --torch_num_threads 1 \
  --eval_device cpu \
  --eval_workers 8 \
  --vtw_time_step_s 60 \
  --out_dir runs/main_compare_v2/indep \
  --run_name indep_ppo_stress \
  --no_viz \
  --device cuda:0
```

### 3.3 Vanilla MAPPO

```bash
python compare_methods.py \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --scenario_cache_dir runs/scenario_cache/cva_stress_sat12_r1200_d300_gs4_seed42 \
  --vtw_cache_dir runs/scenario_cache/cva_stress_sat12_r1200_d300_gs4_seed42/vtw_cache \
  --n_satellites 12 \
  --train_iters 30 \
  --eval_episodes 20 \
  --n_routine 1200 \
  --n_dynamic 300 \
  --n_ground_stations 4 \
  --downlink_time_s 300 \
  --satellite_storage_capacity 8 \
  --enable_inter_satellite_transfer \
  --inter_satellite_transfer_time_s 300 \
  --methods mappo \
  --no_episode_assignment \
  --candidate_action_top_k 0 \
  --rollout_steps 512 \
  --ppo_epochs 4 \
  --ppo_batch_size 512 \
  --train_env_workers 8 \
  --torch_num_threads 1 \
  --eval_device cpu \
  --eval_workers 8 \
  --vtw_time_step_s 60 \
  --out_dir runs/main_compare_v2/vanilla_mappo \
  --run_name vanilla_mappo_full_action_stress \
  --no_viz \
  --device cuda:0
```

### 3.4 MAPPO + Mixed Top-K

```bash
python compare_methods.py \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --scenario_cache_dir runs/scenario_cache/cva_stress_sat12_r1200_d300_gs4_seed42 \
  --vtw_cache_dir runs/scenario_cache/cva_stress_sat12_r1200_d300_gs4_seed42/vtw_cache \
  --n_satellites 12 \
  --train_iters 30 \
  --eval_episodes 20 \
  --n_routine 1200 \
  --n_dynamic 300 \
  --n_ground_stations 4 \
  --downlink_time_s 300 \
  --satellite_storage_capacity 8 \
  --enable_inter_satellite_transfer \
  --inter_satellite_transfer_time_s 300 \
  --methods mappo \
  --no_episode_assignment \
  --candidate_action_top_k 128 \
  --rollout_steps 512 \
  --ppo_epochs 4 \
  --ppo_batch_size 512 \
  --train_env_workers 8 \
  --torch_num_threads 1 \
  --eval_device cpu \
  --eval_workers 8 \
  --vtw_time_step_s 60 \
  --out_dir runs/main_compare_v2/mixed_topk \
  --run_name mappo_mixed_topk128_stress \
  --no_viz \
  --device cuda:0
```

### 3.5 CVA-MAPPO v2

```bash
python -m cva_mappo_v2.run_experiment \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --scenario_cache_dir runs/scenario_cache/cva_stress_sat12_r1200_d300_gs4_seed42 \
  --vtw_cache_dir runs/scenario_cache/cva_stress_sat12_r1200_d300_gs4_seed42/vtw_cache \
  --n_satellites 12 \
  --train_iters 30 \
  --eval_episodes 20 \
  --n_routine 1200 \
  --n_dynamic 300 \
  --n_ground_stations 4 \
  --downlink_time_s 300 \
  --satellite_storage_capacity 8 \
  --enable_inter_satellite_transfer \
  --inter_satellite_transfer_time_s 300 \
  --routine_slots 64 \
  --dynamic_slots 32 \
  --flex_slots 32 \
  --routine_candidate_owners 1 \
  --dynamic_candidate_owners 2 \
  --urgent_candidate_owners 3 \
  --stale_candidate_owners 3 \
  --capacity_slack_ratio 0.05 \
  --cva_load_penalty 0.15 \
  --w_quality 0.42 \
  --w_priority 0.18 \
  --w_deadline 0.14 \
  --w_dynamic 0.10 \
  --w_scarcity 0.10 \
  --w_future_opportunity_loss 0.08 \
  --w_load 0.16 \
  --w_owner_stability 0.04 \
  --release_before_deadline_s 1800 \
  --dynamic_broadcast_window_s 1800 \
  --assignment_replan_interval_s 3600 \
  --assignment_replan_horizon_s 7200 \
  --assignment_replan_trigger periodic,dynamic,stale_owner,deadline \
  --assignment_switch_penalty 0.05 \
  --owner_switch_margin 0.08 \
  --ownership_mask_mode soft \
  --candidate_owner_bonus 0.06 \
  --slot_selection_mode mixed \
  --assignment_lock_window_s 600 \
  --assignment_max_switches_per_task 2 \
  --rollout_steps 512 \
  --ppo_epochs 4 \
  --ppo_batch_size 512 \
  --train_env_workers 8 \
  --torch_num_threads 1 \
  --eval_device cpu \
  --eval_workers 8 \
  --vtw_time_step_s 60 \
  --out_dir runs/main_compare_v2/cva_mappo_v2 \
  --run_name cva_mappo_v2_stress \
  --no_viz \
  --device cuda:0
```

## 4. 候选动作空间消融

Full Action、Mixed Top-K 走旧版 `compare_methods.py`; Typed Slots 走 v2。

```bash
python compare_methods.py \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --scenario_cache_dir runs/scenario_cache/cva_stress_sat12_r1200_d300_gs4_seed42 \
  --vtw_cache_dir runs/scenario_cache/cva_stress_sat12_r1200_d300_gs4_seed42/vtw_cache \
  --n_satellites 12 --train_iters 30 --eval_episodes 20 \
  --n_routine 1200 --n_dynamic 300 \
  --n_ground_stations 4 \
  --downlink_time_s 300 \
  --satellite_storage_capacity 8 \
  --enable_inter_satellite_transfer \
  --inter_satellite_transfer_time_s 300 \
  --methods mappo --no_episode_assignment \
  --candidate_action_top_k 0 \
  --rollout_steps 512 --ppo_epochs 4 --ppo_batch_size 512 \
  --train_env_workers 8 --torch_num_threads 1 \
  --eval_device cpu --eval_workers 8 \
  --vtw_time_step_s 60 \
  --out_dir runs/ablation_v2/action_space/full_action \
  --run_name full_action \
  --no_viz --device cuda:0
```

```bash
python compare_methods.py \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --scenario_cache_dir runs/scenario_cache/cva_stress_sat12_r1200_d300_gs4_seed42 \
  --vtw_cache_dir runs/scenario_cache/cva_stress_sat12_r1200_d300_gs4_seed42/vtw_cache \
  --n_satellites 12 --train_iters 30 --eval_episodes 20 \
  --n_routine 1200 --n_dynamic 300 \
  --n_ground_stations 4 \
  --downlink_time_s 300 \
  --satellite_storage_capacity 8 \
  --enable_inter_satellite_transfer \
  --inter_satellite_transfer_time_s 300 \
  --methods mappo --no_episode_assignment \
  --candidate_action_top_k 128 \
  --rollout_steps 512 --ppo_epochs 4 --ppo_batch_size 512 \
  --train_env_workers 8 --torch_num_threads 1 \
  --eval_device cpu --eval_workers 8 \
  --vtw_time_step_s 60 \
  --out_dir runs/ablation_v2/action_space/mixed_topk128 \
  --run_name mixed_topk128 \
  --no_viz --device cuda:0
```

```bash
python -m cva_mappo_v2.run_experiment \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --scenario_cache_dir runs/scenario_cache/cva_stress_sat12_r1200_d300_gs4_seed42 \
  --vtw_cache_dir runs/scenario_cache/cva_stress_sat12_r1200_d300_gs4_seed42/vtw_cache \
  --n_satellites 12 --train_iters 30 --eval_episodes 20 \
  --n_routine 1200 --n_dynamic 300 \
  --n_ground_stations 4 \
  --downlink_time_s 300 \
  --satellite_storage_capacity 8 \
  --enable_inter_satellite_transfer \
  --inter_satellite_transfer_time_s 300 \
  --routine_slots 64 --dynamic_slots 32 --flex_slots 0 \
  --slot_selection_mode typed \
  --rollout_steps 512 --ppo_epochs 4 --ppo_batch_size 512 \
  --train_env_workers 8 --torch_num_threads 1 \
  --eval_device cpu --eval_workers 8 \
  --vtw_time_step_s 60 \
  --out_dir runs/ablation_v2/action_space/typed_no_flex \
  --run_name typed_no_flex \
  --no_viz --device cuda:0
```

```bash
python -m cva_mappo_v2.run_experiment \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --scenario_cache_dir runs/scenario_cache/cva_stress_sat12_r1200_d300_gs4_seed42 \
  --vtw_cache_dir runs/scenario_cache/cva_stress_sat12_r1200_d300_gs4_seed42/vtw_cache \
  --n_satellites 12 --train_iters 30 --eval_episodes 20 \
  --n_routine 1200 --n_dynamic 300 \
  --n_ground_stations 4 \
  --downlink_time_s 300 \
  --satellite_storage_capacity 8 \
  --enable_inter_satellite_transfer \
  --inter_satellite_transfer_time_s 300 \
  --routine_slots 64 --dynamic_slots 32 --flex_slots 32 \
  --slot_selection_mode typed \
  --rollout_steps 512 --ppo_epochs 4 --ppo_batch_size 512 \
  --train_env_workers 8 --torch_num_threads 1 \
  --eval_device cpu --eval_workers 8 \
  --vtw_time_step_s 60 \
  --out_dir runs/ablation_v2/action_space/typed_flex \
  --run_name typed_flex \
  --no_viz --device cuda:0
```

## 5. 槽位比例消融

只改变槽位配置，其余参数保持一致。

```bash
python -m cva_mappo_v2.run_experiment \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --scenario_cache_dir runs/scenario_cache/cva_stress_sat12_r1200_d300_gs4_seed42 \
  --vtw_cache_dir runs/scenario_cache/cva_stress_sat12_r1200_d300_gs4_seed42/vtw_cache \
  --n_satellites 12 --train_iters 30 --eval_episodes 20 \
  --n_routine 1200 --n_dynamic 300 \
  --n_ground_stations 4 \
  --downlink_time_s 300 \
  --satellite_storage_capacity 8 \
  --enable_inter_satellite_transfer \
  --inter_satellite_transfer_time_s 300 \
  --routine_slots 56 --dynamic_slots 8 --flex_slots 0 \
  --slot_selection_mode typed \
  --rollout_steps 512 --ppo_epochs 4 --ppo_batch_size 512 \
  --train_env_workers 8 --torch_num_threads 1 \
  --eval_device cpu --eval_workers 8 \
  --vtw_time_step_s 60 \
  --out_dir runs/ablation_v2/slot_ratio/r56_d8_f0 \
  --run_name r56_d8_f0 \
  --no_viz --device cuda:0
```

将 `--routine_slots --dynamic_slots --flex_slots` 分别替换为:

- `48 16 0`
- `40 16 8`
- `32 24 8`
- `64 32 32`

## 6. 归属机制消融

### 6.1 Hard Single Owner

```bash
python -m cva_mappo_v2.run_experiment \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --scenario_cache_dir runs/scenario_cache/cva_stress_sat12_r1200_d300_gs4_seed42 \
  --vtw_cache_dir runs/scenario_cache/cva_stress_sat12_r1200_d300_gs4_seed42/vtw_cache \
  --n_satellites 12 --train_iters 30 --eval_episodes 20 \
  --n_routine 1200 --n_dynamic 300 \
  --n_ground_stations 4 \
  --downlink_time_s 300 \
  --satellite_storage_capacity 8 \
  --enable_inter_satellite_transfer \
  --inter_satellite_transfer_time_s 300 \
  --routine_candidate_owners 1 \
  --dynamic_candidate_owners 1 \
  --urgent_candidate_owners 1 \
  --stale_candidate_owners 1 \
  --ownership_mask_mode hard \
  --candidate_owner_bonus 0 \
  --assignment_replan_trigger none \
  --assignment_replan_interval_s 0 \
  --rollout_steps 512 --ppo_epochs 4 --ppo_batch_size 512 \
  --train_env_workers 8 --torch_num_threads 1 \
  --eval_device cpu --eval_workers 8 \
  --vtw_time_step_s 60 \
  --out_dir runs/ablation_v2/ownership/hard_single_owner \
  --run_name hard_single_owner \
  --no_viz --device cuda:0
```

### 6.2 Static Multi-Owner

```bash
python -m cva_mappo_v2.run_experiment \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --scenario_cache_dir runs/scenario_cache/cva_stress_sat12_r1200_d300_gs4_seed42 \
  --vtw_cache_dir runs/scenario_cache/cva_stress_sat12_r1200_d300_gs4_seed42/vtw_cache \
  --n_satellites 12 --train_iters 30 --eval_episodes 20 \
  --n_routine 1200 --n_dynamic 300 \
  --n_ground_stations 4 \
  --downlink_time_s 300 \
  --satellite_storage_capacity 8 \
  --enable_inter_satellite_transfer \
  --inter_satellite_transfer_time_s 300 \
  --routine_candidate_owners 1 \
  --dynamic_candidate_owners 2 \
  --urgent_candidate_owners 3 \
  --stale_candidate_owners 3 \
  --assignment_replan_trigger none \
  --assignment_replan_interval_s 0 \
  --rollout_steps 512 --ppo_epochs 4 --ppo_batch_size 512 \
  --train_env_workers 8 --torch_num_threads 1 \
  --eval_device cpu --eval_workers 8 \
  --vtw_time_step_s 60 \
  --out_dir runs/ablation_v2/ownership/static_multi_owner \
  --run_name static_multi_owner \
  --no_viz --device cuda:0
```

### 6.3 Full CVA Ownership

使用主对比中的 CVA-MAPPO v2 命令。

## 7. 匹配分数项消融

Priority-only:

```bash
python -m cva_mappo_v2.run_experiment \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --scenario_cache_dir runs/scenario_cache/cva_stress_sat12_r1200_d300_gs4_seed42 \
  --vtw_cache_dir runs/scenario_cache/cva_stress_sat12_r1200_d300_gs4_seed42/vtw_cache \
  --n_satellites 12 --train_iters 30 --eval_episodes 20 \
  --n_routine 1200 --n_dynamic 300 \
  --n_ground_stations 4 \
  --downlink_time_s 300 \
  --satellite_storage_capacity 8 \
  --enable_inter_satellite_transfer \
  --inter_satellite_transfer_time_s 300 \
  --w_quality 0 --w_priority 1 --w_deadline 0 --w_dynamic 0 \
  --w_scarcity 0 --w_future_opportunity_loss 0 --w_load 0 --w_owner_stability 0 \
  --rollout_steps 512 --ppo_epochs 4 --ppo_batch_size 512 \
  --train_env_workers 8 --torch_num_threads 1 \
  --eval_device cpu --eval_workers 8 \
  --vtw_time_step_s 60 \
  --out_dir runs/ablation_v2/scorer/priority_only \
  --run_name priority_only \
  --no_viz --device cuda:0
```

Visibility/quality-only:

```bash
python -m cva_mappo_v2.run_experiment \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --scenario_cache_dir runs/scenario_cache/cva_stress_sat12_r1200_d300_gs4_seed42 \
  --vtw_cache_dir runs/scenario_cache/cva_stress_sat12_r1200_d300_gs4_seed42/vtw_cache \
  --n_satellites 12 --train_iters 30 --eval_episodes 20 \
  --n_routine 1200 --n_dynamic 300 \
  --n_ground_stations 4 \
  --downlink_time_s 300 \
  --satellite_storage_capacity 8 \
  --enable_inter_satellite_transfer \
  --inter_satellite_transfer_time_s 300 \
  --w_quality 1 --w_priority 0 --w_deadline 0 --w_dynamic 0 \
  --w_scarcity 0 --w_future_opportunity_loss 0 --w_load 0 --w_owner_stability 0 \
  --rollout_steps 512 --ppo_epochs 4 --ppo_batch_size 512 \
  --train_env_workers 8 --torch_num_threads 1 \
  --eval_device cpu --eval_workers 8 \
  --vtw_time_step_s 60 \
  --out_dir runs/ablation_v2/scorer/quality_only \
  --run_name quality_only \
  --no_viz --device cuda:0
```

Full score 使用主对比 CVA-MAPPO v2 默认权重。`+Urgency/+Scarcity/+Load/+Future` 可以在 priority/quality base 上逐项打开对应 `--w_*`。

## 8. 重分配触发消融

将主对比 CVA-MAPPO v2 命令中的 `--assignment_replan_trigger` 替换为:

- `none`
- `periodic`
- `dynamic`
- `deadline`
- `stale_owner`
- `periodic,dynamic,stale_owner,deadline`

并分别修改 `--out_dir` 与 `--run_name`。

## 9. 负载均衡消融

将主对比 CVA-MAPPO v2 命令中的 `--cva_load_penalty` 替换为:

- `0.0`
- `0.05`
- `0.15`
- `0.30`

如需同时消融分数项中的 load penalty, 同步修改 `--w_load`。

## 10. 动态压力与规模扩展

动态压力通过不同场景缓存实现。每组先运行 `precompute_scenarios.py`, 再运行主 CVA-MAPPO v2 命令。

推荐压力组:

- low: `--n_routine 1200 --n_dynamic 100`
- medium: `--n_routine 1200 --n_dynamic 300`
- high: `--n_routine 1200 --n_dynamic 600`

规模扩展推荐:

- small: `--n_satellites 3 --n_routine 300 --n_dynamic 75`
- medium: `--n_satellites 6 --n_routine 600 --n_dynamic 150`
- large: `--n_satellites 12 --n_routine 1200 --n_dynamic 300`
- stress: `--n_satellites 18 --n_routine 1800 --n_dynamic 450`

## 11. 当前仍需补充的接口

- Random Feasible-K: 需要新增随机可行候选采样器, 用于证明“不是任意缩小动作空间都有效”。
- Adaptive Slot Ratio: 需要按当前任务压力动态调整 routine/dynamic/flex 槽位。
- 可训练 MLP/History-aware Perceiver: 当前 v2 是显式价值评分器, 支持权重消融, 但还不是端到端训练的高层感知器。
- 更完整的候选诊断指标: candidate overlap、valid action ratio、slot utilization、coverage@K 可作为论文图表指标继续补。
