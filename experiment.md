# MRL-DMS 多 CPU 实验运行方案

## 1. 并行逻辑

MRL-DMS 不是 GPU 密集型任务。主要耗时来自环境 rollout、VTW 计算、任务调度和评估。

普通消融分两层并行:

- 训练阶段:用 `run_ablation.py --jobs N` 并行运行多个子实验。
- 评估阶段:用 `--eval_workers M` 在每个子实验内部并行多个 eval episode。

注意:

- `--eval_workers` 的有效上限是 `--eval_episodes`。
- `--num_workers` 只对 `meta_encoder_v1` 这种训练型消融有效,对普通消融无效。
- 总 CPU 压力大约是 `jobs × eval_workers`;如果机器是 16 核,建议从 `--jobs 4 --eval_workers 4` 开始。

## 2. 普通消融推荐命令

适用于:

- `assignment_v2`
- `assignment_rolling_v1`
- `hier_assignment_v1`
- `learned_assignment_v1`
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

## 3. 训练型消融推荐命令

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

## 4. 结果检查

确认参数是否生效:

```bash
grep -R '"eval_workers"' runs/ablation_assignment_rolling_v1_stress | head
grep -R '"command"' runs/ablation_assignment_rolling_v1_stress | head
```

普通消融并行评估时,日志应出现:

```text
并行评估 MAPPO: episodes=8, eval_workers=4
```

训练型消融并行评估时,日志应出现:

```text
并行评估: episodes=3, eval_workers=3
```

## 5. 推荐默认值

16 核 CPU:

```bash
--jobs 4 --eval_workers 4
```

32 核 CPU:

```bash
--jobs 4 --eval_workers 8
```

如果出现系统负载过高、单个子实验变慢,先降低 `--jobs`,再降低 `--eval_workers`。
