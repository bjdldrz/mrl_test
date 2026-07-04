# MRL-DMS 实验流程与结果对比文档

> 目标:用统一、可复现、可断点续跑的方式完成论文复现、多星协同优化消融和后续结果对比。
> 当前版本建议先建立完整 baseline,之后所有优化消融默认只跑 `MAPPO`,避免反复训练不变的 `Single-PPO` / `Indep-PPO`。

---

## 1. 实验总路线

建议按下面顺序执行。每一步都写入独立目录,便于后续画表、画图和回溯版本。

| 阶段 | 实验 | 主要目的 | 推荐方法 |
|---|---|---|---|
| S0 | 冒烟测试 | 确认环境、依赖、输出目录可用 | `--train_iters 0/--fast` |
| S1 | 论文/基础 baseline | 建立 Single-PPO、Indep-PPO、MAPPO 的统一对照 | `--methods single,indep,mappo` |
| S2 | 任务分配消融 | 比较全局指派、学习式 scorer、滚动重分配、CVA-MAPPO 主方案 | 默认 `--methods mappo` |
| S3 | 协同机制消融 | 比较奖励、critic state、训练稳定性、通信机制 | 默认 `--methods mappo` |
| S4 | 元学习结构消融 | 比较 LSTM/GRU/MLP/Transformer/Set Transformer 外循环 | `meta_encoder_v1` |
| S5 | Oracle/上界实验 | 判断 MAPPO 离强启发式上界还有多远 | `--methods mappo,oracle` |
| S6 | 结果汇总 | 生成最终表格、图和论文叙述材料 | `ablation_summary.csv/json` |

推荐原则:

1. 先跑一次完整三方案 baseline,后续消融只改变 MAPPO 配置。
2. 每类实验使用单独 `--out_root`,不要把不同任务混到一个目录。
3. 长实验中断后使用 `--resume_latest --skip_existing`,不要重新开新批次重复跑。
4. 每个结果目录里的 `manifest.json` 会记录命令、参数、git commit 和 dirty 状态,用于版本追溯。

---

## 2. 通用运行参数

服务器/AutoDL 推荐直接使用真实 ACLED 数据路径:

```text
./DynamicMission/DynamicMission.shp
```

消融实验通用参数:

```bash
--n_satellites 6 \
--train_iters 30 \
--eval_episodes 5 \
--n_routine 200 \
--n_dynamic 50 \
--seed 42 \
--device cuda:0
```

快速检查命令是否正确,不真正运行:

```bash
python run_ablation.py \
  --python python \
  --preset assignment_v2 \
  --dry_run \
  --train_iters 0 \
  --eval_episodes 1 \
  --device cuda:0
```

断点续跑:

```bash
python run_ablation.py \
  --python python \
  --preset learned_assignment_v1 \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --methods mappo \
  --out_root runs/ablation_learned_assignment_v1 \
  --device cuda:0 \
  --resume_latest \
  --skip_existing
```

`--resume_latest` 会复用当前参数匹配的最新批次目录;`--skip_existing` 会跳过已有 `manifest.json` 或 `summary.json` 的子实验。

---

## 3. S0 冒烟测试

目的:先确认代码、依赖、目录输出和最小训练流程正常。

```bash
python train.py --method mrl_dms --fast --device cuda:0
```

MAPPO-only 对比 smoke:

```bash
python compare_methods.py \
  --methods mappo \
  --train_iters 0 \
  --eval_episodes 1 \
  --n_satellites 2 \
  --n_routine 8 \
  --n_dynamic 1 \
  --out_dir runs/smoke_mappo_only \
  --flat_out_dir \
  --device cuda:0
```

通过标准:

- 生成 `comparison_results.json` 和 `manifest.json`。
- 控制台表格能列出 `n_total_tasks`、`n_feasible_tasks`、完成率、重复率等指标。
- MAPPO-only 时 `coordination_gain` 和 `oracle_relative_completion` 可为空,这是正常的。

---

## 4. S1 基础三方案 baseline

目的:建立后续所有消融的共同参照。

```bash
python compare_methods.py \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --n_satellites 6 \
  --train_iters 30 \
  --eval_episodes 5 \
  --n_routine 200 \
  --n_dynamic 50 \
  --methods single,indep,mappo \
  --out_dir runs/compare_baseline \
  --device cuda:0
```

重点对比:

| 指标 | 解释 | 期待现象 |
|---|---|---|
| `n_feasible_tasks` | 可观测任务数,完成率的主要分母 | 用来解释 raw 完成率偏低 |
| `observation_success_rate` | feasible 口径观测完成率 | MAPPO 应保持较高 |
| `duplicate_rate` | 多星重复观测率 | MAPPO 应显著低于 Indep-PPO |
| `load_balance_cv` | 负载均衡程度 | 越低越均衡 |
| `avg_dynamic_response_s` | 动态任务响应延迟 | 越低越好 |
| `avg_off_nadir_deg` | 平均观测角 | 越低表示质量越高 |
| `coordination_gain` | 多星相对单星的协同增益 | 仅在包含 Single-PPO 时有意义 |

说明:

- `Indep-PPO` 的 `total_reward` 可能因为重复观测而偏高,不能只看奖励。
- 完成率默认使用可观测任务数作为分母,同时保留 `*_raw` 和 `feasible_ratio` 诊断全部任务口径。

---

## 5. S2 任务分配优化实验

这一组是当前多星优化主线。建议先跑低成本 MAPPO-only 消融,找到较优配置后再挑关键组合做完整三方案对比。

### 5.1 全局任务指派参数消融

目的:比较是否启用 episode 级指派、容量模式、负载权重和截止释放窗口。

```bash
python compare_methods.py \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --n_satellites 6 \
  --train_iters 30 \
  --eval_episodes 5 \
  --n_routine 200 \
  --n_dynamic 50 \
  --methods mappo \
  --out_root runs/ablation_assignment_v2 \
  --device cuda:0
```

若要完整三方案对比:

```bash
python run_ablation.py \
  --python python \
  --preset assignment_v2 \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --methods single,indep,mappo \
  --out_root runs/ablation_assignment_v2_full \
  --device cuda:0
```

比较目的:

- `no_assignment` vs `assign_*`:全局任务指派是否带来收益。
- `equal` vs `proportional`:等额容量和按覆盖能力容量的差异。
- `assign_w_load`:负载均衡与吞吐之间的权衡。
- `release_before_deadline_s`:截止前释放 owner 是否能减少硬指派带来的损失。

### 5.2 基础三方案压力试验

目的:在更大任务规模下直接比较 `Single-PPO / Indep-PPO / MAPPO`,验证资源更紧张时 MAPPO 的去重、负载均衡和观测质量优势是否能进一步转化为有效完成数优势。该实验只改变任务规模,不引入滚动重分配、学习式 scorer 或高层 manager,适合作为主对比后的第一组压力测试。

```bash
python compare_methods.py \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --n_satellites 6 \
  --train_iters 30 \
  --eval_episodes 5 \
  --n_routine 600 \
  --n_dynamic 150 \
  --methods single,indep,mappo \
  --out_dir runs/compare_stress \
  --device cuda:0
```

重点比较:

- `duplicate_rate`:任务量变大后 Indep-PPO 的重复观测是否更严重,MAPPO 是否继续保持 0 或接近 0。
- `n_scheduled`:资源紧张时 MAPPO 是否完成更多有效任务。
- `observation_success_rate` / `dynamic_completion_rate`:压力场景下完成率是否拉开差距。
- `avg_dynamic_response_s`:动态任务更多时 MAPPO 响应是否改善。
- `load_balance_cv` / `avg_off_nadir_deg`:MAPPO 是否仍保持负载和观测质量优势。

压力测试需要 `600 + 3×150 = 1050` 个任务槽位。`compare_methods.py` 会按任务规模自动扩容 `max_action_dim`,避免动态任务因默认 800 槽位不足而被丢弃;如需手动指定,可追加 `--max_action_dim 1200`。

若要验证更大星座下 MAPPO 是否仍能协同规划、避免 Indep-PPO 随卫星数增加而产生更多重复观测,可运行大规模星座压力测试:

```bash
python compare_methods.py \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --n_satellites 12 \
  --train_iters 30 \
  --eval_episodes 5 \
  --n_routine 1200 \
  --n_dynamic 300 \
  --methods single,indep,mappo \
  --out_dir runs/compare_scale_sat12 \
  --device cuda:0
```

当 `--n_satellites` 超过默认 6 颗时,`compare_methods.py` 会基于原 6 颗 SSO 卫星生成带 RAAN/相位偏移的派生星座。该实验重点看 `duplicate_rate` 是否随 Indep-PPO 星数放大而升高,以及 MAPPO 是否在保持完成率的同时继续把重复观测压到 0 或接近 0。

后续优化版本消融统一采用该压力口径:

```text
n_satellites = 12
n_routine = 1200
n_dynamic = 300
methods = mappo
```

这样可以避免轻载场景下所有方法都接近可观测任务上限,导致优化收益不明显。

### 5.3 学习式任务分配 scorer 压力消融

目的:在大规模星座压力场景下比较 heuristic、MLP、LSTM、GRU、Transformer、Set Transformer、GNN 分配 scorer。后续任务分配类优化默认使用该压力口径,避免轻载场景下优化空间不足。

```bash
python run_ablation.py \
  --python python \
  --preset learned_assignment_v1 \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --n_satellites 12 \
  --train_iters 30 \
  --eval_episodes 5 \
  --n_routine 1200 \
  --n_dynamic 300 \
  --methods mappo \
  --ppo_batch_size 1024\
  --assignment_scorer_mixes 0.1,0.25,0.5 \
  --assignment_sequence_scorers lstm,gru \
  --assignment_sequence_mixes 0.25 \
  --assignment_attention_scorers transformer,set_transformer \
  --assignment_attention_mixes 0.25 \
  --assignment_graph_scorers gnn \
  --assignment_graph_mixes 0.25 \
  --out_root runs/ablation_learned_assignment_v1_stress \
  --device cuda:0
```

建议先看:

- `mappo_n_scheduled`
- `mappo_observation_success_rate`
- `mappo_dynamic_completion_rate`
- `mappo_load_balance_cv`
- `mappo_avg_dynamic_response_s`
- `mappo_avg_off_nadir_deg`

### 5.4 滚动重分配压力消融

目的:比较静态 owner、周期重分配、事件触发重分配、MPC 窗口重分配。

```bash
python run_ablation.py \
  --python python \
  --preset assignment_rolling_v1 \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --n_satellites 12 \
  --train_iters 30 \
  --eval_episodes 5 \
  --n_routine 1200 \
  --n_dynamic 300 \
  --methods mappo \
  --out_root runs/ablation_assignment_rolling_v1_stress \
  --device cuda:0 \
  --jobs 2 \
  --eval_workers 4 \
  --train_env_workers 4 \
  --torch_num_threads 1 \
  --rollout_steps 512 \
  --ppo_epochs 4 \
  --ppo_batch_size 512 \
  --vtw_time_step_s 60
```

说明:`assignment_rolling_v1` 属于普通消融;批量压力测试建议用 `--jobs 2 --train_env_workers 4 --eval_workers 4`,避免 16 核机器过载。`--eval_workers` 的有效上限是 `--eval_episodes`。

重点新增指标:

- `mappo_n_replans`:重分配次数。
- `mappo_n_owner_switches`:owner 切换次数。
- `mappo_owner_churn_rate`:owner 切换率。
- `mappo_stale_owner_rate`:失效 owner 比例。
- `mappo_deadline_rescue_rate`:临近截止救援比例。

### 5.5 高层分配 manager + 低层 MAPPO 压力消融

目的:验证规则式高层 manager 是否优于普通滚动重分配,为后续学习式高层策略铺接口。

```bash
python run_ablation.py \
  --python python \
  --preset hier_assignment_v1 \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --n_satellites 12 \
  --train_iters 30 \
  --eval_episodes 5 \
  --n_routine 1200 \
  --n_dynamic 300 \
  --methods mappo \
  --out_root runs/ablation_hier_assignment_v1_stress \
  --device cuda:0
```

比较目的:

- `hier_no_manager`:仅使用滚动/MPC 分配。
- `hier_rule_manager`:增加规则式高层 assignment manager。

### 5.6 CVA-MAPPO 主方案压力消融

目的:将外循环思想接入任务分配阶段,验证“上下文价值感知 owner 分配 + 滚动重分配 + 低层 MAPPO”是否优于纯规则分配。该组建议作为论文主方法消融。

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
  --device cuda:0 \
  --jobs 2 \
  --eval_workers 4 \
  --train_env_workers 4 \
  --torch_num_threads 1 \
  --rollout_steps 512 \
  --ppo_epochs 4 \
  --ppo_batch_size 512 \
  --vtw_time_step_s 60 \
  --resume_latest \
  --skip_existing
```

默认子实验:

- `heuristic_static`:静态规则 owner 分配。
- `heuristic_rolling`:规则 owner + rolling replan。
- `cva_lstm_static`:CVA-LSTM,不启用 rolling,用于隔离 CVA scorer 本身。
- `cva_mlp_rolling`:无序/无记忆边价值 baseline。
- `cva_lstm/gru_rolling`:序列外循环上下文。
- `cva_transformer/set_transformer_rolling`:任务集合/注意力上下文。

主要结论对应:

- `heuristic_static` vs `heuristic_rolling`:滚动重分配收益。
- `heuristic_rolling` vs `cva_*_rolling`:上下文价值感知分配收益。
- `cva_lstm_static` vs `cva_lstm_rolling`:CVA 与 rolling 的互补收益。
- `cva_mlp_rolling` vs 其他 encoder:外循环/上下文编码器收益。

---

## 6. S3 协同机制消融

### 6.1 协同奖励压力消融

目的:判断团队奖励、负载奖励、团队完成 bonus 和奖励归一化是否改善协同。

```bash
python run_ablation.py \
  --python python \
  --preset reward_v1 \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --n_satellites 12 \
  --train_iters 30 \
  --eval_episodes 5 \
  --n_routine 1200 \
  --n_dynamic 300 \
  --methods mappo \
  --out_root runs/ablation_reward_v1_stress \
  --device cuda:0
```

重点看:

- 完成率是否上升。
- `duplicate_rate` 是否保持低位。
- `load_balance_cv` 是否下降。
- `total_reward` 是否和任务指标方向一致。

### 6.2 Critic 全局状态压力消融

目的:比较 mean pooling、追加任务统计、concat 全局状态等 critic 输入方式。

```bash
python run_ablation.py \
  --python python \
  --preset state_v1 \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --n_satellites 12 \
  --train_iters 30 \
  --eval_episodes 5 \
  --n_routine 1200 \
  --n_dynamic 300 \
  --methods mappo \
  --out_root runs/ablation_state_v1_stress \
  --device cuda:0
```

比较目的:

- mean pooling 是否丢失关键信息。
- 任务统计是否帮助 critic 估计全局价值。
- concat 是否在 6 星规模下带来更好结果。

### 6.3 训练稳定性压力消融

目的:比较卫星数量 curriculum、联合探索、组合策略对 MAPPO 训练稳定性的影响。

```bash
python run_ablation.py \
  --python python \
  --preset train_stability_v1 \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --n_satellites 12 \
  --train_iters 30 \
  --eval_episodes 5 \
  --n_routine 1200 \
  --n_dynamic 300 \
  --methods mappo \
  --out_root runs/ablation_train_stability_v1_stress \
  --device cuda:0
```

### 6.4 执行期通信压力消融

目的:比较无通信、意图广播、意图广播 + 稳定训练策略。

```bash
python run_ablation.py \
  --python python \
  --preset communication_v1 \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --n_satellites 12 \
  --train_iters 30 \
  --eval_episodes 5 \
  --n_routine 1200 \
  --n_dynamic 300 \
  --methods mappo \
  --out_root runs/ablation_communication_v1_stress \
  --device cuda:0
```

重点看:

- 冲突后改派是否更有效。
- 重复率是否继续保持 0 或接近 0。
- 动态响应延迟是否下降。

---

## 7. S4 外循环元学习结构消融

目的:比较原论文 LSTM 外循环与 GRU、MLP、Transformer、Set Transformer,并验证 MAPPO + LSTM 外循环分支。

```bash
python run_ablation.py \
  --python python \
  --preset meta_encoder_v1 \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --seed 42 \
  --meta_iterations 12 \
  --meta_encoder_types lstm,gru,mlp,transformer,set_transformer \
  --meta_mappo_n_satellites 2 \
  --n_routine 200 \
  --n_dynamic 50 \
  --num_workers 8 \
  --meta_batch_size 8 \
  --inner_steps 2 \
  --rollout_steps 512 \
  --eval_workers 8 \
  --eval_interval 20 \
  --ppo_epochs 4 \
  --ppo_batch_size 512 \
  --out_root runs/ablation_meta_encoder_v1_eval \
  --device cuda:0
```

说明:

- 默认是高吞吐配置,适合验证接口和中等规模趋势。
- MRL-DMS 的瓶颈通常在 CPU 环境 rollout、VTW 和评估;训练型消融优先用 `--num_workers/--meta_batch_size`、`--eval_workers` 和更大的 PPO 更新批量控制吞吐。
- `train_log.csv` 会记录 `sample_s`、`modulation_s`、`worker_map_s`、`meta_apply_s`、`meta_opt_s`、`eval_s`。若 `worker_map_s` 占比最高,说明主要卡在 worker 内的环境模拟/VTW;若 `eval_s` 高,调大 `--eval_interval`。
- `--fast` 下评估间隔为 5;`--meta_iterations 2` 不会触发评估,短跑主要看 `best_train_reward`、`last_train_reward` 和 `last_train_dynamic_rate`。
- 若要比较评估奖励,将 `--meta_iterations` 提高到至少 6;此时重点看 `best_eval_reward` / `best_reward`。
- `train.py` 会按训练池和评估规模自动扩容 `max_action_dim`;如需手动指定,可追加 `--max_action_dim 800`。
- `meta_encoder_v1` 的 `--n_routine/--n_dynamic` 用作训练型消融的 eval 任务规模,默认 `200 + 3×50`。
- 如果日志出现“未指定 ACLED 数据”,说明 `./DynamicMission/DynamicMission.shp` 文件不存在或路径无效,当前实验会退回合成动态任务。
- 若要完整训练,加 `--full_train`。
- 每个子实验输出 `summary.json`、`train_log.csv`、`eval_log.csv`。
- 这组不是 `compare_methods.py` 三方案对比,而是调用 `train.py` 的训练型消融。

---

## 8. S5 Oracle/上界压力实验

目的:估计 MAPPO 与集中式启发式上界的距离。

```bash
python run_ablation.py \
  --python python \
  --preset oracle_v1 \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --n_satellites 12 \
  --train_iters 30 \
  --eval_episodes 5 \
  --n_routine 1200 \
  --n_dynamic 300 \
  --methods mappo,oracle \
  --out_root runs/ablation_oracle_v1_stress \
  --device cuda:0
```

重点指标:

- `mappo_oracle_gap_n_scheduled`:Oracle 完成数 - MAPPO 完成数。
- `mappo_oracle_relative_completion`:MAPPO 相对 Oracle 的完成比例。

注意:当前 Oracle 是 Greedy-Oracle,不是严格 ILP 最优,但适合做统一强启发式参考。

---

## 9. S6 结果汇总与横向对比

每个批次根目录会生成:

```text
runs/<ablation_root>/<batch_name>/
├── ablation_summary.csv
├── ablation_summary.json
├── <tag_1>/
│   ├── comparison_results.json
│   ├── manifest.json
│   └── *_viz_data.json
└── <tag_2>/
    ├── comparison_results.json
    └── manifest.json
```

建议最终报告至少做 4 张表:

| 表 | 数据来源 | 目的 |
|---|---|---|
| Baseline 三方案表 | `runs/compare_baseline/**/comparison_results.json` | 证明 MAPPO 协同价值 |
| 任务分配消融表 | `assignment_v2/learned_assignment/rolling/hier` 的 summary | 找最佳任务分配策略 |
| 协同机制消融表 | `reward/state/train_stability/communication` 的 summary | 判断收益来自奖励、状态还是通信 |
| Oracle gap 表 | `oracle_v1` summary | 说明离启发式上界还有多少空间 |

推荐排序字段:

1. 先按 `mappo_observation_success_rate` 或 `mappo_n_scheduled` 降序。
2. 再按 `mappo_duplicate_rate` 升序。
3. 再按 `mappo_avg_dynamic_response_s` 升序。
4. 最后看 `mappo_load_balance_cv` 和 `mappo_avg_off_nadir_deg`。

MAPPO-only 消融的 summary 只会包含 `mappo_*` 字段。只有运行了 `--methods single,indep,mappo` 时,才会出现 `indep_*`、`single_*` 和 `delta_*` 对比列。

---

## 10. 可视化

方案对比图:

```bash
python visualize.py --compare_json runs/compare_baseline/<run_name>/comparison_results.json
```

训练曲线:

```bash
python visualize.py --run_dir runs/<train_run_dir>
```

多次训练对比:

```bash
python visualize.py \
  --run_dirs runs/exp_a runs/exp_b \
  --labels LSTM GRU
```

`compare_methods.py` 会写出 `*_viz_data.json`,用于任务分布图和任务调度甘特图。若某个子实验没有可视化文件,优先检查该子实验是否真实运行完成,而不是只做了 `--dry_run`。

---

## 11. 结果解读口径

完成率:

- `observation_success_rate`、`dynamic_completion_rate`、`routine_completion_rate` 使用可观测任务数作为分母。
- `*_raw` 使用全部任务数作为分母,主要用于诊断轨道覆盖稀疏性。
- `n_feasible_tasks`、`n_feasible_routine`、`n_feasible_dynamic` 必须和完成率一起报告。

协同指标:

- `duplicate_rate`:越低越好,MAPPO 应显著低于 Indep-PPO。
- `load_balance_cv`:越低越均衡,但过低可能牺牲吞吐。
- `avg_dynamic_response_s`:越低越好,动态任务优化重点。
- `avg_off_nadir_deg`:越低表示观测质量越好。
- `coordination_gain`:只有包含 Single-PPO baseline 时才解释。
- `oracle_relative_completion`:只有包含 Oracle 时才解释。

任务分配类指标:

- `n_replans` 太高可能表示重分配过于频繁。
- `owner_churn_rate` 太高可能导致训练非平稳。
- `stale_owner_rate` 下降通常说明滚动重分配有效。
- `deadline_rescue_rate` 上升说明释放/救援机制在发挥作用。

---

## 12. 推荐实验组合

当前先优先跑基础 baseline 和基础三方案压力测试:

```bash
# 1. 完整 baseline
python compare_methods.py --acled_path ./DynamicMission/DynamicMission.shp --methods single,indep,mappo \
  --n_satellites 6 --train_iters 30 --eval_episodes 5 \
  --n_routine 200 --n_dynamic 50 --out_dir runs/compare_baseline --device cuda:0

# 2. 基础三方案压力测试
python compare_methods.py --acled_path ./DynamicMission/DynamicMission.shp --methods single,indep,mappo \
  --n_satellites 6 --train_iters 30 --eval_episodes 5 \
  --n_routine 600 --n_dynamic 150 --out_dir runs/compare_stress --device cuda:0
```

压力测试完成后,后续优化版本消融也使用同一压力口径,以保证有足够优化空间:

```bash
# 3. 全局任务指派
python run_ablation.py --python python --preset assignment_v2 --acled_path ./DynamicMission/DynamicMission.shp \
  --n_satellites 12 --train_iters 30 --eval_episodes 5 \
  --n_routine 1200 --n_dynamic 300 --methods mappo \
  --rollout_steps 512 --ppo_epochs 4 --ppo_batch_size 512 --train_env_workers 4 --torch_num_threads 1 --eval_workers 4 --vtw_time_step_s 60 \
  --out_root runs/ablation_assignment_v2_stress --device cuda:0

# 4. 学习式任务分配 scorer
python run_ablation.py --python python --preset learned_assignment_v1 --acled_path ./DynamicMission/DynamicMission.shp \
  --n_satellites 12 --train_iters 30 --eval_episodes 5 \
  --n_routine 1200 --n_dynamic 300 --methods mappo \
  --rollout_steps 512 --ppo_epochs 4 --ppo_batch_size 512 --train_env_workers 4 --torch_num_threads 1 --eval_workers 4 --vtw_time_step_s 60 \
  --out_root runs/ablation_learned_assignment_v1_stress --device cuda:0

# 5. 滚动重分配 + 层级 manager
python run_ablation.py --python python --preset assignment_rolling_v1 --acled_path ./DynamicMission/DynamicMission.shp \
  --n_satellites 12 --train_iters 30 --eval_episodes 5 \
  --n_routine 1200 --n_dynamic 300 --methods mappo \
  --rollout_steps 512 --ppo_epochs 4 --ppo_batch_size 512 --train_env_workers 4 --torch_num_threads 1 --eval_workers 4 --vtw_time_step_s 60 \
  --out_root runs/ablation_assignment_rolling_v1_stress --device cuda:0
python run_ablation.py --python python --preset hier_assignment_v1 --acled_path ./DynamicMission/DynamicMission.shp \
  --n_satellites 12 --train_iters 30 --eval_episodes 5 \
  --n_routine 1200 --n_dynamic 300 --methods mappo \
  --rollout_steps 512 --ppo_epochs 4 --ppo_batch_size 512 --train_env_workers 4 --torch_num_threads 1 --eval_workers 4 --vtw_time_step_s 60 \
  --out_root runs/ablation_hier_assignment_v1_stress --device cuda:0

# 6. Oracle gap
python run_ablation.py --python python --preset oracle_v1 --acled_path ./DynamicMission/DynamicMission.shp \
  --n_satellites 12 --train_iters 30 --eval_episodes 5 \
  --n_routine 1200 --n_dynamic 300 --methods mappo,oracle \
  --rollout_steps 512 --ppo_epochs 4 --ppo_batch_size 512 --train_env_workers 4 --torch_num_threads 1 --eval_workers 4 --vtw_time_step_s 60 \
  --out_root runs/ablation_oracle_v1_stress --device cuda:0
```

如果需要论文复现完整性,再补:

```bash
python run_ablation.py --python python --preset reward_v1 --acled_path ./DynamicMission/DynamicMission.shp \
  --n_satellites 12 --train_iters 30 --eval_episodes 5 \
  --n_routine 1200 --n_dynamic 300 --methods mappo \
  --rollout_steps 512 --ppo_epochs 4 --ppo_batch_size 512 --train_env_workers 4 --torch_num_threads 1 --eval_workers 4 --vtw_time_step_s 60 \
  --out_root runs/ablation_reward_v1_stress --device cuda:0
python run_ablation.py --python python --preset state_v1 --acled_path ./DynamicMission/DynamicMission.shp \
  --n_satellites 12 --train_iters 30 --eval_episodes 5 \
  --n_routine 1200 --n_dynamic 300 --methods mappo \
  --rollout_steps 512 --ppo_epochs 4 --ppo_batch_size 512 --train_env_workers 4 --torch_num_threads 1 --eval_workers 4 --vtw_time_step_s 60 \
  --out_root runs/ablation_state_v1_stress --device cuda:0
python run_ablation.py --python python --preset communication_v1 --acled_path ./DynamicMission/DynamicMission.shp \
  --n_satellites 12 --train_iters 30 --eval_episodes 5 \
  --n_routine 1200 --n_dynamic 300 --methods mappo \
  --rollout_steps 512 --ppo_epochs 4 --ppo_batch_size 512 --train_env_workers 4 --torch_num_threads 1 --eval_workers 4 --vtw_time_step_s 60 \
  --out_root runs/ablation_communication_v1_stress --device cuda:0
python run_ablation.py --python python --preset meta_encoder_v1 --acled_path ./DynamicMission/DynamicMission.shp \
  --meta_iterations 12 --n_routine 200 --n_dynamic 50 \
  --num_workers 8 --meta_batch_size 8 --inner_steps 2 --rollout_steps 512 --eval_workers 8 --eval_interval 20 \
  --ppo_epochs 4 --ppo_batch_size 512 \
  --out_root runs/ablation_meta_encoder_v1 --device cuda:0
```

---

## 13. 实验记录模板

每完成一批实验,建议在记录中写:

```text
实验名称:
运行日期:
git commit:
命令:
输出目录:
是否使用 ACLED:
训练迭代 / 评估 episode:
主要结论:
最佳 tag:
关键指标:
  n_feasible_tasks:
  mappo_n_scheduled:
  mappo_observation_success_rate:
  mappo_dynamic_completion_rate:
  mappo_duplicate_rate:
  mappo_load_balance_cv:
  mappo_avg_dynamic_response_s:
  mappo_avg_off_nadir_deg:
问题/下一步:
```

最终对比时优先引用 `manifest.json` 中的命令和 git 信息,不要只依赖手写记录。
