# MRL-DMS 实验运行路线

本文档用于梳理当前项目中较多优化方案的运行顺序、对比目的和建议观察指标。建议按阶段推进,不要一开始把所有 preset 全部跑完,否则结果很多但难以解释。

## 0. 统一实验配置

正式对比建议固定以下配置,保证横向可比:

```bash
--python /Users/zhouzidie/miniconda3/envs/myenv/bin/python
--n_satellites 6
--train_iters 30
--eval_episodes 5
--n_routine 200
--n_dynamic 50
--seed 42
--device cpu
```

如果时间充足,最终关键实验再补多随机种子,例如 `--seed 42/43/44`。

每个 `run_ablation.py` 批次会自动创建唯一结果目录,每个子实验会输出:

- `comparison_results.json`: 指标结果
- `manifest.json`: 参数、git commit、运行环境
- `*_viz_data.json`: 可视化数据
- 批次根目录 `ablation_summary.json/csv`: 汇总表

## 1. 主复现结果

**目的**:回答 MAPPO 是否优于 Single-PPO / Indep-PPO,作为论文复现和多星扩展的主结果。

```bash
python compare_methods.py \
  --n_satellites 6 --train_iters 30 --eval_episodes 5 \
  --n_routine 200 --n_dynamic 50 \
  --out_dir runs/compare_main \
  --device cpu
```

**对比对象**:

- `Single-PPO`: 单星 PPO
- `Indep-PPO`: 多星但无协同
- `MAPPO`: 多星协同,含全局任务指派和集中式 critic

**重点指标**:

- `n_scheduled`
- `observation_success_rate`
- `dynamic_completion_rate`
- `duplicate_rate`
- `avg_dynamic_response_s`
- `load_balance_cv`
- `coordination_gain`

## 2. 全局任务分配消融

**目的**:验证 episode 级任务所有权、容量比例分配、截止释放机制是否真的带来收益。

```bash
python run_ablation.py \
  --python /Users/zhouzidie/miniconda3/envs/myenv/bin/python \
  --preset assignment_v2 \
  --n_satellites 6 --train_iters 30 --eval_episodes 5 \
  --n_routine 200 --n_dynamic 50 \
  --out_root runs/ablation_assignment_v2 \
  --device cpu
```

**对比目的**:

- `no_episode_assignment`: 关闭全局任务分配,验证所有权机制是否必要
- `assignment_capacity_mode=equal/proportional`: 比较等额分配和按覆盖能力分配
- `assign_w_load=0.05/0.1/0.2`: 观察负载均衡和吞吐的权衡
- `release_before_deadline_s=0/1800`: 验证临近 deadline 释放 owner 是否提升完成率

**重点指标**:

- `n_scheduled`
- `duplicate_rate`
- `load_balance_cv`
- `avg_off_nadir_deg`
- `avg_dynamic_response_s`

## 3. Oracle 上界对比

**目的**:判断当前 MAPPO/分配器距离集中式强启发式参考还有多大差距。

```bash
python run_ablation.py \
  --python /Users/zhouzidie/miniconda3/envs/myenv/bin/python \
  --preset oracle_v1 \
  --n_satellites 6 --train_iters 30 --eval_episodes 5 \
  --n_routine 200 --n_dynamic 50 \
  --out_root runs/ablation_oracle_v1 \
  --device cpu
```

**对比目的**:

- `oracle_no_assignment`: 无全局分配时的 Greedy-Oracle 参考
- `oracle_assignment_v2`: 默认全局分配下的 Greedy-Oracle 参考

**重点指标**:

- `oracle_relative_completion`
- `mappo_oracle_gap_n_scheduled`
- `n_scheduled`

如果 oracle gap 很大,后续应优先研究任务分配、解码器或高层策略;如果 gap 很小,瓶颈可能更多来自任务可见性或数据本身。

## 4. 学习式任务分配 scorer

**目的**:比较不同任务分配 scorer 是否优于手写 heuristic,判断哪类结构最有研究价值。

```bash
python run_ablation.py \
  --python /Users/zhouzidie/miniconda3/envs/myenv/bin/python \
  --preset learned_assignment_v1 \
  --n_satellites 6 --train_iters 30 --eval_episodes 5 \
  --n_routine 200 --n_dynamic 50 \
  --out_root runs/ablation_learned_assignment_v1 \
  --device cpu
```

**对比目的**:

- `heuristic`: 当前手写分配分数
- `mlp`: 只看任务-卫星边特征是否足够
- `lstm/gru`: 任务序列上下文是否有帮助
- `transformer/set_transformer`: 集合建模是否优于人工排序序列
- `gnn`: 卫星-任务二分图结构是否更适合分配问题

**重点指标**:

- `n_scheduled`
- `avg_dynamic_response_s`
- `load_balance_cv`
- `avg_off_nadir_deg`
- `oracle_relative_completion`

## 5. 滚动重分配 / Rolling Horizon

**目的**:验证静态 episode 分配是否会因为动态任务、错过窗口、owner 失效而变差。

```bash
python run_ablation.py \
  --python /Users/zhouzidie/miniconda3/envs/myenv/bin/python \
  --preset assignment_rolling_v1 \
  --n_satellites 6 --train_iters 30 --eval_episodes 5 \
  --n_routine 200 --n_dynamic 50 \
  --out_root runs/ablation_assignment_rolling_v1 \
  --device cpu
```

**对比目的**:

- `rolling_static`: 当前静态 owner
- `rolling_periodic_1h`: 每小时重分配
- `rolling_event`: 动态任务/owner 失效/deadline 触发
- `rolling_mpc_2h`: 只看未来 2 小时窗口的 rolling horizon

**重点指标**:

- `n_replans`
- `n_owner_switches`
- `owner_churn_rate`
- `stale_owner_rate`
- `deadline_rescue_rate`
- `avg_dynamic_response_s`
- `dynamic_completion_rate_raw`

## 6. 层级任务分配接口

**目的**:验证“高层 Assignment Manager + 低层 MAPPO”的接口是否值得继续推进为可学习 manager。

```bash
python run_ablation.py \
  --python /Users/zhouzidie/miniconda3/envs/myenv/bin/python \
  --preset hier_assignment_v1 \
  --n_satellites 6 --train_iters 30 --eval_episodes 5 \
  --n_routine 200 --n_dynamic 50 \
  --out_root runs/ablation_hier_assignment_v1 \
  --device cpu
```

**对比目的**:

- `hier_no_manager`: rolling/MPC,无额外高层 manager
- `hier_rule_manager`: 规则式高层 manager 读取 `get_assignment_state()` 并提出 owner 建议

**重点指标**:

- `n_scheduled`
- `avg_dynamic_response_s`
- `owner_churn_rate`
- `stale_owner_rate`
- `deadline_rescue_rate`

如果 `hier_rule_manager` 相比 `hier_no_manager` 有收益,下一步可以实现 `supervised_manager` 或 GNN/Transformer manager。

## 7. MAPPO 训练机制消融

这些实验用于解释 MAPPO 为什么好或不好,属于辅助分析。

### 7.1 奖励塑形

**目的**:验证团队奖励、负载均衡奖励、团队完成 bonus 是否帮助协同学习。

```bash
python run_ablation.py \
  --python /Users/zhouzidie/miniconda3/envs/myenv/bin/python \
  --preset reward_v1 \
  --n_satellites 6 --train_iters 30 --eval_episodes 5 \
  --n_routine 200 --n_dynamic 50 \
  --out_root runs/ablation_reward_v1 \
  --device cpu
```

### 7.2 集中式 critic 状态

**目的**:验证 critic 是否需要更完整的全局状态,例如 concat 各星观测或追加任务/负载统计。

```bash
python run_ablation.py \
  --python /Users/zhouzidie/miniconda3/envs/myenv/bin/python \
  --preset state_v1 \
  --n_satellites 6 --train_iters 30 --eval_episodes 5 \
  --n_routine 200 --n_dynamic 50 \
  --out_root runs/ablation_state_v1 \
  --device cpu
```

### 7.3 训练稳定性

**目的**:验证卫星数量课程学习和联合探索是否提升 MAPPO 训练稳定性。

```bash
python run_ablation.py \
  --python /Users/zhouzidie/miniconda3/envs/myenv/bin/python \
  --preset train_stability_v1 \
  --n_satellites 6 --train_iters 30 --eval_episodes 5 \
  --n_routine 200 --n_dynamic 50 \
  --out_root runs/ablation_train_stability_v1 \
  --device cpu
```

### 7.4 意图广播通信

**目的**:验证规则式意图广播是否减少多星动作冲突和重复选择。

```bash
python run_ablation.py \
  --python /Users/zhouzidie/miniconda3/envs/myenv/bin/python \
  --preset communication_v1 \
  --n_satellites 6 --train_iters 30 --eval_episodes 5 \
  --n_routine 200 --n_dynamic 50 \
  --out_root runs/ablation_communication_v1 \
  --device cpu
```

## 8. MRL-DMS 外循环结构消融

**目的**:比较论文默认 LSTM 外循环和 GRU/MLP/Transformer/Set Transformer,同时保留 MAPPO+LSTM 外循环对照。

```bash
python run_ablation.py \
  --python /Users/zhouzidie/miniconda3/envs/myenv/bin/python \
  --preset meta_encoder_v1 \
  --meta_encoder_types lstm,gru,mlp,transformer,set_transformer \
  --meta_iterations 2 \
  --meta_mappo_n_satellites 6 \
  --full_train \
  --out_root runs/ablation_meta_encoder_v1 \
  --device cpu
```

**对比目的**:

- `lstm`: 原论文外循环
- `gru`: 更轻量的循环结构
- `mlp`: 无序列记忆 baseline
- `transformer/set_transformer`: 注意力/集合式外循环
- `MAPPO + LSTM`: 多星协同下保留原论文 LSTM 外循环

## 9. 推荐最终论文表格顺序

建议最终论文或报告按以下顺序组织结果:

1. 主对比表: `Single-PPO / Indep-PPO / MAPPO`
2. 任务统计表: 全部任务数、可观测任务数、完成数
3. 全局分配消融: `assignment_v2`
4. Oracle gap: `oracle_v1`
5. 学习式分配 scorer: `learned_assignment_v1`
6. 滚动/层级分配: `assignment_rolling_v1` + `hier_assignment_v1`
7. 训练辅助机制: `reward_v1`、`state_v1`、`train_stability_v1`、`communication_v1`
8. 外循环结构: `meta_encoder_v1`

## 10. 最小优先运行路线

如果计算资源有限,优先运行:

```text
compare_main
assignment_v2
oracle_v1
learned_assignment_v1
assignment_rolling_v1
hier_assignment_v1
```

这条路线最能支撑“多星协同 + 任务分配优化”的主线。

