# MRL-DMS 实验结果记录

> 本文档用于记录已经完成的关键实验结果、主要结论和后续待验证问题。实验运行流程与命令见 [EXPERIMENT_README.md](EXPERIMENT_README.md)。

---

## 1. 主对比实验: Single-PPO vs Indep-PPO vs MAPPO

### 1.1 实验信息

| 项目 | 内容 |
|---|---|
| 实验名称 | `compare_main` 主对比 |
| 输出目录 | `runs/compare_main/single_compare_sat6_iter30_assign_on_seed42_20260701_170916/` |
| 结果文件 | `comparison_results.json` |
| 实验记录 | `manifest.json` |
| 卫星数量 | 6 |
| 训练迭代 | 30 |
| 评估 episode | 5 |
| 每 episode 任务 | 200 routine + 50 dynamic 插入配置,输出表中平均全部任务数为 350 |
| 随机种子 | 42 |
| 总耗时 | 4503.4s |

### 1.2 原始指标表

| 指标 | Single-PPO | Indep-PPO | MAPPO |
|---|---:|---:|---:|
| 观测成功率 | 97.5% | 98.0% | 98.2% |
| 动态完成率 | 100.0% | 99.1% | 99.1% |
| 常规完成率 | 97.2% | 97.8% | 98.1% |
| 累积奖励 | 48.81 | 313.28 | 183.97 |
| 全部任务数 | 350.00 | 350.00 | 350.00 |
| 可观测任务数 | 47.60 | 175.60 | 175.60 |
| 可观测常规任务数 | 43.00 | 154.60 | 154.60 |
| 可观测动态任务数 | 4.60 | 21.00 | 21.00 |
| 完成任务数 | 46.40 | 172.00 | 172.40 |
| 重复观测数 | 0.00 | 124.40 | 0.00 |
| 重复观测率 | - | 42.0% | 0.0% |
| 负载变异系数 | - | 0.20 | 0.19 |
| 平均 off-nadir | 14.99 | 14.79 | 12.63 |
| 动态响应延迟(s) | 10793.40 | 9718.47 | 10084.04 |
| 重分配次数 | - | 0.00 | 0.00 |
| owner 切换数 | - | 0.00 | 0.00 |
| owner 切换率 | - | 0.0% | 0.0% |
| 失效 owner 比例 | - | 0.0% | 100.0% |
| deadline 救援率 | - | 0.0% | 0.0% |
| 协同增益 | - | 0.62 | 0.62 |
| Oracle 相对完成率 | - | - | - |

### 1.3 关键对比结论

| 对比点 | 结论 |
|---|---|
| MAPPO vs Indep-PPO 完成数 | MAPPO 平均完成 172.40 个任务,略高于 Indep-PPO 的 172.00。两者可观测任务数相同,说明 MAPPO 没有牺牲吞吐。 |
| MAPPO vs Indep-PPO 重复观测 | Indep-PPO 平均重复观测 124.40 次,重复率 42.0%;MAPPO 重复观测为 0。这是当前最强的协同证据。 |
| MAPPO vs Indep-PPO 观测质量 | MAPPO 平均 off-nadir 为 12.63°,优于 Indep-PPO 的 14.79°,说明协同分配不仅去重,还改善了观测质量。 |
| MAPPO vs Indep-PPO 负载均衡 | MAPPO `load_balance_cv=0.19`,略优于 Indep-PPO 的 0.20,但优势不大。后续还需要通过分配/奖励消融继续压低 CV。 |
| MAPPO vs Indep-PPO 动态响应 | MAPPO 动态响应 10084.04s,慢于 Indep-PPO 的 9718.47s,但快于 Single-PPO 的 10793.40s。动态任务响应仍是后续优化重点。 |
| MAPPO vs Indep-PPO 奖励 | Indep-PPO 奖励更高,但它存在大量重复观测,重复刷分会抬高奖励。因此该实验不能只用 `total_reward` 判断优劣。 |
| MAPPO vs Single-PPO | 多星方案可观测任务数从 47.60 提升到 175.60,完成任务数从 46.40 提升到 172.40,体现多星覆盖范围提升。 |

### 1.4 当前可写进总结的主结论

1. **MAPPO 的核心优势是协同去重**:在与 Indep-PPO 可观测任务数相同的条件下,MAPPO 将重复观测率从 42.0% 降到 0.0%,同时完成任务数略有提升。
2. **MAPPO 提升了观测质量**:平均 off-nadir 从 Indep-PPO 的 14.79° 降到 12.63°,说明任务所有权和冲突处理让卫星更倾向于执行质量更好的观测窗口。
3. **MAPPO 的完成率保持稳定**:观测成功率 98.2%,高于 Single-PPO 和 Indep-PPO;常规任务完成率也最高。
4. **奖励指标需要谨慎解释**:Indep-PPO 的 `total_reward=313.28` 高于 MAPPO 的 183.97,但它伴随 124.40 次重复观测,说明奖励被重复任务污染。后续报告中应把 `duplicate_rate`、`n_duplicates` 与奖励一起展示。
5. **动态响应仍有优化空间**:MAPPO 相比 Single-PPO 更快,但慢于 Indep-PPO,后续应优先验证滚动重分配、动态任务容量预留和 deadline rescue 机制。

### 1.5 需要继续审计的问题

| 问题 | 现象 | 后续动作 |
|---|---|---|
| `stale_owner_rate=100.0%` | MAPPO 输出失效 owner 比例为 100%,但完成率和重复率正常 | 审计 `stale_owner_rate` 的定义和分母,确认它是否只统计带 owner 的未完成任务,避免误读 |
| `coordination_gain=0.62` | 多星完成数相对 `6 × Single-PPO` 只有 0.62 | 该指标受可观测任务重叠和单星可达任务分布影响,后续建议结合 Oracle gap 解释 |
| 动态响应慢于 Indep-PPO | MAPPO 10084.04s vs Indep-PPO 9718.47s | 运行 `assignment_rolling_v1`、`hier_assignment_v1` 和 Oracle 对比,判断瓶颈在静态 owner 还是策略执行 |
| MAPPO 奖励低于 Indep-PPO | MAPPO 去重后奖励降低 | 检查奖励是否需要对重复观测惩罚或团队去重收益重标定 |

---

## 2. 后续推荐实验

基于当前主对比结果,下一步先做基础三方案压力测试即可。暂时不展开滚动重分配、高层 manager 或 Oracle 上界实验,先验证在更大任务规模和更紧张资源条件下,MAPPO 的协同去重优势是否会进一步转化为有效完成数、动态响应和观测质量优势。

### 2.1 基础三方案压力测试

```bash
python compare_methods.py \
  --acled_path "$ACLED" \
  --n_satellites 6 --train_iters 30 --eval_episodes 5 \
  --n_routine 600 --n_dynamic 150 \
  --methods single,indep,mappo \
  --out_dir runs/compare_stress \
  --device cuda:0
```

目的:在任务规模扩大到 `600 routine + 150 dynamic` 后,重新比较 Single-PPO、Indep-PPO 和 MAPPO。

该压力配置需要 `1050` 个任务槽位,当前 `compare_methods.py` 会自动扩容 `max_action_dim`,避免动态任务被丢弃导致 `dynamic_completion_rate` 失真。

重点看 `duplicate_rate`、`n_scheduled`、`observation_success_rate`、`dynamic_completion_rate`、`avg_dynamic_response_s`、`load_balance_cv` 和 `avg_off_nadir_deg`。如果压力测试中 MAPPO 相比 Indep-PPO 的完成数或动态响应优势明显扩大,就能支撑“资源越紧张,协同调度越有价值”的结论。

---

## 3. 写作建议

最终报告中建议把该主对比作为第一张核心表。叙述顺序:

1. 先说明多星带来可观测任务数提升:Single-PPO 47.60,多星 175.60。
2. 再说明无协同多星虽然完成率高,但重复观测严重:Indep-PPO 重复率 42.0%。
3. 然后强调 MAPPO 的协同收益:完成率最高、重复率 0、off-nadir 最低。
4. 最后诚实指出动态响应和 owner 失效指标仍需优化,引出后续滚动重分配和高层 manager 实验。
