# 多星协同(MAPPO)优化路线图

> 本文档记录 MRL-DMS 多星协同方案的优化方向,供后续逐项落地。
> 创建于 2026-06-21,基于 `compare_methods.py` 的对比结果分析。

---

## 2026-07-06 CVA-guided Mixed-TopK 调整记录

最新压力实验显示 Mixed-TopK 是当前最强基线,而 hard-owner CVA-v2 会因为过强的 owner mask 降低当前可执行槽位暴露。进一步代码审计发现,上一版虽然改成 soft owner,但仍按 routine/dynamic/flex 固定类型槽位配额截断,并不是真正的 Mixed-TopK。因此本轮将默认槽位选择改为共享 Top-K:

- 默认 `--slot_selection_mode mixed`:routine/dynamic/flex 不再按固定配额截断,所有候选在一个共享 Top-K 列表中排序。
- 默认 `--ownership_mask_mode soft`:不再用 owner 硬屏蔽当前可执行任务,保留 Mixed-TopK 的高可执行性。
- 修正 soft+mixed 候选来源:共享 Top-K 不再受 `candidate_ids` 和短重分配 horizon 限制,而是扫描本星全任务池;CVA owner 只作为排序 bonus。
- 新增 `--candidate_owner_bonus`:候选 owner 只获得排序加分,降低重复竞争倾向,但不剥夺非 owner 的可执行动作。
- 保留 `--slot_selection_mode typed` 和 `--ownership_mask_mode hard`:用于复现/消融原 typed/hard-owner v2。

核心对比:

- Mixed-TopK-like: `--slot_selection_mode mixed --ownership_mask_mode soft --candidate_owner_bonus 0`
- CVA-guided Mixed-TopK: `--slot_selection_mode mixed --ownership_mask_mode soft --candidate_owner_bonus 0.06`
- Typed soft-owner ablation: `--slot_selection_mode typed --ownership_mask_mode soft --candidate_owner_bonus 0.06`
- Hard-owner CVA-v2: `--slot_selection_mode typed --ownership_mask_mode hard --candidate_owner_bonus 0`

重点验证 CVA 软引导是否能在接近 Mixed-TopK 完成率的前提下,降低重复观测率、改善负载均衡或动态响应。

## 2026-07-06 CVA-MAPPO v2 压力场景修正记录

压力测试中出现 `avg_filled_slots` 上升但 `avg_valid_slots` 仍接近 0 的问题,说明扩大候选 owner 只增加了未来候选任务,没有增加当前可执行动作;同时 `owner_churn_rate` 过高,导致低层 MAPPO 面对非平稳槽位语义。

本轮实现:

- **Executable-first slot exposure**:构造类型化候选槽位时,强制把当前 action mask 允许执行的任务纳入候选排序池,避免未来高分任务挤掉当前可执行任务。
- **Dynamic short-window broadcast**:动态任务到达后的短窗口内,当前可执行卫星即使不是 owner 也可以临时看到该任务,提高动态任务响应概率。
- **Stable primary owner margin**:重分配时保留旧 owner,除非新 owner 的容量修正分数超过 `owner_switch_margin + switch_penalty`,降低 owner churn。
- **诊断指标增强**:增加 routine/dynamic/flex 分类型槽位有效率和平均有效槽位数,用于区分“槽位填充不足”和“填充了但不可执行”。

新增消融接口:

- `--dynamic_broadcast_window_s`:0 表示关闭动态短时广播,建议对比 0/1800/3600。
- `--owner_switch_margin`:0 表示允许更频繁切换,建议对比 0/0.08/0.12。

重点观察指标:

- `dynamic_completion_rate`, `avg_dynamic_response_s`
- `avg_valid_slots`, `avg_valid_dynamic_slots`
- `owner_churn_rate`, `n_owner_switches`, `duplicate_rate`

## 一、对比结果暴露的 4 个真问题

| 现象 | 数据(smoke 测试) | 说明 |
|---|---|---|
| **协同反而让负载更不均** | MAPPO CV=0.46 > Indep CV=0.21 | 最反常——协同应更均衡却更差 |
| **协同增益没到位** | gain=0.92 < 1 | 6 星没达到"接近 N× 单星"的理想 |
| **完成率/奖励没压过无协同** | 96.4% vs 98.0% | 只在重复观测、响应延迟上赢 |
| **奖励是各星独立的** | `mappo_trainer` 每星各算各的 GAE | 没有团队奖励、没有反事实信用分配,协同只靠环境去冲突"被动"产生 |

**根源**:当前协同只发生在**环境层**(去冲突 + 状态同步),策略层 / 奖励层 / 通信层几乎没有真正的协同机制。

当前实现关键点(优化前基线):
- 冲突解决 = `envs/multi_satellite_env.py` 中「先到先得,其余强制 idle」
- 全局状态 = 各星局部观测的 mean pooling(`get_global_state()`)
- 奖励 = 每星独立的 env reward,各自算 GAE(`algo/mappo_trainer.py`)
- 执行期零通信(纯 CTDE 分布式 actor)

> 难度标注:🟢 快速 / 🟡 中等 / 🔴 研究级

---

## 二、优化方案清单(按问题根源分类)

### A. 冲突解决机制(最大快赢点)
当前「先到先得,其余强制 idle」→ 抢输的星白白浪费一步,直接导致负载不均 + gain<1。

- A1. 🟢 **抢输后改派次优任务**:冲突时败者不 idle,从其 action_mask 选下一个可行任务,杜绝空步。
- A2. 🟡 **基于边际价值择优分配**:同一任务多星争抢时,分给 VTW 质量最好 / off-nadir 最小 / 负载最轻的星。
- A3. 🟡 **匈牙利算法 / 拍卖机制最优指派**:每步「星×任务」建二分图,用 Hungarian 或贪心拍卖求全局最优分配。
- A4. 🔴 **自回归联合动作**:各星按序决策,后者看到前者已选动作,结构上避免冲突。

### B. 负载均衡(CV 反而变差,必须专门处理)
- B5. 🟢 **加负载均衡奖励/惩罚项**:对各星完成数方差/CV 加惩罚,或给"最闲的星完成任务"额外奖励。
- B6. 🟢 **冲突 tie-break 改为按负载**:把 A 的"先到先得"换成"派给最闲的星",一行逻辑压低 CV。
- B7. 🟡 **每星单窗任务数软上限**:超阈值后该星对新任务 mask 收紧,逼任务流向其他星。

### C. 奖励与信用分配(当前完全是个体奖励)
- C8. 🟡 **引入团队奖励**:`r = α·个体 + (1-α)·全队总奖励`,让策略为集体优化。
- C9. 🔴 **COMA 反事实优势**:用集中式 critic 估计"第 i 星改 idle/改选别的"的反事实基线,得到真实边际贡献。
- C10. 🔴 **值分解(QMIX/VDN/QTRAN)**:联合价值分解到各星,单调性保证个体最优≈联合最优。
- C11. 🟡 **互补覆盖奖励**:奖励"覆盖不同地理区域/时间窗"的行为,鼓励分工。
- C12. 🟡 **势能塑形(potential-based shaping)**:用"剩余可行任务数下降"做势函数,不改最优策略前提下加速收敛。

### D. 全局状态表示(mean pooling 丢信息)
- D13. 🟡 **注意力聚合替代均值**:attention 聚合各星观测,让 critic 知道"哪颗星此刻重要"。
- D14. 🟡 **拼接式全局状态 + agent-id 嵌入**:critic 看完整拼接状态(含身份),信息无损。
- D15. 🔴 **GNN 建模星座**:N 颗星建图(边=轨道邻近/可见性重叠),图网络做 critic。
- D16. 🟢 **全局状态加入任务级信息**:补"哪些任务已认领/待处理"的全局任务表,而非只有 obs 均值。

### E. 执行期通信(现在执行时零通信)
- E17. 🟡 **意图广播**:决策前各星广播"打算做哪个任务",据此二次决策——轻量去冲突。
- E18. 🔴 **可学习通信(CommNet/TarMAC/IC3Net)**:各星学习收发消息向量,端到端训练。
- E19. 🔴 **注意力通信**:学习"该跟谁通信、关注什么",稀疏高效。

### F. 算法与策略结构
- F20. 🟡 **MAPPO → HAPPO/HATRPO**:异构智能体顺序更新,有单调改进保证。
- F21. 🟡 **参数共享 + agent-id 嵌入**:兼顾共享与异构轨道差异化。
- F22. 🟡 **执行期中心协调器(放松纯 CTDE)**:若部署允许地面站统一调度,可作对比"上界"方案。

### G. 分层 / 混合优化(解耦"分配"与"排序")
- G23. 🔴 **两级架构**:高层任务分配器(区域/任务簇指派)+ 低层单星调度器。
- G24. 🔴 **优化 + RL 混合**:ILP/拍卖做星间指派(全局最优),RL 只负责单星内时序排程。
- G25. 🟡 **基于轨道的地理分区先验**:按纬度带/经度扇区预分责任区作 mask 先验/warm-start,再 RL 微调。

### H. 领域知识 / 前瞻调度
- H26. 🟡 **前瞻让位**:若另一颗星稍后能以更小 off-nadir 看同一任务,当前星不抢,用轨道传播预判。
- H27. 🟡 **互补时间窗协同**:协调各星覆盖同一热点的不同过境窗,扩大有效覆盖时长。
- H28. 🟡 **为动态任务预留 capacity**:别用常规任务贪婪填满,留机动余量给突发动态任务(动态响应是已领先点,可放大)。
- H29. 🔴 **预测式定位/重规划**:动态任务到达时增量重规划。

### I. 训练层面
- I30. 🟢 **卫星数量课程**:从 1 星逐步加到 6 星训练,协同策略更易收敛。
- I31. 🟡 **联合探索**:加联合熵正则/相关探索噪声,避免各星独立探索学不到配合。
- I32. 🟢 **每星奖励归一化**:异构轨道奖励量纲不同,归一化后 critic 更稳。

### J. 评估与上界(量化"还有多大空间")
- J33. 🟢 **加 ILP/贪心 Oracle 上界**:离线最优调度器作天花板,量化 MAPPO 与上界差距。
- J34. 🟢 **修正 coordination_gain 定义**:分母改成"相对最优分配的比值",避免任务可行性重叠天然压低。

---

## 三、推荐落地顺序

1. **第一梯度(快赢,纯环境层为主)**:A1 + A2/A3 + B6
   修复"负载更差、gain<1"两个硬伤,性价比最高。
2. **第二梯度(协同进奖励信号)**:C8/C11 + H28
   让协同真正进奖励,放大已有优势。
3. **第三梯度(论文亮点/研究级)**:C9(COMA)、E17/E18(通信)、G24(优化+RL 混合)、J33(Oracle 上界)
   最能写进论文当卖点。

---

## 四、进度跟踪

| 方案 | 状态 | 备注 |
|---|---|---|
| 实验 manifest | ✅ 已实现(2026-06-22) | `compare_methods.py` 每次输出 `manifest.json` |
| 批量消融 runner | ✅ 已实现(2026-06-22) | `run_ablation.py --preset assignment_v2` |
| 结果目录隔离 | ✅ 已实现(2026-06-22) | compare/ablation/train 默认自动创建唯一输出目录 |
| 可视化与可观测任务统计 | ✅ 已实现(2026-06-22) | 输出 `*_viz_data.json`,支持任务分布图/调度甘特图/可观测任务数 |
| 外循环编码器消融 | ✅ 已实现(2026-06-23) | `meta_encoder_v1`: LSTM/GRU/MLP/Transformer/Set Transformer + MAPPO-LSTM |
| 学习式任务分配器 | ✅ 已实现(2026-06-23) | `learned_assignment_v1`: heuristic/MLP/LSTM/GRU/Transformer/Set Transformer/GNN scorer |
| 滚动重分配 | ✅ 已实现(2026-06-23) | `assignment_rolling_v1`: static/periodic/event/2h rolling horizon |
| 层级任务分配接口 | ✅ 已实现(2026-06-23) | `hier_assignment_v1`: RuleBasedAssignmentManager + 低层 MAPPO |
| A1 败者改派 | ✅ 已实现(评估期) | `_resolve_actions` + `eval_mode`;训练期关闭以保信用分配 |
| A2/A3 择优指派 | ✅ 已实现 | 边际价值竞价(优先级+off-nadir 质量),胜者得 |
| B6 负载均衡 tie-break | ✅ 已实现 | 竞价含负载惩罚 `coord_w_load` |

### 学习式任务分配器路线(2026-06-23,learned_assignment)

**动机**:当前 episode 级任务指派 `_assign_tasks()` 依赖手写启发式分数 `quality - assign_w_load * load_pressure`。它稳定、可解释,但只做局部贪心,无法学习不同轨道重叠、任务密度、动态任务预留和负载权衡之间的复杂关系。

**递进实现顺序**:
1. **L0 接口化 + MLP scorer(已实现)**:保留当前所有硬约束、候选可见性、所有权掩码和贪心/拍卖解码,只把候选边 `(satellite, task)` 的分数抽象为 `assignment_scorer`。支持 `heuristic` 与 `mlp`,并用 `assignment_scorer_mix` 控制 MLP 分数与旧启发式的混合比例。
2. **L1 序列分配器消融(已实现)**:在同一接口下加入 `lstm/gru` scorer,把按候选数/任务约束排序后的任务序列作为输入,验证“任务序列记忆”是否优于无记忆 MLP。
3. **L2 集合/注意力分配器(已实现)**:加入 `transformer/set_transformer` scorer,把任务集作为集合建模,减少 LSTM 对人工排序的依赖。
4. **L3 图匹配分配器(已实现第一版)**:把卫星-任务可见关系建成二分图,用轻量 GNN 消息传递输出边上下文,再用 greedy 解码,保持硬约束不被神经网络破坏。后续可把当前 deterministic GNN scorer 替换为监督/强化训练得到的参数。

**消融指标**:重点看 `n_scheduled`、`n_feasible_tasks`、`duplicate_rate`、`load_balance_cv`、`avg_dynamic_response_s`、`avg_off_nadir_deg` 和 oracle gap。

### 任务分配优化后续头脑风暴(2026-06-23)

**当前结论**:任务分配的第一阶段已经完成,包括 episode 级全局指派、容量比例负载均衡、动态任务增量指派、以及 `heuristic/mlp/lstm/gru/transformer/set_transformer/gnn` 多种 scorer 消融接口。该阶段的核心价值是把原来的手写局部贪心拆成可替换模块,并提供稳定的实验记录。但从研究角度看,它仍主要是"固定参数 scorer + greedy/拍卖解码",还没有进入真正的可训练分配策略或强优化解码阶段。

| 方向 | 难度 | 研究价值 | 与现有代码关系 | 建议版本名 |
|---|---:|---:|---|---|
| 更强解码器: Hungarian / min-cost flow / auction / ILP 小规模 oracle | 中 | 高 | 保留现有 scorer,替换或并列比较 `_assign_tasks()` 的 greedy 解码;可直接衡量 greedy gap | `assignment_decode_v1` |
| 监督学习 scorer: 用 oracle/启发式最优标签训练 MLP/GNN/Transformer | 中 | 高 | 复用当前 scorer 接口,新增离线数据生成和训练脚本;先 imitation,再接 RL | `assignment_supervised_v1` |
| Rolling horizon / MPC 动态重分配 | 中高 | 高 | 当前是 episode 初始分配 + 动态任务增量分配;可改成每隔 K 步重估未完成任务所有权 | `assignment_rolling_v1` |
| 动态任务容量预留 | 低中 | 中高 | 在 `_assignment_targets()` 中为未来动态任务保留卫星容量,重点优化 `avg_dynamic_response_s` | `assignment_reserve_v1` |
| Differentiable assignment: Sinkhorn / Gumbel-Sinkhorn | 高 | 高 | 把边分数转成软匹配矩阵,用于端到端训练;硬约束仍需后处理 | `assignment_sinkhorn_v1` |
| 高层策略 + 低层 MAPPO 联合训练 | 高 | 很高 | 分配器作为 high-level policy,每颗卫星 PPO/MAPPO 作为 low-level scheduler | `assignment_hier_marl_v1` |
| 后处理局部修复: swap/2-opt/late rescue | 低 | 中 | 在 greedy 指派后交换少量边,修复负载不均、临近 deadline 和低质量观测 | `assignment_repair_v1` |
| 不确定性鲁棒分配 | 中高 | 中 | 对天气/姿态机动/任务取消等扰动做保守分配或重分配 | `assignment_robust_v1` |

**推荐递进顺序**:
1. **先做 `assignment_repair_v1` 或 `assignment_reserve_v1`**:改动小,能直接验证是否改善负载、动态响应和临近截止任务。
2. **再做 `assignment_decode_v1`**:在同一 scorer 下比较 greedy、Hungarian/min-cost flow 和小规模 ILP/oracle,回答"当前瓶颈是打分还是解码"。
3. **随后做 `assignment_supervised_v1`**:用 decode/oracle 生成标签,训练 MLP/GNN/Transformer scorer,把当前 deterministic scorer 推进为可学习分配器。
4. **最后推进 `assignment_rolling_v1` 与 `assignment_hier_marl_v1`**:面向动态任务和协同决策,研究价值最高,但也最容易影响训练稳定性。

**版本与消融约定**:后续每个任务分配优化都必须提供独立 CLI 开关、`run_ablation.py` preset、唯一输出目录、`manifest.json` 记录和 README 命令。新增策略默认不能覆盖当前 `learned_assignment_v1` 结果,应以 `heuristic + greedy`、`best learned scorer + greedy`、`best learned scorer + new decoder` 三组作为最小对照。

### 滚动重分配 + 层级 MAPPO 深化方案(2026-06-23)

**核心判断**:滚动重分配和"高层分配策略 + 低层 MAPPO"不是两条互斥路线,而是同一条递进路线。先做规则/MPC 版滚动重分配,可以暴露稳定的环境 API、日志指标和对照组;之后再把这个高层控制器替换为可训练策略。这样能避免一开始就端到端训练导致难以定位收益来源。

#### A. Rolling horizon / MPC 重分配路线

**要解决的问题**:当前 `task_owner` 在 episode 初始确定,动态任务只做增量指派,临近 deadline 或 owner 无未来窗口时才释放。这比静态指派可靠,但仍可能出现三类失效:① owner 因策略行为错过早期窗口;② 动态任务到达后改变全局负载和稀缺窗口;③ 初始贪心分配没有考虑未来一段时间的真实执行状态。

| 阶段 | 机制 | 难度 | 预期收益 | 主要风险 |
|---|---|---:|---|---|
| R0 诊断版 | 只记录 owner 失效、deadline rescue、owner backlog、任务被释放次数 | 低 | 判断重分配是否有足够空间 | 无直接性能收益 |
| R1 周期重分配 | 每隔 `K` 秒对未完成任务重算 owner,加入 switch penalty 和 lock window 防抖 | 中 | 降低 stale owner,改善动态响应和吞吐 | mask 非平稳,训练波动 |
| R2 事件触发重分配 | 动态任务到达、owner 无未来窗口、负载严重不均、deadline 风险时触发 | 中 | 比固定周期更少扰动,更像任务调度系统 | 触发阈值需要消融 |
| R3 MPC 窗口重分配 | 只看未来 `H` 秒可执行窗口,优化优先级、质量、slack、负载和切换成本 | 中高 | 更贴近动态调度,可解释性强 | 计算量随任务/卫星增长 |
| R4 学习式重分配门控 | 学习"保持 owner / 交换 owner / 释放 owner"的决策 | 高 | 为层级策略过渡 | 需要标签或稳定奖励 |

**推荐实现接口**:
- `--assignment_replan_interval_s`:周期重分配间隔;0 表示关闭。
- `--assignment_replan_horizon_s`:MPC 只考虑未来窗口长度;0 表示看完整剩余 horizon。
- `--assignment_replan_trigger`:可选 `periodic,dynamic,stale_owner,deadline,imbalance`。
- `--assignment_switch_penalty`:切换 owner 的惩罚,抑制频繁改派。
- `--assignment_lock_window_s`:任务在下一可行窗口前多少秒锁定 owner,避免临门换人。
- `--assignment_max_switches_per_task`:每个任务最多改派次数。

**建议代码落点**:
- `MultiSatelliteEnv.step()` 在动态任务同步之后、构建下一步 mask 之前调用 `_maybe_reassign_tasks()`。
- 新增 `_eligible_replan_missions()` 过滤已完成、锁定、不可达和切换次数超限任务。
- 复用 `_assign_tasks()` 的 scorer,但在重分配时加入当前时间、未来窗口、旧 owner switch penalty 和 deadline pressure。
- `get_metrics()` 增加 `n_replans`、`n_owner_switches`、`owner_churn_rate`、`stale_owner_rate`、`deadline_rescue_rate`、`rescued_tasks`。
- `run_ablation.py` 新增 `assignment_rolling_v1` preset:静态 owner、周期重分配、事件触发、MPC horizon 四组对照。

**最小消融矩阵**:
1. `static_assignment`:当前默认 `assignment_v2/learned_assignment_v1`。
2. `rolling_periodic_1h`:每 3600s 重分配,带 switch penalty。
3. `rolling_event`:只在动态到达、owner 无未来窗口、deadline 风险时重分配。
4. `rolling_mpc_2h`:未来 7200s 窗口 MPC,强调动态任务响应。

#### B. 高层分配策略 + 低层 MAPPO 路线

**基本框架**:把系统拆成两个时间尺度。高层 Assignment Manager 每隔若干步输出任务所有权、容量配额或局部重分配动作;低层 MAPPO 在高层给定的 owner mask/option 下选择具体观测动作。低层解决"当前窗口怎么排",高层解决"谁负责哪些任务和何时重分配"。

| 阶段 | 高层动作 | 低层策略 | 训练方式 | 研究价值 |
|---|---|---|---|---|
| H0 规则 manager | 使用 R1/R2/R3 的滚动重分配规则 | 当前 MAPPO | 不训练高层 | 建立层级接口和强基线 |
| H1 监督 manager | 预测 task-owner 边分数或 owner switch | 当前/冻结 MAPPO | 模仿 MPC/oracle 标签 | 证明学习式高层能复现强规则 |
| H2 Bandit manager | 每个重分配 epoch 输出少量 owner switch 或容量 quota | 冻结 MAPPO | PPO/REINFORCE,奖励为下个区间团队收益 | 降低动作空间和训练难度 |
| H3 Hierarchical PPO + MAPPO | 高层集中式策略输出 task-owner 矩阵/图匹配;低层 MAPPO 调度 | 先冻结后交替训练 | 双 buffer,高层用区间奖励,低层用步级奖励 | 最完整的层级 MARL 贡献 |

**高层状态设计**:
- 卫星侧:当前负载、未来 `H` 秒可见任务数、平均质量、动态任务 backlog、上次完成时间。
- 任务侧:priority、duration、slack、dynamic flag、候选卫星数、当前 owner、已切换次数、最近释放状态。
- 边特征:未来窗口最早开始时间、最佳 off-nadir、窗口数量、是否旧 owner、switch cost。
- 全局统计:完成率、动态响应均值、load CV、重复率、owner churn。

**高层动作设计优先级**:
1. **最稳妥**:输出每颗卫星容量 quota 或负载目标,再由现有 `_assign_tasks()` 解码。
2. **中等动作空间**:输出 top-K owner switch,每次只改少量最危险任务。
3. **最强但最难**:输出完整 task-owner 图匹配,需要 Hungarian/min-cost flow/Sinkhorn 等解码。

**训练建议**:
1. 先冻结低层 MAPPO,用 R3/MPC 生成 `(state, assignment)` 标签训练高层 GNN/Transformer manager。
2. 再用高层 PPO 做区间奖励微调,奖励定义为 `Δcompleted + dynamic_bonus + quality_bonus - churn_penalty - imbalance_penalty`。
3. 最后交替训练:固定高层跑若干低层 MAPPO update,再固定低层跑若干高层 update。避免两个策略同时漂移。

**需要新增模块**:
- `models/assignment_manager.py`:GNN/Transformer 高层策略,输入卫星-任务二分图,输出边 logits 或 switch logits。
- `algo/hier_mappo_trainer.py`:高层 rollout buffer + 区间奖励 PPO 更新。
- `envs/multi_satellite_env.py`:暴露 `get_assignment_state()`、`set_task_owner()`、`replan_assignment()`。
- `run_ablation.py`:新增 `assignment_rolling_v1` 和 `hier_assignment_v1` preset。

**建议执行顺序**:
1. **先实现 `assignment_rolling_v1` 的 R0/R1/R2**:改动小,能快速判断重分配是否值得。
2. **补 R3 MPC**:作为强规则上界,同时给监督高层 manager 产标签。
3. **实现 H0/H1**:先只训练高层 imitation,低层 MAPPO 不动。
4. **实现 H2/H3**:再进入真正的高层策略 + 低层 MAPPO 联合训练。

**关键评估指标**:
- 性能: `n_scheduled`、`dynamic_completion_rate_raw`、`avg_dynamic_response_s`、`avg_off_nadir_deg`、`oracle_relative_completion`。
- 稳定性: `owner_churn_rate`、`n_owner_switches`、`stale_owner_rate`、`deadline_rescue_rate`。
- 训练:MAPPO reward 方差、高层 entropy、高层 value loss、低层 invalid/idle 比例。
- 消融结论必须区分三件事:重分配时机收益、解码器收益、高层学习收益。

#### assignment_rolling_v1 落地记录(2026-06-23)

**本次实现范围**:完成 R0/R1/R2,并提供 R3 的 rolling horizon 参数接口。默认配置保持关闭,不影响 `assignment_v2` 与 `learned_assignment_v1` 旧结果;显式打开后在每步动态任务同步之后、下一步 mask 构建之前调用 `_maybe_reassign_tasks()`。

**代码改动**:
- `MultiSatelliteEnv` 新增 `assignment_replan_interval_s`、`assignment_replan_horizon_s`、`assignment_replan_trigger`、`assignment_switch_penalty`、`assignment_lock_window_s`、`assignment_max_switches_per_task`。
- 新增 `_maybe_reassign_tasks()`、`_eligible_replan_missions()`、`_reassign_tasks()`、`_task_quality_window()` 等方法,支持周期、动态任务到达、owner 失效、deadline 风险和负载不均触发。
- `get_metrics()` 新增 `n_replans`、`n_owner_switches`、`n_tasks_switched`、`owner_churn_rate`、`stale_owner_rate`、`deadline_rescue_rate`、`n_rescued_tasks` 等诊断指标。
- `compare_methods.py` 暴露 rolling CLI 参数并写入 manifest;控制台摘要显示 rolling 指标。
- `run_ablation.py` 新增 `assignment_rolling_v1` preset,包含 `rolling_static`、`rolling_periodic_1h`、`rolling_event`、`rolling_mpc_2h` 四组。

**推荐命令**:
```bash
python run_ablation.py \
  --python /Users/zhouzidie/miniconda3/envs/myenv/bin/python \
  --preset assignment_rolling_v1 \
  --n_satellites 6 --train_iters 30 --eval_episodes 5 \
  --n_routine 200 --n_dynamic 50 \
  --out_root runs/ablation_assignment_rolling_v1 \
  --device cpu
```

**下一步**:若 `rolling_mpc_2h` 能降低 stale owner 或动态响应延迟,再实现 H0/H1:提取 `get_assignment_state()` 图状态,用 rolling/MPC 结果训练监督式 Assignment Manager;否则优先回到 `assignment_decode_v1` 或 `assignment_reserve_v1`。

#### hier_assignment_v1 H0 落地记录(2026-06-23)

**本次实现范围**:完成层级任务分配的 H0 接口版。高层 manager 暂时采用规则策略,低层仍是当前 MAPPO actor/critic;目标是先把高层状态、owner 建议、硬约束校验和消融入口稳定下来。

**代码改动**:
- 新增 `models/assignment_manager.py`,提供 `RuleBasedAssignmentManager.select_owners(assignment_state)`。
- `MultiSatelliteEnv` 新增 `assignment_manager_mode=none/rule`。
- `MultiSatelliteEnv.get_assignment_state()` 导出框架无关的卫星-任务图状态:卫星负载/目标、任务 slack/dynamic/owner stale、候选边 quality/load pressure/score。
- `MultiSatelliteEnv.set_task_owner()` 与 `replan_assignment()` 作为未来可训练高层策略的稳定环境 API。
- `_reassign_tasks()` 优先采用 manager proposal,再回退到原 scorer,并继续由环境校验可见性、切换次数和锁定窗口。
- `run_ablation.py --preset hier_assignment_v1` 对比 `hier_no_manager` 与 `hier_rule_manager`。

**推荐命令**:
```bash
python run_ablation.py \
  --python /Users/zhouzidie/miniconda3/envs/myenv/bin/python \
  --preset hier_assignment_v1 \
  --n_satellites 6 --train_iters 30 --eval_episodes 5 \
  --n_routine 200 --n_dynamic 50 \
  --out_root runs/ablation_hier_assignment_v1 \
  --device cpu
```

**下一步**:实现 H1 supervised manager。用 `rolling_mpc_2h` 或 Greedy-Oracle 生成 `(assignment_state, owner)` 标签,训练 GNN/Transformer manager 替换 `RuleBasedAssignmentManager`,并保留 `hier_no_manager/rule_manager/supervised_manager` 三组对照。

### 实验框架落地(2026-06-22,v2_experiment_harness)

**目标**:后续会逐步加入 reward/state/communication 等多个优化簇,如果只保存单个 `comparison_results.json`,很难追踪每个结果对应的代码版本、参数组合和运行环境。因此先把实验记录标准化。

**实现**:
- `compare_methods.py` 新增 `manifest.json`,记录 `args`、git commit/branch/dirty 状态、Python/NumPy/PyTorch 版本、输出路径和完整 results。
- `compare_methods.py` 新增 `--experiment_tag`,供批量实验给每个 run 命名。
- 新增 `run_ablation.py`,默认 `assignment_v2` preset 会运行 no-assignment baseline,以及 `assignment_capacity_mode × assign_w_load × release_before_deadline_s` 的网格消融。
- `run_ablation.py` 每完成一个子实验就增量写出 `ablation_summary.json/csv`,避免长实验中断后丢失已完成结果。

**推荐正式运行**:
```bash
python run_ablation.py \
  --python /Users/zhouzidie/miniconda3/envs/myenv/bin/python \
  --preset assignment_v2 \
  --n_satellites 6 --train_iters 30 --eval_episodes 5 \
  --n_routine 200 --n_dynamic 50 \
  --out_root runs/ablation_assignment_v2 \
  --device cpu
```

**后续所有优化版本约定**:每个新方案必须提供开关,并进入 `run_ablation.py` 或新增 preset 做消融;路线图记录方案、默认值、对照组、结果路径和结论。

**结果目录约定**:
- `train.py` 未指定 `--exp_name` 时自动写入 `runs/<method>[_tag][_fast]_<timestamp>/`;显式 `--exp_name` 可加 `--append_timestamp` 避免覆盖。
- `compare_methods.py` 默认在 `--out_dir` 下创建唯一子目录;需要旧行为时使用 `--flat_out_dir`。
- `run_ablation.py` 默认在 `--out_root` 下创建唯一批次目录,子实验写入 `<batch>/<tag>/`;需要旧行为时使用 `--flat_out_root`。

**本地验证**:
- 语法检查: `PYTHONPYCACHEPREFIX=/private/tmp/mrl_dms_pycache python3 -m compileall compare_methods.py run_ablation.py`。
- dry-run: `python3 run_ablation.py --dry_run --train_iters 0 --eval_episodes 1 --n_routine 8 --n_dynamic 1 --assign_w_loads 0.1 --release_windows 0 --capacity_modes proportional --no_baseline --out_root runs/ablation_dry_run`。
- 冒烟运行: `/Users/zhouzidie/miniconda3/envs/myenv/bin/python run_ablation.py --python /Users/zhouzidie/miniconda3/envs/myenv/bin/python --train_iters 0 --eval_episodes 1 --n_satellites 2 --n_routine 8 --n_dynamic 1 --assign_w_loads 0.1 --release_windows 0 --capacity_modes proportional --no_baseline --out_root runs/ablation_harness_smoke --device cpu`。
- 验证输出: `runs/ablation_harness_smoke/assign_proportional_w0p1_rel0/manifest.json` 和 `runs/ablation_harness_smoke/ablation_summary.csv/json`。

### 第一梯度落地复盘(2026-06-21)

**已完成**:用「负载感知贪心拍卖 + 败者改派」替换了原 `multi_satellite_env.py` 的「先到先得+败者强制 idle」。改动全部封装在 `_resolve_actions` / `_obs_value` / `_next_best_action` 三个方法,不动 trainer/eval 接口。合成单元测试验证逻辑正确(改派、训练期信用分配、无冲突通过)。

**关键发现(重要)**:在当前 ACLED + SSO 近极轨场景下,这套**逐时刻冲突解决机制杠杆极低**:
- 诊断显示 6 星 × 1487 步中只有 238 次非 idle 决策——**97% 的 agent-step 每颗星根本没有可行任务**(轨道过境窗短、稀疏)。
- "同一时刻多星抢多任务"几乎不发生,冲突基本是"N 星抢同 1 个任务",**A1 没有次优任务可派、B6 没有多任务可分摊**。
- 在同一固定策略上对比新旧解析,结果**完全一致**(n_scheduled、load CV 不变)→ 机制正确但处于休眠态。

**对"负载不均/gain<1"的重新认识**:
- 这两个问题**主要是结构性的**(各星轨道对 ACLED 热点的覆盖天然不均),不是逐时刻 tie-break 能解决的——空闲星根本看不到那些任务。
- 无协同 baseline 的 load CV 更低(0.21)其实是**假象**:它靠"每颗星都重复观测同样的简单高纬任务"凑出均衡的计数,代价是 43% 重复率。去掉重复后真实的轨道不均才显现。

**结论与建议(下一步)**:第一梯度作为对"先到先得"的正确替代保留(无害、密度变大时自动生效),但**高杠杆点不在逐时刻冲突**,而在**全局/episode 级任务指派**——综合每颗星在整个 24h 内对各任务的窗口质量,提前决定"哪颗星负责哪个任务",从根上做负载均衡。这更接近 **A3 的 episode 级版本**或 **G24(优化+RL 混合)**。建议把它作为优化后续的重点。

---

### 全局 episode 级任务指派落地(2026-06-21,A3-episode / G24)

| 方案 | 状态 | 备注 |
|---|---|---|
| 全局任务指派 | ✅ 已实现 | `_assign_tasks` + `_apply_ownership_mask` + `task_owner` |
| 动态任务增量指派 | ✅ 已实现 | 动态任务到达后在当前负载基础上继续均衡 |
| 负载/吞吐权衡旋钮 | ✅ 已暴露 | `assign_w_load` + compare CLI `--assign_w_load/--no_episode_assignment` |
| 容量比例指派 | ✅ 已实现(2026-06-22) | `assignment_capacity_mode=proportional`;按候选窗口质量估算各星目标份额 |
| 截止前所有权释放 | ✅ 已实现(2026-06-22) | `release_before_deadline_s`;临近 deadline 或 owner 无未来窗口时允许非 owner 接手 |
| C8 团队奖励混合 | ✅ 已实现(2026-06-22) | `team_reward_mix`;默认 0 关闭 |
| B5 负载均衡奖励 | ✅ 已实现(2026-06-22) | `load_balance_reward_coeff`;低负载星完成任务获 bonus |
| I32 每星奖励归一化 | ✅ 已实现(2026-06-22) | `normalize_agent_rewards`;MAPPO 更新前按 agent rollout 归一化 |
| reward_v1 消融 | ✅ 已实现(2026-06-22) | `run_ablation.py --preset reward_v1` |
| D14 拼接式全局状态 | ✅ 已实现(2026-06-22) | `global_state_mode=concat`;critic 看完整各星观测 |
| D16 任务级全局统计 | ✅ 已实现(2026-06-22) | `global_state_task_stats`;追加任务/负载/重复率统计 |
| state_v1 消融 | ✅ 已实现(2026-06-22) | `run_ablation.py --preset state_v1` |
| J33 Greedy-Oracle 参考 | ✅ 已实现(2026-06-22) | `--run_oracle`;集中式启发式参考上界 |
| J34 Oracle 相对增益 | ✅ 已实现(2026-06-22) | `oracle_relative_completion` + `mappo_oracle_gap_n_scheduled` |
| I30 卫星数量课程 | ✅ 已实现(2026-06-22) | `satellite_curriculum`;训练期活跃卫星数线性增加 |
| I31 轻量联合探索 | ✅ 已实现(2026-06-22) | `joint_explore_prob`;训练期随机选择互不重复可行动作 |
| train_stability_v1 消融 | ✅ 已实现(2026-06-22) | `run_ablation.py --preset train_stability_v1` |
| E17 意图广播(规则版) | ✅ 已实现(2026-06-22) | `intent_broadcast`;冲突败者基于广播意图重采样 |
| communication_v1 消融 | ✅ 已实现(2026-06-22) | `run_ablation.py --preset communication_v1` |

**实现**:`reset()` 时综合每颗星对每个任务在全 24h 的窗口质量(最小 off-nadir),用「**最少候选优先 + 负载惩罚**」贪心广义指派算出 `task_owner`(每任务归属一颗星);通过**所有权掩码**让各星只在自己负责的任务上行动。动态任务到达时增量指派。仅 `coordinate=True` 生效;训练/评估都套用所有权掩码(纯掩码,不破坏信用分配)。

**固定随机策略诊断(6 星, 300+60 任务, 隔离训练噪声)**:

| 配置 | 完成数 | 成功率 | 负载CV | 重复率 | off-nadir |
|---|---|---|---|---|---|
| 无协同 baseline | 164 | 68.3% | 0.25 | 27.8% | 15.34 |
| 协同(无全局指派) | 166 | 69.2% | 0.42 | 0.0% | 14.35 |
| 协同+全局指派 w=0.02 | 125 | 52.1% | **0.09** | 0.0% | **12.62** |

**冒烟测试(3 星, 3 训练迭代)**:MAPPO 负载CV **0.15**(< Indep 0.21,原为 0.46)、重复率 **0%**、off-nadir **13.88**(最优)、协同增益 **0.93**(与 Indep 持平,吞吐未掉)。**最初"MAPPO 负载比无协同还差"的核心问题已修复。**

**本质权衡(重要)**:硬所有权下存在**负载均衡 vs 吞吐**的不可约权衡——覆盖好的卫星被限额后,只有它能看到的任务可能无人完成。随机策略下吞吐损失明显(~22%),但**(哪怕极少量)训练后即可大幅恢复**(冒烟中吞吐与 Indep 持平)。`assign_w_load` 把这个权衡变成可调的帕累托曲线:越大越均衡、吞吐越低。

**待办/可改进**:① 在服务器上跑 `--train_iters 30+` 的正式对比,确认训练后吞吐恢复程度并调 `assign_w_load`;② 加 ILP/贪心 Oracle 上界,量化当前 MAPPO 与离线最优差距;③ 探索团队奖励/互补覆盖奖励,让协同从掩码进入奖励信号。

---

### 全局指派 v2: 容量比例 + 截止释放(2026-06-22,A3-episode / G24)

**目标**:修复硬所有权的两个副作用:一是等额负载惩罚会把任务压给覆盖能力弱的卫星,导致吞吐下降;二是 owner 错过窗口后,其他卫星即使当前可行也会被所有权掩码挡住。

**实现**:
- `_assignment_targets()` 新增 `assignment_capacity_mode`:默认 `proportional`,按每颗星对候选任务的质量和估算目标容量;保留 `equal` 作为消融对照。
- `_load_pressure()` 用"当前指派负载 / 目标容量"替代绝对指派数惩罚,使覆盖能力强的卫星可以承担更多任务。
- `_ownership_released()` 新增截止释放:任务进入 `release_before_deadline_s` 窗口,或 owner 已无未来可行窗口时,非 owner 可在自己的动作掩码中重新看到该任务。
- `_resolve_actions()` 现在也叠加所有权掩码,避免评估期 A1 败者改派绕开 `task_owner`。
- 动态任务增量指派前用 `_refresh_assignment_load()` 把"已实际完成数 + 未完成 owner backlog"作为负载基线,避免 release 后的真实负载偏移影响后续动态任务。

**新增对比参数**:
```bash
python compare_methods.py \
  --assignment_capacity_mode proportional \
  --release_before_deadline_s 1800 \
  --assign_w_load 0.1
```

**实验建议**:
- 指派消融: `--assignment_capacity_mode equal` vs `proportional`。
- 释放窗口消融: `--release_before_deadline_s 0/900/1800/3600`。
- 权衡曲线: `--assign_w_load 0.02/0.05/0.1/0.2`,同时看 `n_scheduled`、`load_balance_cv`、`duplicate_rate`、`avg_off_nadir_deg`。

**本地验证**:
- 语法检查: `PYTHONPYCACHEPREFIX=/private/tmp/mrl_dms_pycache python3 -m compileall envs compare_methods.py`。
- 冒烟运行: `/Users/zhouzidie/miniconda3/envs/myenv/bin/python compare_methods.py --n_satellites 2 --train_iters 0 --eval_episodes 1 --n_routine 12 --n_dynamic 2 --out_dir runs/compare_opt_smoke --device cpu`。
- 结果文件: `runs/compare_opt_smoke/comparison_results.json`。该冒烟仅验证代码路径;因 `train_iters=0` 且样本极小,不作为算法效果结论。

---

### 奖励塑形 v1(2026-06-22,B5 / C8 / I32)

**目标**:当前协同主要靠 mask/指派发生在环境层,策略本身仍以个体奖励学习。reward_v1 把协同信号显式放进训练奖励,但保持所有开关默认关闭,便于与原 MAPPO 做严格消融。

**实现**:
- C8 `team_reward_mix`: 将个体奖励与全队平均奖励混合,`0` 表示原个体奖励,`1` 表示完全团队平均奖励。为避免奖励尺度随卫星数线性放大,这里采用 team mean 而非 team sum。
- B5 `load_balance_reward_coeff`: 若某星在本步完成任务,根据执行前负载与全队平均负载的差给 bonus/penalty,鼓励相对空闲卫星承担任务。
- 团队完成 bonus `team_completion_bonus`: 本步每新增完成一个全局任务,给所有 agent 小额团队 bonus,作为轻量协作塑形信号。
- I32 `normalize_agent_rewards`: 在 MAPPO update 前对每颗星 rollout 奖励做标准化,缓解异构轨道导致的奖励尺度差异。
- `run_ablation.py --preset reward_v1`: 统一跑 default/team/load/completion/combined_norm 五个对照。

**推荐正式运行**:
```bash
python run_ablation.py \
  --python /Users/zhouzidie/miniconda3/envs/myenv/bin/python \
  --preset reward_v1 \
  --n_satellites 6 --train_iters 30 --eval_episodes 5 \
  --n_routine 200 --n_dynamic 50 \
  --out_root runs/ablation_reward_v1 \
  --device cpu
```

**注意**:`get_metrics()` 仍统计原环境完成奖励,不统计 shaped training reward;这样评估指标不被奖励塑形本身污染,只反映训练后策略行为变化。

**本地验证**:
- 语法检查: `PYTHONPYCACHEPREFIX=/private/tmp/mrl_dms_pycache python3 -m compileall envs algo compare_methods.py run_ablation.py`。
- dry-run: `python3 run_ablation.py --preset reward_v1 --dry_run --train_iters 0 --eval_episodes 1 --n_satellites 2 --n_routine 8 --n_dynamic 1 --out_root runs/ablation_reward_dry_run`。
- 冒烟运行: `/Users/zhouzidie/miniconda3/envs/myenv/bin/python run_ablation.py --python /Users/zhouzidie/miniconda3/envs/myenv/bin/python --preset reward_v1 --train_iters 0 --eval_episodes 1 --n_satellites 2 --n_routine 8 --n_dynamic 1 --out_root runs/ablation_reward_smoke --device cpu`。
- 验证输出: `runs/ablation_reward_smoke/ablation_summary.csv/json`。该冒烟仅验证 5 个 reward 组合链路,不作为效果结论。

**下一步**:C11 互补覆盖奖励和 C12 势能塑形建议独立成 reward_v2,因为它们对"覆盖互补"和"剩余可行任务"的定义会显著影响实验解释。

---

### Critic 全局状态 v1(2026-06-22,D14 / D16)

**目标**:旧 MAPPO critic 只看各星局部观测的 mean pooling,会丢失"哪颗星拥有哪些窗口/负载/任务归属"的信息。state_v1 先增强 centralized critic,不改 actor 执行期输入,保持 CTDE 部署假设。

**实现**:
- D14 `global_state_mode=concat`: critic 输入从 mean pooling 改为拼接所有卫星局部观测,信息无损但维度随卫星数增长。
- D16 `global_state_task_stats`: 在 mean/concat 后追加任务级统计,包括 per-agent load fraction、已完成比例、待完成比例、动态待完成比例、已指派待完成比例、load CV、当前重复率。
- `compare_methods.py` 使用 `env.global_state_dim` 初始化 critic,支持不同全局状态维度。
- `run_ablation.py --preset state_v1`: 统一比较 `mean`、`mean_task_stats`、`concat`、`concat_task_stats`。

**推荐正式运行**:
```bash
python run_ablation.py \
  --python /Users/zhouzidie/miniconda3/envs/myenv/bin/python \
  --preset state_v1 \
  --n_satellites 6 --train_iters 30 --eval_episodes 5 \
  --n_routine 200 --n_dynamic 50 \
  --out_root runs/ablation_state_v1 \
  --device cpu
```

**注意**:concat 维度约为 `n_satellites × local_obs_dim`,会明显增加 critic 参数量和训练耗时;若正式实验中收益有限,后续优先发展 attention critic(D13)而不是继续堆 concat 维度。

**本地验证**:
- 语法检查: `PYTHONPYCACHEPREFIX=/private/tmp/mrl_dms_pycache python3 -m compileall envs compare_methods.py run_ablation.py`。
- dry-run: `python3 run_ablation.py --preset state_v1 --dry_run --train_iters 0 --eval_episodes 1 --n_satellites 2 --n_routine 8 --n_dynamic 1 --out_root runs/ablation_state_dry_run`。
- 冒烟运行: `/Users/zhouzidie/miniconda3/envs/myenv/bin/python run_ablation.py --python /Users/zhouzidie/miniconda3/envs/myenv/bin/python --preset state_v1 --train_iters 0 --eval_episodes 1 --n_satellites 2 --n_routine 8 --n_dynamic 1 --out_root runs/ablation_state_smoke --device cpu`。
- 验证输出: `runs/ablation_state_smoke/ablation_summary.csv/json`。该冒烟仅验证 4 个 critic 状态组合链路,不作为效果结论。

---

### Greedy-Oracle 参考上界 v1(2026-06-22,J33 / J34)

**目标**:原 `coordination_gain = MAPPO / (N × Single-PPO)` 会受任务可见性重叠影响,不一定反映“离当前场景可达上界还有多远”。oracle_v1 加入集中式启发式参考,用于量化 MAPPO 相对强调度器的差距。

**实现**:
- `compare_methods.py --run_oracle` 新增 `Greedy-Oracle` 方法,不训练策略,在同一固定测试集上运行。
- Oracle 每一步读取所有卫星当前动作掩码,按 `priority + quality + urgency + dynamic_bonus - load_penalty` 给候选动作打分,集中式贪心选择非冲突的星-任务匹配。
- 输出 `oracle_relative_completion = method_n_scheduled / oracle_n_scheduled`。
- `run_ablation.py --preset oracle_v1` 比较 `no_episode_assignment` 与默认 `assignment_v2`,并同时输出 oracle gap。
- `run_ablation.py --run_oracle` 可给任意 preset 追加 Oracle 参考。

**推荐正式运行**:
```bash
python run_ablation.py \
  --python /Users/zhouzidie/miniconda3/envs/myenv/bin/python \
  --preset oracle_v1 \
  --n_satellites 6 --train_iters 30 --eval_episodes 5 \
  --n_routine 200 --n_dynamic 50 \
  --out_root runs/ablation_oracle_v1 \
  --device cpu
```

**注意**:这是 Greedy 启发式参考,不是严格 ILP 最优。若论文需要“数学意义上的上界”,下一版应实现 ILP/最大权匹配滚动规划,或至少把 Greedy 明确称为 `heuristic upper reference`。

**本地验证**:
- 语法检查: `PYTHONPYCACHEPREFIX=/private/tmp/mrl_dms_pycache python3 -m compileall compare_methods.py run_ablation.py`。
- 回归 dry-run: `python3 run_ablation.py --preset reward_v1 --dry_run --train_iters 0 --eval_episodes 1 --n_satellites 2 --n_routine 8 --n_dynamic 1 --out_root runs/ablation_reward_dry_run`。
- oracle dry-run: `python3 run_ablation.py --preset oracle_v1 --dry_run --train_iters 0 --eval_episodes 1 --n_satellites 2 --n_routine 8 --n_dynamic 1 --out_root runs/ablation_oracle_dry_run`。
- 冒烟运行: `/Users/zhouzidie/miniconda3/envs/myenv/bin/python run_ablation.py --python /Users/zhouzidie/miniconda3/envs/myenv/bin/python --preset oracle_v1 --train_iters 0 --eval_episodes 1 --n_satellites 2 --n_routine 8 --n_dynamic 1 --out_root runs/ablation_oracle_smoke --device cpu`。
- 验证输出: `runs/ablation_oracle_smoke/ablation_summary.csv/json`。该冒烟仅验证 Greedy-Oracle 与 oracle-relative 指标链路,不作为效果结论。

---

### 训练稳定性 v1(2026-06-22,I30 / I31)

**目标**:前面已加入指派、奖励、critic 状态等优化,但多星协同策略可能仍因早期探索空间过大而收敛慢。train_stability_v1 提供两个训练层面的低侵入改动,帮助策略先学简单协同再逐步扩展。

**实现**:
- I30 `satellite_curriculum`: 训练期只让前 `k` 颗卫星参与 rollout/update,`k` 从 `curriculum_min_satellites` 在 `curriculum_iters` 个训练迭代内线性增加到全部卫星。评估期始终使用全部卫星。
- I31 `joint_explore_prob`: 训练 rollout 中以给定概率执行轻量联合探索,集中式随机选择互不重复的可行动作,减少多星同时撞同一任务的探索样本。
- `MAPPOTrainer.collect_rollout()` 新增 `active_agent_ids` 和 `joint_explore_prob`;默认关闭,保持原训练行为。
- `run_ablation.py --preset train_stability_v1`: 比较 default/curriculum/joint_explore/combined 四组。

**推荐正式运行**:
```bash
python run_ablation.py \
  --python /Users/zhouzidie/miniconda3/envs/myenv/bin/python \
  --preset train_stability_v1 \
  --n_satellites 6 --train_iters 30 --eval_episodes 5 \
  --n_routine 200 --n_dynamic 50 \
  --out_root runs/ablation_train_stability_v1 \
  --device cpu
```

**注意**:联合探索会让部分训练动作不是直接从当前 policy sample 出来,因此只作为轻量探索消融,不应默认开启。若正式结果显示收益明显,后续应实现更严格的 correlated policy / shared latent exploration。

**本地验证**:
- 语法检查: `PYTHONPYCACHEPREFIX=/private/tmp/mrl_dms_pycache python3 -m compileall algo compare_methods.py run_ablation.py`。
- state 回归 dry-run: `python3 run_ablation.py --preset state_v1 --dry_run --train_iters 0 --eval_episodes 1 --n_satellites 2 --n_routine 8 --n_dynamic 1 --out_root runs/ablation_state_dry_run`。
- train stability dry-run: `python3 run_ablation.py --preset train_stability_v1 --dry_run --train_iters 1 --eval_episodes 1 --n_satellites 2 --n_routine 8 --n_dynamic 1 --out_root runs/ablation_train_stability_dry_run`。
- 训练路径冒烟: `/Users/zhouzidie/miniconda3/envs/myenv/bin/python compare_methods.py --n_satellites 2 --train_iters 1 --eval_episodes 1 --n_routine 8 --n_dynamic 1 --out_dir runs/train_stability_smoke --device cpu --satellite_curriculum --curriculum_min_satellites 1 --curriculum_iters 2 --joint_explore_prob 1.0 --experiment_tag train_stability_smoke`。
- 验证输出: `runs/train_stability_smoke/manifest.json`。该冒烟仅验证 active-agent curriculum 与 joint exploration 训练链路,不作为效果结论。

---

### 执行期通信 v1: 意图广播(2026-06-22,E17)

**目标**:当前 MAPPO actor 独立采样动作,冲突主要由环境 `_resolve_actions()` 事后处理。communication_v1 在策略执行层加入轻量通信:先广播各星初选任务,发现同一任务冲突后,败者在屏蔽已声明任务的 mask 上重新采样。

**实现**:
- `MAPPOTrainer.sample_actions()` 统一训练/评估 action 采样逻辑。
- `intent_broadcast=True`: 对同一非 idle 任务的多个意图,保留当前 policy log_prob 最高者,其余 agent 重采样。
- `intent_replan_rounds`: 最多重采样轮数,默认 1。
- 重采样后的 `action/log_prob/action_mask` 一起写入 buffer,保证 PPO 更新看到的是最终执行前策略动作。
- `run_ablation.py --preset communication_v1`: 对比 default、intent_broadcast、intent_broadcast + train_stability。

**推荐正式运行**:
```bash
python run_ablation.py \
  --python /Users/zhouzidie/miniconda3/envs/myenv/bin/python \
  --preset communication_v1 \
  --n_satellites 6 --train_iters 30 --eval_episodes 5 \
  --n_routine 200 --n_dynamic 50 \
  --out_root runs/ablation_communication_v1 \
  --device cpu
```

**注意**:这是规则式意图广播,不是可学习通信。若有效,下一步可升级为 E18/E19:学习消息向量、注意力通信或 TarMAC/CommNet 风格模块。

**本地验证**:
- 语法检查: `PYTHONPYCACHEPREFIX=/private/tmp/mrl_dms_pycache python3 -m compileall algo compare_methods.py run_ablation.py`。
- train stability 回归 dry-run: `python3 run_ablation.py --preset train_stability_v1 --dry_run --train_iters 1 --eval_episodes 1 --n_satellites 2 --n_routine 8 --n_dynamic 1 --out_root runs/ablation_train_stability_dry_run`。
- communication dry-run: `python3 run_ablation.py --preset communication_v1 --dry_run --train_iters 1 --eval_episodes 1 --n_satellites 2 --n_routine 8 --n_dynamic 1 --out_root runs/ablation_communication_dry_run`。
- 通信路径冒烟: `/Users/zhouzidie/miniconda3/envs/myenv/bin/python compare_methods.py --n_satellites 2 --train_iters 1 --eval_episodes 1 --n_routine 8 --n_dynamic 1 --out_dir runs/communication_smoke --device cpu --intent_broadcast --intent_replan_rounds 1 --experiment_tag communication_smoke`。
- 验证输出: `runs/communication_smoke/manifest.json`。该冒烟仅验证训练/评估共用意图广播 action 采样链路,不作为效果结论。

---

### CVA-MAPPO 主方案 v1(2026-07-04)

**论文定位**:将外循环思想从“单独替换 PPO/LSTM 结构”收敛为任务分配阶段的上下文价值建模器。最终方法暂定为 **CVA-MAPPO: Contextual Value-aware Assignment MAPPO**。

**核心框架**:
1. 高层候选生成:根据 VTW、deadline、任务到达时间和 off-nadir 生成可分配 `(satellite, task)` 边。
2. 上下文价值编码:使用 `MLP/LSTM/GRU/Transformer/Set Transformer/GNN` 编码任务集合、任务序列和卫星-任务二分图上下文。
3. CVA 边价值打分:融合观测质量、优先级、动态任务标记、deadline 压力、候选稀缺性、负载压力、owner stale/release/switch 历史和上下文编码器价值。
4. 约束 owner 分配:保留现有可见性硬约束、容量比例、所有权掩码、switch penalty、lock window 和 deadline release。
5. 低层 MAPPO 调度:MAPPO 在 owner 约束下执行具体观测动作,负责局部时序调度和协同执行。
6. 滚动重分配:动态任务到达、周期触发、stale owner 和 deadline 风险时重新计算 CVA owner。

**代码实现**:
- `envs/multi_satellite_env.py`
  - 新增 `assignment_scorer="cva"`。
  - 新增 `assignment_context_encoder` 与 `assignment_context_weight`。
  - `_assignment_cva_score()` 显式实现上下文价值感知分配分数。
  - 旧 `heuristic/mlp/lstm/gru/transformer/set_transformer/gnn` scorer 保持兼容。
- `compare_methods.py`
  - 新增 CLI: `--assignment_context_encoder`、`--assignment_context_weight`。
  - manifest 自动记录 CVA 参数。
- `run_ablation.py`
  - 新增 `--preset cva_assignment_v1`。
  - 默认 8 个子实验: `heuristic_static`、`heuristic_rolling`、`cva_lstm_static`、`cva_mlp/gru/lstm/transformer/set_transformer_rolling`。
  - 支持 `--cva_context_encoders`、`--cva_scorer_mixes`、`--cva_context_weight`。

**推荐压力消融命令**:
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

**论文实验解释**:
- `heuristic_static -> heuristic_rolling`:验证滚动 owner 重分配收益。
- `heuristic_rolling -> cva_*_rolling`:验证上下文价值感知分配是否优于规则分配。
- `cva_lstm_static -> cva_lstm_rolling`:验证 CVA 与滚动重分配的互补性。
- `cva_mlp/lstm/gru/transformer/set_transformer`:验证外循环/上下文编码器在分配阶段的作用。

**后续研究升级**:
1. 用 Greedy/ILP/MPC 标签监督训练 CVA scorer,把当前 deterministic encoder 推进为可学习 assignment value model。
2. 使用 MAPPO critic 或低层 rollout return 作为任务-owner 边价值标签。
3. 将 `assignment_context_encoder` 参数保存为 checkpoint,允许离线训练后加载权重。

---

### 基站下传约束 v1(2026-07-06)

**研究动机**:原环境中任务在卫星完成观测后立即计为完成,没有刻画遥感任务中“观测-下传-交付”的完整链路。为让调度结果更接近真实任务闭环,新增基站下传约束:卫星观测目标后,图像必须传输到基站才算任务完成。

**当前建模假设**:
1. 基站数量为 `n_ground_stations`,所有卫星共享同一组基站资源。
2. 每幅图像下传时间固定为 `downlink_time_s`。
3. 暂不建模基站空间位置、星地可见窗口、带宽差异和存储容量。
4. 下传由环境自动分配给最早可用基站;若下传完成时间超过 horizon 或任务 deadline,该任务只算“已观测”,不算“已完成”。
5. 默认 `n_ground_stations=0` 或 `downlink_time_s=0` 时保持旧口径:观测结束即完成。

**代码实现**:
- `data/mission_generator.py`:任务新增 `is_downlinked/downlink_start_s/downlink_end_s/ground_station_id`。
- `envs/satellite_env.py`:单星环境新增基站队列、固定下传耗时和下传完成口径。
- `envs/multi_satellite_env.py`:多星环境共享同一组基站可用时间,完成率按去重后的下传完成任务统计。
- `compare_methods.py`:新增 CLI `--n_ground_stations`、`--downlink_time_s`,并输出 `n_observed/n_downlinked/n_pending_downlink/avg_downlink_queue_s`。
- `cva_mappo_v2/run_experiment.py`:CVA-MAPPO-v2 主方案支持同样的基站下传参数。
- `run_ablation.py`、`train.py`、`algo/task_worker.py`:消融、单独训练和并行 worker 均支持该口径。

**推荐主对比命令**:
```bash
python compare_methods.py \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --n_satellites 12 \
  --train_iters 30 \
  --eval_episodes 8 \
  --n_routine 1200 \
  --n_dynamic 300 \
  --methods single,indep,mappo \
  --n_ground_stations 4 \
  --downlink_time_s 300 \
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
  --candidate_action_top_k 128 \
  --rollout_steps 256 \
  --ppo_epochs 2 \
  --ppo_batch_size 256 \
  --train_env_workers 4 \
  --eval_workers 4 \
  --vtw_time_step_s 60 \
  --out_dir runs/compare_ground_station_v1 \
  --device cpu
```

**推荐消融方向**:
- 基站数量: `--n_ground_stations 1/2/4/8`。
- 下传耗时: `--downlink_time_s 60/300/600/1200`。
- 对照旧口径: `--n_ground_stations 0 --downlink_time_s 0`。

**后续升级**:
1. 引入真实基站经纬度与星地可见窗口,从固定服务台升级为星地链路调度。
2. 加入卫星存储容量和下传队列,让未下传图像占用星上存储。
3. 把“是否优先观测高价值但下传拥堵的任务”纳入 CVA scorer,形成观测-下传联合价值评估。

---

### 基站可见 VTW 下传约束 v2(2026-07-06)

**本次修正**:v1 中基站仅作为共享服务台,没有判断卫星是否能看到基站。v2 将基站建模为真实地理位置,下传必须落在“卫星-基站通信可见时间窗口”内。

**实现要点**:
1. `config.py` 新增 `GroundStationConfig` 和默认基站表 `DEFAULT_GROUND_STATIONS`。
2. `data/orbit_utils.py` 新增 `compute_ground_station_vtw()`:
   - 使用卫星轨道传播和基站经纬度计算星地链路窗口;
   - 只约束最低仰角 `min_elevation_deg`,不使用光学成像的 FOV/roll/日照约束;
   - 复用全局内存缓存和 `MRL_DMS_VTW_CACHE_DIR` 磁盘缓存。
3. `envs/satellite_env.py`:
   - reset 时预计算当前卫星到所有基站的 `ground_station_vtw`;
   - `_schedule_downlink()` 在基站空闲时间和卫星-基站 VTW 的交集中选择最早可行下传;
   - 若没有可见窗口或排队后超过 deadline/horizon,任务只算已观测,不算下传完成。
4. `envs/multi_satellite_env.py`:
   - 多颗卫星共享同一组基站资源;
   - 每颗卫星保留自己的 satellite-ground VTW;
   - 同一 multi-agent step 内的新增观测按真实 `obs_end_s` 重排下传,避免 Python 遍历顺序影响基站分配。
5. `precompute_scenarios.py`:
   - 新增 `--n_ground_stations`;
   - 预热任务 VTW 的同时预热卫星-基站通信 VTW,降低后续训练/评估开销。

**推荐预计算命令**:
```bash
python precompute_scenarios.py \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --n_satellites 12 \
  --n_train_scenarios 200 \
  --n_eval_scenarios 20 \
  --n_routine 1200 \
  --n_dynamic 300 \
  --n_ground_stations 4 \
  --curriculum_stages 300:75,600:150,900:225,1200:300 \
  --vtw_time_step_s 60 \
  --vtw_workers 12 \
  --out_dir runs/scenario_cache/cva_stress_sat12_r1200_d300_gs4_seed42
```

**推荐主对比命令**:
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
  --methods single,indep,mappo \
  --n_ground_stations 4 \
  --downlink_time_s 300 \
  --assignment_capacity_mode proportional \
  --assign_w_load 0.1 \
  --release_before_deadline_s 1800 \
  --assignment_scorer cva \
  --assignment_scorer_mix 0.35 \
  --assignment_context_encoder lstm \
  --assignment_context_weight 0.25 \
  --assignment_replan_interval_s 3600 \
  --assignment_replan_horizon_s 7200 \
  --assignment_replan_trigger periodic,dynamic,stale_owner,deadline \
  --candidate_action_top_k 128 \
  --rollout_steps 256 \
  --ppo_epochs 2 \
  --ppo_batch_size 256 \
  --train_env_workers 4 \
  --eval_workers 4 \
  --vtw_time_step_s 60 \
  --out_dir runs/compare_ground_station_v2 \
  --device cpu
```

**重点观察指标**:
- `n_observed`: 已完成观测数量。
- `n_downlinked` / `n_scheduled`: 已下传并交付数量。
- `n_pending_downlink`: 已观测但未能下传完成数量。
- `avg_downlink_queue_s`: 基站排队造成的平均等待。
- `n_ground_station_vtws` / `avg_ground_station_vtws`: 卫星-基站可见窗口数量,用于诊断基站网络是否过稀。

---

### 星上存储容量约束与星间转发 v1(2026-07-06)

**研究动机**:仅有基站下传约束仍不完整。真实卫星观测后图像会占用星上存储,若存储满则不能继续拍摄;只有下传到基站或转发给其他卫星后,源卫星存储才会释放。

**当前建模假设**:
1. 每颗卫星最多同时存储 `satellite_storage_capacity` 张未交付图片; `0` 表示关闭容量限制。
2. 图片在 `obs_end_s` 后占用源卫星存储。
3. 若成功下传基站,源卫星在 `downlink_end_s` 释放存储。
4. 若无法在 deadline/horizon 前下传,图片在 deadline 被视为过期丢弃,任务只算已观测,不算完成。
5. 星间转发为规则式 fallback,默认关闭;开启后仅在源卫星无法直接下传时尝试转给其他有空余存储且能下传的卫星。
6. 当前星间转发使用固定耗时 `inter_satellite_transfer_time_s`,暂不建模星间链路可见窗口。

**代码实现**:
- `envs/satellite_env.py`
  - 新增 `satellite_storage_capacity`;
  - 新增 `StorageRecord`;
  - action mask 在存储满时只保留 idle;
  - idle 会推进到最近的存储释放时刻;
  - 指标新增 `current_onboard_images/max_onboard_images/avg_onboard_images/n_storage_expired_drops`。
- `envs/multi_satellite_env.py`
  - 多星环境为每颗子卫星维护独立存储占用;
  - 新增规则式星间转发 fallback;
  - 指标新增 `n_inter_satellite_transfers/n_relay_storage_images/inter_satellite_transfer_time_s`。
- `compare_methods.py`、`run_ablation.py`、`train.py`、`cva_mappo_v2/run_experiment.py`
  - 新增 CLI: `--satellite_storage_capacity`;
  - 新增 CLI: `--enable_inter_satellite_transfer`;
  - 新增 CLI: `--inter_satellite_transfer_time_s`。

**推荐容量压力命令**:
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
  --methods single,indep,mappo \
  --n_ground_stations 4 \
  --downlink_time_s 300 \
  --satellite_storage_capacity 8 \
  --enable_inter_satellite_transfer \
  --inter_satellite_transfer_time_s 300 \
  --assignment_capacity_mode proportional \
  --assign_w_load 0.1 \
  --release_before_deadline_s 1800 \
  --assignment_scorer cva \
  --assignment_scorer_mix 0.35 \
  --assignment_context_encoder lstm \
  --assignment_context_weight 0.25 \
  --assignment_replan_interval_s 3600 \
  --assignment_replan_horizon_s 7200 \
  --assignment_replan_trigger periodic,dynamic,stale_owner,deadline \
  --candidate_action_top_k 128 \
  --rollout_steps 256 \
  --ppo_epochs 2 \
  --ppo_batch_size 256 \
  --train_env_workers 4 \
  --eval_workers 4 \
  --vtw_time_step_s 60 \
  --out_dir runs/compare_storage_capacity_v1 \
  --device cpu
```

**建议消融**:
- 存储容量: `--satellite_storage_capacity 2/4/8/16/0`。
- 下传耗时: `--downlink_time_s 60/300/600/1200`。
- 星间转发: 对比不开启与开启 `--enable_inter_satellite_transfer`。

**后续升级**:
1. 给星间转发加入卫星-卫星可见窗口和链路速率。
2. 把“主动下传/主动转发”从规则式 fallback 升级为策略动作或高层调度动作。
3. 在 CVA scorer 中加入存储压力、未来下传机会和转发代价,做观测-存储-下传联合价值评估。
