# 当前项目工作报告

更新时间：2026-07-20

## 1. 项目定位

当前分支是 `DAS-CVA-MAPPO` 的研发分支，目标是面向多卫星、多地面站、动态任务到达、星上存储、数传和星间转移约束的动态任务调度问题，构建一个可用于论文实验的 MAPPO 强化学习调度方法。

项目已经从早期 `cva_mappo_v2` 的固定槽位、规则候选分配路线，逐步转向 `das_cva_mappo` 主线。当前推荐的论文方法中心是 `das_cva_mappo/`，而 `cva_mappo_v2/` 主要作为候选生成、环境封装和兼容层继续被使用。

当前主线版本为 `DAS-CVA-MAPPO V0.33.0`。

## 2. 已完成的主要工作

### 2.1 项目清理与主线收敛

已完成历史代码清理，保留与当前方法有关的模块：

- `das_cva_mappo/`：DAS-CVA-MAPPO 主方法，包括动作集感知 actor、候选 scorer、特征构建、环境 adapter、rollout buffer 和训练入口。
- `cva_mappo_v2/`：兼容层，负责候选生成、任务 owner 分配、可执行性判断和部分环境逻辑。
- `envs/`：单星与多星调度环境。
- `scripts/run_stage_ablation_suite.py`：阶段化实验与消融汇总脚本。
- `scripts/run_paper_experiment_suite.py`：论文实验总入口，按论文问题组织当前可运行的实验组合，并保存实验计划和汇总结果。
- `docs/`：设计总结与版本历史。

这一点对论文叙事很重要：方法主体已经不再依赖旧的固定 slot actor 作为主要贡献，而是围绕动态动作集、候选集选择和调度环境一致性展开。

### 2.2 DAS 动态动作集策略

当前实现已经完成以下核心结构：

- 将每颗卫星当前可选动作表示为 action entities，而不是让固定输出神经元永久绑定固定任务语义。
- 支持 set-transformer matcher，用动作特征、局部状态和动作集合上下文计算每个候选动作的 logit。
- 保留 additive、dot、set_transformer matcher 的消融接口。
- 支持 `full`、`minimal`、`no_score` 等动作特征模式。
- 通过 idle auxiliary loss 抑制“有可行动作时仍高概率 idle”的问题。

这部分已经形成可写进论文方法章节的主体：策略网络不是直接预测固定任务编号，而是在每个时刻对候选动作集合打分。

### 2.3 CVA 候选评分与辅助训练

候选评分路线已经从纯规则启发式扩展为：

- `v2_heuristic`：保留兼容层规则分数，便于诊断和消融。
- `learned`：使用学习型边评分器。
- `hybrid`：启发式和学习型 scorer 混合。

当前已实现：

- CVA scorer warmup。
- rollout advantage 辅助更新。
- hard-negative candidate sampling。
- conflict penalty 与 load penalty。
- scorer feature 中加入任务动态性、等待压力、可执行性、响应预算、下传交付压力和未来窗口时序摘要。

这为“候选生成不是静态规则，而是可学习调度价值估计”提供了实验基础。

### 2.4 候选 owner 修复与可执行性暴露

项目已经围绕候选可见性和 owner 分配做了多轮修复：

- typed slot：routine、dynamic、flex 分槽，减少候选类型相互淹没。
- dynamic、urgent、stale owner 多 owner 广播，缓解动态任务只被少数卫星看到的问题。
- stale-owner release 与 deadline release，释放已经不可执行或临近截止的任务。
- executable-aware candidate exposure，用当前可执行任务优先占用候选槽。
- future-task metadata，包括当前可执行、未来可执行、等待时间、下一可执行时间、截止时间等。
- visible-candidate idle advancement：当策略选择 idle 时，任务窗口跳转只考虑策略实际看得到的任务槽，同时动态任务到达和存储释放仍作为全局事件保留。

这些修改解决了早期实验中最明显的问题：`avg_filled_slots` 很高，但 `avg_valid_slots` 极低，策略看到大量“填充了但当前不可执行”的动作，导致有效决策率很低。

### 2.5 未来任务宏动作

已实现 bounded future-task macro execution：

- 允许候选任务虽然当前不可执行，但如果在短时间内有可行观测窗口，可以作为未来宏动作被选择。
- 对 routine future 与 dynamic future 使用不同等待上限。
- 对 routine future 加入 dynamic guard，避免 routine 未来任务抢占动态任务机会。
- 增加 `n_future_task_executions`、`n_future_dynamic_task_executions`、`avg_future_task_wait_s` 等指标。

该路线经过几轮验证后证明：完全开放未来宏动作会导致吞吐崩塌；完全禁用未来宏动作又浪费未来窗口信息。目前采用的是有边界、有类型区分、带动态保护的折中方案。

### 2.6 动态任务响应优化

针对动态任务表现过弱的问题，已完成多轮专项优化：

- 动态候选打分加入 response pressure。
- future dynamic window 的等待惩罚按 `dynamic_response_target_s` 归一化，而不是按整天 horizon 稀释。
- 当前可执行 dynamic slot 额外加权。
- `stage2_dynamic_priority_recovery` 提供动态优先配置。
- 增加 dynamic current/future slot exposure 诊断指标。
- 增加 dynamic-priority downlink replanning，让未开始的 routine downlink 可以被动态任务图像重排到后面。

最近一轮 V0.29 验证表明，观测后再做 dynamic-priority downlink replanning 并没有稳定降低动态响应时间。V0.30 因此把下传队列和交付延迟前置到候选边价值中，让策略在选择观测任务前就能感知预计下传代价。V0.31 进一步把动态任务响应预算显式接入 actor 局部状态、动作实体特征和学习型候选 scorer 的边特征中。V0.32 在此基础上加入未来窗口时序摘要，并提供 GRU state-history encoder 对比版本。V0.32 结果显示未来窗口特征改善了下传闭环，但平均动态响应时间被拉长，因此 V0.33 增加 early-delivery temporal features，把时序信号从“最终可交付”进一步推向“尽早交付”。

### 2.7 V0.33 时序模块

当前 V0.33 已实现三条可对比的时序路线：

- 未来窗口特征版：对每个卫星-任务候选抽取未来 top-K 可行观测窗口的摘要，包含等待时间、窗口质量、质量趋势、预计下传队列、交付延迟、下传可行性和响应/截止预算余量；这些特征同时进入 actor action entity 和学习型 candidate edge scorer。
- early-delivery temporal 版：在未来窗口摘要上继续加入首个窗口交付延迟、首个窗口剩余响应预算、首个窗口超预算标记、最早可行交付延迟、最早可行交付预算余量，以及质量最优窗口相对最早交付窗口的延迟差；动态任务的窗口选择 key 也通过 `--temporal_early_delivery_weight` 偏向早交付。
- GRU state-history 版：通过 `--temporal_state_encoder gru --temporal_state_history_len 4` 将最近若干步局部 state 拼成固定历史序列，由 GRU 编码后再进入 action-set actor。该版本不改变 PPO buffer 的主体结构，也不在多进程 worker 间维护 recurrent hidden state，因此适合先做模型侧时序对比。

当前建议优先比较 early-delivery temporal 版、V0.32-like 未来窗口特征版、GRU state-history 版和关闭 temporal 的消融版，而不是直接重构成完整 recurrent MAPPO。完整 recurrent MAPPO 需要改 rollout buffer、hidden state reset、done mask、并行采样和 eval 路径，工程风险明显更高。

### 2.8 训练与评估一致性

已经修复了一个对论文结果很关键的问题：早期 evaluation 默认启用了 eval-only repair/rescue，而 training 没有完全相同的处理。这会让评估结果带有规则后处理增益，不能严格代表策略本身。

当前状态：

- eval 默认走与 train 一致的动作处理路径。
- 旧的 eval repair 只保留在 `--eval_use_repair` 诊断开关后。
- 增加静态回归测试，防止 eval 默认路径再次偏离 train。

这提高了实验结论的可信度。

### 2.9 并行训练、评估与性能诊断

已完成：

- `--train_env_workers` 并行 rollout。
- 默认训练设备为 `cuda:0`。
- 默认评估设备回到 `cpu`，避免 CUDA eval 在当前环境下与论文迭代路线冲突。
- 支持 CPU 多进程评估。
- 支持单进程 batched CUDA/MPS eval，但当前建议后续统一用 CPU eval。
- 增加 `--eval_profile`，可记录 eval wall time、env step、actor forward、feature build 等耗时。
- 增加 all-idle fast path 和低层环境 fast step，减少高 idle 场景下的重复 Python 计算。

从 profiling 结果看，eval 时间长的主要瓶颈不在 GPU 前向，而在 Python 环境步进、候选检查、任务/数传状态推进等模拟逻辑，尤其 `eval_env_step_time_s` 占比极高。

### 2.10 实验脚本与结果汇总

已完成阶段化实验脚本：

- 顺序运行 stage1 到 stage4。
- 支持只运行指定阶段或消融。
- 默认 `train_iters=50`、`eval_episodes=10`、`eval_workers=24`、`train_env_workers=16`、训练设备 `cuda:0`、评估设备 `cpu`。
- 自动生成 `summary.csv` 和 markdown 汇总表。
- 新增论文实验 wrapper：`scripts/run_paper_experiment_suite.py`，可用 `--plan quick_temporal`、`--plan progression`、`--plan mechanism_core`、`--plan paper_core` 或 `--plan paper_full` 运行当前可做的论文实验列表，并在 suite 目录保存 `paper_experiment_plan.json` 与 `paper_experiment_plan.md`。其中 `paper_core` 会避开当前已知的 Stage-2 重复标签，使用 `stage2_candidate_owner_repair` 作为 V0.33 early-delivery temporal 基线；`paper_full` 保留历史标签，便于复现旧结果表。
- 已加入多个关键消融：
  - 禁用未来任务执行。
  - future macro with current valid。
  - 禁用动态响应压力。
  - 禁用动态 downlink priority。
  - 动态优先恢复配置。

这已经满足阶段性验证、论文表格汇总和单因素消融的基本需要。

## 3. 当前工作的优点

### 3.1 方法贡献点比较清晰

当前项目已经形成了相对明确的论文贡献结构：

- 动态动作集感知 MAPPO 策略。
- CVA 候选评分与学习型边价值估计。
- 候选 owner 分配、可执行性暴露和未来窗口宏动作。
- 面向动态任务的响应时间优化。
- 多星、多站、存储、下传、星间转移一体化评估环境。

这些点比单纯调参更像一个完整方法。

### 3.2 实验诊断指标丰富

当前指标已经不只看 total reward 和 completion rate，还能定位问题来源：

- 候选集是否有效：`avg_valid_slots`、`avg_filled_invalid_slots`、`eval_valid_decision_rate`。
- 动态任务是否被看到：`dynamic_current_slot_exposure_rate`、`dynamic_future_slot_exposure_rate`。
- 未来宏动作是否过度使用：`n_future_task_executions`、`avg_future_task_wait_s`。
- owner 是否失效：`stale_owner_rate`、`owner_churn_rate`。
- 下传是否拖慢响应：`avg_downlink_queue_s`、`avg_dynamic_downlink_replan_gain_s`。
- 评估性能瓶颈：`eval_env_step_time_s`、`eval_actor_forward_time_s`、`eval_feature_build_time_s`。

这使得后续优化可以按瓶颈推进，而不是盲目调 reward。

### 3.3 Train/Eval 一致性得到修复

论文结果最怕评估阶段隐藏规则修复。当前已经把 eval-only repair 移到显式开关后，并加测试保护，这是一个重要优点。

### 3.4 工程路线可复现实验

README 里已有完整运行命令，stage suite 可以复现实验组合，summary 表可以直接用于论文筛选。默认训练使用 CUDA，评估使用 CPU，多环境参数也已经显式化。

### 3.5 动态任务问题被正确拆解

当前分析已经从“动态任务完成率低”拆成了几类原因：

- 动态任务本身 feasible ratio 较低，raw rate 有天然上限。
- 候选集有效性不足导致策略看不到可执行动作。
- future macro 可能牺牲动态响应。
- downlink queue 是动态响应时间的重要组成。
- routine throughput 与 dynamic response 存在调度权衡。

这比只追求单个指标更适合论文呈现。

## 4. 当前工作的缺点与风险

### 4.1 动态任务 raw completion 仍偏低

从最近的实验看，动态任务 feasible-normalized completion 已经能达到较高水平，但 `dynamic_completion_rate_raw` 仍大约在 0.11 到 0.13 区间。由于 `dynamic_feasible_ratio` 约为 0.14 到 0.21，raw 指标受可行性上限影响较大，但论文读者首先看到 raw rate 时仍可能认为动态任务效果不够强。

后续需要同时报告：

- raw completion。
- feasible-normalized completion。
- dynamic feasible ratio。
- 在可行动态任务集合上的 response time。

否则动态任务结果容易被误读。

### 4.2 `stale_owner_rate` 长期偏高

多轮结果中 `stale_owner_rate` 经常接近 1。这说明 owner 分配仍有大量任务最终处于“当前 owner 已无未来可行窗口”的状态。

虽然系统通过 release、multi-owner、dynamic broadcast 缓解了执行问题，但 owner 机制本身仍不够稳定。论文中如果强调“分配质量”，需要进一步降低 stale owner，或者把它解释为动态释放机制的一部分，而不是静态 owner 最优分配。

### 4.3 候选有效槽位仍很稀疏

即使经过优化，`avg_valid_slots` 仍明显低于 `avg_filled_slots`。这说明策略输入里仍有大量上下文候选不是真正立即可执行动作。

未来宏动作缓解了一部分问题，但也带来新风险：策略可能过早承诺未来任务，跳过中间动态任务到达或更优窗口。因此候选集“信息展示”和“动作可执行”之间的边界仍需要继续打磨。

### 4.4 动态响应时间仍主要受 downlink 影响

`avg_downlink_queue_s` 经常达到数小时量级，而 `avg_dynamic_response_s` 定义包含从任务最早时间到下传完成的全过程。因此即使观测调度变好，若动态图像排在 routine downlink 后面，响应时间仍会很差。

V0.29 的 dynamic-priority downlink replanning 没有在最近结果中形成稳定收益。V0.30 已把优化重心改为 downlink-aware candidate edge value，需要继续确认它是否能稳定降低：

- `avg_dynamic_response_s`
- `avg_downlink_queue_s`
- `avg_dynamic_downlink_replan_gain_s`

如果收益不稳定，说明 ground segment 还需要更强的调度策略，而不只是局部重排。

### 4.5 当前仍依赖兼容层

虽然论文主线是 `das_cva_mappo`，但候选生成和大量环境逻辑仍在 `cva_mappo_v2` 与 `envs/` 中。方法结构上已经较清楚，但工程上还不是完全独立的 DAS allocator。

这不一定影响论文实验，但会影响代码叙事：需要明确说明 `cva_mappo_v2` 是环境和候选生成支撑层，DAS 贡献在动作集策略、候选 scorer 和训练逻辑。

### 4.6 评估时间仍偏长

当前 profiling 显示评估耗时主要来自 Python 环境模拟和步进，而不是模型前向。评估慢会带来两个问题：

- 实验迭代速度慢。
- 论文结果难以扩大 seed、场景规模和消融数量。

已做 fast path 优化，但如果后续要跑大规模论文表格，还需要进一步减少环境步进次数或做事件驱动评估。

### 4.7 学习型 scorer 的论文说服力仍需补强

当前 hybrid scorer 已有工程实现，但从已有结果看，heuristic 与 dynamic-priority 配置有时已经很强，learned/hybrid scorer 的稳定增益还不够明确。

如果论文要把 learnable CVA scorer 作为主要创新，需要补充：

- 与纯 heuristic 的多 seed 对比。
- scorer 消融。
- scorer 预测质量分析，例如正负边排序准确率。
- hard-negative 是否减少 invalid/future/stale 候选曝光。

## 5. 对当前实验结果的判断

已有结果说明：

- 候选/owner 修复显著改善了早期“几乎没有有效动作”的问题。
- future macro 如果不受限制，会提高 valid decision，但会牺牲总完成率和动态任务表现。
- 有限制的 future macro 加 dynamic guard 是目前更合理的路线。
- 动态任务 feasible-normalized completion 已有改善，但 raw completion 仍受场景可行性上限和资源竞争限制。
- 动态响应时间的关键瓶颈很可能不是观测本身，而是观测后的下传排队。
- CPU eval 虽慢，但当前更稳定、与后续实验路线更一致。

因此，目前项目已经具备论文实验原型，但还没达到“最终结果可直接定稿”的状态。核心短板集中在动态任务 raw 表现、响应时间、stale owner 和评估效率。

## 6. 方法创新性强化方案

当前方法如果只描述为“在 MAPPO 上加入候选筛选和若干规则修复”，创新性会显得偏工程调参。论文中更有说服力的表述应当把方法提升为一个面向动态卫星任务调度的约束感知动作集决策框架。

建议将方法主张收敛为：

```text
Response-aware Dynamic Action-Set CVA-MAPPO
```

中文可以表述为“响应感知的动态动作集 CVA-MAPPO”。核心不是简单使用 MAPPO，而是解决动态调度中的三个结构性问题：

- 固定动作空间无法表达每颗卫星在不同时刻完全不同的可行任务集合。
- 候选任务如果只按观测收益排序，会忽略未来可执行性、任务响应时间和下传队列。
- 动态任务的完成质量不只取决于是否观测成功，还取决于能否及时下传并交付。

### 6.1 可作为论文贡献的创新点

第一，动态动作集感知策略。每颗卫星的动作不是固定 slot 语义，而是由当前候选任务、转移动作和 idle 动作组成的 action set。策略网络对动作实体打分，使同一个策略可以处理不同卫星、不同时间、不同候选规模下的异构动作集合。这比传统 fixed-head MAPPO 更适合动态任务调度。

第二，时序可执行性候选表示。候选集不只记录任务是否存在，还显式编码当前可执行、未来可执行、等待时间、下一观测窗口、截止时间压力等信息。这样策略可以区分“现在就能做的动作”和“未来有价值但需要等待的上下文任务”，避免把未来任务简单当成 invalid action 丢弃。

第三，响应感知 CVA 候选评分。动态任务的候选排序引入 response pressure，使任务年龄、响应目标和未来窗口等待共同影响卫星-任务边价值。该设计比单纯按优先级或 off-nadir 质量排序更贴合动态应急任务。

第四，观测-存储-下传耦合的调度评价。V0.30 已把 downlink queue、预计交付延迟和下传可行性纳入候选边价值，让动态图像在观测选择前就能根据端到端交付代价排序。论文中可以强调方法不只优化观测覆盖率，而是优化从任务到达到数据交付的端到端响应。

第五，训练评估一致的约束决策流程。当前 eval 默认不再启用 train 中没有的 repair 逻辑，保证论文结果反映策略和候选机制本身，而不是评估阶段额外规则后处理。这一点虽然不是算法创新，但能显著增强实验可信度。

### 6.2 还需要补强的机制创新

为了让方法不只停留在“已有实现加权重”，建议后续围绕四个机制继续强化，其中前三项已经完成第一版实现。

第一，downlink-aware edge value。V0.30 已完成第一版下传感知候选边价值，在候选评分阶段估计：

- earliest downlink start。
- earliest downlink finish。
- ground-station queue delay。
- onboard storage risk。
- relay usefulness。
- dynamic delivery deadline margin。

这样 CVA scorer 预测的是“观测并交付”的边价值，而不是只预测“观测”的边价值。后续还可以把该估计从当前的启发式预览升级为学习型 delivery-value head。

第二，response-budget-aware actor/scorer feature。V0.31 已把动态任务剩余响应预算接入 actor 局部状态、动作实体特征和学习型 candidate scorer 边特征。动态任务可以定义剩余响应预算：

```text
response_budget = dynamic_response_target_s - (current_time_s - arrival_time_s)
```

策略和 scorer 都可以使用该预算作为特征。后续还可以在奖励或 critic feature 中对超出响应目标的动态任务递增建模，而不是只在完成后统计 `avg_dynamic_response_s`。这样论文可以说明方法是 response-aware，而不是事后报告响应时间。

第三，future-window temporal feature / early-delivery temporal feature / GRU state-history。V0.32 已把候选任务未来 top-K 可行窗口的等待、质量、下传队列、交付延迟和预算余量接入 actor action entity 与 candidate edge scorer，同时提供 GRU 局部状态历史编码作为对比版本。V0.33 进一步加入早交付特征，直接表达“第一个可交付窗口是否会超出动态响应预算”和“质量最优窗口相对最早交付窗口要晚多少”。该机制用于回答“未来窗口序列是否值得等待，以及是否值得为了质量牺牲响应时间”，补足 V0.31 只能表达“当前任务是否紧急”的不足。

第四，候选 exposure 的可解释约束。对于每个动态任务，记录它从到达到完成之间被多少个卫星看到、看到时是否当前可执行、是否被 future slot 挤出、是否被 routine 下传阻塞。这可以形成一组可解释诊断指标，让方法改进与动态任务表现之间有因果链条。

### 6.3 论文实验需要证明的点

为了让创新点有说服力，实验表格应当围绕“每个机制解决一个结构性问题”设计，而不是只放最终 reward。

建议至少保留以下对比：

- fixed-slot MAPPO 或 v2 compatibility runner：证明动态动作集策略必要。
- DAS action-set actor without set context：证明动作集合上下文有用。
- heuristic CVA vs learned/hybrid CVA：证明学习型候选价值有增益。
- no future task execution：证明时序可执行性表示有用。
- open future macro：证明无约束未来宏动作会损害动态任务。
- no dynamic response pressure：证明响应感知候选排序有效。
- no downlink-aware edge value：证明端到端交付代价前置到候选评分中是有效的。
- post-hoc dynamic downlink priority：证明 V0.30 的前置评分优于 V0.29 的事后重排路线。

对应指标应包括：

- 总体：`total_reward`、`observation_success_rate_raw`。
- 动态：`dynamic_completion_rate_raw`、`dynamic_completion_rate`、`dynamic_feasible_ratio`。
- 响应：`avg_dynamic_response_s`、`avg_downlink_queue_s`。
- 候选：`avg_valid_slots`、`avg_filled_invalid_slots`、`dynamic_current_slot_exposure_rate`。
- 稳定性：`stale_owner_rate`、`owner_churn_rate`、`load_balance_cv`。
- 效率：`eval_wall_time_s`、`eval_steps_per_wall_s`，作为工程复现信息，不作为主要算法指标。

### 6.4 主要实验与验证目的

当前建议以 `paper_core` 作为论文主实验集合，先完成单 seed 或少量 seed 的结果筛选，再对关键结论补多 seed。主要实验和验证目的如下：

| 实验类别 | 实验名 | 对照关系 | 验证目的 | 重点指标 |
| --- | --- | --- | --- | --- |
| 阶段推进 | `stage1_slot_diagnosis` | 诊断起点 | 验证固定候选槽位和基础候选暴露下，策略是否能看到足够的当前可执行动作；定位 invalid slot、idle 和 stale owner 问题。 | `avg_valid_slots`、`avg_filled_invalid_slots`、`eval_valid_decision_rate`、`stale_owner_rate` |
| 阶段推进 | `stage2_candidate_owner_repair` | 对比 Stage 1 | 验证 typed slot、多 owner 广播、stale release、future-task metadata 和 V0.33 early-delivery temporal 默认配置是否改善候选可见性与动态任务完成。 | `total_reward`、`dynamic_completion_rate`、`dynamic_task_candidate_seen_rate`、`avg_dynamic_response_s` |
| 阶段推进 | `stage2_dynamic_priority_recovery` | 对比 Stage 2 | 验证动态任务优先候选配置是否能提高动态任务被看到、被选择和完成的比例。 | `dynamic_completion_rate_raw`、`dynamic_current_slot_exposure_rate`、`dynamic_task_policy_selected_rate` |
| 阶段推进 | `stage3_dynamic_hybrid` | 对比 Stage 2 dynamic | 验证 hybrid CVA scorer 相对纯启发式候选评分是否带来更好的任务选择和负载分配。 | `total_reward`、`load_balance_cv`、`owner_churn_rate`、`dynamic_completion_rate` |
| 阶段推进 | `stage4_storage_pressure` | 对比 Stage 3 | 验证加入更强存储/下传压力后，是否缓解图像堆积和下传队列拥塞。 | `avg_downlink_queue_s`、`n_storage_expired_drops`、`dynamic_task_downlink_queue_block_rate` |
| 时序对比 | `cmp_stage2_temporal_future_features` | 对比 `stage2_candidate_owner_repair` | 验证只有 V0.32-like 未来窗口摘要、没有早交付信号时，是否会提升最终交付但拉长动态响应。 | `dynamic_completion_rate`、`avg_dynamic_response_s`、`avg_future_task_wait_s` |
| 时序对比 | `cmp_stage2_temporal_gru_state` | 对比 `stage2_candidate_owner_repair` | 验证 GRU 局部状态历史编码是否能利用过去状态变化，提供区别于未来窗口特征的时序收益。 | `dynamic_completion_rate_raw`、`avg_dynamic_response_s`、`eval_actor_forward_time_s` |
| 时序消融 | `abl_stage2_no_temporal_window_features` | 对比 `stage2_candidate_owner_repair` | 验证未来窗口时序特征是否是当前主方法收益来源之一。 | `total_reward`、`dynamic_completion_rate`、`n_future_task_executions` |
| 机制消融 | `abl_stage2_no_future_task_execution` | 对比 `stage2_candidate_owner_repair` | 验证 bounded future-task macro execution 是否有助于利用短期未来观测窗口。 | `n_future_task_executions`、`total_reward`、`observation_success_rate_raw` |
| 机制消融 | `abl_stage2_no_dynamic_response_pressure` | 对比 `stage2_candidate_owner_repair` | 验证动态响应压力项是否能推动候选排序优先选择更紧急的动态任务。 | `avg_dynamic_response_s`、`dynamic_task_policy_selected_rate`、`dynamic_completion_rate` |
| 机制消融 | `abl_stage2_no_response_budget_features` | 对比 `stage2_candidate_owner_repair` | 验证显式响应预算特征进入 actor/scorer 后，是否有助于区分仍可及时交付和已经接近超时的动态任务。 | `avg_dynamic_response_s`、`dynamic_completion_rate_raw`、`dynamic_feasible_ratio` |
| 机制消融 | `abl_stage2_no_downlink_aware_edge_value` | 对比 `stage2_candidate_owner_repair` | 验证把下传队列、预计交付延迟和交付可行性前置到候选边价值中是否必要。 | `avg_downlink_queue_s`、`dynamic_task_downlinked_after_observed_rate`、`dynamic_task_downlink_queue_block_rate` |
| 机制消融 | `abl_stage2_posthoc_dynamic_downlink_priority` | 对比 `stage2_candidate_owner_repair` | 验证事后动态下传重排是否优于当前的观测前下传感知边价值路线。 | `avg_dynamic_response_s`、`n_downlink_priority_replans`、`avg_dynamic_downlink_replan_gain_s` |
| 模型消融 | `abl_v2_heuristic_scorer` | 对比 `stage4_storage_pressure` | 验证 hybrid/learned CVA scorer 相比纯启发式 scorer 是否提供稳定增益。 | `total_reward`、`dynamic_completion_rate`、`load_balance_cv` |
| 模型消融 | `abl_no_candidate_aux_update` | 对比 `stage4_storage_pressure` | 验证 rollout advantage 辅助更新和 hard-negative 排序训练是否改善 scorer 的候选排序能力。 | `total_reward`、`eval_valid_decision_rate`、`avg_valid_slots` |
| 模型消融 | `abl_no_action_type_gate` | 对比 `stage4_storage_pressure` | 验证 action-type gate 是否有助于区分观测、转发、idle 等不同动作实体类型。 | `total_reward`、`eval_idle_action_rate`、`dynamic_completion_rate` |
| 模型消融 | `abl_no_set_context` | 对比 `stage4_storage_pressure` | 验证 action-set context 是否能缓解动态动作集中候选之间的相互竞争和语义错位。 | `total_reward`、`eval_valid_decision_rate`、`load_balance_cv` |
| 模型消融 | `abl_no_idle_aux` | 对比 `stage4_storage_pressure` | 验证 idle auxiliary loss 是否减少“有可行动作仍选择 idle”的无效等待。 | `eval_idle_when_valid_rate`、`eval_idle_action_rate`、`total_reward` |
| 模型消融 | `abl_no_future_task_execution` | 对比 `stage4_storage_pressure` | 在 Stage-4 强配置下复验 future macro 对最终方法是否仍有贡献。 | `n_future_task_executions`、`total_reward`、`dynamic_completion_rate` |
| 模型消融 | `abl_no_storage_pressure` | 对比 `stage4_storage_pressure` | 在 Stage-4 强配置下验证存储/下传压力项是否降低过期丢弃和队列阻塞。 | `avg_downlink_queue_s`、`n_storage_expired_drops`、`dynamic_task_downlink_queue_block_rate` |
| 压力测试 | `stress_12sat_double_tasks` | 12 星、routine 1200、dynamic 300；内部对比 `stage2_candidate_owner_repair`、`stage4_storage_pressure`、`abl_no_storage_pressure`、`abl_stage2_no_downlink_aware_edge_value` | 验证主方法在星座规模和任务负载同时增大时，是否仍能维持候选有效性、动态响应和下传闭环；同时检验 storage/downlink-aware 机制在高压场景下是否更关键。 | `total_reward`、`dynamic_completion_rate_raw`、`avg_dynamic_response_s`、`avg_downlink_queue_s`、`eval_wall_time_s` |

表格使用时应注意：`stage2_candidate_owner_repair` 在当前默认参数下可作为 V0.33 early-delivery temporal 主基线；`paper_full` 中保留的 `cmp_stage2_temporal_early_delivery_features`、`abl_stage2_no_dynamic_downlink_priority` 和 `abl_stage2_no_early_delivery_temporal_features` 更多用于复现历史表或显式标注，不一定都需要进入论文主表。

### 6.5 论文叙事建议

论文中不要把方法写成“MAPPO 加规则候选筛选”。更强的叙事是：

1. 动态卫星调度的动作空间是时变、异构、强约束的，固定离散动作头会产生语义错位。
2. 因此提出动态动作集策略，将每个可选调度决策表示为动作实体，由共享策略对动作实体打分。
3. 候选集由 CVA scorer 生成，边价值同时考虑观测质量、任务紧迫性、可执行窗口、存储压力和下传交付压力。
4. 通过 future macro 和 visible-candidate idle advancement，让策略既能利用未来窗口，又不跳过不可见的关键事件。
5. 通过下传感知边价值和未来窗口时序编码，把动态任务目标从“观测完成”扩展为“及时交付”。

这样的叙事比单独强调某个 reward 权重更稳，也更容易解释为什么该方法适合动态任务场景。

### 6.6 不建议作为主要创新点的内容

以下内容可以放在实现细节或实验设置里，但不建议作为主要创新点：

- 单纯把 `--train_env_workers` 调大。
- CUDA 或 CPU eval 的选择。
- 增加更多日志列。
- 某个具体 reward 权重调参。
- 只靠 heuristic priority 提升动态任务。

这些内容对工程有效，但论文创新性较弱。它们应服务于主方法，而不是成为主方法本身。

## 7. 建议的后续路线

### 7.1 优先完成 V0.33 早交付时序对比

建议先运行当前短验证，对比 early-delivery temporal、V0.32-like 未来窗口特征版、GRU state-history 版和关闭未来窗口特征的消融版：

```bash
python3 scripts/run_paper_experiment_suite.py \
  --plan quick_temporal \
  --suite_name das_v033_quick_temporal \
  --train_iters 50 \
  --val_episodes 10 \
  --eval_workers 10 \
  --eval_device cpu \
  --train_env_workers 16 \
  --device cuda:0 \
  --no_progress
```

重点看：

- `dynamic_completion_rate_raw`
- `dynamic_completion_rate`
- `avg_dynamic_response_s`
- `avg_downlink_queue_s`
- `n_downlink_priority_replans`
- `avg_dynamic_downlink_replan_gain_s`
- `dynamic_task_candidate_seen_rate`
- `dynamic_task_policy_selected_rate`
- `n_future_dynamic_task_executions`
- `avg_future_task_wait_s`
- `dynamic_task_downlink_queue_block_rate`
- `dynamic_current_slot_exposure_rate`
- `dynamic_future_slot_exposure_rate`

如果 V0.33 early-delivery temporal 相比 V0.32-like future-window features 同时保住 `dynamic_task_downlinked_after_observed_rate` 并降低 `avg_dynamic_response_s`，则早交付特征可以保留为主线模型设计。若 GRU state-history 版仍只改善完成率但不改善响应时间，则暂时不把 GRU 作为论文主方法核心。

### 7.2 动态任务指标单独成表

论文结果不建议只放总表。应单独建立动态任务表：

- dynamic feasible ratio
- dynamic raw completion
- dynamic feasible-normalized completion
- dynamic response time
- dynamic downlink queue
- dynamic future/current slot exposure

这样可以说明 raw completion 的上限来自场景可行性，而方法贡献主要在“可行动态任务完成率”和“响应延迟”。

### 7.3 继续压低 downlink queue

如果 V0.33 的早交付时序特征仍不能显著压低下传队列或动态响应时间，下一步应考虑：

- 为动态图像设置更强的 downlink deadline 或 priority key。
- 将当前启发式预计 downlink finish time 升级为 learned delivery-value head。
- 将 ground-station queue pressure 加入候选 edge feature。
- 对即将产生动态图像的任务预留或提前释放下传窗口。

### 7.4 降低 stale owner

建议增加一个 owner quality 表：

- owner 是否还有未来可行窗口。
- owner 的 earliest feasible start。
- 当前候选 owner 与最早可执行卫星之间的时间差。
- release 后任务是否被新 owner 完成。

如果 stale owner 仍高，应把 owner 从“任务归属”弱化成“候选曝光优先级”，避免论文中把它描述为稳定分配机制。

### 7.5 加强多 seed 与最终表格

当前多数实验更像单轮开发验证。论文最终应至少跑：

- 3 到 5 个随机种子。
- 固定 scenario cache。
- stage2 baseline、stage2_dynamic_priority、stage3 hybrid、stage4 storage/downlink。
- 关键消融：no future task、no temporal window features、GRU state-history、no dynamic response pressure、no downlink-aware edge value、post-hoc dynamic downlink priority、heuristic vs hybrid。

最终报告均值和标准差，避免单次结果波动影响结论。

当前已提供论文核心实验总入口：

```bash
python3 scripts/run_paper_experiment_suite.py \
  --plan paper_core \
  --suite_name das_paper_core_v033 \
  --train_iters 50 \
  --val_episodes 10 \
  --eval_workers 10 \
  --eval_device cpu \
  --train_env_workers 16 \
  --device cuda:0 \
  --continue_on_error \
  --no_progress
```

`paper_core` 覆盖阶段推进、V0.33 时序对比、Stage-2 响应/交付机制消融和 Stage-4 模型组件消融，并跳过当前参数下等价的 Stage-2 重复标签；`paper_full` 会额外加入历史标签和探索性诊断消融，耗时明显更长。运行完成后优先读取对应 suite 下的 `summary.md`、`summary.csv` 和 `paper_experiment_plan.md`。

### 7.6 12 星任务翻倍压力测试

为了验证方法在更高负载下是否仍然稳定，新增一个聚焦压力测试计划 `stress_12sat_double_tasks`。该计划自动使用：

- `n_satellites=12`
- `n_routine=1200`
- `n_dynamic=300`
- `eval_max_steps=12000`
- `scenario_cache_dir=runs/scenario_cache/das_cva_stress_12sat_double_seed42`
- `vtw_cache_dir=runs/scenario_cache/das_cva_stress_12sat_double_seed42/vtw_cache`
- `n_ground_stations=4`

理论上应先预生成压力场景和 VTW cache，再让后续训练/评估都从该 cache 采样。这样可以保证不同实验共享同一组 train/eval 场景，避免一边生成一边训练造成额外耗时和对照不公平。

先生成压力环境：

```bash
python precompute_scenarios.py \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --n_satellites 12 \
  --n_train_scenarios 200 \
  --n_eval_scenarios 10 \
  --n_routine 1200 \
  --n_dynamic 300 \
  --n_ground_stations 4 \
  --curriculum_stages 300:75,600:150,900:225,1200:300 \
  --vtw_time_step_s 60 \
  --vtw_workers 12 \
  --out_dir runs/scenario_cache/das_cva_stress_12sat_double_seed42
```

然后运行压力测试。其中地面站数量保持 4 个不变，是为了刻意保留共享下传瓶颈，检验下传感知边价值和存储压力机制在高压场景下是否仍有作用：

```bash
python3 scripts/run_paper_experiment_suite.py \
  --plan stress_12sat_double_tasks \
  --suite_name das_stress_12sat_double_tasks_v033 \
  --train_iters 50 \
  --val_episodes 10 \
  --eval_workers 10 \
  --eval_device cpu \
  --train_env_workers 16 \
  --device cuda:0 \
  --continue_on_error \
  --no_progress
```

该压力测试默认只跑四组关键对照：`stage2_candidate_owner_repair`、`stage4_storage_pressure`、`abl_no_storage_pressure` 和 `abl_stage2_no_downlink_aware_edge_value`。重点观察 `avg_downlink_queue_s`、`dynamic_task_downlink_queue_block_rate`、`avg_dynamic_response_s`、`n_storage_expired_drops` 和 `eval_wall_time_s`。

## 8. 当前结论

当前项目已经完成了从兼容层规则调度到 DAS-CVA-MAPPO 动态动作集策略的主体改造，并建立了较完整的阶段实验、消融、指标诊断和 train/eval 一致性保护。项目的主要优势是方法结构清晰、实验可诊断、动态任务问题被拆解得比较细；主要不足是动态任务 raw completion 仍不够强、stale owner 偏高、downlink queue 对响应时间影响大、评估成本仍偏高。

从论文方法角度看，后续应把主张从“候选筛选优化”提升为“响应感知、交付感知、时序感知的动态动作集约束调度”。V0.30 已把 downlink finish time 前置到候选评分中，V0.31 已把 response budget 前置到模型输入中，V0.32 已加入未来窗口时序摘要并提供 GRU state-history 对比版本，V0.33 已把时序模块进一步改为早交付导向。下一步应优先跑 V0.33 四组短验证，确认早交付特征能否在保住下传闭环收益的同时降低动态响应时间。
