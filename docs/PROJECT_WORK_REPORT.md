# 当前项目工作报告

更新时间：2026-07-18

## 1. 项目定位

当前分支是 `DAS-CVA-MAPPO` 的研发分支，目标是面向多卫星、多地面站、动态任务到达、星上存储、数传和星间转移约束的动态任务调度问题，构建一个可用于论文实验的 MAPPO 强化学习调度方法。

项目已经从早期 `cva_mappo_v2` 的固定槽位、规则候选分配路线，逐步转向 `das_cva_mappo` 主线。当前推荐的论文方法中心是 `das_cva_mappo/`，而 `cva_mappo_v2/` 主要作为候选生成、环境封装和兼容层继续被使用。

当前主线版本为 `DAS-CVA-MAPPO V0.29.0`。

## 2. 已完成的主要工作

### 2.1 项目清理与主线收敛

已完成历史代码清理，保留与当前方法有关的模块：

- `das_cva_mappo/`：DAS-CVA-MAPPO 主方法，包括动作集感知 actor、候选 scorer、特征构建、环境 adapter、rollout buffer 和训练入口。
- `cva_mappo_v2/`：兼容层，负责候选生成、任务 owner 分配、可执行性判断和部分环境逻辑。
- `envs/`：单星与多星调度环境。
- `scripts/run_stage_ablation_suite.py`：阶段化实验与消融汇总脚本。
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
- scorer feature 中加入任务动态性、等待压力、可执行性和未来窗口相关信息。

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

最近一轮 V0.29 的重点已经从“是否能完成动态任务”推进到“完成后是否能更快下传”，这是因为实验中 `avg_downlink_queue_s` 对 `avg_dynamic_response_s` 的贡献很大。

### 2.7 训练与评估一致性

已经修复了一个对论文结果很关键的问题：早期 evaluation 默认启用了 eval-only repair/rescue，而 training 没有完全相同的处理。这会让评估结果带有规则后处理增益，不能严格代表策略本身。

当前状态：

- eval 默认走与 train 一致的动作处理路径。
- 旧的 eval repair 只保留在 `--eval_use_repair` 诊断开关后。
- 增加静态回归测试，防止 eval 默认路径再次偏离 train。

这提高了实验结论的可信度。

### 2.8 并行训练、评估与性能诊断

已完成：

- `--train_env_workers` 并行 rollout。
- 默认训练设备为 `cuda:0`。
- 默认评估设备回到 `cpu`，避免 CUDA eval 在当前环境下与论文迭代路线冲突。
- 支持 CPU 多进程评估。
- 支持单进程 batched CUDA/MPS eval，但当前建议后续统一用 CPU eval。
- 增加 `--eval_profile`，可记录 eval wall time、env step、actor forward、feature build 等耗时。
- 增加 all-idle fast path 和低层环境 fast step，减少高 idle 场景下的重复 Python 计算。

从 profiling 结果看，eval 时间长的主要瓶颈不在 GPU 前向，而在 Python 环境步进、候选检查、任务/数传状态推进等模拟逻辑，尤其 `eval_env_step_time_s` 占比极高。

### 2.9 实验脚本与结果汇总

已完成阶段化实验脚本：

- 顺序运行 stage1 到 stage4。
- 支持只运行指定阶段或消融。
- 默认 `train_iters=50`、`eval_episodes=10`、`eval_workers=24`、`train_env_workers=16`、训练设备 `cuda:0`、评估设备 `cpu`。
- 自动生成 `summary.csv` 和 markdown 汇总表。
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

V0.29 已加入 dynamic-priority downlink replanning，但还需要实验确认它能稳定降低：

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

第四，观测-存储-下传耦合的调度评价。V0.29 已开始把 downlink queue 纳入动态响应优化，通过 dynamic-priority downlink replanning 让动态图像在未开始的数传计划中获得更高优先级。论文中可以强调方法不只优化观测覆盖率，而是优化从任务到达到数据交付的端到端响应。

第五，训练评估一致的约束决策流程。当前 eval 默认不再启用 train 中没有的 repair 逻辑，保证论文结果反映策略和候选机制本身，而不是评估阶段额外规则后处理。这一点虽然不是算法创新，但能显著增强实验可信度。

### 6.2 还需要补强的机制创新

为了让方法不只停留在“已有实现加权重”，建议后续优先补三个机制。

第一，downlink-aware edge value。当前候选评分主要面向观测可行性和动态等待压力，downlink priority 更多发生在观测之后。更强的版本应在候选评分阶段估计：

- earliest downlink start。
- earliest downlink finish。
- ground-station queue delay。
- onboard storage risk。
- relay usefulness。
- dynamic delivery deadline margin。

这样 CVA scorer 预测的是“观测并交付”的边价值，而不是只预测“观测”的边价值。该点最容易提升论文创新性，因为它把卫星观测调度和地面站下传资源真正耦合起来。

第二，response-budget-aware reward 或 critic feature。动态任务可以定义剩余响应预算：

```text
response_budget = dynamic_response_target_s - (current_time_s - arrival_time_s)
```

策略和 scorer 都可以使用该预算作为特征。奖励中也可以对超出响应目标的动态任务递增惩罚，而不是只在完成后统计 `avg_dynamic_response_s`。这样论文可以说明方法是 response-aware，而不是事后报告响应时间。

第三，候选 exposure 的可解释约束。对于每个动态任务，记录它从到达到完成之间被多少个卫星看到、看到时是否当前可执行、是否被 future slot 挤出、是否被 routine 下传阻塞。这可以形成一组可解释诊断指标，让方法改进与动态任务表现之间有因果链条。

### 6.3 论文实验需要证明的点

为了让创新点有说服力，实验表格应当围绕“每个机制解决一个结构性问题”设计，而不是只放最终 reward。

建议至少保留以下对比：

- fixed-slot MAPPO 或 v2 compatibility runner：证明动态动作集策略必要。
- DAS action-set actor without set context：证明动作集合上下文有用。
- heuristic CVA vs learned/hybrid CVA：证明学习型候选价值有增益。
- no future task execution：证明时序可执行性表示有用。
- open future macro：证明无约束未来宏动作会损害动态任务。
- no dynamic response pressure：证明响应感知候选排序有效。
- no dynamic downlink priority：证明端到端交付优化有效。

对应指标应包括：

- 总体：`total_reward`、`observation_success_rate_raw`。
- 动态：`dynamic_completion_rate_raw`、`dynamic_completion_rate`、`dynamic_feasible_ratio`。
- 响应：`avg_dynamic_response_s`、`avg_downlink_queue_s`。
- 候选：`avg_valid_slots`、`avg_filled_invalid_slots`、`dynamic_current_slot_exposure_rate`。
- 稳定性：`stale_owner_rate`、`owner_churn_rate`、`load_balance_cv`。
- 效率：`eval_wall_time_s`、`eval_steps_per_wall_s`，作为工程复现信息，不作为主要算法指标。

### 6.4 论文叙事建议

论文中不要把方法写成“MAPPO 加规则候选筛选”。更强的叙事是：

1. 动态卫星调度的动作空间是时变、异构、强约束的，固定离散动作头会产生语义错位。
2. 因此提出动态动作集策略，将每个可选调度决策表示为动作实体，由共享策略对动作实体打分。
3. 候选集由 CVA scorer 生成，边价值同时考虑观测质量、任务紧迫性、可执行窗口、存储压力和下传交付压力。
4. 通过 future macro 和 visible-candidate idle advancement，让策略既能利用未来窗口，又不跳过不可见的关键事件。
5. 通过 dynamic-priority downlink replanning，把动态任务目标从“观测完成”扩展为“及时交付”。

这样的叙事比单独强调某个 reward 权重更稳，也更容易解释为什么该方法适合动态任务场景。

### 6.5 不建议作为主要创新点的内容

以下内容可以放在实现细节或实验设置里，但不建议作为主要创新点：

- 单纯把 `--train_env_workers` 调大。
- CUDA 或 CPU eval 的选择。
- 增加更多日志列。
- 某个具体 reward 权重调参。
- 只靠 heuristic priority 提升动态任务。

这些内容对工程有效，但论文创新性较弱。它们应服务于主方法，而不是成为主方法本身。

## 7. 建议的后续路线

### 7.1 优先完成 V0.29 验证

建议先运行当前短验证：

```bash
python3 scripts/run_stage_ablation_suite.py \
  --suite_name das_v029_dynamic_response_iter \
  --only stage2_candidate_owner_repair stage2_dynamic_priority_recovery abl_stage2_no_dynamic_downlink_priority \
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
- `dynamic_current_slot_exposure_rate`
- `dynamic_future_slot_exposure_rate`

如果禁用 dynamic downlink priority 后响应时间明显变差，则 V0.29 可以作为有效改进保留。

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

如果 V0.29 的 downlink priority 收益有限，下一步应考虑：

- 为动态图像设置更强的 downlink deadline 或 priority key。
- 在观测候选评分阶段加入预计 downlink finish time，而不是只在观测后重排。
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
- 关键消融：no future task、no dynamic response pressure、no dynamic downlink priority、heuristic vs hybrid。

最终报告均值和标准差，避免单次结果波动影响结论。

## 8. 当前结论

当前项目已经完成了从兼容层规则调度到 DAS-CVA-MAPPO 动态动作集策略的主体改造，并建立了较完整的阶段实验、消融、指标诊断和 train/eval 一致性保护。项目的主要优势是方法结构清晰、实验可诊断、动态任务问题被拆解得比较细；主要不足是动态任务 raw completion 仍不够强、stale owner 偏高、downlink queue 对响应时间影响大、评估成本仍偏高。

从论文方法角度看，后续应把主张从“候选筛选优化”提升为“响应感知的动态动作集约束调度”。下一步不建议大范围重构，而应沿 V0.29 路线做有针对性的验证：先确认 dynamic downlink priority 是否显著降低响应时间，再决定是否把 downlink finish time 前置到候选评分和 owner 分配中。
