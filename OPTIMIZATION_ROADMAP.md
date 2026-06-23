# 多星协同(MAPPO)优化路线图

> 本文档记录 MRL-DMS 多星协同方案的优化方向,供后续逐项落地。
> 创建于 2026-06-21,基于 `compare_methods.py` 的对比结果分析。

---

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
| A1 败者改派 | ✅ 已实现(评估期) | `_resolve_actions` + `eval_mode`;训练期关闭以保信用分配 |
| A2/A3 择优指派 | ✅ 已实现 | 边际价值竞价(优先级+off-nadir 质量),胜者得 |
| B6 负载均衡 tie-break | ✅ 已实现 | 竞价含负载惩罚 `coord_w_load` |

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
