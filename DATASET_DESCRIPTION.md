# MRL-DMS 数据集与实验场景说明

本文档描述当前仓库用于论文复现与多星协同优化实验的数据集、任务生成方式和卫星星座配置。项目采用“常规任务合成生成 + 动态任务 ACLED 候选池/合成回退”的混合任务集，调度周期为 24 小时。

## 1. 数据来源与生成方式

项目中的任务集不是单一固定文件，而是在每个 episode 中按随机种子生成一个调度场景：

- 常规任务由 `data/mission_generator.py` 随机生成。
- 动态任务优先从本地 `DynamicMission/DynamicMission.shp` 采样。
- 若未通过 `--acled_path` 加载 shapefile，则自动生成合成动态任务。
- 任务生成随机种子默认是 `42`，可通过 `--seed` 修改。

本地动态任务候选文件：

| 文件 | 内容 |
|---|---|
| `DynamicMission/DynamicMission.shp` | 动态任务候选点位 |
| `DynamicMission/DynamicMission.dbf` | 动态任务属性表 |
| `DynamicMission/DynamicMission.prj` | WGS84 坐标参考 |

本地 shapefile 统计如下：

| 项目 | 数值 |
|---|---:|
| 动态任务候选记录数 | 1184 |
| 坐标字段 | `lat`, `lon` |
| 属性字段 | `frequency`, `region`, `country`, `location`, `disorder`, `event_type` |
| 纬度范围 | -45.8644 到 64.1461 |
| 经度范围 | -123.1004 到 151.1825 |

主要事件类型分布：

| 事件类型 | 数量 |
|---|---:|
| Protests | 515 |
| Explosions/Remote violence | 421 |
| Battles | 131 |
| Violence against civilians | 51 |
| Strategic developments | 44 |
| Riots | 22 |

主要区域分布：

| 区域 | 数量 |
|---|---:|
| Europe | 427 |
| Middle East | 279 |
| South Asia | 99 |
| North America | 98 |
| South America | 90 |
| Southeast Asia | 56 |
| Northern Africa | 41 |
| East Asia | 33 |
| Eastern Africa | 13 |
| Middle Africa | 10 |

## 2. 任务数量设置

默认任务规模来自 `config.py` 和各实验脚本参数。

| 场景 | 常规任务数 | 动态任务设置 | episode 总任务数 |
|---|---:|---:|---:|
| 论文/正式对比常用设置 | 200 | 每次插入 50 个, 插入 3 次 | 350 |
| `compare_methods.py` 默认 | 200 | 每次插入 50 个, 插入 3 次 | 350 |
| 训练默认任务池 | 100/200/300/400/500 | 每次插入 5/10/50/100 个, 插入 3 次 | 随采样而变 |
| `--fast` 快速测试 | 20 | 每次插入 5 个, 插入 3 次 | 35 |
| 动作空间上限 | - | - | `max_action_dim=800` |

动态任务每天插入 3 次，插入时刻均匀分布在 24 小时规划周期的 10% 到 90% 之间：

| 插入批次 | 到达时刻 | 小时 |
|---|---:|---:|
| 第 1 批 | 8640 s | 2.4 h |
| 第 2 批 | 43200 s | 12.0 h |
| 第 3 批 | 77760 s | 21.6 h |

## 3. 任务基本属性

每个任务由 `Mission` 数据结构表示，核心字段如下：

| 字段 | 含义 |
|---|---|
| `id` | 任务唯一编号 |
| `lat`, `lon` | WGS84 经纬度 |
| `priority` | 任务优先级/奖励代理 |
| `duration_s` | 观测持续时间 |
| `earliest_time_s` | 最早可执行时间 |
| `deadline_s` | 截止时间 |
| `is_dynamic` | 是否为动态任务 |
| `arrival_time_s` | 动态任务到达时刻 |
| `event_type` | ACLED 事件类型 |

常规任务设置：

| 属性 | 取值 |
|---|---|
| 空间范围 | 纬度 [-60, 70], 经度 [-180, 180] |
| 优先级 | 均匀采样 [0, 10] |
| 观测时长 | 均匀采样 10-60 s |
| 最早时间 | 0 s |
| 截止时间 | 86400 s |
| 时间窗口 | 覆盖完整 24 h 调度周期 |

动态任务设置：

| 属性 | 取值 |
|---|---|
| 来源 | ACLED 候选池或合成随机点 |
| 优先级 | [5, 10]，若有 `frequency` 则按频率映射 |
| 观测时长 | 均匀采样 10-30 s |
| 到达时间 | 第 1/2/3 批动态插入时刻 |
| 截止时间 | 到达后 2-8 h 内，且不超过 86400 s |
| 采样策略 | `uniform` 跨区域均匀采样；`hotspot` 按 frequency 加权采样 |

注意：完成率指标默认采用“可观测任务数/物理可达任务数”作为分母，而不是全部生成任务数。`comparison_results.json` 中已记录 `n_total_tasks`、`n_feasible_tasks`、`n_feasible_routine` 和 `n_feasible_dynamic`，用于解释完成率口径。

## 4. 卫星数量与基本情况

项目默认定义 6 颗异构太阳同步轨道卫星。单星论文复现默认使用第 1 颗卫星；多星 MAPPO 对比实验通常通过 `--n_satellites 6` 使用全部 6 颗卫星。

所有卫星共有传感器/机动设置：

| 参数 | 数值 |
|---|---:|
| 最大滚转角 | +/-25 deg |
| 视场角 | 45 deg |
| 姿态机动速度 | 3 deg/s |

轨道参数如下：

| 卫星 | 高度 km | 半长轴 km | 偏心率 | 倾角 deg | RAAN deg | 近地点幅角 deg | 平近点角 deg |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sat1 | 644 | 7015 | 0.00190 | 98.7 | 271 | 5 | 355 |
| Sat2 | 705 | 7076 | 0.00000 | 98.0 | 78 | 296 | 64 |
| Sat3 | 705 | 7076 | 0.00000 | 98.2 | 269 | 108 | 252 |
| Sat4 | 822 | 7193 | 0.00000 | 98.7 | 255 | 288 | 70 |
| Sat5 | 694 | 7065 | 0.00000 | 98.2 | 266 | 102 | 258 |
| Sat6 | 496 | 6867 | 0.00042 | 97.2 | 318 | 185 | 175 |

## 5. 可见时间窗与可观测任务

卫星可观测性由 `data/orbit_utils.py` 中的轨道传播与 VTW 计算决定：

- 轨道传播基于 SGP4/TEME-ECEF 坐标转换。
- 默认 VTW 采样步长为 60 s。
- 任务必须满足可见时间窗、姿态机动时间、截止时间、单任务最多执行一次等约束。
- 多星环境中，任一卫星完成任务后，协同 MAPPO 会同步全局完成状态，避免重复观测。

因此，生成任务数不等于可完成任务数。正式分析时建议同时报告：

| 指标 | 含义 |
|---|---|
| `n_total_tasks` | episode 生成的全部任务数 |
| `n_feasible_tasks` | 至少被当前卫星/星座物理可观测的任务数 |
| `n_feasible_routine` | 可观测常规任务数 |
| `n_feasible_dynamic` | 可观测动态任务数 |
| `n_scheduled` | 实际成功调度完成的任务数 |
| `observation_success_rate` | 基于可观测任务数计算的观测成功率 |

## 6. 推荐论文实验口径

用于论文复现/对比时，建议明确写出如下设置：

- 调度周期：24 h。
- 常规任务：200 个，随机均匀分布。
- 动态任务：每次插入 50 个，每天插入 3 次，共 150 个。
- 总任务数：350 个/episode。
- 动态任务来源：`DynamicMission.shp` ACLED 候选池；未加载时使用合成动态任务。
- 卫星数量：单星 MRL-DMS 使用 1 颗；多星 MAPPO 使用 6 颗异构 SSO 卫星。
- 完成率分母：可观测任务数，而不是全部任务数。

对应命令示例：

```bash
python compare_methods.py \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --n_satellites 6 \
  --n_routine 200 \
  --n_dynamic 50 \
  --eval_episodes 5 \
  --device cpu
```
