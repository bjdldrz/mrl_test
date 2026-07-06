"""
任务生成器
==========
实现论文 Section 3.2 (Element definition) 和 Section 4.1.1 (Hybrid dataset):
  - 加载 ACLED Shapefile 作为动态任务候选池
  - 生成常规任务 (随机坐标 + 优先级)
  - 按论文采样策略 (均匀/热点) 采样动态任务实例
  - 动态任务按 "每天3次、每次 5/10/50/100" 插入
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import logging
import os

logger = logging.getLogger(__name__)

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False


# -----------------------------------------------------------------------
# 任务数据结构 (论文 Section 3.2)
# -----------------------------------------------------------------------
@dataclass
class Mission:
    """
    单个观测任务, 对应论文 Section 3.2 中的 M_i。
    """
    id: int                        # 唯一标识符 i
    lat: float                     # 空间位置 l_i^a (纬度)
    lon: float                     # 空间位置 l_i^a (经度)
    priority: float                # 奖励/优先级 p_i
    duration_s: float              # 执行时长 d_i (秒)
    max_executions: int = 1        # 可执行次数 x_i (论文 Constraint 9)
    earliest_time_s: float = 0.0   # 最早可执行时间 t_i^a
    deadline_s: float = 86400.0    # 截止时间 t_i^d
    is_dynamic: bool = False       # 是否为动态任务
    arrival_time_s: float = 0.0    # 动态任务的到达时刻
    event_type: str = ""           # ACLED 事件类型 (仅动态任务)

    # 运行时状态 (调度过程中更新)
    is_observed: bool = False      # D_i (论文 Eq.3)
    obs_start_s: float = -1.0     # t_i^s
    obs_end_s: float = -1.0       # t_i^e
    is_downlinked: bool = False    # 图像是否已下传到基站; 关闭基站约束时与 is_observed 等价
    downlink_start_s: float = -1.0
    downlink_end_s: float = -1.0
    ground_station_id: int = -1
    relay_satellite_name: str = ""
    relay_start_s: float = -1.0
    relay_end_s: float = -1.0


# -----------------------------------------------------------------------
# ACLED 数据加载
# -----------------------------------------------------------------------
def load_acled_shapefile(shp_path: str) -> pd.DataFrame:
    """
    加载 DynamicMission.shp, 返回标准化 DataFrame。

    字段映射 (参见 Data_Description.docx):
      Frequency → 事件频率 (用作任务回报代理)
      Latitude/Longitude → WGS84 坐标
      Region, Country, Location → 地理信息
      Disorder, EventType → 冲突分类
    """
    if not HAS_GEOPANDAS:
        raise ImportError("需要安装 geopandas: pip install geopandas")

    if not os.path.exists(shp_path):
        raise FileNotFoundError(f"未找到: {shp_path}")

    gdf = gpd.read_file(shp_path)

    # 字段名标准化 (兼容 shapefile 10字符截断)
    rename_map = {}
    for col in gdf.columns:
        c = col.lower().replace('_', '').replace(' ', '')
        if c.startswith('freq'):
            rename_map[col] = 'frequency'
        elif c.startswith('lat'):
            rename_map[col] = 'lat'
        elif c.startswith('lon') or c.startswith('lng'):
            rename_map[col] = 'lon'
        elif c.startswith('region'):
            rename_map[col] = 'region'
        elif c.startswith('country'):
            rename_map[col] = 'country'
        elif c.startswith('disorder'):
            rename_map[col] = 'disorder'
        elif c.startswith('event'):
            rename_map[col] = 'event_type'

    gdf = gdf.rename(columns=rename_map)

    # 若坐标列缺失, 从 geometry 中提取
    if 'lat' not in gdf.columns:
        gdf['lat'] = gdf.geometry.y
    if 'lon' not in gdf.columns:
        gdf['lon'] = gdf.geometry.x

    logger.info(f"加载 ACLED 数据: {len(gdf)} 条记录")
    return pd.DataFrame(gdf.drop(columns='geometry', errors='ignore'))


# -----------------------------------------------------------------------
# 任务生成器
# -----------------------------------------------------------------------
class MissionGenerator:
    """
    生成论文实验所需的混合任务集 M = M_r ∪ M_d (Eq.1)。
    """

    def __init__(
        self,
        acled_df: Optional[pd.DataFrame] = None,
        seed: int = 42,
    ):
        """
        参数
        ----
        acled_df : pd.DataFrame or None
            ACLED 数据 (由 load_acled_shapefile 返回)。
            若为 None, 将自动生成合成动态任务数据。
        seed : int
            随机种子
        """
        self.rng = np.random.RandomState(seed)
        self.acled_df = acled_df
        self._mission_id_counter = 0

    def _next_id(self) -> int:
        self._mission_id_counter += 1
        return self._mission_id_counter

    # -------------------------------------------------------------------
    # 常规任务生成
    # -------------------------------------------------------------------
    def generate_routine_missions(
        self,
        n_missions: int = 100,
        priority_range: Tuple[float, float] = (0.0, 10.0),
        duration_range_s: Tuple[float, float] = (10.0, 60.0),
        lat_range: Tuple[float, float] = (-60.0, 70.0),
        lon_range: Tuple[float, float] = (-180.0, 180.0),
        horizon_s: float = 86400.0,
    ) -> List[Mission]:
        """
        生成常规任务 M_r (论文 Section 3.2)。

        常规任务坐标随机分布, 优先级在 [0, 10] 内随机赋值,
        时间窗口覆盖整个规划周期。
        """
        missions = []
        for _ in range(n_missions):
            lat = self.rng.uniform(*lat_range)
            lon = self.rng.uniform(*lon_range)
            priority = self.rng.uniform(*priority_range)
            duration = self.rng.uniform(*duration_range_s)

            missions.append(Mission(
                id=self._next_id(),
                lat=lat,
                lon=lon,
                priority=priority,
                duration_s=duration,
                earliest_time_s=0.0,
                deadline_s=horizon_s,
                is_dynamic=False,
            ))

        logger.info(f"生成 {len(missions)} 个常规任务")
        return missions

    # -------------------------------------------------------------------
    # 动态任务采样
    # -------------------------------------------------------------------
    def sample_dynamic_missions(
        self,
        n_missions: int = 10,
        arrival_time_s: float = 0.0,
        horizon_s: float = 86400.0,
        priority_range: Tuple[float, float] = (5.0, 10.0),
        duration_range_s: Tuple[float, float] = (10.0, 30.0),
        sampling_strategy: str = "uniform",
    ) -> List[Mission]:
        """
        从 ACLED 候选池中采样动态任务 (论文 Section 4.1.1)。

        参数
        ----
        n_missions : int
            本次采样的动态任务数量 (论文中的 5/10/50/100)
        arrival_time_s : float
            这批动态任务的到达时刻 (秒)
        sampling_strategy : str
            "uniform" - 跨大洲地理均匀采样
            "hotspot" - 高冲突区域热点采样 (按 Frequency 加权)

        返回
        ----
        List[Mission]
        """
        if self.acled_df is not None and len(self.acled_df) > 0:
            return self._sample_from_acled(
                n_missions, arrival_time_s, horizon_s,
                priority_range, duration_range_s, sampling_strategy,
            )
        else:
            return self._generate_synthetic_dynamic(
                n_missions, arrival_time_s, horizon_s,
                priority_range, duration_range_s,
            )

    def _sample_from_acled(
        self, n, arrival_s, horizon_s, pri_range, dur_range, strategy,
    ) -> List[Mission]:
        """从 ACLED 数据中按指定策略采样"""
        df = self.acled_df

        if strategy == "hotspot" and 'frequency' in df.columns:
            # 按频率加权采样 (高冲突区优先)
            weights = df['frequency'].values.astype(float)
            weights = weights / weights.sum()
        elif strategy == "uniform" and 'region' in df.columns:
            # 跨区域均匀采样: 先均匀选区域, 再在区域内随机选
            regions = df['region'].unique()
            per_region = max(1, n // len(regions))
            sampled_indices = []
            for region in regions:
                region_df = df[df['region'] == region]
                k = min(per_region, len(region_df))
                idx = self.rng.choice(region_df.index, size=k, replace=False)
                sampled_indices.extend(idx)
            # 补足数量
            if len(sampled_indices) < n:
                remaining = n - len(sampled_indices)
                extra = self.rng.choice(
                    df.index, size=remaining, replace=True
                )
                sampled_indices.extend(extra)
            sampled = df.loc[sampled_indices[:n]]
            return self._df_to_missions(
                sampled, arrival_s, horizon_s, pri_range, dur_range
            )
        else:
            weights = None

        n_actual = min(n, len(df))
        indices = self.rng.choice(
            df.index, size=n_actual, replace=False, p=weights
        )
        sampled = df.loc[indices]
        return self._df_to_missions(
            sampled, arrival_s, horizon_s, pri_range, dur_range
        )

    def _df_to_missions(self, df, arrival_s, horizon_s, pri_range, dur_range):
        """将采样到的 DataFrame 行转为 Mission 对象列表"""
        missions = []
        for _, row in df.iterrows():
            # 优先级: 若 ACLED 有 frequency, 做归一化映射到 pri_range
            if 'frequency' in row and pd.notna(row['frequency']):
                freq_norm = min(row['frequency'] / 1000.0, 1.0)
                priority = pri_range[0] + freq_norm * (pri_range[1] - pri_range[0])
            else:
                priority = self.rng.uniform(*pri_range)

            duration = self.rng.uniform(*dur_range)

            # 动态任务的截止时间: 到达后 2-8 小时内 (模拟紧迫性)
            deadline_offset = self.rng.uniform(2 * 3600, 8 * 3600)
            deadline = min(arrival_s + deadline_offset, horizon_s)

            event_type = row.get('event_type', '')

            missions.append(Mission(
                id=self._next_id(),
                lat=float(row['lat']),
                lon=float(row['lon']),
                priority=priority,
                duration_s=duration,
                earliest_time_s=arrival_s,
                deadline_s=deadline,
                is_dynamic=True,
                arrival_time_s=arrival_s,
                event_type=str(event_type),
            ))
        return missions

    def _generate_synthetic_dynamic(
        self, n, arrival_s, horizon_s, pri_range, dur_range,
    ) -> List[Mission]:
        """无 ACLED 数据时, 随机生成合成动态任务"""
        logger.info(f"无 ACLED 数据, 合成 {n} 个动态任务")
        missions = []
        for _ in range(n):
            deadline_offset = self.rng.uniform(2 * 3600, 8 * 3600)
            missions.append(Mission(
                id=self._next_id(),
                lat=self.rng.uniform(-60, 70),
                lon=self.rng.uniform(-180, 180),
                priority=self.rng.uniform(*pri_range),
                duration_s=self.rng.uniform(*dur_range),
                earliest_time_s=arrival_s,
                deadline_s=min(arrival_s + deadline_offset, horizon_s),
                is_dynamic=True,
                arrival_time_s=arrival_s,
            ))
        return missions

    # -------------------------------------------------------------------
    # 完整混合任务场景生成 (一个 episode)
    # -------------------------------------------------------------------
    def generate_episode_missions(
        self,
        n_routine: int = 100,
        n_dynamic_per_insertion: int = 10,
        n_insertions: int = 3,
        horizon_s: float = 86400.0,
        sampling_strategy: str = "uniform",
    ) -> Tuple[List[Mission], List[Tuple[float, List[Mission]]]]:
        """
        生成一个完整调度 episode 的任务集。

        返回
        ----
        routine_missions : List[Mission]
            初始常规任务列表 (规划开始时已知)
        dynamic_schedule : List[Tuple[float, List[Mission]]]
            动态任务插入计划: [(到达时刻, [Mission, ...]), ...]
            按时间排序, 到达时刻等间距分布在规划周期内
        """
        # 生成常规任务
        routine = self.generate_routine_missions(
            n_missions=n_routine,
            horizon_s=horizon_s,
        )

        # 生成动态任务插入计划
        # 论文: 每天插入3次, 等间距分布
        insertion_times = np.linspace(
            horizon_s * 0.1,   # 首次插入在规划开始后 10%
            horizon_s * 0.9,   # 末次插入在规划结束前 10%
            n_insertions,
        )

        dynamic_schedule = []
        for t_arrival in insertion_times:
            dyn_missions = self.sample_dynamic_missions(
                n_missions=n_dynamic_per_insertion,
                arrival_time_s=t_arrival,
                horizon_s=horizon_s,
                sampling_strategy=sampling_strategy,
            )
            dynamic_schedule.append((t_arrival, dyn_missions))

        total_dynamic = sum(len(m) for _, m in dynamic_schedule)
        logger.info(
            f"Episode 任务集: {n_routine} routine + {total_dynamic} dynamic "
            f"({n_insertions} insertions × {n_dynamic_per_insertion})"
        )
        return routine, dynamic_schedule
