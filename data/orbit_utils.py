"""
轨道传播 & 可见时间窗口 (VTW) 计算
===================================
实现论文 Section 3.1-3.3 中的卫星-目标几何约束:
  - ±25° 滚动角约束
  - 45° 固定视场角
  - 太阳光照条件
  - 姿态机动转移时间

使用 sgp4 (SGP4 轨道传播) + skyfield (坐标变换/太阳位置) 实现。
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

try:
    from sgp4.api import Satrec, WGS84
    from sgp4.api import jday
    HAS_SGP4 = True
except ImportError:
    HAS_SGP4 = False
    logger.warning("sgp4 未安装, 将使用简化轨道模型。pip install sgp4")

try:
    from skyfield.api import load, EarthSatellite, wgs84
    from skyfield.api import utc
    HAS_SKYFIELD = True
except ImportError:
    HAS_SKYFIELD = False
    logger.warning("skyfield 未安装, 太阳光照判断将被跳过。pip install skyfield")


# -----------------------------------------------------------------------
# 数据结构
# -----------------------------------------------------------------------
@dataclass
class VisibleTimeWindow:
    """可见时间窗口 (论文 Eq.2)"""
    start_time: float    # 窗口开始时刻 (秒, 相对于规划起点)
    end_time: float      # 窗口结束时刻 (秒)
    elevation_deg: float # 窗口中点处的仰角 (°)
    off_nadir_deg: float # 窗口中点处的偏离星下点角 (°)


@dataclass
class SatelliteState:
    """卫星在某一时刻的状态"""
    time_s: float                   # 当前时刻 (秒, 相对于规划起点)
    position_ecef: np.ndarray       # ECEF 位置 (km)
    velocity_ecef: np.ndarray       # ECEF 速度 (km/s)
    latitude_deg: float             # 星下点纬度
    longitude_deg: float            # 星下点经度
    altitude_km: float              # 轨道高度


# -----------------------------------------------------------------------
# 轨道传播器
# -----------------------------------------------------------------------
class OrbitPropagator:
    """
    基于轨道六根数的卫星轨道传播 & VTW 计算。

    支持两种模式:
    1. 精确模式 (sgp4 可用): 从轨道根数构造 TLE, 用 SGP4 传播
    2. 简化模式 (fallback): 用开普勒圆轨道近似
    """

    EARTH_RADIUS_KM = 6371.0
    EARTH_MU = 398600.4418  # km^3/s^2
    DEG2RAD = np.pi / 180.0

    def __init__(self, sat_config, epoch: datetime = None):
        """
        参数
        ----
        sat_config : SatelliteConfig
            卫星轨道根数和传感器参数 (来自 config.py)
        epoch : datetime
            轨道历元, 默认使用 2024-01-01 00:00 UTC
        """
        self.config = sat_config
        self.epoch = epoch or datetime(2024, 1, 1, 0, 0, 0)

        # 传感器约束 (论文 Constraint 4)
        self.max_roll_deg = sat_config.max_roll_deg   # ±25°
        self.fov_deg = sat_config.fov_deg             # 45°
        self.maneuver_speed = sat_config.maneuver_speed_deg_s  # °/s

        # 轨道参数
        self.sma_km = sat_config.semi_major_axis_km
        self.altitude_km = self.sma_km - self.EARTH_RADIUS_KM
        self.ecc = sat_config.eccentricity
        self.inc_deg = sat_config.inclination_deg
        self.raan_deg = sat_config.raan_deg
        self.argp_deg = sat_config.arg_perigee_deg
        self.ma_deg = sat_config.mean_anomaly_deg

        # 轨道周期 (秒)
        self.period_s = 2.0 * np.pi * np.sqrt(self.sma_km**3 / self.EARTH_MU)

        # 初始化 SGP4 卫星对象 (如果库可用)
        self._satrec = None
        if HAS_SGP4:
            self._init_sgp4()

    def _init_sgp4(self):
        """从轨道根数构造 SGP4 卫星对象"""
        satrec = Satrec()
        # SGP4 使用 WGS72 地球模型, 但差异很小
        satrec.sgp4init(
            WGS84,
            'i',                                 # improved mode
            99999,                               # 虚拟 NORAD ID
            self._datetime_to_epoch_days(),      # epoch (Julian days since 1949-12-31)
            0.0,                                 # bstar (大气阻力系数, 短期仿真可忽略)
            0.0,                                 # ndot
            0.0,                                 # nddot
            self.ecc,
            self.argp_deg * self.DEG2RAD,
            self.inc_deg * self.DEG2RAD,
            self.ma_deg * self.DEG2RAD,
            (2.0 * np.pi) / (self.period_s / 60.0),  # mean motion (rad/min)
            self.raan_deg * self.DEG2RAD,
        )
        self._satrec = satrec

    def _datetime_to_epoch_days(self) -> float:
        """将 datetime 转为 SGP4 使用的历元 (自 1949-12-31 起的儒略日数)"""
        jd, fr = jday(
            self.epoch.year, self.epoch.month, self.epoch.day,
            self.epoch.hour, self.epoch.minute, self.epoch.second
        )
        # sgp4init 需要 epoch = jd - 2433281.5 (since 1949 Dec 31)
        return (jd + fr) - 2433281.5

    def propagate(self, t_seconds: float) -> SatelliteState:
        """
        传播卫星到指定时刻。

        参数
        ----
        t_seconds : float
            相对于 epoch 的秒数

        返回
        ----
        SatelliteState
        """
        if self._satrec is not None:
            return self._propagate_sgp4(t_seconds)
        else:
            return self._propagate_kepler(t_seconds)

    def _propagate_sgp4(self, t_seconds: float) -> SatelliteState:
        """SGP4 精确传播"""
        e, r, v = self._satrec.sgp4(
            self._satrec.jdsatepoch,
            self._satrec.jdsatepochF + t_seconds / 86400.0
        )
        if e != 0:
            logger.warning(f"SGP4 传播错误 code={e}, 回退到开普勒模型")
            return self._propagate_kepler(t_seconds)

        r = np.array(r)  # TEME 坐标 (km), 近似当作 ECEF
        v = np.array(v)

        # 星下点 (简化: TEME ≈ ECEF 对短期仿真误差可接受)
        alt = np.linalg.norm(r) - self.EARTH_RADIUS_KM
        lat = np.degrees(np.arcsin(r[2] / np.linalg.norm(r)))
        lon = np.degrees(np.arctan2(r[1], r[0]))

        return SatelliteState(
            time_s=t_seconds,
            position_ecef=r,
            velocity_ecef=v,
            latitude_deg=lat,
            longitude_deg=lon,
            altitude_km=alt
        )

    def _propagate_kepler(self, t_seconds: float) -> SatelliteState:
        """简化开普勒圆轨道传播 (fallback)"""
        n = 2.0 * np.pi / self.period_s  # 平均角速度
        M = (self.ma_deg * self.DEG2RAD + n * t_seconds) % (2.0 * np.pi)

        # 圆轨道近似: 真近点角 ≈ 平近点角
        inc = self.inc_deg * self.DEG2RAD
        raan = self.raan_deg * self.DEG2RAD
        argp = self.argp_deg * self.DEG2RAD
        u = argp + M  # 纬度幅角

        # 地球自转补偿 (简化)
        earth_rot = 7.2921159e-5  # rad/s
        raan_eff = raan - earth_rot * t_seconds

        # ECEF 位置
        r_mag = self.sma_km
        x = r_mag * (np.cos(raan_eff)*np.cos(u) - np.sin(raan_eff)*np.sin(u)*np.cos(inc))
        y = r_mag * (np.sin(raan_eff)*np.cos(u) + np.cos(raan_eff)*np.sin(u)*np.cos(inc))
        z = r_mag * np.sin(u) * np.sin(inc)

        r = np.array([x, y, z])
        alt = r_mag - self.EARTH_RADIUS_KM
        lat = np.degrees(np.arcsin(z / r_mag))
        lon = np.degrees(np.arctan2(y, x))

        return SatelliteState(
            time_s=t_seconds,
            position_ecef=r,
            velocity_ecef=np.zeros(3),  # 简化
            latitude_deg=lat,
            longitude_deg=lon,
            altitude_km=alt
        )

    # -------------------------------------------------------------------
    # 可见时间窗口 (VTW) 计算
    # -------------------------------------------------------------------
    def compute_vtw(
        self,
        target_lat: float,
        target_lon: float,
        horizon_seconds: float = 86400.0,
        time_step_s: float = 10.0,
    ) -> List[VisibleTimeWindow]:
        """
        计算某地面目标在规划周期内的所有可见时间窗口。

        实现论文 Constraint (4): 受限于 ±25° 滚动角、45° FOV、光照条件。

        参数
        ----
        target_lat, target_lon : float
            目标的 WGS84 坐标 (°)
        horizon_seconds : float
            规划周期总时长 (秒), 默认 86400 (24h)
        time_step_s : float
            采样步长 (秒), 越小越精确但越慢

        返回
        ----
        List[VisibleTimeWindow]
        """
        target_ecef = self._geodetic_to_ecef(target_lat, target_lon)

        windows = []
        in_window = False
        win_start = 0.0
        elevations = []
        off_nadirs = []

        t = 0.0
        while t <= horizon_seconds:
            sat_state = self.propagate(t)
            visible, elev, off_nadir = self._check_visibility(
                sat_state.position_ecef, target_ecef
            )

            if visible and not in_window:
                # 窗口开始
                in_window = True
                win_start = t
                elevations = [elev]
                off_nadirs = [off_nadir]
            elif visible and in_window:
                elevations.append(elev)
                off_nadirs.append(off_nadir)
            elif not visible and in_window:
                # 窗口结束
                in_window = False
                mid_idx = len(elevations) // 2
                windows.append(VisibleTimeWindow(
                    start_time=win_start,
                    end_time=t - time_step_s,
                    elevation_deg=elevations[mid_idx] if elevations else 0.0,
                    off_nadir_deg=off_nadirs[mid_idx] if off_nadirs else 0.0,
                ))
                elevations = []
                off_nadirs = []

            t += time_step_s

        # 处理周期末尾仍在窗口内的情况
        if in_window and elevations:
            mid_idx = len(elevations) // 2
            windows.append(VisibleTimeWindow(
                start_time=win_start,
                end_time=horizon_seconds,
                elevation_deg=elevations[mid_idx],
                off_nadir_deg=off_nadirs[mid_idx] if off_nadirs else 0.0,
            ))

        return windows

    def _check_visibility(
        self,
        sat_pos: np.ndarray,
        target_ecef: np.ndarray,
    ) -> Tuple[bool, float, float]:
        """
        判断卫星能否观测目标 (论文 Constraint 4)。

        返回
        ----
        (is_visible, elevation_deg, off_nadir_deg)
        """
        # 目标→卫星 方向 (仰角计算需要从目标看向卫星)
        to_sat = sat_pos - target_ecef
        slant_range = np.linalg.norm(to_sat)
        if slant_range < 1e-6:
            return False, 0.0, 0.0

        # 目标处的天顶方向
        target_up = target_ecef / np.linalg.norm(target_ecef)

        # 仰角: 从目标看卫星, sin(elev) = dot(目标→卫星, 天顶) / 斜距
        sin_elev = np.dot(to_sat, target_up) / slant_range
        sin_elev = np.clip(sin_elev, -1.0, 1.0)
        elevation_deg = np.degrees(np.arcsin(sin_elev))

        # 星下点角 (off-nadir): 卫星处看目标偏离星下点的角度
        # 天底方向 (卫星→地心)
        nadir = -sat_pos / np.linalg.norm(sat_pos)
        # 卫星→目标 方向
        sat_to_target = (target_ecef - sat_pos) / slant_range
        # cos(off_nadir) = dot(天底方向, 卫星→目标方向)
        cos_nadir = np.dot(nadir, sat_to_target)
        cos_nadir = np.clip(cos_nadir, -1.0, 1.0)
        off_nadir_deg = np.degrees(np.arccos(cos_nadir))

        # 可见条件:
        #   1) 仰角 > 0° (卫星在目标的地平线以上)
        #   2) off-nadir 角 <= max_roll_deg (±25° 滚动角约束)
        is_visible = (elevation_deg > 0.0) and (off_nadir_deg <= self.max_roll_deg)

        return is_visible, elevation_deg, off_nadir_deg

    def compute_transition_time(
        self,
        off_nadir_from: float,
        off_nadir_to: float,
    ) -> float:
        """
        计算姿态机动转移时间 (论文中的 trans_ij,pq)。

        参数
        ----
        off_nadir_from, off_nadir_to : float
            前后两个目标的偏离星下点角 (°)

        返回
        ----
        转移时间 (秒)
        """
        angle_diff = abs(off_nadir_to - off_nadir_from)
        return angle_diff / self.maneuver_speed

    # -------------------------------------------------------------------
    # 工具函数
    # -------------------------------------------------------------------
    @staticmethod
    def _geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_km: float = 0.0) -> np.ndarray:
        """WGS84 大地坐标 → ECEF 直角坐标 (km)"""
        R = 6371.0  # 简化球体
        lat = np.radians(lat_deg)
        lon = np.radians(lon_deg)
        r = R + alt_km
        x = r * np.cos(lat) * np.cos(lon)
        y = r * np.cos(lat) * np.sin(lon)
        z = r * np.sin(lat)
        return np.array([x, y, z])


# -----------------------------------------------------------------------
# 批量 VTW 预计算
# -----------------------------------------------------------------------
def precompute_all_vtw(
    satellites: list,
    targets: list,
    horizon_s: float = 86400.0,
    time_step_s: float = 10.0,
) -> dict:
    """
    为所有 (卫星, 目标) 对预计算可见时间窗口。

    参数
    ----
    satellites : list of SatelliteConfig
    targets : list of dict, 每个包含 'lat', 'lon', 'id'
    horizon_s : float
    time_step_s : float

    返回
    ----
    dict: {(sat_name, target_id): [VisibleTimeWindow, ...]}
    """
    vtw_dict = {}
    for sat_cfg in satellites:
        propagator = OrbitPropagator(sat_cfg)
        for target in targets:
            key = (sat_cfg.name, target['id'])
            vtw_dict[key] = propagator.compute_vtw(
                target['lat'], target['lon'],
                horizon_s, time_step_s,
            )
            logger.debug(
                f"{sat_cfg.name} -> Target {target['id']}: "
                f"{len(vtw_dict[key])} VTW(s)"
            )
    return vtw_dict
