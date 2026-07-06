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
import os
import pickle
import hashlib
from dataclasses import dataclass
from typing import List, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


_GLOBAL_VTW_CACHE = {}


def _vtw_disk_cache_path(cache_dir: str, key) -> str:
    digest = hashlib.sha1(repr(key).encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{digest}.pkl")


def _load_vtw_disk_cache(key):
    cache_dir = os.environ.get("MRL_DMS_VTW_CACHE_DIR", "").strip()
    if not cache_dir:
        return None
    path = _vtw_disk_cache_path(cache_dir, key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _save_vtw_disk_cache(key, windows):
    cache_dir = os.environ.get("MRL_DMS_VTW_CACHE_DIR", "").strip()
    if not cache_dir:
        return
    try:
        os.makedirs(cache_dir, exist_ok=True)
        path = _vtw_disk_cache_path(cache_dir, key)
        tmp_path = f"{path}.{os.getpid()}.tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(windows, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)
    except Exception:
        return

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

        # VTW 缓存: (lat_rounded, lon_rounded, horizon_s_int, step_int) → List[VTW]
        # 同一任务在内循环和评估中被 reset 两次，缓存后第二次直接返回
        self._vtw_cache: dict = {}
        self._sat_cache_key = (
            getattr(sat_config, "name", ""),
            round(float(self.sma_km), 6),
            round(float(self.ecc), 8),
            round(float(self.inc_deg), 6),
            round(float(self.raan_deg), 6),
            round(float(self.argp_deg), 6),
            round(float(self.ma_deg), 6),
            round(float(self.max_roll_deg), 6),
            round(float(self.fov_deg), 6),
        )

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
        # 命中缓存则直接返回（同任务在内循环和评估中重复调用）
        cache_key = (round(target_lat, 4), round(target_lon, 4),
                     int(horizon_seconds), int(time_step_s))
        if cache_key in self._vtw_cache:
            return self._vtw_cache[cache_key]
        global_key = (self._sat_cache_key, *cache_key)
        if global_key in _GLOBAL_VTW_CACHE:
            windows = _GLOBAL_VTW_CACHE[global_key]
            self._vtw_cache[cache_key] = windows
            return windows
        windows = _load_vtw_disk_cache(global_key)
        if windows is not None:
            self._vtw_cache[cache_key] = windows
            _GLOBAL_VTW_CACHE[global_key] = windows
            return windows

        target_ecef = self._geodetic_to_ecef(target_lat, target_lon)
        windows = self._compute_vtw_vectorized(target_ecef, horizon_seconds, time_step_s)

        self._vtw_cache[cache_key] = windows
        _GLOBAL_VTW_CACHE[global_key] = windows
        _save_vtw_disk_cache(global_key, windows)
        return windows

    def compute_ground_station_vtw(
        self,
        station_lat: float,
        station_lon: float,
        horizon_seconds: float = 86400.0,
        time_step_s: float = 10.0,
        min_elevation_deg: float = 5.0,
    ) -> List[VisibleTimeWindow]:
        """
        计算卫星-基站通信可见窗口。

        与光学任务观测 VTW 不同, 基站下传只要求星地链路视线满足最低仰角,
        不施加成像 FOV、roll 或目标日照约束。
        """
        cache_key = (
            "ground_station",
            round(station_lat, 4),
            round(station_lon, 4),
            int(horizon_seconds),
            int(time_step_s),
            round(float(min_elevation_deg), 3),
        )
        if cache_key in self._vtw_cache:
            return self._vtw_cache[cache_key]
        global_key = (self._sat_cache_key, *cache_key)
        if global_key in _GLOBAL_VTW_CACHE:
            windows = _GLOBAL_VTW_CACHE[global_key]
            self._vtw_cache[cache_key] = windows
            return windows
        windows = _load_vtw_disk_cache(global_key)
        if windows is not None:
            self._vtw_cache[cache_key] = windows
            _GLOBAL_VTW_CACHE[global_key] = windows
            return windows

        station_ecef = self._geodetic_to_ecef(station_lat, station_lon)
        windows = self._compute_ground_station_vtw_vectorized(
            station_ecef,
            horizon_seconds,
            time_step_s,
            min_elevation_deg,
        )
        self._vtw_cache[cache_key] = windows
        _GLOBAL_VTW_CACHE[global_key] = windows
        _save_vtw_disk_cache(global_key, windows)
        return windows

    def _compute_ground_station_vtw_vectorized(
        self,
        station_ecef: np.ndarray,
        horizon_seconds: float,
        time_step_s: float,
        min_elevation_deg: float,
    ) -> List[VisibleTimeWindow]:
        times = np.arange(0.0, horizon_seconds + time_step_s * 0.5, time_step_s)
        n_steps = len(times)
        if self._satrec is not None:
            sat_pos_all = self._propagate_sgp4_batch(times)
        else:
            sat_pos_all = self._propagate_kepler_batch(times)

        to_sat = sat_pos_all - station_ecef[np.newaxis, :]
        slant_range = np.linalg.norm(to_sat, axis=1)
        valid = slant_range > 1e-6
        station_up = station_ecef / np.linalg.norm(station_ecef)
        sin_elev = np.einsum('ij,j->i', to_sat, station_up) / np.where(valid, slant_range, 1.0)
        sin_elev = np.clip(sin_elev, -1.0, 1.0)
        elevation_deg = np.degrees(np.arcsin(sin_elev))

        sat_norm = np.linalg.norm(sat_pos_all, axis=1, keepdims=True)
        nadir = -sat_pos_all / np.where(sat_norm > 1e-6, sat_norm, 1.0)
        sat_to_station = (station_ecef[np.newaxis, :] - sat_pos_all) / np.where(
            valid[:, np.newaxis],
            slant_range[:, np.newaxis],
            1.0,
        )
        cos_nadir = np.einsum('ij,ij->i', nadir, sat_to_station)
        cos_nadir = np.clip(cos_nadir, -1.0, 1.0)
        off_nadir_deg = np.degrees(np.arccos(cos_nadir))

        visible_arr = valid & (elevation_deg >= float(min_elevation_deg))
        vis_int = visible_arr.astype(np.int8)
        transitions = np.diff(vis_int, prepend=0, append=0)
        starts = np.where(transitions == 1)[0]
        ends = np.where(transitions == -1)[0]

        windows = []
        for s, e in zip(starts, ends):
            mid = (s + e) // 2
            windows.append(VisibleTimeWindow(
                start_time=times[s],
                end_time=times[min(e, n_steps - 1)],
                elevation_deg=float(elevation_deg[mid]),
                off_nadir_deg=float(off_nadir_deg[mid]),
            ))
        return windows

    def _compute_vtw_vectorized(
        self,
        target_ecef: np.ndarray,
        horizon_seconds: float,
        time_step_s: float,
    ) -> List[VisibleTimeWindow]:
        """
        向量化 VTW 计算：一次性生成所有时间步的卫星位置，批量判断可见性。
        比逐步 Python while 循环快约 5-10x。
        """
        # 生成所有时间步
        times = np.arange(0.0, horizon_seconds + time_step_s * 0.5, time_step_s)
        n_steps = len(times)

        # 批量传播：根据模式分别调用
        if self._satrec is not None:
            sat_pos_all = self._propagate_sgp4_batch(times)
        else:
            sat_pos_all = self._propagate_kepler_batch(times)

        # 批量可见性判断 (传入 times 以计算太阳光照约束)
        visible_arr, elev_arr, off_nadir_arr = self._check_visibility_batch(
            sat_pos_all, target_ecef, times
        )

        # 用 diff 找窗口边界（0→1 为开始，1→0 为结束）
        vis_int = visible_arr.astype(np.int8)
        transitions = np.diff(vis_int, prepend=0, append=0)
        starts = np.where(transitions == 1)[0]
        ends = np.where(transitions == -1)[0]

        windows = []
        for s, e in zip(starts, ends):
            mid = (s + e) // 2
            windows.append(VisibleTimeWindow(
                start_time=times[s],
                end_time=times[min(e, n_steps - 1)],
                elevation_deg=float(elev_arr[mid]),
                off_nadir_deg=float(off_nadir_arr[mid]),
            ))
        return windows

    def _gmst_rad(self, times: np.ndarray) -> np.ndarray:
        """格林尼治平恒星时 (弧度), 用于 TEME/ECI → ECEF 旋转。times: 相对 epoch 秒数。"""
        jd0, fr0 = jday(
            self.epoch.year, self.epoch.month, self.epoch.day,
            self.epoch.hour, self.epoch.minute, self.epoch.second
        )
        d = (jd0 + fr0) + times / 86400.0 - 2451545.0   # 自 J2000.0 的天数
        return np.radians((280.46061837 + 360.98564736629 * d) % 360.0)

    @staticmethod
    def _teme_to_ecef(pos_teme: np.ndarray, gmst: np.ndarray) -> np.ndarray:
        """绕 z 轴旋转 GMST: ECEF = Rz(GMST)·TEME。pos_teme [N,3], gmst [N]。"""
        cos_g, sin_g = np.cos(gmst), np.sin(gmst)
        x, y, z = pos_teme[:, 0], pos_teme[:, 1], pos_teme[:, 2]
        x_e = cos_g * x + sin_g * y
        y_e = -sin_g * x + cos_g * y
        return np.stack([x_e, y_e, z], axis=1)

    def _propagate_sgp4_batch(self, times: np.ndarray) -> np.ndarray:
        """批量 SGP4 传播，返回 [N, 3] ECEF 位置数组 (已做 TEME→ECEF 旋转)"""
        n = len(times)
        pos_teme = np.zeros((n, 3))
        for i, t in enumerate(times):
            e, r, _ = self._satrec.sgp4(
                self._satrec.jdsatepoch,
                self._satrec.jdsatepochF + t / 86400.0
            )
            if e == 0:
                pos_teme[i] = r
            else:
                # fallback 到开普勒单点 (已是 ECEF), 先存后续不再旋转
                pos_teme[i] = self._propagate_kepler(float(t)).position_ecef
        # TEME → ECEF (关键: SGP4 输出 TEME, 必须绕 z 轴旋转 GMST 才能与地面 ECEF 目标一致)
        gmst = self._gmst_rad(times)
        return self._teme_to_ecef(pos_teme, gmst)

    def _propagate_kepler_batch(self, times: np.ndarray) -> np.ndarray:
        """批量开普勒传播，返回 [N, 3] ECEF 位置数组（全向量化）"""
        n = 2.0 * np.pi / self.period_s
        M = (self.ma_deg * self.DEG2RAD + n * times) % (2.0 * np.pi)

        inc = self.inc_deg * self.DEG2RAD
        raan = self.raan_deg * self.DEG2RAD
        argp = self.argp_deg * self.DEG2RAD
        u = argp + M

        earth_rot = 7.2921159e-5
        raan_eff = raan - earth_rot * times

        r_mag = self.sma_km
        x = r_mag * (np.cos(raan_eff) * np.cos(u) - np.sin(raan_eff) * np.sin(u) * np.cos(inc))
        y = r_mag * (np.sin(raan_eff) * np.cos(u) + np.cos(raan_eff) * np.sin(u) * np.cos(inc))
        z = r_mag * np.sin(u) * np.sin(inc)

        return np.stack([x, y, z], axis=1)  # [N, 3]

    def _sun_ecef_batch(self, times: np.ndarray) -> np.ndarray:
        """
        批量计算太阳单位方向向量 (ECEF), 用于太阳光照约束 (论文 Constraint 4)。

        采用标准低精度天文公式 (NOAA/天文年历), 纯 numpy 实现, 不依赖外部库。
        返回 [N, 3] 的太阳方向单位向量 (ECEF 坐标系)。
        """
        # 各时刻的儒略日 (UT)
        jd0, fr0 = jday(
            self.epoch.year, self.epoch.month, self.epoch.day,
            self.epoch.hour, self.epoch.minute, self.epoch.second
        )
        jd = (jd0 + fr0) + times / 86400.0      # [N]

        # 自 J2000.0 起的儒略世纪/天数
        d = jd - 2451545.0                      # 天数
        # 太阳平黄经与平近点角 (度)
        g = np.radians((357.529 + 0.98560028 * d) % 360.0)   # 平近点角
        q = (280.459 + 0.98564736 * d) % 360.0               # 平黄经
        # 黄道经度 (度)
        L = np.radians((q + 1.915 * np.sin(g) + 0.020 * np.sin(2 * g)) % 360.0)
        # 黄赤交角 (度)
        eps = np.radians(23.439 - 0.00000036 * d)

        # 太阳在地心赤道惯性系 (ECI/GCRF 近似) 的单位方向
        x_eci = np.cos(L)
        y_eci = np.cos(eps) * np.sin(L)
        z_eci = np.sin(eps) * np.sin(L)

        # 格林尼治平恒星时 (GMST, 弧度), 用于 ECI→ECEF 旋转
        gmst = np.radians((280.46061837 + 360.98564736629 * d) % 360.0)   # [N]
        cos_g, sin_g = np.cos(gmst), np.sin(gmst)
        # 绕 z 轴旋转 -GMST: ECEF = Rz(GMST) · ECI
        x_ecef = cos_g * x_eci + sin_g * y_eci
        y_ecef = -sin_g * x_eci + cos_g * y_eci
        z_ecef = z_eci

        sun = np.stack([x_ecef, y_ecef, z_ecef], axis=1)     # [N, 3]
        sun /= np.linalg.norm(sun, axis=1, keepdims=True)
        return sun

    def _check_visibility_batch(
        self,
        sat_pos_all: np.ndarray,
        target_ecef: np.ndarray,
        times: np.ndarray = None,
    ) -> tuple:
        """
        批量可见性判断，返回 (visible[N], elevation_deg[N], off_nadir_deg[N])。
        全程 NumPy 向量化，无 Python 循环。

        论文 Constraint 4 三项几何/光照约束:
          - off-nadir ≤ ±25° roll
          - off-nadir ≤ FOV/2 (45° 视场)
          - 目标处太阳光照 (光学成像需白天)
        """
        # to_sat: [N, 3]
        to_sat = sat_pos_all - target_ecef[np.newaxis, :]
        slant_range = np.linalg.norm(to_sat, axis=1)  # [N]

        # 避免除零
        valid = slant_range > 1e-6

        target_up = target_ecef / np.linalg.norm(target_ecef)  # [3]

        # 仰角
        sin_elev = np.einsum('ij,j->i', to_sat, target_up) / np.where(valid, slant_range, 1.0)
        sin_elev = np.clip(sin_elev, -1.0, 1.0)
        elevation_deg = np.degrees(np.arcsin(sin_elev))

        # off-nadir 角
        sat_norm = np.linalg.norm(sat_pos_all, axis=1, keepdims=True)
        nadir = -sat_pos_all / np.where(sat_norm > 1e-6, sat_norm, 1.0)  # [N, 3]
        slant_range_kp = np.where(valid, slant_range, 1.0)
        sat_to_target = (target_ecef[np.newaxis, :] - sat_pos_all) / slant_range_kp[:, np.newaxis]
        cos_nadir = np.einsum('ij,ij->i', nadir, sat_to_target)
        cos_nadir = np.clip(cos_nadir, -1.0, 1.0)
        off_nadir_deg = np.degrees(np.arccos(cos_nadir))

        # 可见条件: (1) 仰角>0  (2) off-nadir ≤ ±25° roll  (3) off-nadir ≤ FOV/2
        is_visible = (
            valid
            & (elevation_deg > 0.0)
            & (off_nadir_deg <= self.max_roll_deg)
            & (off_nadir_deg <= self.fov_deg / 2.0)
        )

        # (4) 太阳光照约束 (论文 Constraint 4): 目标处太阳高度角 > 阈值
        # 光学成像需目标处于日照. 阈值 0° = 目标在地平线以上受日照(白天)
        if times is not None:
            sun_dir = self._sun_ecef_batch(times)                 # [N, 3] 太阳方向
            sun_elev_sin = np.einsum('ij,j->i', sun_dir, target_up)  # 太阳高度角的 sin
            sunlit = sun_elev_sin > 0.0                           # 太阳在目标地平线以上
            is_visible = is_visible & sunlit

        return is_visible, elevation_deg, off_nadir_deg

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
        #   3) off-nadir 角 <= FOV/2 (45° 视场约束)
        # 注: 太阳光照约束需时间参数, 仅在批量版 _check_visibility_batch 中施加
        is_visible = (
            (elevation_deg > 0.0)
            and (off_nadir_deg <= self.max_roll_deg)
            and (off_nadir_deg <= self.fov_deg / 2.0)
        )

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
