"""
卫星任务调度环境
================
实现论文 Section 3.4 的 MDP 〈S, A, R, P, γ〉:
  - State:  任务观测状态 s_I × 卫星运行状态 s_Sat (Eq.11-12)
  - Action: 自适应动作空间 + 掩码 (Eq.13-14, Fig.3)
  - Reward: 三部分自适应奖励 R = R_p + R_t + R_d (Eq.15-20)
  - Transition: 基于 VTW 和姿态机动约束的状态转移

封装为 Gymnasium 标准接口, 可直接对接 RL 训练循环。
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import logging

from data.mission_generator import Mission
from data.orbit_utils import OrbitPropagator, VisibleTimeWindow

logger = logging.getLogger(__name__)


@dataclass
class ScheduleRecord:
    """一次观测的调度记录"""
    mission_id: int
    satellite_name: str
    obs_start_s: float
    obs_end_s: float
    reward: float
    off_nadir_deg: float = 0.0   # 观测时的偏离星下点角(图像质量, 越小越好)
    is_dynamic: bool = False     # 是否动态任务
    earliest_time_s: float = 0.0 # 任务可用/到达时间(算响应延迟用)
    downlink_start_s: float = 0.0
    downlink_end_s: float = 0.0
    ground_station_id: int = -1
    storage_start_s: float = 0.0
    storage_release_s: float = 0.0
    storage_release_reason: str = "none"
    relay_satellite_name: str = ""
    relay_start_s: float = -1.0
    relay_end_s: float = -1.0


@dataclass
class StorageRecord:
    """一张图片在某颗卫星星上存储中的占用记录"""
    mission_id: int
    satellite_name: str
    storage_start_s: float
    storage_release_s: float
    release_reason: str
    source_satellite_name: str = ""


class SatelliteSchedulingEnv(gym.Env):
    """
    单星混合任务调度环境。

    一个 episode = 一颗卫星在 24h 内的任务调度过程。
    多星场景通过并行运行多个环境实例 + 共享经验池实现。
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        satellite_config,
        max_action_dim: int = 800,
        horizon_s: float = 86400.0,
        reward_config=None,
        precomputed_vtw: Optional[Dict] = None,
        vtw_time_step_s: float = 120.0,
        n_ground_stations: int = 0,
        downlink_time_s: float = 0.0,
        ground_station_configs: Optional[List[Any]] = None,
        satellite_storage_capacity: int = 0,
    ):
        """
        参数
        ----
        satellite_config : SatelliteConfig
        max_action_dim : int
            最大动作空间维度 A_max (论文 Section 3.4)
        horizon_s : float
            规划周期 (秒)
        reward_config : RewardConfig
            奖励函数权重
        precomputed_vtw : dict or None
            预计算的 VTW, key=(sat_name, mission_id), value=List[VTW]
        """
        super().__init__()

        self.sat_config = satellite_config
        self.propagator = OrbitPropagator(satellite_config)
        self.max_action_dim = max_action_dim
        self.horizon_s = horizon_s

        # 奖励权重
        if reward_config is None:
            from config import RewardConfig
            reward_config = RewardConfig()
        self.rw_cfg = reward_config

        # 预计算的 VTW (可选, 提高训练速度)
        self.precomputed_vtw = precomputed_vtw or {}
        self.vtw_time_step_s = vtw_time_step_s
        self.ground_station_configs = self._build_ground_station_configs(
            n_ground_stations,
            ground_station_configs,
        )
        self.n_ground_stations = len(self.ground_station_configs)
        self.downlink_time_s = max(0.0, float(downlink_time_s or 0.0))
        self._ground_station_available_s: List[float] = [0.0] * self.n_ground_stations
        self.ground_station_vtw: Dict[int, List[VisibleTimeWindow]] = {}
        self.satellite_storage_capacity = max(0, int(satellite_storage_capacity or 0))

        # ----- 状态空间 (论文 Eq.11-12) -----
        # 每个任务的状态向量: [obs_status, w_start, w_end, t_obs_start, t_obs_end, priority, is_dynamic]
        # 维度 = max_action_dim * 7 + 卫星状态 4
        self._mission_feat_dim = 7
        self._sat_feat_dim = 4  # [current_time_norm, lat_norm, lon_norm, status]
        obs_dim = self.max_action_dim * self._mission_feat_dim + self._sat_feat_dim
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # ----- 动作空间 (论文 Eq.13-14) -----
        # 离散动作: 0 ~ max_action_dim-1 对应各任务槽位
        # 额外加一个 "idle" 动作 (跳过当前时刻不做观测)
        self.action_space = spaces.Discrete(self.max_action_dim + 1)
        self.IDLE_ACTION = self.max_action_dim

        # ----- 运行时状态 -----
        self.current_time_s = 0.0
        self.missions: List[Optional[Mission]] = []      # 当前已知的所有任务
        self.mission_vtw: Dict[int, List[VisibleTimeWindow]] = {}  # VTW 缓存
        self.schedule_log: List[ScheduleRecord] = []
        self.storage_log: List[StorageRecord] = []
        self._last_off_nadir_deg = 0.0  # 跟踪上一次观测的 off-nadir 角

        # 动态任务插入队列
        self._dynamic_queue: List[Tuple[float, List[Mission]]] = []
        self._n_routine = 0

    @property
    def downlink_required(self) -> bool:
        return self.n_ground_stations > 0 and self.downlink_time_s > 0.0

    @property
    def storage_limited(self) -> bool:
        return self.satellite_storage_capacity > 0

    def set_ground_station_state(self, availability: List[float]):
        """让多星环境共享同一组基站可用时间。"""
        self._ground_station_available_s = availability
        self.n_ground_stations = len(availability)

    @staticmethod
    def _build_ground_station_configs(n_ground_stations: int, ground_station_configs=None) -> List[Any]:
        n = max(0, int(n_ground_stations or 0))
        if n <= 0:
            return []
        if ground_station_configs is None:
            try:
                from config import DEFAULT_GROUND_STATIONS, GroundStationConfig
                base = list(DEFAULT_GROUND_STATIONS)
            except Exception:
                from dataclasses import make_dataclass
                GroundStationConfig = make_dataclass(
                    "GroundStationConfig",
                    [("name", str), ("lat", float), ("lon", float), ("min_elevation_deg", float)],
                )
                base = []
            if not base:
                base = [
                    GroundStationConfig(f"GS{i+1}", 0.0, -180.0 + 360.0 * i / max(n, 1), 5.0)
                    for i in range(n)
                ]
            stations = list(base[:n])
            while len(stations) < n:
                idx = len(stations)
                stations.append(GroundStationConfig(
                    f"GS_auto_{idx + 1}",
                    0.0,
                    -180.0 + 360.0 * idx / max(n, 1),
                    5.0,
                ))
            return stations
        stations = list(ground_station_configs[:n])
        if len(stations) < n:
            try:
                from config import GroundStationConfig
            except Exception:
                from dataclasses import make_dataclass
                GroundStationConfig = make_dataclass(
                    "GroundStationConfig",
                    [("name", str), ("lat", float), ("lon", float), ("min_elevation_deg", float)],
                )
            while len(stations) < n:
                idx = len(stations)
                stations.append(GroundStationConfig(
                    f"GS_auto_{idx + 1}",
                    0.0,
                    -180.0 + 360.0 * idx / max(n, 1),
                    5.0,
                ))
        return stations

    def _mission_completed(self, mission: Mission) -> bool:
        if mission is None:
            return False
        return bool(mission.is_downlinked) if self.downlink_required else bool(mission.is_observed)

    # ===================================================================
    # Gymnasium 接口
    # ===================================================================
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        重置环境, 开始新的调度 episode。

        options 字典应包含:
          "routine_missions": List[Mission]
          "dynamic_schedule": List[Tuple[float, List[Mission]]]
        """
        super().reset(seed=seed)
        self.current_time_s = 0.0
        self.schedule_log = []
        self.storage_log = []
        self.mission_vtw = {}
        self._last_off_nadir_deg = 0.0
        if self.n_ground_stations > 0:
            self._ground_station_available_s[:] = [0.0] * self.n_ground_stations
        self._compute_ground_station_vtws()

        options = options or {}
        routine_missions = options.get("routine_missions", [])
        dynamic_schedule = options.get("dynamic_schedule", [])

        # 初始化任务槽位 (论文 Fig.3)
        self._n_routine = len(routine_missions)
        self.missions = [None] * self.max_action_dim

        # 填充常规任务到前 N_routine 个槽位
        for i, m in enumerate(routine_missions):
            if i < self.max_action_dim:
                self.missions[i] = m

        # 动态任务队列 (按到达时间排序)
        self._dynamic_queue = sorted(dynamic_schedule, key=lambda x: x[0])
        self._next_dynamic_slot = self._n_routine  # 下一个可用的动态槽位

        # 预计算 VTW (常规任务)
        self._compute_vtw_for_missions(routine_missions)

        obs = self._build_observation()
        info = {"action_mask": self._build_action_mask()}
        return obs, info

    def step(
        self,
        action: int,
        build_observation: bool = True,
        check_done: bool = True,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行一步调度决策。

        参数
        ----
        action : int
            选择的任务槽位 (0 ~ max_action_dim-1) 或 IDLE_ACTION

        返回
        ----
        obs, reward, terminated, truncated, info
        """
        reward = 0.0
        info = {}

        # 1) 检查并插入已到达的动态任务
        self._insert_arrived_dynamic_missions()

        # 2) 执行动作
        if action == self.IDLE_ACTION:
            # 空闲: 推进到下一次真正可能执行观测的事件时刻。
            # 旧逻辑会跳到任意 VTW 起点, 即使任务未到达、deadline/持续时间/
            # 姿态转移/存储约束已经使该窗口不可执行, 会制造大量 only-idle 状态。
            next_event_t = self._next_idle_event_time()
            reward = self.rw_cfg.penalty_idle
            self.current_time_s = max(self.current_time_s + 1.0, next_event_t)
        elif 0 <= action < self.max_action_dim:
            mission = self.missions[action]
            if mission is None or mission.is_observed:
                # 无效动作 (空槽位或已完成任务)
                reward = self.rw_cfg.penalty_invalid
                self.current_time_s += 10.0
            else:
                # 尝试调度该任务
                reward, success = self._execute_observation(mission)
                if not success:
                    reward = self.rw_cfg.penalty_invalid
                    self.current_time_s += 10.0
        else:
            reward = self.rw_cfg.penalty_invalid
            self.current_time_s += 10.0

        # 3) 判断终止条件
        terminated = False
        truncated = False

        if self.current_time_s >= self.horizon_s:
            truncated = True  # 规划周期结束

        if check_done and self._all_missions_done():
            terminated = True  # 所有可行任务已完成

        # 4) 构建下一步状态
        info["current_time_s"] = self.current_time_s
        if build_observation:
            obs = self._build_observation()
            info["action_mask"] = self._build_action_mask()
            info["schedule_log"] = self.schedule_log
        else:
            obs = np.zeros(0, dtype=np.float32)

        return obs, reward, terminated, truncated, info

    def _next_idle_event_time(self) -> float:
        """Return the next useful time after an idle action.

        The returned event is the earliest known time where either a currently
        loaded mission can actually start an observation, a dynamic batch
        arrives, or storage may be released.  A short fallback keeps progress
        monotonic when no future executable event is known.
        """
        current = float(self.current_time_s)
        candidates = []

        if self._dynamic_queue:
            arrival_t = float(self._dynamic_queue[0][0])
            if arrival_t > current:
                candidates.append(min(arrival_t, self.horizon_s))

        storage_release_t = self._next_storage_release_time(current)
        if storage_release_t is not None and storage_release_t > current:
            candidates.append(min(float(storage_release_t), self.horizon_s))

        if not (self.storage_limited and not self._has_storage_capacity(current)):
            for mission in self.missions:
                obs_start = self._earliest_feasible_observation_start(mission, current)
                if obs_start is not None and obs_start > current:
                    candidates.append(min(float(obs_start), self.horizon_s))

        if candidates:
            return min(candidates)
        return min(current + 60.0, self.horizon_s)

    def _earliest_feasible_observation_start(
        self,
        mission: Optional[Mission],
        from_time_s: Optional[float] = None,
    ) -> Optional[float]:
        """Earliest future start that can satisfy local observation constraints."""
        if mission is None or mission.is_observed:
            return None
        current = self.current_time_s if from_time_s is None else float(from_time_s)
        earliest_from = max(current, float(mission.earliest_time_s))
        if earliest_from > mission.deadline_s:
            return None

        last_obs_end = self.schedule_log[-1].obs_end_s if self.schedule_log else None
        best_start = None
        for vtw in self.mission_vtw.get(mission.id, []):
            latest_start = min(
                float(vtw.end_time),
                float(mission.deadline_s),
                float(self.horizon_s),
            ) - float(mission.duration_s)
            if latest_start < earliest_from:
                continue

            obs_start = max(float(vtw.start_time), earliest_from)
            if last_obs_end is not None:
                transition = self.propagator.compute_transition_time(
                    self._last_off_nadir_deg,
                    vtw.off_nadir_deg,
                )
                obs_start = max(obs_start, float(last_obs_end) + float(transition))

            if not self._has_storage_capacity(obs_start):
                release_t = self._next_storage_release_time(obs_start)
                if release_t is None:
                    continue
                obs_start = max(obs_start, float(release_t))

            if obs_start > latest_start:
                continue
            if self._conflicts_with_schedule(obs_start, obs_start + mission.duration_s):
                continue
            if best_start is None or obs_start < best_start:
                best_start = obs_start
        return best_start

    def _conflicts_with_schedule(self, obs_start: float, obs_end: float) -> bool:
        for record in self.schedule_log:
            if not (obs_end <= record.obs_start_s or obs_start >= record.obs_end_s):
                return True
        return False

    # ===================================================================
    # 动作掩码 (论文 Eq.13-14)
    # ===================================================================
    def _build_action_mask(self) -> np.ndarray:
        """
        构建二值掩码向量 M_t (论文 Eq.13)。
        Valid(a_i | s_t) = 1 当且仅当三条判据同时满足:
          (1) 目标在 VTW 内可见
          (2) 姿态机动时间可行 (Constraint 8)
          (3) 无先前执行冲突: 未执行过 + 与已调度任务时间窗不重叠 (Constraint 10)
        论文用掩码"先验排除"不可行动作 (Eq.14), 而非事后惩罚。
        """
        mask = np.zeros(self.max_action_dim + 1, dtype=np.float32)
        mask[self.IDLE_ACTION] = 1.0  # idle 动作始终可用

        if not self._has_storage_capacity(self.current_time_s):
            return mask

        # 上一次观测的结束时刻与姿态角 (用于机动时间判据)
        last_obs_end = self.schedule_log[-1].obs_end_s if self.schedule_log else None

        for i in range(self.max_action_dim):
            m = self.missions[i]
            if m is None or m.is_observed:
                continue  # 判据(3): 已执行过
            if m.earliest_time_s > self.current_time_s:
                continue  # 动态任务尚未到达
            if self.current_time_s > m.deadline_s:
                continue  # 已过截止时间

            # 判据(1): 找到当前可用的 VTW
            usable_vtw = None
            for vtw in self.mission_vtw.get(m.id, []):
                if vtw.start_time <= self.current_time_s <= vtw.end_time - m.duration_s:
                    usable_vtw = vtw
                    break
            if usable_vtw is None:
                continue

            # 判据(2): 姿态机动时间可行 (Constraint 8)
            obs_start = self.current_time_s
            if last_obs_end is not None:
                transition = self.propagator.compute_transition_time(
                    self._last_off_nadir_deg, usable_vtw.off_nadir_deg
                )
                earliest_feasible = last_obs_end + transition
                if obs_start < earliest_feasible:
                    obs_start = earliest_feasible
                # 机动顺延后仍须落在 VTW 与截止时间内
                if obs_start > usable_vtw.end_time - m.duration_s:
                    continue
                if obs_start + m.duration_s > m.deadline_s:
                    continue
            obs_end = obs_start + m.duration_s

            # 判据(3): 与已调度任务时间窗不重叠 (Constraint 10)
            if self._conflicts_with_schedule(obs_start, obs_end):
                continue

            mask[i] = 1.0

        return mask

    # ===================================================================
    # 状态构建 (论文 Eq.11-12)
    # ===================================================================
    def _build_observation(self) -> np.ndarray:
        """
        构建状态向量 S = s_I × s_Sat。

        s_I: 每个任务的 [obs_status, w_start, w_end, t_obs_start, t_obs_end, priority, is_dynamic]
        s_Sat: [current_time_norm, lat_norm, lon_norm, status]
        """
        mission_feats = np.zeros(
            (self.max_action_dim, self._mission_feat_dim), dtype=np.float32
        )

        for i in range(self.max_action_dim):
            m = self.missions[i]
            if m is None:
                continue

            # 观测/交付状态编码: 0=未观测, 0.5=观测中, 0.75=待/未下传, 1=已完成
            if self._mission_completed(m):
                obs_status = 1.0
            elif m.is_observed:
                obs_status = 0.75
            elif m.obs_start_s > 0:
                obs_status = 0.5
            else:
                obs_status = 0.0

            # 下一个可见窗口
            w_start, w_end = self._get_next_vtw_times(m.id)

            mission_feats[i] = [
                obs_status,
                w_start / self.horizon_s,       # 归一化
                w_end / self.horizon_s,
                m.obs_start_s / self.horizon_s if m.obs_start_s > 0 else 0.0,
                m.obs_end_s / self.horizon_s if m.obs_end_s > 0 else 0.0,
                m.priority / 10.0,              # 归一化到 [0, 1]
                1.0 if m.is_dynamic else 0.0,
            ]

        # 卫星状态
        sat_state = self.propagator.propagate(self.current_time_s)
        sat_feats = np.array([
            self.current_time_s / self.horizon_s,
            sat_state.latitude_deg / 90.0,
            sat_state.longitude_deg / 180.0,
            0.0,  # 状态标志: 0=空闲, 1=观测中, 2=机动中
        ], dtype=np.float32)

        obs = np.concatenate([mission_feats.flatten(), sat_feats])
        return obs

    # ===================================================================
    # 奖励计算 (论文 Section 3.4, Eq.15-20)
    # ===================================================================
    def compute_reward(
        self, mission: Mission, completion_time_s: float,
        off_nadir_deg: float = 0.0,
    ) -> float:
        """
        计算完成任务的总奖励 R = R_p + R_t + R_d + R_q (论文 Eq.15 + 质量扩展)

        参数
        ----
        off_nadir_deg : float
            观测时的偏离星下点角 (°), 越小图像质量越高
        """
        # (a) 优先级奖励 R_p (Eq.16)
        p_norm = mission.priority / 10.0  # P_t ∈ [0, 1]
        R_p = self.rw_cfg.w_priority * p_norm

        # (b) 时间奖励 R_t (Eq.17)
        t0 = mission.earliest_time_s  # 任务创建时间
        dc = mission.deadline_s       # 截止时间
        tau = completion_time_s       # 完成时间

        if tau <= dc and dc > t0:
            R_t = (dc - tau) / (dc - t0)
        else:
            R_t = 0.0

        # (c) 动态任务奖励 R_d (Eq.18-19)
        if mission.is_dynamic and tau <= dc and dc > t0:
            delta_t = 1.0  # δ_t = 1 (动态任务)
            time_ratio = (tau - t0) / (dc - t0) if dc > t0 else 1.0
            # f=exp(-k·time_ratio): 越早完成 f 越大(与 Eq.17 自洽); k 可配置
            f_decay = np.exp(-self.rw_cfg.dynamic_decay_k * time_ratio)
            R_d = delta_t * self.rw_cfg.w_dynamic * f_decay
        else:
            R_d = 0.0

        # (d) 观测质量奖励 R_q —— 论文外扩展, 仅 w_quality>0 时启用 (论文 Eq.15 仅 Rp+Rt+Rd)
        R_q = 0.0
        if self.rw_cfg.w_quality > 0:
            max_roll = self.sat_config.max_roll_deg
            if max_roll > 0:
                R_q = self.rw_cfg.w_quality * np.cos(
                    off_nadir_deg / max_roll * (np.pi / 2.0)
                )
            else:
                R_q = self.rw_cfg.w_quality

        total_reward = R_p + R_t + R_d + R_q
        return total_reward

    # ===================================================================
    # 内部逻辑
    # ===================================================================
    def _execute_observation(self, mission: Mission) -> Tuple[float, bool]:
        """
        尝试执行一次观测任务。

        返回 (reward, success)
        """
        # 检查截止时间
        if self.current_time_s > mission.deadline_s:
            return self.rw_cfg.penalty_deadline_miss, False

        # 找到当前可用的 VTW
        vtws = self.mission_vtw.get(mission.id, [])
        usable_vtw = None
        for vtw in vtws:
            if vtw.start_time <= self.current_time_s <= vtw.end_time - mission.duration_s:
                usable_vtw = vtw
                break

        if usable_vtw is None:
            return 0.0, False

        # 确认不与已调度任务冲突 (论文 Constraint 10)
        obs_start = self.current_time_s
        obs_end = obs_start + mission.duration_s

        for record in self.schedule_log:
            if not (obs_end <= record.obs_start_s or obs_start >= record.obs_end_s):
                return 0.0, False  # 时间重叠

        # 检查姿态机动时间 (论文 Constraint 8)
        # 使用跟踪的上次观测角度，而非硬编码 0.0
        if self.schedule_log:
            last = self.schedule_log[-1]
            transition = self.propagator.compute_transition_time(
                self._last_off_nadir_deg, usable_vtw.off_nadir_deg
            )
            earliest_feasible = last.obs_end_s + transition
            if obs_start < earliest_feasible:
                obs_start = earliest_feasible
                obs_end = obs_start + mission.duration_s
                if obs_end > usable_vtw.end_time:
                    return 0.0, False
                if obs_end > mission.deadline_s:
                    return self.rw_cfg.penalty_deadline_miss, False

        if not self._has_storage_capacity(obs_start):
            return self.rw_cfg.penalty_invalid, False

        # 执行观测
        mission.is_observed = True
        mission.obs_start_s = obs_start
        mission.obs_end_s = obs_end
        self._last_off_nadir_deg = usable_vtw.off_nadir_deg  # 更新跟踪

        downlink_start, downlink_end, ground_station_id = self._schedule_downlink(
            obs_end,
            latest_end_s=min(self.horizon_s, mission.deadline_s),
        )
        mission.downlink_start_s = downlink_start
        mission.downlink_end_s = downlink_end
        mission.ground_station_id = ground_station_id
        mission.is_downlinked = (
            (not self.downlink_required)
            or (ground_station_id >= 0 and downlink_end <= self.horizon_s and downlink_end <= mission.deadline_s)
        )
        storage_release_s, storage_release_reason = self._storage_release_from_delivery(
            mission,
            obs_end,
            downlink_end,
            ground_station_id,
        )

        completion_time_s = downlink_end if self.downlink_required else obs_end
        if self.downlink_required and not mission.is_downlinked:
            reward = self.rw_cfg.penalty_deadline_miss
        else:
            reward = self.compute_reward(
                mission, completion_time_s, off_nadir_deg=usable_vtw.off_nadir_deg
            )

        self.schedule_log.append(ScheduleRecord(
            mission_id=mission.id,
            satellite_name=self.sat_config.name,
            obs_start_s=obs_start,
            obs_end_s=obs_end,
            reward=reward,
            off_nadir_deg=usable_vtw.off_nadir_deg,
            is_dynamic=mission.is_dynamic,
            earliest_time_s=mission.earliest_time_s,
            downlink_start_s=downlink_start,
            downlink_end_s=downlink_end,
            ground_station_id=ground_station_id,
            storage_start_s=obs_end,
            storage_release_s=storage_release_s,
            storage_release_reason=storage_release_reason,
        ))
        self._set_storage_record(
            mission_id=mission.id,
            storage_start_s=obs_end,
            storage_release_s=storage_release_s,
            release_reason=storage_release_reason,
        )

        # 时间推进到观测结束
        self.current_time_s = obs_end

        return reward, True

    def _storage_release_from_delivery(
        self,
        mission: Mission,
        obs_end_s: float,
        downlink_end_s: float,
        ground_station_id: int,
    ) -> Tuple[float, str]:
        if not self.storage_limited:
            return obs_end_s, "disabled"
        if not self.downlink_required:
            return obs_end_s, "immediate"
        if ground_station_id >= 0 and downlink_end_s >= obs_end_s:
            return downlink_end_s, "downlink"
        return min(float(mission.deadline_s), self.horizon_s), "expired_drop"

    def _onboard_image_count(self, time_s: Optional[float] = None) -> int:
        if not self.storage_limited:
            return 0
        t = self.current_time_s if time_s is None else float(time_s)
        return sum(
            1 for rec in self.storage_log
            if rec.storage_start_s <= t < rec.storage_release_s
        )

    def _has_storage_capacity(self, time_s: Optional[float] = None) -> bool:
        if not self.storage_limited:
            return True
        return self._onboard_image_count(time_s) < self.satellite_storage_capacity

    def _next_storage_release_time(self, time_s: Optional[float] = None) -> Optional[float]:
        if not self.storage_limited:
            return None
        t = self.current_time_s if time_s is None else float(time_s)
        releases = [
            rec.storage_release_s for rec in self.storage_log
            if rec.storage_start_s <= t < rec.storage_release_s
        ]
        return min(releases) if releases else None

    def _set_storage_record(
        self,
        mission_id: int,
        storage_start_s: float,
        storage_release_s: float,
        release_reason: str,
        source_satellite_name: str = "",
    ) -> None:
        if not self.storage_limited:
            return
        if storage_release_s <= storage_start_s:
            self.storage_log = [
                rec for rec in self.storage_log
                if not (rec.mission_id == mission_id and abs(rec.storage_start_s - storage_start_s) < 1e-6)
            ]
            return
        for rec in self.storage_log:
            if rec.mission_id == mission_id and abs(rec.storage_start_s - storage_start_s) < 1e-6:
                rec.storage_release_s = storage_release_s
                rec.release_reason = release_reason
                rec.source_satellite_name = source_satellite_name
                return
        self.storage_log.append(StorageRecord(
            mission_id=mission_id,
            satellite_name=self.sat_config.name,
            storage_start_s=storage_start_s,
            storage_release_s=storage_release_s,
            release_reason=release_reason,
            source_satellite_name=source_satellite_name,
        ))

    def _storage_stats(self) -> Dict[str, float]:
        if not self.storage_limited:
            return {
                "current_onboard_images": 0.0,
                "max_onboard_images": 0.0,
                "avg_onboard_images": 0.0,
                "n_storage_expired_drops": 0.0,
                "n_relay_storage_images": 0.0,
            }
        events = []
        for rec in self.storage_log:
            start = max(0.0, min(float(rec.storage_start_s), self.horizon_s))
            end = max(start, min(float(rec.storage_release_s), self.horizon_s))
            events.append((start, 1))
            events.append((end, -1))
        events.sort(key=lambda item: (item[0], item[1]))
        count = 0
        max_count = 0
        prev_t = 0.0
        area = 0.0
        for t, delta in events:
            area += count * max(0.0, t - prev_t)
            count += delta
            max_count = max(max_count, count)
            prev_t = t
        area += count * max(0.0, self.horizon_s - prev_t)
        return {
            "current_onboard_images": float(self._onboard_image_count(self.current_time_s)),
            "max_onboard_images": float(max_count),
            "avg_onboard_images": float(area / max(self.horizon_s, 1.0)),
            "n_storage_expired_drops": float(sum(1 for rec in self.storage_log if rec.release_reason == "expired_drop")),
            "n_relay_storage_images": float(sum(1 for rec in self.storage_log if rec.release_reason == "relay_downlink")),
        }

    def _find_downlink_slot(
        self,
        obs_end_s: float,
        latest_end_s: Optional[float] = None,
        station_available_s: Optional[List[float]] = None,
    ) -> Tuple[float, float, int]:
        """
        预览最早可行的基站下传窗口, 不修改基站可用时间。
        """
        if not self.downlink_required:
            return obs_end_s, obs_end_s, -1
        if not self._ground_station_available_s:
            self._ground_station_available_s = [0.0] * self.n_ground_stations
        if not self.ground_station_vtw:
            self._compute_ground_station_vtws()
        latest_end = self.horizon_s if latest_end_s is None else min(float(latest_end_s), self.horizon_s)
        availability = station_available_s if station_available_s is not None else self._ground_station_available_s
        best = None
        for station_id in range(self.n_ground_stations):
            ready_time = max(float(obs_end_s), float(availability[station_id]))
            for vtw in self.ground_station_vtw.get(station_id, []):
                if vtw.end_time < ready_time + self.downlink_time_s:
                    continue
                start = max(ready_time, float(vtw.start_time))
                end = start + self.downlink_time_s
                if end <= vtw.end_time and end <= latest_end:
                    cand = (end, start, station_id)
                    if best is None or cand < best:
                        best = cand
                    break
        if best is None:
            return -1.0, -1.0, -1
        downlink_end, downlink_start, station_id = best
        return downlink_start, downlink_end, station_id

    def _schedule_downlink(self, obs_end_s: float, latest_end_s: Optional[float] = None) -> Tuple[float, float, int]:
        """
        将观测图像自动分配给最早可用且对卫星可见的基站下传。

        n_ground_stations=0 或 downlink_time_s=0 时保持旧口径: 观测结束即完成。
        """
        downlink_start, downlink_end, station_id = self._find_downlink_slot(
            obs_end_s,
            latest_end_s=latest_end_s,
        )
        if station_id < 0:
            return downlink_start, downlink_end, station_id
        self._ground_station_available_s[station_id] = downlink_end
        return downlink_start, downlink_end, station_id

    def _compute_ground_station_vtws(self):
        """预计算该卫星对所有基站的通信 VTW, 避免每次下传重复算星地可见性。"""
        self.ground_station_vtw = {}
        if not self.downlink_required:
            return
        for idx, gs in enumerate(self.ground_station_configs):
            self.ground_station_vtw[idx] = self.propagator.compute_ground_station_vtw(
                getattr(gs, "lat"),
                getattr(gs, "lon"),
                self.horizon_s,
                time_step_s=self.vtw_time_step_s,
                min_elevation_deg=getattr(gs, "min_elevation_deg", 5.0),
            )

    def _insert_arrived_dynamic_missions(self):
        """检查并插入已到达的动态任务 (论文 Fig.3 动态槽位更新)"""
        n_discarded = 0
        while self._dynamic_queue:
            arrival_time, missions = self._dynamic_queue[0]
            if arrival_time <= self.current_time_s:
                self._dynamic_queue.pop(0)
                for m in missions:
                    if self._next_dynamic_slot < self.max_action_dim:
                        self.missions[self._next_dynamic_slot] = m
                        self._next_dynamic_slot += 1
                        # 计算新任务的 VTW
                        self._compute_vtw_for_missions([m])
                    else:
                        n_discarded += 1
            else:
                break
        if n_discarded > 0:
            logger.warning(
                f"动态槽位已满 (max_action_dim={self.max_action_dim}), 丢弃 {n_discarded} 个任务; "
                f"这会污染 dynamic_completion_rate 指标，应调大 max_action_dim"
            )

    def _compute_vtw_for_missions(self, missions: List[Mission]):
        """为指定任务计算 VTW 并缓存"""
        for m in missions:
            key = (self.sat_config.name, m.id)
            if key in self.precomputed_vtw:
                self.mission_vtw[m.id] = self.precomputed_vtw[key]
            else:
                vtws = self.propagator.compute_vtw(
                    m.lat, m.lon,
                    self.horizon_s,
                    time_step_s=self.vtw_time_step_s,
                )
                self.mission_vtw[m.id] = vtws

    def _get_next_vtw_times(self, mission_id: int) -> Tuple[float, float]:
        """获取该任务下一个可见窗口的起止时间"""
        vtws = self.mission_vtw.get(mission_id, [])
        for vtw in vtws:
            if vtw.end_time > self.current_time_s:
                return vtw.start_time, vtw.end_time
        return 0.0, 0.0

    def _all_missions_done(self) -> bool:
        """检查是否所有可行任务都已完成或不可行"""
        for m in self.missions:
            if m is None or m.is_observed:
                continue
            if self._earliest_feasible_observation_start(m, self.current_time_s) is not None:
                return False
        # 还要检查动态任务队列
        if self._dynamic_queue:
            return False
        return True

    # ===================================================================
    # 评估指标 (论文 Table 4)
    # ===================================================================
    def _is_feasible(self, mission) -> bool:
        """
        判断任务是否"可行"(feasible): 在其可用时间之后, 至少存在一个
        满足观测时长、且不晚于截止时间的可见窗口 (VTW)。

        论文 Table 4 用 "feasible missions" 作为成功率/完成率的分母——
        即排除那些物理上根本看不到的任务 (SSO 近极轨对中低纬度覆盖差,
        大量任务一整天无 VTW)。把不可见任务计入分母会不合理地稀释指标。
        """
        if mission is None:
            return False
        for vtw in self.mission_vtw.get(mission.id, []):
            # 窗口须能容纳观测时长
            if vtw.end_time - vtw.start_time < mission.duration_s:
                continue
            # 窗口须在任务可用时间之后结束 (动态任务到达前的窗口无效)
            if vtw.end_time < mission.earliest_time_s + mission.duration_s:
                continue
            # 观测须能在截止时间前完成
            earliest_obs_end = max(vtw.start_time, mission.earliest_time_s) + mission.duration_s
            if earliest_obs_end <= min(vtw.end_time, mission.deadline_s):
                return True
        return False

    def get_metrics(self) -> Dict[str, float]:
        """
        计算当前 episode 的评估指标 (对应论文 Table 4)。

        完成率/成功率的分母采用论文口径 = "feasible(可行)任务", 即至少有一个
        可用 VTW 的任务; 排除物理上无可见窗口、不可能完成的任务。
        同时额外给出 *_raw (以全部任务为分母) 与 feasible 统计以便诊断。
        """
        all_missions = [m for m in self.missions if m is not None]
        total_missions = len(all_missions)
        observed = sum(1 for m in all_missions if m.is_observed)
        completed = sum(1 for m in all_missions if self._mission_completed(m))

        # feasible 划分 (论文 Table 4 分母)
        feasible = [m for m in all_missions if self._is_feasible(m)]
        feas_total = len(feasible)
        feas_observed_only = sum(1 for m in feasible if m.is_observed)
        feas_observed = sum(1 for m in feasible if self._mission_completed(m))

        routine_feas = [m for m in feasible if not m.is_dynamic]
        dynamic_feas = [m for m in feasible if m.is_dynamic]
        routine_feas_done = sum(1 for m in routine_feas if self._mission_completed(m))
        dynamic_feas_done = sum(1 for m in dynamic_feas if self._mission_completed(m))

        # 全部任务口径 (诊断用)
        routine_total = sum(1 for m in all_missions if not m.is_dynamic)
        routine_done = sum(1 for m in all_missions if not m.is_dynamic and self._mission_completed(m))
        dynamic_total = sum(1 for m in all_missions if m.is_dynamic)
        dynamic_done = sum(1 for m in all_missions if m.is_dynamic and self._mission_completed(m))

        total_reward = sum(r.reward for r in self.schedule_log)
        dynamic_reward = sum(
            r.reward for r in self.schedule_log
            if any(m is not None and m.id == r.mission_id and m.is_dynamic for m in self.missions)
        )

        # --- 协同/质量指标 ---
        # 平均观测质量: off-nadir 越小图像质量越高
        if self.schedule_log:
            avg_off_nadir = float(np.mean([r.off_nadir_deg for r in self.schedule_log]))
        else:
            avg_off_nadir = 0.0
        # 动态任务平均响应延迟: 从任务可用(到达)到观测完成的时间(秒)
        dyn_delays = [
            (r.downlink_end_s if self.downlink_required else r.obs_end_s) - r.earliest_time_s
            for r in self.schedule_log
            if r.is_dynamic and ((not self.downlink_required) or r.ground_station_id >= 0)
        ]
        avg_dynamic_response_s = float(np.mean(dyn_delays)) if dyn_delays else 0.0
        downlink_durations = [
            max(0.0, r.downlink_end_s - r.downlink_start_s)
            for r in self.schedule_log
            if r.ground_station_id >= 0
        ]
        downlink_queue_delays = [
            max(0.0, r.downlink_start_s - r.obs_end_s)
            for r in self.schedule_log
            if r.ground_station_id >= 0
        ]
        n_ground_station_windows = sum(len(vtws) for vtws in self.ground_station_vtw.values())
        storage_stats = self._storage_stats()

        return {
            "total_reward": total_reward,
            # 论文 Table 4 口径: 分母 = feasible 任务
            "observation_success_rate": feas_observed / feas_total if feas_total > 0 else 0.0,
            "dynamic_completion_rate": dynamic_feas_done / len(dynamic_feas) if dynamic_feas else 0.0,
            "routine_completion_rate": routine_feas_done / len(routine_feas) if routine_feas else 0.0,
            # 全部任务口径 (诊断对照)
            "observation_success_rate_raw": completed / total_missions if total_missions > 0 else 0.0,
            "dynamic_completion_rate_raw": dynamic_done / dynamic_total if dynamic_total > 0 else 0.0,
            "routine_completion_rate_raw": routine_done / routine_total if routine_total > 0 else 0.0,
            "observation_only_success_rate": feas_observed_only / feas_total if feas_total > 0 else 0.0,
            "observation_only_success_rate_raw": observed / total_missions if total_missions > 0 else 0.0,
            # feasible 比例 (反映物理可达性)
            "n_total_tasks": total_missions,
            "n_routine_tasks": routine_total,
            "n_dynamic_tasks": dynamic_total,
            "n_feasible_tasks": feas_total,
            "n_feasible_routine": len(routine_feas),
            "n_feasible_dynamic": len(dynamic_feas),
            "n_feasible_observed": feas_observed,
            "n_feasible_observed_only": feas_observed_only,
            "n_feasible_routine_done": routine_feas_done,
            "n_feasible_dynamic_done": dynamic_feas_done,
            "feasible_ratio": feas_total / total_missions if total_missions > 0 else 0.0,
            "dynamic_feasible_ratio": len(dynamic_feas) / dynamic_total if dynamic_total > 0 else 0.0,
            # 协同/质量指标
            "avg_off_nadir_deg": avg_off_nadir,
            "avg_dynamic_response_s": avg_dynamic_response_s,
            "n_observed": observed,
            "n_downlinked": completed,
            "n_pending_downlink": max(observed - completed, 0),
            "n_ground_stations": self.n_ground_stations,
            "downlink_time_s": self.downlink_time_s,
            "satellite_storage_capacity": self.satellite_storage_capacity,
            **storage_stats,
            "avg_downlink_duration_s": float(np.mean(downlink_durations)) if downlink_durations else 0.0,
            "avg_downlink_queue_s": float(np.mean(downlink_queue_delays)) if downlink_queue_delays else 0.0,
            "n_ground_station_vtws": n_ground_station_windows,
            "avg_ground_station_vtws": (
                n_ground_station_windows / max(self.n_ground_stations, 1)
                if self.downlink_required else 0.0
            ),
            "dynamic_reward": dynamic_reward,
            "routine_reward": total_reward - dynamic_reward,
            "n_scheduled": completed,
            "n_duplicates": 0,  # 单星无重复观测(占位, 多星覆盖)
        }
