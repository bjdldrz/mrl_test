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
        self._last_off_nadir_deg = 0.0  # 跟踪上一次观测的 off-nadir 角

        # 动态任务插入队列
        self._dynamic_queue: List[Tuple[float, List[Mission]]] = []
        self._n_routine = 0

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
        self.mission_vtw = {}
        self._last_off_nadir_deg = 0.0

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

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
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
            # 空闲: 智能推进到下一个有意义的事件时刻
            # (下一个 VTW 开始、动态任务到达、或兜底 60 秒)
            next_event_t = self.current_time_s + 60.0  # 兜底
            # 检查最近的 VTW 开始时刻
            for m in self.missions:
                if m is not None and not m.is_observed:
                    for vtw in self.mission_vtw.get(m.id, []):
                        if vtw.start_time > self.current_time_s:
                            next_event_t = min(next_event_t, vtw.start_time)
                            break
            # 检查最近的动态任务到达时刻
            if self._dynamic_queue:
                next_event_t = min(next_event_t, self._dynamic_queue[0][0])
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

        if self._all_missions_done():
            terminated = True  # 所有可行任务已完成

        # 4) 构建下一步状态
        obs = self._build_observation()
        info["action_mask"] = self._build_action_mask()
        info["current_time_s"] = self.current_time_s
        info["schedule_log"] = self.schedule_log

        return obs, reward, terminated, truncated, info

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
            conflict = False
            for record in self.schedule_log:
                if not (obs_end <= record.obs_start_s or obs_start >= record.obs_end_s):
                    conflict = True
                    break
            if conflict:
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

            # 观测状态编码: 0=未观测, 0.5=观测中, 1=已完成
            if m.is_observed:
                obs_status = 1.0
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

        # 执行观测
        mission.is_observed = True
        mission.obs_start_s = obs_start
        mission.obs_end_s = obs_end
        self._last_off_nadir_deg = usable_vtw.off_nadir_deg  # 更新跟踪

        reward = self.compute_reward(
            mission, obs_end, off_nadir_deg=usable_vtw.off_nadir_deg
        )

        self.schedule_log.append(ScheduleRecord(
            mission_id=mission.id,
            satellite_name=self.sat_config.name,
            obs_start_s=obs_start,
            obs_end_s=obs_end,
            reward=reward,
        ))

        # 时间推进到观测结束
        self.current_time_s = obs_end

        return reward, True

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
            if m.deadline_s <= self.current_time_s:
                continue  # 已过截止时间, 不可行
            if m.earliest_time_s > self.current_time_s:
                return False  # 尚有未到达的动态任务
            # 检查是否还有可用的 VTW
            vtws = self.mission_vtw.get(m.id, [])
            for vtw in vtws:
                if vtw.end_time > self.current_time_s:
                    return False  # 仍有可用窗口
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

        # feasible 划分 (论文 Table 4 分母)
        feasible = [m for m in all_missions if self._is_feasible(m)]
        feas_total = len(feasible)
        feas_observed = sum(1 for m in feasible if m.is_observed)

        routine_feas = [m for m in feasible if not m.is_dynamic]
        dynamic_feas = [m for m in feasible if m.is_dynamic]
        routine_feas_done = sum(1 for m in routine_feas if m.is_observed)
        dynamic_feas_done = sum(1 for m in dynamic_feas if m.is_observed)

        # 全部任务口径 (诊断用)
        routine_total = sum(1 for m in all_missions if not m.is_dynamic)
        routine_done = sum(1 for m in all_missions if not m.is_dynamic and m.is_observed)
        dynamic_total = sum(1 for m in all_missions if m.is_dynamic)
        dynamic_done = sum(1 for m in all_missions if m.is_dynamic and m.is_observed)

        total_reward = sum(r.reward for r in self.schedule_log)
        dynamic_reward = sum(
            r.reward for r in self.schedule_log
            if any(m is not None and m.id == r.mission_id and m.is_dynamic for m in self.missions)
        )

        return {
            "total_reward": total_reward,
            # 论文 Table 4 口径: 分母 = feasible 任务
            "observation_success_rate": feas_observed / feas_total if feas_total > 0 else 0.0,
            "dynamic_completion_rate": dynamic_feas_done / len(dynamic_feas) if dynamic_feas else 0.0,
            "routine_completion_rate": routine_feas_done / len(routine_feas) if routine_feas else 0.0,
            # 全部任务口径 (诊断对照)
            "observation_success_rate_raw": observed / total_missions if total_missions > 0 else 0.0,
            "dynamic_completion_rate_raw": dynamic_done / dynamic_total if dynamic_total > 0 else 0.0,
            "routine_completion_rate_raw": routine_done / routine_total if routine_total > 0 else 0.0,
            # feasible 比例 (反映物理可达性)
            "feasible_ratio": feas_total / total_missions if total_missions > 0 else 0.0,
            "dynamic_feasible_ratio": len(dynamic_feas) / dynamic_total if dynamic_total > 0 else 0.0,
            "dynamic_reward": dynamic_reward,
            "routine_reward": total_reward - dynamic_reward,
            "n_scheduled": len(self.schedule_log),
        }
