"""
多星协同调度环境
================
CTDE (Centralized Training Decentralized Execution) 多智能体环境。

核心设计:
  - N 颗卫星共享同一任务池 M = M_r ∪ M_d
  - 每颗卫星有独立的轨道/VTW，但观测结果全局同步
  - 当 Sat_i 完成 M_j 后，M_j 对所有卫星标记为已完成
  - 提供全局状态接口 (给集中式 Critic) 和局部观测接口 (给分布式 Actor)

接口风格参考 PettingZoo parallel API:
  reset() → {agent_id: (obs, info)}
  step({agent_id: action}) → {agent_id: (obs, reward, term, trunc, info)}
  get_global_state() → np.ndarray  (仅训练时使用)
"""

import numpy as np
import copy
from typing import Dict, List, Tuple, Optional, Any
import logging

from data.mission_generator import Mission
from data.orbit_utils import OrbitPropagator
from envs.satellite_env import SatelliteSchedulingEnv

logger = logging.getLogger(__name__)


class MultiSatelliteEnv:
    """
    多星协同调度环境。

    内部持有 N 个 SatelliteSchedulingEnv 实例，
    通过共享任务池和观测状态同步实现多星协调。
    """

    def __init__(
        self,
        satellite_configs: list,
        max_action_dim: int = 600,
        horizon_s: float = 86400.0,
        reward_config=None,
        vtw_time_step_s: float = 120.0,
    ):
        self.sat_configs = satellite_configs
        self.n_agents = len(satellite_configs)
        self.agent_ids = [cfg.name for cfg in satellite_configs]
        self.max_action_dim = max_action_dim
        self.horizon_s = horizon_s

        # 为每颗卫星创建独立的单星环境
        self.envs: Dict[str, SatelliteSchedulingEnv] = {}
        for cfg in satellite_configs:
            self.envs[cfg.name] = SatelliteSchedulingEnv(
                satellite_config=cfg,
                max_action_dim=max_action_dim,
                horizon_s=horizon_s,
                reward_config=reward_config,
                vtw_time_step_s=vtw_time_step_s,
            )

        # 维度信息 (所有卫星共享相同的 obs/action 维度)
        sample_env = list(self.envs.values())[0]
        self.local_obs_dim = sample_env.observation_space.shape[0]
        self.action_dim = sample_env.action_space.n
        # mean pooling：全局状态 = 所有卫星局部观测的均值，维度与单卫星观测相同
        self.global_state_dim = self.local_obs_dim

        # 共享任务池 (在 reset 时初始化)
        self._shared_missions: List[Optional[Mission]] = []

    # ===================================================================
    # 核心接口
    # ===================================================================
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Dict[str, Tuple[np.ndarray, Dict]]:
        """
        重置所有卫星环境。

        options 字典应包含:
          "routine_missions": List[Mission]    (所有卫星共享)
          "dynamic_schedule": List[Tuple[float, List[Mission]]]
        """
        options = options or {}
        routine_missions = options.get("routine_missions", [])
        dynamic_schedule = options.get("dynamic_schedule", [])

        # 深拷贝任务列表——每颗卫星需要独立的 Mission 对象引用，
        # 但观测状态通过 _sync_mission_status() 同步
        self._shared_missions = routine_missions  # 保持原始引用
        self._dynamic_schedule_template = dynamic_schedule

        results = {}
        for agent_id, env in self.envs.items():
            # 每颗卫星获得任务的独立副本 (不同轨道有不同 VTW)
            local_routine = copy.deepcopy(routine_missions)
            local_dynamic = copy.deepcopy(dynamic_schedule)
            obs, info = env.reset(options={
                "routine_missions": local_routine,
                "dynamic_schedule": local_dynamic,
            })
            results[agent_id] = (obs, info)

        return results

    def step(
        self,
        actions: Dict[str, int],
    ) -> Dict[str, Tuple[np.ndarray, float, bool, bool, Dict]]:
        """
        所有卫星同时执行一步决策。

        包含冲突检测：如果多颗卫星选择同一任务，
        只允许第一颗执行，其余强制 idle（避免双重奖励）。
        """
        results = {}

        # 1) 冲突检测：同一任务只能被一颗卫星执行
        resolved_actions = {}
        claimed_missions = set()
        for agent_id, action in actions.items():
            idle = self.max_action_dim
            if action != idle and action not in claimed_missions:
                resolved_actions[agent_id] = action
                claimed_missions.add(action)
            elif action != idle and action in claimed_missions:
                # 冲突：强制 idle
                resolved_actions[agent_id] = idle
            else:
                resolved_actions[agent_id] = action

        # 2) 每颗卫星执行（已去冲突的）动作
        for agent_id in self.agent_ids:
            env = self.envs[agent_id]
            obs, reward, term, trunc, info = env.step(resolved_actions[agent_id])
            results[agent_id] = (obs, reward, term, trunc, info)

        # 3) 同步观测状态
        self._sync_mission_status()

        # 4) 用同步后的状态重新构建观测和掩码
        for agent_id, env in self.envs.items():
            obs = env._build_observation()
            mask = env._build_action_mask()
            old_result = results[agent_id]
            results[agent_id] = (
                obs,
                old_result[1],
                old_result[2],
                old_result[3],
                {**old_result[4], "action_mask": mask},
            )

        return results

    def _sync_mission_status(self):
        """
        跨卫星同步任务观测状态。

        如果任何一颗卫星完成了任务 M_j (is_observed=True)，
        则所有卫星的 M_j 都标记为已完成。
        这是多星协调的核心——避免重复观测。
        """
        # 收集所有已完成任务的 ID
        observed_ids = set()
        for env in self.envs.values():
            for m in env.missions:
                if m is not None and m.is_observed:
                    observed_ids.add(m.id)

        # 同步到所有卫星
        if observed_ids:
            for env in self.envs.values():
                for m in env.missions:
                    if m is not None and m.id in observed_ids:
                        m.is_observed = True

    # ===================================================================
    # 全局状态 (仅训练时 Critic 使用)
    # ===================================================================
    def get_global_state(self) -> np.ndarray:
        """
        构建全局状态向量 (给集中式 Critic)。

        全局状态 = 所有卫星局部观测的均值 (mean pooling)。
        维度 = local_obs_dim，与卫星数量无关，避免参数爆炸。
        """
        local_obs_list = []
        for agent_id in self.agent_ids:
            env = self.envs[agent_id]
            local_obs_list.append(env._build_observation())
        return np.mean(local_obs_list, axis=0)

    # ===================================================================
    # 评估指标 (聚合所有卫星)
    # ===================================================================
    def get_metrics(self) -> Dict[str, float]:
        """聚合所有卫星的调度指标"""
        # 合并所有卫星的调度记录
        all_scheduled_ids = set()
        total_reward = 0.0
        total_time = 0.0

        for env in self.envs.values():
            metrics = env.get_metrics()
            total_reward += metrics["total_reward"]
            for record in env.schedule_log:
                all_scheduled_ids.add(record.mission_id)

        # 基于共享任务池统计
        # 使用第一颗卫星的任务列表作为基准 (它们的 id 相同)
        first_env = list(self.envs.values())[0]
        all_missions = [m for m in first_env.missions if m is not None]
        total_missions = len(all_missions)
        routine_total = sum(1 for m in all_missions if not m.is_dynamic)
        dynamic_total = sum(1 for m in all_missions if m.is_dynamic)

        # 统计哪些任务被任意一颗卫星完成
        observed_total = len(all_scheduled_ids)
        routine_done = sum(
            1 for m in all_missions
            if not m.is_dynamic and m.id in all_scheduled_ids
        )
        dynamic_done = sum(
            1 for m in all_missions
            if m.is_dynamic and m.id in all_scheduled_ids
        )

        return {
            "total_reward": total_reward,
            "observation_success_rate": observed_total / max(total_missions, 1),
            "dynamic_completion_rate": dynamic_done / max(dynamic_total, 1),
            "routine_completion_rate": routine_done / max(routine_total, 1),
            "n_scheduled": observed_total,
            "n_duplicates": sum(
                len(env.schedule_log) for env in self.envs.values()
            ) - observed_total,  # 重复观测数
        }

    def is_done(self) -> bool:
        """检查是否所有卫星的 episode 都结束"""
        for env in self.envs.values():
            if env.current_time_s < env.horizon_s and not env._all_missions_done():
                return False
        return True

    @property
    def idle_action(self) -> int:
        return self.max_action_dim  # 与单星环境一致
