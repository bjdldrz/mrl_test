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
        max_action_dim: int = 800,
        horizon_s: float = 86400.0,
        reward_config=None,
        vtw_time_step_s: float = 120.0,
        coordinate: bool = True,
        reassign_losers: bool = True,
        coord_w_priority: float = 1.0,
        coord_w_quality: float = 0.5,
        coord_w_load: float = 0.05,
        episode_assignment: bool = True,
        assign_w_load: float = 0.1,
        assignment_capacity_mode: str = "proportional",
        release_before_deadline_s: float = 1800.0,
        assignment_scorer: str = "heuristic",
        assignment_scorer_mix: float = 0.25,
        assignment_context_encoder: str = "lstm",
        assignment_context_weight: float = 0.25,
        assignment_mlp_hidden_dim: int = 16,
        assignment_mlp_seed: int = 42,
        assignment_sequence_hidden_dim: int = 16,
        assignment_replan_interval_s: float = 0.0,
        assignment_replan_horizon_s: float = 0.0,
        assignment_replan_trigger: str = "none",
        assignment_switch_penalty: float = 0.05,
        assignment_lock_window_s: float = 600.0,
        assignment_max_switches_per_task: int = 2,
        assignment_manager_mode: str = "none",
        team_reward_mix: float = 0.0,
        load_balance_reward_coeff: float = 0.0,
        team_completion_bonus: float = 0.0,
        global_state_mode: str = "mean",
        global_state_task_stats: bool = False,
        candidate_action_top_k: int = 0,
        n_ground_stations: int = 0,
        downlink_time_s: float = 0.0,
        ground_station_configs: Optional[List[Any]] = None,
        satellite_storage_capacity: int = 0,
        enable_inter_satellite_transfer: bool = False,
        inter_satellite_transfer_time_s: float = 300.0,
    ):
        self.sat_configs = satellite_configs
        self.n_agents = len(satellite_configs)
        self.agent_ids = [cfg.name for cfg in satellite_configs]
        self.max_action_dim = max_action_dim
        self.horizon_s = horizon_s
        # coordinate=True: 冲突协调 + 观测状态同步(MAPPO 协同).
        # coordinate=False: 各卫星完全独立决策, 不去冲突、不共享观测(无协同 baseline).
        self.coordinate = coordinate
        # --- 协同冲突解决参数 (优化路线图 A1+A2/A3+B6) ---
        # reassign_losers: 抢输的卫星是否改派到次优任务 (A1); False 则退化为"败者 idle".
        self.reassign_losers = reassign_losers
        # A1 改派仅在评估期启用. 训练期启用会破坏信用分配: 智能体采样动作 X(buffer 记 X),
        # 抢输后被改派到 Y 并领 Y 的奖励 → "选 X"被 Y 的奖励错误强化, 教坏协同.
        # 训练期只做 A2/A3(择优胜者)+B6(负载均衡): 胜者拿自己选的任务奖励、败者 idle, 信用分配干净.
        self.eval_mode = False
        # 边际价值竞价权重: 价值 = w_priority·优先级 + w_quality·质量 − w_load·负载.
        self.coord_w_priority = coord_w_priority   # 任务优先级权重
        self.coord_w_quality = coord_w_quality     # 观测质量权重 (off-nadir 越小越高)
        self.coord_w_load = coord_w_load           # 负载惩罚权重 (B6: 偏向空闲卫星)
        # --- 全局 episode 级任务指派 (优化路线图 A3-episode / G24) ---
        # episode_assignment: reset 时综合全 24h 窗口质量为每个任务预指派归属卫星,
        # 用所有权掩码让各星只考虑自己负责的任务 → 从构造上消重 + 均衡负载.
        self.episode_assignment = episode_assignment
        self.assign_w_load = assign_w_load          # 指派时的负载均衡权重 (越大越均衡)
        self.assignment_capacity_mode = assignment_capacity_mode
        self.assignment_scorer = assignment_scorer
        self.assignment_scorer_mix = float(np.clip(assignment_scorer_mix, 0.0, 1.0))
        self.assignment_context_encoder = self._validate_assignment_context_encoder(
            assignment_context_encoder
        )
        self.assignment_context_weight = max(0.0, float(assignment_context_weight))
        self.assignment_mlp_hidden_dim = max(1, int(assignment_mlp_hidden_dim))
        self.assignment_mlp_seed = int(assignment_mlp_seed)
        self.assignment_sequence_hidden_dim = max(1, int(assignment_sequence_hidden_dim))
        self._assignment_mlp = None
        self._assignment_sequence = None
        self._assignment_sequence_context: Dict[int, np.ndarray] = {}
        self._assignment_sat_context: Dict[str, np.ndarray] = {}
        self._init_assignment_scorer()
        # --- Rolling horizon / MPC 风格重分配 (assignment_rolling_v1) ---
        # 默认关闭, 仅记录诊断指标; 通过 CLI/preset 显式开启周期或事件触发重分配。
        self.assignment_replan_interval_s = max(0.0, float(assignment_replan_interval_s))
        self.assignment_replan_horizon_s = max(0.0, float(assignment_replan_horizon_s))
        self.assignment_replan_triggers = self._parse_replan_triggers(assignment_replan_trigger)
        self.assignment_switch_penalty = max(0.0, float(assignment_switch_penalty))
        self.assignment_lock_window_s = max(0.0, float(assignment_lock_window_s))
        self.assignment_max_switches_per_task = max(0, int(assignment_max_switches_per_task))
        self._last_replan_time_s = 0.0
        self._owner_switch_counts: Dict[int, int] = {}
        self._n_replans = 0
        self._n_owner_switches = 0
        self._n_replan_checks = 0
        self._n_stale_owner_events = 0
        self._released_mission_ids = set()
        self._deadline_release_mission_ids = set()
        self._rescued_mission_ids = set()
        self._deadline_rescue_mission_ids = set()
        self.assignment_manager_mode = self._validate_assignment_manager_mode(assignment_manager_mode)
        self._assignment_manager = self._init_assignment_manager()
        # 截止前释放窗口: owner 尚未完成时, 非 owner 可在任务临近 deadline 时接手,
        # 回收硬所有权导致的吞吐损失. 设为 0 可关闭释放机制.
        self.release_before_deadline_s = release_before_deadline_s
        self.task_owner: Dict[int, str] = {}        # mission_id → 负责的卫星 agent_id
        self._assign_load: Dict[str, int] = {}      # 各卫星已被指派的任务数
        # --- 多智能体奖励塑形 (优化路线图 B5+C8) ---
        # 默认全为 0, 保持论文/基线奖励不变.
        self.team_reward_mix = team_reward_mix
        self.load_balance_reward_coeff = load_balance_reward_coeff
        self.team_completion_bonus = team_completion_bonus
        # --- 集中式 Critic 全局状态表示 (优化路线图 D14/D16) ---
        # mean: 兼容旧实现; concat: 保留每颗星完整局部观测, 信息无损但维度随星数增长.
        self.global_state_mode = global_state_mode
        self.global_state_task_stats = global_state_task_stats
        # --- 分层候选动作空间 (CVA-MAPPO scale-up) ---
        # 0 表示保持旧版 full action space; >0 时对每颗星只暴露 Top-K 当前可行任务 + idle。
        # 底层 SatelliteSchedulingEnv 仍保留 max_action_dim 个真实槽位,用于 VTW、统计和结果可视化。
        self.candidate_action_top_k = max(0, int(candidate_action_top_k or 0))
        self._candidate_action_maps: Dict[str, List[Optional[int]]] = {}
        # --- 基站下传约束 ---
        # n_ground_stations=0 时保持旧口径: 观测结束即完成。
        # >0 时所有卫星共享 n 个基站服务台, 观测图像排队下传, 下传完成才计入任务完成。
        self.ground_station_configs = SatelliteSchedulingEnv._build_ground_station_configs(
            n_ground_stations,
            ground_station_configs,
        )
        self.n_ground_stations = len(self.ground_station_configs)
        self.downlink_time_s = max(0.0, float(downlink_time_s or 0.0))
        self._ground_station_available_s: List[float] = [0.0] * self.n_ground_stations
        self.satellite_storage_capacity = max(0, int(satellite_storage_capacity or 0))
        # enable_inter_satellite_transfer=True 时,星间转发不再是环境自动 fallback,
        # 而是暴露给智能体的显式动作:选择目标星后一次性发送源星当前全部未交付图片。
        self.enable_inter_satellite_transfer = bool(enable_inter_satellite_transfer)
        self.inter_satellite_transfer_time_s = max(0.0, float(inter_satellite_transfer_time_s or 0.0))
        self._transfer_targets: Dict[str, List[str]] = {}

        # 为每颗卫星创建独立的单星环境
        self.envs: Dict[str, SatelliteSchedulingEnv] = {}
        for cfg in satellite_configs:
            self.envs[cfg.name] = SatelliteSchedulingEnv(
                satellite_config=cfg,
                max_action_dim=max_action_dim,
                horizon_s=horizon_s,
                reward_config=reward_config,
                vtw_time_step_s=vtw_time_step_s,
                n_ground_stations=self.n_ground_stations,
                downlink_time_s=self.downlink_time_s,
                ground_station_configs=self.ground_station_configs,
                satellite_storage_capacity=self.satellite_storage_capacity,
            )
            self.envs[cfg.name].set_ground_station_state(self._ground_station_available_s)

        # 维度信息 (所有卫星共享相同的 obs/action 维度)
        sample_env = list(self.envs.values())[0]
        if self._candidate_actions_enabled():
            self.local_obs_dim = (
                self.candidate_action_top_k * sample_env._mission_feat_dim
                + sample_env._sat_feat_dim
            )
            self.action_dim = self.candidate_action_top_k + self._transfer_action_count() + 1
        else:
            self.local_obs_dim = sample_env.observation_space.shape[0]
            self.action_dim = self._raw_idle_action() + 1
        if self.global_state_mode == "concat":
            base_global_dim = self.local_obs_dim * self.n_agents
        else:
            base_global_dim = self.local_obs_dim
        stats_dim = self.n_agents + 6 if self.global_state_task_stats else 0
        self.global_state_dim = base_global_dim + stats_dim

        # 共享任务池 (在 reset 时初始化)
        self._shared_missions: List[Optional[Mission]] = []

    def _init_assignment_scorer(self):
        """初始化 episode 级任务指派 scorer; 默认 heuristic 完全保持旧逻辑。"""
        allowed = {"heuristic", "mlp", "lstm", "gru", "transformer", "set_transformer", "gnn", "cva"}
        if self.assignment_scorer not in allowed:
            raise ValueError(
                f"未知 assignment_scorer={self.assignment_scorer!r}; "
                f"可选: {sorted(allowed)}"
            )

        context_encoder = self._effective_assignment_context_encoder()
        needs_mlp = self.assignment_scorer == "mlp" or context_encoder == "mlp"
        if needs_mlp:
            self._init_assignment_mlp_scorer()
        if context_encoder in {"lstm", "gru"}:
            self._init_assignment_sequence_scorer()
        elif context_encoder in {"transformer", "set_transformer"}:
            self._init_assignment_attention_scorer()
        elif context_encoder == "gnn":
            self._init_assignment_gnn_scorer()

    @staticmethod
    def _validate_assignment_context_encoder(encoder: str) -> str:
        normalized = str(encoder or "lstm").strip().lower()
        allowed = {"mlp", "lstm", "gru", "transformer", "set_transformer", "gnn"}
        if normalized not in allowed:
            raise ValueError(
                f"未知 assignment_context_encoder={encoder!r}; 可选: {sorted(allowed)}"
            )
        return normalized

    def _effective_assignment_context_encoder(self) -> str:
        """返回当前 scorer 实际使用的上下文编码器类型。"""
        if self.assignment_scorer == "cva":
            return self.assignment_context_encoder
        if self.assignment_scorer in {"lstm", "gru", "transformer", "set_transformer", "gnn"}:
            return self.assignment_scorer
        return "none"

    def _init_assignment_mlp_scorer(self):
        """初始化确定性 MLP 边价值 scorer。"""
        rng = np.random.RandomState(self.assignment_mlp_seed)
        in_dim = 8
        hidden = self.assignment_mlp_hidden_dim
        # 小初始化 + 启发式友好的输出偏置: 第一版只做可复现 scorer 消融,
        # 后续可把这组权重替换为监督/强化训练得到的参数。
        self._assignment_mlp = {
            "w1": rng.normal(0.0, 0.15, size=(in_dim, hidden)).astype(np.float32),
            "b1": np.zeros(hidden, dtype=np.float32),
            "w2": rng.normal(0.0, 0.15, size=(hidden, 1)).astype(np.float32),
            "b2": np.zeros(1, dtype=np.float32),
        }

    @staticmethod
    def _parse_replan_triggers(trigger_text: str) -> set:
        """解析滚动重分配触发器列表。"""
        if trigger_text is None:
            return set()
        normalized = str(trigger_text).strip().lower()
        if normalized in {"", "none", "off", "false", "0"}:
            return set()
        triggers = {x.strip() for x in normalized.split(",") if x.strip()}
        allowed = {"periodic", "dynamic", "stale_owner", "deadline", "imbalance"}
        unknown = triggers - allowed
        if unknown:
            raise ValueError(
                f"未知 assignment_replan_trigger={sorted(unknown)}; "
                f"可选: {sorted(allowed)} 或 none"
            )
        return triggers

    @staticmethod
    def _validate_assignment_manager_mode(mode: str) -> str:
        normalized = str(mode or "none").strip().lower()
        allowed = {"none", "rule"}
        if normalized not in allowed:
            raise ValueError(
                f"未知 assignment_manager_mode={mode!r}; 可选: {sorted(allowed)}"
            )
        return normalized

    def _init_assignment_manager(self):
        if self.assignment_manager_mode == "none":
            return None
        from models.assignment_manager import RuleBasedAssignmentManager
        return RuleBasedAssignmentManager()

    def _init_assignment_sequence_scorer(self):
        """初始化确定性 LSTM/GRU 风格序列 scorer 权重。"""
        rng = np.random.RandomState(self.assignment_mlp_seed)
        encoder = self._effective_assignment_context_encoder()
        in_dim = 7
        hidden = self.assignment_sequence_hidden_dim
        out_dim = 8
        def init(shape, scale=0.12):
            return rng.normal(0.0, scale, size=shape).astype(np.float32)

        self._assignment_sequence = {
            "w_out": init((hidden, out_dim)),
            "b_out": np.zeros(out_dim, dtype=np.float32),
        }
        if encoder == "lstm":
            self._assignment_sequence.update({
                "w_ix": init((in_dim, hidden)), "w_ih": init((hidden, hidden)), "b_i": np.zeros(hidden, dtype=np.float32),
                "w_fx": init((in_dim, hidden)), "w_fh": init((hidden, hidden)), "b_f": np.ones(hidden, dtype=np.float32),
                "w_ox": init((in_dim, hidden)), "w_oh": init((hidden, hidden)), "b_o": np.zeros(hidden, dtype=np.float32),
                "w_gx": init((in_dim, hidden)), "w_gh": init((hidden, hidden)), "b_g": np.zeros(hidden, dtype=np.float32),
            })
        else:
            self._assignment_sequence.update({
                "w_zx": init((in_dim, hidden)), "w_zh": init((hidden, hidden)), "b_z": np.zeros(hidden, dtype=np.float32),
                "w_rx": init((in_dim, hidden)), "w_rh": init((hidden, hidden)), "b_r": np.zeros(hidden, dtype=np.float32),
                "w_nx": init((in_dim, hidden)), "w_nh": init((hidden, hidden)), "b_n": np.zeros(hidden, dtype=np.float32),
            })

    def _init_assignment_attention_scorer(self):
        """初始化确定性 Transformer/Set Transformer 风格 scorer 权重。"""
        rng = np.random.RandomState(self.assignment_mlp_seed)
        encoder = self._effective_assignment_context_encoder()
        in_dim = 7
        hidden = self.assignment_sequence_hidden_dim
        out_dim = 8
        def init(shape, scale=0.12):
            return rng.normal(0.0, scale, size=shape).astype(np.float32)

        self._assignment_sequence = {
            "w_embed": init((in_dim, hidden)),
            "b_embed": np.zeros(hidden, dtype=np.float32),
            "w_q": init((hidden, hidden)),
            "w_k": init((hidden, hidden)),
            "w_v": init((hidden, hidden)),
            "w_ff": init((hidden, hidden)),
            "b_ff": np.zeros(hidden, dtype=np.float32),
            "w_out": init((hidden, out_dim)),
            "b_out": np.zeros(out_dim, dtype=np.float32),
        }
        if encoder == "transformer":
            self._assignment_sequence["pos_scale"] = init((hidden,), scale=0.04)

    def _init_assignment_gnn_scorer(self):
        """初始化确定性二分图 GNN 风格 scorer 权重。"""
        rng = np.random.RandomState(self.assignment_mlp_seed)
        task_in_dim = 7
        sat_in_dim = 5
        hidden = self.assignment_sequence_hidden_dim
        out_dim = 8
        def init(shape, scale=0.12):
            return rng.normal(0.0, scale, size=shape).astype(np.float32)

        self._assignment_sequence = {
            "w_task": init((task_in_dim, hidden)),
            "b_task": np.zeros(hidden, dtype=np.float32),
            "w_sat": init((sat_in_dim, hidden)),
            "b_sat": np.zeros(hidden, dtype=np.float32),
            "w_edge_task": init((hidden, hidden)),
            "w_edge_sat": init((hidden, hidden)),
            "w_out": init((hidden, out_dim)),
            "b_out": np.zeros(out_dim, dtype=np.float32),
        }

    # ===================================================================
    # 核心接口
    # ===================================================================
    def _candidate_actions_enabled(self) -> bool:
        return self.candidate_action_top_k > 0

    def _raw_idle_action(self) -> int:
        return self.max_action_dim + self._transfer_action_count()

    def _transfer_actions_enabled(self) -> bool:
        return (
            self.coordinate
            and self.enable_inter_satellite_transfer
            and self.n_agents > 1
            and self.downlink_time_s > 0
            and self.n_ground_stations > 0
            and self.inter_satellite_transfer_time_s > 0
        )

    def _transfer_action_count(self) -> int:
        return (self.n_agents - 1) if self._transfer_actions_enabled() else 0

    def _transfer_target_list(self, agent_id: str) -> List[str]:
        if agent_id not in self._transfer_targets:
            self._transfer_targets[agent_id] = [
                aid for aid in self.agent_ids if aid != agent_id
            ]
        return self._transfer_targets[agent_id]

    def _raw_transfer_action(self, agent_id: str, target_agent_id: str) -> Optional[int]:
        targets = self._transfer_target_list(agent_id)
        if target_agent_id not in targets:
            return None
        return self.max_action_dim + targets.index(target_agent_id)

    def _raw_transfer_target(self, agent_id: str, action: int) -> Optional[str]:
        if not self._transfer_actions_enabled():
            return None
        idx = int(action) - self.max_action_dim
        targets = self._transfer_target_list(agent_id)
        if 0 <= idx < len(targets):
            return targets[idx]
        return None

    def _is_raw_transfer_action(self, agent_id: str, action: int) -> bool:
        return self._raw_transfer_target(agent_id, action) is not None

    def _full_action_mask(self, agent_id: str) -> np.ndarray:
        env = self.envs[agent_id]
        task_mask = env._build_action_mask()
        if self.coordinate and self.episode_assignment:
            task_mask = self._apply_ownership_mask(agent_id, task_mask)
        mask = np.zeros(self._raw_idle_action() + 1, dtype=np.float32)
        mask[:self.max_action_dim] = task_mask[:self.max_action_dim]
        if self._transfer_actions_enabled():
            for target_id in self._transfer_target_list(agent_id):
                action = self._raw_transfer_action(agent_id, target_id)
                if action is not None and self._can_transfer_all_images(agent_id, target_id):
                    mask[action] = 1.0
        mask[self._raw_idle_action()] = 1.0
        return mask

    def _expose_obs_info(self, agent_id: str, full_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        将底层 full-slot 状态转换为对策略暴露的状态/掩码。

        full action: 直接返回原始观测和原始 mask。
        Top-K action: 选择当前可行任务 Top-K,动作 index 映射到真实任务槽位。
        """
        env = self.envs[agent_id]
        if full_mask is None:
            full_mask = self._full_action_mask(agent_id)
        if not self._candidate_actions_enabled():
            return env._build_observation(), {"action_mask": full_mask}

        mapping = self._select_candidate_actions(agent_id, full_mask)
        self._candidate_action_maps[agent_id] = mapping
        obs = self._build_candidate_observation(agent_id, mapping)
        transfer_count = self._transfer_action_count()
        exposed_idle = self.idle_action
        mask = np.zeros(self.candidate_action_top_k + transfer_count + 1, dtype=np.float32)
        for idx, raw_action in enumerate(mapping):
            if raw_action is not None and full_mask[raw_action] > 0:
                mask[idx] = 1.0
        for offset, target_id in enumerate(self._transfer_target_list(agent_id)):
            raw_action = self._raw_transfer_action(agent_id, target_id)
            exposed_action = self.candidate_action_top_k + offset
            if raw_action is not None and raw_action < len(full_mask) and full_mask[raw_action] > 0:
                mask[exposed_action] = 1.0
        mask[exposed_idle] = 1.0
        return obs, {
            "action_mask": mask,
            "candidate_action_slots": [
                int(a) if a is not None else None for a in mapping
            ],
            "transfer_action_targets": list(self._transfer_target_list(agent_id)),
        }

    def _select_candidate_actions(self, agent_id: str, full_mask: np.ndarray) -> List[Optional[int]]:
        current_actions = np.nonzero(full_mask[:self.max_action_dim])[0].tolist()
        current_scored = [
            (self._candidate_action_score(agent_id, int(action)), int(action))
            for action in current_actions
        ]
        current_scored = [item for item in current_scored if item[0] is not None]
        current_scored.sort(key=lambda item: item[0], reverse=True)
        selected = [action for _, action in current_scored[:self.candidate_action_top_k]]

        # 若当前可行动作不足 K,用未来高价值 owner 任务补齐观测槽。
        # 这些未来任务在 mask 中仍为 0,只帮助策略判断是否 idle 等待。
        if len(selected) < self.candidate_action_top_k:
            selected_set = set(selected)
            future_scored = []
            env = self.envs[agent_id]
            for action, mission in enumerate(env.missions[:self.max_action_dim]):
                if action in selected_set or mission is None or mission.is_observed:
                    continue
                if mission.deadline_s <= env.current_time_s:
                    continue
                if self.coordinate and self.episode_assignment:
                    owner = self.task_owner.get(mission.id)
                    if owner is not None and owner != agent_id:
                        continue
                score = self._candidate_action_score(agent_id, action, allow_future=True)
                if score is not None:
                    future_scored.append((score, action))
            future_scored.sort(key=lambda item: item[0], reverse=True)
            for _, action in future_scored:
                if len(selected) >= self.candidate_action_top_k:
                    break
                selected.append(action)

        if len(selected) < self.candidate_action_top_k:
            selected.extend([None] * (self.candidate_action_top_k - len(selected)))
        return selected

    def _candidate_action_score(
        self,
        agent_id: str,
        action: int,
        allow_future: bool = False,
    ) -> Optional[float]:
        env = self.envs[agent_id]
        mission = env.missions[action]
        if mission is None or mission.is_observed:
            return None

        off_nadir = None
        wait_s = 0.0
        for vtw in env.mission_vtw.get(mission.id, []):
            if vtw.start_time <= env.current_time_s <= vtw.end_time - mission.duration_s:
                off_nadir = vtw.off_nadir_deg
                break
            if allow_future and vtw.end_time > env.current_time_s:
                obs_start = max(vtw.start_time, env.current_time_s, mission.earliest_time_s)
                obs_end = obs_start + mission.duration_s
                if obs_end <= min(vtw.end_time, mission.deadline_s):
                    off_nadir = vtw.off_nadir_deg
                    wait_s = max(obs_start - env.current_time_s, 0.0)
                    break
        if off_nadir is None:
            return None

        max_roll = max(env.sat_config.max_roll_deg, 1e-6)
        quality = 1.0 - min(off_nadir / max_roll, 1.0)
        priority = np.clip(mission.priority / 10.0, 0.0, 1.0)
        slack_s = max(mission.deadline_s - max(env.current_time_s, mission.earliest_time_s), 0.0)
        slack_norm = np.clip(slack_s / max(self.horizon_s, 1.0), 0.0, 1.0)
        deadline_pressure = 1.0 - slack_norm
        dynamic = 1.0 if mission.is_dynamic else 0.0
        wait_penalty = np.clip(wait_s / max(self.horizon_s, 1.0), 0.0, 1.0)
        owner_bonus = 0.0
        owner = self.task_owner.get(mission.id)
        if owner == agent_id:
            owner_bonus = 0.04
        elif owner is not None:
            # 能进入候选集说明 ownership mask 已判定该任务可由当前卫星执行。
            owner_bonus = 0.12

        base = (
            0.52 * quality
            + 0.20 * priority
            + 0.16 * deadline_pressure
            + 0.12 * dynamic
            + owner_bonus
            - 0.08 * wait_penalty
            - 0.02 * len(env.schedule_log)
        )
        if self.coordinate and self.episode_assignment:
            # 用当前分配器作为候选排序的价值项;目标容量取当前 backlog 的平滑估计。
            avg_target = max(
                (sum(self._assign_load.values()) + 1.0) / max(self.n_agents, 1),
                1.0,
            )
            targets = {aid: avg_target for aid in self.agent_ids}
            assignment_value = self._assignment_score(
                agent_id, mission, quality, targets, n_candidates=1
            )
            base = 0.65 * base + 0.35 * assignment_value
        return float(base)

    def _build_candidate_observation(self, agent_id: str, mapping: List[Optional[int]]) -> np.ndarray:
        env = self.envs[agent_id]
        mission_feats = np.zeros(
            (self.candidate_action_top_k, env._mission_feat_dim), dtype=np.float32
        )
        for idx, raw_action in enumerate(mapping):
            if raw_action is None:
                continue
            mission = env.missions[raw_action]
            if mission is None:
                continue
            if mission.is_observed:
                obs_status = 1.0
            elif mission.obs_start_s > 0:
                obs_status = 0.5
            else:
                obs_status = 0.0
            w_start, w_end = env._get_next_vtw_times(mission.id)
            mission_feats[idx] = [
                obs_status,
                w_start / env.horizon_s,
                w_end / env.horizon_s,
                mission.obs_start_s / env.horizon_s if mission.obs_start_s > 0 else 0.0,
                mission.obs_end_s / env.horizon_s if mission.obs_end_s > 0 else 0.0,
                mission.priority / 10.0,
                1.0 if mission.is_dynamic else 0.0,
            ]

        sat_state = env.propagator.propagate(env.current_time_s)
        sat_feats = np.array([
            env.current_time_s / env.horizon_s,
            sat_state.latitude_deg / 90.0,
            sat_state.longitude_deg / 180.0,
            0.0,
        ], dtype=np.float32)
        return np.concatenate([mission_feats.flatten(), sat_feats]).astype(np.float32)

    def _decode_actions(self, actions: Dict[str, int]) -> Dict[str, int]:
        if not self._candidate_actions_enabled():
            return dict(actions)
        raw_actions = {}
        transfer_count = self._transfer_action_count()
        exposed_idle = self.candidate_action_top_k + transfer_count
        for aid in self.agent_ids:
            action = int(actions.get(aid, exposed_idle))
            if action == exposed_idle:
                raw_actions[aid] = self._raw_idle_action()
                continue
            if self.candidate_action_top_k <= action < exposed_idle:
                target_idx = action - self.candidate_action_top_k
                targets = self._transfer_target_list(aid)
                if 0 <= target_idx < len(targets):
                    raw_action = self._raw_transfer_action(aid, targets[target_idx])
                    raw_actions[aid] = raw_action if raw_action is not None else self._raw_idle_action()
                else:
                    raw_actions[aid] = self._raw_idle_action()
                continue
            mapping = self._candidate_action_maps.get(aid, [])
            if 0 <= action < len(mapping) and mapping[action] is not None:
                raw_actions[aid] = int(mapping[action])
            else:
                raw_actions[aid] = self._raw_idle_action()
        return raw_actions

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

        # 全局 episode 级指派: 为所有常规任务预分配归属卫星 (仅协同模式)
        self.task_owner = {}
        self._assign_load = {aid: 0 for aid in self.agent_ids}
        self._last_replan_time_s = 0.0
        self._owner_switch_counts = {}
        self._n_replans = 0
        self._n_owner_switches = 0
        self._n_replan_checks = 0
        self._n_stale_owner_events = 0
        self._released_mission_ids = set()
        self._deadline_release_mission_ids = set()
        self._rescued_mission_ids = set()
        self._deadline_rescue_mission_ids = set()
        self._candidate_action_maps = {aid: [] for aid in self.agent_ids}
        self._transfer_targets = {}
        if self.n_ground_stations > 0:
            self._ground_station_available_s[:] = [0.0] * self.n_ground_stations
            for env in self.envs.values():
                env.set_ground_station_state(self._ground_station_available_s)
        if self.coordinate and self.episode_assignment:
            self._assign_tasks(self._all_known_missions())

        # 对外暴露 full action 或 Top-K candidate action。
        for agent_id in self.agent_ids:
            obs, info = self._expose_obs_info(agent_id)
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
        raw_actions = self._decode_actions(actions)
        results = {}
        prev_load = {aid: len(self.envs[aid].schedule_log) for aid in self.agent_ids}
        prev_observed = self._completed_mission_ids() if self.coordinate else set()

        # 1) 冲突解决 (优化路线图 A1+A2/A3+B6): 负载感知的贪心拍卖 + 败者改派.
        #    协同模式下用边际价值竞价择优指派, 抢输者改派次优任务;
        #    无协同 baseline 下原样返回各卫星动作 (不去冲突 → 可能重复观测).
        resolved_actions = self._resolve_actions(raw_actions)
        action_owner_before = {}
        if self.coordinate and self.episode_assignment:
            for agent_id, action in resolved_actions.items():
                if 0 <= action < self.max_action_dim:
                    mission = self.envs[agent_id].missions[action]
                    if mission is not None:
                        action_owner_before[agent_id] = (
                            mission.id,
                            self.task_owner.get(mission.id),
                        )

        # 2) 每颗卫星执行（已去冲突的）动作
        prev_schedule_lens = {aid: len(self.envs[aid].schedule_log) for aid in self.agent_ids}
        prev_ground_available = list(self._ground_station_available_s)
        pending_transfer_actions: Dict[str, int] = {}
        for agent_id in self.agent_ids:
            env = self.envs[agent_id]
            action = resolved_actions[agent_id]
            if self._is_raw_transfer_action(agent_id, action):
                env._insert_arrived_dynamic_missions()
                pending_transfer_actions[agent_id] = action
                obs, reward, term, trunc, info = (
                    env._build_observation(),
                    0.0,
                    env._all_missions_done() or env.current_time_s >= env.horizon_s,
                    False,
                    {"pending_inter_satellite_transfer": True},
                )
            else:
                obs, reward, term, trunc, info = env.step(action)
            results[agent_id] = (obs, reward, term, trunc, info)
        if self.n_ground_stations > 0 and self.downlink_time_s > 0:
            self._rebatch_new_downlinks(prev_schedule_lens, prev_ground_available, results)
        for agent_id, action in pending_transfer_actions.items():
            results[agent_id] = self._step_transfer_action(agent_id, action)

        # 3) 同步观测状态 (仅协同模式: 一颗星完成则全体知晓, 避免重复)
        if self.coordinate:
            self._sync_mission_status()
            self._track_release_rescues(action_owner_before)

        # 3.5) 动态任务到达后做增量指派 (在当前负载基础上继续均衡)
        new_missions = []
        if self.coordinate and self.episode_assignment:
            new_missions = [
                m for m in self._all_known_missions()
                if m.id not in self.task_owner
            ]
            if new_missions:
                self._refresh_assignment_load()
                self._assign_tasks(new_missions)
            self._maybe_reassign_tasks(dynamic_event=bool(new_missions))

        # 3.8) 多智能体奖励塑形 (仅协同模式): 团队奖励 + 负载均衡 + 团队完成 bonus.
        if self.coordinate:
            results = self._shape_multi_agent_rewards(results, prev_load, prev_observed)

        # 4) 用同步后的状态重新构建观测和掩码 (协同模式叠加所有权掩码)
        for agent_id, env in self.envs.items():
            obs, info = self._expose_obs_info(agent_id)
            old_result = results[agent_id]
            results[agent_id] = (
                obs,
                old_result[1],
                old_result[2],
                old_result[3],
                {**old_result[4], **info},
            )

        return results

    def _rebatch_new_downlinks(
        self,
        prev_schedule_lens: Dict[str, int],
        prev_ground_available: List[float],
        results: Dict[str, Tuple[np.ndarray, float, bool, bool, Dict]],
    ) -> None:
        """
        同一 multi-agent step 内可能有多颗卫星同时完成观测。

        单星环境在各自 step 中会立即申请基站,但多星 step 是按 agent_id 顺序
        串行调用的。这里把本步新增的观测记录收集起来,按真实 obs_end_s 重新
        分配共享基站,避免 Python 遍历顺序影响下传完成时间和奖励。
        """
        self._ground_station_available_s[:] = list(prev_ground_available)
        new_records = []
        for aid in self.agent_ids:
            env = self.envs[aid]
            start = prev_schedule_lens.get(aid, len(env.schedule_log))
            for record in env.schedule_log[start:]:
                new_records.append((record.obs_end_s, aid, record))
        new_records.sort(key=lambda item: (item[0], item[1], item[2].mission_id))

        for _, aid, record in new_records:
            env = self.envs[aid]
            old_reward = record.reward
            mission = self._mission_for_agent(aid, record.mission_id)
            latest_end_s = min(env.horizon_s, mission.deadline_s) if mission is not None else env.horizon_s
            downlink_start, downlink_end, station_id = env._schedule_downlink(
                record.obs_end_s,
                latest_end_s=latest_end_s,
            )
            record.downlink_start_s = downlink_start
            record.downlink_end_s = downlink_end
            record.ground_station_id = station_id

            if mission is not None:
                mission.downlink_start_s = downlink_start
                mission.downlink_end_s = downlink_end
                mission.ground_station_id = station_id
                mission.is_downlinked = (
                    station_id >= 0
                    and downlink_end <= env.horizon_s
                    and downlink_end <= mission.deadline_s
                )
                completion_time_s = downlink_end
                if mission.is_downlinked:
                    record.reward = env.compute_reward(
                        mission,
                        completion_time_s,
                        off_nadir_deg=record.off_nadir_deg,
                    )
                else:
                    record.reward = env.rw_cfg.penalty_deadline_miss
                storage_release_s, storage_reason = env._storage_release_from_delivery(
                    mission,
                    record.obs_end_s,
                    downlink_end,
                    station_id,
                )
                record.storage_release_s = storage_release_s
                record.storage_release_reason = storage_reason
                env._set_storage_record(
                    mission_id=record.mission_id,
                    storage_start_s=record.obs_end_s,
                    storage_release_s=storage_release_s,
                    release_reason=storage_reason,
                )

            if aid in results and record.reward != old_reward:
                obs, reward, term, trunc, info = results[aid]
                results[aid] = (obs, float(reward + record.reward - old_reward), term, trunc, info)

    def _transferable_records(self, source_aid: str) -> List[Any]:
        """返回源星当前星上仍未交付、可一次性转发的图片记录。"""
        if not self._transfer_actions_enabled():
            return []
        source_env = self.envs[source_aid]
        t = float(source_env.current_time_s)
        records = []
        for record in source_env.schedule_log:
            mission = self._mission_for_agent(source_aid, record.mission_id)
            if mission is None:
                continue
            if getattr(record, "relay_satellite_name", ""):
                continue
            if not (record.storage_start_s <= t < record.storage_release_s):
                continue
            records.append(record)
        records.sort(key=lambda r: (r.obs_end_s, r.mission_id))
        return records

    def _preview_transfer_plan(
        self,
        source_aid: str,
        target_aid: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        预览"源星全部当前图片"转给目标星后的下传计划。

        返回 None 表示动作不可行;否则返回每张图片的目标星下传预约方案。
        """
        if source_aid == target_aid or target_aid not in self.envs:
            return None
        records = self._transferable_records(source_aid)
        if not records:
            return None
        source_env = self.envs[source_aid]
        target_env = self.envs[target_aid]
        transfer_start = float(source_env.current_time_s)
        transfer_end = transfer_start + self.inter_satellite_transfer_time_s
        if transfer_end > source_env.horizon_s:
            return None

        if target_env.storage_limited:
            active = target_env._onboard_image_count(transfer_end)
            if active + len(records) > target_env.satellite_storage_capacity:
                return None

        station_available = list(self._ground_station_available_s)
        plan = []
        for record in records:
            mission = self._mission_for_agent(source_aid, record.mission_id)
            if mission is None:
                return None
            latest_end_s = min(source_env.horizon_s, mission.deadline_s)
            if transfer_end >= latest_end_s:
                return None
            dl_start, dl_end, station_id = target_env._find_downlink_slot(
                transfer_end,
                latest_end_s=latest_end_s,
                station_available_s=station_available,
            )
            if station_id < 0:
                return None
            station_available[station_id] = dl_end
            plan.append({
                "record": record,
                "mission": mission,
                "transfer_start": transfer_start,
                "transfer_end": transfer_end,
                "downlink_start": dl_start,
                "downlink_end": dl_end,
                "ground_station_id": station_id,
            })
        return plan

    def _can_transfer_all_images(self, source_aid: str, target_aid: str) -> bool:
        return self._preview_transfer_plan(source_aid, target_aid) is not None

    def _step_transfer_action(self, source_aid: str, action: int):
        """执行智能体选择的全量星间转发动作。"""
        source_env = self.envs[source_aid]
        source_env._insert_arrived_dynamic_missions()
        target_aid = self._raw_transfer_target(source_aid, action)
        if target_aid is None:
            obs = source_env._build_observation()
            return obs, self.envs[source_aid].rw_cfg.penalty_invalid, False, False, {
                "invalid_transfer": True,
            }

        plan = self._preview_transfer_plan(source_aid, target_aid)
        if not plan:
            obs = source_env._build_observation()
            return obs, self.envs[source_aid].rw_cfg.penalty_invalid, False, False, {
                "invalid_transfer": True,
                "relay_target": target_aid,
            }

        target_env = self.envs[target_aid]
        total_delta_reward = 0.0
        transferred = 0
        for item in plan:
            record = item["record"]
            mission = item["mission"]
            old_reward = float(record.reward)
            transfer_start = float(item["transfer_start"])
            transfer_end = float(item["transfer_end"])
            dl_start = float(item["downlink_start"])
            dl_end = float(item["downlink_end"])
            station_id = int(item["ground_station_id"])

            # 预约真实共享基站窗口。若并发动作已占用该窗口,再次检查并重新找最早可行窗口。
            latest_end_s = min(source_env.horizon_s, mission.deadline_s)
            dl_start, dl_end, station_id = target_env._schedule_downlink(
                transfer_end,
                latest_end_s=latest_end_s,
            )
            if station_id < 0:
                continue

            record.relay_satellite_name = target_aid
            record.relay_start_s = transfer_start
            record.relay_end_s = transfer_end
            record.downlink_start_s = dl_start
            record.downlink_end_s = dl_end
            record.ground_station_id = station_id
            record.storage_release_s = transfer_end
            record.storage_release_reason = "relay_transfer"

            source_env._set_storage_record(
                mission_id=record.mission_id,
                storage_start_s=record.obs_end_s,
                storage_release_s=transfer_end,
                release_reason="relay_transfer",
            )
            target_env._set_storage_record(
                mission_id=record.mission_id,
                storage_start_s=transfer_end,
                storage_release_s=dl_end,
                release_reason="relay_downlink",
                source_satellite_name=source_aid,
            )

            mission.relay_satellite_name = target_aid
            mission.relay_start_s = transfer_start
            mission.relay_end_s = transfer_end
            mission.downlink_start_s = dl_start
            mission.downlink_end_s = dl_end
            mission.ground_station_id = station_id
            mission.is_downlinked = dl_end <= latest_end_s
            if mission.is_downlinked:
                record.reward = source_env.compute_reward(
                    mission,
                    dl_end,
                    off_nadir_deg=record.off_nadir_deg,
                )
            else:
                record.reward = source_env.rw_cfg.penalty_deadline_miss
            total_delta_reward += float(record.reward - old_reward)
            transferred += 1

        if transferred == 0:
            obs = source_env._build_observation()
            return obs, source_env.rw_cfg.penalty_invalid, False, False, {
                "invalid_transfer": True,
                "relay_target": target_aid,
            }

        source_env.current_time_s = min(
            source_env.horizon_s,
            max(source_env.current_time_s, plan[0]["transfer_end"]),
        )
        obs = source_env._build_observation()
        done = source_env._all_missions_done() or source_env.current_time_s >= source_env.horizon_s
        info = {
            "inter_satellite_transfer": True,
            "relay_target": target_aid,
            "n_relay_images_sent": float(transferred),
            "transfer_all_images": True,
        }
        return obs, float(total_delta_reward), bool(done), False, info

    def _try_inter_satellite_transfers(
        self,
        prev_schedule_lens: Dict[str, int],
        results: Dict[str, Tuple[np.ndarray, float, bool, bool, Dict]],
    ) -> None:
        """
        历史保留的自动星间转发 fallback,当前默认不再由 step 调用。

        当前版本只在源卫星没有可行直接下传窗口时尝试转发。转发本身使用固定耗时,
        暂不建模星间链路可见窗口; 接收卫星必须有空余存储,且能在 deadline 前
        找到卫星-基站下传窗口。
        """
        if self.inter_satellite_transfer_time_s <= 0:
            return
        for source_aid in self.agent_ids:
            source_env = self.envs[source_aid]
            start_idx = prev_schedule_lens.get(source_aid, len(source_env.schedule_log))
            for record in source_env.schedule_log[start_idx:]:
                mission = self._mission_for_agent(source_aid, record.mission_id)
                if mission is None or mission.is_downlinked or record.ground_station_id >= 0:
                    continue
                transfer_start = record.obs_end_s
                transfer_end = transfer_start + self.inter_satellite_transfer_time_s
                latest_end_s = min(source_env.horizon_s, mission.deadline_s)
                if transfer_end >= latest_end_s:
                    continue

                best = None
                for target_aid in self.agent_ids:
                    if target_aid == source_aid:
                        continue
                    target_env = self.envs[target_aid]
                    if not target_env._has_storage_capacity(transfer_end):
                        continue
                    dl_start, dl_end, station_id = target_env._find_downlink_slot(
                        transfer_end,
                        latest_end_s=latest_end_s,
                    )
                    if station_id < 0:
                        continue
                    cand = (dl_end, dl_start, station_id, target_aid)
                    if best is None or cand < best:
                        best = cand

                if best is None:
                    continue

                old_reward = record.reward
                dl_end, dl_start, station_id, target_aid = best
                target_env = self.envs[target_aid]
                # 真正预约接收星的下传窗口。
                dl_start, dl_end, station_id = target_env._schedule_downlink(
                    transfer_end,
                    latest_end_s=latest_end_s,
                )
                if station_id < 0:
                    continue

                record.relay_satellite_name = target_aid
                record.relay_start_s = transfer_start
                record.relay_end_s = transfer_end
                record.downlink_start_s = dl_start
                record.downlink_end_s = dl_end
                record.ground_station_id = station_id
                record.storage_release_s = transfer_end
                record.storage_release_reason = "relay_transfer"

                source_env._set_storage_record(
                    mission_id=record.mission_id,
                    storage_start_s=record.obs_end_s,
                    storage_release_s=transfer_end,
                    release_reason="relay_transfer",
                )
                target_env._set_storage_record(
                    mission_id=record.mission_id,
                    storage_start_s=transfer_end,
                    storage_release_s=dl_end,
                    release_reason="relay_downlink",
                    source_satellite_name=source_aid,
                )

                mission.relay_satellite_name = target_aid
                mission.relay_start_s = transfer_start
                mission.relay_end_s = transfer_end
                mission.downlink_start_s = dl_start
                mission.downlink_end_s = dl_end
                mission.ground_station_id = station_id
                mission.is_downlinked = dl_end <= latest_end_s
                if mission.is_downlinked:
                    record.reward = source_env.compute_reward(
                        mission,
                        dl_end,
                        off_nadir_deg=record.off_nadir_deg,
                    )
                else:
                    record.reward = source_env.rw_cfg.penalty_deadline_miss

                if source_aid in results and record.reward != old_reward:
                    obs, reward, term, trunc, info = results[source_aid]
                    info = {
                        **info,
                        "inter_satellite_transfer": True,
                        "relay_target": target_aid,
                    }
                    results[source_aid] = (
                        obs,
                        float(reward + record.reward - old_reward),
                        term,
                        trunc,
                        info,
                    )

    # ===================================================================
    # 冲突解决: 负载感知贪心拍卖 + 败者改派 (优化路线图 A1+A2/A3+B6)
    # ===================================================================
    def set_eval_mode(self, flag: bool = True):
        """切换评估模式. 评估期启用 A1 败者改派以最大化吞吐; 训练期关闭以保信用分配。"""
        self.eval_mode = flag

    # ===================================================================
    # 全局 episode 级任务指派 (优化路线图 A3-episode / G24)
    # ===================================================================
    def _team_current_time_s(self) -> float:
        """全体卫星的当前调度前沿时间。"""
        if not self.envs:
            return 0.0
        return float(max(env.current_time_s for env in self.envs.values()))

    def _all_known_missions_by_id(self) -> Dict[int, Any]:
        """返回所有卫星已经插入/知道的任务, 按 mission_id 去重。"""
        missions: Dict[int, Any] = {}
        for env in self.envs.values():
            for mission in env.missions:
                if mission is None:
                    continue
                current = missions.get(int(mission.id))
                if current is None or (mission.is_observed and not current.is_observed):
                    missions[int(mission.id)] = mission
        return missions

    def _all_known_missions(self) -> List[Any]:
        return list(self._all_known_missions_by_id().values())

    def _mission_from_any_env(self, mission_id: int):
        return self._all_known_missions_by_id().get(int(mission_id))

    def _mission_for_agent(self, agent_id: str, mission_id: int):
        env = self.envs[agent_id]
        for mission in env.missions:
            if mission is not None and mission.id == mission_id:
                return mission
        return self._mission_from_any_env(mission_id)

    def _mission_observed_anywhere(self, mission_id: int) -> bool:
        for env in self.envs.values():
            for mission in env.missions:
                if mission is not None and mission.id == mission_id and mission.is_observed:
                    return True
        return False

    def _mission_completed_anywhere(self, mission_id: int) -> bool:
        for env in self.envs.values():
            for mission in env.missions:
                if mission is not None and mission.id == mission_id and env._mission_completed(mission):
                    return True
        return False

    def _ensure_mission_vtw(self, agent_id: str, mission) -> None:
        """确保某颗卫星已经为已知任务计算 VTW。"""
        if mission is None:
            return
        env = self.envs[agent_id]
        if mission.id not in env.mission_vtw:
            env._compute_vtw_for_missions([mission])

    def _mission_alive_for_any_agent(self, mission) -> bool:
        if mission is None:
            return False
        return any(env.current_time_s < mission.deadline_s for env in self.envs.values())

    def _task_quality(self, agent_id: str, mission) -> Optional[float]:
        """
        卫星 agent_id 在整个 horizon 内对 mission 的观测质量 (∈[0,1], 越大越好);
        返回 None 表示该星全程无可行 VTW (无法负责该任务)。
        质量取所有可行窗口中最小 off-nadir 对应的值 (最佳成像几何)。
        """
        env = self.envs[agent_id]
        self._ensure_mission_vtw(agent_id, mission)
        best_off = None
        for vtw in env.mission_vtw.get(mission.id, []):
            if vtw.end_time - vtw.start_time < mission.duration_s:
                continue
            if vtw.end_time < mission.earliest_time_s + mission.duration_s:
                continue
            earliest_obs_end = max(vtw.start_time, mission.earliest_time_s) + mission.duration_s
            if earliest_obs_end <= min(vtw.end_time, mission.deadline_s):
                if best_off is None or vtw.off_nadir_deg < best_off:
                    best_off = vtw.off_nadir_deg
        if best_off is None:
            return None
        max_roll = max(env.sat_config.max_roll_deg, 1e-6)
        return 1.0 - min(best_off / max_roll, 1.0)

    def _assign_tasks(self, missions: list):
        """
        全局广义指派: 为每个任务选一颗负责卫星, 兼顾观测质量与负载均衡。

        策略 (近似平衡广义指派):
          - 候选 = 全程可行的卫星; 无候选的任务跳过 (物理不可达)。
          - "最少候选优先": 只有一颗星能做的硬任务先指派, 灵活任务后填以平衡负载。
          - 容量比例: 按每星候选任务质量之和估算可服务容量, 覆盖能力强的星获得更高目标份额。
          - 打分 = 质量 − assign_w_load·负载压力; 负载惩罚使任务流向相对空闲卫星 (B6 全局版)。
        结果写入 self.task_owner; self._assign_load 累计各星负载 (供动态任务增量指派延续)。
        """
        pending = []
        for m in missions:
            if m.id in self.task_owner:
                continue
            cands = []
            for aid in self.agent_ids:
                q = self._task_quality(aid, m)
                if q is not None:
                    cands.append((aid, q))
            if cands:
                pending.append((m, cands))
        context_encoder = self._effective_assignment_context_encoder()
        if context_encoder in {"lstm", "gru"}:
            # 最少候选 + 最早截止优先: 让序列 scorer 先看到最受约束/最紧迫的任务。
            pending.sort(key=lambda x: (len(x[1]), x[0].deadline_s, x[0].earliest_time_s, -x[0].priority))
        else:
            # 保持 heuristic/MLP 旧行为, 维护历史实验可比性。
            pending.sort(key=lambda x: len(x[1]))
        self._prepare_assignment_sequence_context(pending)
        targets = self._assignment_targets(pending)
        for mission, cands in pending:
            best_aid = max(
                cands,
                key=lambda c: self._assignment_score(
                    c[0], mission, c[1], targets, n_candidates=len(cands)
                )
            )[0]
            self.task_owner[mission.id] = best_aid
            self._assign_load[best_aid] += 1

    def _maybe_reassign_tasks(self, dynamic_event: bool = False):
        """
        Rolling horizon 重分配入口。

        默认无触发器且 interval=0 时完全关闭。开启后只重分配未完成任务,
        保留已完成记录和当前 step 的动作执行结果, 因而影响的是下一步 action mask。
        """
        if not self.coordinate or not self.episode_assignment:
            return
        if self.assignment_replan_interval_s <= 0 and not self.assignment_replan_triggers:
            self._update_stale_owner_diagnostics()
            return

        self._n_replan_checks += 1
        current_time = self._team_current_time_s()
        elapsed_since_replan = current_time - self._last_replan_time_s
        periodic_enabled = (
            self.assignment_replan_interval_s > 0
            and (not self.assignment_replan_triggers or "periodic" in self.assignment_replan_triggers)
        )
        event_cooldown = (
            self.assignment_replan_interval_s > 0
            and elapsed_since_replan < self.assignment_replan_interval_s
        )
        reasons = []

        if periodic_enabled and elapsed_since_replan >= self.assignment_replan_interval_s:
            reasons.append("periodic")
        if not event_cooldown:
            if dynamic_event and "dynamic" in self.assignment_replan_triggers:
                reasons.append("dynamic")
            if "stale_owner" in self.assignment_replan_triggers and self._has_stale_owner():
                reasons.append("stale_owner")
            if "deadline" in self.assignment_replan_triggers and self._has_deadline_risk():
                reasons.append("deadline")
            if "imbalance" in self.assignment_replan_triggers and self._assignment_load_cv() > 0.5:
                reasons.append("imbalance")

        if not reasons:
            self._update_stale_owner_diagnostics()
            return

        missions = self._eligible_replan_missions(current_time)
        if not missions:
            self._last_replan_time_s = current_time
            self._update_stale_owner_diagnostics()
            return

        switched = self._reassign_tasks(missions)
        self._last_replan_time_s = current_time
        self._n_replans += 1
        self._n_owner_switches += switched
        self._update_stale_owner_diagnostics()

    def replan_assignment(self, mission_ids: Optional[List[int]] = None, reason: str = "external") -> int:
        """
        Public high-level replan API for assignment managers.

        Returns the number of owner switches. This is the stable entry point for
        future trainable high-level policies; default behavior remains unchanged
        unless a caller invokes it or rolling triggers are enabled.
        """
        if not self.coordinate or not self.episode_assignment:
            return 0
        current_time = self._team_current_time_s()
        missions = self._eligible_replan_missions(current_time)
        if mission_ids is not None:
            allowed_ids = set(mission_ids)
            missions = [m for m in missions if m.id in allowed_ids]
        if not missions:
            return 0
        switched = self._reassign_tasks(missions)
        self._last_replan_time_s = current_time
        self._n_replans += 1
        self._n_owner_switches += switched
        return switched

    def set_task_owner(self, mission_id: int, agent_id: str, count_switch: bool = True) -> bool:
        """
        Set one task owner after validating the satellite is a feasible candidate.
        Returns True if ownership changed or was newly assigned.
        """
        if agent_id not in self.agent_ids:
            raise ValueError(f"未知 agent_id={agent_id!r}; 可选: {self.agent_ids}")
        mission = self._mission_from_any_env(mission_id)
        if mission is None or mission.is_observed:
            return False
        if self._task_quality_window(agent_id, mission) is None:
            return False
        old_owner = self.task_owner.get(mission_id)
        if old_owner == agent_id:
            return False
        self.task_owner[mission_id] = agent_id
        if count_switch and old_owner is not None:
            self._owner_switch_counts[mission_id] = self._owner_switch_counts.get(mission_id, 0) + 1
        self._refresh_assignment_load()
        return True

    def get_assignment_state(
        self,
        missions: Optional[list] = None,
        pending: Optional[list] = None,
        targets: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Export a structured high-level assignment graph state.

        The state is JSON-like and intentionally framework-neutral, so it can be
        consumed by a rule manager today and by a trainable manager later.
        """
        current_time = self._team_current_time_s()
        if pending is None:
            if missions is None:
                missions = [
                    m for m in self._all_known_missions()
                    if m is not None and not m.is_observed and m.id in self.task_owner
                ]
            pending = []
            for mission in missions:
                cands = []
                for aid in self.agent_ids:
                    q = self._task_quality_window(aid, mission)
                    if q is not None:
                        cands.append((aid, q))
                if cands:
                    pending.append((mission, cands))
        if targets is None:
            targets = self._assignment_targets(pending)

        loads = {aid: len(self.envs[aid].schedule_log) for aid in self.agent_ids}
        total_load = max(sum(loads.values()), 1)
        satellites = []
        for aid in self.agent_ids:
            env = self.envs[aid]
            pending_owned = sum(
                1 for mission, _ in pending
                if self.task_owner.get(mission.id) == aid
            )
            satellites.append({
                "agent_id": aid,
                "load": loads[aid],
                "load_frac": loads[aid] / total_load,
                "target": float(targets.get(aid, 0.0)),
                "pending_owned": pending_owned,
                "current_time_s": float(env.current_time_s),
            })

        tasks = []
        edges = []
        for mission, cands in pending:
            old_owner = self.task_owner.get(mission.id)
            owner_stale = old_owner is not None and not self._has_future_feasible_window(old_owner, mission.id)
            slack_s = max(mission.deadline_s - max(current_time, mission.earliest_time_s), 0.0)
            deadline_pressure = 1.0 - np.clip(slack_s / max(self.horizon_s, 1.0), 0.0, 1.0)
            candidate_frac = np.clip(len(cands) / max(self.n_agents, 1), 0.0, 1.0)
            tasks.append({
                "mission_id": int(mission.id),
                "priority": float(mission.priority),
                "duration_s": float(mission.duration_s),
                "slack_s": float(slack_s),
                "deadline_pressure": float(deadline_pressure),
                "is_dynamic": bool(mission.is_dynamic),
                "candidate_frac": float(candidate_frac),
                "owner": old_owner,
                "owner_stale": bool(owner_stale),
                "switch_count": int(self._owner_switch_counts.get(mission.id, 0)),
            })
            for aid, quality in cands:
                load_pressure = self._load_pressure(aid, targets)
                score = self._assignment_score(aid, mission, quality, targets, n_candidates=len(cands))
                if old_owner is not None and aid != old_owner:
                    score -= self.assignment_switch_penalty
                edges.append({
                    "mission_id": int(mission.id),
                    "agent_id": aid,
                    "quality": float(quality),
                    "load_pressure": float(load_pressure),
                    "score": float(score),
                    "is_current_owner": aid == old_owner,
                })

        return {
            "current_time_s": float(current_time),
            "horizon_s": float(self.horizon_s),
            "manager_mode": self.assignment_manager_mode,
            "satellites": satellites,
            "tasks": tasks,
            "edges": edges,
        }

    def _eligible_replan_missions(self, current_time: float) -> list:
        """筛选允许滚动重分配的未完成任务。"""
        missions = []
        for mission in self._all_known_missions():
            if mission is None or mission.is_observed or mission.id not in self.task_owner:
                continue
            if not self._mission_alive_for_any_agent(mission):
                continue
            if self.assignment_max_switches_per_task == 0:
                continue
            if self._owner_switch_counts.get(mission.id, 0) >= self.assignment_max_switches_per_task:
                continue
            owner = self.task_owner.get(mission.id)
            owner_time = self.envs[owner].current_time_s if owner in self.envs else current_time
            if self._is_replan_locked(mission, owner_time):
                continue
            missions.append(mission)
        return missions

    def _is_replan_locked(self, mission, current_time: float) -> bool:
        """
        防抖锁定: owner 的下一次可行窗口即将到来时不临门换人。
        如果 owner 已无未来窗口, 不锁定, 交给 stale/deadline 机制救援。
        """
        if self.assignment_lock_window_s <= 0:
            return False
        owner = self.task_owner.get(mission.id)
        if owner is None:
            return False
        next_start = self._next_feasible_window_start(owner, mission.id, current_time)
        if next_start is None:
            return False
        return 0.0 <= next_start - current_time <= self.assignment_lock_window_s

    def _next_feasible_window_start(self, agent_id: str, mission_id: int, from_t: float) -> Optional[float]:
        """返回 agent 对任务从 from_t 起的下一次可完成窗口开始时间。"""
        env = self.envs[agent_id]
        mission = self._mission_for_agent(agent_id, mission_id)
        if mission is None or self._mission_observed_anywhere(mission_id):
            return None
        self._ensure_mission_vtw(agent_id, mission)
        starts = []
        for vtw in env.mission_vtw.get(mission.id, []):
            obs_start = max(vtw.start_time, from_t, mission.earliest_time_s)
            obs_end = obs_start + mission.duration_s
            if obs_end <= min(vtw.end_time, mission.deadline_s):
                starts.append(obs_start)
        return min(starts) if starts else None

    def _reassign_tasks(self, missions: list) -> int:
        """对给定未完成任务重新选择 owner, 返回发生 owner 切换的数量。"""
        self._refresh_assignment_load()
        pending = []
        for mission in missions:
            cands = []
            for aid in self.agent_ids:
                q = self._task_quality_window(aid, mission)
                if q is not None:
                    cands.append((aid, q))
            if cands:
                pending.append((mission, cands))
        pending.sort(key=lambda x: (len(x[1]), x[0].deadline_s, x[0].earliest_time_s, -x[0].priority))
        if not pending:
            return 0

        for mission, _ in pending:
            old_owner = self.task_owner.get(mission.id)
            if old_owner in self._assign_load:
                self._assign_load[old_owner] = max(0, self._assign_load.get(old_owner, 0) - 1)

        self._prepare_assignment_sequence_context(pending)
        targets = self._assignment_targets(pending)
        manager_proposals = self._assignment_manager_proposals(pending, targets)
        switched = 0
        for mission, cands in pending:
            old_owner = self.task_owner.get(mission.id)
            cand_ids = {aid for aid, _ in cands}
            proposed = manager_proposals.get(mission.id)
            if proposed in cand_ids:
                best_aid = proposed
            else:
                best_aid = max(
                    cands,
                    key=lambda c: (
                        self._assignment_score(c[0], mission, c[1], targets, n_candidates=len(cands))
                        - (self.assignment_switch_penalty if old_owner is not None and c[0] != old_owner else 0.0)
                    )
                )[0]
            if old_owner is None:
                self.task_owner[mission.id] = best_aid
                self._assign_load[best_aid] = self._assign_load.get(best_aid, 0) + 1
            elif best_aid != old_owner:
                self.task_owner[mission.id] = best_aid
                self._assign_load[best_aid] = self._assign_load.get(best_aid, 0) + 1
                self._owner_switch_counts[mission.id] = self._owner_switch_counts.get(mission.id, 0) + 1
                switched += 1
            else:
                self._assign_load[best_aid] = self._assign_load.get(best_aid, 0) + 1
        return switched

    def _assignment_manager_proposals(self, pending: list, targets: Dict[str, float]) -> Dict[int, str]:
        """Ask the optional high-level assignment manager for owner proposals."""
        if self._assignment_manager is None:
            return {}
        state = self.get_assignment_state(pending=pending, targets=targets)
        return self._assignment_manager.select_owners(state)

    def _task_quality_window(self, agent_id: str, mission) -> Optional[float]:
        """
        从当前时刻起、可选未来 horizon 内的观测质量。
        assignment_replan_horizon_s=0 表示看到任务 deadline/horizon 结束。
        """
        env = self.envs[agent_id]
        self._ensure_mission_vtw(agent_id, mission)
        from_t = max(env.current_time_s, mission.earliest_time_s)
        horizon_end = self.horizon_s
        if self.assignment_replan_horizon_s > 0:
            horizon_end = min(horizon_end, env.current_time_s + self.assignment_replan_horizon_s)
        best_off = None
        for vtw in env.mission_vtw.get(mission.id, []):
            obs_start = max(vtw.start_time, from_t)
            obs_end = obs_start + mission.duration_s
            latest_end = min(vtw.end_time, mission.deadline_s, horizon_end)
            if obs_end <= latest_end:
                if best_off is None or vtw.off_nadir_deg < best_off:
                    best_off = vtw.off_nadir_deg
        if best_off is None:
            return None
        max_roll = max(env.sat_config.max_roll_deg, 1e-6)
        return 1.0 - min(best_off / max_roll, 1.0)

    def _has_stale_owner(self) -> bool:
        """是否存在 owner 已无未来可行窗口的未完成任务。"""
        for mission in self._all_known_missions():
            if mission is None or mission.is_observed:
                continue
            owner = self.task_owner.get(mission.id)
            if owner is not None and not self._has_future_feasible_window(owner, mission.id):
                return True
        return False

    def _has_deadline_risk(self) -> bool:
        """是否存在进入 release 窗口但尚未完成的任务。"""
        if self.release_before_deadline_s <= 0:
            return False
        for mission in self._all_known_missions():
            if mission is None or mission.is_observed or mission.id not in self.task_owner:
                continue
            for env in self.envs.values():
                if env.current_time_s >= mission.deadline_s - self.release_before_deadline_s:
                    if env.current_time_s < mission.deadline_s:
                        return True
        return False

    def _assignment_load_cv(self) -> float:
        """当前各星已完成负载的变异系数。"""
        loads = np.array([len(self.envs[aid].schedule_log) for aid in self.agent_ids], dtype=np.float32)
        mean_load = float(loads.mean()) if loads.size else 0.0
        return float(loads.std() / mean_load) if mean_load > 0 else 0.0

    def _update_stale_owner_diagnostics(self):
        """累计 stale owner 诊断事件, 不改变策略行为。"""
        if self._has_stale_owner():
            self._n_stale_owner_events += 1

    def _track_release_rescues(self, action_owner_before: Dict[str, Tuple[int, Optional[str]]]):
        """统计非 owner 通过释放机制完成任务的情况。"""
        if not action_owner_before:
            return
        observed = self._observed_mission_ids()
        for aid, (mission_id, owner_before) in action_owner_before.items():
            if owner_before is None or owner_before == aid or mission_id not in observed:
                continue
            self._rescued_mission_ids.add(mission_id)
            if mission_id in self._deadline_release_mission_ids:
                self._deadline_rescue_mission_ids.add(mission_id)

    def _assignment_score(
        self,
        agent_id: str,
        mission: Mission,
        quality: float,
        targets: Dict[str, float],
        n_candidates: int,
    ) -> float:
        """候选边 (satellite, task) 的指派分数。"""
        load_pressure = self._load_pressure(agent_id, targets)
        heuristic = quality - self.assign_w_load * load_pressure
        if self.assignment_scorer == "heuristic":
            return heuristic
        if self.assignment_scorer == "cva":
            return self._assignment_cva_score(
                agent_id, mission, quality, load_pressure, n_candidates, heuristic
            )

        if self.assignment_scorer == "mlp":
            learned_score = self._assignment_mlp_score(
                agent_id, mission, quality, load_pressure, n_candidates
            )
        else:
            learned_score = self._assignment_context_score(
                agent_id, mission, quality, load_pressure, n_candidates
            )
        return (1.0 - self.assignment_scorer_mix) * heuristic + self.assignment_scorer_mix * learned_score

    def _prepare_assignment_sequence_context(self, pending: list):
        """为序列/集合 scorer 生成任务上下文; 其他 scorer 清空缓存。"""
        self._assignment_sequence_context = {}
        self._assignment_sat_context = {}
        context_encoder = self._effective_assignment_context_encoder()
        if context_encoder not in {"lstm", "gru", "transformer", "set_transformer", "gnn"} or self._assignment_sequence is None:
            return

        if context_encoder == "gnn":
            self._prepare_assignment_gnn_context(pending)
            return

        if context_encoder in {"transformer", "set_transformer"}:
            self._prepare_assignment_attention_context(pending)
            return

        w = self._assignment_sequence
        hidden = self.assignment_sequence_hidden_dim
        h = np.zeros(hidden, dtype=np.float32)
        c = np.zeros(hidden, dtype=np.float32)
        for mission, cands in pending:
            x = self._assignment_task_sequence_features(mission, cands)
            if context_encoder == "lstm":
                i = self._sigmoid(x @ w["w_ix"] + h @ w["w_ih"] + w["b_i"])
                f = self._sigmoid(x @ w["w_fx"] + h @ w["w_fh"] + w["b_f"])
                o = self._sigmoid(x @ w["w_ox"] + h @ w["w_oh"] + w["b_o"])
                g = np.tanh(x @ w["w_gx"] + h @ w["w_gh"] + w["b_g"])
                c = f * c + i * g
                h = o * np.tanh(c)
            else:
                z = self._sigmoid(x @ w["w_zx"] + h @ w["w_zh"] + w["b_z"])
                r = self._sigmoid(x @ w["w_rx"] + h @ w["w_rh"] + w["b_r"])
                n = np.tanh(x @ w["w_nx"] + (r * h) @ w["w_nh"] + w["b_n"])
                h = (1.0 - z) * h + z * n
            context = np.tanh(h @ w["w_out"] + w["b_out"])
            self._assignment_sequence_context[mission.id] = context.astype(np.float32)

    def _prepare_assignment_attention_context(self, pending: list):
        """为 Transformer/Set Transformer scorer 生成集合上下文。"""
        if not pending:
            return
        w = self._assignment_sequence
        context_encoder = self._effective_assignment_context_encoder()
        feats = np.stack([
            self._assignment_task_sequence_features(mission, cands)
            for mission, cands in pending
        ]).astype(np.float32)
        x = np.tanh(feats @ w["w_embed"] + w["b_embed"])
        if context_encoder == "transformer":
            positions = np.linspace(0.0, 1.0, num=x.shape[0], dtype=np.float32).reshape(-1, 1)
            x = x + positions * w["pos_scale"].reshape(1, -1)

        q = x @ w["w_q"]
        k = x @ w["w_k"]
        v = x @ w["w_v"]
        scale = max(np.sqrt(float(q.shape[-1])), 1.0)
        attn_logits = (q @ k.T) / scale
        attn = self._softmax(attn_logits, axis=1)
        attended = attn @ v
        if context_encoder == "set_transformer":
            pooled = attended.mean(axis=0, keepdims=True)
            attended = attended + pooled
        hidden = np.tanh(attended + np.tanh(attended @ w["w_ff"] + w["b_ff"]))
        contexts = np.tanh(hidden @ w["w_out"] + w["b_out"])
        for (mission, _), context in zip(pending, contexts):
            self._assignment_sequence_context[mission.id] = context.astype(np.float32)

    def _prepare_assignment_gnn_context(self, pending: list):
        """为 GNN scorer 生成卫星-任务二分图上下文。"""
        if not pending:
            return
        w = self._assignment_sequence
        task_base = {}
        task_candidate_count = {}
        sat_edges = {aid: [] for aid in self.agent_ids}

        for mission, cands in pending:
            task_feat = self._assignment_task_sequence_features(mission, cands)
            task_base[mission.id] = np.tanh(task_feat @ w["w_task"] + w["b_task"])
            task_candidate_count[mission.id] = len(cands)
            for aid, quality in cands:
                sat_edges.setdefault(aid, []).append((mission, quality))

        sat_base = {}
        for aid in self.agent_ids:
            load_norm = np.clip(self._assign_load.get(aid, 0) / max(len(self.task_owner) + 1, 1), 0.0, 1.0)
            qualities = [q for _, q in sat_edges.get(aid, [])]
            n_edges = len(qualities)
            sat_feat = np.array([
                load_norm,
                np.clip(n_edges / max(len(pending), 1), 0.0, 1.0),
                max(qualities) if qualities else 0.0,
                float(np.mean(qualities)) if qualities else 0.0,
                float(np.std(qualities)) if len(qualities) > 1 else 0.0,
            ], dtype=np.float32)
            sat_base[aid] = np.tanh(sat_feat @ w["w_sat"] + w["b_sat"])

        for mission, cands in pending:
            sat_msg = np.zeros_like(next(iter(sat_base.values())))
            for aid, quality in cands:
                sat_msg += sat_base[aid] * max(quality, 1e-6)
            sat_msg /= max(sum(max(q, 1e-6) for _, q in cands), 1e-6)
            context = np.tanh(task_base[mission.id] + sat_msg @ w["w_edge_task"])
            self._assignment_sequence_context[mission.id] = np.tanh(context @ w["w_out"] + w["b_out"]).astype(np.float32)

        for aid in self.agent_ids:
            edges = sat_edges.get(aid, [])
            if edges:
                task_msg = np.zeros_like(next(iter(task_base.values())))
                for mission, quality in edges:
                    scarcity = 1.0 / max(task_candidate_count.get(mission.id, 1), 1)
                    task_msg += task_base[mission.id] * max(quality, 1e-6) * scarcity
                task_msg /= max(sum(max(q, 1e-6) for _, q in edges), 1e-6)
            else:
                task_msg = np.zeros_like(next(iter(task_base.values())))
            self._assignment_sat_context[aid] = np.tanh(sat_base[aid] + task_msg @ w["w_edge_sat"]).astype(np.float32)

    def _assignment_task_sequence_features(self, mission: Mission, cands: list) -> np.ndarray:
        """任务级序列特征, 不含具体卫星, 用于 LSTM/GRU 读入任务流。"""
        qualities = [q for _, q in cands]
        priority_norm = np.clip(mission.priority / 10.0, 0.0, 1.0)
        duration_norm = np.clip(mission.duration_s / 60.0, 0.0, 1.0)
        slack_s = max(mission.deadline_s - mission.earliest_time_s, 0.0)
        slack_norm = np.clip(slack_s / max(self.horizon_s, 1.0), 0.0, 1.0)
        dynamic = 1.0 if mission.is_dynamic else 0.0
        candidate_frac = np.clip(len(cands) / max(self.n_agents, 1), 0.0, 1.0)
        best_quality = max(qualities) if qualities else 0.0
        mean_quality = float(np.mean(qualities)) if qualities else 0.0
        return np.array([
            priority_norm,
            duration_norm,
            slack_norm,
            dynamic,
            candidate_frac,
            best_quality,
            mean_quality,
        ], dtype=np.float32)

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))

    @staticmethod
    def _softmax(x, axis=-1):
        x = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(np.clip(x, -30.0, 30.0))
        return exp_x / np.maximum(exp_x.sum(axis=axis, keepdims=True), 1e-8)

    def _assignment_context_score(
        self,
        agent_id: str,
        mission: Mission,
        quality: float,
        load_pressure: float,
        n_candidates: int,
    ) -> float:
        """上下文 scorer: 任务序列/集合上下文 + 当前候选边特征。"""
        context = self._assignment_sequence_context.get(mission.id)
        if context is None:
            return self._assignment_mlp_score(agent_id, mission, quality, load_pressure, n_candidates)
        context_encoder = self._effective_assignment_context_encoder()

        env = self.envs[agent_id]
        priority_norm = np.clip(mission.priority / 10.0, 0.0, 1.0)
        slack_s = max(mission.deadline_s - max(env.current_time_s, mission.earliest_time_s), 0.0)
        slack_norm = np.clip(slack_s / max(self.horizon_s, 1.0), 0.0, 1.0)
        dynamic = 1.0 if mission.is_dynamic else 0.0
        candidate_frac = np.clip(n_candidates / max(self.n_agents, 1), 0.0, 1.0)
        base = quality + 0.2 * priority_norm + 0.1 * dynamic + 0.1 * (1.0 - slack_norm)
        context_term = (
            context[0] * quality
            + context[1] * priority_norm
            + context[2] * dynamic
            + context[3] * (1.0 - candidate_frac)
            - context[4] * load_pressure
        )
        if context_encoder == "gnn":
            sat_context = self._assignment_sat_context.get(agent_id)
            if sat_context is not None:
                context_term += (
                    sat_context[0] * quality
                    + sat_context[1] * (1.0 - load_pressure)
                    + sat_context[2] * priority_norm
                )
        return base - self.assign_w_load * load_pressure + 0.15 * float(context_term)

    def _assignment_cva_score(
        self,
        agent_id: str,
        mission: Mission,
        quality: float,
        load_pressure: float,
        n_candidates: int,
        heuristic: float,
    ) -> float:
        """
        CVA-MAPPO 的高层上下文价值分配分数。

        该分数保留可解释启发式作为锚点,再加入外循环上下文编码器估计的任务价值。
        因此第一版无需端到端训练高层策略,但已经能在任务分配阶段显式利用
        LSTM/GRU/Transformer/Set Transformer/GNN 上下文。
        """
        env = self.envs[agent_id]
        priority_norm = np.clip(mission.priority / 10.0, 0.0, 1.0)
        duration_norm = np.clip(mission.duration_s / 60.0, 0.0, 1.0)
        slack_s = max(mission.deadline_s - max(env.current_time_s, mission.earliest_time_s), 0.0)
        slack_norm = np.clip(slack_s / max(self.horizon_s, 1.0), 0.0, 1.0)
        deadline_pressure = 1.0 - slack_norm
        dynamic = 1.0 if mission.is_dynamic else 0.0
        candidate_frac = np.clip(n_candidates / max(self.n_agents, 1), 0.0, 1.0)
        scarcity = 1.0 - candidate_frac
        old_owner = self.task_owner.get(mission.id)
        is_current_owner = 1.0 if old_owner == agent_id else 0.0
        switch_count_norm = np.clip(
            self._owner_switch_counts.get(mission.id, 0)
            / max(self.assignment_max_switches_per_task, 1),
            0.0,
            1.0,
        )
        owner_stale = (
            old_owner is not None
            and not self._has_future_feasible_window(old_owner, mission.id)
        )
        stale_rescue = 1.0 if owner_stale and old_owner != agent_id else 0.0
        stale_keep_penalty = 1.0 if owner_stale and old_owner == agent_id else 0.0
        released_before = 1.0 if mission.id in self._released_mission_ids else 0.0
        next_start = self._next_feasible_window_start(agent_id, mission.id, env.current_time_s)
        if next_start is None:
            next_window_urgency = 0.0
        else:
            wait_s = max(next_start - env.current_time_s, 0.0)
            next_window_urgency = 1.0 - np.clip(wait_s / max(self.horizon_s, 1.0), 0.0, 1.0)

        # 可解释的边价值: 当前几何质量、任务价值、动态/截止压力和稀缺性,
        # 同时惩罚超过目标容量的卫星。
        value_prior = (
            0.55 * quality
            + 0.18 * priority_norm
            + 0.14 * dynamic
            + 0.14 * deadline_pressure
            + 0.10 * scarcity
            + 0.06 * next_window_urgency
            - self.assign_w_load * load_pressure
        )

        # 历史/滚动上下文: 保持稳定 owner,但当 owner 失效或临近释放时鼓励救援。
        history_value = (
            0.05 * is_current_owner
            + 0.16 * stale_rescue
            + 0.06 * released_before
            - 0.18 * stale_keep_penalty
            - 0.08 * switch_count_norm
            - 0.04 * duration_norm
        )

        context_encoder = self._effective_assignment_context_encoder()
        if context_encoder == "mlp":
            context_value = self._assignment_mlp_score(
                agent_id, mission, quality, load_pressure, n_candidates
            )
        else:
            context_value = self._assignment_context_score(
                agent_id, mission, quality, load_pressure, n_candidates
            )
        contextual_value = (
            value_prior
            + history_value
            + self.assignment_context_weight * context_value
        )
        alpha = self.assignment_scorer_mix
        return (1.0 - alpha) * heuristic + alpha * contextual_value

    def _assignment_mlp_score(
        self,
        agent_id: str,
        mission: Mission,
        quality: float,
        load_pressure: float,
        n_candidates: int,
    ) -> float:
        """轻量 MLP scorer: 第一版用于建立学习式分配器消融接口。"""
        if self._assignment_mlp is None:
            return quality - self.assign_w_load * load_pressure

        env = self.envs[agent_id]
        priority_norm = np.clip(mission.priority / 10.0, 0.0, 1.0)
        duration_norm = np.clip(mission.duration_s / 60.0, 0.0, 1.0)
        slack_s = max(mission.deadline_s - max(env.current_time_s, mission.earliest_time_s), 0.0)
        slack_norm = np.clip(slack_s / max(self.horizon_s, 1.0), 0.0, 1.0)
        candidate_frac = np.clip(n_candidates / max(self.n_agents, 1), 0.0, 1.0)
        load_norm = np.clip(self._assign_load.get(agent_id, 0) / max(len(self.task_owner) + 1, 1), 0.0, 1.0)
        dynamic = 1.0 if mission.is_dynamic else 0.0
        late_pressure = 1.0 - slack_norm
        x = np.array([
            quality,
            priority_norm,
            duration_norm,
            slack_norm,
            dynamic,
            candidate_frac,
            load_pressure,
            load_norm,
        ], dtype=np.float32)
        w = self._assignment_mlp
        h = np.tanh(x @ w["w1"] + w["b1"])
        neural = float(np.tanh((h @ w["w2"] + w["b2"])[0]))
        # 加入一个显式可解释残差, 让未训练 MLP 不至于完全随机化分配。
        residual = quality + 0.2 * priority_norm + 0.1 * dynamic + 0.1 * late_pressure - self.assign_w_load * load_pressure
        return residual + 0.25 * neural

    def _assignment_targets(self, pending: list) -> Dict[str, float]:
        """
        计算各星的目标指派份额。

        equal: 每星目标任务数相同。
        proportional: 按候选质量和估算容量, 覆盖能力强的卫星承担更多任务, 避免硬等额
        指派把任务压给物理窗口很弱的卫星而损失吞吐。
        """
        total_after = sum(self._assign_load.values()) + len(pending)
        if total_after <= 0:
            return {aid: 1.0 for aid in self.agent_ids}

        if self.assignment_capacity_mode == "equal":
            each = total_after / max(len(self.agent_ids), 1)
            return {aid: max(each, 1e-6) for aid in self.agent_ids}

        capacity = {aid: 0.0 for aid in self.agent_ids}
        for _, cands in pending:
            for aid, q in cands:
                capacity[aid] += max(q, 1e-6)
        total_capacity = sum(capacity.values())
        if total_capacity <= 0:
            each = total_after / max(len(self.agent_ids), 1)
            return {aid: max(each, 1e-6) for aid in self.agent_ids}

        return {
            aid: max(total_after * capacity[aid] / total_capacity, 1e-6)
            for aid in self.agent_ids
        }

    def _load_pressure(self, agent_id: str, targets: Dict[str, float]) -> float:
        """相对目标容量的负载压力; >1 表示该星已超过其容量份额。"""
        return self._assign_load.get(agent_id, 0) / max(targets.get(agent_id, 1.0), 1e-6)

    def _refresh_assignment_load(self):
        """
        动态任务增量指派前刷新负载基线。

        负载 = 已实际完成数 + 未完成任务的 owner backlog。这样 release 让非 owner 接手后,
        后续动态任务会参考真实执行负载, 不被初始 owner 表误导。
        """
        load = {aid: len(self.envs[aid].schedule_log) for aid in self.agent_ids}
        for m in self._all_known_missions():
            if m is None or m.is_observed:
                continue
            owner = self.task_owner.get(m.id)
            if owner in load:
                load[owner] += 1
        self._assign_load = load

    def _apply_ownership_mask(self, agent_id: str, mask: np.ndarray) -> np.ndarray:
        """
        所有权掩码: 各星只在自己负责的任务上行动 (从构造上消重 + 均衡负载)。
        归属其他卫星的任务默认屏蔽; idle 始终可用。

        截止前释放: 当任务临近 deadline, 或 owner 已无未来可行窗口时, 非 owner 可接手。
        这保留早期分工带来的去重/均衡, 同时回收硬所有权造成的末段吞吐损失。
        """
        env = self.envs[agent_id]
        for i in range(self.max_action_dim):
            m = env.missions[i]
            if m is None:
                continue
            owner = self.task_owner.get(m.id)
            if owner is not None and owner != agent_id and not self._ownership_released(agent_id, m):
                mask[i] = 0.0
        return mask

    def _ownership_released(self, agent_id: str, mission) -> bool:
        """判断非 owner 是否可以临时接手任务。"""
        owner = self.task_owner.get(mission.id)
        if owner is None or owner == agent_id or mission.is_observed:
            return True
        if self.release_before_deadline_s <= 0:
            return False

        env = self.envs[agent_id]
        near_deadline = env.current_time_s >= mission.deadline_s - self.release_before_deadline_s
        owner_has_future = self._has_future_feasible_window(owner, mission.id)
        released = near_deadline or not owner_has_future
        if released:
            self._released_mission_ids.add(mission.id)
            if near_deadline:
                self._deadline_release_mission_ids.add(mission.id)
        return released

    def _has_future_feasible_window(self, agent_id: str, mission_id: int) -> bool:
        """owner 从当前时刻起是否仍有机会在 deadline 前完成该任务。"""
        env = self.envs[agent_id]
        mission = self._mission_for_agent(agent_id, mission_id)
        if mission is None or self._mission_observed_anywhere(mission_id):
            return False
        self._ensure_mission_vtw(agent_id, mission)
        from_t = max(env.current_time_s, mission.earliest_time_s)
        for vtw in env.mission_vtw.get(mission.id, []):
            obs_start = max(vtw.start_time, from_t)
            obs_end = obs_start + mission.duration_s
            if obs_end <= min(vtw.end_time, mission.deadline_s):
                return True
        return False

    def _resolve_actions(self, actions: Dict[str, int]) -> Dict[str, int]:
        """
        协同冲突解决。

        - A2/A3 择优指派: 同一任务被多颗卫星争抢时, 用"边际价值"竞价,
          价值最高(优先级高 / off-nadir 小 / 负载轻)的卫星赢得该任务。
        - B6 负载均衡: 竞价中对当前负载较重的卫星扣分, 使任务流向较空闲的卫星。
        - A1 败者改派: 抢输的卫星不强制 idle, 而是改派到它当前可行且未被占用
          的"次优"任务, 杜绝浪费的空步。
        主动选择 idle(等待更好窗口)的卫星予以尊重, 不强行改派。

        无协同 baseline (coordinate=False): 原样返回各卫星动作, 不去冲突。
        """
        idle = self.max_action_dim
        if not self.coordinate:
            return dict(actions)

        # 预计算每颗卫星当前可行的(非 idle)动作集合.
        # 此刻各 env 状态尚未改变(动态任务在 env.step 内才插入), 掩码与策略所见一致.
        feasible: Dict[str, set] = {}
        for aid in self.agent_ids:
            mask = self.envs[aid]._build_action_mask()
            if self.episode_assignment:
                mask = self._apply_ownership_mask(aid, mask)
            feasible[aid] = set(np.nonzero(mask[:self.max_action_dim])[0].tolist())

        resolved = {aid: idle for aid in self.agent_ids}
        claimed = set()                 # 已被指派的任务槽位
        # 仅对"想行动"(非 idle)的卫星做指派; 主动 idle 的予以尊重
        desired = {aid: actions.get(aid, idle) for aid in self.agent_ids
                   if actions.get(aid, idle) != idle}
        for aid, action in list(desired.items()):
            if self._is_raw_transfer_action(aid, action):
                resolved[aid] = action
                desired.pop(aid, None)
        unassigned = set(desired.keys())

        max_iters = 4 * self.n_agents + 2
        for _ in range(max_iters):
            if not unassigned:
                break
            # 按当前期望任务分组; 期望已失效/被占用的卫星进入改派队列
            groups: Dict[int, List[str]] = {}
            to_reassign: List[str] = []
            for aid in list(unassigned):
                a = desired.get(aid, idle)
                if a == idle or a in claimed or self._obs_value(aid, a, feasible) is None:
                    to_reassign.append(aid)
                else:
                    groups.setdefault(a, []).append(aid)

            # 解决每个被争抢任务: 价值最高者赢, 其余进入改派 (A2/A3 + B6)
            for a, contenders in groups.items():
                winner = max(contenders, key=lambda x: self._obs_value(x, a, feasible))
                resolved[winner] = a
                claimed.add(a)
                unassigned.discard(winner)
                for loser in contenders:
                    if loser != winner:
                        to_reassign.append(loser)

            # 败者/失效者改派到次优可行任务 (A1, 仅评估期; 训练期保持 idle 以保信用分配)
            progressed = False
            do_reassign = self.reassign_losers and self.eval_mode
            for aid in to_reassign:
                nxt = self._next_best_action(aid, claimed, feasible) if do_reassign else None
                if nxt is None:
                    resolved[aid] = idle
                    unassigned.discard(aid)
                else:
                    desired[aid] = nxt
                    progressed = True

            if not groups and not progressed:
                # 无人可再指派, 剩余全部 idle
                for aid in unassigned:
                    resolved[aid] = idle
                break

        return resolved

    def _observed_mission_ids(self) -> set:
        observed = set()
        for env in self.envs.values():
            for m in env.missions:
                if m is not None and m.is_observed:
                    observed.add(m.id)
        return observed

    def _completed_mission_ids(self) -> set:
        completed = set()
        for env in self.envs.values():
            for m in env.missions:
                if m is not None and env._mission_completed(m):
                    completed.add(m.id)
        return completed

    def _shape_multi_agent_rewards(
        self,
        results: Dict[str, Tuple[np.ndarray, float, bool, bool, Dict]],
        prev_load: Dict[str, int],
        prev_observed: set,
    ) -> Dict[str, Tuple[np.ndarray, float, bool, bool, Dict]]:
        """
        C8/B5 奖励塑形。

        - team_reward_mix: 将个体奖励与全队平均奖励混合, 让每颗星感知团队收益。
        - load_balance_reward_coeff: 完成任务时, 低于平均负载的卫星获 bonus, 高负载卫星受轻惩罚。
        - team_completion_bonus: 本步新增完成任务时, 给全体一个小团队 bonus。
        """
        if (self.team_reward_mix <= 0
                and self.load_balance_reward_coeff == 0
                and self.team_completion_bonus == 0):
            return results

        raw_rewards = {aid: results[aid][1] for aid in self.agent_ids}
        team_mean = float(np.mean(list(raw_rewards.values()))) if raw_rewards else 0.0
        mean_prev_load = float(np.mean(list(prev_load.values()))) if prev_load else 0.0
        new_completed = len(self._completed_mission_ids() - prev_observed)
        completion_bonus = self.team_completion_bonus * new_completed

        shaped = {}
        mix = float(np.clip(self.team_reward_mix, 0.0, 1.0))
        for aid in self.agent_ids:
            obs, reward, term, trunc, info = results[aid]
            r = (1.0 - mix) * reward + mix * team_mean

            if self.load_balance_reward_coeff != 0 and reward > 0:
                # 正值鼓励相对空闲的卫星承担任务; 负值抑制已经偏忙的卫星继续抢任务.
                load_advantage = (mean_prev_load - prev_load.get(aid, 0)) / max(mean_prev_load, 1.0)
                r += self.load_balance_reward_coeff * load_advantage

            r += completion_bonus
            shaped[aid] = (obs, float(r), term, trunc, {
                **info,
                "raw_reward": reward,
                "team_mean_reward": team_mean,
                "team_completion_bonus": completion_bonus,
            })
        return shaped

    def _obs_value(self, agent_id: str, action: int,
                   feasible: Dict[str, set]) -> Optional[float]:
        """
        卫星 agent_id 此刻观测 action 槽位任务的"边际价值"(竞价分)。
        返回 None 表示该动作对该卫星不可行(不能用于指派)。
        价值 = w_priority·优先级 + w_quality·质量 − w_load·负载 (质量: off-nadir 越小越高)。
        """
        if action not in feasible.get(agent_id, ()):
            return None
        env = self.envs[agent_id]
        m = env.missions[action]
        if m is None or m.is_observed:
            return None
        # 当前可用 VTW 的 off-nadir (观测质量)
        off_nadir = None
        for vtw in env.mission_vtw.get(m.id, []):
            if vtw.start_time <= env.current_time_s <= vtw.end_time - m.duration_s:
                off_nadir = vtw.off_nadir_deg
                break
        if off_nadir is None:
            return None
        max_roll = max(env.sat_config.max_roll_deg, 1e-6)
        quality = 1.0 - min(off_nadir / max_roll, 1.0)        # ∈[0,1], 越大越好
        priority = m.priority / 10.0
        load = len(env.schedule_log)
        return (self.coord_w_priority * priority
                + self.coord_w_quality * quality
                - self.coord_w_load * load)

    def _next_best_action(self, agent_id: str, claimed: set,
                          feasible: Dict[str, set]) -> Optional[int]:
        """为卫星挑选当前可行且未被占用的最高价值任务 (A1 改派)。无可选则返回 None。"""
        best, best_val = None, float("-inf")
        for a in feasible.get(agent_id, ()):
            if a in claimed:
                continue
            v = self._obs_value(agent_id, a, feasible)
            if v is not None and v > best_val:
                best_val, best = v, a
        return best

    def _sync_mission_status(self):
        """
        跨卫星同步任务观测状态。

        如果任何一颗卫星完成了任务 M_j (is_observed=True)，
        则所有卫星的 M_j 都标记为已完成。
        这是多星协调的核心——避免重复观测。
        """
        # 收集所有已完成任务的 ID
        observed_state: Dict[int, Any] = {}
        for env in self.envs.values():
            for m in env.missions:
                if m is not None and m.is_observed:
                    old = observed_state.get(m.id)
                    if old is None:
                        observed_state[m.id] = m
                    elif (
                        getattr(m, "is_downlinked", False)
                        and (
                            not getattr(old, "is_downlinked", False)
                            or m.downlink_end_s < old.downlink_end_s
                        )
                    ):
                        observed_state[m.id] = m

        # 同步到所有卫星
        if observed_state:
            for env in self.envs.values():
                for m in env.missions:
                    if m is not None and m.id in observed_state:
                        src = observed_state[m.id]
                        m.is_observed = True
                        m.obs_start_s = src.obs_start_s
                        m.obs_end_s = src.obs_end_s
                        m.is_downlinked = getattr(src, "is_downlinked", False)
                        m.downlink_start_s = getattr(src, "downlink_start_s", -1.0)
                        m.downlink_end_s = getattr(src, "downlink_end_s", -1.0)
                        m.ground_station_id = getattr(src, "ground_station_id", -1)
                        m.relay_satellite_name = getattr(src, "relay_satellite_name", "")
                        m.relay_start_s = getattr(src, "relay_start_s", -1.0)
                        m.relay_end_s = getattr(src, "relay_end_s", -1.0)

    # ===================================================================
    # 全局状态 (仅训练时 Critic 使用)
    # ===================================================================
    def get_global_state(self) -> np.ndarray:
        """
        构建全局状态向量 (给集中式 Critic)。

        global_state_mode:
          - mean: 所有卫星局部观测均值, 维度不随卫星数增长(旧实现).
          - concat: 拼接所有卫星局部观测, 信息无损但维度随卫星数增长(D14).
        global_state_task_stats=True 时追加任务/负载统计(D16).
        """
        local_obs_list = []
        for agent_id in self.agent_ids:
            env = self.envs[agent_id]
            if self._candidate_actions_enabled():
                mapping = self._candidate_action_maps.get(agent_id)
                if mapping is None:
                    _, _ = self._expose_obs_info(agent_id)
                    mapping = self._candidate_action_maps.get(agent_id, [])
                local_obs_list.append(self._build_candidate_observation(agent_id, mapping))
            else:
                local_obs_list.append(env._build_observation())
        if self.global_state_mode == "concat":
            base = np.concatenate(local_obs_list, axis=0)
        else:
            base = np.mean(local_obs_list, axis=0)
        if self.global_state_task_stats:
            base = np.concatenate([base, self._global_task_stats()], axis=0)
        return base.astype(np.float32)

    def _global_task_stats(self) -> np.ndarray:
        """
        D16 任务级全局统计: 给 critic 补充局部观测 mean/concat 不容易表达的团队信息。

        包含:
          per-agent load fraction (n_agents)
          observed/pending/dynamic_pending/assigned_owner_pending fractions
          load CV
          duplicate rate so far
        """
        missions = self._all_known_missions()
        total = max(len(missions), 1)
        def _completed(mission):
            env = next(iter(self.envs.values()))
            return env._mission_completed(mission)

        observed = sum(1 for m in missions if _completed(m))
        pending = sum(1 for m in missions if not _completed(m))
        dynamic_pending = sum(1 for m in missions if m.is_dynamic and not _completed(m))
        assigned_pending = sum(
            1 for m in missions
            if not _completed(m) and m.id in self.task_owner
        )

        loads = np.array([len(self.envs[aid].schedule_log) for aid in self.agent_ids], dtype=np.float32)
        total_load = max(float(loads.sum()), 1.0)
        load_fracs = loads / total_load
        mean_load = float(loads.mean()) if loads.size else 0.0
        load_cv = float(loads.std() / mean_load) if mean_load > 0 else 0.0
        total_records = int(loads.sum())
        unique_records = len(self._observed_mission_ids())
        duplicate_rate = (total_records - unique_records) / max(total_records, 1)

        stats = np.array([
            observed / total,
            pending / total,
            dynamic_pending / total,
            assigned_pending / total,
            load_cv,
            duplicate_rate,
        ], dtype=np.float32)
        return np.concatenate([load_fracs, stats], axis=0)

    # ===================================================================
    # 评估指标 (聚合所有卫星)
    # ===================================================================
    def get_metrics(self) -> Dict[str, float]:
        """聚合所有卫星的调度指标"""
        # 合并所有卫星的调度记录
        all_observed_ids = set()
        all_completed_ids = set()
        total_reward = 0.0
        total_time = 0.0
        sub_metrics = {}

        for env in self.envs.values():
            metrics = env.get_metrics()
            sub_metrics[env.sat_config.name] = metrics
            total_reward += metrics["total_reward"]
            for record in env.schedule_log:
                all_observed_ids.add(record.mission_id)
                mission = self._mission_for_agent(env.sat_config.name, record.mission_id)
                deadline_s = mission.deadline_s if mission is not None else env.horizon_s
                if (not env.downlink_required) or (
                    record.downlink_end_s <= env.horizon_s
                    and record.downlink_end_s <= deadline_s
                ):
                    all_completed_ids.add(record.mission_id)

        # 基于共享任务池统计: 动态任务可能先被某一颗卫星插入,
        # 因此用所有卫星已知任务的并集作为基准。
        all_missions = self._all_known_missions()
        total_missions = len(all_missions)

        # feasible 划分 (论文 Table 4 口径): 任意一颗卫星对该任务有可用 VTW 即可行
        def _feasible_any(mission_id):
            for aid, env in self.envs.items():
                mission = self._mission_for_agent(aid, mission_id)
                if mission is None:
                    continue
                self._ensure_mission_vtw(aid, mission)
                if env._is_feasible(mission):
                    return True
            return False

        feasible_routine = [m for m in all_missions if not m.is_dynamic and _feasible_any(m.id)]
        feasible_dynamic = [m for m in all_missions if m.is_dynamic and _feasible_any(m.id)]
        feas_total = len(feasible_routine) + len(feasible_dynamic)

        # 统计哪些任务被任意一颗卫星完成
        observed_only_total = len(all_observed_ids)
        observed_total = len(all_completed_ids)
        feas_observed = sum(
            1 for m in (feasible_routine + feasible_dynamic) if m.id in all_completed_ids
        )
        feas_observed_only = sum(
            1 for m in (feasible_routine + feasible_dynamic) if m.id in all_observed_ids
        )
        routine_feas_done = sum(1 for m in feasible_routine if m.id in all_completed_ids)
        dynamic_feas_done = sum(1 for m in feasible_dynamic if m.id in all_completed_ids)

        # 全部任务口径 (诊断对照)
        routine_total = sum(1 for m in all_missions if not m.is_dynamic)
        dynamic_total = sum(1 for m in all_missions if m.is_dynamic)
        routine_done = sum(1 for m in all_missions if not m.is_dynamic and m.id in all_completed_ids)
        dynamic_done = sum(1 for m in all_missions if m.is_dynamic and m.id in all_completed_ids)

        # --- 协同质量指标 (体现多星协同 vs 无协同的核心差异) ---
        # 1) 重复观测: 总调度记录数 - 去重后完成数. 协同好 → ≈0
        total_records = sum(len(env.schedule_log) for env in self.envs.values())
        n_duplicates = total_records - observed_only_total
        duplicate_rate = n_duplicates / max(total_records, 1)

        # 2) 负载均衡: 各卫星完成任务数的方差/变异系数. 协同好 → 均衡(方差小)
        per_sat_counts = [len(env.schedule_log) for env in self.envs.values()]
        mean_load = float(np.mean(per_sat_counts)) if per_sat_counts else 0.0
        load_std = float(np.std(per_sat_counts)) if per_sat_counts else 0.0
        load_cv = load_std / mean_load if mean_load > 0 else 0.0  # 变异系数(越小越均衡)

        # 3) 平均观测质量(off-nadir, 跨所有卫星记录)
        all_off = [r.off_nadir_deg for env in self.envs.values() for r in env.schedule_log]
        avg_off_nadir = float(np.mean(all_off)) if all_off else 0.0

        # 4) 动态任务平均响应延迟(到达→完成, 取最早完成那次)
        dyn_delays = []
        for env in self.envs.values():
            for r in env.schedule_log:
                if r.is_dynamic and ((not env.downlink_required) or r.ground_station_id >= 0):
                    completion_time = r.downlink_end_s if env.downlink_required else r.obs_end_s
                    dyn_delays.append(completion_time - r.earliest_time_s)
        avg_dynamic_response_s = float(np.mean(dyn_delays)) if dyn_delays else 0.0
        downlink_queue_delays = [
            max(0.0, r.downlink_start_s - r.obs_end_s)
            for env in self.envs.values() for r in env.schedule_log
            if r.ground_station_id >= 0
        ]
        n_ground_station_windows = sum(
            len(vtws)
            for env in self.envs.values()
            for vtws in env.ground_station_vtw.values()
        )
        n_inter_satellite_transfers = sum(
            1 for env in self.envs.values()
            for r in env.schedule_log
            if getattr(r, "relay_satellite_name", "")
        )
        current_onboard_images = sum(
            metrics.get("current_onboard_images", 0.0)
            for metrics in sub_metrics.values()
        )
        max_onboard_images = max(
            [metrics.get("max_onboard_images", 0.0) for metrics in sub_metrics.values()] or [0.0]
        )
        avg_onboard_images = float(np.mean([
            metrics.get("avg_onboard_images", 0.0)
            for metrics in sub_metrics.values()
        ])) if sub_metrics else 0.0
        n_storage_expired_drops = sum(
            metrics.get("n_storage_expired_drops", 0.0)
            for metrics in sub_metrics.values()
        )
        n_relay_storage_images = sum(
            metrics.get("n_relay_storage_images", 0.0)
            for metrics in sub_metrics.values()
        )

        pending_assigned = [
            m for m in all_missions
            if not m.is_observed and m.id in self.task_owner
        ]
        stale_owner_now = sum(
            1 for m in pending_assigned
            if not self._has_future_feasible_window(self.task_owner[m.id], m.id)
        )
        owner_churn_rate = self._n_owner_switches / max(len(self.task_owner), 1)
        stale_owner_rate = stale_owner_now / max(len(pending_assigned), 1)
        deadline_rescue_rate = (
            len(self._deadline_rescue_mission_ids)
            / max(len(self._deadline_release_mission_ids), 1)
        )

        return {
            "total_reward": total_reward,
            # 论文 Table 4 口径: 分母 = feasible 任务
            "observation_success_rate": feas_observed / max(feas_total, 1),
            "dynamic_completion_rate": dynamic_feas_done / max(len(feasible_dynamic), 1),
            "routine_completion_rate": routine_feas_done / max(len(feasible_routine), 1),
            # 全部任务口径 (诊断对照)
            "observation_success_rate_raw": observed_total / max(total_missions, 1),
            "dynamic_completion_rate_raw": dynamic_done / max(dynamic_total, 1),
            "routine_completion_rate_raw": routine_done / max(routine_total, 1),
            "observation_only_success_rate": feas_observed_only / max(feas_total, 1),
            "observation_only_success_rate_raw": observed_only_total / max(total_missions, 1),
            "n_total_tasks": total_missions,
            "n_routine_tasks": routine_total,
            "n_dynamic_tasks": dynamic_total,
            "n_feasible_tasks": feas_total,
            "n_feasible_routine": len(feasible_routine),
            "n_feasible_dynamic": len(feasible_dynamic),
            "n_feasible_observed": feas_observed,
            "n_feasible_observed_only": feas_observed_only,
            "n_feasible_routine_done": routine_feas_done,
            "n_feasible_dynamic_done": dynamic_feas_done,
            "feasible_ratio": feas_total / max(total_missions, 1),
            "dynamic_feasible_ratio": len(feasible_dynamic) / max(dynamic_total, 1),
            # --- 协同质量指标 ---
            "n_duplicates": n_duplicates,           # 重复观测数 (协同好→0)
            "duplicate_rate": duplicate_rate,       # 重复观测率
            "load_balance_cv": load_cv,             # 负载变异系数 (越小越均衡)
            "avg_off_nadir_deg": avg_off_nadir,     # 平均观测质量 (越小越好)
            "avg_dynamic_response_s": avg_dynamic_response_s,  # 动态响应延迟 (越小越快)
            "n_observed": observed_only_total,
            "n_downlinked": observed_total,
            "n_pending_downlink": max(observed_only_total - observed_total, 0),
            "n_ground_stations": self.n_ground_stations,
            "downlink_time_s": self.downlink_time_s,
            "satellite_storage_capacity": self.satellite_storage_capacity,
            "current_onboard_images": current_onboard_images,
            "max_onboard_images": max_onboard_images,
            "avg_onboard_images": avg_onboard_images,
            "n_storage_expired_drops": n_storage_expired_drops,
            "n_relay_storage_images": n_relay_storage_images,
            "n_inter_satellite_transfers": n_inter_satellite_transfers,
            "inter_satellite_transfer_time_s": self.inter_satellite_transfer_time_s,
            "avg_downlink_queue_s": float(np.mean(downlink_queue_delays)) if downlink_queue_delays else 0.0,
            "n_ground_station_vtws": n_ground_station_windows,
            "avg_ground_station_vtws": (
                n_ground_station_windows / max(self.n_agents * self.n_ground_stations, 1)
                if self.downlink_time_s > 0 and self.n_ground_stations > 0 else 0.0
            ),
            "n_scheduled": observed_total,
            # --- Rolling assignment 诊断指标 ---
            "n_replans": self._n_replans,
            "n_replan_checks": self._n_replan_checks,
            "n_owner_switches": self._n_owner_switches,
            "n_tasks_switched": len(self._owner_switch_counts),
            "owner_churn_rate": owner_churn_rate,
            "stale_owner_rate": stale_owner_rate,
            "n_stale_owner_events": self._n_stale_owner_events,
            "n_released_tasks": len(self._released_mission_ids),
            "n_deadline_release_tasks": len(self._deadline_release_mission_ids),
            "n_rescued_tasks": len(self._rescued_mission_ids),
            "n_deadline_rescue_tasks": len(self._deadline_rescue_mission_ids),
            "deadline_rescue_rate": deadline_rescue_rate,
        }

    def is_done(self) -> bool:
        """检查是否所有卫星的 episode 都结束"""
        for env in self.envs.values():
            if env.current_time_s < env.horizon_s and not env._all_missions_done():
                return False
        return True

    @property
    def idle_action(self) -> int:
        if self._candidate_actions_enabled():
            return self.candidate_action_top_k + self._transfer_action_count()
        return self._raw_idle_action()
