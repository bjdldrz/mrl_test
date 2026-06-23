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
        assignment_mlp_hidden_dim: int = 16,
        assignment_mlp_seed: int = 42,
        assignment_sequence_hidden_dim: int = 16,
        team_reward_mix: float = 0.0,
        load_balance_reward_coeff: float = 0.0,
        team_completion_bonus: float = 0.0,
        global_state_mode: str = "mean",
        global_state_task_stats: bool = False,
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
        self.assignment_mlp_hidden_dim = max(1, int(assignment_mlp_hidden_dim))
        self.assignment_mlp_seed = int(assignment_mlp_seed)
        self.assignment_sequence_hidden_dim = max(1, int(assignment_sequence_hidden_dim))
        self._assignment_mlp = None
        self._assignment_sequence = None
        self._assignment_sequence_context: Dict[int, np.ndarray] = {}
        self._init_assignment_scorer()
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
        allowed = {"heuristic", "mlp", "lstm", "gru", "transformer", "set_transformer"}
        if self.assignment_scorer not in allowed:
            raise ValueError(
                f"未知 assignment_scorer={self.assignment_scorer!r}; "
                f"可选: {sorted(allowed)}"
            )
        if self.assignment_scorer != "mlp":
            if self.assignment_scorer in {"lstm", "gru"}:
                self._init_assignment_sequence_scorer()
            elif self.assignment_scorer in {"transformer", "set_transformer"}:
                self._init_assignment_attention_scorer()
            return

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

    def _init_assignment_sequence_scorer(self):
        """初始化确定性 LSTM/GRU 风格序列 scorer 权重。"""
        rng = np.random.RandomState(self.assignment_mlp_seed)
        in_dim = 7
        hidden = self.assignment_sequence_hidden_dim
        out_dim = 8
        def init(shape, scale=0.12):
            return rng.normal(0.0, scale, size=shape).astype(np.float32)

        self._assignment_sequence = {
            "w_out": init((hidden, out_dim)),
            "b_out": np.zeros(out_dim, dtype=np.float32),
        }
        if self.assignment_scorer == "lstm":
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
        if self.assignment_scorer == "transformer":
            self._assignment_sequence["pos_scale"] = init((hidden,), scale=0.04)

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

        # 全局 episode 级指派: 为所有常规任务预分配归属卫星 (仅协同模式)
        self.task_owner = {}
        self._assign_load = {aid: 0 for aid in self.agent_ids}
        if self.coordinate and self.episode_assignment:
            first_env = list(self.envs.values())[0]
            all_missions = [m for m in first_env.missions if m is not None]
            self._assign_tasks(all_missions)
            # 用所有权掩码重新过滤各星的初始动作掩码
            for agent_id, env in self.envs.items():
                obs, info = results[agent_id]
                mask = self._apply_ownership_mask(agent_id, env._build_action_mask())
                results[agent_id] = (obs, {**info, "action_mask": mask})

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
        prev_load = {aid: len(self.envs[aid].schedule_log) for aid in self.agent_ids}
        prev_observed = self._observed_mission_ids() if self.coordinate else set()

        # 1) 冲突解决 (优化路线图 A1+A2/A3+B6): 负载感知的贪心拍卖 + 败者改派.
        #    协同模式下用边际价值竞价择优指派, 抢输者改派次优任务;
        #    无协同 baseline 下原样返回各卫星动作 (不去冲突 → 可能重复观测).
        resolved_actions = self._resolve_actions(actions)

        # 2) 每颗卫星执行（已去冲突的）动作
        for agent_id in self.agent_ids:
            env = self.envs[agent_id]
            obs, reward, term, trunc, info = env.step(resolved_actions[agent_id])
            results[agent_id] = (obs, reward, term, trunc, info)

        # 3) 同步观测状态 (仅协同模式: 一颗星完成则全体知晓, 避免重复)
        if self.coordinate:
            self._sync_mission_status()

        # 3.5) 动态任务到达后做增量指派 (在当前负载基础上继续均衡)
        if self.coordinate and self.episode_assignment:
            first_env = list(self.envs.values())[0]
            new_missions = [m for m in first_env.missions
                            if m is not None and m.id not in self.task_owner]
            if new_missions:
                self._refresh_assignment_load()
                self._assign_tasks(new_missions)

        # 3.8) 多智能体奖励塑形 (仅协同模式): 团队奖励 + 负载均衡 + 团队完成 bonus.
        if self.coordinate:
            results = self._shape_multi_agent_rewards(results, prev_load, prev_observed)

        # 4) 用同步后的状态重新构建观测和掩码 (协同模式叠加所有权掩码)
        for agent_id, env in self.envs.items():
            obs = env._build_observation()
            mask = env._build_action_mask()
            if self.coordinate and self.episode_assignment:
                mask = self._apply_ownership_mask(agent_id, mask)
            old_result = results[agent_id]
            results[agent_id] = (
                obs,
                old_result[1],
                old_result[2],
                old_result[3],
                {**old_result[4], "action_mask": mask},
            )

        return results

    # ===================================================================
    # 冲突解决: 负载感知贪心拍卖 + 败者改派 (优化路线图 A1+A2/A3+B6)
    # ===================================================================
    def set_eval_mode(self, flag: bool = True):
        """切换评估模式. 评估期启用 A1 败者改派以最大化吞吐; 训练期关闭以保信用分配。"""
        self.eval_mode = flag

    # ===================================================================
    # 全局 episode 级任务指派 (优化路线图 A3-episode / G24)
    # ===================================================================
    def _task_quality(self, agent_id: str, mission) -> Optional[float]:
        """
        卫星 agent_id 在整个 horizon 内对 mission 的观测质量 (∈[0,1], 越大越好);
        返回 None 表示该星全程无可行 VTW (无法负责该任务)。
        质量取所有可行窗口中最小 off-nadir 对应的值 (最佳成像几何)。
        """
        env = self.envs[agent_id]
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
        if self.assignment_scorer in {"lstm", "gru"}:
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
        if self.assignment_scorer not in {"lstm", "gru", "transformer", "set_transformer"} or self._assignment_sequence is None:
            return

        if self.assignment_scorer in {"transformer", "set_transformer"}:
            self._prepare_assignment_attention_context(pending)
            return

        w = self._assignment_sequence
        hidden = self.assignment_sequence_hidden_dim
        h = np.zeros(hidden, dtype=np.float32)
        c = np.zeros(hidden, dtype=np.float32)
        for mission, cands in pending:
            x = self._assignment_task_sequence_features(mission, cands)
            if self.assignment_scorer == "lstm":
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
        feats = np.stack([
            self._assignment_task_sequence_features(mission, cands)
            for mission, cands in pending
        ]).astype(np.float32)
        x = np.tanh(feats @ w["w_embed"] + w["b_embed"])
        if self.assignment_scorer == "transformer":
            positions = np.linspace(0.0, 1.0, num=x.shape[0], dtype=np.float32).reshape(-1, 1)
            x = x + positions * w["pos_scale"].reshape(1, -1)

        q = x @ w["w_q"]
        k = x @ w["w_k"]
        v = x @ w["w_v"]
        scale = max(np.sqrt(float(q.shape[-1])), 1.0)
        attn_logits = (q @ k.T) / scale
        attn = self._softmax(attn_logits, axis=1)
        attended = attn @ v
        if self.assignment_scorer == "set_transformer":
            pooled = attended.mean(axis=0, keepdims=True)
            attended = attended + pooled
        hidden = np.tanh(attended + np.tanh(attended @ w["w_ff"] + w["b_ff"]))
        contexts = np.tanh(hidden @ w["w_out"] + w["b_out"])
        for (mission, _), context in zip(pending, contexts):
            self._assignment_sequence_context[mission.id] = context.astype(np.float32)

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
        return base - self.assign_w_load * load_pressure + 0.15 * float(context_term)

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
        first_env = list(self.envs.values())[0]
        for m in first_env.missions:
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
        return near_deadline or not owner_has_future

    def _has_future_feasible_window(self, agent_id: str, mission_id: int) -> bool:
        """owner 从当前时刻起是否仍有机会在 deadline 前完成该任务。"""
        env = self.envs[agent_id]
        mission = next((m for m in env.missions if m is not None and m.id == mission_id), None)
        if mission is None or mission.is_observed:
            return False
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
        new_completed = len(self._observed_mission_ids() - prev_observed)
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

        global_state_mode:
          - mean: 所有卫星局部观测均值, 维度不随卫星数增长(旧实现).
          - concat: 拼接所有卫星局部观测, 信息无损但维度随卫星数增长(D14).
        global_state_task_stats=True 时追加任务/负载统计(D16).
        """
        local_obs_list = []
        for agent_id in self.agent_ids:
            env = self.envs[agent_id]
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
        first_env = list(self.envs.values())[0]
        missions = [m for m in first_env.missions if m is not None]
        total = max(len(missions), 1)
        observed = sum(1 for m in missions if m.is_observed)
        pending = sum(1 for m in missions if not m.is_observed)
        dynamic_pending = sum(1 for m in missions if m.is_dynamic and not m.is_observed)
        assigned_pending = sum(
            1 for m in missions
            if not m.is_observed and m.id in self.task_owner
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

        # feasible 划分 (论文 Table 4 口径): 任意一颗卫星对该任务有可用 VTW 即可行
        def _feasible_any(mission_id):
            for env in self.envs.values():
                for m in env.missions:
                    if m is not None and m.id == mission_id and env._is_feasible(m):
                        return True
            return False

        feasible_routine = [m for m in all_missions if not m.is_dynamic and _feasible_any(m.id)]
        feasible_dynamic = [m for m in all_missions if m.is_dynamic and _feasible_any(m.id)]
        feas_total = len(feasible_routine) + len(feasible_dynamic)

        # 统计哪些任务被任意一颗卫星完成
        observed_total = len(all_scheduled_ids)
        feas_observed = sum(
            1 for m in (feasible_routine + feasible_dynamic) if m.id in all_scheduled_ids
        )
        routine_feas_done = sum(1 for m in feasible_routine if m.id in all_scheduled_ids)
        dynamic_feas_done = sum(1 for m in feasible_dynamic if m.id in all_scheduled_ids)

        # 全部任务口径 (诊断对照)
        routine_total = sum(1 for m in all_missions if not m.is_dynamic)
        dynamic_total = sum(1 for m in all_missions if m.is_dynamic)
        routine_done = sum(1 for m in all_missions if not m.is_dynamic and m.id in all_scheduled_ids)
        dynamic_done = sum(1 for m in all_missions if m.is_dynamic and m.id in all_scheduled_ids)

        # --- 协同质量指标 (体现多星协同 vs 无协同的核心差异) ---
        # 1) 重复观测: 总调度记录数 - 去重后完成数. 协同好 → ≈0
        total_records = sum(len(env.schedule_log) for env in self.envs.values())
        n_duplicates = total_records - observed_total
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
                if r.is_dynamic:
                    dyn_delays.append(r.obs_end_s - r.earliest_time_s)
        avg_dynamic_response_s = float(np.mean(dyn_delays)) if dyn_delays else 0.0

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
            "n_total_tasks": total_missions,
            "n_routine_tasks": routine_total,
            "n_dynamic_tasks": dynamic_total,
            "n_feasible_tasks": feas_total,
            "n_feasible_routine": len(feasible_routine),
            "n_feasible_dynamic": len(feasible_dynamic),
            "n_feasible_observed": feas_observed,
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
            "n_scheduled": observed_total,
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
