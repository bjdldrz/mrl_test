"""
MRL-DMS 完整训练器
==================
实现论文 Algorithm 1 (Fig. 5) 的元训练过程:

  外循环 (Meta-learner):
    for each meta-iteration:
      采样一批任务 {M_1, ..., M_N} ~ p(M)
      for each task M_t:
        LSTM 编码历史反馈 → 调制 Actor-Critic 初始化
        内循环 PPO 适应 K 步
        评估适应后策略 → 收集 R_routine + R_dynamic
      聚合元损失 L_meta → 更新元参数 Ω

  内循环 (PPO):
    在单个任务场景内执行标准 PPO 训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import time
import logging
from typing import Dict, List, Optional
from pathlib import Path
from tqdm import tqdm

from models.actor_critic import ActorCritic
from models.meta_learner import MetaLearner
from algo.ppo import PPOTrainer, RolloutBuffer
from envs.satellite_env import SatelliteSchedulingEnv
from data.mission_generator import MissionGenerator

logger = logging.getLogger(__name__)


class MRLDMSTrainer:
    """
    MRL-DMS 元强化学习训练器。

    将 LSTM 元学习器 (外循环) 与 PPO (内循环) 结合,
    实现跨任务分布的快速策略适应。
    """

    def __init__(self, config):
        """
        参数
        ----
        config : Config (来自 config.py)
        """
        self.cfg = config
        self.device = self._resolve_device(config.train.device)
        logger.info(f"使用设备: {self.device}")

        # ---- 构建网络 ----
        # 先创建一个临时环境以获取维度信息
        sat_cfg = config.satellites[0]
        dummy_env = SatelliteSchedulingEnv(
            satellite_config=sat_cfg,
            max_action_dim=config.mission.max_action_dim,
        )
        obs_dim = dummy_env.observation_space.shape[0]
        action_dim = dummy_env.action_space.n

        # Actor-Critic (base 网络, 论文中的 θ_base, ϕ_base)
        self.actor_critic = ActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=config.network.hidden_layers,
            activation=config.network.activation,
        ).to(self.device)

        # Meta-Learner (外循环, 论文中的 Ω)
        feedback_dim = 7  # 与 MetaLearner.build_feedback_vector 对齐
        self.meta_learner = MetaLearner(
            feedback_dim=feedback_dim,
            lstm_hidden_dim=config.network.lstm_hidden_dim,
            feedback_embed_dim=config.meta.feedback_dim,
        ).to(self.device)

        # 将调制头绑定到 Actor-Critic
        self.meta_learner.attach_to_actor_critic(self.actor_critic)

        # ---- 优化器 ----
        # 元优化器: 仅更新 LSTM + 调制头 (meta_learner 的参数)
        # base 参数 (actor_critic) 通过 FOMAML 的参数差值方向更新
        self.meta_optimizer = optim.Adam(
            self.meta_learner.parameters(),
            lr=config.meta.meta_lr,
        )
        # base 参数优化器: 用于 FOMAML 的 base 更新
        self.base_optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=config.meta.meta_lr,
        )

        # ---- 复用的内循环 PPO 训练器 ----
        self._inner_ppo = PPOTrainer(
            actor_critic=self.actor_critic,
            lr=config.ppo.learning_rate,
            gamma=config.ppo.discount_factor,
            gae_lambda=config.ppo.gae_lambda,
            clip_ratio=config.ppo.clip_ratio,
            entropy_coeff=config.ppo.entropy_coeff,
            value_loss_coeff=config.ppo.value_loss_coeff,
            ppo_epochs=config.ppo.ppo_epochs,
            batch_size=config.ppo.batch_size,
            device=str(self.device),
        )

        # ---- 任务生成器 ----
        self.mission_gen = None  # 在 setup_data() 中初始化

        # ---- 环境列表 (多星并行) ----
        self.envs = []
        for sat_cfg in config.satellites:
            env = SatelliteSchedulingEnv(
                satellite_config=sat_cfg,
                max_action_dim=config.mission.max_action_dim,
                reward_config=config.reward,
            )
            self.envs.append(env)

        # ---- 训练状态 ----
        self.global_step = 0
        self.meta_iteration = 0
        self.best_reward = -float('inf')
        # 初始化反馈向量 (避免首次 meta_update 时未定义)
        self._prev_feedback = MetaLearner.build_feedback_vector(
            cumulative_reward=0.0, avg_advantage=0.0,
            policy_entropy=1.0, kl_divergence=0.0,
            dynamic_ratio=0.0, n_dynamic_completed=0, n_routine_completed=0,
        )

        # ---- 多智能体组件 (可选) ----
        self.multi_agent = config.mappo.n_satellites > 1
        self._multi_env = None
        self._mappo_model = None
        self._inner_mappo = None

        if self.multi_agent:
            self._init_multi_agent(config, obs_dim, action_dim)

    def setup_data(self, acled_df=None):
        """初始化任务生成器"""
        self.mission_gen = MissionGenerator(
            acled_df=acled_df,
            seed=self.cfg.train.seed,
        )
        logger.info("任务生成器已初始化")

    def _init_multi_agent(self, config, obs_dim, action_dim):
        """初始化多智能体组件"""
        from envs.multi_satellite_env import MultiSatelliteEnv
        from models.mappo import MAPPOActorCritic
        from algo.mappo_trainer import MAPPOTrainer, MultiAgentRolloutBuffer

        n_sat = min(config.mappo.n_satellites, len(config.satellites))
        sat_cfgs = config.satellites[:n_sat]

        # 多星环境
        self._multi_env = MultiSatelliteEnv(
            satellite_configs=sat_cfgs,
            max_action_dim=config.mission.max_action_dim,
            reward_config=config.reward,
        )

        # MAPPO 模型: 共享 Actor + 集中式 Critic
        # global_state_dim = local_obs_dim（mean pooling，维度不随卫星数增长）
        global_state_dim = obs_dim
        self._mappo_model = MAPPOActorCritic(
            local_obs_dim=obs_dim,
            action_dim=action_dim,
            global_state_dim=global_state_dim,
            actor_hidden_dims=config.network.hidden_layers,
            critic_hidden_dims=config.mappo.critic_hidden_dims,
        ).to(self.device)

        # MAPPO 训练器
        self._inner_mappo = MAPPOTrainer(
            mappo_model=self._mappo_model,
            lr=config.ppo.learning_rate,
            gamma=config.ppo.discount_factor,
            gae_lambda=config.ppo.gae_lambda,
            clip_ratio=config.ppo.clip_ratio,
            entropy_coeff=config.ppo.entropy_coeff,
            value_loss_coeff=config.ppo.value_loss_coeff,
            ppo_epochs=config.ppo.ppo_epochs,
            batch_size=config.ppo.batch_size,
            device=str(self.device),
        )

        # 元学习器绑定到 MAPPO 的 Actor（替换之前绑定到 actor_critic 的调制头）
        self.meta_learner.attach_to_actor_critic(self._mappo_model.actor)

        # 重建元优化器（attach 创建了新的 ModulationHead，旧优化器引用已失效）
        self.meta_optimizer = optim.Adam(
            self.meta_learner.parameters(),
            lr=config.meta.meta_lr,
        )

        logger.info(f"多智能体模式: {n_sat} 颗卫星, "
                    f"global_state_dim={global_state_dim}")

    # ===================================================================
    # 元训练主循环 (论文 Algorithm 1)
    # ===================================================================
    def train(self, total_meta_iterations: int = None):
        """
        元训练主循环。

        对应论文 Algorithm 1, Fig. 4-5。
        带 tqdm 进度条实时显示训练指标。
        """
        if self.mission_gen is None:
            self.setup_data()

        total_iters = total_meta_iterations or (
            self.cfg.train.total_training_steps // self.cfg.meta.rollout_steps
        )

        logger.info(f"开始元训练, 共 {total_iters} 个元迭代, 设备: {self.device}")

        pbar = tqdm(
            range(total_iters),
            desc="Meta-Train",
            unit="iter",
            dynamic_ncols=True,
        )

        for meta_iter in pbar:
            self.meta_iteration = meta_iter
            meta_loss, meta_info = self._meta_update()

            # 进度条实时指标
            pbar.set_postfix(
                loss=f"{meta_loss:.4f}",
                R=f"{meta_info['avg_reward']:.1f}",
                dyn=f"{meta_info['avg_dynamic_rate']:.1%}",
                step=self.global_step,
            )

            # 评估
            if meta_iter % self.cfg.train.eval_interval == 0 and meta_iter > 0:
                eval_metrics = self.evaluate()
                tqdm.write(
                    f"  [Eval iter={meta_iter}] "
                    f"reward={eval_metrics['total_reward']:.1f} "
                    f"obs_rate={eval_metrics['observation_success_rate']:.1%} "
                    f"dyn_rate={eval_metrics['dynamic_completion_rate']:.1%}"
                )
                if eval_metrics['total_reward'] > self.best_reward:
                    self.best_reward = eval_metrics['total_reward']
                    self.save_checkpoint("best")

            # 定期保存
            if meta_iter % self.cfg.train.save_interval == 0 and meta_iter > 0:
                self.save_checkpoint(f"iter_{meta_iter}")

        pbar.close()
        logger.info("训练完成")

    # -------------------------------------------------------------------
    # 单次元更新 (Algorithm 1 的一次外循环迭代)
    # -------------------------------------------------------------------
    def _meta_update(self) -> tuple:
        """
        执行一次元更新 (FOMAML 实现):
        自动适配单星/多星模式。
        """
        # 选择正确的 base 模型
        base_model = self._mappo_model.actor if self.multi_agent else self.actor_critic

        # 保存 base 参数
        base_state = copy.deepcopy(base_model.state_dict())

        # 重置 LSTM 隐状态 (Algorithm 1, line 4)
        self.meta_learner.reset_hidden(batch_size=1, device=self.device)

        # 采样任务 (Algorithm 1, line 3)
        tasks = self._sample_task_batch(self.cfg.meta.meta_batch_size)

        # 收集所有任务的参数差值
        param_diffs = []
        total_reward = 0.0
        total_dynamic_rate = 0.0

        for task_idx, (routine, dynamic_schedule) in enumerate(tasks):
            # (a) 恢复 base 参数
            base_model.load_state_dict(copy.deepcopy(base_state))

            # (b) LSTM 编码 + 参数调制
            fb_t = torch.FloatTensor(self._prev_feedback).unsqueeze(0).to(self.device)
            h_t = self.meta_learner(fb_t)
            actor_mods, critic_mods = self.meta_learner.get_modulations(h_t)
            self.meta_learner.apply_modulations(
                base_model, actor_mods, critic_mods
            )

            # 记录调制后的初始参数
            init_state = copy.deepcopy(base_model.state_dict())

            # (c) 内循环适应
            task_reward, task_info = self._inner_loop_adapt(
                routine, dynamic_schedule
            )

            # (d) 收集适应后参数差值
            adapted_state = base_model.state_dict()
            diff = {}
            for name in adapted_state:
                diff[name] = adapted_state[name] - init_state[name]
            param_diffs.append(diff)

            # (e) 评估
            eval_reward, eval_metrics = self._evaluate_adapted_policy(
                routine, dynamic_schedule
            )

            # (f) 更新反馈向量
            self._prev_feedback = MetaLearner.build_feedback_vector(
                cumulative_reward=eval_reward,
                avg_advantage=task_info.get('avg_advantage', 0.0),
                policy_entropy=task_info.get('entropy', 0.0),
                kl_divergence=task_info.get('kl_divergence', 0.0),
                dynamic_ratio=eval_metrics.get('dynamic_completion_rate', 0.0),
                n_dynamic_completed=int(
                    eval_metrics.get('dynamic_completion_rate', 0) * 100),
                n_routine_completed=int(
                    eval_metrics.get('routine_completion_rate', 0) * 100),
            )

            total_reward += eval_reward
            total_dynamic_rate += eval_metrics.get('dynamic_completion_rate', 0.0)

        # ===== FOMAML base 参数更新 =====
        base_model.load_state_dict(base_state)
        n_tasks = len(tasks)
        meta_lr = self.cfg.meta.meta_lr

        with torch.no_grad():
            for name, param in base_model.named_parameters():
                avg_diff = sum(d[name] for d in param_diffs) / n_tasks
                param.data.add_(meta_lr * avg_diff)

        # ===== LSTM / 调制头参数更新 =====
        self.meta_optimizer.zero_grad()
        self.meta_learner.reset_hidden(batch_size=1, device=self.device)
        fb_t = torch.FloatTensor(self._prev_feedback).unsqueeze(0).to(self.device)
        h_t = self.meta_learner(fb_t)
        lstm_loss = -total_reward / (n_tasks * 1000.0) * h_t.norm()
        lstm_loss.backward()
        nn.utils.clip_grad_norm_(self.meta_learner.parameters(), 1.0)
        self.meta_optimizer.step()

        info = {
            'avg_reward': total_reward / n_tasks,
            'avg_dynamic_rate': total_dynamic_rate / n_tasks,
        }
        return lstm_loss.item(), info

    # -------------------------------------------------------------------
    # 内循环 PPO 适应
    # -------------------------------------------------------------------
    def _inner_loop_adapt(
        self,
        routine_missions,
        dynamic_schedule,
    ) -> tuple:
        """
        在单个任务上执行内循环适应。
        根据 self.multi_agent 自动选择单星 PPO 或多星 MAPPO。
        """
        if self.multi_agent:
            return self._inner_loop_mappo(routine_missions, dynamic_schedule)
        else:
            return self._inner_loop_single(routine_missions, dynamic_schedule)

    def _inner_loop_single(self, routine_missions, dynamic_schedule) -> tuple:
        """单星 PPO 内循环 (原始逻辑)"""
        env = self.envs[0]
        obs, info = env.reset(options={
            "routine_missions": copy.deepcopy(routine_missions),
            "dynamic_schedule": copy.deepcopy(dynamic_schedule),
        })

        buffer = RolloutBuffer()
        total_reward = 0.0
        update_info = {}

        inner_pbar = tqdm(
            range(self.cfg.meta.inner_steps),
            desc="  Inner PPO",
            leave=False,
            dynamic_ncols=True,
        )
        for k in inner_pbar:
            buffer.clear()
            obs, info, ep_reward = self._inner_ppo.collect_rollout(
                env, buffer, self.cfg.meta.rollout_steps, obs, info
            )
            total_reward += ep_reward

            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                last_value = self.actor_critic.get_value(obs_t).cpu().item()

            update_info = self._inner_ppo.update(buffer, last_value)
            self.global_step += self.cfg.meta.rollout_steps

            inner_pbar.set_postfix(
                ploss=f"{update_info.get('policy_loss', 0):.3f}",
                R=f"{total_reward:.0f}",
            )

        inner_pbar.close()
        return total_reward, update_info

    def _inner_loop_mappo(self, routine_missions, dynamic_schedule) -> tuple:
        """多星 MAPPO 内循环"""
        from algo.mappo_trainer import MultiAgentRolloutBuffer

        multi_env = self._multi_env
        reset_result = multi_env.reset(options={
            "routine_missions": copy.deepcopy(routine_missions),
            "dynamic_schedule": copy.deepcopy(dynamic_schedule),
        })
        current_obs = {aid: r[0] for aid, r in reset_result.items()}
        current_infos = {aid: r[1] for aid, r in reset_result.items()}

        buffer = MultiAgentRolloutBuffer()
        buffer.init_agents(multi_env.agent_ids)
        total_reward = 0.0
        update_info = {}

        inner_pbar = tqdm(
            range(self.cfg.meta.inner_steps),
            desc="  Inner MAPPO",
            leave=False,
            dynamic_ncols=True,
        )
        for k in inner_pbar:
            buffer.clear()
            buffer.init_agents(multi_env.agent_ids)
            current_obs, current_infos, ep_reward = self._inner_mappo.collect_rollout(
                multi_env, buffer, self.cfg.meta.rollout_steps,
                current_obs, current_infos,
            )
            total_reward += ep_reward

            last_gs = multi_env.get_global_state()
            update_info = self._inner_mappo.update(buffer, last_gs)
            self.global_step += self.cfg.meta.rollout_steps

            inner_pbar.set_postfix(
                ploss=f"{update_info.get('policy_loss', 0):.3f}",
                R=f"{total_reward:.0f}",
            )

        inner_pbar.close()
        return total_reward, update_info

    def _evaluate_adapted_policy(
        self,
        routine_missions,
        dynamic_schedule,
    ) -> tuple:
        """评估适应后的策略 (自动选择单星/多星)"""
        if self.multi_agent:
            return self._evaluate_multi(routine_missions, dynamic_schedule)
        else:
            return self._evaluate_single(routine_missions, dynamic_schedule)

    def _evaluate_single(self, routine_missions, dynamic_schedule) -> tuple:
        """单星评估"""
        env = self.envs[0]
        obs, info = env.reset(options={
            "routine_missions": copy.deepcopy(routine_missions),
            "dynamic_schedule": copy.deepcopy(dynamic_schedule),
        })

        total_reward = 0.0
        done = False
        max_steps = int(env.horizon_s / 10.0) + 100  # 防御性上限

        for _ in range(max_steps):
            if done:
                break
            action_mask = info.get("action_mask", np.ones(env.action_space.n))
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                mask_t = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)
                action, _, _, _ = self.actor_critic.get_action_and_value(obs_t, mask_t)

            obs, reward, terminated, truncated, info = env.step(action.cpu().item())
            total_reward += reward
            done = terminated or truncated

        metrics = env.get_metrics()
        return total_reward, metrics

    def _evaluate_multi(self, routine_missions, dynamic_schedule) -> tuple:
        """多星 MAPPO 评估"""
        multi_env = self._multi_env
        reset_result = multi_env.reset(options={
            "routine_missions": copy.deepcopy(routine_missions),
            "dynamic_schedule": copy.deepcopy(dynamic_schedule),
        })
        current_obs = {aid: r[0] for aid, r in reset_result.items()}
        current_infos = {aid: r[1] for aid, r in reset_result.items()}

        total_reward = 0.0
        # 步数上限：规划周期(秒) / 最小推进步长(1秒) 的合理上界
        max_steps = int(multi_env.horizon_s / 10.0) + 100

        for _ in range(max_steps):
            actions = {}
            for aid in multi_env.agent_ids:
                obs = current_obs[aid]
                mask = current_infos[aid].get(
                    "action_mask", np.ones(multi_env.action_dim)
                )
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    mask_t = torch.FloatTensor(mask).unsqueeze(0).to(self.device)
                    action, _, _ = self._mappo_model.actor.get_action(obs_t, mask_t)
                actions[aid] = action.cpu().item()

            results = multi_env.step(actions)
            for aid, (obs, reward, term, trunc, info) in results.items():
                total_reward += reward
                current_obs[aid] = obs
                current_infos[aid] = info

            if multi_env.is_done():
                break

        metrics = multi_env.get_metrics()
        return total_reward, metrics

    # -------------------------------------------------------------------
    # 任务采样
    # -------------------------------------------------------------------
    def _sample_task_batch(self, n_tasks: int) -> list:
        """
        从任务分布 p(T) 中采样一批任务。
        每个"任务"= (routine_missions, dynamic_schedule)
        """
        tasks = []
        routine_sizes = self.cfg.mission.routine_pool_sizes
        dynamic_sizes = self.cfg.mission.dynamic_pool_sizes

        for _ in range(n_tasks):
            n_routine = np.random.choice(routine_sizes)
            n_dynamic = np.random.choice(dynamic_sizes)
            strategy = np.random.choice(["uniform", "hotspot"])

            routine, dynamic = self.mission_gen.generate_episode_missions(
                n_routine=n_routine,
                n_dynamic_per_insertion=n_dynamic,
                n_insertions=self.cfg.mission.dynamic_insertions_per_day,
                sampling_strategy=strategy,
            )
            tasks.append((routine, dynamic))
        return tasks

    # ===================================================================
    # 评估
    # ===================================================================
    def evaluate(self, n_episodes: int = 3) -> Dict[str, float]:
        """在测试任务上评估当前策略"""
        all_metrics = []

        for _ in range(n_episodes):
            routine, dynamic = self.mission_gen.generate_episode_missions(
                n_routine=200,
                n_dynamic_per_insertion=50,
            )
            _, metrics = self._evaluate_adapted_policy(routine, dynamic)
            all_metrics.append(metrics)

        # 平均
        avg = {}
        for key in all_metrics[0]:
            avg[key] = np.mean([m[key] for m in all_metrics])
        return avg

    # ===================================================================
    # 保存 / 加载
    # ===================================================================
    def save_checkpoint(self, tag: str = "latest"):
        """保存模型检查点"""
        save_dir = Path(self.cfg.train.checkpoint_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"mrl_dms_{tag}.pt"

        torch.save({
            'actor_critic': self.actor_critic.state_dict(),
            'meta_learner': self.meta_learner.state_dict(),
            'meta_optimizer': self.meta_optimizer.state_dict(),
            'global_step': self.global_step,
            'meta_iteration': self.meta_iteration,
            'best_reward': self.best_reward,
        }, path)
        logger.info(f"检查点已保存: {path}")

    def load_checkpoint(self, path: str):
        """加载模型检查点"""
        ckpt = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(ckpt['actor_critic'])
        self.meta_learner.load_state_dict(ckpt['meta_learner'])
        self.meta_optimizer.load_state_dict(ckpt['meta_optimizer'])
        self.global_step = ckpt.get('global_step', 0)
        self.meta_iteration = ckpt.get('meta_iteration', 0)
        self.best_reward = ckpt.get('best_reward', -float('inf'))
        logger.info(f"检查点已加载: {path}")

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device_str)
