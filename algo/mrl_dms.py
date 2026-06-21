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
import csv
import json
import time
import logging
from multiprocessing import Pool, get_context
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
                vtw_time_step_s=config.train.vtw_time_step_s,
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

        # ---- 全局跨任务 VTW 缓存 ----
        # key: (sat_name, round(lat,4), round(lon,4), horizon_s_int, step_s_int)
        # 相同坐标在不同任务/迭代中完全复用，避免重复计算
        self._global_vtw_cache: dict = {}

        # ---- 多智能体组件 (可选) ----
        self.multi_agent = config.mappo.n_satellites > 1
        self._multi_env = None
        self._mappo_model = None
        self._inner_mappo = None

        if self.multi_agent:
            self._init_multi_agent(config, obs_dim, action_dim)

        # 多进程模式：子进程自己创建模型和环境，主进程只保存构建所需的参数
        self._worker_obs_dim = obs_dim
        self._worker_action_dim = action_dim

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
            vtw_time_step_s=config.train.vtw_time_step_s,
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
    def train(self, total_meta_iterations: int = None, exp_name: str = None):
        """
        元训练主循环。

        对应论文 Algorithm 1, Fig. 4-5。
        带 tqdm 进度条实时显示训练指标，并将指标写入 CSV / JSON 文件。
        """
        if self.mission_gen is None:
            self.setup_data()


        total_iters = total_meta_iterations or (
            self.cfg.train.total_training_steps
            // (self.cfg.meta.meta_batch_size
                * self.cfg.meta.inner_steps
                * self.cfg.meta.rollout_steps)
        )

        # ---- 日志目录 ----
        run_name = exp_name or f"mrl_dms_{int(time.time())}"
        log_dir = Path(self.cfg.train.log_dir) / run_name
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_dir = log_dir  # 供 save_checkpoint 使用

        train_log_path = log_dir / "train_log.csv"
        eval_log_path = log_dir / "eval_log.csv"

        train_fieldnames = ["iter", "global_step", "meta_loss", "avg_reward", "avg_dynamic_rate"]
        eval_fieldnames = ["iter", "total_reward", "observation_success_rate",
                           "dynamic_completion_rate", "routine_completion_rate",
                           "dynamic_reward", "routine_reward", "n_scheduled"]

        train_csv_f = open(train_log_path, "w", newline="")
        eval_csv_f = open(eval_log_path, "w", newline="")
        train_writer = csv.DictWriter(train_csv_f, fieldnames=train_fieldnames)
        eval_writer = csv.DictWriter(eval_csv_f, fieldnames=eval_fieldnames)
        train_writer.writeheader()
        eval_writer.writeheader()

        logger.info(f"开始元训练, 共 {total_iters} 个元迭代, 设备: {self.device}")
        logger.info(f"日志目录: {log_dir}")

        # 余弦退火：meta_lr 在训练过程中从初始值衰减到 1/10
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.meta_optimizer, T_max=total_iters, eta_min=self.cfg.meta.meta_lr / 10
        )

        pbar = tqdm(
            range(total_iters),
            desc="Meta-Train",
            unit="iter",
            dynamic_ncols=True,
        )

        try:
            for meta_iter in pbar:
                self.meta_iteration = meta_iter
                meta_loss, meta_info = self._meta_update()
                lr_scheduler.step()

                # 进度条实时指标
                pbar.set_postfix(
                    loss=f"{meta_loss:.4f}",
                    R=f"{meta_info['avg_reward']:.1f}",
                    dyn=f"{meta_info['avg_dynamic_rate']:.1%}",
                    step=self.global_step,
                )

                # 写训练行
                train_writer.writerow({
                    "iter": meta_iter,
                    "global_step": self.global_step,
                    "meta_loss": round(meta_loss, 6),
                    "avg_reward": round(meta_info['avg_reward'], 4),
                    "avg_dynamic_rate": round(meta_info['avg_dynamic_rate'], 4),
                })
                train_csv_f.flush()

                # 评估
                if meta_iter % self.cfg.train.eval_interval == 0 and meta_iter > 0:
                    eval_metrics = self.evaluate()
                    tqdm.write(
                        f"  [Eval iter={meta_iter}] "
                        f"reward={eval_metrics['total_reward']:.1f} "
                        f"obs_rate={eval_metrics['observation_success_rate']:.1%} "
                        f"dyn_rate={eval_metrics['dynamic_completion_rate']:.1%}"
                    )

                    eval_writer.writerow({"iter": meta_iter, **{
                        k: round(eval_metrics.get(k, 0.0), 4) for k in eval_fieldnames[1:]
                    }})
                    eval_csv_f.flush()

                    if eval_metrics['total_reward'] > self.best_reward:
                        self.best_reward = eval_metrics['total_reward']
                        self.save_checkpoint("best")

                # 定期保存
                if meta_iter % self.cfg.train.save_interval == 0 and meta_iter > 0:
                    self.save_checkpoint(f"iter_{meta_iter}")

        finally:
            train_csv_f.close()
            eval_csv_f.close()
            pbar.close()

        # ---- 训练结束：写 JSON 摘要 ----
        summary = {
            "exp_name": run_name,
            "total_iters": total_iters,
            "global_step": self.global_step,
            "best_reward": self.best_reward,
            "train_log": str(train_log_path),
            "eval_log": str(eval_log_path),
        }
        summary_path = log_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"训练完成, 摘要已保存至: {summary_path}")

    # -------------------------------------------------------------------
    # 单次元更新 (Algorithm 1 的一次外循环迭代)
    # -------------------------------------------------------------------
    def _meta_update(self) -> tuple:
        """
        执行一次元更新 (FOMAML 实现)。

        单星模式：meta_batch 内各任务通过 multiprocessing.Pool 并行执行，
        充分利用多核 CPU。
        多星模式：MAPPO 内部已经是 N 星并发，meta_batch 串行执行避免多环境
        共享状态竞态问题。
        """
        base_model = self._mappo_model.actor if self.multi_agent else self.actor_critic
        base_state = copy.deepcopy(base_model.state_dict())

        self.meta_learner.reset_hidden(batch_size=1, device=self.device)

        tasks = self._sample_task_batch(self.cfg.meta.meta_batch_size)
        n_tasks = len(tasks)

        # 预先为每个任务计算 LSTM 调制（串行，需共享 LSTM 状态）
        # 注意：单星模式下 VTW 预计算推迟到 worker 内部执行（并行化）
        task_h_norms = []
        task_init_states = []
        for i, (routine, dynamic_schedule) in enumerate(tasks):
            if self.multi_agent:
                # 多星模式串行执行，VTW 在主线程预计算
                self._precompute_task_vtw(routine, dynamic_schedule)
            base_model.load_state_dict(copy.deepcopy(base_state))
            fb_t = torch.FloatTensor(self._prev_feedback).unsqueeze(0).to(self.device)
            h_t = self.meta_learner(fb_t)
            actor_mods, critic_mods = self.meta_learner.get_modulations(h_t)
            self.meta_learner.apply_modulations(base_model, actor_mods, critic_mods)
            task_h_norms.append(h_t.norm())
            task_init_states.append(copy.deepcopy(base_model.state_dict()))

        if self.multi_agent:
            results = self._meta_update_mappo(tasks, task_init_states, base_model, base_state)
        else:
            results = self._meta_update_single(tasks, task_init_states)

        param_diffs, meta_rewards, total_dynamic_rate, total_reward = [], [], 0.0, 0.0
        for param_diff, eval_reward, eval_metrics, steps in results:
            param_diffs.append(param_diff)
            meta_rewards.append(eval_reward)
            total_reward += eval_reward
            total_dynamic_rate += eval_metrics.get('dynamic_completion_rate', 0.0)
            self.global_step += steps

        # 更新反馈向量（对所有任务取平均，减少 batch=16 时的单任务噪声）
        avg_dyn_rate = total_dynamic_rate / n_tasks
        avg_cum_reward = sum(meta_rewards) / n_tasks
        avg_routine_rate = sum(
            r[2].get('routine_completion_rate', 0.0) for r in results
        ) / n_tasks
        avg_n_dynamic = int(avg_dyn_rate * 100)
        avg_n_routine = int(avg_routine_rate * 100)
        self._prev_feedback = MetaLearner.build_feedback_vector(
            cumulative_reward=avg_cum_reward,
            avg_advantage=0.0,
            policy_entropy=0.0,
            kl_divergence=0.0,
            dynamic_ratio=avg_dyn_rate,
            n_dynamic_completed=avg_n_dynamic,
            n_routine_completed=avg_n_routine,
        )

        # FOMAML base 参数更新
        base_model.load_state_dict(base_state)
        meta_lr = self.cfg.meta.meta_lr
        with torch.no_grad():
            for name, param in base_model.named_parameters():
                diffs = torch.stack([d[name].to(param.device) for d in param_diffs])
                avg_diff = diffs.mean(dim=0)
                param.data.add_(meta_lr * avg_diff)

        # LSTM / 调制头更新
        self.meta_optimizer.zero_grad()
        rewards_t = torch.tensor(meta_rewards, dtype=torch.float32, device=self.device)
        r_std = rewards_t.std().clamp(min=1.0)
        rewards_normalized = (rewards_t - rewards_t.mean()) / r_std
        h_norms = torch.stack(task_h_norms)
        lstm_loss = -(rewards_normalized * h_norms).mean()
        lstm_loss.backward()
        nn.utils.clip_grad_norm_(self.meta_learner.parameters(), 1.0)
        self.meta_optimizer.step()

        info = {
            'avg_reward': total_reward / n_tasks,
            'avg_dynamic_rate': total_dynamic_rate / n_tasks,
        }
        return lstm_loss.item(), info

    def _meta_update_single(self, tasks, task_init_states) -> list:
        """单星并行内循环：multiprocessing.Pool 跑 meta_batch 中所有任务，绕开 GIL"""
        from algo.task_worker import run_single_task

        cfg_workers = self.cfg.train.num_workers
        n_workers = cfg_workers if cfg_workers > 0 else len(tasks)
        n_workers = min(n_workers, len(tasks))

        # 将 state_dict 张量转为 numpy（减少跨进程序列化体积）
        def _state_to_numpy(sd):
            return {k: v.cpu().numpy() for k, v in sd.items()}

        task_args = []
        for i, (routine, dynamic_schedule) in enumerate(tasks):
            task_args.append({
                'idx': i,
                'routine': routine,
                'dynamic_schedule': dynamic_schedule,
                'init_state': _state_to_numpy(task_init_states[i]),
                'sat_config': self.cfg.satellites[0],
                'reward_config': self.cfg.reward,
                'vtw_time_step_s': self.cfg.train.vtw_time_step_s,
                'max_action_dim': self.cfg.mission.max_action_dim,
                'vtw_cache': dict(self._global_vtw_cache),  # 只读快照
                'cfg_ppo': self.cfg.ppo,
                'cfg_meta': self.cfg.meta,
                'obs_dim': self._worker_obs_dim,
                'action_dim': self._worker_action_dim,
                'hidden_dims': self.cfg.network.hidden_layers,
                'activation': self.cfg.network.activation,
            })

        # spawn 避免 CUDA / tqdm fork 问题
        ctx = get_context('spawn')
        with ctx.Pool(processes=n_workers) as pool:
            results_raw = pool.map(run_single_task, task_args)

        raw = [None] * len(tasks)
        for r in results_raw:
            idx = r['idx']
            # numpy → torch tensor
            param_diff = {
                name: torch.from_numpy(arr).to(self.device)
                for name, arr in r['param_diff_np'].items()
            }
            # 将子进程新计算的 VTW 合并回主进程缓存
            self._global_vtw_cache.update(r['new_vtw'])
            raw[idx] = (param_diff, r['eval_reward'], r['eval_metrics'], r['steps_consumed'])
        return raw

    def _meta_update_mappo(self, tasks, task_init_states, base_model, base_state) -> list:
        """多星串行内循环：每个任务用 MAPPOTrainer + MultiSatelliteEnv 完整跑一遍"""
        from algo.mappo_trainer import MAPPOTrainer, MultiAgentRolloutBuffer

        results = []
        for i, (routine, dynamic_schedule) in enumerate(tasks):
            init_state = task_init_states[i]
            # 每个任务独立加载调制后的初始参数（actor 是 base_model）
            base_model.load_state_dict(copy.deepcopy(init_state))

            multi_env = self._multi_env
            reset_options = {
                "routine_missions": copy.deepcopy(routine),
                "dynamic_schedule": copy.deepcopy(dynamic_schedule),
            }
            reset_result = multi_env.reset(options=reset_options)
            current_obs = {aid: r[0] for aid, r in reset_result.items()}
            current_infos = {aid: r[1] for aid, r in reset_result.items()}

            buffer = MultiAgentRolloutBuffer()
            buffer.init_agents(multi_env.agent_ids)

            # 每个任务重置 Adam 动量
            self._inner_mappo.optimizer = optim.Adam(
                self._mappo_model.parameters(),
                lr=self.cfg.ppo.learning_rate,
            )

            for _ in range(self.cfg.meta.inner_steps):
                buffer.clear()
                buffer.init_agents(multi_env.agent_ids)
                current_obs, current_infos, ep_reward = self._inner_mappo.collect_rollout(
                    multi_env, buffer, self.cfg.meta.rollout_steps,
                    current_obs, current_infos,
                )
                last_gs = multi_env.get_global_state()
                self._inner_mappo.update(buffer, last_gs)

            # 评估
            eval_reward, eval_metrics = self._evaluate_multi(routine, dynamic_schedule)

            adapted_state = base_model.state_dict()
            param_names = {name for name, _ in base_model.named_parameters()}
            param_diff = {
                name: adapted_state[name] - init_state[name]
                for name in param_names
            }
            steps_consumed = self.cfg.meta.inner_steps * self.cfg.meta.rollout_steps
            results.append((param_diff, eval_reward, eval_metrics, steps_consumed))

        return results

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
        reset_options = {
            "routine_missions": copy.deepcopy(routine_missions),
            "dynamic_schedule": copy.deepcopy(dynamic_schedule),
        }
        obs, info = env.reset(options=reset_options)

        buffer = RolloutBuffer()
        total_reward = 0.0
        update_info = {}

        # 每个新任务重置 Adam 动量，避免上一个任务的梯度历史污染新任务适应
        self._inner_ppo.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=self.cfg.ppo.learning_rate,
        )

        inner_pbar = tqdm(
            range(self.cfg.meta.inner_steps),
            desc="  Inner PPO",
            leave=False,
            dynamic_ncols=True,
        )
        for k in inner_pbar:
            buffer.clear()
            obs, info, ep_reward = self._inner_ppo.collect_rollout(
                env, buffer, self.cfg.meta.rollout_steps, obs, info,
                reset_options=reset_options,
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

        # 每个新任务重置 Adam 动量
        self._inner_mappo.optimizer = optim.Adam(
            self._mappo_model.parameters(),
            lr=self.cfg.ppo.learning_rate,
        )

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
    # VTW 预计算
    # -------------------------------------------------------------------
    def _precompute_task_vtw(
        self,
        routine_missions: list,
        dynamic_schedule: list,
    ) -> None:
        """
        为本次任务批量预计算 VTW，并注入每个 env.precomputed_vtw。

        两级缓存策略：
          1. _global_vtw_cache：跨 meta-iteration / 跨任务，按坐标去重；
             相同坐标只计算一次，即使 mission_id 不同。
          2. env.precomputed_vtw：按 (sat_name, mission_id) 索引，供
             env._compute_vtw_for_missions 走 O(1) 字典查找。
        """
        all_missions = list(routine_missions)
        for _, dyn_batch in dynamic_schedule:
            all_missions.extend(dyn_batch)

        ref_env = self.envs[0]
        horizon_s = ref_env.horizon_s
        step_s = ref_env.vtw_time_step_s

        def _get_vtw_for_env(env, mission, h_s=None, s_s=None):
            h_s = h_s if h_s is not None else horizon_s
            s_s = s_s if s_s is not None else step_s
            sat_name = env.sat_config.name
            coord_key = (sat_name, round(mission.lat, 4), round(mission.lon, 4),
                         int(h_s), int(s_s))
            if coord_key not in self._global_vtw_cache:
                self._global_vtw_cache[coord_key] = env.propagator.compute_vtw(
                    mission.lat, mission.lon, h_s, time_step_s=s_s
                )
            return self._global_vtw_cache[coord_key]

        # 单星模式：注入 self.envs；多星模式：只注入 _multi_env 子环境
        if not self.multi_agent:
            for env in self.envs:
                pv = {}
                for m in all_missions:
                    pv[(env.sat_config.name, m.id)] = _get_vtw_for_env(env, m)
                env.precomputed_vtw = pv

        # 多星模式：同样处理 _multi_env 各子环境
        if self.multi_agent and self._multi_env is not None:
            for sub_env in self._multi_env.envs.values():
                pv = {}
                for m in all_missions:
                    pv[(sub_env.sat_config.name, m.id)] = _get_vtw_for_env(
                        sub_env, m, sub_env.horizon_s, sub_env.vtw_time_step_s
                    )
                sub_env.precomputed_vtw = pv

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
        """在测试任务上评估当前策略（含内循环适应，反映 MAML 快适应能力）"""
        base_model = self._mappo_model.actor if self.multi_agent else self.actor_critic
        base_state = copy.deepcopy(base_model.state_dict())

        all_metrics = []

        for _ in range(n_episodes):
            routine, dynamic = self.mission_gen.generate_episode_missions(
                n_routine=200,
                n_dynamic_per_insertion=50,
            )
            self._precompute_task_vtw(routine, dynamic)

            # 恢复 base 参数，执行内循环适应后再评估
            base_model.load_state_dict(copy.deepcopy(base_state))
            self._inner_loop_adapt(routine, dynamic)
            _, metrics = self._evaluate_adapted_policy(routine, dynamic)
            all_metrics.append(metrics)

        # 评估后恢复 base 参数（不污染外循环训练状态）
        base_model.load_state_dict(base_state)

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

        ckpt = {
            'actor_critic': self.actor_critic.state_dict(),
            'meta_learner': self.meta_learner.state_dict(),
            'meta_optimizer': self.meta_optimizer.state_dict(),
            'base_optimizer': self.base_optimizer.state_dict(),
            'global_step': self.global_step,
            'meta_iteration': self.meta_iteration,
            'best_reward': self.best_reward,
        }
        if self.multi_agent and self._mappo_model is not None:
            ckpt['mappo_model'] = self._mappo_model.state_dict()
        if hasattr(self, '_log_dir'):
            ckpt['log_dir'] = str(self._log_dir)

        torch.save(ckpt, path)
        logger.info(f"检查点已保存: {path}")

    def load_checkpoint(self, path: str):
        """加载模型检查点"""
        ckpt = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(ckpt['actor_critic'])
        self.meta_learner.load_state_dict(ckpt['meta_learner'])
        self.meta_optimizer.load_state_dict(ckpt['meta_optimizer'])
        if 'base_optimizer' in ckpt:
            self.base_optimizer.load_state_dict(ckpt['base_optimizer'])
        if self.multi_agent and self._mappo_model is not None and 'mappo_model' in ckpt:
            self._mappo_model.load_state_dict(ckpt['mappo_model'])
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
