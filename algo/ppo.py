"""
PPO 内循环训练器
================
实现论文 Section 3.5.2:
  - Clipped surrogate objective (Eq.22)
  - GAE (Generalized Advantage Estimation)
  - 共享经验池
  - 与动作掩码兼容的策略更新
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field


@dataclass
class RolloutBuffer:
    """
    PPO 经验回放缓冲区 (on-policy)。
    存储一个 rollout 周期内的所有 transitions。
    """
    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    action_masks: List[np.ndarray] = field(default_factory=list)

    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.action_masks.clear()

    def add(
        self, obs, action, log_prob, reward, value, done, action_mask
    ):
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.action_masks.append(action_mask)

    def __len__(self):
        return len(self.observations)


class PPOTrainer:
    """
    PPO 训练器 (论文 Section 3.5.2, Eq.22)。

    在 MRL-DMS 中作为内循环使用, 也可独立作为 baseline 使用。
    """

    def __init__(
        self,
        actor_critic: nn.Module,
        lr: float = 0.005,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coeff: float = 0.01,
        value_loss_coeff: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 128,
        device: str = "cpu",
    ):
        self.actor_critic = actor_critic
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr)

        self.actor_critic.to(self.device)

    # -------------------------------------------------------------------
    # GAE 计算
    # -------------------------------------------------------------------
    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算 Generalized Advantage Estimation。

        返回
        ----
        advantages : np.ndarray [T]
        returns : np.ndarray [T]
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = last_value
                next_done = 0.0
            else:
                next_value = values[t + 1]
                next_done = float(dones[t])

            delta = rewards[t] + self.gamma * next_value * (1 - next_done) - values[t]
            advantages[t] = last_gae = (
                delta + self.gamma * self.gae_lambda * (1 - next_done) * last_gae
            )

        returns = advantages + values
        return advantages, returns

    # -------------------------------------------------------------------
    # PPO 更新 (论文 Eq.22)
    # -------------------------------------------------------------------
    def update(self, buffer: RolloutBuffer, last_value: float) -> Dict[str, float]:
        """
        用缓冲区中的经验执行 PPO 更新。

        返回
        ----
        info : dict 包含 policy_loss, value_loss, entropy, kl_div 等训练指标
        """
        # 转为数组
        obs = np.array(buffer.observations, dtype=np.float32)
        actions = np.array(buffer.actions, dtype=np.int64)
        old_log_probs = np.array(buffer.log_probs, dtype=np.float32)
        rewards = np.array(buffer.rewards, dtype=np.float32)
        values = np.array(buffer.values, dtype=np.float32)
        dones = np.array(buffer.dones, dtype=np.float32)
        masks = np.array(buffer.action_masks, dtype=np.float32)

        # GAE
        advantages, returns = self.compute_gae(rewards, values, dones, last_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 转为 tensor
        obs_t = torch.FloatTensor(obs).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        old_log_probs_t = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        masks_t = torch.FloatTensor(masks).to(self.device)

        # PPO 多轮更新
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        n_updates = 0

        dataset_size = len(obs)

        for _ in range(self.ppo_epochs):
            # Mini-batch 随机采样
            indices = np.random.permutation(dataset_size)

            for start in range(0, dataset_size, self.batch_size):
                end = min(start + self.batch_size, dataset_size)
                batch_idx = indices[start:end]

                b_obs = obs_t[batch_idx]
                b_actions = actions_t[batch_idx]
                b_old_lp = old_log_probs_t[batch_idx]
                b_adv = advantages_t[batch_idx]
                b_ret = returns_t[batch_idx]
                b_mask = masks_t[batch_idx]

                # 前向传播
                _, new_log_probs, entropy, new_values = (
                    self.actor_critic.get_action_and_value(
                        b_obs, b_mask, b_actions
                    )
                )

                # 概率比 r_t(θ) (Eq.22)
                ratio = torch.exp(new_log_probs - b_old_lp)

                # Clipped surrogate objective (Eq.22)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (MSE)
                value_loss = F.mse_loss(new_values, b_ret)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.value_loss_coeff * value_loss
                    + self.entropy_coeff * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                # 统计
                with torch.no_grad():
                    kl = (b_old_lp - new_log_probs).mean().item()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_kl += kl
                n_updates += 1

        n_updates = max(n_updates, 1)
        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "kl_divergence": total_kl / n_updates,
        }

    # -------------------------------------------------------------------
    # Rollout 采集
    # -------------------------------------------------------------------
    def collect_rollout(
        self,
        env,
        buffer: RolloutBuffer,
        n_steps: int,
        obs: Optional[np.ndarray] = None,
        info: Optional[Dict] = None,
        reset_options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict, float]:
        """
        从环境中采集 n_steps 步经验。

        参数
        ----
        reset_options : dict or None
            episode 结束后重置时传入的 options（含 routine_missions / dynamic_schedule）。
            若为 None 则不传 options（适用于非任务调度类环境）。

        返回
        ----
        last_obs, last_info, episode_reward
        """
        if obs is None:
            obs, info = env.reset(options=reset_options)

        episode_reward = 0.0

        for _ in range(n_steps):
            action_mask = info.get("action_mask", np.ones(env.action_space.n))

            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                mask_t = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)

                action, log_prob, _, value = (
                    self.actor_critic.get_action_and_value(obs_t, mask_t)
                )

            action_np = action.cpu().item()
            log_prob_np = log_prob.cpu().item()
            value_np = value.cpu().item()

            next_obs, reward, terminated, truncated, next_info = env.step(action_np)
            done = terminated or truncated

            buffer.add(obs, action_np, log_prob_np, reward, value_np, done, action_mask)
            episode_reward += reward

            obs = next_obs
            info = next_info

            if done:
                obs, info = env.reset(options=reset_options)

        # 最后一步的价值估计 (用于 GAE)
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            last_value = self.actor_critic.get_value(obs_t).cpu().item()

        return obs, info, episode_reward
