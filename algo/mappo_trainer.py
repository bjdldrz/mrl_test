"""
MAPPO 训练器
============
多智能体 PPO 训练循环，实现 CTDE 范式:
  - 并行收集所有卫星的经验
  - 用集中式 Critic 的全局状态价值计算 GAE
  - 聚合所有卫星的梯度更新共享 Actor
  - Critic 用全局状态训练

参考: Yu et al., "The Surprising Effectiveness of PPO in Cooperative MARL"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class MultiAgentRolloutBuffer:
    """
    多智能体经验缓冲区。

    为每个智能体独立存储轨迹，同时记录全局状态。
    """
    # 按 agent_id 索引的局部数据
    local_obs: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    actions: Dict[str, List[int]] = field(default_factory=dict)
    log_probs: Dict[str, List[float]] = field(default_factory=dict)
    rewards: Dict[str, List[float]] = field(default_factory=dict)
    dones: Dict[str, List[bool]] = field(default_factory=dict)
    action_masks: Dict[str, List[np.ndarray]] = field(default_factory=dict)

    # 全局数据 (所有 agent 共享)
    global_states: List[np.ndarray] = field(default_factory=list)
    values: List[float] = field(default_factory=list)  # 集中式 Critic 的输出

    def init_agents(self, agent_ids: List[str]):
        for aid in agent_ids:
            self.local_obs[aid] = []
            self.actions[aid] = []
            self.log_probs[aid] = []
            self.rewards[aid] = []
            self.dones[aid] = []
            self.action_masks[aid] = []

    def clear(self):
        for container in [self.local_obs, self.actions, self.log_probs,
                          self.rewards, self.dones, self.action_masks]:
            for k in container:
                container[k].clear()
        self.global_states.clear()
        self.values.clear()

    def add(
        self,
        agent_id: str,
        obs: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        done: bool,
        mask: np.ndarray,
        global_state: np.ndarray,
        value: float,
    ):
        self.local_obs[agent_id].append(obs)
        self.actions[agent_id].append(action)
        self.log_probs[agent_id].append(log_prob)
        self.rewards[agent_id].append(reward)
        self.dones[agent_id].append(done)
        self.action_masks[agent_id].append(mask)
        # 全局数据只在第一个 agent 添加时写入 (同一步只需一份)
        if agent_id == list(self.local_obs.keys())[0]:
            self.global_states.append(global_state)
            self.values.append(value)

    def __len__(self):
        if not self.global_states:
            return 0
        return len(self.global_states)


class MAPPOTrainer:
    """
    MAPPO 训练器。

    与单智能体 PPOTrainer 对应，但处理多个并行智能体。
    """

    def __init__(
        self,
        mappo_model: nn.Module,
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
        self.model = mappo_model
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.optimizer = optim.Adam(mappo_model.parameters(), lr=lr)
        self.model.to(self.device)

    # -------------------------------------------------------------------
    # Rollout 采集
    # -------------------------------------------------------------------
    def collect_rollout(
        self,
        multi_env,
        buffer: MultiAgentRolloutBuffer,
        n_steps: int,
        current_obs: Optional[Dict] = None,
        current_infos: Optional[Dict] = None,
    ) -> Tuple[Dict, Dict, float]:
        """
        从多星环境中并行采集经验。

        返回: last_obs, last_infos, total_reward
        """
        if current_obs is None:
            reset_result = multi_env.reset()
            current_obs = {aid: r[0] for aid, r in reset_result.items()}
            current_infos = {aid: r[1] for aid, r in reset_result.items()}

        agent_ids = multi_env.agent_ids
        total_reward = 0.0

        for step in range(n_steps):
            # 获取全局状态 (给 Critic)
            global_state = multi_env.get_global_state()

            # 用集中式 Critic 估计当前全局价值
            with torch.no_grad():
                gs_t = torch.FloatTensor(global_state).unsqueeze(0).to(self.device)
                value = self.model.get_values(gs_t).cpu().item()

            # 每颗卫星独立选择动作 (用共享 Actor)
            actions_dict = {}
            for aid in agent_ids:
                obs = current_obs[aid]
                mask = current_infos[aid].get(
                    "action_mask", np.ones(multi_env.action_dim)
                )

                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    mask_t = torch.FloatTensor(mask).unsqueeze(0).to(self.device)
                    action, log_prob, _ = self.model.actor.get_action(obs_t, mask_t)

                action_np = action.cpu().item()
                log_prob_np = log_prob.cpu().item()
                actions_dict[aid] = action_np

                # 先存入 buffer (reward 在 step 后补充)
                buffer.add(
                    agent_id=aid,
                    obs=obs,
                    action=action_np,
                    log_prob=log_prob_np,
                    reward=0.0,  # 占位，下面替换
                    done=False,
                    mask=mask,
                    global_state=global_state,
                    value=value,
                )

            # 环境执行
            step_results = multi_env.step(actions_dict)

            # 更新 buffer 中的 reward 和 done
            for aid in agent_ids:
                obs, reward, term, trunc, info = step_results[aid]
                done = term or trunc
                # 替换占位值
                buffer.rewards[aid][-1] = reward
                buffer.dones[aid][-1] = done
                total_reward += reward

                current_obs[aid] = obs
                current_infos[aid] = info

            # 检查是否所有卫星都完成
            if multi_env.is_done():
                # episode 结束，终止本轮采集（不做空 reset）
                break

        return current_obs, current_infos, total_reward

    # -------------------------------------------------------------------
    # GAE (使用集中式 Critic 的价值估计)
    # -------------------------------------------------------------------
    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
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
    # MAPPO 更新
    # -------------------------------------------------------------------
    def update(
        self,
        buffer: MultiAgentRolloutBuffer,
        last_global_state: np.ndarray,
    ) -> Dict[str, float]:
        """
        MAPPO 更新：聚合所有智能体的经验。

        关键区别: GAE 使用集中式 Critic 的价值，
        但策略梯度来自每个智能体的局部动作/观测。
        """
        agent_ids = list(buffer.local_obs.keys())
        n_steps = len(buffer)

        if n_steps == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        # 全局数据
        global_states = np.array(buffer.global_states, dtype=np.float32)
        values = np.array(buffer.values, dtype=np.float32)

        # 用集中式 Critic 计算最后一步的价值
        with torch.no_grad():
            gs_t = torch.FloatTensor(last_global_state).unsqueeze(0).to(self.device)
            last_value = self.model.get_values(gs_t).cpu().item()

        # 聚合所有智能体的经验，用共享的全局价值计算 GAE
        all_obs = []
        all_actions = []
        all_old_lps = []
        all_masks = []
        all_advantages = []
        all_returns = []
        all_global_states = []

        for aid in agent_ids:
            rewards = np.array(buffer.rewards[aid], dtype=np.float32)
            dones = np.array(buffer.dones[aid], dtype=np.float32)

            # 所有智能体共享同一 Critic 价值
            advantages, returns = self.compute_gae(rewards, values, dones, last_value)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            all_obs.append(np.array(buffer.local_obs[aid], dtype=np.float32))
            all_actions.append(np.array(buffer.actions[aid], dtype=np.int64))
            all_old_lps.append(np.array(buffer.log_probs[aid], dtype=np.float32))
            all_masks.append(np.array(buffer.action_masks[aid], dtype=np.float32))
            all_advantages.append(advantages)
            all_returns.append(returns)
            all_global_states.append(global_states)

        # 拼接所有智能体的数据 (参数共享 → 所有智能体的梯度叠加)
        obs_all = torch.FloatTensor(np.concatenate(all_obs)).to(self.device)
        act_all = torch.LongTensor(np.concatenate(all_actions)).to(self.device)
        lp_all = torch.FloatTensor(np.concatenate(all_old_lps)).to(self.device)
        mask_all = torch.FloatTensor(np.concatenate(all_masks)).to(self.device)
        adv_all = torch.FloatTensor(np.concatenate(all_advantages)).to(self.device)
        ret_all = torch.FloatTensor(np.concatenate(all_returns)).to(self.device)
        gs_all = torch.FloatTensor(np.concatenate(all_global_states)).to(self.device)

        dataset_size = len(obs_all)
        total_ploss = total_vloss = total_ent = 0.0
        n_updates = 0

        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, self.batch_size):
                end = min(start + self.batch_size, dataset_size)
                idx = indices[start:end]

                b_obs = obs_all[idx]
                b_act = act_all[idx]
                b_olp = lp_all[idx]
                b_mask = mask_all[idx]
                b_adv = adv_all[idx]
                b_ret = ret_all[idx]
                b_gs = gs_all[idx]

                # Actor forward (共享参数)
                _, new_lp, entropy = self.model.actor.get_action(
                    b_obs, b_mask, b_act
                )

                # Critic forward (集中式)
                new_values = self.model.critic(b_gs)

                # PPO clipped surrogate
                ratio = torch.exp(new_lp - b_olp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(
                    ratio, 1 - self.clip_ratio, 1 + self.clip_ratio
                ) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(new_values, b_ret)
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.value_loss_coeff * value_loss
                    + self.entropy_coeff * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_ploss += policy_loss.item()
                total_vloss += value_loss.item()
                total_ent += entropy.mean().item()
                n_updates += 1

        n_updates = max(n_updates, 1)
        return {
            "policy_loss": total_ploss / n_updates,
            "value_loss": total_vloss / n_updates,
            "entropy": total_ent / n_updates,
        }
