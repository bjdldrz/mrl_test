from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .feature_builder import ActionSetFeatureBuilder
from .action_entities import ACTION_OBSERVE, ACTION_RELAY, ACTION_WAIT, EdgeDecisionRecord
from .rollout_buffer import ActionSetRolloutBuffer


@dataclass
class CandidateAuxSamples:
    edge_features: np.ndarray
    advantages: np.ndarray
    negative_features: np.ndarray
    negative_anchor_indices: np.ndarray
    n_conflict_edges: int = 0
    conflict_penalty_sum: float = 0.0
    load_penalty_sum: float = 0.0
    edge_records: List[EdgeDecisionRecord] = field(default_factory=list)


class ActionSetMAPPOTrainer:
    def __init__(
        self,
        model,
        feature_builder: ActionSetFeatureBuilder,
        lr: float = 0.005,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coeff: float = 0.01,
        value_loss_coeff: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 128,
        candidate_dropout_prob: float = 0.0,
        idle_aux_coeff: float = 0.0,
        device: str = "cpu",
    ):
        self.model = model
        self.feature_builder = feature_builder
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_ratio = float(clip_ratio)
        self.entropy_coeff = float(entropy_coeff)
        self.value_loss_coeff = float(value_loss_coeff)
        self.max_grad_norm = float(max_grad_norm)
        self.ppo_epochs = int(ppo_epochs)
        self.batch_size = int(batch_size)
        self.candidate_dropout_prob = float(candidate_dropout_prob)
        self.idle_aux_coeff = max(0.0, float(idle_aux_coeff))
        self.device = torch.device(device)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.model.to(self.device)

    def collect_rollout(
        self,
        env,
        buffer: ActionSetRolloutBuffer,
        n_steps: int,
        current_infos: Dict,
    ) -> Tuple[Dict, float]:
        total_reward = 0.0
        for _ in range(max(1, int(n_steps))):
            global_state = env.get_global_state()
            with torch.no_grad():
                value = self.model.get_values(
                    torch.FloatTensor(global_state).unsqueeze(0).to(self.device)
                ).cpu().item()
            buffer.add_step_value(global_state, value)

            batch = self.feature_builder.build_many(env, current_infos)
            actions, log_probs, masks = self.sample_actions(env, batch, training=True)
            for aid in env.agent_ids:
                item = batch[aid]
                buffer.add_agent(
                    aid,
                    state_obs=item.state,
                    action_features=item.action_features,
                    candidate_edge_features=item.candidate_edge_features,
                    candidate_task_ids=item.candidate_task_ids,
                    action_mask=masks[aid],
                    action=actions[aid],
                    log_prob=log_probs[aid],
                    decision_time_s=float(env.envs[aid].current_time_s),
                )

            step_results = env.step(actions)
            for aid in env.agent_ids:
                _, reward, term, trunc, info = step_results[aid]
                buffer.rewards[aid][-1] = float(reward)
                buffer.dones[aid][-1] = bool(term or trunc)
                current_infos[aid] = info
                total_reward += float(reward)
            if env.is_done():
                break
        return current_infos, total_reward

    def sample_actions(self, env, batch: Dict, training: bool = False, deterministic: bool = False):
        agent_ids = env.agent_ids
        states = np.stack([batch[aid].state for aid in agent_ids], axis=0)
        action_features = np.stack([batch[aid].action_features for aid in agent_ids], axis=0)
        masks = {
            aid: self._dropout_mask(env, batch[aid].action_mask) if training else batch[aid].action_mask.copy()
            for aid in agent_ids
        }
        mask_arr = np.stack([masks[aid] for aid in agent_ids], axis=0)
        with torch.no_grad():
            state_t = torch.FloatTensor(states).to(self.device)
            action_feat_t = torch.FloatTensor(action_features).to(self.device)
            mask_t = torch.FloatTensor(mask_arr).to(self.device)
            action_t, log_prob_t, _ = self.model.actor.get_action(
                state_t,
                action_feat_t,
                mask_t,
                deterministic=deterministic,
            )
        actions_np = action_t.cpu().numpy()
        log_probs_np = log_prob_t.cpu().numpy()
        actions = {aid: int(actions_np[idx]) for idx, aid in enumerate(agent_ids)}
        log_probs = {aid: float(log_probs_np[idx]) for idx, aid in enumerate(agent_ids)}
        return actions, log_probs, masks

    def select_eval_actions(self, env, infos: Dict, deterministic: bool = False) -> Dict[str, int]:
        batch = self.feature_builder.build_many(env, infos)
        actions, _, _ = self.sample_actions(env, batch, training=False, deterministic=deterministic)
        return actions

    def _dropout_mask(self, env, mask: np.ndarray) -> np.ndarray:
        out = mask.copy()
        p = self.candidate_dropout_prob
        if p <= 0:
            return out
        idle = int(env.idle_action)
        task_limit = int(getattr(env, "candidate_action_top_k", max(0, idle)))
        valid_tasks = np.nonzero(out[:task_limit] > 0)[0]
        if len(valid_tasks) <= 1:
            return out
        drop = np.random.rand(len(valid_tasks)) < p
        if np.all(drop):
            drop[np.random.randint(0, len(drop))] = False
        out[valid_tasks[drop]] = 0.0
        if 0 <= idle < len(out):
            out[idle] = 1.0
        return out

    def update(self, buffer: ActionSetRolloutBuffer, last_global_state: np.ndarray) -> Dict[str, float]:
        return self.update_many([buffer], [last_global_state])

    def candidate_aux_samples(
        self,
        buffer: ActionSetRolloutBuffer,
        last_global_state: np.ndarray,
        max_negatives_per_positive: int = 0,
        valid_negatives_only: bool = True,
        conflict_penalty: float = 0.0,
        load_penalty: float = 0.0,
    ) -> CandidateAuxSamples:
        if len(buffer) == 0:
            return self._empty_candidate_aux_samples()

        values = np.array(buffer.values, dtype=np.float32)
        with torch.no_grad():
            last_value = self.model.get_values(
                torch.FloatTensor(last_global_state).unsqueeze(0).to(self.device)
            ).cpu().item()

        edge_rows: List[np.ndarray] = []
        targets: List[float] = []
        negative_rows: List[np.ndarray] = []
        negative_anchors: List[int] = []
        edge_records: List[EdgeDecisionRecord] = []
        selected_task_counts = self._selected_task_counts(buffer)
        n_conflict_edges = 0
        conflict_penalty_sum = 0.0
        load_penalty_sum = 0.0
        for aid in buffer.state_obs.keys():
            if aid not in buffer.candidate_edge_features:
                continue
            rewards = np.array(buffer.rewards[aid], dtype=np.float32)
            dones = np.array(buffer.dones[aid], dtype=np.float32)
            advantages, _ = self.compute_gae(rewards, values, dones, last_value)
            for t, action in enumerate(buffer.actions[aid]):
                edge_snapshot = buffer.candidate_edge_features[aid][t]
                if not 0 <= int(action) < edge_snapshot.shape[0]:
                    continue
                edge = np.asarray(edge_snapshot[int(action)], dtype=np.float32)
                if edge.size == 0 or not np.any(np.abs(edge) > 0):
                    continue
                task_id = self._selected_task_id(buffer, aid, t, int(action))
                conflict_count = selected_task_counts[t].get(task_id, 0) if task_id >= 0 else 0
                conflict_cost = max(conflict_count - 1, 0) * max(float(conflict_penalty), 0.0)
                load_cost = float(edge[6]) * max(float(load_penalty), 0.0) if edge.shape[0] > 6 else 0.0
                target = float(advantages[t]) - conflict_cost - load_cost
                if conflict_cost > 0:
                    n_conflict_edges += 1
                    conflict_penalty_sum += float(conflict_cost)
                load_penalty_sum += float(load_cost)
                edge_records.append(
                    EdgeDecisionRecord(
                        decision_id=len(edge_records),
                        time_s=self._decision_time(buffer, aid, t),
                        agent_id=aid,
                        action_idx=int(action),
                        action_type=self._action_entity_type(buffer.action_features[aid][t][int(action)]),
                        target_id=task_id,
                        edge_features=edge,
                        target=target,
                    )
                )
                positive_idx = len(edge_rows)
                edge_rows.append(edge)
                targets.append(target)
                if max_negatives_per_positive > 0:
                    neg_edges = self._hard_negative_edges(
                        edge_snapshot=edge_snapshot,
                        action_features=buffer.action_features[aid][t],
                        action_mask=buffer.action_masks[aid][t],
                        selected_action=int(action),
                        max_negatives=max_negatives_per_positive,
                        valid_only=valid_negatives_only,
                    )
                    for neg_edge in neg_edges:
                        negative_rows.append(neg_edge)
                        negative_anchors.append(positive_idx)

        if not edge_rows:
            return self._empty_candidate_aux_samples()
        return CandidateAuxSamples(
            edge_features=np.stack(edge_rows, axis=0).astype(np.float32),
            advantages=np.asarray(targets, dtype=np.float32),
            negative_features=(
                np.stack(negative_rows, axis=0).astype(np.float32)
                if negative_rows else np.zeros((0, edge_rows[0].shape[0]), dtype=np.float32)
            ),
            negative_anchor_indices=np.asarray(negative_anchors, dtype=np.int64),
            n_conflict_edges=int(n_conflict_edges),
            conflict_penalty_sum=float(conflict_penalty_sum),
            load_penalty_sum=float(load_penalty_sum),
            edge_records=edge_records,
        )

    @staticmethod
    def _empty_candidate_aux_samples() -> CandidateAuxSamples:
        return CandidateAuxSamples(
            edge_features=np.zeros((0, 0), dtype=np.float32),
            advantages=np.zeros(0, dtype=np.float32),
            negative_features=np.zeros((0, 0), dtype=np.float32),
            negative_anchor_indices=np.zeros(0, dtype=np.int64),
        )

    @staticmethod
    def _decision_time(buffer: ActionSetRolloutBuffer, aid: str, t: int) -> float:
        times = buffer.decision_times.get(aid, [])
        if 0 <= int(t) < len(times):
            return float(times[int(t)])
        return float(t)

    @staticmethod
    def _action_entity_type(row: np.ndarray) -> int:
        row = np.asarray(row, dtype=np.float32)
        if row.shape[0] > 2 and row[2] > 0.5:
            return ACTION_WAIT
        if row.shape[0] > 1 and row[1] > 0.5:
            return ACTION_RELAY
        return ACTION_OBSERVE

    @staticmethod
    def _selected_task_counts(buffer: ActionSetRolloutBuffer) -> List[Dict[int, int]]:
        counts: List[Dict[int, int]] = []
        for t in range(len(buffer)):
            row: Dict[int, int] = {}
            for aid in buffer.actions.keys():
                action = int(buffer.actions[aid][t])
                task_id = ActionSetMAPPOTrainer._selected_task_id(buffer, aid, t, action)
                if task_id < 0:
                    continue
                row[task_id] = row.get(task_id, 0) + 1
            counts.append(row)
        return counts

    @staticmethod
    def _selected_task_id(buffer: ActionSetRolloutBuffer, aid: str, t: int, action: int) -> int:
        if aid not in buffer.candidate_task_ids:
            return -1
        task_ids = buffer.candidate_task_ids[aid][t]
        if not 0 <= int(action) < len(task_ids):
            return -1
        return int(task_ids[int(action)])

    @staticmethod
    def _hard_negative_edges(
        edge_snapshot: np.ndarray,
        action_features: np.ndarray,
        action_mask: np.ndarray,
        selected_action: int,
        max_negatives: int,
        valid_only: bool,
    ) -> List[np.ndarray]:
        nonzero = np.any(np.abs(edge_snapshot) > 0, axis=1)
        candidates = np.nonzero(nonzero)[0]
        candidates = candidates[candidates != int(selected_action)]
        if valid_only:
            candidates = candidates[np.asarray(action_mask)[candidates] > 0]
        if len(candidates) == 0:
            return []

        score_col = np.asarray(action_features)[candidates, 4]
        if np.any(np.abs(score_col) > 0):
            hardness = score_col
        else:
            hardness = np.sum(np.abs(edge_snapshot[candidates, :8]), axis=1)
        order = np.argsort(-hardness)
        selected = candidates[order[: max(0, int(max_negatives))]]
        return [np.asarray(edge_snapshot[idx], dtype=np.float32) for idx in selected]

    def update_many(self, buffers: List[ActionSetRolloutBuffer], last_global_states: List[np.ndarray]) -> Dict[str, float]:
        valid = [
            (buffer, last_global_state)
            for buffer, last_global_state in zip(buffers, last_global_states)
            if len(buffer) > 0
        ]
        if not valid:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        all_states, all_action_features, all_masks = [], [], []
        all_actions, all_old_lps, all_advantages, all_returns, all_global_states = [], [], [], [], []
        for buffer, last_global_state in valid:
            values = np.array(buffer.values, dtype=np.float32)
            global_states = np.array(buffer.global_states, dtype=np.float32)
            with torch.no_grad():
                last_value = self.model.get_values(
                    torch.FloatTensor(last_global_state).unsqueeze(0).to(self.device)
                ).cpu().item()
            for aid in buffer.state_obs.keys():
                rewards = np.array(buffer.rewards[aid], dtype=np.float32)
                dones = np.array(buffer.dones[aid], dtype=np.float32)
                advantages, returns = self.compute_gae(rewards, values, dones, last_value)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                all_states.append(np.array(buffer.state_obs[aid], dtype=np.float32))
                all_action_features.append(np.array(buffer.action_features[aid], dtype=np.float32))
                all_masks.append(np.array(buffer.action_masks[aid], dtype=np.float32))
                all_actions.append(np.array(buffer.actions[aid], dtype=np.int64))
                all_old_lps.append(np.array(buffer.log_probs[aid], dtype=np.float32))
                all_advantages.append(advantages)
                all_returns.append(returns)
                all_global_states.append(global_states)

        state_t = torch.FloatTensor(np.concatenate(all_states)).to(self.device)
        action_feat_t = torch.FloatTensor(np.concatenate(all_action_features)).to(self.device)
        mask_t = torch.FloatTensor(np.concatenate(all_masks)).to(self.device)
        action_t = torch.LongTensor(np.concatenate(all_actions)).to(self.device)
        old_lp_t = torch.FloatTensor(np.concatenate(all_old_lps)).to(self.device)
        adv_t = torch.FloatTensor(np.concatenate(all_advantages)).to(self.device)
        ret_t = torch.FloatTensor(np.concatenate(all_returns)).to(self.device)
        global_t = torch.FloatTensor(np.concatenate(all_global_states)).to(self.device)

        dataset_size = len(state_t)
        total_policy = total_value = total_entropy = total_idle_aux = 0.0
        n_updates = 0
        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, self.batch_size):
                idx = indices[start:min(start + self.batch_size, dataset_size)]
                dist = self.model.actor.forward(
                    state_t[idx],
                    action_feat_t[idx],
                    mask_t[idx],
                )
                new_lp = dist.log_prob(action_t[idx])
                entropy = dist.entropy()
                values = self.model.critic(global_t[idx])
                ratio = torch.exp(new_lp - old_lp_t[idx])
                surr1 = ratio * adv_t[idx]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_t[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, ret_t[idx])
                entropy_loss = entropy.mean()
                idle_aux_loss = self._idle_aux_loss(dist, action_feat_t[idx], mask_t[idx])
                loss = (
                    policy_loss
                    + self.value_loss_coeff * value_loss
                    - self.entropy_coeff * entropy_loss
                    + self.idle_aux_coeff * idle_aux_loss
                )
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                total_policy += float(policy_loss.item())
                total_value += float(value_loss.item())
                total_entropy += float(entropy_loss.item())
                total_idle_aux += float(idle_aux_loss.item())
                n_updates += 1

        denom = max(n_updates, 1)
        return {
            "policy_loss": total_policy / denom,
            "value_loss": total_value / denom,
            "entropy": total_entropy / denom,
            "idle_aux_loss": total_idle_aux / denom,
        }

    @staticmethod
    def _idle_aux_loss(dist, action_features: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        """Penalize idle probability only when non-idle actions are executable."""
        if action_features.shape[-1] <= 2:
            return torch.zeros((), device=action_mask.device)
        idle = torch.clamp(action_features[..., 2], 0.0, 1.0)
        valid_non_idle = torch.sum(action_mask * (1.0 - idle), dim=1) > 0
        if not torch.any(valid_non_idle):
            return torch.zeros((), device=action_mask.device)
        idle_prob = torch.sum(dist.probs * idle, dim=1)
        return -torch.log(torch.clamp(1.0 - idle_prob[valid_non_idle], min=1e-6)).mean()

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros(len(rewards), dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            next_value = last_value if t == len(rewards) - 1 else values[t + 1]
            next_done = float(dones[t])
            delta = rewards[t] + self.gamma * next_value * (1.0 - next_done) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1.0 - next_done) * last_gae
            advantages[t] = last_gae
        return advantages, advantages + values
