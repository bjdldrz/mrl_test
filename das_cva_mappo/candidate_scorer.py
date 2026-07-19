from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from cva_mappo_v2.scorer import CandidateScore, CandidateValueScorer
from .env_adapter import V2CandidateAdapter
from .temporal_features import TEMPORAL_WINDOW_FEATURE_DIM, temporal_window_features


EDGE_FEATURE_DIM = 28 + TEMPORAL_WINDOW_FEATURE_DIM


@dataclass
class ScorerWarmupStats:
    n_edges: int = 0
    final_loss: float = 0.0


@dataclass
class ScorerAuxUpdateStats:
    n_edges: int = 0
    n_positive_edges: int = 0
    n_negative_edges: int = 0
    value_loss: float = 0.0
    rank_loss: float = 0.0
    total_loss: float = 0.0
    target_mean: float = 0.0
    target_std: float = 0.0


class EdgeValueMLP(nn.Module):
    def __init__(self, input_dim: int = EDGE_FEATURE_DIM, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class TrainableCandidateValueScorer:
    """DAS candidate scorer with the same score_pair API as CVA-MAPPO v2.

    The scorer uses the v2 heuristic scorer as a feasibility oracle and warm-start
    teacher.  In `hybrid` mode the learned edge value is mixed with the heuristic
    value; in `learned` mode candidate ranking is driven by the learned value.
    """

    def __init__(
        self,
        v2_cfg,
        mode: str = "hybrid",
        mix: float = 0.35,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        device: str = "cpu",
        use_response_budget_features: bool = True,
        use_temporal_window_features: bool = True,
        use_early_delivery_temporal_features: bool = True,
        temporal_window_top_k: int = 3,
        temporal_early_delivery_weight: float = 0.35,
    ):
        if mode not in {"v2_heuristic", "learned", "hybrid"}:
            raise ValueError("candidate scorer mode must be v2_heuristic, learned, or hybrid")
        self.heuristic = CandidateValueScorer(v2_cfg)
        self.mode = str(mode)
        self.mix = float(np.clip(mix, 0.0, 1.0))
        self.device = torch.device(device)
        self.use_response_budget_features = bool(use_response_budget_features)
        self.use_temporal_window_features = bool(use_temporal_window_features)
        self.use_early_delivery_temporal_features = bool(use_early_delivery_temporal_features)
        self.temporal_window_top_k = max(int(temporal_window_top_k), 1)
        self.temporal_early_delivery_weight = max(float(temporal_early_delivery_weight), 0.0)
        self.model = EdgeValueMLP(EDGE_FEATURE_DIM, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=float(lr))
        self.warmup_stats = ScorerWarmupStats()
        self.last_aux_update_stats = ScorerAuxUpdateStats()
        self.aux_update_count = 0
        self.aux_edges_seen = 0

    def score_pair(
        self,
        env,
        agent_id: str,
        mission,
        current_time_s: float,
        load_pressure: float,
        n_visible_agents: int,
        n_agents: int,
        current_owner: Optional[str],
        owner_stale: bool,
        allow_future: bool = True,
    ) -> Optional[CandidateScore]:
        heuristic_score = self.heuristic.score_pair(
            env=env,
            agent_id=agent_id,
            mission=mission,
            current_time_s=current_time_s,
            load_pressure=load_pressure,
            n_visible_agents=n_visible_agents,
            n_agents=n_agents,
            current_owner=current_owner,
            owner_stale=owner_stale,
            allow_future=allow_future,
        )
        if heuristic_score is None or self.mode == "v2_heuristic":
            return heuristic_score

        features = self.edge_features(
            env=env,
            mission=mission,
            heuristic_score=heuristic_score,
            current_time_s=current_time_s,
            n_visible_agents=n_visible_agents,
            n_agents=n_agents,
            current_owner=current_owner,
            owner_stale=owner_stale,
            allow_future=allow_future,
        )
        with torch.no_grad():
            learned = self.model(
                torch.FloatTensor(features).unsqueeze(0).to(self.device)
            ).cpu().item()

        if self.mode == "learned":
            score = float(learned)
        elif self.mode == "hybrid":
            score = (1.0 - self.mix) * float(heuristic_score.score) + self.mix * float(learned)
        else:
            raise ValueError("candidate scorer mode must be v2_heuristic, learned, or hybrid")

        return CandidateScore(
            agent_id=heuristic_score.agent_id,
            mission_id=heuristic_score.mission_id,
            score=score,
            quality=heuristic_score.quality,
            deadline_pressure=heuristic_score.deadline_pressure,
            scarcity=heuristic_score.scarcity,
            future_loss=heuristic_score.future_loss,
            load_pressure=heuristic_score.load_pressure,
            visible=heuristic_score.visible,
            wait_s=float(getattr(heuristic_score, "wait_s", 0.0)),
            dynamic_response_pressure=float(getattr(heuristic_score, "dynamic_response_pressure", 0.0)),
            dynamic_wait_pressure=float(getattr(heuristic_score, "dynamic_wait_pressure", 0.0)),
            downlink_queue_pressure=float(getattr(heuristic_score, "downlink_queue_pressure", 0.0)),
            delivery_delay_pressure=float(getattr(heuristic_score, "delivery_delay_pressure", 0.0)),
            downlink_feasible=float(getattr(heuristic_score, "downlink_feasible", 1.0)),
            estimated_downlink_queue_s=float(getattr(heuristic_score, "estimated_downlink_queue_s", 0.0)),
            estimated_delivery_delay_s=float(getattr(heuristic_score, "estimated_delivery_delay_s", 0.0)),
        )

    def edge_features(
        self,
        env,
        mission,
        heuristic_score: CandidateScore,
        current_time_s: float,
        n_visible_agents: int,
        n_agents: int,
        current_owner: Optional[str],
        owner_stale: bool,
        allow_future: bool,
    ) -> np.ndarray:
        priority = float(np.clip(mission.priority / 10.0, 0.0, 1.0))
        dynamic = 1.0 if mission.is_dynamic else 0.0
        owner_stability = 1.0 if current_owner == heuristic_score.agent_id else (0.5 if owner_stale else 0.0)
        wait_norm = self._wait_norm(env, mission, current_time_s)
        current_norm = float(np.clip(current_time_s / max(env.horizon_s, 1.0), 0.0, 1.0))
        earliest_norm = float(np.clip(mission.earliest_time_s / max(env.horizon_s, 1.0), 0.0, 1.0))
        deadline_norm = float(np.clip(mission.deadline_s / max(env.horizon_s, 1.0), 0.0, 1.0))
        duration_norm = float(np.clip(mission.duration_s / max(env.horizon_s, 1.0), 0.0, 1.0))
        visible_frac = float(np.clip(n_visible_agents / max(n_agents, 1), 0.0, 1.0))
        schedule_load = float(np.clip(len(getattr(env, "schedule_log", [])) / 256.0, 0.0, 1.0))
        storage_pressure = 0.0
        if hasattr(env, "_onboard_image_count"):
            capacity = max(int(getattr(env, "satellite_storage_capacity", 0) or 0), 1)
            storage_pressure = float(np.clip(env._onboard_image_count(current_time_s) / capacity, 0.0, 1.0))
        downlink_enabled = 1.0 if getattr(env, "n_ground_stations", 0) > 0 and getattr(env, "downlink_time_s", 0.0) > 0 else 0.0
        dynamic_response_pressure = float(getattr(heuristic_score, "dynamic_response_pressure", 0.0))
        dynamic_wait_pressure = float(getattr(heuristic_score, "dynamic_wait_pressure", 0.0))
        downlink_queue_pressure = float(getattr(heuristic_score, "downlink_queue_pressure", 0.0))
        delivery_delay_pressure = float(getattr(heuristic_score, "delivery_delay_pressure", 0.0))
        downlink_feasible = float(getattr(heuristic_score, "downlink_feasible", 1.0))
        downlink_queue_norm = float(
            np.clip(
                float(getattr(heuristic_score, "estimated_downlink_queue_s", 0.0))
                / max(float(getattr(env, "horizon_s", 1.0)), 1.0),
                0.0,
                1.0,
            )
        )
        response_budget = (
            self._response_budget_features(env, mission, heuristic_score, current_time_s)
            if self.use_response_budget_features
            else (0.0, 0.0, 0.0, 0.0)
        )
        temporal_windows = (
            temporal_window_features(
                env=env,
                mission=mission,
                current_time_s=current_time_s,
                top_k=self.temporal_window_top_k,
                response_target_s=getattr(self.heuristic.cfg, "dynamic_response_target_s", 3600.0),
                downlink_queue_target_s=getattr(self.heuristic.cfg, "downlink_queue_target_s", 3600.0),
                downlink_feature_fn=self.heuristic._downlink_features,
                use_early_delivery_features=self.use_early_delivery_temporal_features,
                early_delivery_weight=self.temporal_early_delivery_weight,
            )
            if self.use_temporal_window_features
            else np.zeros(TEMPORAL_WINDOW_FEATURE_DIM, dtype=np.float32)
        )
        return np.array([
            heuristic_score.quality,
            priority,
            heuristic_score.deadline_pressure,
            dynamic,
            heuristic_score.scarcity,
            heuristic_score.future_loss,
            heuristic_score.load_pressure,
            owner_stability,
            wait_norm,
            current_norm,
            earliest_norm,
            deadline_norm,
            duration_norm,
            visible_frac,
            schedule_load,
            storage_pressure,
            downlink_enabled,
            1.0 if allow_future else 0.0,
            dynamic_response_pressure,
            dynamic_wait_pressure,
            downlink_queue_pressure,
            delivery_delay_pressure,
            downlink_feasible,
            downlink_queue_norm,
            *response_budget,
            *temporal_windows,
        ], dtype=np.float32)

    def warm_start(
        self,
        env_factory: Callable[[], object],
        scenarios: Iterable[Tuple[list, list]],
        max_edges: int = 4096,
        epochs: int = 2,
        batch_size: int = 256,
        candidate_adapter=None,
    ) -> ScorerWarmupStats:
        if self.mode == "v2_heuristic" or epochs <= 0 or max_edges <= 0:
            self.warmup_stats = ScorerWarmupStats()
            return self.warmup_stats

        features, targets = self._collect_teacher_edges(
            env_factory,
            scenarios,
            max_edges,
            candidate_adapter=candidate_adapter,
        )
        if len(features) == 0:
            self.warmup_stats = ScorerWarmupStats()
            return self.warmup_stats

        x = torch.FloatTensor(np.stack(features, axis=0)).to(self.device)
        y = torch.FloatTensor(np.array(targets, dtype=np.float32)).to(self.device)
        final_loss = 0.0
        batch_size = max(int(batch_size), 1)
        for _ in range(int(epochs)):
            indices = torch.randperm(len(x), device=self.device)
            for start in range(0, len(x), batch_size):
                idx = indices[start:start + batch_size]
                pred = self.model(x[idx])
                loss = torch.mean((pred - y[idx]) ** 2)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                final_loss = float(loss.detach().cpu().item())
        self.warmup_stats = ScorerWarmupStats(n_edges=len(features), final_loss=final_loss)
        return self.warmup_stats

    def update_from_rollout(
        self,
        edge_features: np.ndarray,
        advantages: np.ndarray,
        negative_features: Optional[np.ndarray] = None,
        negative_anchor_indices: Optional[np.ndarray] = None,
        epochs: int = 1,
        batch_size: int = 256,
        rank_weight: float = 0.2,
        target_clip: float = 3.0,
        min_edges: int = 4,
        negative_margin: float = 0.25,
        negative_value_weight: float = 0.5,
    ) -> ScorerAuxUpdateStats:
        if self.mode == "v2_heuristic" or epochs <= 0:
            self.last_aux_update_stats = ScorerAuxUpdateStats()
            return self.last_aux_update_stats

        features = np.asarray(edge_features, dtype=np.float32)
        targets = np.asarray(advantages, dtype=np.float32).reshape(-1)
        if features.ndim != 2 or features.shape[1] != EDGE_FEATURE_DIM or len(features) != len(targets):
            self.last_aux_update_stats = ScorerAuxUpdateStats()
            return self.last_aux_update_stats

        finite = np.isfinite(features).all(axis=1) & np.isfinite(targets)
        nonzero = np.any(np.abs(features) > 0, axis=1)
        keep = finite & nonzero
        old_to_new = np.full(len(features), -1, dtype=np.int64)
        old_to_new[np.nonzero(keep)[0]] = np.arange(int(np.sum(keep)))
        pos_features = features[keep]
        pos_targets = targets[keep]
        if len(pos_features) == 0:
            self.last_aux_update_stats = ScorerAuxUpdateStats()
            return self.last_aux_update_stats

        target_mean = float(np.mean(pos_targets))
        target_std = float(np.std(pos_targets) + 1e-8)
        pos_y = (pos_targets - target_mean) / target_std
        pos_y = np.clip(pos_y, -float(target_clip), float(target_clip)).astype(np.float32)

        neg_features, neg_anchor = self._prepare_negative_edges(
            negative_features=negative_features,
            negative_anchor_indices=negative_anchor_indices,
            old_to_new=old_to_new,
        )
        if len(neg_features) > 0:
            neg_y = pos_y[neg_anchor] - float(negative_margin)
            neg_y = np.clip(neg_y, -float(target_clip), float(target_clip)).astype(np.float32)
            train_features = np.concatenate([pos_features, neg_features], axis=0)
            train_targets = np.concatenate([pos_y, neg_y], axis=0)
            weights = np.concatenate([
                np.ones(len(pos_features), dtype=np.float32),
                np.full(len(neg_features), max(float(negative_value_weight), 0.0), dtype=np.float32),
            ])
        else:
            train_features = pos_features
            train_targets = pos_y
            weights = np.ones(len(pos_features), dtype=np.float32)

        if len(train_features) < int(min_edges):
            self.last_aux_update_stats = ScorerAuxUpdateStats(
                n_edges=int(len(train_features)),
                n_positive_edges=int(len(pos_features)),
                n_negative_edges=int(len(neg_features)),
            )
            return self.last_aux_update_stats

        x = torch.FloatTensor(train_features).to(self.device)
        y = torch.FloatTensor(train_targets).to(self.device)
        sample_w = torch.FloatTensor(weights).to(self.device)
        batch_size = max(int(batch_size), 1)
        rank_weight = max(float(rank_weight), 0.0)
        final_value = final_rank = final_total = 0.0
        n_updates = 0

        for _ in range(int(epochs)):
            indices = torch.randperm(len(x), device=self.device)
            for start in range(0, len(x), batch_size):
                idx = indices[start:start + batch_size]
                pred = self.model(x[idx])
                y_batch = y[idx]
                w_batch = sample_w[idx]
                value_loss = torch.mean(w_batch * (pred - y_batch) ** 2) / torch.mean(w_batch).clamp_min(1e-6)
                rank_loss = pred.new_tensor(0.0)

                if rank_weight > 0.0 and len(idx) > 1:
                    perm = torch.randperm(len(idx), device=self.device)
                    delta = y_batch - y_batch[perm]
                    pair_mask = torch.abs(delta) > 1e-6
                    if torch.any(pair_mask):
                        direction = torch.sign(delta[pair_mask])
                        rank_loss = F.softplus(
                            -direction * (pred[pair_mask] - pred[perm][pair_mask])
                        ).mean()

                loss = value_loss + rank_weight * rank_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                final_value = float(value_loss.detach().cpu().item())
                final_rank = float(rank_loss.detach().cpu().item())
                final_total = float(loss.detach().cpu().item())
                n_updates += 1

        self.aux_update_count += int(n_updates > 0)
        self.aux_edges_seen += int(len(train_features))
        self.last_aux_update_stats = ScorerAuxUpdateStats(
            n_edges=int(len(train_features)),
            n_positive_edges=int(len(pos_features)),
            n_negative_edges=int(len(neg_features)),
            value_loss=final_value,
            rank_loss=final_rank,
            total_loss=final_total,
            target_mean=target_mean,
            target_std=target_std,
        )
        return self.last_aux_update_stats

    @staticmethod
    def _prepare_negative_edges(
        negative_features: Optional[np.ndarray],
        negative_anchor_indices: Optional[np.ndarray],
        old_to_new: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if negative_features is None or negative_anchor_indices is None:
            return (
                np.zeros((0, EDGE_FEATURE_DIM), dtype=np.float32),
                np.zeros(0, dtype=np.int64),
            )
        neg = np.asarray(negative_features, dtype=np.float32)
        anchors = np.asarray(negative_anchor_indices, dtype=np.int64).reshape(-1)
        if neg.ndim != 2 or neg.shape[1] != EDGE_FEATURE_DIM or len(neg) != len(anchors):
            return (
                np.zeros((0, EDGE_FEATURE_DIM), dtype=np.float32),
                np.zeros(0, dtype=np.int64),
            )
        valid_anchor = (anchors >= 0) & (anchors < len(old_to_new))
        remapped = np.full(len(anchors), -1, dtype=np.int64)
        remapped[valid_anchor] = old_to_new[anchors[valid_anchor]]
        keep = (
            valid_anchor
            & (remapped >= 0)
            & np.isfinite(neg).all(axis=1)
            & np.any(np.abs(neg) > 0, axis=1)
        )
        return neg[keep], remapped[keep]

    def _collect_teacher_edges(
        self,
        env_factory: Callable[[], object],
        scenarios: Iterable[Tuple[list, list]],
        max_edges: int,
        candidate_adapter=None,
    ) -> Tuple[List[np.ndarray], List[float]]:
        adapter = candidate_adapter or V2CandidateAdapter()
        features: List[np.ndarray] = []
        targets: List[float] = []
        for routine, dynamic in scenarios:
            env = env_factory()
            env.scorer = self.heuristic
            env.reset(options={
                "routine_missions": copy.deepcopy(routine),
                "dynamic_schedule": copy.deepcopy(dynamic),
            })
            for mission_id, agent_id, score in adapter.score_edges(env):
                mission = adapter.mission_by_id(env, agent_id, mission_id)
                if mission is None:
                    continue
                sub_env = env.envs[agent_id]
                features.append(
                    self.edge_features(
                        env=sub_env,
                        mission=mission,
                        heuristic_score=score,
                        current_time_s=float(sub_env.current_time_s),
                        n_visible_agents=adapter.n_visible_agents(env, mission_id, score=score),
                        n_agents=getattr(env, "n_agents", 1),
                        current_owner=adapter.current_owner(env, mission.id),
                        owner_stale=adapter.owner_stale(env, mission.id),
                        allow_future=True,
                    )
                )
                targets.append(float(score.score))
                if len(features) >= max_edges:
                    return features, targets
        return features, targets

    @staticmethod
    def _wait_norm(env, mission, current_time_s: float) -> float:
        wait_s = env.horizon_s
        for vtw in env.mission_vtw.get(mission.id, []):
            if vtw.end_time <= current_time_s:
                continue
            obs_start = max(vtw.start_time, current_time_s, mission.earliest_time_s)
            obs_end = obs_start + mission.duration_s
            if obs_end <= min(vtw.end_time, mission.deadline_s):
                wait_s = max(obs_start - current_time_s, 0.0)
                break
        return float(np.clip(wait_s / max(env.horizon_s, 1.0), 0.0, 1.0))

    def _response_budget_features(
        self,
        env,
        mission,
        heuristic_score: CandidateScore,
        current_time_s: float,
    ) -> Tuple[float, float, float, float]:
        if not getattr(mission, "is_dynamic", False):
            return 0.0, 0.0, 0.0, 0.0
        response_target_s = max(
            float(getattr(self.heuristic.cfg, "dynamic_response_target_s", 3600.0) or 3600.0),
            1.0,
        )
        # Prefer the scorer's normalized pressure when available so the feature
        # stays aligned with the v2 candidate scoring configuration.
        age_pressure = float(np.clip(
            getattr(heuristic_score, "dynamic_response_pressure", 0.0),
            0.0,
            1.0,
        ))
        if age_pressure <= 0.0:
            arrival_s = float(getattr(mission, "arrival_time_s", mission.earliest_time_s))
            age_s = max(float(current_time_s) - arrival_s, 0.0)
            age_pressure = float(np.clip(age_s / response_target_s, 0.0, 1.0))
        budget_remaining = float(np.clip(1.0 - age_pressure, 0.0, 1.0))
        overrun = 1.0 if age_pressure >= 1.0 else 0.0
        delivery_budget_pressure = float(np.clip(
            getattr(heuristic_score, "estimated_delivery_delay_s", 0.0) / response_target_s,
            0.0,
            1.0,
        ))
        return age_pressure, budget_remaining, overrun, delivery_budget_pressure
