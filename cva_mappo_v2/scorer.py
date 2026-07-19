from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from data.mission_generator import Mission
from .config import CVAMAPPOV2Config


@dataclass
class CandidateScore:
    agent_id: str
    mission_id: int
    score: float
    quality: float
    deadline_pressure: float
    scarcity: float
    future_loss: float
    load_pressure: float
    visible: bool
    wait_s: float = 0.0
    dynamic_response_pressure: float = 0.0
    dynamic_wait_pressure: float = 0.0
    downlink_queue_pressure: float = 0.0
    delivery_delay_pressure: float = 0.0
    downlink_feasible: float = 1.0
    estimated_downlink_queue_s: float = 0.0
    estimated_delivery_delay_s: float = 0.0


@dataclass
class PairFeatures:
    quality: float
    wait_s: float
    future_loss: float
    obs_end_s: float
    downlink_queue_s: float
    delivery_delay_s: float
    downlink_feasible: float


class CandidateValueScorer:
    """State-aware satellite-task matching scorer.

    The scorer is deliberately transparent in v2.  It exposes the pair features
    the paper text describes, and the weights are configured through
    CVAMAPPOV2Config so they can later be replaced by a trainable module.
    """

    def __init__(self, cfg: CVAMAPPOV2Config):
        self.cfg = cfg

    def score_pair(
        self,
        env,
        agent_id: str,
        mission: Mission,
        current_time_s: float,
        load_pressure: float,
        n_visible_agents: int,
        n_agents: int,
        current_owner: Optional[str],
        owner_stale: bool,
        allow_future: bool = True,
    ) -> Optional[CandidateScore]:
        pair = self._pair_features(
            env=env,
            mission=mission,
            current_time_s=current_time_s,
            allow_future=allow_future,
        )
        if pair is None:
            return None

        quality = pair.quality
        wait_s = pair.wait_s
        future_loss = pair.future_loss
        priority = float(np.clip(mission.priority / 10.0, 0.0, 1.0))
        slack_s = max(mission.deadline_s - max(current_time_s, mission.earliest_time_s), 0.0)
        deadline_pressure = 1.0 - float(np.clip(slack_s / max(env.horizon_s, 1.0), 0.0, 1.0))
        dynamic = 1.0 if mission.is_dynamic else 0.0
        scarcity = 1.0 - float(np.clip(n_visible_agents / max(1, n_agents), 0.0, 1.0))
        owner_stability = 0.0
        if current_owner == agent_id:
            owner_stability = 0.25 if owner_stale else 1.0
        elif owner_stale:
            owner_stability = 0.75

        wait_penalty = float(np.clip(wait_s / max(env.horizon_s, 1.0), 0.0, 1.0))
        response_target_s = max(float(getattr(self.cfg, "dynamic_response_target_s", 3600.0) or 3600.0), 1.0)
        dynamic_response_pressure = 0.0
        dynamic_wait_pressure = 0.0
        if mission.is_dynamic:
            arrival_s = float(getattr(mission, "arrival_time_s", mission.earliest_time_s))
            age_s = max(float(current_time_s) - arrival_s, 0.0)
            dynamic_response_pressure = float(np.clip(age_s / response_target_s, 0.0, 1.0))
            dynamic_wait_pressure = float(np.clip(wait_s / response_target_s, 0.0, 1.0))
        storage_pressure = 0.0
        if getattr(env, "storage_limited", False) and hasattr(env, "_onboard_image_count"):
            capacity = max(int(getattr(env, "satellite_storage_capacity", 0) or 0), 1)
            storage_pressure = float(np.clip(env._onboard_image_count(current_time_s) / capacity, 0.0, 1.0))
        dynamic_urgency = dynamic * max(deadline_pressure, dynamic_response_pressure)
        downlink_queue_target_s = max(
            float(getattr(self.cfg, "downlink_queue_target_s", 3600.0) or 3600.0),
            1.0,
        )
        downlink_queue_pressure = float(
            np.clip(pair.downlink_queue_s / downlink_queue_target_s, 0.0, 1.0)
        )
        delivery_delay_pressure = 0.0
        dynamic_delivery_score = 0.0
        if mission.is_dynamic:
            delivery_delay_pressure = float(
                np.clip(pair.delivery_delay_s / response_target_s, 0.0, 1.0)
            )
            if pair.downlink_feasible > 0.0:
                dynamic_delivery_score = 1.0 - delivery_delay_pressure
        cfg = self.cfg
        downlink_terms_enabled = bool(
            getattr(cfg, "downlink_aware_candidate_score", True)
            and getattr(env, "downlink_required", False)
        )
        score = (
            cfg.w_quality * quality
            + cfg.w_priority * priority
            + cfg.w_deadline * deadline_pressure
            + cfg.w_dynamic * dynamic
            + getattr(cfg, "w_dynamic_urgency", 0.0) * dynamic_urgency
            + getattr(cfg, "w_dynamic_response", 0.0) * dynamic_response_pressure
            + cfg.w_scarcity * scarcity
            + cfg.w_future_opportunity_loss * future_loss
            + cfg.w_owner_stability * owner_stability
            - cfg.w_load * load_pressure
            - getattr(cfg, "w_wait", 0.05) * wait_penalty
            - getattr(cfg, "w_dynamic_wait", 0.0) * dynamic_wait_pressure
            - getattr(cfg, "w_storage_pressure", 0.0) * storage_pressure
        )
        if downlink_terms_enabled:
            score += (
                getattr(cfg, "w_dynamic_delivery", 0.0) * dynamic * dynamic_delivery_score
                - getattr(cfg, "w_dynamic_delivery_delay", 0.0) * dynamic * delivery_delay_pressure
                - getattr(cfg, "w_downlink_queue", 0.0) * downlink_queue_pressure
                - getattr(cfg, "w_downlink_miss", 0.0) * (1.0 - float(pair.downlink_feasible))
            )
        return CandidateScore(
            agent_id=agent_id,
            mission_id=int(mission.id),
            score=float(score),
            quality=float(quality),
            deadline_pressure=float(deadline_pressure),
            scarcity=float(scarcity),
            future_loss=float(future_loss),
            load_pressure=float(load_pressure),
            visible=True,
            wait_s=float(wait_s),
            dynamic_response_pressure=float(dynamic_response_pressure),
            dynamic_wait_pressure=float(dynamic_wait_pressure),
            downlink_queue_pressure=float(downlink_queue_pressure),
            delivery_delay_pressure=float(delivery_delay_pressure),
            downlink_feasible=float(pair.downlink_feasible),
            estimated_downlink_queue_s=float(pair.downlink_queue_s),
            estimated_delivery_delay_s=float(pair.delivery_delay_s),
        )

    def _pair_features(self, env, mission: Mission, current_time_s: float, allow_future: bool):
        if mission.id not in env.mission_vtw and hasattr(env, "_compute_vtw_for_missions"):
            env._compute_vtw_for_missions([mission])

        best_quality = None
        best_key = None
        best_wait_s = 0.0
        best_obs_end_s = 0.0
        best_downlink_queue_s = 0.0
        best_delivery_delay_s = 0.0
        best_downlink_feasible = 1.0
        future_windows = 0
        response_target_s = max(float(getattr(self.cfg, "dynamic_response_target_s", 3600.0) or 3600.0), 1.0)
        dynamic_wait_weight = max(float(getattr(self.cfg, "dynamic_window_wait_weight", 0.0) or 0.0), 0.0)
        for vtw in env.mission_vtw.get(mission.id, []):
            obs_start = max(vtw.start_time, current_time_s, mission.earliest_time_s)
            obs_end = obs_start + mission.duration_s
            if obs_end > min(vtw.end_time, mission.deadline_s):
                continue
            if vtw.end_time <= current_time_s:
                continue
            future_windows += 1
            if not allow_future and not (
                vtw.start_time <= current_time_s <= vtw.end_time - mission.duration_s
            ):
                continue
            max_roll = max(env.sat_config.max_roll_deg, 1e-6)
            quality = 1.0 - min(vtw.off_nadir_deg / max_roll, 1.0)
            wait_s = max(obs_start - current_time_s, 0.0)
            if mission.is_dynamic:
                wait_pressure = float(np.clip(wait_s / response_target_s, 0.0, 1.0))
                candidate_key = float(quality) - dynamic_wait_weight * wait_pressure
            else:
                wait_norm = float(np.clip(wait_s / max(env.horizon_s, 1.0), 0.0, 1.0))
                candidate_key = float(quality) - 0.25 * float(getattr(self.cfg, "w_wait", 0.0)) * wait_norm
            downlink_queue_s, delivery_delay_s, downlink_feasible = self._downlink_features(
                env=env,
                mission=mission,
                obs_end_s=obs_end,
            )
            if bool(
                getattr(self.cfg, "downlink_aware_candidate_score", True)
                and getattr(env, "downlink_required", False)
            ):
                queue_target_s = max(
                    float(getattr(self.cfg, "downlink_queue_target_s", 3600.0) or 3600.0),
                    1.0,
                )
                queue_pressure = float(np.clip(downlink_queue_s / queue_target_s, 0.0, 1.0))
                candidate_key -= float(getattr(self.cfg, "w_downlink_queue", 0.0)) * queue_pressure
                candidate_key -= float(getattr(self.cfg, "w_downlink_miss", 0.0)) * (1.0 - downlink_feasible)
                if mission.is_dynamic:
                    delivery_pressure = float(np.clip(delivery_delay_s / response_target_s, 0.0, 1.0))
                    candidate_key += float(getattr(self.cfg, "w_dynamic_delivery", 0.0)) * (1.0 - delivery_pressure)
                    candidate_key -= float(getattr(self.cfg, "w_dynamic_delivery_delay", 0.0)) * delivery_pressure
            if best_key is None or candidate_key > best_key:
                best_key = float(candidate_key)
                best_quality = float(quality)
                best_wait_s = float(wait_s)
                best_obs_end_s = float(obs_end)
                best_downlink_queue_s = float(downlink_queue_s)
                best_delivery_delay_s = float(delivery_delay_s)
                best_downlink_feasible = float(downlink_feasible)

        if best_quality is None:
            return None
        # Few future windows means losing this opportunity is more costly.
        future_loss = 1.0 / max(future_windows, 1)
        return PairFeatures(
            quality=float(best_quality),
            wait_s=float(best_wait_s),
            future_loss=float(future_loss),
            obs_end_s=float(best_obs_end_s),
            downlink_queue_s=float(best_downlink_queue_s),
            delivery_delay_s=float(best_delivery_delay_s),
            downlink_feasible=float(best_downlink_feasible),
        )

    def _downlink_features(self, env, mission: Mission, obs_end_s: float):
        origin_s = (
            float(getattr(mission, "arrival_time_s", mission.earliest_time_s))
            if mission.is_dynamic
            else float(mission.earliest_time_s)
        )
        if not bool(getattr(self.cfg, "downlink_aware_candidate_score", True)):
            return 0.0, max(float(obs_end_s) - origin_s, 0.0), 1.0
        if not getattr(env, "downlink_required", False):
            return 0.0, max(float(obs_end_s) - origin_s, 0.0), 1.0
        if not hasattr(env, "_find_downlink_slot"):
            return 0.0, max(float(obs_end_s) - origin_s, 0.0), 1.0

        latest_end_s = min(float(getattr(env, "horizon_s", mission.deadline_s)), float(mission.deadline_s))
        availability = list(getattr(env, "_ground_station_available_s", []) or [])
        downlink_start, downlink_end, station_id = env._find_downlink_slot(
            float(obs_end_s),
            latest_end_s=latest_end_s,
            station_available_s=availability,
        )
        if station_id < 0:
            miss_end = max(float(obs_end_s), latest_end_s)
            return (
                max(miss_end - float(obs_end_s), 0.0),
                max(miss_end - origin_s, 0.0),
                0.0,
            )
        return (
            max(float(downlink_start) - float(obs_end_s), 0.0),
            max(float(downlink_end) - origin_s, 0.0),
            1.0,
        )


def score_table_to_debug(scores: Dict[int, Dict[str, CandidateScore]]) -> Dict[int, Dict[str, float]]:
    return {
        mission_id: {agent_id: score.score for agent_id, score in agent_scores.items()}
        for mission_id, agent_scores in scores.items()
    }
