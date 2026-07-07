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

        quality, wait_s, future_loss = pair
        priority = float(np.clip(mission.priority / 10.0, 0.0, 1.0))
        slack_s = max(mission.deadline_s - max(current_time_s, mission.earliest_time_s), 0.0)
        deadline_pressure = 1.0 - float(np.clip(slack_s / max(env.horizon_s, 1.0), 0.0, 1.0))
        dynamic = 1.0 if mission.is_dynamic else 0.0
        scarcity = 1.0 - float(np.clip(n_visible_agents / max(1, n_agents), 0.0, 1.0))
        owner_stability = 0.0
        if current_owner == agent_id:
            owner_stability = 1.0
        elif owner_stale:
            owner_stability = 0.5

        wait_penalty = float(np.clip(wait_s / max(env.horizon_s, 1.0), 0.0, 1.0))
        cfg = self.cfg
        score = (
            cfg.w_quality * quality
            + cfg.w_priority * priority
            + cfg.w_deadline * deadline_pressure
            + cfg.w_dynamic * dynamic
            + cfg.w_scarcity * scarcity
            + cfg.w_future_opportunity_loss * future_loss
            + cfg.w_owner_stability * owner_stability
            - cfg.w_load * load_pressure
            - 0.05 * wait_penalty
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
        )

    def _pair_features(self, env, mission: Mission, current_time_s: float, allow_future: bool):
        if mission.id not in env.mission_vtw and hasattr(env, "_compute_vtw_for_missions"):
            env._compute_vtw_for_missions([mission])

        best_quality = None
        best_wait_s = 0.0
        future_windows = 0
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
            if best_quality is None or quality > best_quality:
                best_quality = float(quality)
                best_wait_s = float(wait_s)

        if best_quality is None:
            return None
        # Few future windows means losing this opportunity is more costly.
        future_loss = 1.0 / max(future_windows, 1)
        return best_quality, best_wait_s, float(future_loss)


def score_table_to_debug(scores: Dict[int, Dict[str, CandidateScore]]) -> Dict[int, Dict[str, float]]:
    return {
        mission_id: {agent_id: score.score for agent_id, score in agent_scores.items()}
        for mission_id, agent_scores in scores.items()
    }
