from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from data.mission_generator import Mission
from .config import CVAMAPPOV2Config
from .scorer import CandidateScore


@dataclass
class CandidateAssignment:
    """Task-centered candidate ownership."""

    task_candidates: Dict[int, List[str]] = field(default_factory=dict)
    primary_owner: Dict[int, str] = field(default_factory=dict)
    owner_switches: int = 0


class CapacityAwareTaskAllocator:
    """Assign each task to one or more candidate satellites under capacities."""

    def __init__(self, cfg: CVAMAPPOV2Config, agent_ids: List[str]):
        self.cfg = cfg
        self.agent_ids = list(agent_ids)

    def allocate(
        self,
        missions: Iterable[Mission],
        score_table: Dict[int, Dict[str, CandidateScore]],
        current_candidates: Dict[int, List[str]],
        current_primary: Dict[int, str],
        current_load: Dict[str, int],
        current_time_s: float,
        horizon_s: float,
        stale_tasks: Optional[set] = None,
    ) -> CandidateAssignment:
        stale_tasks = stale_tasks or set()
        missions = [m for m in missions if m is not None and not m.is_observed]
        capacities = self._capacities(missions, current_load)
        load = {aid: int(current_load.get(aid, 0)) for aid in self.agent_ids}

        assignment = CandidateAssignment(
            task_candidates=dict(current_candidates),
            primary_owner=dict(current_primary),
            owner_switches=0,
        )

        ordered = sorted(
            missions,
            key=lambda m: (
                len(score_table.get(m.id, {})),
                m.deadline_s,
                m.earliest_time_s,
                -m.priority,
            ),
        )

        for mission in ordered:
            agent_scores = score_table.get(mission.id, {})
            if not agent_scores:
                continue
            target_k = self._target_candidate_count(
                mission=mission,
                current_time_s=current_time_s,
                horizon_s=horizon_s,
                stale=mission.id in stale_tasks,
            )
            old_primary = assignment.primary_owner.get(mission.id)
            ranked_pairs = sorted(
                (
                    (score, self._capacity_adjusted_score(score, load, capacities))
                    for score in agent_scores.values()
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            ranked = self._stabilize_primary(
                ranked_pairs=ranked_pairs,
                old_primary=old_primary,
                stale=mission.id in stale_tasks,
            )
            chosen = self._choose_candidates(ranked, target_k, load, capacities)
            if not chosen:
                continue

            new_primary = chosen[0]
            if old_primary is not None and old_primary != new_primary:
                assignment.owner_switches += 1
            assignment.primary_owner[mission.id] = new_primary
            assignment.task_candidates[mission.id] = chosen
            for aid in chosen:
                load[aid] = load.get(aid, 0) + 1

        return assignment

    def _capacities(self, missions: List[Mission], current_load: Dict[str, int]) -> Dict[str, int]:
        total_slots = self.cfg.slots.total_slots
        # Capacity is expressed in candidate slots, not final executed tasks.
        total_capacity = max(len(self.agent_ids) * total_slots, len(missions))
        base = int(np.ceil(total_capacity / max(len(self.agent_ids), 1)))
        slack = int(np.ceil(base * self.cfg.capacity_slack_ratio))
        return {
            aid: max(base + slack, int(current_load.get(aid, 0)) + 1)
            for aid in self.agent_ids
        }

    def _target_candidate_count(
        self,
        mission: Mission,
        current_time_s: float,
        horizon_s: float,
        stale: bool,
    ) -> int:
        if stale:
            return self.cfg.stale_candidate_owners
        slack_s = max(mission.deadline_s - max(current_time_s, mission.earliest_time_s), 0.0)
        urgent = slack_s <= self.cfg.release_before_deadline_s
        if urgent:
            return self.cfg.urgent_candidate_owners
        if mission.is_dynamic:
            return self.cfg.dynamic_candidate_owners
        return self.cfg.routine_candidate_owners

    def _capacity_adjusted_score(
        self,
        score: CandidateScore,
        load: Dict[str, int],
        capacity: Dict[str, int],
    ) -> float:
        pressure = load.get(score.agent_id, 0) / max(capacity.get(score.agent_id, 1), 1)
        return score.score - self.cfg.load_penalty * pressure

    def _choose_candidates(
        self,
        ranked: List[CandidateScore],
        target_k: int,
        load: Dict[str, int],
        capacity: Dict[str, int],
    ) -> List[str]:
        chosen = []
        for score in ranked:
            if score.agent_id in chosen:
                continue
            if load.get(score.agent_id, 0) >= capacity.get(score.agent_id, 0):
                continue
            chosen.append(score.agent_id)
            if len(chosen) >= target_k:
                break
        if chosen:
            return chosen

        # If all candidates are at capacity, keep the best candidate rather than
        # dropping the task entirely.
        return [ranked[0].agent_id] if ranked else []

    def _stabilize_primary(
        self,
        ranked_pairs: List[Tuple[CandidateScore, float]],
        old_primary: Optional[str],
        stale: bool,
    ) -> List[CandidateScore]:
        if not ranked_pairs:
            return []
        if old_primary is None or stale:
            return [score for score, _ in ranked_pairs]

        best_score, best_adjusted = ranked_pairs[0]
        if best_score.agent_id == old_primary:
            return [score for score, _ in ranked_pairs]

        old_pair = next(
            ((score, adjusted) for score, adjusted in ranked_pairs if score.agent_id == old_primary),
            None,
        )
        if old_pair is None:
            return [score for score, _ in ranked_pairs]

        old_score, old_adjusted = old_pair
        required_gain = max(0.0, self.cfg.owner_switch_margin + self.cfg.switch_penalty)
        if best_adjusted < old_adjusted + required_gain:
            rest = [score for score, _ in ranked_pairs if score.agent_id != old_primary]
            return [old_score] + rest
        return [score for score, _ in ranked_pairs]
