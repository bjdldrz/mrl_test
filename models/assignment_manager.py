"""
High-level assignment managers.

The first implementation is intentionally rule based. It consumes a structured
assignment state from MultiSatelliteEnv and proposes task-owner decisions. Later
versions can replace this class with a trainable GNN/Transformer manager while
keeping the environment API stable.
"""

from collections import defaultdict
from typing import Dict, Any


class RuleBasedAssignmentManager:
    """Risk-aware high-level task owner selector."""

    def __init__(
        self,
        deadline_weight: float = 0.25,
        stale_owner_weight: float = 0.35,
        scarcity_weight: float = 0.15,
    ):
        self.deadline_weight = float(deadline_weight)
        self.stale_owner_weight = float(stale_owner_weight)
        self.scarcity_weight = float(scarcity_weight)

    def select_owners(self, assignment_state: Dict[str, Any]) -> Dict[int, str]:
        """
        Return {mission_id: agent_id} owner proposals.

        The environment still validates candidates, switch limits, locks, and
        physical feasibility before applying these proposals.
        """
        edges_by_mission = defaultdict(list)
        for edge in assignment_state.get("edges", []):
            edges_by_mission[int(edge["mission_id"])].append(edge)

        task_by_id = {
            int(task["mission_id"]): task
            for task in assignment_state.get("tasks", [])
        }
        proposals = {}
        for mission_id, edges in edges_by_mission.items():
            task = task_by_id.get(mission_id, {})
            deadline_pressure = float(task.get("deadline_pressure", 0.0))
            stale_owner = 1.0 if task.get("owner_stale", False) else 0.0
            candidate_frac = float(task.get("candidate_frac", 1.0))
            scarcity = 1.0 - candidate_frac

            def value(edge):
                return (
                    float(edge.get("score", 0.0))
                    + self.deadline_weight * deadline_pressure
                    + self.stale_owner_weight * stale_owner
                    + self.scarcity_weight * scarcity
                )

            best = max(edges, key=value)
            proposals[mission_id] = str(best["agent_id"])
        return proposals
