from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple


@dataclass
class CandidateEdgeContext:
    score: object
    load_pressure: float
    n_visible_agents: int
    n_agents: int
    current_owner: Optional[str]
    owner_stale: bool


class V2CandidateAdapter:
    """Compatibility adapter for the current candidate-generation environment.

    DAS modules should use this adapter instead of reading compatibility-layer
    internals directly.  That keeps the replacement path for a DAS-native
    allocator narrow and explicit.
    """

    mode = "v2_compat"

    def candidate_slots(self, info: dict) -> list:
        return list(info.get("candidate_action_slots") or [])

    def slot_types(self, info: dict) -> list:
        return list(info.get("slot_types") or [])

    def slot_scores(self, info: dict) -> list:
        return list(info.get("slot_scores") or [])

    def slot_timing(self, info: dict) -> list:
        return list(info.get("slot_timing") or [])

    def n_task_candidate_owners(self, env, info: dict) -> int:
        return int(info.get("n_task_candidate_owners", len(getattr(env, "task_candidate_owners", {}))))

    def candidate_owners(self, env, mission_id: int) -> list:
        return list(getattr(env, "task_candidate_owners", {}).get(int(mission_id), []))

    def current_owner(self, env, mission_id: int) -> Optional[str]:
        return getattr(env, "task_owner", {}).get(int(mission_id))

    def owner_stale(self, env, mission_id: int) -> bool:
        if hasattr(env, "_stale_task_ids"):
            return int(mission_id) in env._stale_task_ids()
        return False

    def candidate_score(self, env, mission_id: int, agent_id: str):
        return getattr(env, "_candidate_score_table", {}).get(int(mission_id), {}).get(agent_id)

    def score_edges(self, env) -> Iterable[Tuple[int, str, object]]:
        for mission_id, agent_scores in getattr(env, "_candidate_score_table", {}).items():
            for agent_id, score in agent_scores.items():
                yield int(mission_id), agent_id, score

    def mission_by_id(self, env, agent_id: str, mission_id: int):
        if hasattr(env, "_mission_by_id"):
            return env._mission_by_id(agent_id).get(int(mission_id))
        sub_env = env.envs.get(agent_id)
        if sub_env is None:
            return None
        for mission in getattr(sub_env, "missions", []):
            if mission is not None and int(mission.id) == int(mission_id):
                return mission
        return None

    def load_pressure(self, env, agent_id: str) -> float:
        if not hasattr(env, "_candidate_load"):
            return 0.0
        loads = env._candidate_load()
        max_load = max(max(loads.values(), default=0), 1)
        return float(loads.get(agent_id, 0) / max_load)

    def n_visible_agents(self, env, mission_id: int, score=None) -> int:
        if score is not None and hasattr(score, "scarcity"):
            n_agents = max(int(getattr(env, "n_agents", 1)), 1)
            return max(1, int(round((1.0 - float(score.scarcity)) * n_agents)))
        score_row = getattr(env, "_candidate_score_table", {}).get(int(mission_id), {})
        if score_row:
            return max(1, len(score_row))
        return max(1, len(self.candidate_owners(env, mission_id)))

    def edge_context(self, env, agent_id: str, mission, scorer, current_time_s: float) -> Optional[CandidateEdgeContext]:
        score = self.candidate_score(env, mission.id, agent_id)
        if score is None:
            sub_env = env.envs[agent_id]
            score = scorer.heuristic.score_pair(
                env=sub_env,
                agent_id=agent_id,
                mission=mission,
                current_time_s=current_time_s,
                load_pressure=self.load_pressure(env, agent_id),
                n_visible_agents=self.n_visible_agents(env, mission.id),
                n_agents=getattr(env, "n_agents", 1),
                current_owner=self.current_owner(env, mission.id),
                owner_stale=self.owner_stale(env, mission.id),
                allow_future=True,
            )
        if score is None:
            return None
        return CandidateEdgeContext(
            score=score,
            load_pressure=float(getattr(score, "load_pressure", self.load_pressure(env, agent_id))),
            n_visible_agents=self.n_visible_agents(env, mission.id, score=score),
            n_agents=max(int(getattr(env, "n_agents", 1)), 1),
            current_owner=self.current_owner(env, mission.id),
            owner_stale=self.owner_stale(env, mission.id),
        )

    def edge_features(self, env, agent_id: str, sub_env, mission, scorer):
        context = self.edge_context(
            env=env,
            agent_id=agent_id,
            mission=mission,
            scorer=scorer,
            current_time_s=float(sub_env.current_time_s),
        )
        if context is None:
            return None
        return scorer.edge_features(
            env=sub_env,
            mission=mission,
            heuristic_score=context.score,
            current_time_s=float(sub_env.current_time_s),
            n_visible_agents=context.n_visible_agents,
            n_agents=context.n_agents,
            current_owner=context.current_owner,
            owner_stale=context.owner_stale,
            allow_future=True,
        )
