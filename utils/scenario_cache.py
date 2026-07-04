"""
Scenario cache helpers.

The cache keeps training and evaluation scenarios separate on disk. Training
scenarios may be organized into curriculum stages, while evaluation scenarios
are always read as a held-out fixed set.
"""

from __future__ import annotations

import copy
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union


Scenario = Tuple[list, list]


def save_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_scenario_cache(cache_dir: Union[str, Path]) -> Dict[str, Any]:
    root = Path(cache_dir)
    train_path = root / "train_scenarios.pkl"
    eval_path = root / "eval_scenarios.pkl"
    if not train_path.exists():
        raise FileNotFoundError(f"未找到训练场景缓存: {train_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"未找到评估场景缓存: {eval_path}")

    manifest_path = root / "manifest.json"
    return {
        "root": root,
        "train": load_pickle(train_path),
        "eval": load_pickle(eval_path),
        "manifest_path": manifest_path if manifest_path.exists() else None,
    }


def get_eval_scenarios(eval_payload: Any) -> List[Scenario]:
    if isinstance(eval_payload, dict):
        scenarios = eval_payload.get("scenarios", [])
    else:
        scenarios = eval_payload
    return list(scenarios)


def get_train_stage_count(train_payload: Any) -> int:
    if isinstance(train_payload, dict) and train_payload.get("stages"):
        return len(train_payload["stages"])
    return 1


def get_train_scenario_count(train_payload: Any) -> int:
    if isinstance(train_payload, dict) and train_payload.get("stages"):
        return sum(len(stage.get("scenarios", [])) for stage in train_payload["stages"])
    return len(train_payload or [])


def select_train_scenario(
    train_payload: Any,
    iteration: int,
    total_iterations: int,
    rng,
) -> Scenario:
    """
    Select one training scenario.

    If the training cache contains curriculum stages, the current iteration
    chooses a stage from simple to complex. Within that stage, scenarios are
    sampled randomly.
    """
    if train_payload is None:
        raise ValueError("train_payload 不能为空")

    if isinstance(train_payload, dict) and train_payload.get("stages"):
        stages = train_payload["stages"]
        total_iterations = max(int(total_iterations), 1)
        progress = min(max(iteration / total_iterations, 0.0), 0.999999)
        stage_idx = min(int(progress * len(stages)), len(stages) - 1)
        scenarios = stages[stage_idx].get("scenarios", [])
        if not scenarios:
            raise ValueError(f"训练场景 stage={stage_idx} 为空")
        return copy_scenario(scenarios[int(rng.randint(0, len(scenarios)))])

    scenarios = list(train_payload)
    if not scenarios:
        raise ValueError("训练场景缓存为空")
    return copy_scenario(scenarios[int(rng.randint(0, len(scenarios)))])


def copy_scenario(scenario: Scenario) -> Scenario:
    routine, dynamic = scenario
    return copy.deepcopy(routine), copy.deepcopy(dynamic)


def iter_scenario_missions(scenarios: Iterable[Scenario]):
    for routine, dynamic in scenarios:
        for mission in routine:
            yield mission
        for _, missions in dynamic:
            for mission in missions:
                yield mission


def flatten_train_scenarios(train_payload: Any) -> List[Scenario]:
    if isinstance(train_payload, dict) and train_payload.get("stages"):
        out: List[Scenario] = []
        for stage in train_payload["stages"]:
            out.extend(stage.get("scenarios", []))
        return out
    return list(train_payload or [])


def scenario_summary(cache_payload: Dict[str, Any]) -> Dict[str, Any]:
    train_payload = cache_payload["train"]
    eval_scenarios = get_eval_scenarios(cache_payload["eval"])
    return {
        "root": str(cache_payload["root"]),
        "train_scenarios": get_train_scenario_count(train_payload),
        "train_stages": get_train_stage_count(train_payload),
        "eval_scenarios": len(eval_scenarios),
    }
