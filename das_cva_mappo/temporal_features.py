from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np


TEMPORAL_WINDOW_FEATURE_DIM = 10


def temporal_window_features(
    env,
    mission,
    current_time_s: float,
    top_k: int = 3,
    response_target_s: float = 3600.0,
    downlink_queue_target_s: float = 3600.0,
    downlink_feature_fn: Optional[Callable[[object, object, float], Tuple[float, float, float]]] = None,
) -> np.ndarray:
    """Summarize the future feasible observation/downlink sequence for an edge."""

    windows = _feasible_windows(
        env=env,
        mission=mission,
        current_time_s=float(current_time_s),
        top_k=max(int(top_k), 1),
        downlink_feature_fn=downlink_feature_fn,
    )
    if not windows:
        return np.zeros(TEMPORAL_WINDOW_FEATURE_DIM, dtype=np.float32)

    horizon_s = max(float(getattr(env, "horizon_s", 1.0)), 1.0)
    response_target_s = max(float(response_target_s or 0.0), 1.0)
    downlink_queue_target_s = max(float(downlink_queue_target_s or 0.0), 1.0)
    dynamic = bool(getattr(mission, "is_dynamic", False))
    time_target_s = response_target_s if dynamic else horizon_s

    count = len(windows)
    first = windows[0]
    best = max(
        windows,
        key=lambda item: (
            item["quality"]
            - 0.25 * _clip01(item["wait_s"] / time_target_s)
            - 0.10 * _clip01(item["delivery_delay_s"] / time_target_s)
            - 0.10 * _clip01(item["downlink_queue_s"] / downlink_queue_target_s)
        ),
    )
    qualities = [item["quality"] for item in windows]
    min_delivery_delay_s = min(item["delivery_delay_s"] for item in windows)
    min_downlink_queue_s = min(item["downlink_queue_s"] for item in windows)
    any_downlink_feasible = max(item["downlink_feasible"] for item in windows)
    first_quality = float(first["quality"])
    last_quality = float(windows[-1]["quality"])

    if dynamic:
        budget_remaining = _clip01(
            (response_target_s - float(best["delivery_delay_s"])) / response_target_s
        )
    else:
        budget_remaining = _clip01(
            (float(mission.deadline_s) - float(best["obs_end_s"])) / horizon_s
        )

    return np.array([
        _clip01(count / max(int(top_k), 1)),
        _clip01(float(first["wait_s"]) / time_target_s),
        _clip01(float(best["wait_s"]) / time_target_s),
        _clip01(float(best["quality"])),
        _clip01(float(np.mean(qualities))),
        _clip11(last_quality - first_quality),
        _clip01(float(min_delivery_delay_s) / time_target_s),
        _clip01(float(min_downlink_queue_s) / downlink_queue_target_s),
        _clip01(float(any_downlink_feasible)),
        _clip01(float(budget_remaining)),
    ], dtype=np.float32)


def _feasible_windows(
    env,
    mission,
    current_time_s: float,
    top_k: int,
    downlink_feature_fn: Optional[Callable[[object, object, float], Tuple[float, float, float]]],
) -> List[dict]:
    if mission.id not in getattr(env, "mission_vtw", {}) and hasattr(env, "_compute_vtw_for_missions"):
        env._compute_vtw_for_missions([mission])

    horizon_s = max(float(getattr(env, "horizon_s", 1.0)), 1.0)
    max_roll = max(float(getattr(getattr(env, "sat_config", None), "max_roll_deg", 1.0)), 1e-6)
    rows: List[dict] = []
    vtws = sorted(
        getattr(env, "mission_vtw", {}).get(mission.id, []),
        key=lambda item: float(getattr(item, "start_time", 0.0)),
    )
    for vtw in vtws:
        if float(vtw.end_time) <= current_time_s:
            continue
        obs_start = max(float(vtw.start_time), current_time_s, float(mission.earliest_time_s))
        obs_end = obs_start + float(mission.duration_s)
        if obs_end > min(float(vtw.end_time), float(mission.deadline_s)):
            continue
        wait_s = max(obs_start - current_time_s, 0.0)
        quality = 1.0 - min(float(getattr(vtw, "off_nadir_deg", max_roll)) / max_roll, 1.0)
        downlink_queue_s, delivery_delay_s, downlink_feasible = _downlink_features(
            env=env,
            mission=mission,
            obs_end_s=obs_end,
            downlink_feature_fn=downlink_feature_fn,
        )
        rows.append({
            "obs_start_s": float(obs_start),
            "obs_end_s": float(obs_end),
            "wait_s": float(wait_s),
            "quality": float(quality),
            "downlink_queue_s": float(downlink_queue_s),
            "delivery_delay_s": float(delivery_delay_s),
            "downlink_feasible": float(downlink_feasible),
        })
        if len(rows) >= top_k:
            break

    return rows[:top_k]


def _downlink_features(
    env,
    mission,
    obs_end_s: float,
    downlink_feature_fn: Optional[Callable[[object, object, float], Tuple[float, float, float]]],
) -> Tuple[float, float, float]:
    if downlink_feature_fn is not None:
        return downlink_feature_fn(env, mission, float(obs_end_s))

    origin_s = (
        float(getattr(mission, "arrival_time_s", mission.earliest_time_s))
        if getattr(mission, "is_dynamic", False)
        else float(mission.earliest_time_s)
    )
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


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _clip11(value: float) -> float:
    return float(np.clip(float(value), -1.0, 1.0))
