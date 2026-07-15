"""Shared experiment helpers for the clean CVA-MAPPO branch."""

from __future__ import annotations

import copy
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from config import SatelliteConfig


def make_test_scenarios(
    mission_gen,
    n_episodes: int,
    n_routine: int,
    n_dynamic: int,
    n_insertions: int = 3,
    seed: int = 123,
) -> list:
    rng = np.random.RandomState(seed)
    scenarios = []
    for _ in range(n_episodes):
        strategy = "hotspot" if rng.rand() < 0.5 else "uniform"
        scenarios.append(
            mission_gen.generate_episode_missions(
                n_routine=n_routine,
                n_dynamic_per_insertion=n_dynamic,
                n_insertions=n_insertions,
                sampling_strategy=strategy,
            )
        )
    return scenarios


def _configure_torch_threads(torch_num_threads) -> None:
    if torch_num_threads is None:
        return
    torch.set_num_threads(max(1, int(torch_num_threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def expand_satellite_configs(base_satellites: Iterable[SatelliteConfig], n_satellites: int) -> list:
    base_satellites = list(base_satellites)
    if n_satellites <= len(base_satellites):
        return copy.deepcopy(base_satellites[:n_satellites])

    satellites = []
    base_n = len(base_satellites)
    for idx in range(n_satellites):
        base = base_satellites[idx % base_n]
        replica = idx // base_n
        n_replicas = max((n_satellites + base_n - 1) // base_n, 1)
        raan_offset = (replica * 360.0 / n_replicas) % 360.0
        phase_offset = (idx * 360.0 / n_satellites) % 360.0
        satellites.append(
            SatelliteConfig(
                name=f"{base.name}_r{replica + 1}" if replica else base.name,
                semi_major_axis_km=base.semi_major_axis_km,
                eccentricity=base.eccentricity,
                inclination_deg=base.inclination_deg,
                raan_deg=(base.raan_deg + raan_offset) % 360.0,
                arg_perigee_deg=base.arg_perigee_deg,
                mean_anomaly_deg=(base.mean_anomaly_deg + phase_offset) % 360.0,
                max_roll_deg=base.max_roll_deg,
                fov_deg=base.fov_deg,
                maneuver_speed_deg_s=base.maneuver_speed_deg_s,
            )
        )
    return satellites


def _avg_metrics(metrics_list: list[dict]) -> dict:
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    return {k: float(np.mean([m.get(k, 0.0) for m in metrics_list])) for k in keys}


def _run_git(args: list[str]) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parents[1],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _git_metadata() -> dict:
    status = _run_git(["status", "--short"])
    return {
        "commit": _run_git(["rev-parse", "--short", "HEAD"]),
        "branch": _run_git(["branch", "--show-current"]),
        "dirty": bool(status),
        "status_short": status.splitlines(),
    }


def _record_to_dict(record) -> dict:
    return {
        "mission_id": int(record.mission_id),
        "satellite_name": record.satellite_name,
        "obs_start_s": float(record.obs_start_s),
        "obs_end_s": float(record.obs_end_s),
        "reward": float(record.reward),
        "off_nadir_deg": float(record.off_nadir_deg),
        "is_dynamic": bool(record.is_dynamic),
        "earliest_time_s": float(record.earliest_time_s),
        "downlink_start_s": float(getattr(record, "downlink_start_s", record.obs_end_s)),
        "downlink_end_s": float(getattr(record, "downlink_end_s", record.obs_end_s)),
        "ground_station_id": int(getattr(record, "ground_station_id", -1)),
        "storage_start_s": float(getattr(record, "storage_start_s", record.obs_end_s)),
        "storage_release_s": float(getattr(record, "storage_release_s", record.obs_end_s)),
        "storage_release_reason": getattr(record, "storage_release_reason", "none"),
        "relay_satellite_name": getattr(record, "relay_satellite_name", ""),
        "relay_start_s": float(getattr(record, "relay_start_s", -1.0)),
        "relay_end_s": float(getattr(record, "relay_end_s", -1.0)),
    }


def _torch_state_to_numpy(state_dict: dict) -> dict:
    return {key: value.detach().cpu().numpy() for key, value in state_dict.items()}


def _numpy_state_to_torch(state_dict: dict) -> dict:
    return {key: torch.from_numpy(value.copy()) for key, value in state_dict.items()}
