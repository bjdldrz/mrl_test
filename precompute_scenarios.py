"""
Precompute train/eval mission scenarios and warm VTW disk cache.

This script intentionally keeps train and eval environments separate:

  - train_scenarios.pkl contains curriculum stages from simple to complex.
  - eval_scenarios.pkl contains held-out full-scale evaluation scenarios.
  - vtw_cache/ stores OrbitPropagator disk-cache entries for generated tasks.
"""

import argparse
import json
import logging
import os
import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import numpy as np

from config import SatelliteConfig, get_default_config
from data.mission_generator import MissionGenerator, load_acled_shapefile
from data.orbit_utils import OrbitPropagator
from utils.scenario_cache import (
    flatten_train_scenarios,
    get_eval_scenarios,
    iter_scenario_missions,
    save_pickle,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("precompute_scenarios")


def expand_satellite_configs(base_satellites, n_satellites):
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
        satellites.append(SatelliteConfig(
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
        ))
    return satellites


def parse_curriculum_stages(text: str, n_routine: int, n_dynamic: int):
    if text.strip().lower() in {"", "auto"}:
        factors = [0.25, 0.5, 0.75, 1.0]
        return [
            (max(1, int(round(n_routine * f))), max(1, int(round(n_dynamic * f))))
            for f in factors
        ]
    if text.strip().lower() in {"none", "off", "false"}:
        return [(n_routine, n_dynamic)]

    stages = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            routine_text, dynamic_text = part.split(":")
            stages.append((int(routine_text), int(dynamic_text)))
        except ValueError as exc:
            raise ValueError(
                "--curriculum_stages 格式应为 'routine:dynamic,routine:dynamic', "
                "例如 300:75,600:150,900:225,1200:300"
            ) from exc
    if not stages:
        raise ValueError("--curriculum_stages 解析为空")
    return stages


def _scenario_strategy(rng):
    return "hotspot" if rng.rand() < 0.5 else "uniform"


def generate_scenarios(
    mission_gen: MissionGenerator,
    n_scenarios: int,
    n_routine: int,
    n_dynamic: int,
    n_insertions: int,
    strategy_rng,
) -> List[Tuple[list, list]]:
    scenarios = []
    for _ in range(n_scenarios):
        scenarios.append(
            mission_gen.generate_episode_missions(
                n_routine=n_routine,
                n_dynamic_per_insertion=n_dynamic,
                n_insertions=n_insertions,
                sampling_strategy=_scenario_strategy(strategy_rng),
            )
        )
    return scenarios


def build_train_payload(
    acled_df,
    seed: int,
    total_scenarios: int,
    stages,
    n_insertions: int,
):
    mission_gen = MissionGenerator(acled_df=acled_df, seed=seed)
    strategy_rng = np.random.RandomState(seed + 17)
    n_stages = len(stages)
    base = total_scenarios // n_stages
    remainder = total_scenarios % n_stages
    payload_stages = []

    for idx, (n_routine, n_dynamic) in enumerate(stages):
        n_stage_scenarios = base + (1 if idx < remainder else 0)
        scenarios = generate_scenarios(
            mission_gen=mission_gen,
            n_scenarios=n_stage_scenarios,
            n_routine=n_routine,
            n_dynamic=n_dynamic,
            n_insertions=n_insertions,
            strategy_rng=strategy_rng,
        )
        payload_stages.append({
            "name": f"stage{idx + 1}_r{n_routine}_d{n_dynamic}",
            "stage_index": idx,
            "n_routine": n_routine,
            "n_dynamic_per_insertion": n_dynamic,
            "n_insertions": n_insertions,
            "n_scenarios": len(scenarios),
            "scenarios": scenarios,
        })
    return {
        "schema_version": 1,
        "kind": "train",
        "curriculum": True,
        "selection": "stage_by_training_progress_random_within_stage",
        "seed": seed,
        "stages": payload_stages,
    }


def build_eval_payload(
    acled_df,
    seed: int,
    n_scenarios: int,
    n_routine: int,
    n_dynamic: int,
    n_insertions: int,
):
    mission_gen = MissionGenerator(acled_df=acled_df, seed=seed)
    strategy_rng = np.random.RandomState(seed + 23)
    scenarios = generate_scenarios(
        mission_gen=mission_gen,
        n_scenarios=n_scenarios,
        n_routine=n_routine,
        n_dynamic=n_dynamic,
        n_insertions=n_insertions,
        strategy_rng=strategy_rng,
    )
    return {
        "schema_version": 1,
        "kind": "eval",
        "seed": seed,
        "n_routine": n_routine,
        "n_dynamic_per_insertion": n_dynamic,
        "n_insertions": n_insertions,
        "scenarios": scenarios,
    }


def warm_vtw_cache(satellites, scenarios, horizon_s: float, time_step_s: float):
    missions = list(iter_scenario_missions(scenarios))
    unique = {}
    for mission in missions:
        key = (round(float(mission.lat), 4), round(float(mission.lon), 4))
        unique.setdefault(key, mission)
    unique_missions = list(unique.values())
    logger.info(
        "预热 VTW: satellites=%s, missions=%s, unique_locations=%s, step=%ss",
        len(satellites),
        len(missions),
        len(unique_missions),
        time_step_s,
    )
    for sat_idx, sat_cfg in enumerate(satellites, start=1):
        logger.info("预热 VTW [%s/%s]: %s", sat_idx, len(satellites), sat_cfg.name)
        propagator = OrbitPropagator(sat_cfg)
        for mission in unique_missions:
            propagator.compute_vtw(
                mission.lat,
                mission.lon,
                horizon_seconds=horizon_s,
                time_step_s=time_step_s,
            )


def main():
    parser = argparse.ArgumentParser(description="预生成 MRL-DMS 训练/评估场景并预热 VTW")
    parser.add_argument("--acled_path", type=str, default=None)
    parser.add_argument("--n_satellites", type=int, default=12)
    parser.add_argument("--n_train_scenarios", type=int, default=200)
    parser.add_argument("--n_eval_scenarios", type=int, default=20)
    parser.add_argument("--n_routine", type=int, default=1200)
    parser.add_argument("--n_dynamic", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_seed", type=int, default=None,
                        help="评估集随机种子; 默认 seed+100000, 保证 train/eval 不同")
    parser.add_argument("--curriculum_stages", type=str, default="auto",
                        help="训练 curriculum, 如 300:75,600:150,900:225,1200:300; "
                             "auto 表示 25/50/75/100%, none 表示只用 full scale")
    parser.add_argument("--vtw_time_step_s", type=float, default=60.0)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--no_warm_vtw", action="store_true",
                        help="只保存场景, 不主动预热 VTW 磁盘缓存")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vtw_cache_dir = out_dir / "vtw_cache"
    os.environ["MRL_DMS_VTW_CACHE_DIR"] = str(vtw_cache_dir)

    cfg = get_default_config()
    cfg.satellites = expand_satellite_configs(cfg.satellites, args.n_satellites)
    if args.vtw_time_step_s is not None:
        cfg.train.vtw_time_step_s = args.vtw_time_step_s
    horizon_s = float(cfg.mission.schedule_horizon_hours) * 3600.0

    eval_seed = args.eval_seed if args.eval_seed is not None else args.seed + 100000
    if eval_seed == args.seed:
        raise ValueError("eval_seed 不应等于 seed, 否则 train/eval 可能发生场景泄漏")

    acled_df = load_acled_shapefile(args.acled_path) if args.acled_path else None
    stages = parse_curriculum_stages(args.curriculum_stages, args.n_routine, args.n_dynamic)
    logger.info("训练 curriculum stages: %s", stages)

    train_payload = build_train_payload(
        acled_df=acled_df,
        seed=args.seed,
        total_scenarios=args.n_train_scenarios,
        stages=stages,
        n_insertions=cfg.mission.dynamic_insertions_per_day,
    )
    eval_payload = build_eval_payload(
        acled_df=acled_df,
        seed=eval_seed,
        n_scenarios=args.n_eval_scenarios,
        n_routine=args.n_routine,
        n_dynamic=args.n_dynamic,
        n_insertions=cfg.mission.dynamic_insertions_per_day,
    )

    save_pickle(out_dir / "train_scenarios.pkl", train_payload)
    save_pickle(out_dir / "eval_scenarios.pkl", eval_payload)

    train_flat = flatten_train_scenarios(train_payload)
    eval_scenarios = get_eval_scenarios(eval_payload)
    if not args.no_warm_vtw:
        warm_vtw_cache(
            satellites=cfg.satellites,
            scenarios=[*train_flat, *eval_scenarios],
            horizon_s=horizon_s,
            time_step_s=cfg.train.vtw_time_step_s,
        )

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "train_seed": args.seed,
        "eval_seed": eval_seed,
        "train_eval_separated": True,
        "train_scenarios": len(train_flat),
        "eval_scenarios": len(eval_scenarios),
        "curriculum_stages": [
            {
                "name": stage["name"],
                "n_routine": stage["n_routine"],
                "n_dynamic_per_insertion": stage["n_dynamic_per_insertion"],
                "n_scenarios": stage["n_scenarios"],
            }
            for stage in train_payload["stages"]
        ],
        "vtw_cache_dir": str(vtw_cache_dir),
        "outputs": {
            "train_scenarios": str(out_dir / "train_scenarios.pkl"),
            "eval_scenarios": str(out_dir / "eval_scenarios.pkl"),
            "manifest": str(out_dir / "manifest.json"),
        },
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info("场景缓存完成: %s", out_dir)


if __name__ == "__main__":
    main()
