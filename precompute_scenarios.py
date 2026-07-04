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
import csv
from datetime import datetime, timezone
from multiprocessing import get_all_start_methods, get_context
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

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

_VTW_WORKER_TARGETS = None
_VTW_WORKER_HORIZON_S = 86400.0
_VTW_WORKER_TIME_STEP_S = 60.0
_VTW_WORKER_CACHE_DIR = ""


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
    desc: str = "generate scenarios",
    show_progress: bool = True,
) -> List[Tuple[list, list]]:
    scenarios = []
    iterator = tqdm(
        range(n_scenarios),
        desc=desc,
        unit="scenario",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for _ in iterator:
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
    show_progress: bool = True,
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
            desc=f"train stage {idx + 1}/{n_stages} r{n_routine} d{n_dynamic}",
            show_progress=show_progress,
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
    show_progress: bool = True,
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
        desc=f"eval full r{n_routine} d{n_dynamic}",
        show_progress=show_progress,
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


def _init_warm_vtw_worker(targets, horizon_s, time_step_s, cache_dir):
    global _VTW_WORKER_TARGETS
    global _VTW_WORKER_HORIZON_S
    global _VTW_WORKER_TIME_STEP_S
    global _VTW_WORKER_CACHE_DIR
    _VTW_WORKER_TARGETS = targets
    _VTW_WORKER_HORIZON_S = horizon_s
    _VTW_WORKER_TIME_STEP_S = time_step_s
    _VTW_WORKER_CACHE_DIR = cache_dir
    if cache_dir:
        os.environ["MRL_DMS_VTW_CACHE_DIR"] = cache_dir


def _warm_vtw_satellite_worker(args):
    idx, total, sat_cfg = args
    targets = _VTW_WORKER_TARGETS or []
    if _VTW_WORKER_CACHE_DIR:
        os.environ["MRL_DMS_VTW_CACHE_DIR"] = _VTW_WORKER_CACHE_DIR

    worker_logger = logging.getLogger("precompute_scenarios")
    worker_logger.info("预热 VTW [%s/%s]: %s", idx, total, sat_cfg.name)
    propagator = OrbitPropagator(sat_cfg)
    for lat, lon in targets:
        propagator.compute_vtw(
            lat,
            lon,
            horizon_seconds=_VTW_WORKER_HORIZON_S,
            time_step_s=_VTW_WORKER_TIME_STEP_S,
        )
    return {
        "idx": idx,
        "satellite": sat_cfg.name,
        "n_targets": len(targets),
    }


def warm_vtw_cache(
    satellites,
    scenarios,
    horizon_s: float,
    time_step_s: float,
    workers: int = 1,
    cache_dir: str = "",
    show_progress: bool = True,
):
    missions = list(iter_scenario_missions(scenarios))
    unique = {}
    for mission in missions:
        key = (round(float(mission.lat), 4), round(float(mission.lon), 4))
        unique.setdefault(key, (float(mission.lat), float(mission.lon)))
    targets = list(unique.values())
    n_workers = max(1, min(int(workers or 1), len(satellites)))
    logger.info(
        "预热 VTW: satellites=%s, missions=%s, unique_locations=%s, step=%ss, workers=%s",
        len(satellites),
        len(missions),
        len(targets),
        time_step_s,
        n_workers,
    )
    task_args = [
        (sat_idx, len(satellites), sat_cfg)
        for sat_idx, sat_cfg in enumerate(satellites, start=1)
    ]
    if n_workers <= 1:
        _init_warm_vtw_worker(targets, horizon_s, time_step_s, cache_dir)
        results = [
            _warm_vtw_satellite_worker(args)
            for args in tqdm(
                task_args,
                desc="warm VTW satellites",
                unit="sat",
                dynamic_ncols=True,
                disable=not show_progress,
            )
        ]
    else:
        start_method = "fork" if "fork" in get_all_start_methods() else "spawn"
        with get_context(start_method).Pool(
            processes=n_workers,
            initializer=_init_warm_vtw_worker,
            initargs=(targets, horizon_s, time_step_s, cache_dir),
        ) as pool:
            results = list(tqdm(
                pool.imap_unordered(_warm_vtw_satellite_worker, task_args),
                total=len(task_args),
                desc="warm VTW satellites",
                unit="sat",
                dynamic_ncols=True,
                disable=not show_progress,
            ))
    logger.info("VTW 预热完成: %s satellites", len(results))


def _scenario_points(scenario):
    routine, dynamic_schedule = scenario
    rows = []
    for mission in routine:
        rows.append({
            "mission_id": int(mission.id),
            "lat": float(mission.lat),
            "lon": float(mission.lon),
            "priority": float(mission.priority),
            "is_dynamic": False,
            "arrival_time_s": 0.0,
            "stage": "routine",
        })
    for arrival_time, missions in dynamic_schedule:
        for mission in missions:
            rows.append({
                "mission_id": int(mission.id),
                "lat": float(mission.lat),
                "lon": float(mission.lon),
                "priority": float(mission.priority),
                "is_dynamic": True,
                "arrival_time_s": float(arrival_time),
                "stage": f"dynamic_{int(round(arrival_time))}",
            })
    return rows


def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_scenario_map(scenario, out_path: Path, title: str, dpi: int = 160):
    plt = _setup_matplotlib()
    rows = _scenario_points(scenario)
    routine = [row for row in rows if not row["is_dynamic"]]
    dynamic = [row for row in rows if row["is_dynamic"]]

    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_facecolor("#F7FAFC")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-70, 75)
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-60, 76, 30))
    ax.grid(True, color="#D8DEE9", linewidth=0.6, alpha=0.8)
    ax.axhline(0, color="#A0AEC0", linewidth=0.8)
    ax.axvline(0, color="#A0AEC0", linewidth=0.8)

    if routine:
        ax.scatter(
            [row["lon"] for row in routine],
            [row["lat"] for row in routine],
            s=8,
            c="#2B6CB0",
            alpha=0.34,
            linewidths=0,
            label=f"Routine ({len(routine)})",
        )
    if dynamic:
        arrivals = sorted({row["arrival_time_s"] for row in dynamic})
        colors = ["#E53E3E", "#DD6B20", "#C53030", "#9B2C2C", "#B7791F"]
        for idx, arrival in enumerate(arrivals):
            points = [row for row in dynamic if row["arrival_time_s"] == arrival]
            ax.scatter(
                [row["lon"] for row in points],
                [row["lat"] for row in points],
                s=13,
                c=colors[idx % len(colors)],
                alpha=0.62,
                linewidths=0,
                label=f"Dynamic t={arrival / 3600:.1f}h ({len(points)})",
            )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_scenario_points_csv(scenario, out_path: Path):
    rows = _scenario_points(scenario)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mission_id",
                "lat",
                "lon",
                "priority",
                "is_dynamic",
                "arrival_time_s",
                "stage",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_scenario_maps(
    train_payload,
    eval_payload,
    out_dir: Path,
    max_scenarios: int,
    dpi: int,
    show_progress: bool = True,
):
    maps_dir = out_dir / "maps"
    n_saved = 0
    limit = None if max_scenarios <= 0 else int(max_scenarios)

    for stage in tqdm(
        train_payload["stages"],
        desc="plot train map stages",
        unit="stage",
        dynamic_ncols=True,
        disable=not show_progress,
    ):
        stage_name = stage["name"]
        scenarios = stage["scenarios"]
        if limit is not None:
            scenarios = scenarios[:limit]
        for idx, scenario in enumerate(tqdm(
            scenarios,
            desc=f"plot {stage_name}",
            unit="map",
            dynamic_ncols=True,
            leave=False,
            disable=not show_progress,
        )):
            stem = f"{idx + 1:04d}_{stage_name}"
            plot_scenario_map(
                scenario,
                maps_dir / "train" / stage_name / f"{stem}.png",
                title=f"Train Scenario {idx + 1} | {stage_name}",
                dpi=dpi,
            )
            write_scenario_points_csv(
                scenario,
                maps_dir / "train" / stage_name / f"{stem}.csv",
            )
            n_saved += 1

    eval_scenarios = get_eval_scenarios(eval_payload)
    if limit is not None:
        eval_scenarios = eval_scenarios[:limit]
    for idx, scenario in enumerate(tqdm(
        eval_scenarios,
        desc="plot eval maps",
        unit="map",
        dynamic_ncols=True,
        disable=not show_progress,
    )):
        stem = f"{idx + 1:04d}_eval_full"
        plot_scenario_map(
            scenario,
            maps_dir / "eval" / f"{stem}.png",
            title=f"Eval Scenario {idx + 1} | full stress",
            dpi=dpi,
        )
        write_scenario_points_csv(
            scenario,
            maps_dir / "eval" / f"{stem}.csv",
        )
        n_saved += 1
    logger.info("场景分布地图完成: %s scenario maps under %s", n_saved, maps_dir)
    return maps_dir, n_saved


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
    parser.add_argument("--vtw_workers", type=int, default=0,
                        help="VTW 预热并行进程数; 0 表示自动 min(n_satellites, cpu_count)")
    parser.add_argument("--no_plot_maps", action="store_true",
                        help="不为每个 train/eval scenario 输出任务分布地图")
    parser.add_argument("--map_max_scenarios", type=int, default=0,
                        help="每个 train stage / eval 最多绘制多少个场景; 0 表示全部绘制")
    parser.add_argument("--map_dpi", type=int, default=160,
                        help="任务分布地图 PNG 分辨率")
    parser.add_argument("--no_progress", action="store_true",
                        help="关闭 tqdm 进度条")
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
        show_progress=not args.no_progress,
    )
    eval_payload = build_eval_payload(
        acled_df=acled_df,
        seed=eval_seed,
        n_scenarios=args.n_eval_scenarios,
        n_routine=args.n_routine,
        n_dynamic=args.n_dynamic,
        n_insertions=cfg.mission.dynamic_insertions_per_day,
        show_progress=not args.no_progress,
    )

    save_pickle(out_dir / "train_scenarios.pkl", train_payload)
    save_pickle(out_dir / "eval_scenarios.pkl", eval_payload)

    train_flat = flatten_train_scenarios(train_payload)
    eval_scenarios = get_eval_scenarios(eval_payload)
    maps_dir = None
    n_maps = 0
    effective_vtw_workers = 0
    if not args.no_plot_maps:
        maps_dir, n_maps = save_scenario_maps(
            train_payload=train_payload,
            eval_payload=eval_payload,
            out_dir=out_dir,
            max_scenarios=args.map_max_scenarios,
            dpi=args.map_dpi,
            show_progress=not args.no_progress,
        )
    if not args.no_warm_vtw:
        vtw_workers = args.vtw_workers
        if vtw_workers <= 0:
            vtw_workers = min(len(cfg.satellites), os.cpu_count() or 1)
        effective_vtw_workers = vtw_workers
        warm_vtw_cache(
            satellites=cfg.satellites,
            scenarios=[*train_flat, *eval_scenarios],
            horizon_s=horizon_s,
            time_step_s=cfg.train.vtw_time_step_s,
            workers=vtw_workers,
            cache_dir=str(vtw_cache_dir),
            show_progress=not args.no_progress,
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
        "vtw_workers": args.vtw_workers,
        "effective_vtw_workers": effective_vtw_workers,
        "maps_dir": str(maps_dir) if maps_dir is not None else "",
        "n_maps": n_maps,
        "outputs": {
            "train_scenarios": str(out_dir / "train_scenarios.pkl"),
            "eval_scenarios": str(out_dir / "eval_scenarios.pkl"),
            "manifest": str(out_dir / "manifest.json"),
            "maps_dir": str(maps_dir) if maps_dir is not None else "",
        },
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info("场景缓存完成: %s", out_dir)


if __name__ == "__main__":
    main()
