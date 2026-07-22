#!/usr/bin/env python3
"""Run paper baseline comparisons that use the fixed-slot MAPPO actor."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

from run_stage_ablation_suite import SUMMARY_COLUMNS, markdown_table, safe_name, timestamp


REPO_ROOT = Path(__file__).resolve().parents[1]


def kv(flag: str, value: Any) -> list[str]:
    return [flag, str(value)]


def fixed_slot_stage1_args() -> list[str]:
    return [
        *kv("--routine_slots", 64),
        *kv("--dynamic_slots", 32),
        *kv("--flex_slots", 32),
        *kv("--routine_candidate_owners", 1),
        *kv("--dynamic_candidate_owners", 6),
        *kv("--urgent_candidate_owners", 6),
        *kv("--stale_candidate_owners", 6),
        *kv("--candidate_owner_bonus", "0.06"),
        *kv("--dynamic_broadcast_window_s", 3600),
        *kv("--dynamic_takeover_margin_s", 300),
        *kv("--candidate_wait_penalty", "0.08"),
        *kv("--candidate_storage_penalty", "0.08"),
        *kv("--candidate_dynamic_urgency_bonus", "0.12"),
        *kv("--allocator_wait_penalty", "0.10"),
        *kv("--allocator_stale_rescue_bonus", "0.25"),
        *kv("--allocator_dynamic_urgency_bonus", "0.10"),
    ]


def fixed_slot_plain_args(slot_mode: str = "mixed") -> list[str]:
    return [
        *kv("--routine_slots", 64),
        *kv("--dynamic_slots", 32),
        *kv("--flex_slots", 32),
        *kv("--slot_selection_mode", slot_mode),
        *kv("--ownership_mask_mode", "hard"),
        *kv("--routine_candidate_owners", 1),
        *kv("--dynamic_candidate_owners", 1),
        *kv("--urgent_candidate_owners", 1),
        *kv("--stale_candidate_owners", 1),
        *kv("--candidate_owner_bonus", "0.00"),
        *kv("--executable_slot_reserve_ratio", "0.00"),
        *kv("--assignment_replan_trigger", "periodic"),
        *kv("--assignment_switch_penalty", "0.00"),
        *kv("--owner_switch_margin", "0.00"),
        *kv("--dynamic_broadcast_window_s", 0),
        *kv("--dynamic_takeover_margin_s", 0),
        *kv("--routine_future_dynamic_penalty", "0.00"),
        *kv("--dynamic_future_bonus", "0.00"),
        *kv("--dynamic_current_slot_bonus", "0.00"),
        *kv("--dynamic_window_wait_weight", "0.00"),
        "--no_future_task_execution",
        "--no_downlink_aware_candidate_score",
        *kv("--cva_load_penalty", "0.00"),
        *kv("--w_dynamic", "0.00"),
        *kv("--w_scarcity", "0.00"),
        *kv("--w_future_opportunity_loss", "0.00"),
        *kv("--w_load", "0.00"),
        *kv("--w_owner_stability", "0.00"),
        *kv("--candidate_wait_penalty", "0.00"),
        *kv("--candidate_storage_penalty", "0.00"),
        *kv("--candidate_dynamic_urgency_bonus", "0.00"),
        *kv("--candidate_dynamic_response_bonus", "0.00"),
        *kv("--candidate_dynamic_wait_penalty", "0.00"),
        *kv("--candidate_downlink_queue_penalty", "0.00"),
        *kv("--candidate_downlink_miss_penalty", "0.00"),
        *kv("--candidate_dynamic_delivery_bonus", "0.00"),
        *kv("--candidate_dynamic_delivery_delay_penalty", "0.00"),
        *kv("--allocator_wait_penalty", "0.00"),
        *kv("--allocator_stale_rescue_bonus", "0.00"),
        *kv("--allocator_dynamic_urgency_bonus", "0.00"),
        *kv("--allocator_dynamic_response_bonus", "0.00"),
        *kv("--allocator_dynamic_wait_penalty", "0.00"),
        *kv("--dynamic_rescue_response_bonus", "0.00"),
    ]


def fixed_slot_stage2_args() -> list[str]:
    return [
        *kv("--routine_slots", 48),
        *kv("--dynamic_slots", 48),
        *kv("--flex_slots", 32),
        *kv("--routine_candidate_owners", 1),
        *kv("--dynamic_candidate_owners", 8),
        *kv("--urgent_candidate_owners", 8),
        *kv("--stale_candidate_owners", 8),
        *kv("--candidate_owner_bonus", "0.08"),
        *kv("--assignment_replan_interval_s", 900),
        *kv("--assignment_replan_horizon_s", 21600),
        *kv("--release_before_deadline_s", 7200),
        *kv("--dynamic_broadcast_window_s", 7200),
        *kv("--dynamic_takeover_margin_s", 120),
        *kv("--candidate_wait_penalty", "0.10"),
        *kv("--candidate_storage_penalty", "0.08"),
        *kv("--candidate_dynamic_urgency_bonus", "0.16"),
        *kv("--allocator_wait_penalty", "0.14"),
        *kv("--allocator_stale_rescue_bonus", "0.35"),
        *kv("--allocator_dynamic_urgency_bonus", "0.16"),
    ]


def fixed_slot_stage4_args() -> list[str]:
    return [
        *kv("--routine_slots", 48),
        *kv("--dynamic_slots", 48),
        *kv("--flex_slots", 32),
        *kv("--routine_candidate_owners", 1),
        *kv("--dynamic_candidate_owners", 8),
        *kv("--urgent_candidate_owners", 8),
        *kv("--stale_candidate_owners", 8),
        *kv("--candidate_owner_bonus", "0.08"),
        *kv("--assignment_replan_interval_s", 900),
        *kv("--assignment_replan_horizon_s", 21600),
        *kv("--release_before_deadline_s", 7200),
        *kv("--dynamic_broadcast_window_s", 7200),
        *kv("--dynamic_takeover_margin_s", 120),
        *kv("--candidate_wait_penalty", "0.10"),
        *kv("--candidate_storage_penalty", "0.16"),
        *kv("--candidate_dynamic_urgency_bonus", "0.18"),
        *kv("--allocator_wait_penalty", "0.14"),
        *kv("--allocator_stale_rescue_bonus", "0.35"),
        *kv("--allocator_dynamic_urgency_bonus", "0.18"),
    ]


BASELINE_SPECS: list[dict[str, Any]] = [
    {
        "name": "plain_fixed_slot_mappo_v2_mixed",
        "group": "plain_fixed_slot_mappo",
        "base_stage": "plain_mixed",
        "paper_use": "Strict fixed-slot MAPPO with shared Top-K slots and no DAS/CVA heuristic priors.",
        "args": fixed_slot_plain_args("mixed"),
    },
    {
        "name": "plain_fixed_slot_mappo_v2_typed",
        "group": "plain_fixed_slot_mappo",
        "base_stage": "plain_typed",
        "paper_use": "Fixed-slot MAPPO with typed slot quotas only; dynamic/downlink/future heuristic priors disabled.",
        "args": fixed_slot_plain_args("typed"),
    },
    {
        "name": "fixed_slot_mappo_v2_stage1",
        "group": "fixed_slot_mappo",
        "base_stage": "v2_stage1",
        "paper_use": "Fixed-slot MAPPO with v2 heuristic candidate slots.",
        "args": fixed_slot_stage1_args(),
    },
    {
        "name": "fixed_slot_mappo_v2_stage2",
        "group": "fixed_slot_mappo",
        "base_stage": "stage2_like",
        "paper_use": "Fixed-slot MAPPO under the Stage-2 candidate/owner configuration.",
        "args": fixed_slot_stage2_args(),
    },
    {
        "name": "fixed_slot_mappo_v2_stage4_heuristic",
        "group": "fixed_slot_mappo",
        "base_stage": "stage4_like",
        "paper_use": "Fixed-slot MAPPO with Stage-4-like heuristic storage/downlink pressure.",
        "args": fixed_slot_stage4_args(),
    },
]


def base_args(args: argparse.Namespace, suite_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "cva_mappo_v2.run_experiment",
        *kv("--acled_path", args.acled_path),
        *kv("--scenario_cache_dir", args.scenario_cache_dir),
        *kv("--vtw_cache_dir", args.vtw_cache_dir),
        *kv("--n_satellites", args.n_satellites),
        *kv("--train_iters", args.train_iters),
        *kv("--eval_episodes", args.eval_episodes),
        *kv("--n_routine", args.n_routine),
        *kv("--n_dynamic", args.n_dynamic),
        *kv("--n_ground_stations", args.n_ground_stations),
        *kv("--downlink_time_s", args.downlink_time_s),
        *kv("--satellite_storage_capacity", args.satellite_storage_capacity),
        "--enable_inter_satellite_transfer",
        *kv("--inter_satellite_transfer_time_s", args.inter_satellite_transfer_time_s),
        *kv("--slot_selection_mode", "typed"),
        *kv("--ownership_mask_mode", "soft"),
        *kv("--executable_slot_reserve_ratio", args.executable_slot_reserve_ratio),
        *kv("--future_task_max_wait_s", args.future_task_max_wait_s),
        *kv("--future_routine_max_wait_s", args.future_routine_max_wait_s),
        *kv("--routine_future_dynamic_guard_s", args.routine_future_dynamic_guard_s),
        *kv("--routine_future_dynamic_penalty", args.routine_future_dynamic_penalty),
        *kv("--dynamic_future_bonus", args.dynamic_future_bonus),
        *kv("--dynamic_current_slot_bonus", args.dynamic_current_slot_bonus),
        *kv("--dynamic_window_wait_weight", args.dynamic_window_wait_weight),
        *(["--no_downlink_aware_candidate_score"] if args.no_downlink_aware_candidate_score else []),
        *kv("--downlink_queue_target_s", args.downlink_queue_target_s),
        *kv("--candidate_downlink_queue_penalty", args.candidate_downlink_queue_penalty),
        *kv("--candidate_downlink_miss_penalty", args.candidate_downlink_miss_penalty),
        *kv("--candidate_dynamic_delivery_bonus", args.candidate_dynamic_delivery_bonus),
        *kv("--candidate_dynamic_delivery_delay_penalty", args.candidate_dynamic_delivery_delay_penalty),
        *(["--dynamic_downlink_priority"] if args.dynamic_downlink_priority else ["--no_dynamic_downlink_priority"]),
        *kv("--candidate_dynamic_response_bonus", args.candidate_dynamic_response_bonus),
        *kv("--candidate_dynamic_wait_penalty", args.candidate_dynamic_wait_penalty),
        *kv("--dynamic_response_target_s", args.dynamic_response_target_s),
        *kv("--allocator_dynamic_response_bonus", args.allocator_dynamic_response_bonus),
        *kv("--allocator_dynamic_wait_penalty", args.allocator_dynamic_wait_penalty),
        *kv("--dynamic_rescue_response_bonus", args.dynamic_rescue_response_bonus),
        *kv("--rollout_steps", args.rollout_steps),
        *kv("--train_env_workers", args.train_env_workers),
        "--split_rollout_steps_across_workers",
        *kv("--ppo_epochs", args.ppo_epochs),
        *kv("--ppo_batch_size", args.ppo_batch_size),
        *kv("--eval_max_steps", args.eval_max_steps),
        *kv("--eval_device", args.eval_device),
        *kv("--eval_workers", args.eval_workers),
        *(["--eval_use_repair"] if args.eval_use_repair else []),
        *kv("--torch_num_threads", args.torch_num_threads),
        *kv("--vtw_time_step_s", args.vtw_time_step_s),
        *kv("--out_dir", suite_dir),
        *kv("--device", args.device),
        "--no_viz",
    ]


def command_for_spec(args: argparse.Namespace, suite_dir: Path, spec: dict[str, Any]) -> list[str]:
    return [
        *base_args(args, suite_dir),
        *kv("--run_name", spec["name"]),
        *spec["args"],
        *(["--no_progress"] if args.no_progress else []),
    ]


def stream_command(cmd: list[str], log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + shlex.join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        return int(proc.wait())


def latest_result_dir(suite_dir: Path, run_name: str) -> Optional[Path]:
    matches = sorted(
        [p for p in suite_dir.glob(f"{safe_name(run_name)}_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in matches:
        if (path / "comparison_results.json").exists():
            return path
    return matches[0] if matches else None


def read_metrics(result_dir: Optional[Path]) -> dict[str, Any]:
    if result_dir is None:
        return {}
    path = result_dir / "comparison_results.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data:
        return {}
    first_key = next(iter(data))
    metrics = data[first_key]
    return metrics if isinstance(metrics, dict) else {}


def write_summary(rows: list[dict[str, Any]], suite_dir: Path) -> None:
    keys = list(SUMMARY_COLUMNS)
    extra_keys = sorted({key for row in rows for key in row.keys()} - set(keys))
    keys.extend(extra_keys)
    with (suite_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    (suite_dir / "summary.md").write_text(markdown_table(rows, SUMMARY_COLUMNS), encoding="utf-8")


def write_plan(rows: list[dict[str, Any]], suite_dir: Path, args: argparse.Namespace) -> None:
    payload = {
        "schema_version": 1,
        "suite_name": args.suite_name,
        "method_family": "fixed_slot_mappo",
        "baselines": [
            {
                "name": row["experiment"],
                "paper_use": row.get("paper_use", ""),
                "result_dir": row.get("result_dir", ""),
                "status": row.get("status", ""),
            }
            for row in rows
        ],
    }
    (suite_dir / "paper_baseline_plan.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Paper Baseline Plan",
        "",
        "| baseline | paper use | status |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        lines.append(f"| `{row['experiment']}` | {row.get('paper_use', '')} | {row.get('status', '')} |")
    lines.extend(["", "## Result Files", "", "- `summary.csv`", "- `summary.md`", "- `logs/`"])
    (suite_dir / "paper_baseline_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_suite(args: argparse.Namespace) -> int:
    suite_name = safe_name(args.suite_name or f"fixed_slot_baselines_{timestamp()}")
    args.suite_name = suite_name
    suite_dir = Path(args.out_dir).resolve() / suite_name
    suite_dir.mkdir(parents=True, exist_ok=False)
    print(f"Baseline suite directory: {suite_dir}")

    specs = list(BASELINE_SPECS)
    if args.only:
        allowed = set(args.only)
        specs = [spec for spec in specs if spec["name"] in allowed]

    env = os.environ.copy()
    if args.vtw_cache_dir:
        env["MRL_DMS_VTW_CACHE_DIR"] = args.vtw_cache_dir

    rows: list[dict[str, Any]] = []
    for idx, spec in enumerate(specs, start=1):
        print(f"\n[{idx}/{len(specs)}] Running {spec['name']}")
        cmd = command_for_spec(args, suite_dir, spec)
        if args.dry_run:
            print(shlex.join(cmd))
            continue
        log_path = suite_dir / "logs" / f"{spec['name']}.log"
        started = time.time()
        returncode = stream_command(cmd, log_path, env)
        elapsed = time.time() - started
        result_dir = latest_result_dir(suite_dir, spec["name"])
        metrics = read_metrics(result_dir)
        row = {
            "experiment": spec["name"],
            "group": spec["group"],
            "base_stage": spec["base_stage"],
            "paper_use": spec["paper_use"],
            "status": "ok" if returncode == 0 else "failed",
            "returncode": returncode,
            "result_dir": str(result_dir) if result_dir else "",
            "elapsed_s": round(elapsed, 3),
            **{
                key: metrics.get(key, "")
                for key in SUMMARY_COLUMNS
                if key not in {"experiment", "group", "base_stage", "status", "returncode", "result_dir"}
            },
        }
        rows.append(row)
        write_summary(rows, suite_dir)
        write_plan(rows, suite_dir, args)
        if returncode != 0 and not args.continue_on_error:
            print(f"Run failed: {spec['name']}. See {log_path}")
            return returncode
    if args.dry_run:
        return 0
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed-slot MAPPO paper baselines.")
    parser.add_argument("--acled_path", default="./DynamicMission/DynamicMission.shp")
    parser.add_argument("--scenario_cache_dir", default="runs/scenario_cache/das_cva_stress_seed42")
    parser.add_argument("--vtw_cache_dir", default="runs/scenario_cache/das_cva_stress_seed42/vtw_cache")
    parser.add_argument("--out_dir", default="runs/paper_baseline_suites")
    parser.add_argument("--suite_name", default="")
    parser.add_argument("--n_satellites", type=int, default=6)
    parser.add_argument("--train_iters", type=int, default=50)
    parser.add_argument("--eval_episodes", "--val_episodes", dest="eval_episodes", type=int, default=10)
    parser.add_argument("--n_routine", type=int, default=600)
    parser.add_argument("--n_dynamic", type=int, default=150)
    parser.add_argument("--n_ground_stations", type=int, default=4)
    parser.add_argument("--downlink_time_s", type=float, default=30.0)
    parser.add_argument("--satellite_storage_capacity", type=int, default=30)
    parser.add_argument("--inter_satellite_transfer_time_s", type=float, default=300.0)
    parser.add_argument("--rollout_steps", type=int, default=512)
    parser.add_argument("--train_env_workers", type=int, default=16)
    parser.add_argument("--executable_slot_reserve_ratio", type=float, default=0.5)
    parser.add_argument("--future_task_max_wait_s", type=float, default=600.0)
    parser.add_argument("--future_routine_max_wait_s", type=float, default=180.0)
    parser.add_argument("--routine_future_dynamic_guard_s", type=float, default=1800.0)
    parser.add_argument("--routine_future_dynamic_penalty", type=float, default=0.35)
    parser.add_argument("--dynamic_future_bonus", type=float, default=0.25)
    parser.add_argument("--dynamic_current_slot_bonus", type=float, default=0.65)
    parser.add_argument("--dynamic_window_wait_weight", type=float, default=0.75)
    parser.add_argument("--no_downlink_aware_candidate_score", action="store_true")
    parser.add_argument("--downlink_queue_target_s", type=float, default=3600.0)
    parser.add_argument("--candidate_downlink_queue_penalty", type=float, default=0.10)
    parser.add_argument("--candidate_downlink_miss_penalty", type=float, default=0.20)
    parser.add_argument("--candidate_dynamic_delivery_bonus", type=float, default=0.24)
    parser.add_argument("--candidate_dynamic_delivery_delay_penalty", type=float, default=0.20)
    parser.add_argument("--dynamic_downlink_priority", dest="dynamic_downlink_priority", action="store_true")
    parser.add_argument("--no_dynamic_downlink_priority", dest="dynamic_downlink_priority", action="store_false")
    parser.set_defaults(dynamic_downlink_priority=False)
    parser.add_argument("--candidate_dynamic_response_bonus", type=float, default=0.24)
    parser.add_argument("--candidate_dynamic_wait_penalty", type=float, default=0.20)
    parser.add_argument("--dynamic_response_target_s", type=float, default=3600.0)
    parser.add_argument("--allocator_dynamic_response_bonus", type=float, default=0.24)
    parser.add_argument("--allocator_dynamic_wait_penalty", type=float, default=0.20)
    parser.add_argument("--dynamic_rescue_response_bonus", type=float, default=1.0)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--ppo_batch_size", type=int, default=512)
    parser.add_argument("--eval_max_steps", type=int, default=8000)
    parser.add_argument("--eval_workers", type=int, default=10)
    parser.add_argument("--torch_num_threads", type=int, default=1)
    parser.add_argument("--vtw_time_step_s", type=float, default=60.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval_device", default="cpu")
    parser.add_argument("--eval_use_repair", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--only", nargs="*", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_suite(parse_args()))
