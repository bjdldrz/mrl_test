#!/usr/bin/env python3
"""Run DAS staged experiments and ablations, then summarize results.

The suite runs Stage 1-4 sequentially with a shared stress configuration:

  train_iters=50, eval_episodes=10, eval_workers=24, train_env_workers=16,
  training device=cuda:0, eval_device=cpu by default.

Most ablations are applied on top of the Stage 4 configuration so the table
compares one removed/changed component at a time against the strongest staged
setting. Targeted Stage 2 ablations are included for changes that should be
validated before hybrid scorer/storage-pressure effects enter the comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]


SUMMARY_COLUMNS = [
    "experiment",
    "group",
    "base_stage",
    "status",
    "returncode",
    "result_dir",
    "total_reward",
    "observation_success_rate_raw",
    "dynamic_completion_rate_raw",
    "routine_completion_rate_raw",
    "observation_success_rate",
    "dynamic_completion_rate",
    "routine_completion_rate",
    "feasible_ratio",
    "dynamic_feasible_ratio",
    "eval_valid_decision_rate",
    "eval_idle_action_rate",
    "eval_idle_when_valid_rate",
    "eval_wall_time_s",
    "eval_total_steps",
    "eval_steps_per_wall_s",
    "eval_setup_time_s",
    "eval_reset_time_s",
    "eval_valid_mask_time_s",
    "eval_feature_build_time_s",
    "eval_actor_forward_time_s",
    "eval_counter_time_s",
    "eval_env_step_time_s",
    "eval_finalize_time_s",
    "eval_timed_to_wall_ratio",
    "eval_feature_build_share",
    "eval_actor_forward_share",
    "eval_env_step_share",
    "eval_actor_batches",
    "eval_feature_batches",
    "eval_env_step_calls",
    "n_multi_steps",
    "n_fast_idle_resolve_steps",
    "fast_idle_resolve_rate",
    "n_low_level_fast_steps",
    "avg_valid_slots",
    "avg_current_valid_slots",
    "avg_future_valid_slots",
    "avg_filled_slots",
    "avg_filled_invalid_slots",
    "n_future_task_executions",
    "n_future_dynamic_task_executions",
    "n_future_routine_task_executions",
    "avg_future_task_wait_s",
    "n_candidate_limited_idle_advances",
    "avg_candidate_limited_idle_advance_s",
    "n_dynamic_candidate_idle_advances",
    "avg_dynamic_candidate_idle_advance_s",
    "dynamic_current_slot_exposure_rate",
    "dynamic_future_slot_exposure_rate",
    "avg_dynamic_current_slot_candidates",
    "avg_dynamic_current_slots_selected",
    "avg_dynamic_future_slot_candidates",
    "avg_dynamic_future_slots_selected",
    "n_downlink_priority_replans",
    "n_downlink_priority_dynamic_records",
    "avg_dynamic_downlink_replan_gain_s",
    "stale_owner_rate",
    "owner_churn_rate",
    "load_balance_cv",
    "avg_dynamic_response_s",
    "n_storage_expired_drops",
    "avg_downlink_queue_s",
    "n_relay_storage_images",
]


def safe_name(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", str(text or "run")).strip("._-")
    return (text or "run")[:max_len]


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def kv(flag: str, value: Any) -> list[str]:
    return [flag, str(value)]


def base_args(args: argparse.Namespace, suite_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "das_cva_mappo.run_experiment",
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
        *kv("--executable_slot_reserve_ratio", args.executable_slot_reserve_ratio),
        *kv("--future_task_max_wait_s", args.future_task_max_wait_s),
        *kv("--future_routine_max_wait_s", args.future_routine_max_wait_s),
        *kv("--routine_future_dynamic_guard_s", args.routine_future_dynamic_guard_s),
        *kv("--routine_future_dynamic_penalty", args.routine_future_dynamic_penalty),
        *kv("--dynamic_future_bonus", args.dynamic_future_bonus),
        *kv("--dynamic_current_slot_bonus", args.dynamic_current_slot_bonus),
        *kv("--dynamic_window_wait_weight", args.dynamic_window_wait_weight),
        *(["--no_dynamic_downlink_priority"] if args.no_dynamic_downlink_priority else []),
        *kv("--candidate_dynamic_response_bonus", args.candidate_dynamic_response_bonus),
        *kv("--candidate_dynamic_wait_penalty", args.candidate_dynamic_wait_penalty),
        *kv("--dynamic_response_target_s", args.dynamic_response_target_s),
        *kv("--allocator_dynamic_response_bonus", args.allocator_dynamic_response_bonus),
        *kv("--allocator_dynamic_wait_penalty", args.allocator_dynamic_wait_penalty),
        *kv("--dynamic_rescue_response_bonus", args.dynamic_rescue_response_bonus),
        *kv("--ownership_mask_mode", "soft"),
        *kv("--matcher", "set_transformer"),
        *kv("--idle_aux_coeff", "0.05"),
        *kv("--action_feature_mode", "full"),
        *kv("--candidate_adapter_mode", "v2_compat"),
        *kv("--candidate_dropout_prob", "0.05"),
        *kv("--rollout_steps", args.rollout_steps),
        *kv("--train_env_workers", args.train_env_workers),
        "--split_rollout_steps_across_workers",
        *kv("--ppo_epochs", args.ppo_epochs),
        *kv("--ppo_batch_size", args.ppo_batch_size),
        *kv("--eval_max_steps", args.eval_max_steps),
        *kv("--eval_device", args.eval_device),
        *kv("--eval_workers", args.eval_workers),
        *(["--eval_use_repair"] if args.eval_use_repair else []),
        *(["--eval_profile"] if args.eval_profile else []),
        *kv("--torch_num_threads", args.torch_num_threads),
        *kv("--vtw_time_step_s", args.vtw_time_step_s),
        *kv("--out_dir", suite_dir),
        *kv("--device", args.device),
    ]


def stage2_common() -> list[str]:
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
        *kv("--assignment_replan_trigger", "periodic,dynamic,stale_owner,deadline"),
        *kv("--release_before_deadline_s", 7200),
        *kv("--dynamic_broadcast_window_s", 7200),
        *kv("--dynamic_takeover_margin_s", 120),
        *kv("--candidate_wait_penalty", "0.10"),
        *kv("--candidate_storage_penalty", "0.08"),
        *kv("--candidate_dynamic_urgency_bonus", "0.16"),
        *kv("--allocator_wait_penalty", "0.14"),
        *kv("--allocator_stale_rescue_bonus", "0.35"),
        *kv("--allocator_dynamic_urgency_bonus", "0.16"),
        *kv("--candidate_scorer_mode", "v2_heuristic"),
    ]


def stage2_dynamic_priority_common() -> list[str]:
    return [
        *stage2_common(),
        *kv("--routine_slots", 32),
        *kv("--dynamic_slots", 64),
        *kv("--flex_slots", 32),
        *kv("--dynamic_candidate_owners", 12),
        *kv("--urgent_candidate_owners", 12),
        *kv("--stale_candidate_owners", 12),
        *kv("--release_before_deadline_s", 10800),
        *kv("--dynamic_broadcast_window_s", 10800),
        *kv("--candidate_wait_penalty", "0.14"),
        *kv("--candidate_dynamic_urgency_bonus", "0.30"),
        *kv("--dynamic_current_slot_bonus", "0.85"),
        *kv("--dynamic_window_wait_weight", "1.00"),
        *kv("--allocator_wait_penalty", "0.18"),
        *kv("--allocator_dynamic_urgency_bonus", "0.30"),
    ]


def stage_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "stage1_slot_diagnosis",
            "group": "stage",
            "base_stage": "stage1",
            "args": [
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
                *kv("--assignment_replan_trigger", "periodic,dynamic,stale_owner,deadline"),
                *kv("--candidate_scorer_mode", "v2_heuristic"),
            ],
        },
        {
            "name": "stage2_candidate_owner_repair",
            "group": "stage",
            "base_stage": "stage2",
            "args": [
                *stage2_common(),
            ],
        },
        {
            "name": "stage2_dynamic_priority_recovery",
            "group": "stage",
            "base_stage": "stage2_dynamic",
            "args": [
                *stage2_dynamic_priority_common(),
            ],
        },
        {
            "name": "stage3_dynamic_hybrid",
            "group": "stage",
            "base_stage": "stage3",
            "args": [
                *stage4_common(candidate_storage_penalty="0.08"),
                *hybrid_scorer_args(candidate_aux_load_penalty="0.10"),
            ],
        },
        {
            "name": "stage4_storage_pressure",
            "group": "stage",
            "base_stage": "stage4",
            "args": [
                *stage4_common(candidate_storage_penalty="0.16"),
                *hybrid_scorer_args(candidate_aux_load_penalty="0.20"),
            ],
        },
    ]


def stage4_common(candidate_storage_penalty: str) -> list[str]:
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
        *kv("--assignment_replan_trigger", "periodic,dynamic,stale_owner,deadline"),
        *kv("--release_before_deadline_s", 7200),
        *kv("--dynamic_broadcast_window_s", 7200),
        *kv("--dynamic_takeover_margin_s", 120),
        *kv("--candidate_wait_penalty", "0.10"),
        *kv("--candidate_storage_penalty", candidate_storage_penalty),
        *kv("--candidate_dynamic_urgency_bonus", "0.18"),
        *kv("--allocator_wait_penalty", "0.14"),
        *kv("--allocator_stale_rescue_bonus", "0.35"),
        *kv("--allocator_dynamic_urgency_bonus", "0.18"),
    ]


def hybrid_scorer_args(candidate_aux_load_penalty: str) -> list[str]:
    return [
        *kv("--candidate_scorer_mode", "hybrid"),
        *kv("--candidate_scorer_mix", "0.45"),
        *kv("--candidate_warmup_edges", 8192),
        *kv("--candidate_warmup_epochs", 3),
        *kv("--candidate_aux_rank_weight", "0.30"),
        *kv("--candidate_hard_negative_samples", 4),
        "--candidate_hard_negative_include_invalid",
        *kv("--candidate_hard_negative_margin", "0.30"),
        *kv("--candidate_aux_conflict_penalty", "0.50"),
        *kv("--candidate_aux_load_penalty", candidate_aux_load_penalty),
    ]


def ablation_specs() -> list[dict[str, Any]]:
    stage2_base = [*stage2_common()]
    stage2_dynamic_base = [*stage2_dynamic_priority_common()]
    base = [*stage4_common(candidate_storage_penalty="0.16")]
    hybrid = [*hybrid_scorer_args(candidate_aux_load_penalty="0.20")]
    return [
        {
            "name": "abl_stage2_no_future_task_execution",
            "group": "ablation",
            "base_stage": "stage2",
            "args": [
                *stage2_base,
                "--no_future_task_execution",
            ],
        },
        {
            "name": "abl_stage2_future_macro_with_current_valid",
            "group": "ablation",
            "base_stage": "stage2",
            "args": [
                *stage2_base,
                "--future_task_allow_with_current_valid",
            ],
        },
        {
            "name": "abl_stage2_no_dynamic_response_pressure",
            "group": "ablation",
            "base_stage": "stage2",
            "args": [
                *stage2_base,
                *kv("--candidate_dynamic_response_bonus", "0.00"),
                *kv("--candidate_dynamic_wait_penalty", "0.00"),
                *kv("--allocator_dynamic_response_bonus", "0.00"),
                *kv("--allocator_dynamic_wait_penalty", "0.00"),
                *kv("--dynamic_rescue_response_bonus", "0.00"),
            ],
        },
        {
            "name": "abl_stage2_no_dynamic_downlink_priority",
            "group": "ablation",
            "base_stage": "stage2",
            "args": [
                *stage2_base,
                "--no_dynamic_downlink_priority",
            ],
        },
        {
            "name": "abl_stage2_dynamic_no_future_task_execution",
            "group": "ablation",
            "base_stage": "stage2_dynamic",
            "args": [
                *stage2_dynamic_base,
                "--no_future_task_execution",
            ],
        },
        {
            "name": "abl_stage2_dynamic_open_routine_future",
            "group": "ablation",
            "base_stage": "stage2_dynamic",
            "args": [
                *stage2_dynamic_base,
                *kv("--future_routine_max_wait_s", 600),
                *kv("--routine_future_dynamic_guard_s", 0),
            ],
        },
        {
            "name": "abl_stage2_dynamic_restricted_future_macro",
            "group": "ablation",
            "base_stage": "stage2_dynamic",
            "args": [
                *stage2_dynamic_base,
                "--future_task_requires_no_current_valid",
            ],
        },
        {
            "name": "abl_future_macro_with_current_valid",
            "group": "ablation",
            "base_stage": "stage4",
            "args": [
                *base,
                "--future_task_allow_with_current_valid",
                *hybrid,
            ],
        },
        {
            "name": "abl_no_future_task_execution",
            "group": "ablation",
            "base_stage": "stage4",
            "args": [
                *base,
                "--no_future_task_execution",
                *hybrid,
            ],
        },
        {
            "name": "abl_no_executable_slot_reserve",
            "group": "ablation",
            "base_stage": "stage4",
            "args": [
                *base,
                *kv("--executable_slot_reserve_ratio", "0.00"),
                *hybrid,
            ],
        },
        {
            "name": "abl_no_storage_pressure",
            "group": "ablation",
            "base_stage": "stage4",
            "args": [
                *stage4_common(candidate_storage_penalty="0.00"),
                *hybrid,
            ],
        },
        {
            "name": "abl_no_dynamic_urgency",
            "group": "ablation",
            "base_stage": "stage4",
            "args": [
                *base,
                *kv("--candidate_dynamic_urgency_bonus", "0.00"),
                *kv("--allocator_dynamic_urgency_bonus", "0.00"),
                *hybrid,
            ],
        },
        {
            "name": "abl_no_stale_rescue",
            "group": "ablation",
            "base_stage": "stage4",
            "args": [
                *base,
                *kv("--allocator_stale_rescue_bonus", "0.00"),
                *hybrid,
            ],
        },
        {
            "name": "abl_no_wait_penalty",
            "group": "ablation",
            "base_stage": "stage4",
            "args": [
                *base,
                *kv("--candidate_wait_penalty", "0.00"),
                *kv("--allocator_wait_penalty", "0.00"),
                *hybrid,
            ],
        },
        {
            "name": "abl_v2_heuristic_scorer",
            "group": "ablation",
            "base_stage": "stage4",
            "args": [
                *base,
                *kv("--candidate_scorer_mode", "v2_heuristic"),
            ],
        },
        {
            "name": "abl_no_candidate_aux_update",
            "group": "ablation",
            "base_stage": "stage4",
            "args": [
                *base,
                *hybrid,
                "--no_candidate_aux_update",
            ],
        },
        {
            "name": "abl_no_invalid_hard_negatives",
            "group": "ablation",
            "base_stage": "stage4",
            "args": [
                *base,
                *kv("--candidate_scorer_mode", "hybrid"),
                *kv("--candidate_scorer_mix", "0.45"),
                *kv("--candidate_warmup_edges", 8192),
                *kv("--candidate_warmup_epochs", 3),
                *kv("--candidate_aux_rank_weight", "0.30"),
                *kv("--candidate_hard_negative_samples", 4),
                *kv("--candidate_hard_negative_margin", "0.30"),
                *kv("--candidate_aux_conflict_penalty", "0.50"),
                *kv("--candidate_aux_load_penalty", "0.20"),
            ],
        },
        {
            "name": "abl_no_action_type_gate",
            "group": "ablation",
            "base_stage": "stage4",
            "args": [
                *base,
                *hybrid,
                "--no_action_type_gate",
            ],
        },
        {
            "name": "abl_no_set_context",
            "group": "ablation",
            "base_stage": "stage4",
            "args": [
                *base,
                *hybrid,
                "--no_set_context",
            ],
        },
        {
            "name": "abl_no_idle_aux",
            "group": "ablation",
            "base_stage": "stage4",
            "args": [
                *base,
                *kv("--idle_aux_coeff", "0.00"),
                *hybrid,
            ],
        },
    ]


def command_for_spec(
    args: argparse.Namespace,
    suite_dir: Path,
    spec: dict[str, Any],
) -> list[str]:
    return [
        *base_args(args, suite_dir),
        *kv("--run_name", spec["name"]),
        *spec["args"],
        *(["--no_progress"] if args.no_progress else []),
    ]


def stream_command(cmd: list[str], log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
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


def read_metrics(result_dir: Path | None) -> dict[str, Any]:
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
    csv_path = suite_dir / "summary.csv"
    md_path = suite_dir / "summary.md"
    keys = list(SUMMARY_COLUMNS)
    extra_keys = sorted({k for row in rows for k in row.keys()} - set(keys))
    keys.extend(extra_keys)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    md_path.write_text(markdown_table(rows, SUMMARY_COLUMNS), encoding="utf-8")
    print(f"\nSummary CSV: {csv_path}")
    print(f"Summary Markdown: {md_path}")


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        if value is None:
            return ""
        return str(value)

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(c, "")) for c in columns) + " |")
    return "\n".join(lines) + "\n"


def run_suite(args: argparse.Namespace) -> int:
    suite_name = safe_name(args.suite_name or f"das_stage_ablation_{timestamp()}")
    suite_dir = Path(args.out_dir).resolve() / suite_name
    suite_dir.mkdir(parents=True, exist_ok=False)
    print(f"Suite directory: {suite_dir}")

    specs = stage_specs()
    if not args.stages_only:
        specs.extend(ablation_specs())
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
        log_path = suite_dir / "logs" / f"{spec['name']}.log"
        started = time.time()
        returncode = stream_command(cmd, log_path, env)
        elapsed = time.time() - started
        result_dir = latest_result_dir(suite_dir, spec["name"])
        metrics = read_metrics(result_dir)
        row: dict[str, Any] = {
            "experiment": spec["name"],
            "group": spec["group"],
            "base_stage": spec["base_stage"],
            "status": "ok" if returncode == 0 else "failed",
            "returncode": returncode,
            "result_dir": str(result_dir) if result_dir else "",
            "elapsed_s": round(elapsed, 3),
            **{k: metrics.get(k, "") for k in SUMMARY_COLUMNS if k not in {
                "experiment", "group", "base_stage", "status", "returncode", "result_dir"
            }},
        }
        rows.append(row)
        write_summary(rows, suite_dir)
        if returncode != 0 and not args.continue_on_error:
            print(f"Run failed: {spec['name']}. See {log_path}")
            return returncode
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DAS staged experiments plus targeted ablations and summarize metrics."
    )
    parser.add_argument("--acled_path", default="./DynamicMission/DynamicMission.shp")
    parser.add_argument("--scenario_cache_dir", default="runs/scenario_cache/das_cva_stress_seed42")
    parser.add_argument("--vtw_cache_dir", default="runs/scenario_cache/das_cva_stress_seed42/vtw_cache")
    parser.add_argument("--out_dir", default="runs/das_cva_mappo_sweeps")
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
    parser.add_argument("--no_dynamic_downlink_priority", action="store_true")
    parser.add_argument("--candidate_dynamic_response_bonus", type=float, default=0.24)
    parser.add_argument("--candidate_dynamic_wait_penalty", type=float, default=0.20)
    parser.add_argument("--dynamic_response_target_s", type=float, default=3600.0)
    parser.add_argument("--allocator_dynamic_response_bonus", type=float, default=0.24)
    parser.add_argument("--allocator_dynamic_wait_penalty", type=float, default=0.20)
    parser.add_argument("--dynamic_rescue_response_bonus", type=float, default=1.0)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--ppo_batch_size", type=int, default=512)
    parser.add_argument("--eval_max_steps", type=int, default=8000)
    parser.add_argument("--eval_workers", type=int, default=24)
    parser.add_argument("--torch_num_threads", type=int, default=1)
    parser.add_argument("--vtw_time_step_s", type=float, default=60.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval_device", default="cpu")
    parser.add_argument("--eval_use_repair", action="store_true")
    parser.add_argument("--eval_profile", action="store_true")
    parser.add_argument("--stages_only", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Run only selected experiment names, e.g. stage4_storage_pressure abl_no_idle_aux.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_suite(parse_args()))
