#!/usr/bin/env python3
"""Run paper-focused DAS-CVA-MAPPO experiment plans.

This wrapper keeps the lower-level experiment definitions in
`run_stage_ablation_suite.py`, but groups currently runnable experiments by the
paper question they support. Results are saved by the stage suite, and this
script writes an additional paper plan manifest into the suite directory.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE_SUITE = REPO_ROOT / "scripts" / "run_stage_ablation_suite.py"


TEMPORAL_EXPERIMENTS = [
    "cmp_stage2_temporal_early_delivery_features",
    "cmp_stage2_temporal_future_features",
    "cmp_stage2_temporal_gru_state",
    "abl_stage2_no_temporal_window_features",
]

PAPER_CORE_TEMPORAL_EXPERIMENTS = [
    "cmp_stage2_temporal_future_features",
    "cmp_stage2_temporal_gru_state",
    "abl_stage2_no_temporal_window_features",
]

STAGE_PROGRESSION_EXPERIMENTS = [
    "stage1_slot_diagnosis",
    "stage2_candidate_owner_repair",
    "stage2_dynamic_priority_recovery",
    "stage3_dynamic_hybrid",
    "stage4_storage_pressure",
]

STAGE2_MECHANISM_ABLATIONS = [
    "abl_stage2_no_future_task_execution",
    "abl_stage2_no_dynamic_response_pressure",
    "abl_stage2_no_response_budget_features",
    "abl_stage2_no_downlink_aware_edge_value",
    "abl_stage2_posthoc_dynamic_downlink_priority",
]

STAGE4_MODEL_ABLATIONS = [
    "abl_no_future_task_execution",
    "abl_no_storage_pressure",
    "abl_v2_heuristic_scorer",
    "abl_no_candidate_aux_update",
    "abl_no_action_type_gate",
    "abl_no_set_context",
    "abl_no_idle_aux",
]

PAPER_FULL_EXTRA_ABLATIONS = [
    "abl_stage2_no_dynamic_downlink_priority",
    "abl_stage2_no_early_delivery_temporal_features",
    "abl_stage2_future_macro_with_current_valid",
    "abl_stage2_dynamic_no_future_task_execution",
    "abl_stage2_dynamic_open_routine_future",
    "abl_stage2_dynamic_restricted_future_macro",
    "abl_future_macro_with_current_valid",
    "abl_no_executable_slot_reserve",
    "abl_no_dynamic_urgency",
    "abl_no_stale_rescue",
    "abl_no_wait_penalty",
    "abl_no_invalid_hard_negatives",
]

STRESS_SCALE_EXPERIMENTS = [
    "stage2_candidate_owner_repair",
    "stage4_storage_pressure",
    "abl_no_storage_pressure",
    "abl_stage2_no_downlink_aware_edge_value",
]


PLAN_DEFINITIONS: dict[str, dict[str, Any]] = {
    "quick_temporal": {
        "description": "Fast V0.33 temporal table: early delivery, V0.32-like future features, GRU, and no temporal.",
        "experiments": TEMPORAL_EXPERIMENTS,
    },
    "mechanism_core": {
        "description": "Core mechanism ablations for dynamic response, downlink-aware value, response budget, and temporal design.",
        "experiments": [
            *TEMPORAL_EXPERIMENTS,
            *STAGE2_MECHANISM_ABLATIONS,
        ],
    },
    "progression": {
        "description": "Stage progression table from slot diagnosis to storage/downlink pressure.",
        "experiments": STAGE_PROGRESSION_EXPERIMENTS,
    },
    "paper_core": {
        "description": "Recommended thesis suite: deduplicated progression, V0.33 temporal comparison, and the highest-value mechanism/model ablations.",
        "experiments": [
            *STAGE_PROGRESSION_EXPERIMENTS,
            *PAPER_CORE_TEMPORAL_EXPERIMENTS,
            *STAGE2_MECHANISM_ABLATIONS,
            *STAGE4_MODEL_ABLATIONS,
        ],
    },
    "paper_full": {
        "description": "All currently runnable paper-relevant DAS experiments exposed by the stage suite.",
        "experiments": [
            *STAGE_PROGRESSION_EXPERIMENTS,
            *TEMPORAL_EXPERIMENTS,
            *STAGE2_MECHANISM_ABLATIONS,
            *STAGE4_MODEL_ABLATIONS,
            *PAPER_FULL_EXTRA_ABLATIONS,
        ],
    },
    "stress_12sat_double_tasks": {
        "description": "Focused stress test with 12 satellites and doubled routine/dynamic tasks.",
        "experiments": STRESS_SCALE_EXPERIMENTS,
        "default_overrides": {
            "n_satellites": 12,
            "n_routine": 1200,
            "n_dynamic": 300,
            "eval_max_steps": 12000,
        },
    },
}


EXPERIMENT_NOTES = {
    "stage1_slot_diagnosis": "Candidate slot/executable exposure diagnosis.",
    "stage2_candidate_owner_repair": "Strong Stage-2 DAS baseline before learned scorer/storage pressure.",
    "stage2_dynamic_priority_recovery": "Dynamic-priority candidate exposure configuration.",
    "stage3_dynamic_hybrid": "Hybrid learned candidate scorer with dynamic setting.",
    "stage4_storage_pressure": "Hybrid scorer with stronger storage/downlink pressure.",
    "cmp_stage2_temporal_early_delivery_features": "V0.33 main temporal model: future windows plus early-delivery signals.",
    "cmp_stage2_temporal_future_features": "V0.32-like future-window temporal features without early-delivery signals.",
    "cmp_stage2_temporal_gru_state": "GRU local-state history temporal encoder comparison.",
    "abl_stage2_no_temporal_window_features": "Remove future-window temporal action/edge features.",
    "abl_stage2_no_dynamic_downlink_priority": "No post-hoc dynamic downlink priority baseline.",
    "abl_stage2_no_future_task_execution": "Remove bounded future-task macro execution.",
    "abl_stage2_no_dynamic_response_pressure": "Remove dynamic response pressure from candidate/allocator.",
    "abl_stage2_no_response_budget_features": "Remove V0.31 response-budget actor/scorer features.",
    "abl_stage2_no_early_delivery_temporal_features": "Remove only V0.33 early-delivery temporal signals.",
    "abl_stage2_no_downlink_aware_edge_value": "Remove downlink-aware candidate edge value terms.",
    "abl_stage2_posthoc_dynamic_downlink_priority": "Compare rejected post-hoc dynamic downlink priority path.",
    "abl_no_future_task_execution": "Stage-4 future-task macro ablation.",
    "abl_no_storage_pressure": "Stage-4 storage pressure ablation.",
    "abl_v2_heuristic_scorer": "Stage-4 learned/hybrid scorer replaced by heuristic scorer.",
    "abl_no_candidate_aux_update": "Disable rollout-advantage auxiliary scorer updates.",
    "abl_no_action_type_gate": "Remove action-type gate in action-set actor.",
    "abl_no_set_context": "Remove set-context attention contribution.",
    "abl_no_idle_aux": "Remove idle auxiliary loss.",
    "abl_stage2_future_macro_with_current_valid": "Allow future macro even when current valid actions exist.",
    "abl_stage2_dynamic_no_future_task_execution": "Dynamic-priority Stage-2 run without future-task macro execution.",
    "abl_stage2_dynamic_open_routine_future": "Dynamic-priority Stage-2 run with less restricted routine future macro actions.",
    "abl_stage2_dynamic_restricted_future_macro": "Dynamic-priority Stage-2 run requiring no current valid action before future macro.",
    "abl_future_macro_with_current_valid": "Stage-4 run allowing future macro with current valid actions.",
    "abl_no_executable_slot_reserve": "Remove executable-slot reserve from Stage-4 candidate exposure.",
    "abl_no_dynamic_urgency": "Remove dynamic urgency bonuses from candidate and allocator scoring.",
    "abl_no_stale_rescue": "Remove stale-owner rescue bonus.",
    "abl_no_wait_penalty": "Remove candidate and allocator wait penalties.",
    "abl_no_invalid_hard_negatives": "Exclude invalid hard negatives from hybrid scorer auxiliary ranking.",
}


def safe_name(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", str(text or "run")).strip("._-")
    return (text or "run")[:max_len]


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def unique_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def selected_experiments(args: argparse.Namespace) -> list[str]:
    if args.experiments:
        return unique_order(args.experiments)
    return unique_order(list(PLAN_DEFINITIONS[args.plan]["experiments"]))


def provided_option_names(argv: list[str]) -> set[str]:
    provided = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        provided.add(token.split("=", 1)[0])
    return provided


def apply_plan_defaults(args: argparse.Namespace, provided_options: set[str]) -> None:
    defaults = PLAN_DEFINITIONS[args.plan].get("default_overrides", {})
    for key, value in defaults.items():
        flag_variants = {f"--{key}", f"--{key.replace('_', '-')}"}
        if flag_variants & provided_options:
            continue
        setattr(args, key, value)


def build_stage_command(args: argparse.Namespace, experiments: list[str], unknown_args: list[str]) -> list[str]:
    cmd = [
        sys.executable,
        str(STAGE_SUITE),
        "--out_dir",
        args.out_dir,
        "--suite_name",
        args.suite_name,
        "--acled_path",
        args.acled_path,
        "--scenario_cache_dir",
        args.scenario_cache_dir,
        "--vtw_cache_dir",
        args.vtw_cache_dir,
        "--n_satellites",
        str(args.n_satellites),
        "--train_iters",
        str(args.train_iters),
        "--eval_episodes",
        str(args.eval_episodes),
        "--n_routine",
        str(args.n_routine),
        "--n_dynamic",
        str(args.n_dynamic),
        "--n_ground_stations",
        str(args.n_ground_stations),
        "--downlink_time_s",
        str(args.downlink_time_s),
        "--satellite_storage_capacity",
        str(args.satellite_storage_capacity),
        "--inter_satellite_transfer_time_s",
        str(args.inter_satellite_transfer_time_s),
        "--rollout_steps",
        str(args.rollout_steps),
        "--train_env_workers",
        str(args.train_env_workers),
        "--ppo_epochs",
        str(args.ppo_epochs),
        "--ppo_batch_size",
        str(args.ppo_batch_size),
        "--eval_max_steps",
        str(args.eval_max_steps),
        "--eval_workers",
        str(args.eval_workers),
        "--eval_device",
        args.eval_device,
        "--device",
        args.device,
        "--torch_num_threads",
        str(args.torch_num_threads),
        "--vtw_time_step_s",
        str(args.vtw_time_step_s),
    ]
    if args.eval_profile:
        cmd.append("--eval_profile")
    if args.eval_use_repair:
        cmd.append("--eval_use_repair")
    if args.continue_on_error:
        cmd.append("--continue_on_error")
    if args.no_progress:
        cmd.append("--no_progress")
    cmd.extend(unknown_args)
    cmd.extend(["--only", *experiments])
    return cmd


def print_plans() -> None:
    for name, spec in PLAN_DEFINITIONS.items():
        print(f"{name}: {spec['description']}")
        defaults = spec.get("default_overrides", {})
        if defaults:
            default_text = ", ".join(f"{key}={value}" for key, value in defaults.items())
            print(f"  defaults: {default_text}")
        for experiment in unique_order(list(spec["experiments"])):
            print(f"  - {experiment}: {EXPERIMENT_NOTES.get(experiment, '')}")


def write_plan_files(
    suite_dir: Path,
    args: argparse.Namespace,
    experiments: list[str],
    command: list[str],
    returncode: int,
    elapsed_s: float,
) -> None:
    suite_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "plan": args.plan,
        "plan_description": PLAN_DEFINITIONS[args.plan]["description"],
        "suite_name": args.suite_name,
        "elapsed_s": elapsed_s,
        "returncode": returncode,
        "plan_default_overrides": PLAN_DEFINITIONS[args.plan].get("default_overrides", {}),
        "experiments": [
            {
                "name": experiment,
                "paper_use": EXPERIMENT_NOTES.get(experiment, ""),
            }
            for experiment in experiments
        ],
        "command": command,
        "results": {
            "summary_csv": "summary.csv",
            "summary_md": "summary.md",
            "logs_dir": "logs",
        },
    }
    (suite_dir / "paper_experiment_plan.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Paper Experiment Plan",
        "",
        f"- plan: `{args.plan}`",
        f"- suite: `{args.suite_name}`",
        f"- returncode: `{returncode}`",
        f"- elapsed_s: `{elapsed_s:.3f}`",
        "",
        "| experiment | paper use |",
        "| --- | --- |",
    ]
    for experiment in experiments:
        lines.append(f"| `{experiment}` | {EXPERIMENT_NOTES.get(experiment, '')} |")
    lines.extend([
        "",
        "## Result Files",
        "",
        "- `summary.csv`",
        "- `summary.md`",
        "- `logs/`",
    ])
    (suite_dir / "paper_experiment_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run paper-focused DAS-CVA-MAPPO experiment plans and save result summaries."
    )
    parser.add_argument("--plan", choices=sorted(PLAN_DEFINITIONS), default="paper_core")
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Override the selected plan with explicit stage-suite experiment names.",
    )
    parser.add_argument("--list_plans", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--out_dir", default="runs/paper_experiment_suites")
    parser.add_argument("--suite_name", default="")
    parser.add_argument("--acled_path", default="./DynamicMission/DynamicMission.shp")
    parser.add_argument("--scenario_cache_dir", default="runs/scenario_cache/das_cva_stress_seed42")
    parser.add_argument("--vtw_cache_dir", default="runs/scenario_cache/das_cva_stress_seed42/vtw_cache")
    parser.add_argument("--n_satellites", type=int, default=6)
    parser.add_argument("--train_iters", type=int, default=50)
    parser.add_argument(
        "--eval_episodes",
        "--val_episodes",
        dest="eval_episodes",
        type=int,
        default=10,
    )
    parser.add_argument("--n_routine", type=int, default=600)
    parser.add_argument("--n_dynamic", type=int, default=150)
    parser.add_argument("--n_ground_stations", type=int, default=4)
    parser.add_argument("--downlink_time_s", type=float, default=30.0)
    parser.add_argument("--satellite_storage_capacity", type=int, default=30)
    parser.add_argument("--inter_satellite_transfer_time_s", type=float, default=300.0)
    parser.add_argument("--rollout_steps", type=int, default=512)
    parser.add_argument("--train_env_workers", type=int, default=16)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--ppo_batch_size", type=int, default=512)
    parser.add_argument("--eval_max_steps", type=int, default=8000)
    parser.add_argument("--eval_workers", type=int, default=10)
    parser.add_argument("--eval_device", default="cpu")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch_num_threads", type=int, default=1)
    parser.add_argument("--vtw_time_step_s", type=float, default=60.0)
    parser.add_argument("--eval_profile", action="store_true")
    parser.add_argument("--eval_use_repair", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    return parser.parse_known_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else argv
    args, unknown_args = parse_args(raw_argv)
    if args.list_plans:
        print_plans()
        return 0

    apply_plan_defaults(args, provided_option_names(raw_argv))
    experiments = selected_experiments(args)
    if not args.suite_name:
        args.suite_name = safe_name(f"{args.plan}_{timestamp()}")
    suite_dir = Path(args.out_dir).resolve() / args.suite_name
    command = build_stage_command(args, experiments, unknown_args)

    print(f"Plan: {args.plan}")
    print(f"Suite directory: {suite_dir}")
    print("Experiments:")
    for experiment in experiments:
        print(f"  - {experiment}: {EXPERIMENT_NOTES.get(experiment, '')}")
    print("\nCommand:")
    print(shlex.join(command))

    if args.dry_run:
        return 0

    started = time.time()
    returncode = subprocess.call(command, cwd=REPO_ROOT)
    elapsed_s = time.time() - started
    write_plan_files(
        suite_dir=suite_dir,
        args=args,
        experiments=experiments,
        command=command,
        returncode=int(returncode),
        elapsed_s=float(elapsed_s),
    )
    print(f"\nPaper plan: {suite_dir / 'paper_experiment_plan.md'}")
    print(f"Summary: {suite_dir / 'summary.md'}")
    return int(returncode)


if __name__ == "__main__":
    raise SystemExit(main())
