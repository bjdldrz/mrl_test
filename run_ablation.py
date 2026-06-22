"""
批量消融实验 runner
===================
围绕 compare_methods.py 运行一组可复现对比实验, 并汇总关键指标。

默认 preset=assignment_v2:
  - no_episode_assignment baseline
  - assignment_capacity_mode: equal / proportional
  - release_before_deadline_s: 0 / 1800
  - assign_w_load: 0.05 / 0.1 / 0.2

preset=reward_v1:
  - reward_default
  - team_reward_mix
  - load_balance_reward
  - team_completion_bonus
  - combined reward shaping + reward normalization

preset=state_v1:
  - critic mean pooling baseline
  - mean pooling + task stats
  - concat global state
  - concat global state + task stats

preset=oracle_v1:
  - no episode assignment + Greedy-Oracle
  - default assignment_v2 + Greedy-Oracle

每个子实验输出:
  <out_root>/<tag>/comparison_results.json
  <out_root>/<tag>/manifest.json

汇总输出:
  <out_root>/ablation_summary.csv
  <out_root>/ablation_summary.json
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _float_tag(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def build_assignment_v2_specs(assign_w_loads, release_windows, capacity_modes, include_no_assignment):
    specs = []
    if include_no_assignment:
        specs.append({
            "tag": "no_assignment",
            "extra_args": ["--no_episode_assignment"],
            "params": {
                "episode_assignment": False,
                "assignment_capacity_mode": "none",
                "assign_w_load": 0.0,
                "release_before_deadline_s": 0.0,
            },
        })

    for mode in capacity_modes:
        for w in assign_w_loads:
            for release_s in release_windows:
                tag = f"assign_{mode}_w{_float_tag(w)}_rel{int(release_s)}"
                specs.append({
                    "tag": tag,
                    "extra_args": [
                        "--assignment_capacity_mode", mode,
                        "--assign_w_load", str(w),
                        "--release_before_deadline_s", str(release_s),
                    ],
                    "params": {
                        "episode_assignment": True,
                        "assignment_capacity_mode": mode,
                        "assign_w_load": w,
                        "release_before_deadline_s": release_s,
                    },
                })
    return specs


def build_reward_v1_specs():
    base_assignment = [
        "--assignment_capacity_mode", "proportional",
        "--assign_w_load", "0.1",
        "--release_before_deadline_s", "1800",
    ]
    return [
        {
            "tag": "reward_default",
            "extra_args": [*base_assignment],
            "params": {
                "reward_variant": "default",
                "team_reward_mix": 0.0,
                "load_balance_reward_coeff": 0.0,
                "team_completion_bonus": 0.0,
                "normalize_agent_rewards": False,
            },
        },
        {
            "tag": "reward_team_mix_0p25",
            "extra_args": [*base_assignment, "--team_reward_mix", "0.25"],
            "params": {
                "reward_variant": "team_mix",
                "team_reward_mix": 0.25,
                "load_balance_reward_coeff": 0.0,
                "team_completion_bonus": 0.0,
                "normalize_agent_rewards": False,
            },
        },
        {
            "tag": "reward_load_balance_0p1",
            "extra_args": [*base_assignment, "--load_balance_reward_coeff", "0.1"],
            "params": {
                "reward_variant": "load_balance",
                "team_reward_mix": 0.0,
                "load_balance_reward_coeff": 0.1,
                "team_completion_bonus": 0.0,
                "normalize_agent_rewards": False,
            },
        },
        {
            "tag": "reward_team_completion_0p05",
            "extra_args": [*base_assignment, "--team_completion_bonus", "0.05"],
            "params": {
                "reward_variant": "team_completion",
                "team_reward_mix": 0.0,
                "load_balance_reward_coeff": 0.0,
                "team_completion_bonus": 0.05,
                "normalize_agent_rewards": False,
            },
        },
        {
            "tag": "reward_combined_norm",
            "extra_args": [
                *base_assignment,
                "--team_reward_mix", "0.25",
                "--load_balance_reward_coeff", "0.1",
                "--team_completion_bonus", "0.05",
                "--normalize_agent_rewards",
            ],
            "params": {
                "reward_variant": "combined_norm",
                "team_reward_mix": 0.25,
                "load_balance_reward_coeff": 0.1,
                "team_completion_bonus": 0.05,
                "normalize_agent_rewards": True,
            },
        },
    ]


def build_state_v1_specs():
    base_assignment = [
        "--assignment_capacity_mode", "proportional",
        "--assign_w_load", "0.1",
        "--release_before_deadline_s", "1800",
    ]


def build_oracle_v1_specs():
    return [
        {
            "tag": "oracle_no_assignment",
            "extra_args": ["--no_episode_assignment", "--run_oracle"],
            "params": {
                "oracle_variant": "no_assignment",
                "episode_assignment": False,
            },
        },
        {
            "tag": "oracle_assignment_v2",
            "extra_args": [
                "--assignment_capacity_mode", "proportional",
                "--assign_w_load", "0.1",
                "--release_before_deadline_s", "1800",
                "--run_oracle",
            ],
            "params": {
                "oracle_variant": "assignment_v2",
                "episode_assignment": True,
                "assignment_capacity_mode": "proportional",
                "assign_w_load": 0.1,
                "release_before_deadline_s": 1800.0,
            },
        },
    ]
    return [
        {
            "tag": "state_mean",
            "extra_args": [*base_assignment, "--global_state_mode", "mean"],
            "params": {
                "state_variant": "mean",
                "global_state_mode": "mean",
                "global_state_task_stats": False,
            },
        },
        {
            "tag": "state_mean_task_stats",
            "extra_args": [*base_assignment, "--global_state_mode", "mean", "--global_state_task_stats"],
            "params": {
                "state_variant": "mean_task_stats",
                "global_state_mode": "mean",
                "global_state_task_stats": True,
            },
        },
        {
            "tag": "state_concat",
            "extra_args": [*base_assignment, "--global_state_mode", "concat"],
            "params": {
                "state_variant": "concat",
                "global_state_mode": "concat",
                "global_state_task_stats": False,
            },
        },
        {
            "tag": "state_concat_task_stats",
            "extra_args": [*base_assignment, "--global_state_mode", "concat", "--global_state_task_stats"],
            "params": {
                "state_variant": "concat_task_stats",
                "global_state_mode": "concat",
                "global_state_task_stats": True,
            },
        },
    ]


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def metric(results, method, key, default=0.0):
    return float(results.get(method, {}).get(key, default))


def summarize_run(tag, params, out_dir):
    result_path = out_dir / "comparison_results.json"
    manifest_path = out_dir / "manifest.json"
    results = load_json(result_path)
    manifest = load_json(manifest_path) if manifest_path.exists() else {}

    mappo = results.get("MAPPO", {})
    indep = results.get("Indep-PPO", {})
    row = {
        "tag": tag,
        "out_dir": str(out_dir),
        "git_commit": manifest.get("git", {}).get("commit", ""),
        "git_dirty": manifest.get("git", {}).get("dirty", ""),
        **params,
    }

    keys = [
        "n_scheduled",
        "observation_success_rate",
        "dynamic_completion_rate",
        "routine_completion_rate",
        "total_reward",
        "duplicate_rate",
        "load_balance_cv",
        "avg_off_nadir_deg",
        "avg_dynamic_response_s",
        "coordination_gain",
        "oracle_relative_completion",
    ]
    for key in keys:
        row[f"mappo_{key}"] = metric(results, "MAPPO", key)
        row[f"indep_{key}"] = metric(results, "Indep-PPO", key)
        if "Greedy-Oracle" in results:
            row[f"oracle_{key}"] = metric(results, "Greedy-Oracle", key)

    row["delta_n_scheduled"] = row["mappo_n_scheduled"] - row["indep_n_scheduled"]
    row["delta_success_rate"] = row["mappo_observation_success_rate"] - row["indep_observation_success_rate"]
    row["delta_duplicate_rate"] = row["mappo_duplicate_rate"] - row["indep_duplicate_rate"]
    row["delta_load_balance_cv"] = row["mappo_load_balance_cv"] - row["indep_load_balance_cv"]
    row["delta_avg_off_nadir_deg"] = row["mappo_avg_off_nadir_deg"] - row["indep_avg_off_nadir_deg"]
    if "Greedy-Oracle" in results:
        row["mappo_oracle_gap_n_scheduled"] = row["oracle_n_scheduled"] - row["mappo_n_scheduled"]
        row["mappo_oracle_relative_completion"] = metric(results, "MAPPO", "oracle_relative_completion")
    return row


def write_summary(rows, out_root):
    json_path = out_root / "ablation_summary.json"
    csv_path = out_root / "ablation_summary.csv"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    if rows:
        fieldnames = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    return json_path, csv_path


def parse_float_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_str_list(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="批量运行 compare_methods.py 消融实验")
    parser.add_argument("--preset", type=str, default="assignment_v2",
                        choices=["assignment_v2", "reward_v1", "state_v1", "oracle_v1"])
    parser.add_argument("--python", type=str, default=sys.executable,
                        help="运行 compare_methods.py 的 Python 解释器")
    parser.add_argument("--out_root", type=str, default="runs/ablation_assignment_v2")
    parser.add_argument("--acled_path", type=str, default=None)
    parser.add_argument("--n_satellites", type=int, default=6)
    parser.add_argument("--train_iters", type=int, default=30)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--n_routine", type=int, default=200)
    parser.add_argument("--n_dynamic", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--assign_w_loads", type=str, default="0.05,0.1,0.2")
    parser.add_argument("--release_windows", type=str, default="0,1800")
    parser.add_argument("--capacity_modes", type=str, default="equal,proportional")
    parser.add_argument("--no_baseline", action="store_true",
                        help="不运行 --no_episode_assignment baseline")
    parser.add_argument("--skip_existing", action="store_true",
                        help="若子实验已有 manifest.json 则跳过")
    parser.add_argument("--dry_run", action="store_true",
                        help="只打印命令, 不运行")
    parser.add_argument("--run_oracle", action="store_true",
                        help="给每个子实验额外运行 Greedy-Oracle")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.preset == "assignment_v2":
        specs = build_assignment_v2_specs(
            assign_w_loads=parse_float_list(args.assign_w_loads),
            release_windows=parse_float_list(args.release_windows),
            capacity_modes=parse_str_list(args.capacity_modes),
            include_no_assignment=not args.no_baseline,
        )
    elif args.preset == "reward_v1":
        specs = build_reward_v1_specs()
    elif args.preset == "state_v1":
        specs = build_state_v1_specs()
    else:
        specs = build_oracle_v1_specs()

    rows = []
    for idx, spec in enumerate(specs, start=1):
        tag = spec["tag"]
        out_dir = out_root / tag
        manifest_path = out_dir / "manifest.json"
        cmd = [
            args.python,
            str(ROOT / "compare_methods.py"),
            "--n_satellites", str(args.n_satellites),
            "--train_iters", str(args.train_iters),
            "--eval_episodes", str(args.eval_episodes),
            "--n_routine", str(args.n_routine),
            "--n_dynamic", str(args.n_dynamic),
            "--seed", str(args.seed),
            "--out_dir", str(out_dir),
            "--device", args.device,
            "--experiment_tag", tag,
            *spec["extra_args"],
        ]
        if args.run_oracle and "--run_oracle" not in cmd:
            cmd.append("--run_oracle")
        if args.acled_path:
            cmd.extend(["--acled_path", args.acled_path])

        print(f"[{idx}/{len(specs)}] {tag}")
        print(" ".join(cmd))
        if args.dry_run:
            continue
        if args.skip_existing and manifest_path.exists():
            print(f"skip existing: {manifest_path}")
        else:
            subprocess.run(cmd, cwd=ROOT, check=True)
        rows.append(summarize_run(tag, spec["params"], out_dir))
        write_summary(rows, out_root)

    if args.dry_run:
        return
    json_path, csv_path = write_summary(rows, out_root)
    print(f"summary json: {json_path}")
    print(f"summary csv : {csv_path}")


if __name__ == "__main__":
    main()
