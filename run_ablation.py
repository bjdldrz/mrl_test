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

preset=train_stability_v1:
  - default assignment_v2
  - satellite curriculum
  - joint exploration
  - curriculum + joint exploration

preset=communication_v1:
  - default assignment_v2
  - intent broadcast
  - intent broadcast + train stability

preset=assignment_rolling_v1:
  - static assignment baseline
  - periodic rolling reassignment
  - event-triggered reassignment
  - 2h rolling horizon reassignment

preset=meta_encoder_v1:
  - MRL-DMS outer-loop LSTM/GRU/MLP/Transformer/Set Transformer
  - MAPPO + LSTM outer loop

preset=learned_assignment_v1:
  - heuristic assignment scorer baseline
  - deterministic MLP assignment scorer with different mix ratios
  - deterministic LSTM/GRU sequence assignment scorer
  - deterministic Transformer/Set Transformer assignment scorer
  - deterministic bipartite GNN assignment scorer

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

from utils.experiment_dirs import unique_dir, safe_name

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


def build_train_stability_v1_specs():
    base_assignment = [
        "--assignment_capacity_mode", "proportional",
        "--assign_w_load", "0.1",
        "--release_before_deadline_s", "1800",
    ]
    return [
        {
            "tag": "train_default",
            "extra_args": [*base_assignment],
            "params": {
                "train_variant": "default",
                "satellite_curriculum": False,
                "joint_explore_prob": 0.0,
            },
        },
        {
            "tag": "train_curriculum",
            "extra_args": [
                *base_assignment,
                "--satellite_curriculum",
                "--curriculum_min_satellites", "1",
                "--curriculum_iters", "10",
            ],
            "params": {
                "train_variant": "curriculum",
                "satellite_curriculum": True,
                "curriculum_min_satellites": 1,
                "curriculum_iters": 10,
                "joint_explore_prob": 0.0,
            },
        },
        {
            "tag": "train_joint_explore_0p05",
            "extra_args": [
                *base_assignment,
                "--joint_explore_prob", "0.05",
            ],
            "params": {
                "train_variant": "joint_explore",
                "satellite_curriculum": False,
                "joint_explore_prob": 0.05,
            },
        },
        {
            "tag": "train_curriculum_joint_explore",
            "extra_args": [
                *base_assignment,
                "--satellite_curriculum",
                "--curriculum_min_satellites", "1",
                "--curriculum_iters", "10",
                "--joint_explore_prob", "0.05",
            ],
            "params": {
                "train_variant": "curriculum_joint_explore",
                "satellite_curriculum": True,
                "curriculum_min_satellites": 1,
                "curriculum_iters": 10,
                "joint_explore_prob": 0.05,
            },
        },
    ]


def build_communication_v1_specs():
    base_assignment = [
        "--assignment_capacity_mode", "proportional",
        "--assign_w_load", "0.1",
        "--release_before_deadline_s", "1800",
    ]
    return [
        {
            "tag": "comm_default",
            "extra_args": [*base_assignment],
            "params": {
                "communication_variant": "default",
                "intent_broadcast": False,
                "intent_replan_rounds": 0,
            },
        },
        {
            "tag": "comm_intent_broadcast",
            "extra_args": [
                *base_assignment,
                "--intent_broadcast",
                "--intent_replan_rounds", "1",
            ],
            "params": {
                "communication_variant": "intent_broadcast",
                "intent_broadcast": True,
                "intent_replan_rounds": 1,
            },
        },
        {
            "tag": "comm_intent_train_stability",
            "extra_args": [
                *base_assignment,
                "--intent_broadcast",
                "--intent_replan_rounds", "1",
                "--satellite_curriculum",
                "--curriculum_min_satellites", "1",
                "--curriculum_iters", "10",
                "--joint_explore_prob", "0.05",
            ],
            "params": {
                "communication_variant": "intent_train_stability",
                "intent_broadcast": True,
                "intent_replan_rounds": 1,
                "satellite_curriculum": True,
                "curriculum_min_satellites": 1,
                "curriculum_iters": 10,
                "joint_explore_prob": 0.05,
            },
        },
    ]


def build_assignment_rolling_v1_specs():
    base_assignment = [
        "--assignment_capacity_mode", "proportional",
        "--assign_w_load", "0.1",
        "--release_before_deadline_s", "1800",
    ]
    return [
        {
            "tag": "rolling_static",
            "extra_args": [*base_assignment],
            "params": {
                "rolling_variant": "static",
                "assignment_replan_interval_s": 0.0,
                "assignment_replan_horizon_s": 0.0,
                "assignment_replan_trigger": "none",
            },
        },
        {
            "tag": "rolling_periodic_1h",
            "extra_args": [
                *base_assignment,
                "--assignment_replan_interval_s", "3600",
                "--assignment_replan_trigger", "periodic",
                "--assignment_switch_penalty", "0.05",
                "--assignment_lock_window_s", "600",
                "--assignment_max_switches_per_task", "2",
            ],
            "params": {
                "rolling_variant": "periodic_1h",
                "assignment_replan_interval_s": 3600.0,
                "assignment_replan_horizon_s": 0.0,
                "assignment_replan_trigger": "periodic",
                "assignment_switch_penalty": 0.05,
                "assignment_lock_window_s": 600.0,
                "assignment_max_switches_per_task": 2,
            },
        },
        {
            "tag": "rolling_event",
            "extra_args": [
                *base_assignment,
                "--assignment_replan_interval_s", "3600",
                "--assignment_replan_trigger", "dynamic,stale_owner,deadline",
                "--assignment_switch_penalty", "0.05",
                "--assignment_lock_window_s", "600",
                "--assignment_max_switches_per_task", "2",
            ],
            "params": {
                "rolling_variant": "event",
                "assignment_replan_interval_s": 3600.0,
                "assignment_replan_horizon_s": 0.0,
                "assignment_replan_trigger": "dynamic,stale_owner,deadline",
                "assignment_switch_penalty": 0.05,
                "assignment_lock_window_s": 600.0,
                "assignment_max_switches_per_task": 2,
            },
        },
        {
            "tag": "rolling_mpc_2h",
            "extra_args": [
                *base_assignment,
                "--assignment_replan_interval_s", "3600",
                "--assignment_replan_horizon_s", "7200",
                "--assignment_replan_trigger", "periodic,dynamic,stale_owner,deadline",
                "--assignment_switch_penalty", "0.05",
                "--assignment_lock_window_s", "600",
                "--assignment_max_switches_per_task", "2",
            ],
            "params": {
                "rolling_variant": "mpc_2h",
                "assignment_replan_interval_s": 3600.0,
                "assignment_replan_horizon_s": 7200.0,
                "assignment_replan_trigger": "periodic,dynamic,stale_owner,deadline",
                "assignment_switch_penalty": 0.05,
                "assignment_lock_window_s": 600.0,
                "assignment_max_switches_per_task": 2,
            },
        },
    ]


def build_learned_assignment_v1_specs(
    assignment_scorer_mixes,
    assignment_sequence_scorers,
    assignment_sequence_mixes,
    assignment_attention_scorers,
    assignment_attention_mixes,
    assignment_graph_scorers,
    assignment_graph_mixes,
):
    base_assignment = [
        "--assignment_capacity_mode", "proportional",
        "--assign_w_load", "0.1",
        "--release_before_deadline_s", "1800",
    ]
    specs = [
        {
            "tag": "assign_scorer_heuristic",
            "extra_args": [
                *base_assignment,
                "--assignment_scorer", "heuristic",
            ],
            "params": {
                "assignment_variant": "heuristic",
                "assignment_scorer": "heuristic",
                "assignment_scorer_mix": 0.0,
            },
        }
    ]
    for mix in assignment_scorer_mixes:
        tag = f"assign_scorer_mlp_mix{_float_tag(mix)}"
        specs.append({
            "tag": tag,
            "extra_args": [
                *base_assignment,
                "--assignment_scorer", "mlp",
                "--assignment_scorer_mix", str(mix),
                "--assignment_mlp_hidden_dim", "16",
            ],
            "params": {
                "assignment_variant": "mlp",
                "assignment_scorer": "mlp",
                "assignment_scorer_mix": mix,
                "assignment_mlp_hidden_dim": 16,
            },
        })
    for scorer in assignment_sequence_scorers:
        for mix in assignment_sequence_mixes:
            tag = f"assign_scorer_{scorer}_mix{_float_tag(mix)}"
            specs.append({
                "tag": tag,
                "extra_args": [
                    *base_assignment,
                    "--assignment_scorer", scorer,
                    "--assignment_scorer_mix", str(mix),
                    "--assignment_sequence_hidden_dim", "16",
                ],
                "params": {
                    "assignment_variant": scorer,
                    "assignment_scorer": scorer,
                    "assignment_scorer_mix": mix,
                    "assignment_sequence_hidden_dim": 16,
                },
            })
    for scorer in assignment_attention_scorers:
        for mix in assignment_attention_mixes:
            tag = f"assign_scorer_{scorer}_mix{_float_tag(mix)}"
            specs.append({
                "tag": tag,
                "extra_args": [
                    *base_assignment,
                    "--assignment_scorer", scorer,
                    "--assignment_scorer_mix", str(mix),
                    "--assignment_sequence_hidden_dim", "16",
                ],
                "params": {
                    "assignment_variant": scorer,
                    "assignment_scorer": scorer,
                    "assignment_scorer_mix": mix,
                    "assignment_context_hidden_dim": 16,
                },
            })
    for scorer in assignment_graph_scorers:
        for mix in assignment_graph_mixes:
            tag = f"assign_scorer_{scorer}_mix{_float_tag(mix)}"
            specs.append({
                "tag": tag,
                "extra_args": [
                    *base_assignment,
                    "--assignment_scorer", scorer,
                    "--assignment_scorer_mix", str(mix),
                    "--assignment_sequence_hidden_dim", "16",
                ],
                "params": {
                    "assignment_variant": scorer,
                    "assignment_scorer": scorer,
                    "assignment_scorer_mix": mix,
                    "assignment_graph_hidden_dim": 16,
                },
            })
    return specs


def build_meta_encoder_v1_specs(encoder_types, include_mappo_lstm=True, mappo_n_satellites=2):
    specs = []
    for encoder in encoder_types:
        tag = f"meta_single_{encoder}"
        specs.append({
            "tag": tag,
            "script": "train",
            "extra_args": [
                "--method", "mrl_dms",
                "--meta_encoder_type", encoder,
                "--mappo_n_satellites", "1",
            ],
            "params": {
                "meta_variant": tag,
                "meta_encoder_type": encoder,
                "mappo_n_satellites": 1,
                "multi_agent": False,
            },
        })
    if include_mappo_lstm:
        tag = f"meta_mappo_lstm_sat{mappo_n_satellites}"
        specs.append({
            "tag": tag,
            "script": "train",
            "extra_args": [
                "--method", "mrl_dms",
                "--meta_encoder_type", "lstm",
                "--mappo_n_satellites", str(mappo_n_satellites),
            ],
            "params": {
                "meta_variant": tag,
                "meta_encoder_type": "lstm",
                "mappo_n_satellites": mappo_n_satellites,
                "multi_agent": True,
            },
        })
    return specs


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
        "n_replans",
        "n_owner_switches",
        "owner_churn_rate",
        "stale_owner_rate",
        "deadline_rescue_rate",
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


def summarize_train_run(tag, params, out_dir):
    summary_path = out_dir / "summary.json"
    summary = load_json(summary_path)
    row = {
        "tag": tag,
        "out_dir": str(out_dir),
        **params,
        "best_reward": summary.get("best_reward", 0.0),
        "global_step": summary.get("global_step", 0),
        "total_iters": summary.get("total_iters", 0),
        "train_log": summary.get("train_log", ""),
        "eval_log": summary.get("eval_log", ""),
        "summary_json": str(summary_path),
    }
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
    parser = argparse.ArgumentParser(description="批量运行 MRL-DMS/compare_methods.py 消融实验")
    parser.add_argument("--preset", type=str, default="assignment_v2",
                        choices=["assignment_v2", "reward_v1", "state_v1", "oracle_v1",
                                 "train_stability_v1", "communication_v1",
                                 "assignment_rolling_v1",
                                 "meta_encoder_v1", "learned_assignment_v1"])
    parser.add_argument("--python", type=str, default=sys.executable,
                        help="运行 compare_methods.py 的 Python 解释器")
    parser.add_argument("--out_root", type=str, default="runs/ablation_assignment_v2")
    parser.add_argument("--batch_name", type=str, default=None,
                        help="本批消融实验名称; 默认由 preset/关键参数生成")
    parser.add_argument("--flat_out_root", action="store_true",
                        help="直接写入 --out_root, 不自动创建唯一批次子目录")
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
    parser.add_argument("--assignment_scorer_mixes", type=str, default="0.1,0.25,0.5",
                        help="learned_assignment_v1 使用的 MLP scorer 混合比例列表")
    parser.add_argument("--assignment_sequence_scorers", type=str, default="lstm,gru",
                        help="learned_assignment_v1 使用的序列 scorer 列表: lstm,gru")
    parser.add_argument("--assignment_sequence_mixes", type=str, default="0.25",
                        help="learned_assignment_v1 使用的序列 scorer 混合比例列表")
    parser.add_argument("--assignment_attention_scorers", type=str, default="transformer,set_transformer",
                        help="learned_assignment_v1 使用的集合/注意力 scorer 列表")
    parser.add_argument("--assignment_attention_mixes", type=str, default="0.25",
                        help="learned_assignment_v1 使用的集合/注意力 scorer 混合比例列表")
    parser.add_argument("--assignment_graph_scorers", type=str, default="gnn",
                        help="learned_assignment_v1 使用的图 scorer 列表: gnn")
    parser.add_argument("--assignment_graph_mixes", type=str, default="0.25",
                        help="learned_assignment_v1 使用的图 scorer 混合比例列表")
    parser.add_argument("--meta_encoder_types", type=str,
                        default="lstm,gru,mlp,transformer,set_transformer",
                        help="meta_encoder_v1 使用的外循环编码器列表")
    parser.add_argument("--meta_iterations", type=int, default=2,
                        help="meta_encoder_v1 每个子实验的外循环迭代次数")
    parser.add_argument("--meta_mappo_n_satellites", type=int, default=2,
                        help="meta_encoder_v1 中 MAPPO+LSTM 外循环的卫星数量")
    parser.add_argument("--no_mappo_lstm", action="store_true",
                        help="meta_encoder_v1 不运行 MAPPO+LSTM 外循环分支")
    parser.add_argument("--full_train", action="store_true",
                        help="meta_encoder_v1 使用完整训练配置; 默认加 --fast 便于 smoke")
    args = parser.parse_args()

    if args.flat_out_root:
        out_root = Path(args.out_root)
        out_root.mkdir(parents=True, exist_ok=True)
    else:
        batch_name = args.batch_name or (
            f"{args.preset}_sat{args.n_satellites}_iter{args.train_iters}_"
            f"eval{args.eval_episodes}_seed{args.seed}"
        )
        out_root = unique_dir(args.out_root, safe_name(batch_name))

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
    elif args.preset == "oracle_v1":
        specs = build_oracle_v1_specs()
    elif args.preset == "train_stability_v1":
        specs = build_train_stability_v1_specs()
    elif args.preset == "communication_v1":
        specs = build_communication_v1_specs()
    elif args.preset == "assignment_rolling_v1":
        specs = build_assignment_rolling_v1_specs()
    elif args.preset == "learned_assignment_v1":
        seq_scorers = parse_str_list(args.assignment_sequence_scorers)
        allowed_seq_scorers = {"lstm", "gru"}
        invalid_seq = [s for s in seq_scorers if s not in allowed_seq_scorers]
        if invalid_seq:
            raise ValueError(
                f"未知 assignment sequence scorer: {invalid_seq}; "
                f"可选: {sorted(allowed_seq_scorers)}"
            )
        attention_scorers = parse_str_list(args.assignment_attention_scorers)
        allowed_attention_scorers = {"transformer", "set_transformer"}
        invalid_attention = [s for s in attention_scorers if s not in allowed_attention_scorers]
        if invalid_attention:
            raise ValueError(
                f"未知 assignment attention scorer: {invalid_attention}; "
                f"可选: {sorted(allowed_attention_scorers)}"
            )
        graph_scorers = parse_str_list(args.assignment_graph_scorers)
        allowed_graph_scorers = {"gnn"}
        invalid_graph = [s for s in graph_scorers if s not in allowed_graph_scorers]
        if invalid_graph:
            raise ValueError(
                f"未知 assignment graph scorer: {invalid_graph}; "
                f"可选: {sorted(allowed_graph_scorers)}"
            )
        specs = build_learned_assignment_v1_specs(
            assignment_scorer_mixes=parse_float_list(args.assignment_scorer_mixes),
            assignment_sequence_scorers=seq_scorers,
            assignment_sequence_mixes=parse_float_list(args.assignment_sequence_mixes),
            assignment_attention_scorers=attention_scorers,
            assignment_attention_mixes=parse_float_list(args.assignment_attention_mixes),
            assignment_graph_scorers=graph_scorers,
            assignment_graph_mixes=parse_float_list(args.assignment_graph_mixes),
        )
    else:
        encoder_types = parse_str_list(args.meta_encoder_types)
        allowed_encoders = {"lstm", "gru", "mlp", "transformer", "set_transformer"}
        invalid = [e for e in encoder_types if e not in allowed_encoders]
        if invalid:
            raise ValueError(
                f"未知 meta encoder: {invalid}; "
                f"可选: {sorted(allowed_encoders)}"
            )
        specs = build_meta_encoder_v1_specs(
            encoder_types=encoder_types,
            include_mappo_lstm=not args.no_mappo_lstm,
            mappo_n_satellites=args.meta_mappo_n_satellites,
        )

    rows = []
    for idx, spec in enumerate(specs, start=1):
        tag = spec["tag"]
        out_dir = out_root / tag
        manifest_path = out_dir / "manifest.json"
        if spec.get("script") == "train":
            manifest_path = out_dir / "summary.json"
            cmd = [
                args.python,
                str(ROOT / "train.py"),
                "--seed", str(args.seed),
                "--device", args.device,
                "--meta_iterations", str(args.meta_iterations),
                "--log_dir", str(out_root),
                "--exp_name", tag,
                *spec["extra_args"],
            ]
            if not args.full_train:
                cmd.insert(6, "--fast")
            if args.acled_path:
                cmd.extend(["--acled_path", args.acled_path])
        else:
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
                "--flat_out_dir",
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
        if spec.get("script") == "train":
            rows.append(summarize_train_run(tag, spec["params"], out_dir))
        else:
            rows.append(summarize_run(tag, spec["params"], out_dir))
        write_summary(rows, out_root)

    if args.dry_run:
        return
    json_path, csv_path = write_summary(rows, out_root)
    print(f"summary json: {json_path}")
    print(f"summary csv : {csv_path}")


if __name__ == "__main__":
    main()
