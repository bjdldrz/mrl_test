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

preset=hier_assignment_v1:
  - rolling horizon without manager
  - rule-based high-level assignment manager

preset=cva_assignment_v1:
  - heuristic static / rolling assignment baselines
  - contextual value-aware assignment static ablation
  - CVA rolling assignment with MLP/LSTM/GRU/Transformer/Set Transformer encoders

preset=owner_effect_v1:
  - no-owner independent multi-satellite execution
  - no-owner MAPPO with shared evaluation but no owner pre-assignment
  - static owner assignment
  - CVA rolling owner assignment

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from utils.experiment_dirs import unique_dir, safe_name
from utils.json_utils import dump_json

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


def build_hier_assignment_v1_specs():
    base_rolling = [
        "--assignment_capacity_mode", "proportional",
        "--assign_w_load", "0.1",
        "--release_before_deadline_s", "1800",
        "--assignment_replan_interval_s", "3600",
        "--assignment_replan_horizon_s", "7200",
        "--assignment_replan_trigger", "periodic,dynamic,stale_owner,deadline",
        "--assignment_switch_penalty", "0.05",
        "--assignment_lock_window_s", "600",
        "--assignment_max_switches_per_task", "2",
    ]
    return [
        {
            "tag": "hier_no_manager",
            "extra_args": [*base_rolling, "--assignment_manager_mode", "none"],
            "params": {
                "hier_variant": "no_manager",
                "assignment_manager_mode": "none",
                "assignment_replan_horizon_s": 7200.0,
            },
        },
        {
            "tag": "hier_rule_manager",
            "extra_args": [*base_rolling, "--assignment_manager_mode", "rule"],
            "params": {
                "hier_variant": "rule_manager",
                "assignment_manager_mode": "rule",
                "assignment_replan_horizon_s": 7200.0,
            },
        },
    ]


def _cva_args(encoder: str, mix: float, context_weight: float):
    args = [
        "--assignment_scorer", "cva",
        "--assignment_scorer_mix", str(mix),
        "--assignment_context_encoder", encoder,
        "--assignment_context_weight", str(context_weight),
    ]
    if encoder == "mlp":
        args.extend(["--assignment_mlp_hidden_dim", "16"])
    else:
        args.extend(["--assignment_sequence_hidden_dim", "16"])
    return args


def build_cva_assignment_v1_specs(context_encoders, scorer_mixes, context_weight):
    base_assignment = [
        "--assignment_capacity_mode", "proportional",
        "--assign_w_load", "0.1",
        "--release_before_deadline_s", "1800",
    ]
    rolling_args = [
        "--assignment_replan_interval_s", "3600",
        "--assignment_replan_horizon_s", "7200",
        "--assignment_replan_trigger", "periodic,dynamic,stale_owner,deadline",
        "--assignment_switch_penalty", "0.05",
        "--assignment_lock_window_s", "600",
        "--assignment_max_switches_per_task", "2",
    ]
    specs = [
        {
            "tag": "heuristic_static",
            "extra_args": [
                *base_assignment,
                "--assignment_scorer", "heuristic",
            ],
            "params": {
                "cva_variant": "heuristic_static",
                "assignment_scorer": "heuristic",
                "assignment_context_encoder": "",
                "assignment_scorer_mix": 0.0,
                "assignment_replan_trigger": "none",
            },
        },
        {
            "tag": "heuristic_rolling",
            "extra_args": [
                *base_assignment,
                *rolling_args,
                "--assignment_scorer", "heuristic",
            ],
            "params": {
                "cva_variant": "heuristic_rolling",
                "assignment_scorer": "heuristic",
                "assignment_context_encoder": "",
                "assignment_scorer_mix": 0.0,
                "assignment_replan_trigger": "periodic,dynamic,stale_owner,deadline",
                "assignment_replan_horizon_s": 7200.0,
            },
        },
    ]

    if "lstm" in context_encoders:
        mix = scorer_mixes[0] if scorer_mixes else 0.35
        specs.append({
            "tag": f"cva_lstm_static_mix{_float_tag(mix)}",
            "extra_args": [
                *base_assignment,
                *_cva_args("lstm", mix, context_weight),
            ],
            "params": {
                "cva_variant": "cva_static",
                "assignment_scorer": "cva",
                "assignment_context_encoder": "lstm",
                "assignment_scorer_mix": mix,
                "assignment_context_weight": context_weight,
                "assignment_replan_trigger": "none",
            },
        })

    for encoder in context_encoders:
        for mix in scorer_mixes:
            tag = f"cva_{encoder}_rolling_mix{_float_tag(mix)}"
            specs.append({
                "tag": tag,
                "extra_args": [
                    *base_assignment,
                    *rolling_args,
                    *_cva_args(encoder, mix, context_weight),
                ],
                "params": {
                    "cva_variant": "cva_rolling",
                    "assignment_scorer": "cva",
                    "assignment_context_encoder": encoder,
                    "assignment_scorer_mix": mix,
                    "assignment_context_weight": context_weight,
                    "assignment_replan_trigger": "periodic,dynamic,stale_owner,deadline",
                    "assignment_replan_horizon_s": 7200.0,
                    "assignment_switch_penalty": 0.05,
                },
            })
    return specs


def build_candidate_action_v1_specs(top_ks):
    base_args = [
        "--assignment_capacity_mode", "proportional",
        "--assign_w_load", "0.1",
        "--release_before_deadline_s", "1800",
        "--assignment_replan_interval_s", "3600",
        "--assignment_replan_horizon_s", "7200",
        "--assignment_replan_trigger", "periodic,dynamic,stale_owner,deadline",
        "--assignment_switch_penalty", "0.05",
        "--assignment_lock_window_s", "600",
        "--assignment_max_switches_per_task", "2",
        *_cva_args("lstm", 0.35, 0.25),
    ]
    specs = []
    for top_k in top_ks:
        top_k = int(top_k)
        tag = "candidate_full" if top_k <= 0 else f"candidate_topk{top_k}"
        extra_args = [*base_args]
        if top_k > 0:
            extra_args.extend(["--candidate_action_top_k", str(top_k)])
        specs.append({
            "tag": tag,
            "extra_args": extra_args,
            "params": {
                "candidate_variant": tag,
                "candidate_action_top_k": max(0, top_k),
                "assignment_scorer": "cva",
                "assignment_context_encoder": "lstm",
                "assignment_scorer_mix": 0.35,
                "assignment_context_weight": 0.25,
                "assignment_replan_trigger": "periodic,dynamic,stale_owner,deadline",
                "assignment_replan_horizon_s": 7200.0,
            },
        })
    return specs


def build_owner_effect_v1_specs():
    base_assignment = [
        "--assignment_capacity_mode", "proportional",
        "--assign_w_load", "0.1",
        "--release_before_deadline_s", "1800",
    ]
    rolling_args = [
        "--assignment_replan_interval_s", "3600",
        "--assignment_replan_horizon_s", "7200",
        "--assignment_replan_trigger", "periodic,dynamic,stale_owner,deadline",
        "--assignment_switch_penalty", "0.05",
        "--assignment_lock_window_s", "600",
        "--assignment_max_switches_per_task", "2",
    ]
    return [
        {
            "tag": "no_owner_indep_ppo",
            "extra_args": [
                "--methods", "indep",
                "--no_episode_assignment",
            ],
            "params": {
                "owner_effect_variant": "no_owner_indep_ppo",
                "methods": "indep",
                "episode_assignment": False,
                "coordinate": False,
            },
        },
        {
            "tag": "no_owner_mappo",
            "extra_args": [
                "--methods", "mappo",
                "--no_episode_assignment",
            ],
            "params": {
                "owner_effect_variant": "no_owner_mappo",
                "methods": "mappo",
                "episode_assignment": False,
                "coordinate": True,
            },
        },
        {
            "tag": "owner_heuristic_static",
            "extra_args": [
                "--methods", "mappo",
                *base_assignment,
                "--assignment_scorer", "heuristic",
            ],
            "params": {
                "owner_effect_variant": "owner_heuristic_static",
                "methods": "mappo",
                "episode_assignment": True,
                "assignment_scorer": "heuristic",
                "assignment_replan_trigger": "none",
            },
        },
        {
            "tag": "owner_cva_lstm_rolling",
            "extra_args": [
                "--methods", "mappo",
                *base_assignment,
                *rolling_args,
                *_cva_args("lstm", 0.35, 0.25),
            ],
            "params": {
                "owner_effect_variant": "owner_cva_lstm_rolling",
                "methods": "mappo",
                "episode_assignment": True,
                "assignment_scorer": "cva",
                "assignment_context_encoder": "lstm",
                "assignment_scorer_mix": 0.35,
                "assignment_context_weight": 0.25,
                "assignment_replan_trigger": "periodic,dynamic,stale_owner,deadline",
                "assignment_replan_horizon_s": 7200.0,
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


def optional_metric(results, method, key):
    value = results.get(method, {}).get(key, "")
    return "" if value in ("", None) else float(value)


def summarize_train_log(summary, out_dir):
    train_log = summary.get("train_log") or str(out_dir / "train_log.csv")
    train_log_path = Path(train_log)
    if not train_log_path.exists():
        train_log_path = out_dir / "train_log.csv"
    if not train_log_path.exists():
        return {}

    rows = []
    with open(train_log_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("avg_reward") in ("", None):
                continue
            rows.append(row)
    if not rows:
        return {}

    rewards = [float(row["avg_reward"]) for row in rows]
    last = rows[-1]
    metrics = {
        "best_train_reward": max(rewards),
        "last_train_reward": float(last["avg_reward"]),
    }
    if last.get("avg_dynamic_rate") not in ("", None):
        metrics["last_train_dynamic_rate"] = float(last["avg_dynamic_rate"])
    return metrics


def first_present(*values):
    for value in values:
        if value not in ("", None):
            return value
    return ""


def summarize_run(tag, params, out_dir):
    result_path = out_dir / "comparison_results.json"
    manifest_path = out_dir / "manifest.json"
    results = load_json(result_path)
    manifest = load_json(manifest_path) if manifest_path.exists() else {}

    row = {
        "tag": tag,
        "out_dir": str(out_dir),
        "git_commit": manifest.get("git", {}).get("commit", ""),
        "git_dirty": manifest.get("git", {}).get("dirty", ""),
        "methods": ",".join(results.keys()),
        **params,
    }
    manifest_args = manifest.get("args", {})
    for key in [
        "n_satellites",
        "train_iters",
        "eval_episodes",
        "n_routine",
        "n_dynamic",
        "rollout_steps",
        "ppo_epochs",
        "ppo_batch_size",
        "train_env_workers",
        "eval_workers",
        "torch_num_threads",
        "vtw_time_step_s",
        "vtw_cache_dir",
        "scenario_cache_dir",
        "max_action_dim",
        "candidate_action_top_k",
        "no_viz",
        "no_progress",
        "device",
        "assignment_scorer",
        "assignment_scorer_mix",
        "assignment_context_encoder",
        "assignment_context_weight",
        "assignment_replan_interval_s",
        "assignment_replan_horizon_s",
        "assignment_replan_trigger",
    ]:
        if key in manifest_args:
            row[key] = manifest_args.get(key)

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
    method_prefixes = {
        "Single-PPO": "single",
        "Indep-PPO": "indep",
        "MAPPO": "mappo",
        "Greedy-Oracle": "oracle",
    }
    for method, prefix in method_prefixes.items():
        if method not in results:
            continue
        for key in keys:
            row[f"{prefix}_{key}"] = optional_metric(results, method, key)

    if "MAPPO" in results and "Indep-PPO" in results:
        row["delta_n_scheduled"] = row["mappo_n_scheduled"] - row["indep_n_scheduled"]
        row["delta_success_rate"] = row["mappo_observation_success_rate"] - row["indep_observation_success_rate"]
        row["delta_duplicate_rate"] = row["mappo_duplicate_rate"] - row["indep_duplicate_rate"]
        row["delta_load_balance_cv"] = row["mappo_load_balance_cv"] - row["indep_load_balance_cv"]
        row["delta_avg_off_nadir_deg"] = row["mappo_avg_off_nadir_deg"] - row["indep_avg_off_nadir_deg"]
    if "MAPPO" in results and "Greedy-Oracle" in results:
        row["mappo_oracle_gap_n_scheduled"] = row["oracle_n_scheduled"] - row["mappo_n_scheduled"]
        row["mappo_oracle_relative_completion"] = metric(
            results, "MAPPO", "oracle_relative_completion"
        )
    return row


def summarize_train_run(tag, params, out_dir):
    summary_path = out_dir / "summary.json"
    summary = load_json(summary_path)
    train_metrics = summarize_train_log(summary, out_dir)
    row = {
        "tag": tag,
        "out_dir": str(out_dir),
        **params,
        "best_reward": first_present(summary.get("best_reward")),
        "best_eval_reward": first_present(summary.get("best_eval_reward"), summary.get("best_reward")),
        "has_eval": summary.get("has_eval", ""),
        "best_train_reward": first_present(
            summary.get("best_train_reward"),
            train_metrics.get("best_train_reward"),
        ),
        "last_train_reward": first_present(
            summary.get("last_train_reward"),
            train_metrics.get("last_train_reward"),
        ),
        "last_train_dynamic_rate": first_present(
            summary.get("last_train_dynamic_rate"),
            train_metrics.get("last_train_dynamic_rate"),
        ),
        "num_workers": summary.get("num_workers", ""),
        "meta_batch_size": summary.get("meta_batch_size", ""),
        "inner_steps": summary.get("inner_steps", ""),
        "rollout_steps": summary.get("rollout_steps", ""),
        "ppo_epochs": summary.get("ppo_epochs", ""),
        "ppo_batch_size": summary.get("ppo_batch_size", ""),
        "eval_interval": summary.get("eval_interval", ""),
        "eval_workers": summary.get("eval_workers", ""),
        "vtw_time_step_s": summary.get("vtw_time_step_s", ""),
        "profile_timing": summary.get("profile_timing", ""),
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
        dump_json(rows, f, indent=2, ensure_ascii=False)

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


def find_latest_batch_dir(out_root: Path, batch_prefix: str) -> Optional[Path]:
    """Find the newest existing batch directory matching the generated batch prefix."""
    if not out_root.exists():
        return None
    candidates = [
        p for p in out_root.iterdir()
        if p.is_dir() and p.name.startswith(f"{batch_prefix}_")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p.stat().st_mtime, p.name))


def main():
    parser = argparse.ArgumentParser(description="批量运行 MRL-DMS/compare_methods.py 消融实验")
    parser.add_argument("--preset", type=str, default="assignment_v2",
                        choices=["assignment_v2", "reward_v1", "state_v1", "oracle_v1",
                                 "train_stability_v1", "communication_v1",
                                 "assignment_rolling_v1", "hier_assignment_v1",
                                 "cva_assignment_v1", "candidate_action_v1",
                                 "owner_effect_v1",
                                 "meta_encoder_v1", "learned_assignment_v1"])
    parser.add_argument("--python", type=str, default=sys.executable,
                        help="运行 compare_methods.py 的 Python 解释器")
    parser.add_argument("--out_root", type=str, default="runs/ablation_assignment_v2")
    parser.add_argument("--batch_name", type=str, default=None,
                        help="本批消融实验名称; 默认由 preset/关键参数生成")
    parser.add_argument("--flat_out_root", action="store_true",
                        help="直接写入 --out_root, 不自动创建唯一批次子目录")
    parser.add_argument("--resume_latest", action="store_true",
                        help="复用 --out_root 下匹配当前 batch_name 的最新批次目录, 通常配合 --skip_existing 断点续跑")
    parser.add_argument("--resume_root", type=str, default=None,
                        help="复用指定的已有批次目录, 通常配合 --skip_existing 只补跑缺失子实验")
    parser.add_argument("--acled_path", type=str, default=None)
    parser.add_argument("--n_satellites", type=int, default=6)
    parser.add_argument("--train_iters", type=int, default=30)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--n_routine", type=int, default=200)
    parser.add_argument("--n_dynamic", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_action_dim", type=int, default=None,
                        help="传给 compare_methods.py 的动作空间任务槽位数; "
                             "默认由 compare_methods.py 按任务规模自动扩容")
    parser.add_argument("--candidate_action_top_k", type=int, default=None,
                        help="传给 compare_methods.py 的多星候选动作空间大小; 0=full action")
    parser.add_argument("--candidate_action_top_ks", type=str, default="0,64,128,256",
                        help="candidate_action_v1 使用的候选动作 Top-K 列表; 0 表示 full action")
    parser.add_argument("--methods", type=str, default="mappo",
                        help="传给 compare_methods.py 的方法列表; 默认 mappo, "
                             "避免消融子实验重复训练不变的 Single/Indep baseline。"
                             "完整对比可设为 single,indep,mappo 或 all")
    parser.add_argument("--n_ground_stations", type=int, default=None,
                        help="传给 compare_methods.py 的共享基站数量; None 表示旧口径")
    parser.add_argument("--downlink_time_s", type=float, default=None,
                        help="传给 compare_methods.py 的固定图像下传耗时(秒)")
    parser.add_argument("--assign_w_loads", type=str, default="0.05,0.1,0.2")
    parser.add_argument("--release_windows", type=str, default="0,1800")
    parser.add_argument("--capacity_modes", type=str, default="equal,proportional")
    parser.add_argument("--no_baseline", action="store_true",
                        help="不运行 --no_episode_assignment baseline")
    parser.add_argument("--skip_existing", action="store_true",
                        help="若子实验已有 manifest.json 则跳过")
    parser.add_argument("--jobs", type=int, default=1,
                        help="并行运行多少个子实验; 普通消融训练阶段用它吃多 CPU, 1 为串行")
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
    parser.add_argument("--cva_context_encoders", type=str,
                        default="mlp,lstm,gru,transformer,set_transformer",
                        help="cva_assignment_v1 使用的上下文价值编码器列表")
    parser.add_argument("--cva_scorer_mixes", type=str, default="0.35",
                        help="cva_assignment_v1 中 CVA 分数与启发式分数的混合比例列表")
    parser.add_argument("--cva_context_weight", type=float, default=0.25,
                        help="cva_assignment_v1 中上下文编码器价值项权重")
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
    parser.add_argument("--num_workers", type=int, default=None,
                        help="meta_encoder_v1 透传给 train.py 的并行 worker 数")
    parser.add_argument("--meta_batch_size", type=int, default=None,
                        help="meta_encoder_v1 透传给 train.py 的 meta batch size")
    parser.add_argument("--inner_steps", type=int, default=None,
                        help="meta_encoder_v1 透传给 train.py 的内循环步数")
    parser.add_argument("--rollout_steps", type=int, default=None,
                        help="透传给 train.py/compare_methods.py 的 rollout 长度")
    parser.add_argument("--ppo_epochs", type=int, default=None,
                        help="透传给 train.py/compare_methods.py 的 PPO epoch 数")
    parser.add_argument("--ppo_batch_size", type=int, default=None,
                        help="透传给 train.py/compare_methods.py 的 PPO minibatch 大小")
    parser.add_argument("--train_env_workers", type=int, default=None,
                        help="透传给 compare_methods.py 的 MAPPO 训练 rollout 并行环境进程数")
    parser.add_argument("--eval_interval", type=int, default=None,
                        help="meta_encoder_v1 透传给 train.py 的评估间隔")
    parser.add_argument("--eval_workers", type=int, default=None,
                        help="透传给 train.py/compare_methods.py 的评估 episode 并行 worker 数")
    parser.add_argument("--torch_num_threads", type=int, default=None,
                        help="透传给 compare_methods.py 的单训练进程 PyTorch CPU 线程数")
    parser.add_argument("--save_interval", type=int, default=None,
                        help="meta_encoder_v1 透传给 train.py 的 checkpoint 间隔")
    parser.add_argument("--vtw_time_step_s", type=float, default=None,
                        help="meta_encoder_v1 透传给 train.py 的 VTW 采样步长")
    parser.add_argument("--vtw_cache_dir", type=str, default=None,
                        help="普通消融透传给 compare_methods.py 的 VTW 磁盘缓存目录")
    parser.add_argument("--scenario_cache_dir", type=str, default=None,
                        help="普通消融透传给 compare_methods.py 的预生成场景缓存目录")
    parser.add_argument("--no_profile_timing", action="store_true",
                        help="meta_encoder_v1 关闭 train.py 阶段耗时 profile")
    parser.add_argument("--no_viz", action="store_true",
                        help="普通消融透传给 compare_methods.py, 跳过可视化 JSON 生成")
    parser.add_argument("--no_progress", action="store_true",
                        help="普通消融透传给 compare_methods.py, 关闭 tqdm 进度条")
    args = parser.parse_args()

    if sum(bool(x) for x in [args.flat_out_root, args.resume_latest, args.resume_root]) > 1:
        raise ValueError("--flat_out_root、--resume_latest、--resume_root 只能同时使用一个")

    batch_name = args.batch_name or (
        f"{args.preset}_sat{args.n_satellites}_iter{args.train_iters}_"
        f"eval{args.eval_episodes}_seed{args.seed}"
    )
    batch_prefix = safe_name(batch_name)

    if args.resume_root:
        out_root = Path(args.resume_root)
        if not out_root.exists():
            raise FileNotFoundError(f"--resume_root 指定的目录不存在: {out_root}")
    elif args.resume_latest:
        base_root = Path(args.out_root)
        latest = find_latest_batch_dir(base_root, batch_prefix)
        if latest is None:
            out_root = unique_dir(args.out_root, batch_prefix)
            print(f"未找到可恢复批次, 创建新批次: {out_root}")
        else:
            out_root = latest
            print(f"恢复最新批次: {out_root}")
    elif args.flat_out_root:
        out_root = Path(args.out_root)
        out_root.mkdir(parents=True, exist_ok=True)
    else:
        out_root = unique_dir(args.out_root, batch_prefix)

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
    elif args.preset == "hier_assignment_v1":
        specs = build_hier_assignment_v1_specs()
    elif args.preset == "cva_assignment_v1":
        context_encoders = parse_str_list(args.cva_context_encoders)
        allowed_cva_encoders = {"mlp", "lstm", "gru", "transformer", "set_transformer", "gnn"}
        invalid_cva = [e for e in context_encoders if e not in allowed_cva_encoders]
        if invalid_cva:
            raise ValueError(
                f"未知 CVA context encoder: {invalid_cva}; "
                f"可选: {sorted(allowed_cva_encoders)}"
            )
        specs = build_cva_assignment_v1_specs(
            context_encoders=context_encoders,
            scorer_mixes=parse_float_list(args.cva_scorer_mixes),
            context_weight=args.cva_context_weight,
        )
    elif args.preset == "candidate_action_v1":
        specs = build_candidate_action_v1_specs(
            top_ks=[int(x) for x in parse_float_list(args.candidate_action_top_ks)]
        )
    elif args.preset == "owner_effect_v1":
        specs = build_owner_effect_v1_specs()
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

    jobs = []
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
                "--eval_n_routine", str(args.n_routine),
                "--eval_n_dynamic", str(args.n_dynamic),
                "--log_dir", str(out_root),
                "--exp_name", tag,
                *spec["extra_args"],
            ]
            if not args.full_train:
                cmd.insert(6, "--fast")
            if args.max_action_dim is not None:
                cmd.extend(["--max_action_dim", str(args.max_action_dim)])
            for arg_name in [
                "num_workers",
                "meta_batch_size",
                "inner_steps",
                "rollout_steps",
                "ppo_epochs",
                "ppo_batch_size",
                "eval_interval",
                "eval_workers",
                "save_interval",
                "vtw_time_step_s",
            ]:
                value = getattr(args, arg_name)
                if value is not None:
                    cmd.extend([f"--{arg_name}", str(value)])
            if args.no_profile_timing:
                cmd.append("--no_profile_timing")
            if args.acled_path:
                cmd.extend(["--acled_path", args.acled_path])
        else:
            has_method_override = "--methods" in spec["extra_args"]
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
            ]
            if not has_method_override:
                cmd.extend(["--methods", args.methods])
            cmd.extend(spec["extra_args"])
            if args.max_action_dim is not None:
                cmd.extend(["--max_action_dim", str(args.max_action_dim)])
            for arg_name in [
                "rollout_steps",
                "ppo_epochs",
                "ppo_batch_size",
                "train_env_workers",
                "eval_workers",
                "torch_num_threads",
                "vtw_time_step_s",
                "vtw_cache_dir",
                "scenario_cache_dir",
                "candidate_action_top_k",
                "n_ground_stations",
                "downlink_time_s",
            ]:
                value = getattr(args, arg_name)
                if value is not None:
                    cmd.extend([f"--{arg_name}", str(value)])
            if args.no_viz:
                cmd.append("--no_viz")
            if args.no_progress:
                cmd.append("--no_progress")
            if args.run_oracle and "--run_oracle" not in cmd:
                cmd.append("--run_oracle")
            if args.acled_path:
                cmd.extend(["--acled_path", args.acled_path])

        jobs.append({
            "idx": idx,
            "total": len(specs),
            "tag": tag,
            "spec": spec,
            "out_dir": out_dir,
            "manifest_path": manifest_path,
            "cmd": cmd,
        })

    for job in jobs:
        print(f"[{job['idx']}/{job['total']}] {job['tag']}")
        print(" ".join(job["cmd"]))

    if args.dry_run:
        return

    def run_job(job):
        spec = job["spec"]
        manifest_path = job["manifest_path"]
        if args.skip_existing and manifest_path.exists():
            print(f"skip existing: {manifest_path}")
        else:
            subprocess.run(job["cmd"], cwd=ROOT, check=True)
        if spec.get("script") == "train":
            return summarize_train_run(job["tag"], spec["params"], job["out_dir"])
        else:
            return summarize_run(job["tag"], spec["params"], job["out_dir"])

    rows_by_tag = {}
    max_jobs = max(1, min(args.jobs, len(jobs)))
    if max_jobs == 1:
        for job in jobs:
            row = run_job(job)
            rows_by_tag[job["tag"]] = row
            write_summary(
                [rows_by_tag[j["tag"]] for j in jobs if j["tag"] in rows_by_tag],
                out_root,
            )
    else:
        print(f"并行子实验: jobs={max_jobs}")
        with ThreadPoolExecutor(max_workers=max_jobs) as executor:
            future_to_job = {executor.submit(run_job, job): job for job in jobs}
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                row = future.result()
                rows_by_tag[job["tag"]] = row
                write_summary(
                    [rows_by_tag[j["tag"]] for j in jobs if j["tag"] in rows_by_tag],
                    out_root,
                )

    rows = [rows_by_tag[j["tag"]] for j in jobs if j["tag"] in rows_by_tag]
    json_path, csv_path = write_summary(rows, out_root)
    print(f"summary json: {json_path}")
    print(f"summary csv : {csv_path}")


if __name__ == "__main__":
    main()
