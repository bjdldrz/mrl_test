"""Regression checks for train/eval environment consistency.

These checks are intentionally static so they can run without GPU, pytest-only
fixtures, or scenario data. They guard the policy that eval must not enable
environment-only repair unless the explicit diagnostic flag is supplied.
"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _set_eval_mode_args(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    args: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "set_eval_mode":
            continue
        if not node.args:
            args.append("")
            continue
        args.append(ast.get_source_segment(source, node.args[0]) or "")
    return args


def test_das_eval_mode_defaults_to_train_path() -> None:
    path = ROOT / "das_cva_mappo" / "run_experiment.py"
    args = _set_eval_mode_args(path)
    assert args
    assert all("eval_use_repair" in item for item in args)
    assert all(item.strip() not in {"True", "1"} for item in args)


def test_compat_runner_eval_mode_defaults_to_train_path() -> None:
    path = ROOT / "cva_mappo_v2" / "run_experiment.py"
    args = _set_eval_mode_args(path)
    assert args
    assert all("eval_use_repair" in item for item in args)
    assert all(item.strip() not in {"True", "1"} for item in args)


def test_stage_suite_exposes_diagnostic_repair_flag_only() -> None:
    text = (ROOT / "scripts" / "run_stage_ablation_suite.py").read_text(
        encoding="utf-8"
    )
    assert "--eval_use_repair" in text
    assert '["--eval_use_repair"] if args.eval_use_repair else []' in text


def test_stage_suite_exposes_eval_profile_flag() -> None:
    text = (ROOT / "scripts" / "run_stage_ablation_suite.py").read_text(
        encoding="utf-8"
    )
    assert "--eval_profile" in text
    assert '["--eval_profile"] if args.eval_profile else []' in text


def test_multi_env_uses_low_level_fast_step() -> None:
    text = (ROOT / "envs" / "multi_satellite_env.py").read_text(encoding="utf-8")
    assert "build_observation=False" in text
    assert "check_done=False" in text
    assert "_resolve_actions_with_masks" in text
    assert "_n_fast_idle_resolve_steps" in text
    assert "idle_allowed_actions=idle_allowed_actions" in text
    assert "_idle_allowed_actions" in text


def test_eval_device_defaults_to_cpu() -> None:
    runner = (ROOT / "das_cva_mappo" / "run_experiment.py").read_text(encoding="utf-8")
    suite = (ROOT / "scripts" / "run_stage_ablation_suite.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--eval_device", type=str, default="cpu")' in runner
    assert 'parser.add_argument("--eval_device", default="cpu")' in suite


def test_dynamic_iteration_controls_are_exposed() -> None:
    runner = (ROOT / "das_cva_mappo" / "run_experiment.py").read_text(encoding="utf-8")
    suite = (ROOT / "scripts" / "run_stage_ablation_suite.py").read_text(encoding="utf-8")
    config = (ROOT / "cva_mappo_v2" / "config.py").read_text(encoding="utf-8")
    env = (ROOT / "envs" / "multi_satellite_env.py").read_text(encoding="utf-8")
    v2_env = (ROOT / "cva_mappo_v2" / "env.py").read_text(encoding="utf-8")
    scorer = (ROOT / "cva_mappo_v2" / "scorer.py").read_text(encoding="utf-8")
    das_scorer = (ROOT / "das_cva_mappo" / "candidate_scorer.py").read_text(encoding="utf-8")
    assert "--dynamic_current_slot_bonus" in runner
    assert "--dynamic_window_wait_weight" in runner
    assert "--no_downlink_aware_candidate_score" in runner
    assert "--candidate_downlink_queue_penalty" in runner
    assert "--candidate_dynamic_delivery_delay_penalty" in runner
    assert "--dynamic_downlink_priority" in runner
    assert "--no_dynamic_downlink_priority" in runner
    assert "--no_response_budget_features" in runner
    assert "--no_temporal_window_features" in runner
    assert "--no_early_delivery_temporal_features" in runner
    assert "--temporal_early_delivery_weight" in runner
    assert "--temporal_state_encoder" in runner
    assert "--temporal_state_history_len" in runner
    assert "--candidate_scorer_mix_start" in runner
    assert "--candidate_scorer_mix_end" in runner
    assert "--candidate_scorer_mix_anneal_epochs" in runner
    assert "--dynamic_task_logit_bonus" in runner
    assert "--dynamic_current_logit_bonus" in runner
    assert "--routine_task_logit_penalty" in runner
    assert "--dynamic_select_aux_coeff" in runner
    compat_runner = (ROOT / "cva_mappo_v2" / "run_experiment.py").read_text(encoding="utf-8")
    assert "--candidate_storage_penalty" in compat_runner
    assert "--candidate_dynamic_response_bonus" in compat_runner
    assert "--no_downlink_aware_candidate_score" in compat_runner
    assert "--dynamic_current_slot_bonus" in compat_runner
    assert "--no_response_budget_features" in suite
    assert "--no_temporal_window_features" in suite
    assert "--no_early_delivery_temporal_features" in suite
    assert "--temporal_early_delivery_weight" in suite
    assert "--temporal_state_encoder" in suite
    assert "--candidate_scorer_mix_start" in suite
    assert "--candidate_scorer_mix_end" in suite
    assert "--candidate_scorer_mix_anneal_epochs" in suite
    assert "--dynamic_task_logit_bonus" in suite
    assert "--dynamic_current_logit_bonus" in suite
    assert "--routine_task_logit_penalty" in suite
    assert "--dynamic_select_aux_coeff" in suite
    assert "avg_dynamic_downlink_replan_gain_s" in suite
    assert "abl_stage2_no_response_budget_features" in suite
    assert "abl_stage2_no_temporal_window_features" in suite
    assert "abl_stage2_no_early_delivery_temporal_features" in suite
    assert "cmp_stage2_temporal_early_delivery_features" in suite
    assert "cmp_stage2_temporal_future_features" in suite
    assert "cmp_stage2_temporal_gru_state" in suite
    assert "abl_stage2_no_dynamic_downlink_priority" in suite
    assert "abl_stage2_no_downlink_aware_edge_value" in suite
    assert "abl_stage2_posthoc_dynamic_downlink_priority" in suite
    assert "cmp_v034_gru_no_storage_no_aux_no_idle" in suite
    assert "cmp_v034_mlp_no_storage_no_aux_no_idle" in suite
    assert "cmp_v034_gru_weak_storage_no_aux_no_idle" in suite
    assert "cmp_v034_gru_storage_no_aux_no_idle" in suite
    assert "cmp_v035_gru_weak_storage_no_aux_idle_0p005" in suite
    assert "cmp_v035_gru_weak_storage_no_aux_idle_0p01" in suite
    assert "cmp_v035_gru_weak_storage_no_aux_idle_0p02" in suite
    assert "cmp_v035_gru_weak_storage_no_aux_idle_0p05" in suite
    assert "cmp_v037_dynamic_bias_0p25_current_0p25" in suite
    assert "cmp_v037_dynamic_bias_0p50_current_0p50" in suite
    assert "cmp_v038_dyn_select_aux_0p05" in suite
    assert "dynamic_downlink_priority: bool = False" in config
    assert "downlink_aware_candidate_score: bool = True" in config
    assert "_rebatch_all_downlinks_priority" in env
    assert "get_dynamic_task_diagnostics" in v2_env
    assert "_write_eval_dynamic_task_diagnostics" in runner
    assert "estimated_downlink_queue_s" in scorer
    assert "EDGE_FEATURE_DIM = 28 + TEMPORAL_WINDOW_FEATURE_DIM" in das_scorer
    assert "EdgeDecisionRecord" in (ROOT / "das_cva_mappo" / "action_entities.py").read_text(
        encoding="utf-8"
    )
    assert "test_action_set_actor_permutation_equivariance" in (
        ROOT / "tests" / "test_action_set_actor_equivariance.py"
    ).read_text(encoding="utf-8")


def test_paper_experiment_suite_wraps_stage_suite() -> None:
    text = (ROOT / "scripts" / "run_paper_experiment_suite.py").read_text(
        encoding="utf-8"
    )
    assert "run_stage_ablation_suite.py" in text
    assert '"paper_core"' in text
    assert '"quick_temporal"' in text
    assert '"paper_full"' in text
    assert '"v034_candidate"' in text
    assert '"v035_idle_sweep"' in text
    assert '"v037_dynamic_recovery"' in text
    assert '"v038_dynamic_select_aux"' in text
    assert '"stress_12sat_double_tasks"' in text
    assert '"n_satellites": 12' in text
    assert '"n_routine": 1200' in text
    assert '"n_dynamic": 300' in text
    assert "das_cva_stress_12sat_double_seed42" in text
    assert "paper_experiment_plan.json" in text
    assert "paper_experiment_plan.md" in text
    assert "summary.csv" in text
    assert "summary.md" in text
    assert "cmp_stage2_temporal_early_delivery_features" in text
    assert "cmp_stage2_temporal_future_features" in text
    assert "cmp_stage2_temporal_gru_state" in text
    assert "abl_stage2_no_temporal_window_features" in text
    assert "abl_stage2_no_downlink_aware_edge_value" in text
    assert "cmp_v034_gru_no_storage_no_aux_no_idle" in text
    assert "cmp_v035_gru_weak_storage_no_aux_idle_0p01" in text
    assert "cmp_v037_dynamic_bias_0p50_current_0p50" in text
    assert "cmp_v038_dyn_select_aux_0p05" in text


def test_paper_baseline_suite_runs_fixed_slot_mappo() -> None:
    text = (ROOT / "scripts" / "run_paper_baseline_suite.py").read_text(
        encoding="utf-8"
    )
    assert "cva_mappo_v2.run_experiment" in text
    assert "fixed_slot_mappo_v2_stage1" in text
    assert "fixed_slot_mappo_v2_stage2" in text
    assert "fixed_slot_mappo_v2_stage4_heuristic" in text
    assert "paper_baseline_plan.md" in text
    assert "summary.csv" in text


if __name__ == "__main__":
    test_das_eval_mode_defaults_to_train_path()
    test_compat_runner_eval_mode_defaults_to_train_path()
    test_stage_suite_exposes_diagnostic_repair_flag_only()
    test_stage_suite_exposes_eval_profile_flag()
    test_multi_env_uses_low_level_fast_step()
    test_eval_device_defaults_to_cpu()
    test_dynamic_iteration_controls_are_exposed()
    test_paper_experiment_suite_wraps_stage_suite()
    test_paper_baseline_suite_runs_fixed_slot_mappo()
