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


if __name__ == "__main__":
    test_das_eval_mode_defaults_to_train_path()
    test_compat_runner_eval_mode_defaults_to_train_path()
    test_stage_suite_exposes_diagnostic_repair_flag_only()
    test_stage_suite_exposes_eval_profile_flag()
