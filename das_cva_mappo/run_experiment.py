"""
Run DAS-CVA-MAPPO V0.16.

This runner uses the current CVA-MAPPO v2 environment as the scheduling
compatibility layer, adds a DAS-owned candidate edge scorer, and trains an
action-set-aware MAPPO policy over action entities. V0.16 adds an idle-valid
penalty and diagnostics for idle decisions when executable actions exist.
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import get_default_config
from cva_mappo_v2.config import CandidateSlotConfig, CVAMAPPOV2Config
from cva_mappo_v2.env import CVAMAPPOV2Env
from data.mission_generator import MissionGenerator, load_acled_shapefile
from utils.experiment_common import (
    _avg_metrics,
    _configure_torch_threads,
    _git_metadata,
    expand_satellite_configs,
    make_test_scenarios,
)
from utils.experiment_dirs import safe_name, unique_dir
from utils.json_utils import dump_json
from utils.scenario_cache import (
    flatten_train_scenarios,
    get_eval_scenarios,
    load_scenario_cache,
    scenario_summary,
    select_train_scenario,
)

from .action_set_actor import ActionSetActorCritic
from .candidate_scorer import TrainableCandidateValueScorer
from .config import DASConfig
from .env_adapter import V2CandidateAdapter
from .feature_builder import ActionSetFeatureBuilder
from .rollout_buffer import ActionSetRolloutBuffer
from .trainer import ActionSetMAPPOTrainer


def _parse_int_list(text: str) -> List[int]:
    values = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not values:
        raise ValueError("列表参数不能为空")
    return values


def _build_v2_config(args) -> CVAMAPPOV2Config:
    triggers = tuple(
        item for item in str(args.assignment_replan_trigger).split(",")
        if item and item.lower() != "none"
    )
    cfg = CVAMAPPOV2Config(
        slots=CandidateSlotConfig(
            routine_slots=args.routine_slots,
            dynamic_slots=args.dynamic_slots,
            flex_slots=args.flex_slots,
        ),
        routine_candidate_owners=args.routine_candidate_owners,
        dynamic_candidate_owners=args.dynamic_candidate_owners,
        urgent_candidate_owners=args.urgent_candidate_owners,
        stale_candidate_owners=args.stale_candidate_owners,
        capacity_slack_ratio=args.capacity_slack_ratio,
        load_penalty=args.cva_load_penalty,
        switch_penalty=args.assignment_switch_penalty,
        owner_switch_margin=args.owner_switch_margin,
        ownership_mask_mode=args.ownership_mask_mode,
        candidate_owner_bonus=args.candidate_owner_bonus,
        slot_selection_mode=args.slot_selection_mode,
        replan_interval_s=args.assignment_replan_interval_s,
        replan_horizon_s=args.assignment_replan_horizon_s,
        release_before_deadline_s=args.release_before_deadline_s,
        dynamic_broadcast_window_s=args.dynamic_broadcast_window_s,
        lock_window_s=args.assignment_lock_window_s,
        max_switches_per_task=args.assignment_max_switches_per_task,
        triggers=triggers,
    )
    cfg.validate()
    return cfg


def _build_das_config(args) -> DASConfig:
    use_score = not args.no_candidate_score_feature and args.action_feature_mode != "no_score"
    cfg = DASConfig(
        matcher=args.matcher,
        action_feature_mode=args.action_feature_mode,
        use_candidate_score_feature=use_score,
        use_set_context=not args.no_set_context,
        use_action_type_gate=not args.no_action_type_gate,
        idle_valid_penalty=args.idle_valid_penalty,
        candidate_dropout_prob=args.candidate_dropout_prob,
        candidate_scorer_mode=args.candidate_scorer_mode,
        candidate_scorer_mix=args.candidate_scorer_mix,
        candidate_scorer_hidden_dim=args.candidate_scorer_hidden_dim,
        candidate_scorer_lr=args.candidate_scorer_lr,
        candidate_warmup_edges=args.candidate_warmup_edges,
        candidate_warmup_epochs=args.candidate_warmup_epochs,
        candidate_warmup_batch_size=args.candidate_warmup_batch_size,
        candidate_aux_update=(
            not args.no_candidate_aux_update
            and args.candidate_scorer_mode != "v2_heuristic"
        ),
        candidate_aux_epochs=args.candidate_aux_epochs,
        candidate_aux_batch_size=args.candidate_aux_batch_size,
        candidate_aux_rank_weight=args.candidate_aux_rank_weight,
        candidate_aux_target_clip=args.candidate_aux_target_clip,
        candidate_aux_min_edges=args.candidate_aux_min_edges,
        candidate_hard_negative_samples=args.candidate_hard_negative_samples,
        candidate_hard_negative_valid_only=not args.candidate_hard_negative_include_invalid,
        candidate_hard_negative_margin=args.candidate_hard_negative_margin,
        candidate_hard_negative_value_weight=args.candidate_hard_negative_value_weight,
        candidate_aux_conflict_penalty=args.candidate_aux_conflict_penalty,
        candidate_aux_load_penalty=args.candidate_aux_load_penalty,
        candidate_adapter_mode=args.candidate_adapter_mode,
        actor_hidden_dims=tuple(_parse_int_list(args.actor_hidden_dims)),
        action_hidden_dim=args.action_hidden_dim,
        critic_hidden_dims=tuple(_parse_int_list(args.critic_hidden_dims)),
    )
    cfg.validate()
    return cfg


def _build_candidate_adapter(das_cfg: DASConfig):
    if das_cfg.candidate_adapter_mode == "v2_compat":
        return V2CandidateAdapter()
    raise ValueError(f"unsupported candidate_adapter_mode: {das_cfg.candidate_adapter_mode}")


def _scenario_required_action_dim(scenarios: List[Tuple[list, list]], fallback: int) -> int:
    max_count = int(fallback)
    for routine, dynamic in scenarios:
        count = len(routine)
        for _, missions in dynamic:
            count += len(missions)
        max_count = max(max_count, count)
    return max_count


def _load_or_generate_scenarios(cfg, args):
    acled = load_acled_shapefile(args.acled_path) if args.acled_path else None
    mission_gen = MissionGenerator(acled_df=acled, seed=args.seed)
    train_payload = None
    cache_summary = None
    if args.scenario_cache_dir:
        cache = load_scenario_cache(args.scenario_cache_dir)
        train_payload = cache["train"]
        eval_scenarios = get_eval_scenarios(cache["eval"])
        cache_summary = scenario_summary(cache)
        train_scenarios = flatten_train_scenarios(train_payload)
        all_scenarios = list(train_scenarios) + list(eval_scenarios)
        cfg.mission.max_action_dim = max(
            cfg.mission.max_action_dim,
            _scenario_required_action_dim(all_scenarios, cfg.mission.max_action_dim),
        )
        print("场景缓存:", cache_summary)
    else:
        eval_scenarios = make_test_scenarios(
            mission_gen,
            args.eval_episodes,
            args.n_routine,
            args.n_dynamic,
            n_insertions=cfg.mission.dynamic_insertions_per_day,
            seed=args.seed + 1000,
        )
    return mission_gen, train_payload, eval_scenarios, cache_summary


def _make_env(
    cfg,
    args,
    v2_cfg: CVAMAPPOV2Config,
    candidate_scorer: Optional[TrainableCandidateValueScorer] = None,
) -> CVAMAPPOV2Env:
    n_sat = min(args.n_satellites, len(cfg.satellites))
    env = CVAMAPPOV2Env(
        satellite_configs=cfg.satellites[:n_sat],
        max_action_dim=cfg.mission.max_action_dim,
        reward_config=cfg.reward,
        vtw_time_step_s=cfg.train.vtw_time_step_s,
        coordinate=True,
        reassign_losers=True,
        cva_config=v2_cfg,
        n_ground_stations=args.n_ground_stations,
        downlink_time_s=args.downlink_time_s,
        ground_station_configs=getattr(cfg, "ground_stations", None),
        satellite_storage_capacity=args.satellite_storage_capacity,
        enable_inter_satellite_transfer=args.enable_inter_satellite_transfer,
        inter_satellite_transfer_time_s=args.inter_satellite_transfer_time_s,
        global_state_mode=args.global_state_mode,
        global_state_task_stats=args.global_state_task_stats,
    )
    if candidate_scorer is not None:
        env.scorer = candidate_scorer
    return env


def _reset_infos(env, routine, dynamic) -> Dict[str, Dict]:
    reset = env.reset(options={
        "routine_missions": copy.deepcopy(routine),
        "dynamic_schedule": copy.deepcopy(dynamic),
    })
    return {aid: item[1] for aid, item in reset.items()}


def _eval_step_limit(args, env) -> int:
    if args.eval_max_steps > 0:
        return int(args.eval_max_steps)
    return int(env.horizon_s / 10.0) + 100


def _eval_policy(
    cfg,
    args,
    v2_cfg,
    das_cfg,
    model,
    scenarios,
    candidate_scorer: Optional[TrainableCandidateValueScorer] = None,
    candidate_adapter=None,
    show_progress=True,
) -> Dict[str, float]:
    device = torch.device(args.eval_device if args.eval_device != "same" else args.device)
    if device != next(model.parameters()).device:
        model = copy.deepcopy(model).to(device)
    feature_builder = ActionSetFeatureBuilder(
        state_dim=das_cfg.state_dim,
        action_feature_dim=das_cfg.action_feature_dim,
        mode=das_cfg.action_feature_mode,
        use_candidate_score=das_cfg.use_candidate_score_feature,
        candidate_adapter=candidate_adapter,
    )
    trainer = ActionSetMAPPOTrainer(
        model,
        feature_builder=feature_builder,
        lr=cfg.ppo.learning_rate,
        gamma=cfg.ppo.discount_factor,
        gae_lambda=cfg.ppo.gae_lambda,
        clip_ratio=cfg.ppo.clip_ratio,
        entropy_coeff=cfg.ppo.entropy_coeff,
        value_loss_coeff=cfg.ppo.value_loss_coeff,
        ppo_epochs=cfg.ppo.ppo_epochs,
        batch_size=cfg.ppo.batch_size,
        candidate_dropout_prob=0.0,
        device=str(device),
    )

    metrics = []
    for routine, dynamic in tqdm(
        scenarios,
        desc="eval DAS-CVA-MAPPO",
        unit="ep",
        dynamic_ncols=True,
        disable=not show_progress,
    ):
        env = _make_env(cfg, args, v2_cfg, candidate_scorer=candidate_scorer)
        env.set_eval_mode(True)
        infos = _reset_infos(env, routine, dynamic)
        idle_actions = 0
        idle_with_valid_actions = 0
        idle_without_valid_actions = 0
        valid_decision_points = 0
        agent_actions = 0
        for _ in range(_eval_step_limit(args, env)):
            valid_by_agent = {}
            for aid in env.agent_ids:
                mask = np.asarray(infos[aid].get("action_mask", []), dtype=np.float32)
                if mask.size == 0:
                    valid_by_agent[aid] = False
                    continue
                idle = int(env.idle_action)
                idle_valid = mask[idle] if 0 <= idle < len(mask) else 0.0
                valid_by_agent[aid] = bool(float(np.sum(mask)) - float(idle_valid) > 0)
            actions = trainer.select_eval_actions(
                env,
                infos,
                deterministic=args.eval_deterministic,
            )
            for aid, action in actions.items():
                is_idle = int(action) == env.idle_action
                has_valid = bool(valid_by_agent.get(aid, False))
                valid_decision_points += int(has_valid)
                if is_idle:
                    idle_actions += 1
                    if has_valid:
                        idle_with_valid_actions += 1
                    else:
                        idle_without_valid_actions += 1
            agent_actions += len(actions)
            step = env.step(actions)
            infos = {aid: item[4] for aid, item in step.items()}
            if env.is_done():
                break
        row = env.get_metrics()
        row["eval_idle_action_rate"] = idle_actions / max(agent_actions, 1)
        row["eval_valid_decision_rate"] = valid_decision_points / max(agent_actions, 1)
        row["eval_idle_when_valid_rate"] = idle_with_valid_actions / max(valid_decision_points, 1)
        row["eval_idle_without_valid_rate"] = idle_without_valid_actions / max(agent_actions - valid_decision_points, 1)
        metrics.append(row)
    return _avg_metrics(metrics)


def _write_train_log(out_dir: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with open(out_dir / "train_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def train_and_eval(
    cfg,
    args,
    v2_cfg,
    das_cfg,
    train_payload,
    eval_scenarios,
    mission_gen,
    out_dir: Path,
    candidate_scorer: Optional[TrainableCandidateValueScorer] = None,
    candidate_adapter=None,
):
    device = torch.device(args.device)
    env = _make_env(cfg, args, v2_cfg, candidate_scorer=candidate_scorer)
    feature_builder = ActionSetFeatureBuilder(
        state_dim=das_cfg.state_dim,
        action_feature_dim=das_cfg.action_feature_dim,
        mode=das_cfg.action_feature_mode,
        use_candidate_score=das_cfg.use_candidate_score_feature,
        candidate_scorer=candidate_scorer if das_cfg.candidate_aux_update else None,
        candidate_adapter=candidate_adapter,
    )
    model = ActionSetActorCritic(
        state_dim=das_cfg.state_dim,
        action_feature_dim=das_cfg.action_feature_dim,
        global_state_dim=env.global_state_dim,
        actor_hidden_dims=das_cfg.actor_hidden_dims,
        action_hidden_dim=das_cfg.action_hidden_dim,
        critic_hidden_dims=das_cfg.critic_hidden_dims,
        matcher=das_cfg.matcher,
        use_set_context=das_cfg.use_set_context,
        use_action_type_gate=das_cfg.use_action_type_gate,
        idle_valid_penalty=das_cfg.idle_valid_penalty,
    ).to(device)
    trainer = ActionSetMAPPOTrainer(
        model,
        feature_builder=feature_builder,
        lr=cfg.ppo.learning_rate,
        gamma=cfg.ppo.discount_factor,
        gae_lambda=cfg.ppo.gae_lambda,
        clip_ratio=cfg.ppo.clip_ratio,
        entropy_coeff=cfg.ppo.entropy_coeff,
        value_loss_coeff=cfg.ppo.value_loss_coeff,
        ppo_epochs=cfg.ppo.ppo_epochs,
        batch_size=cfg.ppo.batch_size,
        candidate_dropout_prob=das_cfg.candidate_dropout_prob,
        device=str(device),
    )

    rng = np.random.RandomState(args.seed + 500)
    logs: List[Dict[str, Any]] = []
    iterator = tqdm(
        range(args.train_iters),
        desc="train DAS-CVA-MAPPO",
        unit="iter",
        dynamic_ncols=True,
        disable=args.no_progress,
    )
    for it in iterator:
        if train_payload is not None:
            routine, dynamic = select_train_scenario(train_payload, it, args.train_iters, rng)
        else:
            routine, dynamic = mission_gen.generate_episode_missions(
                n_routine=int(rng.choice(cfg.mission.routine_pool_sizes)),
                n_dynamic_per_insertion=int(rng.choice(cfg.mission.dynamic_pool_sizes)),
                n_insertions=cfg.mission.dynamic_insertions_per_day,
            )
        infos = _reset_infos(env, routine, dynamic)
        buffer = ActionSetRolloutBuffer()
        buffer.init_agents(env.agent_ids)
        _, reward = trainer.collect_rollout(env, buffer, cfg.meta.rollout_steps, infos)
        last_global_state = env.get_global_state()
        aux_samples = trainer.candidate_aux_samples(
            buffer,
            last_global_state,
            max_negatives_per_positive=das_cfg.candidate_hard_negative_samples,
            valid_negatives_only=das_cfg.candidate_hard_negative_valid_only,
            conflict_penalty=das_cfg.candidate_aux_conflict_penalty,
            load_penalty=das_cfg.candidate_aux_load_penalty,
        )
        metrics = trainer.update(buffer, last_global_state)
        if candidate_scorer is not None and das_cfg.candidate_aux_update:
            aux_stats = candidate_scorer.update_from_rollout(
                aux_samples.edge_features,
                aux_samples.advantages,
                negative_features=aux_samples.negative_features,
                negative_anchor_indices=aux_samples.negative_anchor_indices,
                epochs=das_cfg.candidate_aux_epochs,
                batch_size=das_cfg.candidate_aux_batch_size,
                rank_weight=das_cfg.candidate_aux_rank_weight,
                target_clip=das_cfg.candidate_aux_target_clip,
                min_edges=das_cfg.candidate_aux_min_edges,
                negative_margin=das_cfg.candidate_hard_negative_margin,
                negative_value_weight=das_cfg.candidate_hard_negative_value_weight,
            )
            metrics.update({
                "candidate_aux_edges": float(aux_stats.n_edges),
                "candidate_aux_positive_edges": float(aux_stats.n_positive_edges),
                "candidate_aux_negative_edges": float(aux_stats.n_negative_edges),
                "candidate_aux_value_loss": float(aux_stats.value_loss),
                "candidate_aux_rank_loss": float(aux_stats.rank_loss),
                "candidate_aux_total_loss": float(aux_stats.total_loss),
                "candidate_aux_conflict_edges": float(aux_samples.n_conflict_edges),
                "candidate_aux_conflict_penalty_sum": float(aux_samples.conflict_penalty_sum),
                "candidate_aux_load_penalty_sum": float(aux_samples.load_penalty_sum),
            })
        row = {
            "iter": it,
            "reward": float(reward),
            "rollout_steps": len(buffer),
            **{k: float(v) for k, v in metrics.items()},
        }
        logs.append(row)
        iterator.set_postfix(
            reward=f"{reward:.2f}",
            steps=len(buffer),
            ploss=f"{metrics.get('policy_loss', 0.0):.3f}",
            vloss=f"{metrics.get('value_loss', 0.0):.3f}",
        )
    _write_train_log(out_dir, logs)
    return _eval_policy(
        cfg=cfg,
        args=args,
        v2_cfg=v2_cfg,
        das_cfg=das_cfg,
        model=model,
        scenarios=eval_scenarios,
        candidate_scorer=candidate_scorer,
        candidate_adapter=candidate_adapter,
        show_progress=not args.no_progress,
    )


def _generate_candidate_warmup_scenarios(cfg, args, mission_gen) -> List[Tuple[list, list]]:
    if args.candidate_warmup_edges <= 0 or args.candidate_warmup_epochs <= 0:
        return []
    warmup_gen = MissionGenerator(
        acled_df=getattr(mission_gen, "acled_df", None),
        seed=args.seed + 2000,
    )
    rng = np.random.RandomState(args.seed + 2001)
    n_scenarios = max(1, min(16, max(args.train_iters, args.eval_episodes, 1)))
    scenarios: List[Tuple[list, list]] = []
    for _ in range(n_scenarios):
        scenarios.append(
            warmup_gen.generate_episode_missions(
                n_routine=int(rng.choice(cfg.mission.routine_pool_sizes)),
                n_dynamic_per_insertion=int(rng.choice(cfg.mission.dynamic_pool_sizes)),
                n_insertions=cfg.mission.dynamic_insertions_per_day,
            )
        )
    return scenarios


def _build_candidate_scorer(cfg, args, v2_cfg, das_cfg, train_payload, mission_gen, candidate_adapter):
    if das_cfg.candidate_scorer_mode == "v2_heuristic":
        return None, {
            "mode": das_cfg.candidate_scorer_mode,
            "warmup_edges": 0,
            "warmup_loss": 0.0,
            "checkpoint": None,
        }

    scorer = TrainableCandidateValueScorer(
        v2_cfg,
        mode=das_cfg.candidate_scorer_mode,
        mix=das_cfg.candidate_scorer_mix,
        hidden_dim=das_cfg.candidate_scorer_hidden_dim,
        lr=das_cfg.candidate_scorer_lr,
        device=args.device,
    )
    if train_payload is not None:
        warmup_scenarios = flatten_train_scenarios(train_payload)
    else:
        warmup_scenarios = _generate_candidate_warmup_scenarios(cfg, args, mission_gen)

    stats = scorer.warm_start(
        env_factory=lambda: _make_env(cfg, args, v2_cfg),
        scenarios=warmup_scenarios,
        max_edges=das_cfg.candidate_warmup_edges,
        epochs=das_cfg.candidate_warmup_epochs,
        batch_size=das_cfg.candidate_warmup_batch_size,
        candidate_adapter=candidate_adapter,
    )
    if not args.no_progress:
        print(
            "DAS candidate scorer warm-start: "
            f"mode={das_cfg.candidate_scorer_mode}, "
            f"edges={stats.n_edges}, loss={stats.final_loss:.6f}"
        )
    return scorer, {
        "mode": das_cfg.candidate_scorer_mode,
        "mix": das_cfg.candidate_scorer_mix,
        "hidden_dim": das_cfg.candidate_scorer_hidden_dim,
        "lr": das_cfg.candidate_scorer_lr,
        "warmup_edges": int(stats.n_edges),
        "warmup_epochs": das_cfg.candidate_warmup_epochs,
        "warmup_batch_size": das_cfg.candidate_warmup_batch_size,
        "warmup_loss": float(stats.final_loss),
        "checkpoint": None,
    }


def _save_candidate_scorer(
    out_dir: Path,
    candidate_scorer: Optional[TrainableCandidateValueScorer],
    info: Dict[str, Any],
) -> Optional[str]:
    if candidate_scorer is None:
        return None
    path = out_dir / "candidate_scorer.pt"
    torch.save(
        {
            "mode": candidate_scorer.mode,
            "mix": candidate_scorer.mix,
            "warmup_stats": candidate_scorer.warmup_stats.__dict__,
            "last_aux_update_stats": candidate_scorer.last_aux_update_stats.__dict__,
            "aux_update_count": candidate_scorer.aux_update_count,
            "aux_edges_seen": candidate_scorer.aux_edges_seen,
            "model_state_dict": candidate_scorer.model.state_dict(),
            "optimizer_state_dict": candidate_scorer.optimizer.state_dict(),
        },
        path,
    )
    info["checkpoint"] = path.name
    info["aux_update_count"] = int(candidate_scorer.aux_update_count)
    info["aux_edges_seen"] = int(candidate_scorer.aux_edges_seen)
    info["last_aux_update"] = candidate_scorer.last_aux_update_stats.__dict__
    return path.name


def main() -> None:
    parser = argparse.ArgumentParser(description="DAS-CVA-MAPPO V0.16 experiment")
    parser.add_argument("--acled_path", type=str, default=None)
    parser.add_argument("--scenario_cache_dir", type=str, default=None)
    parser.add_argument("--vtw_cache_dir", type=str, default=None)
    parser.add_argument("--n_satellites", type=int, default=12)
    parser.add_argument("--train_iters", type=int, default=30)
    parser.add_argument("--eval_episodes", type=int, default=8)
    parser.add_argument("--eval_device", type=str, default="same")
    parser.add_argument("--eval_deterministic", action="store_true", default=False)
    parser.add_argument("--eval_max_steps", type=int, default=0)
    parser.add_argument("--n_routine", type=int, default=1200)
    parser.add_argument("--n_dynamic", type=int, default=300)
    parser.add_argument("--train_routine_pool_sizes", type=str, default=None)
    parser.add_argument("--train_dynamic_pool_sizes", type=str, default=None)
    parser.add_argument("--n_ground_stations", type=int, default=0)
    parser.add_argument("--downlink_time_s", type=float, default=0.0)
    parser.add_argument("--satellite_storage_capacity", type=int, default=0)
    parser.add_argument("--enable_inter_satellite_transfer", action="store_true")
    parser.add_argument("--inter_satellite_transfer_time_s", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out_dir", type=str, default="runs/das_cva_mappo")
    parser.add_argument("--run_name", type=str, default="das_cva_mappo_v0_16")
    parser.add_argument("--rollout_steps", type=int, default=256)
    parser.add_argument("--ppo_epochs", type=int, default=2)
    parser.add_argument("--ppo_batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--entropy_coeff", type=float, default=None)
    parser.add_argument("--vtw_time_step_s", type=float, default=60.0)
    parser.add_argument("--routine_slots", type=int, default=64)
    parser.add_argument("--dynamic_slots", type=int, default=32)
    parser.add_argument("--flex_slots", type=int, default=32)
    parser.add_argument("--routine_candidate_owners", type=int, default=1)
    parser.add_argument("--dynamic_candidate_owners", type=int, default=6)
    parser.add_argument("--urgent_candidate_owners", type=int, default=6)
    parser.add_argument("--stale_candidate_owners", type=int, default=6)
    parser.add_argument("--capacity_slack_ratio", type=float, default=0.05)
    parser.add_argument("--cva_load_penalty", type=float, default=0.15)
    parser.add_argument("--release_before_deadline_s", type=float, default=3600.0)
    parser.add_argument("--dynamic_broadcast_window_s", type=float, default=3600.0)
    parser.add_argument("--assignment_replan_interval_s", type=float, default=3600.0)
    parser.add_argument("--assignment_replan_horizon_s", type=float, default=21600.0)
    parser.add_argument("--assignment_replan_trigger", type=str, default="periodic,dynamic,stale_owner,deadline")
    parser.add_argument("--assignment_switch_penalty", type=float, default=0.05)
    parser.add_argument("--owner_switch_margin", type=float, default=0.08)
    parser.add_argument("--ownership_mask_mode", choices=["soft", "hard"], default="soft")
    parser.add_argument("--candidate_owner_bonus", type=float, default=0.06)
    parser.add_argument("--slot_selection_mode", choices=["mixed", "typed"], default="typed")
    parser.add_argument("--assignment_lock_window_s", type=float, default=600.0)
    parser.add_argument("--assignment_max_switches_per_task", type=int, default=2)
    parser.add_argument("--global_state_mode", choices=["mean", "concat"], default="mean")
    parser.add_argument("--global_state_task_stats", action="store_true")

    # DAS ablation interface.
    parser.add_argument("--matcher", choices=["additive", "dot", "set_transformer"], default="set_transformer")
    parser.add_argument("--action_feature_mode", choices=["full", "minimal", "no_score"], default="full")
    parser.add_argument("--no_candidate_score_feature", action="store_true")
    parser.add_argument("--no_set_context", action="store_true")
    parser.add_argument("--no_action_type_gate", action="store_true")
    parser.add_argument("--idle_valid_penalty", type=float, default=2.0)
    parser.add_argument("--candidate_dropout_prob", type=float, default=0.0)
    parser.add_argument("--candidate_scorer_mode", choices=["v2_heuristic", "learned", "hybrid"], default="hybrid")
    parser.add_argument("--candidate_scorer_mix", type=float, default=0.35)
    parser.add_argument("--candidate_scorer_hidden_dim", type=int, default=64)
    parser.add_argument("--candidate_scorer_lr", type=float, default=1e-3)
    parser.add_argument("--candidate_warmup_edges", type=int, default=4096)
    parser.add_argument("--candidate_warmup_epochs", type=int, default=2)
    parser.add_argument("--candidate_warmup_batch_size", type=int, default=256)
    parser.add_argument("--no_candidate_aux_update", action="store_true")
    parser.add_argument("--candidate_aux_epochs", type=int, default=1)
    parser.add_argument("--candidate_aux_batch_size", type=int, default=256)
    parser.add_argument("--candidate_aux_rank_weight", type=float, default=0.2)
    parser.add_argument("--candidate_aux_target_clip", type=float, default=3.0)
    parser.add_argument("--candidate_aux_min_edges", type=int, default=4)
    parser.add_argument("--candidate_hard_negative_samples", type=int, default=2)
    parser.add_argument("--candidate_hard_negative_include_invalid", action="store_true")
    parser.add_argument("--candidate_hard_negative_margin", type=float, default=0.25)
    parser.add_argument("--candidate_hard_negative_value_weight", type=float, default=0.5)
    parser.add_argument("--candidate_aux_conflict_penalty", type=float, default=0.5)
    parser.add_argument("--candidate_aux_load_penalty", type=float, default=0.1)
    parser.add_argument("--candidate_adapter_mode", choices=["v2_compat"], default="v2_compat")
    parser.add_argument("--actor_hidden_dims", type=str, default="256,256")
    parser.add_argument("--critic_hidden_dims", type=str, default="256,256")
    parser.add_argument("--action_hidden_dim", type=int, default=128)

    parser.add_argument("--torch_num_threads", type=int, default=None)
    parser.add_argument("--no_progress", action="store_true")
    args = parser.parse_args()

    _configure_torch_threads(args.torch_num_threads)
    if args.vtw_cache_dir:
        os.environ["MRL_DMS_VTW_CACHE_DIR"] = args.vtw_cache_dir
        Path(args.vtw_cache_dir).mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = get_default_config()
    cfg.satellites = expand_satellite_configs(cfg.satellites, args.n_satellites)
    cfg.mappo.n_satellites = args.n_satellites
    cfg.meta.rollout_steps = args.rollout_steps
    cfg.ppo.ppo_epochs = args.ppo_epochs
    cfg.ppo.batch_size = args.ppo_batch_size
    if args.learning_rate is not None:
        cfg.ppo.learning_rate = args.learning_rate
    if args.entropy_coeff is not None:
        cfg.ppo.entropy_coeff = args.entropy_coeff
    cfg.train.vtw_time_step_s = args.vtw_time_step_s
    cfg.mission.n_ground_stations = args.n_ground_stations
    cfg.mission.downlink_time_s = args.downlink_time_s
    cfg.mission.satellite_storage_capacity = args.satellite_storage_capacity
    cfg.mission.enable_inter_satellite_transfer = args.enable_inter_satellite_transfer
    cfg.mission.inter_satellite_transfer_time_s = args.inter_satellite_transfer_time_s
    cfg.mission.routine_pool_sizes = [args.n_routine]
    cfg.mission.dynamic_pool_sizes = [args.n_dynamic]
    if args.train_routine_pool_sizes:
        cfg.mission.routine_pool_sizes = _parse_int_list(args.train_routine_pool_sizes)
    if args.train_dynamic_pool_sizes:
        cfg.mission.dynamic_pool_sizes = _parse_int_list(args.train_dynamic_pool_sizes)
    required_action_dim = args.n_routine + cfg.mission.dynamic_insertions_per_day * args.n_dynamic
    cfg.mission.max_action_dim = max(cfg.mission.max_action_dim, required_action_dim)

    mission_gen, train_payload, eval_scenarios, cache_summary = _load_or_generate_scenarios(cfg, args)
    v2_cfg = _build_v2_config(args)
    das_cfg = _build_das_config(args)
    candidate_adapter = _build_candidate_adapter(das_cfg)

    out_dir = unique_dir(args.out_dir, safe_name(args.run_name))
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    candidate_scorer, candidate_scorer_info = _build_candidate_scorer(
        cfg, args, v2_cfg, das_cfg, train_payload, mission_gen, candidate_adapter
    )

    method_name = "DAS-CVA-MAPPO-v0.16"
    results = {
        method_name: train_and_eval(
            cfg,
            args,
            v2_cfg,
            das_cfg,
            train_payload,
            eval_scenarios,
            mission_gen,
            out_dir,
            candidate_scorer=candidate_scorer,
            candidate_adapter=candidate_adapter,
        )
    }
    _save_candidate_scorer(out_dir, candidate_scorer, candidate_scorer_info)
    elapsed = time.time() - start
    with open(out_dir / "comparison_results.json", "w") as f:
        dump_json(results, f, ensure_ascii=False, indent=2)
    manifest = {
        "schema_version": 1,
        "method": method_name,
        "elapsed_s": elapsed,
        "args": vars(args),
        "scenario_cache_summary": cache_summary,
        "das_config": {
            "version": das_cfg.version,
            "matcher": das_cfg.matcher,
            "action_feature_mode": das_cfg.action_feature_mode,
            "use_candidate_score_feature": das_cfg.use_candidate_score_feature,
            "use_set_context": das_cfg.use_set_context,
            "use_action_type_gate": das_cfg.use_action_type_gate,
            "idle_valid_penalty": das_cfg.idle_valid_penalty,
            "candidate_dropout_prob": das_cfg.candidate_dropout_prob,
            "candidate_scorer_mode": das_cfg.candidate_scorer_mode,
            "candidate_scorer_mix": das_cfg.candidate_scorer_mix,
            "candidate_scorer_hidden_dim": das_cfg.candidate_scorer_hidden_dim,
            "candidate_scorer_lr": das_cfg.candidate_scorer_lr,
            "candidate_warmup_edges": das_cfg.candidate_warmup_edges,
            "candidate_warmup_epochs": das_cfg.candidate_warmup_epochs,
            "candidate_warmup_batch_size": das_cfg.candidate_warmup_batch_size,
            "candidate_aux_update": das_cfg.candidate_aux_update,
            "candidate_aux_epochs": das_cfg.candidate_aux_epochs,
            "candidate_aux_batch_size": das_cfg.candidate_aux_batch_size,
            "candidate_aux_rank_weight": das_cfg.candidate_aux_rank_weight,
            "candidate_aux_target_clip": das_cfg.candidate_aux_target_clip,
            "candidate_aux_min_edges": das_cfg.candidate_aux_min_edges,
            "candidate_hard_negative_samples": das_cfg.candidate_hard_negative_samples,
            "candidate_hard_negative_valid_only": das_cfg.candidate_hard_negative_valid_only,
            "candidate_hard_negative_margin": das_cfg.candidate_hard_negative_margin,
            "candidate_hard_negative_value_weight": das_cfg.candidate_hard_negative_value_weight,
            "candidate_aux_conflict_penalty": das_cfg.candidate_aux_conflict_penalty,
            "candidate_aux_load_penalty": das_cfg.candidate_aux_load_penalty,
            "candidate_adapter_mode": das_cfg.candidate_adapter_mode,
        },
        "candidate_scorer": candidate_scorer_info,
        "candidate_adapter": {
            "mode": das_cfg.candidate_adapter_mode,
        },
        "candidate_layer_config": {
            "routine_slots": v2_cfg.slots.routine_slots,
            "dynamic_slots": v2_cfg.slots.dynamic_slots,
            "flex_slots": v2_cfg.slots.flex_slots,
            "routine_candidate_owners": v2_cfg.routine_candidate_owners,
            "dynamic_candidate_owners": v2_cfg.dynamic_candidate_owners,
            "urgent_candidate_owners": v2_cfg.urgent_candidate_owners,
            "stale_candidate_owners": v2_cfg.stale_candidate_owners,
            "replan_interval_s": v2_cfg.replan_interval_s,
            "replan_horizon_s": v2_cfg.replan_horizon_s,
            "release_before_deadline_s": v2_cfg.release_before_deadline_s,
            "dynamic_broadcast_window_s": v2_cfg.dynamic_broadcast_window_s,
            "ownership_mask_mode": v2_cfg.ownership_mask_mode,
            "slot_selection_mode": v2_cfg.slot_selection_mode,
            "candidate_owner_bonus": v2_cfg.candidate_owner_bonus,
        },
        "git": _git_metadata(),
        "results": results,
    }
    with open(out_dir / "manifest.json", "w") as f:
        dump_json(manifest, f, ensure_ascii=False, indent=2)
    print(f"结果: {out_dir / 'comparison_results.json'}")


if __name__ == "__main__":
    main()
