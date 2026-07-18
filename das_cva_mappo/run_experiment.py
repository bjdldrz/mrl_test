"""
Run DAS-CVA-MAPPO V0.28.

This runner uses the current CVA-MAPPO v2 environment as the scheduling
compatibility layer, adds a DAS-owned candidate edge scorer, and trains an
action-set-aware MAPPO policy over action entities. V0.26 makes the default
evaluation environment use the same action-resolution path as training. V0.27
adds optional evaluation profiling so slow evaluation runs can be diagnosed.
V0.28 removes repeated low-level obs/mask construction inside multi-agent
environment steps.
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
    _numpy_state_to_torch,
    _torch_state_to_numpy,
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
from .trainer import ActionSetMAPPOTrainer, CandidateAuxSamples


def _parse_int_list(text: str) -> List[int]:
    values = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not values:
        raise ValueError("列表参数不能为空")
    return values


def _rollout_step_counts(total_steps: int, n_workers: int, split_across_workers: bool) -> List[int]:
    total_steps = max(int(total_steps), 1)
    n_workers = max(1, min(int(n_workers), total_steps))
    if not split_across_workers or n_workers <= 1:
        return [total_steps] * n_workers
    step_counts = [total_steps // n_workers] * n_workers
    for worker_idx in range(total_steps % n_workers):
        step_counts[worker_idx] += 1
    return [steps for steps in step_counts if steps > 0]


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
        executable_slot_reserve_ratio=args.executable_slot_reserve_ratio,
        allow_future_task_execution=not args.no_future_task_execution,
        future_task_requires_no_current_valid=(
            args.future_task_requires_no_current_valid
            and not args.future_task_allow_with_current_valid
        ),
        future_task_max_wait_s=args.future_task_max_wait_s,
        future_routine_max_wait_s=args.future_routine_max_wait_s,
        routine_future_dynamic_guard_s=args.routine_future_dynamic_guard_s,
        routine_future_dynamic_penalty=args.routine_future_dynamic_penalty,
        dynamic_future_bonus=args.dynamic_future_bonus,
        drop_ineligible_future_candidates=not args.keep_ineligible_future_candidates,
        replan_interval_s=args.assignment_replan_interval_s,
        replan_horizon_s=args.assignment_replan_horizon_s,
        release_before_deadline_s=args.release_before_deadline_s,
        dynamic_broadcast_window_s=args.dynamic_broadcast_window_s,
        dynamic_takeover_margin_s=args.dynamic_takeover_margin_s,
        lock_window_s=args.assignment_lock_window_s,
        max_switches_per_task=args.assignment_max_switches_per_task,
        w_wait=args.candidate_wait_penalty,
        w_storage_pressure=args.candidate_storage_penalty,
        w_dynamic_urgency=args.candidate_dynamic_urgency_bonus,
        w_dynamic_response=args.candidate_dynamic_response_bonus,
        w_dynamic_wait=args.candidate_dynamic_wait_penalty,
        dynamic_response_target_s=args.dynamic_response_target_s,
        allocator_wait_penalty=args.allocator_wait_penalty,
        allocator_stale_rescue_bonus=args.allocator_stale_rescue_bonus,
        allocator_dynamic_urgency_bonus=args.allocator_dynamic_urgency_bonus,
        allocator_dynamic_response_bonus=args.allocator_dynamic_response_bonus,
        allocator_dynamic_wait_penalty=args.allocator_dynamic_wait_penalty,
        dynamic_rescue_response_bonus=args.dynamic_rescue_response_bonus,
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
        idle_aux_coeff=args.idle_aux_coeff,
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


def _build_action_model(das_cfg: DASConfig, env) -> ActionSetActorCritic:
    return ActionSetActorCritic(
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
    )


def _candidate_scorer_model_state(
    candidate_scorer: Optional[TrainableCandidateValueScorer],
) -> Optional[Dict[str, np.ndarray]]:
    if candidate_scorer is None:
        return None
    return _torch_state_to_numpy(candidate_scorer.model.state_dict())


def _build_worker_candidate_scorer(
    v2_cfg: CVAMAPPOV2Config,
    das_cfg: DASConfig,
    state: Optional[Dict[str, np.ndarray]],
    device: str = "cpu",
) -> Optional[TrainableCandidateValueScorer]:
    if state is None or das_cfg.candidate_scorer_mode == "v2_heuristic":
        return None
    scorer = TrainableCandidateValueScorer(
        v2_cfg,
        mode=das_cfg.candidate_scorer_mode,
        mix=das_cfg.candidate_scorer_mix,
        hidden_dim=das_cfg.candidate_scorer_hidden_dim,
        lr=das_cfg.candidate_scorer_lr,
        device=device,
    )
    scorer.model.load_state_dict(_numpy_state_to_torch(state))
    scorer.model.eval()
    return scorer


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


def _eval_counter() -> Dict[str, int]:
    return {
        "idle_actions": 0,
        "idle_with_valid_actions": 0,
        "idle_without_valid_actions": 0,
        "valid_decision_points": 0,
        "agent_actions": 0,
    }


def _eval_valid_by_agent(env, infos: Dict[str, Dict]) -> Dict[str, bool]:
    valid_by_agent: Dict[str, bool] = {}
    for aid in env.agent_ids:
        mask = np.asarray(infos[aid].get("action_mask", []), dtype=np.float32)
        if mask.size == 0:
            valid_by_agent[aid] = False
            continue
        idle = int(env.idle_action)
        idle_valid = mask[idle] if 0 <= idle < len(mask) else 0.0
        valid_by_agent[aid] = bool(float(np.sum(mask)) - float(idle_valid) > 0)
    return valid_by_agent


def _update_eval_counter(env, actions: Dict[str, int], valid_by_agent: Dict[str, bool], counter: Dict[str, int]) -> None:
    for aid, action in actions.items():
        is_idle = int(action) == env.idle_action
        has_valid = bool(valid_by_agent.get(aid, False))
        counter["valid_decision_points"] += int(has_valid)
        if is_idle:
            counter["idle_actions"] += 1
            if has_valid:
                counter["idle_with_valid_actions"] += 1
            else:
                counter["idle_without_valid_actions"] += 1
    counter["agent_actions"] += len(actions)


EVAL_PROFILE_SECTIONS = (
    "setup",
    "reset",
    "valid_mask",
    "feature_build",
    "actor_forward",
    "counter",
    "env_step",
    "finalize",
)


def _new_eval_profile() -> Dict[str, float]:
    return {
        **{f"{section}_time_s": 0.0 for section in EVAL_PROFILE_SECTIONS},
        "actor_batches": 0.0,
        "feature_batches": 0.0,
        "env_step_calls": 0.0,
    }


def _profile_add(profile: Optional[Dict[str, float]], section: str, elapsed_s: float) -> None:
    if profile is None:
        return
    profile[f"{section}_time_s"] = profile.get(f"{section}_time_s", 0.0) + float(elapsed_s)


def _profile_start(profile: Optional[Dict[str, float]]) -> float:
    return time.perf_counter() if profile is not None else 0.0


def _profile_stop(profile: Optional[Dict[str, float]], section: str, started_s: float) -> None:
    if profile is None:
        return
    _profile_add(profile, section, time.perf_counter() - started_s)


def _profile_incr(profile: Optional[Dict[str, float]], key: str, amount: float = 1.0) -> None:
    if profile is None:
        return
    profile[key] = profile.get(key, 0.0) + float(amount)


def _sync_eval_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        try:
            torch.mps.synchronize()
        except Exception:
            pass


def _merge_eval_profiles(profiles: List[Optional[Dict[str, float]]]) -> Dict[str, float]:
    merged = _new_eval_profile()
    for profile in profiles:
        if not profile:
            continue
        for key, value in profile.items():
            merged[key] = merged.get(key, 0.0) + float(value)
    return merged


def _eval_profile_metrics(
    profile: Dict[str, float],
    wall_time_s: float,
    total_steps: float,
    n_episodes: int,
) -> Dict[str, float]:
    wall_time_s = max(float(wall_time_s), 0.0)
    total_steps = float(total_steps)
    timed_time_s = sum(float(profile.get(f"{section}_time_s", 0.0)) for section in EVAL_PROFILE_SECTIONS)
    metrics: Dict[str, float] = {
        "eval_profile_enabled": 1.0,
        "eval_wall_time_s": wall_time_s,
        "eval_total_steps": total_steps,
        "eval_profile_episodes": float(n_episodes),
        "eval_steps_per_wall_s": total_steps / max(wall_time_s, 1e-9),
        "eval_profile_timed_time_s": timed_time_s,
        "eval_timed_to_wall_ratio": timed_time_s / max(wall_time_s, 1e-9),
        "eval_actor_batches": float(profile.get("actor_batches", 0.0)),
        "eval_feature_batches": float(profile.get("feature_batches", 0.0)),
        "eval_env_step_calls": float(profile.get("env_step_calls", 0.0)),
    }
    for section in EVAL_PROFILE_SECTIONS:
        value = float(profile.get(f"{section}_time_s", 0.0))
        metrics[f"eval_{section}_time_s"] = value
        metrics[f"eval_{section}_share"] = value / max(timed_time_s, 1e-9)
    return metrics


def _finalize_eval_metrics(env, n_steps: int, max_steps: int, counter: Dict[str, int]) -> Dict[str, float]:
    row = env.get_metrics()
    current_times = [sub_env.current_time_s for sub_env in env.envs.values()]
    agent_actions = int(counter.get("agent_actions", 0))
    valid_decision_points = int(counter.get("valid_decision_points", 0))
    idle_actions = int(counter.get("idle_actions", 0))
    idle_with_valid_actions = int(counter.get("idle_with_valid_actions", 0))
    idle_without_valid_actions = int(counter.get("idle_without_valid_actions", 0))
    row["eval_steps"] = float(n_steps)
    row["eval_end_time_s"] = float(np.mean(current_times)) if current_times else 0.0
    row["eval_finished_early"] = 1.0 if n_steps < max_steps else 0.0
    row["eval_idle_action_rate"] = idle_actions / max(agent_actions, 1)
    row["eval_valid_decision_rate"] = valid_decision_points / max(agent_actions, 1)
    row["eval_idle_when_valid_rate"] = idle_with_valid_actions / max(valid_decision_points, 1)
    row["eval_idle_without_valid_rate"] = idle_without_valid_actions / max(agent_actions - valid_decision_points, 1)
    return row


def _merge_candidate_aux_samples(samples: List[CandidateAuxSamples]) -> CandidateAuxSamples:
    edge_rows: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    negative_rows: List[np.ndarray] = []
    negative_anchors: List[np.ndarray] = []
    positive_offset = 0
    n_conflict_edges = 0
    conflict_penalty_sum = 0.0
    load_penalty_sum = 0.0

    for sample in samples:
        n_conflict_edges += int(sample.n_conflict_edges)
        conflict_penalty_sum += float(sample.conflict_penalty_sum)
        load_penalty_sum += float(sample.load_penalty_sum)

        edges = np.asarray(sample.edge_features, dtype=np.float32)
        advantages = np.asarray(sample.advantages, dtype=np.float32).reshape(-1)
        if edges.ndim != 2 or len(edges) == 0 or len(edges) != len(advantages):
            continue

        edge_rows.append(edges)
        targets.append(advantages)

        negatives = np.asarray(sample.negative_features, dtype=np.float32)
        anchors = np.asarray(sample.negative_anchor_indices, dtype=np.int64).reshape(-1)
        if negatives.ndim == 2 and len(negatives) == len(anchors) and len(negatives) > 0:
            negative_rows.append(negatives)
            negative_anchors.append(anchors + positive_offset)
        positive_offset += len(edges)

    if not edge_rows:
        return ActionSetMAPPOTrainer._empty_candidate_aux_samples()

    return CandidateAuxSamples(
        edge_features=np.concatenate(edge_rows, axis=0).astype(np.float32),
        advantages=np.concatenate(targets, axis=0).astype(np.float32),
        negative_features=(
            np.concatenate(negative_rows, axis=0).astype(np.float32)
            if negative_rows else np.zeros((0, edge_rows[0].shape[1]), dtype=np.float32)
        ),
        negative_anchor_indices=(
            np.concatenate(negative_anchors, axis=0).astype(np.int64)
            if negative_anchors else np.zeros(0, dtype=np.int64)
        ),
        n_conflict_edges=int(n_conflict_edges),
        conflict_penalty_sum=float(conflict_penalty_sum),
        load_penalty_sum=float(load_penalty_sum),
    )


def _eval_worker(payload):
    torch.set_num_threads(1)
    profile = _new_eval_profile() if payload.get("eval_profile", False) else None
    setup_started = _profile_start(profile)
    cfg = payload["cfg"]
    args = payload["args"]
    v2_cfg = payload["v2_cfg"]
    das_cfg = payload["das_cfg"]
    routine, dynamic = payload["scenario"]
    eval_device = torch.device(payload.get("eval_device", "cpu"))
    candidate_adapter = _build_candidate_adapter(das_cfg)
    candidate_scorer = _build_worker_candidate_scorer(
        v2_cfg,
        das_cfg,
        payload.get("candidate_scorer_state"),
        device=str(eval_device),
    )

    env = _make_env(cfg, args, v2_cfg, candidate_scorer=candidate_scorer)
    env.set_eval_mode(bool(getattr(args, "eval_use_repair", False)))
    model = _build_action_model(das_cfg, env).to(eval_device)
    model.load_state_dict(_numpy_state_to_torch(payload["model_state"]))
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
        idle_aux_coeff=0.0,
        device=str(eval_device),
    )
    _profile_stop(profile, "setup", setup_started)

    reset_started = _profile_start(profile)
    infos = _reset_infos(env, routine, dynamic)
    _profile_stop(profile, "reset", reset_started)
    counter = _eval_counter()
    n_steps = 0
    max_steps = _eval_step_limit(args, env)
    for _ in range(max_steps):
        valid_started = _profile_start(profile)
        valid_by_agent = _eval_valid_by_agent(env, infos)
        _profile_stop(profile, "valid_mask", valid_started)

        feature_started = _profile_start(profile)
        batch = feature_builder.build_many(env, infos)
        _profile_stop(profile, "feature_build", feature_started)
        _profile_incr(profile, "feature_batches")

        actor_started = _profile_start(profile)
        actions, _, _ = trainer.sample_actions(
            env,
            batch,
            training=False,
            deterministic=payload.get("eval_deterministic", False),
        )
        _profile_stop(profile, "actor_forward", actor_started)
        _profile_incr(profile, "actor_batches")

        counter_started = _profile_start(profile)
        _update_eval_counter(env, actions, valid_by_agent, counter)
        _profile_stop(profile, "counter", counter_started)

        step_started = _profile_start(profile)
        step = env.step(actions)
        infos = {aid: item[4] for aid, item in step.items()}
        _profile_stop(profile, "env_step", step_started)
        _profile_incr(profile, "env_step_calls")
        n_steps += 1
        if env.is_done():
            break
    finalize_started = _profile_start(profile)
    row = _finalize_eval_metrics(env, n_steps, max_steps, counter)
    _profile_stop(profile, "finalize", finalize_started)
    return {"idx": payload["idx"], "metrics": row, "profile": profile}


def _make_eval_runtime(
    cfg,
    args,
    v2_cfg,
    scenario,
    idx: int,
    candidate_scorer: Optional[TrainableCandidateValueScorer],
    profile: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    routine, dynamic = scenario
    env = _make_env(cfg, args, v2_cfg, candidate_scorer=candidate_scorer)
    env.set_eval_mode(bool(getattr(args, "eval_use_repair", False)))
    reset_started = _profile_start(profile)
    infos = _reset_infos(env, routine, dynamic)
    _profile_stop(profile, "reset", reset_started)
    return {
        "idx": int(idx),
        "env": env,
        "infos": infos,
        "counter": _eval_counter(),
        "n_steps": 0,
        "max_steps": _eval_step_limit(args, env),
    }


def _select_batched_eval_actions(
    model,
    feature_builder: ActionSetFeatureBuilder,
    runtimes: List[Dict[str, Any]],
    device: torch.device,
    deterministic: bool = False,
    profile: Optional[Dict[str, float]] = None,
) -> List[Tuple[Dict[str, Any], Dict[str, int], Dict[str, bool]]]:
    rows = []
    for runtime in runtimes:
        env = runtime["env"]
        infos = runtime["infos"]
        valid_started = _profile_start(profile)
        valid_by_agent = _eval_valid_by_agent(env, infos)
        _profile_stop(profile, "valid_mask", valid_started)

        feature_started = _profile_start(profile)
        batch = feature_builder.build_many(env, infos)
        _profile_stop(profile, "feature_build", feature_started)
        _profile_incr(profile, "feature_batches")
        for aid in env.agent_ids:
            item = batch[aid]
            rows.append((
                runtime,
                aid,
                valid_by_agent,
                item.state,
                item.action_features,
                item.action_mask,
            ))

    if not rows:
        return []

    states = np.stack([row[3] for row in rows], axis=0)
    action_features = np.stack([row[4] for row in rows], axis=0)
    masks = np.stack([row[5] for row in rows], axis=0)
    if profile is not None:
        _sync_eval_device(device)
    actor_started = _profile_start(profile)
    with torch.no_grad():
        action_t, _, _ = model.actor.get_action(
            torch.FloatTensor(states).to(device),
            torch.FloatTensor(action_features).to(device),
            torch.FloatTensor(masks).to(device),
            deterministic=deterministic,
        )
    if profile is not None:
        _sync_eval_device(device)
    _profile_stop(profile, "actor_forward", actor_started)
    _profile_incr(profile, "actor_batches")
    actions_np = action_t.cpu().numpy()

    action_by_runtime: Dict[int, Tuple[Dict[str, Any], Dict[str, int], Dict[str, bool]]] = {}
    for row, action in zip(rows, actions_np):
        runtime, aid, valid_by_agent = row[0], row[1], row[2]
        key = id(runtime)
        if key not in action_by_runtime:
            action_by_runtime[key] = (runtime, {}, valid_by_agent)
        action_by_runtime[key][1][aid] = int(action)
    return list(action_by_runtime.values())


def _batched_eval_single_process(
    cfg,
    args,
    v2_cfg,
    das_cfg,
    model_state: Dict[str, np.ndarray],
    scenarios,
    candidate_scorer_state: Optional[Dict[str, np.ndarray]],
    eval_device: str,
    batch_envs: int,
    show_progress: bool = True,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, float]]]:
    device = torch.device(eval_device)
    profile = _new_eval_profile() if getattr(args, "eval_profile", False) else None
    setup_started = _profile_start(profile)
    candidate_adapter = _build_candidate_adapter(das_cfg)
    candidate_scorer = _build_worker_candidate_scorer(
        v2_cfg,
        das_cfg,
        candidate_scorer_state,
        device=str(device),
    )
    _profile_stop(profile, "setup", setup_started)
    runtimes: List[Dict[str, Any]] = []
    pending = list(enumerate(scenarios))
    initial_count = min(max(int(batch_envs), 1), len(pending))
    for _ in range(initial_count):
        idx, scenario = pending.pop(0)
        runtimes.append(_make_eval_runtime(cfg, args, v2_cfg, scenario, idx, candidate_scorer, profile))

    if not runtimes:
        return [], profile

    setup_started = _profile_start(profile)
    model = _build_action_model(das_cfg, runtimes[0]["env"]).to(device)
    model.load_state_dict(_numpy_state_to_torch(model_state))
    model.eval()
    feature_builder = ActionSetFeatureBuilder(
        state_dim=das_cfg.state_dim,
        action_feature_dim=das_cfg.action_feature_dim,
        mode=das_cfg.action_feature_mode,
        use_candidate_score=das_cfg.use_candidate_score_feature,
        candidate_adapter=candidate_adapter,
    )
    _profile_stop(profile, "setup", setup_started)

    raw: List[Dict[str, Any]] = []
    pbar = tqdm(
        total=len(scenarios),
        desc="eval DAS-CVA-MAPPO batch",
        unit="ep",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    try:
        while runtimes:
            selected = _select_batched_eval_actions(
                model,
                feature_builder,
                runtimes,
                device=device,
                deterministic=args.eval_deterministic,
                profile=profile,
            )
            finished: List[Dict[str, Any]] = []
            for runtime, actions, valid_by_agent in selected:
                env = runtime["env"]
                counter_started = _profile_start(profile)
                _update_eval_counter(env, actions, valid_by_agent, runtime["counter"])
                _profile_stop(profile, "counter", counter_started)

                step_started = _profile_start(profile)
                step = env.step(actions)
                runtime["infos"] = {aid: item[4] for aid, item in step.items()}
                _profile_stop(profile, "env_step", step_started)
                _profile_incr(profile, "env_step_calls")
                runtime["n_steps"] += 1
                if env.is_done() or runtime["n_steps"] >= runtime["max_steps"]:
                    finalize_started = _profile_start(profile)
                    row = _finalize_eval_metrics(
                        env,
                        runtime["n_steps"],
                        runtime["max_steps"],
                        runtime["counter"],
                    )
                    _profile_stop(profile, "finalize", finalize_started)
                    raw.append({"idx": runtime["idx"], "metrics": row})
                    finished.append(runtime)
                    pbar.update(1)

            if finished:
                finished_ids = {id(runtime) for runtime in finished}
                runtimes = [runtime for runtime in runtimes if id(runtime) not in finished_ids]
                while pending and len(runtimes) < max(int(batch_envs), 1):
                    idx, scenario = pending.pop(0)
                    runtimes.append(_make_eval_runtime(cfg, args, v2_cfg, scenario, idx, candidate_scorer, profile))
            elif not selected:
                break
    finally:
        pbar.close()
    return raw, profile


def _parallel_eval(
    cfg,
    args,
    v2_cfg,
    das_cfg,
    model,
    scenarios,
    candidate_scorer: Optional[TrainableCandidateValueScorer] = None,
    show_progress=True,
) -> Dict[str, float]:
    from multiprocessing import get_context

    if not scenarios:
        return {}
    n_workers = max(1, min(int(args.eval_workers or 1), len(scenarios)))
    eval_device = args.eval_device if args.eval_device != "same" else args.device
    eval_started = time.perf_counter()

    model_state = _torch_state_to_numpy(model.state_dict())
    candidate_scorer_state = _candidate_scorer_model_state(candidate_scorer)
    if str(eval_device) != "cpu":
        if n_workers > 1 and not args.no_progress:
            print(
                f"评估设备为 {eval_device}; 使用单进程 batched eval 并发 {n_workers} 个环境, "
                "避免多个进程抢占同一张 GPU。"
            )
        raw, profile = _batched_eval_single_process(
            cfg=cfg,
            args=args,
            v2_cfg=v2_cfg,
            das_cfg=das_cfg,
            model_state=model_state,
            scenarios=scenarios,
            candidate_scorer_state=candidate_scorer_state,
            eval_device=eval_device,
            batch_envs=n_workers,
            show_progress=show_progress,
        )
        raw = sorted(raw, key=lambda row: row["idx"])
        metrics = _avg_metrics([row["metrics"] for row in raw])
        if getattr(args, "eval_profile", False):
            total_steps = sum(float(row["metrics"].get("eval_steps", 0.0)) for row in raw)
            metrics.update(_eval_profile_metrics(
                profile or _new_eval_profile(),
                time.perf_counter() - eval_started,
                total_steps,
                len(raw),
            ))
        return metrics

    payloads = [
        {
            "idx": idx,
            "cfg": copy.deepcopy(cfg),
            "args": args,
            "v2_cfg": copy.deepcopy(v2_cfg),
            "das_cfg": copy.deepcopy(das_cfg),
            "scenario": scenario,
            "model_state": model_state,
            "candidate_scorer_state": candidate_scorer_state,
            "eval_device": eval_device,
            "eval_deterministic": args.eval_deterministic,
            "eval_profile": bool(getattr(args, "eval_profile", False)),
        }
        for idx, scenario in enumerate(scenarios)
    ]
    if n_workers <= 1:
        raw = [
            _eval_worker(payload)
            for payload in tqdm(
                payloads,
                desc="eval DAS-CVA-MAPPO",
                unit="ep",
                dynamic_ncols=True,
                disable=not show_progress,
            )
        ]
    else:
        with get_context("spawn").Pool(processes=n_workers) as pool:
            raw = list(tqdm(
                pool.imap_unordered(_eval_worker, payloads),
                total=len(payloads),
                desc="eval DAS-CVA-MAPPO",
                unit="ep",
                dynamic_ncols=True,
                disable=not show_progress,
            ))
    raw = sorted(raw, key=lambda row: row["idx"])
    metrics = _avg_metrics([row["metrics"] for row in raw])
    if getattr(args, "eval_profile", False):
        total_steps = sum(float(row["metrics"].get("eval_steps", 0.0)) for row in raw)
        profile = _merge_eval_profiles([row.get("profile") for row in raw])
        metrics.update(_eval_profile_metrics(
            profile,
            time.perf_counter() - eval_started,
            total_steps,
            len(raw),
        ))
    return metrics


def _write_train_log(out_dir: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with open(out_dir / "train_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _collect_rollout_worker(payload):
    seed = int(payload["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    cfg = payload["cfg"]
    args = payload["args"]
    v2_cfg = payload["v2_cfg"]
    das_cfg = payload["das_cfg"]
    routine, dynamic = payload["scenario"]
    n_steps = int(payload["rollout_steps"])
    candidate_adapter = _build_candidate_adapter(das_cfg)
    candidate_scorer = _build_worker_candidate_scorer(
        v2_cfg,
        das_cfg,
        payload.get("candidate_scorer_state"),
        device="cpu",
    )

    env = _make_env(cfg, args, v2_cfg, candidate_scorer=candidate_scorer)
    feature_builder = ActionSetFeatureBuilder(
        state_dim=das_cfg.state_dim,
        action_feature_dim=das_cfg.action_feature_dim,
        mode=das_cfg.action_feature_mode,
        use_candidate_score=das_cfg.use_candidate_score_feature,
        candidate_scorer=candidate_scorer if das_cfg.candidate_aux_update else None,
        candidate_adapter=candidate_adapter,
    )
    model = _build_action_model(das_cfg, env).to("cpu")
    model.load_state_dict(_numpy_state_to_torch(payload["model_state"]))
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
        idle_aux_coeff=das_cfg.idle_aux_coeff,
        device="cpu",
    )
    infos = _reset_infos(env, routine, dynamic)
    buffer = ActionSetRolloutBuffer()
    buffer.init_agents(env.agent_ids)
    _, reward = trainer.collect_rollout(env, buffer, n_steps, infos)
    return {
        "idx": payload["idx"],
        "buffer": buffer,
        "last_global_state": env.get_global_state(),
        "total_reward": float(reward),
        "steps": len(buffer),
    }


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
    model = _build_action_model(das_cfg, env).to(device)
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
        idle_aux_coeff=das_cfg.idle_aux_coeff,
        device=str(device),
    )

    rng = np.random.RandomState(args.seed + 500)
    logs: List[Dict[str, Any]] = []
    step_counts = _rollout_step_counts(
        cfg.meta.rollout_steps,
        int(args.train_env_workers or 1),
        args.split_rollout_steps_across_workers,
    )
    max_train_workers = len(step_counts) if args.train_iters > 0 else 1
    train_pool = None
    if max_train_workers > 1:
        from multiprocessing import get_context

        train_pool = get_context("spawn").Pool(processes=max_train_workers)
    iterator = tqdm(
        range(args.train_iters),
        desc="train DAS-CVA-MAPPO",
        unit="iter",
        dynamic_ncols=True,
        disable=args.no_progress,
    )
    try:
        for it in iterator:
            if max_train_workers > 1:
                model_state = _torch_state_to_numpy(model.state_dict())
                scorer_state = _candidate_scorer_model_state(candidate_scorer)
                payloads = []
                for worker_idx, worker_steps in enumerate(step_counts):
                    if train_payload is not None:
                        routine, dynamic = select_train_scenario(train_payload, it, args.train_iters, rng)
                    else:
                        routine, dynamic = mission_gen.generate_episode_missions(
                            n_routine=int(rng.choice(cfg.mission.routine_pool_sizes)),
                            n_dynamic_per_insertion=int(rng.choice(cfg.mission.dynamic_pool_sizes)),
                            n_insertions=cfg.mission.dynamic_insertions_per_day,
                        )
                    payloads.append({
                        "idx": worker_idx,
                        "cfg": copy.deepcopy(cfg),
                        "args": args,
                        "v2_cfg": copy.deepcopy(v2_cfg),
                        "das_cfg": copy.deepcopy(das_cfg),
                        "scenario": (routine, dynamic),
                        "model_state": model_state,
                        "candidate_scorer_state": scorer_state,
                        "rollout_steps": worker_steps,
                        "seed": int(rng.randint(0, 2**31 - 1)),
                    })
                worker_results = train_pool.map(_collect_rollout_worker, payloads)
                worker_results = sorted(worker_results, key=lambda row: row["idx"])
                buffers = [row["buffer"] for row in worker_results]
                last_global_states = [row["last_global_state"] for row in worker_results]
                aux_samples = _merge_candidate_aux_samples([
                    trainer.candidate_aux_samples(
                        buffer,
                        last_global_state,
                        max_negatives_per_positive=das_cfg.candidate_hard_negative_samples,
                        valid_negatives_only=das_cfg.candidate_hard_negative_valid_only,
                        conflict_penalty=das_cfg.candidate_aux_conflict_penalty,
                        load_penalty=das_cfg.candidate_aux_load_penalty,
                    )
                    for buffer, last_global_state in zip(buffers, last_global_states)
                ])
                metrics = trainer.update_many(buffers, last_global_states)
                reward = sum(float(row.get("total_reward", 0.0)) for row in worker_results)
                rollout_steps_done = sum(int(row.get("steps", 0)) for row in worker_results)
            else:
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
                rollout_steps_done = len(buffer)

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
                "rollout_steps": int(rollout_steps_done),
                "train_env_workers": int(max_train_workers),
                **{k: float(v) for k, v in metrics.items()},
            }
            logs.append(row)
            iterator.set_postfix(
                reward=f"{reward:.2f}",
                steps=rollout_steps_done,
                ploss=f"{metrics.get('policy_loss', 0.0):.3f}",
                vloss=f"{metrics.get('value_loss', 0.0):.3f}",
            )
    finally:
        if train_pool is not None:
            train_pool.close()
            train_pool.join()
    _write_train_log(out_dir, logs)
    return _parallel_eval(
        cfg=cfg,
        args=args,
        v2_cfg=v2_cfg,
        das_cfg=das_cfg,
        model=model,
        scenarios=eval_scenarios,
        candidate_scorer=candidate_scorer,
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


def _runtime_plan(cfg, args, v2_cfg, train_payload, eval_scenarios) -> Dict[str, Any]:
    step_counts: List[int] = []
    train_workers = 0
    if args.train_iters > 0:
        step_counts = _rollout_step_counts(
            cfg.meta.rollout_steps,
            int(args.train_env_workers or 1),
            args.split_rollout_steps_across_workers,
        )
        train_workers = len(step_counts)

    requested_eval_workers = int(args.eval_workers or 1)
    effective_eval_workers = max(1, min(requested_eval_workers, max(len(eval_scenarios), 1)))
    eval_device = args.eval_device if args.eval_device != "same" else args.device
    eval_execution_mode = "multiprocess" if str(eval_device) == "cpu" else "batched_single_process"

    task_slots = int(v2_cfg.slots.total_slots)
    transfer_slots = max(int(args.n_satellites) - 1, 0) if args.enable_inter_satellite_transfer else 0
    return {
        "train_source": "scenario_cache" if train_payload is not None else "generated",
        "routine_pool_sizes": [int(x) for x in cfg.mission.routine_pool_sizes],
        "dynamic_pool_sizes": [int(x) for x in cfg.mission.dynamic_pool_sizes],
        "dynamic_insertions_per_day": int(cfg.mission.dynamic_insertions_per_day),
        "eval_scenarios": int(len(eval_scenarios)),
        "eval_max_steps": int(args.eval_max_steps or 0),
        "requested_eval_workers": requested_eval_workers,
        "effective_eval_workers": int(effective_eval_workers),
        "eval_device": str(eval_device),
        "eval_execution_mode": eval_execution_mode,
        "eval_use_repair": bool(getattr(args, "eval_use_repair", False)),
        "eval_profile": bool(getattr(args, "eval_profile", False)),
        "max_action_dim": int(cfg.mission.max_action_dim),
        "task_slots": task_slots,
        "transfer_slots": transfer_slots,
        "exposed_action_dim": int(task_slots + transfer_slots + 1),
        "train_env_workers": int(train_workers),
        "rollout_steps_arg": int(cfg.meta.rollout_steps),
        "rollout_steps_per_worker": [int(x) for x in step_counts],
        "rollout_steps_total_per_iter": int(sum(step_counts)),
        "rollout_steps_semantics": (
            "total_split_across_workers"
            if args.split_rollout_steps_across_workers
            else "per_worker"
        ),
    }


def _print_runtime_plan(plan: Dict[str, Any]) -> None:
    print(
        "运行计划: "
        f"train_source={plan['train_source']}, "
        f"routine_pool={plan['routine_pool_sizes']}, "
        f"dynamic_pool={plan['dynamic_pool_sizes']}x{plan['dynamic_insertions_per_day']}, "
        f"eval_scenarios={plan['eval_scenarios']}"
    )
    print(
        "采样计划: "
        f"workers={plan['train_env_workers']}, "
        f"rollout_steps_arg={plan['rollout_steps_arg']}, "
        f"per_worker={plan['rollout_steps_per_worker']}, "
        f"total_per_iter={plan['rollout_steps_total_per_iter']}, "
        f"semantics={plan['rollout_steps_semantics']}"
    )
    print(
        "评估计划: "
        f"requested_workers={plan['requested_eval_workers']}, "
        f"effective_workers={plan['effective_eval_workers']}, "
        f"device={plan['eval_device']}, "
        f"mode={plan['eval_execution_mode']}, "
        f"repair={plan['eval_use_repair']}, "
        f"profile={plan['eval_profile']}, "
        f"eval_max_steps={plan['eval_max_steps']}"
    )
    print(
        "动作空间: "
        f"max_action_dim={plan['max_action_dim']}, "
        f"task_slots={plan['task_slots']}, "
        f"transfer_slots={plan['transfer_slots']}, "
        f"exposed_action_dim={plan['exposed_action_dim']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="DAS-CVA-MAPPO V0.28 experiment")
    parser.add_argument("--acled_path", type=str, default=None)
    parser.add_argument("--scenario_cache_dir", type=str, default=None)
    parser.add_argument("--vtw_cache_dir", type=str, default=None)
    parser.add_argument("--n_satellites", type=int, default=12)
    parser.add_argument("--train_iters", type=int, default=30)
    parser.add_argument("--eval_episodes", type=int, default=8)
    parser.add_argument("--eval_workers", type=int, default=24)
    parser.add_argument("--eval_device", type=str, default="same")
    parser.add_argument("--eval_deterministic", action="store_true", default=False)
    parser.add_argument("--eval_use_repair", action="store_true")
    parser.add_argument("--eval_profile", action="store_true")
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
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out_dir", type=str, default="runs/das_cva_mappo")
    parser.add_argument("--run_name", type=str, default="das_cva_mappo_v0_28")
    parser.add_argument("--rollout_steps", type=int, default=256)
    parser.add_argument("--train_env_workers", type=int, default=16)
    parser.add_argument(
        "--split_rollout_steps_across_workers",
        dest="split_rollout_steps_across_workers",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--rollout_steps_per_worker",
        dest="split_rollout_steps_across_workers",
        action="store_false",
    )
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
    parser.add_argument("--dynamic_takeover_margin_s", type=float, default=300.0)
    parser.add_argument("--candidate_wait_penalty", type=float, default=0.08)
    parser.add_argument("--candidate_storage_penalty", type=float, default=0.08)
    parser.add_argument("--candidate_dynamic_urgency_bonus", type=float, default=0.12)
    parser.add_argument("--candidate_dynamic_response_bonus", type=float, default=0.24)
    parser.add_argument("--candidate_dynamic_wait_penalty", type=float, default=0.20)
    parser.add_argument("--dynamic_response_target_s", type=float, default=3600.0)
    parser.add_argument("--allocator_wait_penalty", type=float, default=0.10)
    parser.add_argument("--allocator_stale_rescue_bonus", type=float, default=0.25)
    parser.add_argument("--allocator_dynamic_urgency_bonus", type=float, default=0.10)
    parser.add_argument("--allocator_dynamic_response_bonus", type=float, default=0.24)
    parser.add_argument("--allocator_dynamic_wait_penalty", type=float, default=0.20)
    parser.add_argument("--dynamic_rescue_response_bonus", type=float, default=1.0)
    parser.add_argument("--assignment_replan_interval_s", type=float, default=3600.0)
    parser.add_argument("--assignment_replan_horizon_s", type=float, default=21600.0)
    parser.add_argument("--assignment_replan_trigger", type=str, default="periodic,dynamic,stale_owner,deadline")
    parser.add_argument("--assignment_switch_penalty", type=float, default=0.05)
    parser.add_argument("--owner_switch_margin", type=float, default=0.08)
    parser.add_argument("--ownership_mask_mode", choices=["soft", "hard"], default="soft")
    parser.add_argument("--candidate_owner_bonus", type=float, default=0.06)
    parser.add_argument("--slot_selection_mode", choices=["mixed", "typed"], default="typed")
    parser.add_argument("--executable_slot_reserve_ratio", type=float, default=0.5)
    parser.add_argument("--no_future_task_execution", action="store_true")
    parser.add_argument("--future_task_requires_no_current_valid", action="store_true")
    parser.add_argument("--future_task_allow_with_current_valid", action="store_true")
    parser.add_argument("--future_task_max_wait_s", type=float, default=600.0)
    parser.add_argument("--future_routine_max_wait_s", type=float, default=180.0)
    parser.add_argument("--routine_future_dynamic_guard_s", type=float, default=1800.0)
    parser.add_argument("--routine_future_dynamic_penalty", type=float, default=0.35)
    parser.add_argument("--dynamic_future_bonus", type=float, default=0.25)
    parser.add_argument("--keep_ineligible_future_candidates", action="store_true")
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
    parser.add_argument("--idle_valid_penalty", type=float, default=0.0)
    parser.add_argument("--idle_aux_coeff", type=float, default=0.05)
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
    runtime_plan = _runtime_plan(cfg, args, v2_cfg, train_payload, eval_scenarios)
    _print_runtime_plan(runtime_plan)

    out_dir = unique_dir(args.out_dir, safe_name(args.run_name))
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    candidate_scorer, candidate_scorer_info = _build_candidate_scorer(
        cfg, args, v2_cfg, das_cfg, train_payload, mission_gen, candidate_adapter
    )

    method_name = "DAS-CVA-MAPPO-v0.28"
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
        "runtime_plan": runtime_plan,
        "das_config": {
            "version": das_cfg.version,
            "matcher": das_cfg.matcher,
            "action_feature_mode": das_cfg.action_feature_mode,
            "use_candidate_score_feature": das_cfg.use_candidate_score_feature,
            "use_set_context": das_cfg.use_set_context,
            "use_action_type_gate": das_cfg.use_action_type_gate,
            "idle_valid_penalty": das_cfg.idle_valid_penalty,
            "idle_aux_coeff": das_cfg.idle_aux_coeff,
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
            "dynamic_takeover_margin_s": v2_cfg.dynamic_takeover_margin_s,
            "ownership_mask_mode": v2_cfg.ownership_mask_mode,
            "slot_selection_mode": v2_cfg.slot_selection_mode,
            "candidate_owner_bonus": v2_cfg.candidate_owner_bonus,
            "executable_slot_reserve_ratio": v2_cfg.executable_slot_reserve_ratio,
            "allow_future_task_execution": v2_cfg.allow_future_task_execution,
            "future_task_requires_no_current_valid": v2_cfg.future_task_requires_no_current_valid,
            "future_task_max_wait_s": v2_cfg.future_task_max_wait_s,
            "future_routine_max_wait_s": v2_cfg.future_routine_max_wait_s,
            "routine_future_dynamic_guard_s": v2_cfg.routine_future_dynamic_guard_s,
            "routine_future_dynamic_penalty": v2_cfg.routine_future_dynamic_penalty,
            "dynamic_future_bonus": v2_cfg.dynamic_future_bonus,
            "drop_ineligible_future_candidates": v2_cfg.drop_ineligible_future_candidates,
            "candidate_wait_penalty": v2_cfg.w_wait,
            "candidate_storage_penalty": v2_cfg.w_storage_pressure,
            "candidate_dynamic_urgency_bonus": v2_cfg.w_dynamic_urgency,
            "candidate_dynamic_response_bonus": v2_cfg.w_dynamic_response,
            "candidate_dynamic_wait_penalty": v2_cfg.w_dynamic_wait,
            "dynamic_response_target_s": v2_cfg.dynamic_response_target_s,
            "allocator_wait_penalty": v2_cfg.allocator_wait_penalty,
            "allocator_stale_rescue_bonus": v2_cfg.allocator_stale_rescue_bonus,
            "allocator_dynamic_urgency_bonus": v2_cfg.allocator_dynamic_urgency_bonus,
            "allocator_dynamic_response_bonus": v2_cfg.allocator_dynamic_response_bonus,
            "allocator_dynamic_wait_penalty": v2_cfg.allocator_dynamic_wait_penalty,
            "dynamic_rescue_response_bonus": v2_cfg.dynamic_rescue_response_bonus,
        },
        "git": _git_metadata(),
        "results": results,
    }
    with open(out_dir / "manifest.json", "w") as f:
        dump_json(manifest, f, ensure_ascii=False, indent=2)
    print(f"结果: {out_dir / 'comparison_results.json'}")


if __name__ == "__main__":
    main()
