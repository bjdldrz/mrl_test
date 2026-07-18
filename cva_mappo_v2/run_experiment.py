"""
Run CVA-MAPPO v2 as a clean standalone experiment.

Example:
    python -m cva_mappo_v2.run_experiment \
      --acled_path ./DynamicMission/DynamicMission.shp \
      --scenario_cache_dir runs/scenario_cache/cva_stress_sat12_r1200_d300_seed42 \
      --vtw_cache_dir runs/scenario_cache/cva_stress_sat12_r1200_d300_seed42/vtw_cache \
      --n_satellites 12 --train_iters 30 --device cuda:0
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algo.mappo_trainer import MAPPOTrainer, MultiAgentRolloutBuffer
from config import get_default_config
from data.mission_generator import MissionGenerator, load_acled_shapefile
from models.mappo import MAPPOActorCritic
from utils.experiment_dirs import safe_name, unique_dir
from utils.json_utils import dump_json
from utils.scenario_cache import (
    get_eval_scenarios,
    load_scenario_cache,
    scenario_summary,
    select_train_scenario,
)
from utils.experiment_common import (
    _avg_metrics,
    _configure_torch_threads,
    _git_metadata,
    _record_to_dict,
    _torch_state_to_numpy,
    _numpy_state_to_torch,
    expand_satellite_configs,
    make_test_scenarios,
)
try:
    from .config import CandidateSlotConfig, CVAMAPPOV2Config
    from .env import CVAMAPPOV2Env
except ImportError:
    from cva_mappo_v2.config import CandidateSlotConfig, CVAMAPPOV2Config
    from cva_mappo_v2.env import CVAMAPPOV2Env


def _build_v2_config(args) -> CVAMAPPOV2Config:
    triggers = tuple(
        x for x in args.assignment_replan_trigger.split(",")
        if x and x.lower() != "none"
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
        w_quality=args.w_quality,
        w_priority=args.w_priority,
        w_deadline=args.w_deadline,
        w_dynamic=args.w_dynamic,
        w_scarcity=args.w_scarcity,
        w_future_opportunity_loss=args.w_future_opportunity_loss,
        w_load=args.w_load,
        w_owner_stability=args.w_owner_stability,
    )
    cfg.validate()
    return cfg


def _parse_int_list(text: str) -> list:
    values = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not values:
        raise ValueError("列表参数不能为空")
    return values


def _rollout_step_counts(total_steps: int, n_workers: int, split_across_workers: bool) -> list:
    total_steps = max(int(total_steps), 1)
    n_workers = max(1, min(int(n_workers), total_steps))
    if not split_across_workers or n_workers <= 1:
        return [total_steps] * n_workers
    step_counts = [total_steps // n_workers] * n_workers
    for wi in range(total_steps % n_workers):
        step_counts[wi] += 1
    return [steps for steps in step_counts if steps > 0]


def _eval_step_limit(args, env) -> int:
    if getattr(args, "eval_max_steps", 0):
        return max(1, int(args.eval_max_steps))
    return int(env.horizon_s / 10.0) + 100


def _info_valid_action_count(info: dict, key: str, mask_key: str = "action_mask") -> Optional[float]:
    value = info.get(key)
    if value is not None:
        return float(value)
    mask = info.get(mask_key)
    if mask is None:
        return None
    return max(float(np.sum(mask)) - 1.0, 0.0)


def _make_env(cfg, args, v2_cfg: CVAMAPPOV2Config) -> CVAMAPPOV2Env:
    n_sat = min(args.n_satellites, len(cfg.satellites))
    return CVAMAPPOV2Env(
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
    )


def _eval_worker(payload):
    torch.set_num_threads(1)
    cfg = payload["cfg"]
    args = payload["args"]
    v2_cfg = payload["v2_cfg"]
    routine, dynamic = payload["scenario"]
    model_state = payload["model_state"]
    eval_device = torch.device(payload.get("eval_device", "cpu"))

    env = _make_env(cfg, args, v2_cfg)
    model = MAPPOActorCritic(
        local_obs_dim=env.local_obs_dim,
        action_dim=env.action_dim,
        global_state_dim=env.global_state_dim,
        actor_hidden_dims=cfg.network.hidden_layers,
        critic_hidden_dims=cfg.mappo.critic_hidden_dims,
    ).to(eval_device)
    model.load_state_dict(_numpy_state_to_torch(model_state))
    trainer = MAPPOTrainer(
        model,
        lr=cfg.ppo.learning_rate,
        gamma=cfg.ppo.discount_factor,
        gae_lambda=cfg.ppo.gae_lambda,
        clip_ratio=cfg.ppo.clip_ratio,
        entropy_coeff=cfg.ppo.entropy_coeff,
        value_loss_coeff=cfg.ppo.value_loss_coeff,
        ppo_epochs=cfg.ppo.ppo_epochs,
        batch_size=cfg.ppo.batch_size,
        device=str(eval_device),
    )

    env.set_eval_mode(bool(getattr(args, "eval_use_repair", False)))
    reset = env.reset(options={
        "routine_missions": copy.deepcopy(routine),
        "dynamic_schedule": copy.deepcopy(dynamic),
    })
    cur_obs = {aid: item[0] for aid, item in reset.items()}
    cur_info = {aid: item[1] for aid, item in reset.items()}
    max_steps = _eval_step_limit(args, env)
    n_steps = 0
    n_idle_actions = 0
    n_agent_actions = 0
    valid_action_sum = 0.0
    raw_valid_action_sum = 0.0
    for _ in range(max_steps):
        for aid in env.agent_ids:
            valid_count = _info_valid_action_count(
                cur_info[aid],
                "exposed_valid_action_count",
            )
            if valid_count is not None:
                valid_action_sum += valid_count
            raw_valid_count = _info_valid_action_count(
                cur_info[aid],
                "raw_valid_action_count",
            )
            if raw_valid_count is None:
                raw_mask = env._full_action_mask(aid)
                raw_valid_count = max(float(np.sum(raw_mask)) - 1.0, 0.0)
            raw_valid_action_sum += raw_valid_count
        actions = trainer.select_eval_actions(
            env,
            cur_obs,
            cur_info,
            deterministic=payload.get("eval_deterministic", False),
        )
        n_steps += 1
        n_agent_actions += len(actions)
        n_idle_actions += sum(1 for action in actions.values() if int(action) == env.idle_action)
        step_res = env.step(actions)
        for aid, (obs, _, _, _, info) in step_res.items():
            cur_obs[aid] = obs
            cur_info[aid] = info
        if env.is_done():
            break
    metrics = env.get_metrics()
    current_times = [sub_env.current_time_s for sub_env in env.envs.values()]
    metrics.update({
        "eval_steps": float(n_steps),
        "eval_end_time_s": float(np.mean(current_times)) if current_times else 0.0,
        "eval_finished_early": 1.0 if n_steps < max_steps else 0.0,
        "eval_idle_action_rate": n_idle_actions / max(n_agent_actions, 1),
        "eval_avg_valid_action_count": valid_action_sum / max(n_agent_actions, 1),
        "eval_avg_raw_valid_action_count": raw_valid_action_sum / max(n_agent_actions, 1),
    })
    return {"idx": payload["idx"], "metrics": metrics}


def _parallel_eval(cfg, args, v2_cfg, model, scenarios, show_progress=True):
    from multiprocessing import get_context

    n_workers = max(1, min(args.eval_workers, len(scenarios)))
    eval_device = args.eval_device
    if eval_device == "same":
        eval_device = args.device
    if str(eval_device) != "cpu" and n_workers > 1:
        print(
            f"评估设备为 {eval_device}; 单卡 GPU 评估将 eval_workers 从 {n_workers} 降为 1, "
            "避免多个进程抢占同一张 GPU。"
        )
        n_workers = 1
    model_state = _torch_state_to_numpy(model.state_dict())
    payloads = [
        {
            "idx": idx,
            "cfg": copy.deepcopy(cfg),
            "args": args,
            "v2_cfg": copy.deepcopy(v2_cfg),
            "scenario": scenario,
            "model_state": model_state,
            "eval_device": eval_device,
            "eval_deterministic": args.eval_deterministic,
        }
        for idx, scenario in enumerate(scenarios)
    ]
    if n_workers <= 1:
        raw = [
            _eval_worker(payload)
            for payload in tqdm(
                payloads,
                desc="eval CVA-MAPPO-v2",
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
                desc="eval CVA-MAPPO-v2",
                unit="ep",
                dynamic_ncols=True,
                disable=not show_progress,
            ))
    raw = sorted(raw, key=lambda row: row["idx"])
    return _avg_metrics([row["metrics"] for row in raw])


def _save_viz_data(env, out_dir: Path):
    payload = {
        "method": "CVA-MAPPO-v2",
        "missions": [
            {
                "id": int(m.id),
                "lat": float(m.lat),
                "lon": float(m.lon),
                "priority": float(m.priority),
                "duration_s": float(m.duration_s),
                "earliest_time_s": float(m.earliest_time_s),
                "deadline_s": float(m.deadline_s),
                "is_dynamic": bool(m.is_dynamic),
                "arrival_time_s": float(getattr(m, "arrival_time_s", m.earliest_time_s)),
            }
            for m in list(env.envs.values())[0].missions
            if m is not None
        ],
        "schedules": {
            aid: [_record_to_dict(record) for record in sub_env.schedule_log]
            for aid, sub_env in env.envs.items()
        },
        "task_candidate_owners": {
            int(k): list(v) for k, v in env.task_candidate_owners.items()
        },
    }
    with open(out_dir / "cva_mappo_v2_viz_data.json", "w") as f:
        dump_json(payload, f, ensure_ascii=False, indent=2)


def _collect_rollout_worker(payload):
    seed = int(payload["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    cfg = payload["cfg"]
    args = payload["args"]
    v2_cfg = payload["v2_cfg"]
    routine, dynamic = payload["scenario"]
    model_state = payload["model_state"]
    n_steps = int(payload["rollout_steps"])

    env = _make_env(cfg, args, v2_cfg)
    model = MAPPOActorCritic(
        local_obs_dim=env.local_obs_dim,
        action_dim=env.action_dim,
        global_state_dim=env.global_state_dim,
        actor_hidden_dims=cfg.network.hidden_layers,
        critic_hidden_dims=cfg.mappo.critic_hidden_dims,
    ).to("cpu")
    model.load_state_dict(_numpy_state_to_torch(model_state))
    trainer = MAPPOTrainer(
        model,
        lr=cfg.ppo.learning_rate,
        gamma=cfg.ppo.discount_factor,
        gae_lambda=cfg.ppo.gae_lambda,
        clip_ratio=cfg.ppo.clip_ratio,
        entropy_coeff=cfg.ppo.entropy_coeff,
        value_loss_coeff=cfg.ppo.value_loss_coeff,
        ppo_epochs=cfg.ppo.ppo_epochs,
        batch_size=cfg.ppo.batch_size,
        device="cpu",
    )

    reset = env.reset(options={
        "routine_missions": copy.deepcopy(routine),
        "dynamic_schedule": copy.deepcopy(dynamic),
    })
    cur_obs = {aid: item[0] for aid, item in reset.items()}
    cur_info = {aid: item[1] for aid, item in reset.items()}
    buffer = MultiAgentRolloutBuffer()
    buffer.init_agents(env.agent_ids)
    _, _, reward = trainer.collect_rollout(
        env, buffer, n_steps, cur_obs, cur_info
    )
    return {
        "idx": payload["idx"],
        "buffer": buffer,
        "last_global_state": env.get_global_state(),
        "total_reward": float(reward),
        "steps": len(buffer),
    }


def _runtime_plan(cfg, args, v2_cfg, train_payload, eval_scenarios, training_scale_mode: str) -> dict:
    worker_count = 0
    step_counts = []
    if args.train_iters > 0:
        worker_count = max(1, min(int(args.train_env_workers or 1), int(cfg.meta.rollout_steps)))
        step_counts = _rollout_step_counts(
            cfg.meta.rollout_steps,
            worker_count,
            args.split_rollout_steps_across_workers,
        )
        worker_count = len(step_counts)

    if train_payload is not None:
        train_source = "scenario_cache"
    else:
        train_source = "generated"

    task_slots = int(v2_cfg.slots.total_slots)
    transfer_slots = max(int(args.n_satellites) - 1, 0) if args.enable_inter_satellite_transfer else 0
    plan = {
        "train_source": train_source,
        "training_scale_mode": training_scale_mode,
        "routine_pool_sizes": [int(x) for x in cfg.mission.routine_pool_sizes],
        "dynamic_pool_sizes": [int(x) for x in cfg.mission.dynamic_pool_sizes],
        "dynamic_insertions_per_day": int(cfg.mission.dynamic_insertions_per_day),
        "eval_scenarios": int(len(eval_scenarios)),
        "eval_max_steps": int(getattr(args, "eval_max_steps", 0) or 0),
        "eval_use_repair": bool(getattr(args, "eval_use_repair", False)),
        "max_action_dim": int(cfg.mission.max_action_dim),
        "task_slots": task_slots,
        "transfer_slots": transfer_slots,
        "exposed_action_dim": int(task_slots + transfer_slots + 1),
        "train_env_workers": int(worker_count),
        "rollout_steps_arg": int(cfg.meta.rollout_steps),
        "rollout_steps_per_worker": [int(x) for x in step_counts],
        "rollout_steps_total_per_iter": int(sum(step_counts)),
        "rollout_steps_semantics": (
            "total_split_across_workers"
            if args.split_rollout_steps_across_workers
            else "per_worker"
        ),
    }
    return plan


def _print_runtime_plan(plan: dict) -> None:
    print(
        "运行计划: "
        f"train_source={plan['train_source']}, "
        f"training_scale={plan['training_scale_mode']}, "
        f"routine_pool={plan['routine_pool_sizes']}, "
        f"dynamic_pool={plan['dynamic_pool_sizes']}x{plan['dynamic_insertions_per_day']}, "
        f"eval_scenarios={plan['eval_scenarios']}, "
        f"eval_repair={plan['eval_use_repair']}"
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
        "动作空间: "
        f"max_action_dim={plan['max_action_dim']}, "
        f"task_slots={plan['task_slots']}, "
        f"transfer_slots={plan['transfer_slots']}, "
        f"exposed_action_dim={plan['exposed_action_dim']}"
    )
    if plan["eval_max_steps"] > 0:
        print(f"评估上限: eval_max_steps={plan['eval_max_steps']}")


def train_and_eval(cfg, args, v2_cfg, train_payload, eval_scenarios, mission_gen, out_dir: Path):
    device = torch.device(args.device)
    env = _make_env(cfg, args, v2_cfg)
    model = MAPPOActorCritic(
        local_obs_dim=env.local_obs_dim,
        action_dim=env.action_dim,
        global_state_dim=env.global_state_dim,
        actor_hidden_dims=cfg.network.hidden_layers,
        critic_hidden_dims=cfg.mappo.critic_hidden_dims,
    ).to(device)
    trainer = MAPPOTrainer(
        model,
        lr=cfg.ppo.learning_rate,
        gamma=cfg.ppo.discount_factor,
        gae_lambda=cfg.ppo.gae_lambda,
        clip_ratio=cfg.ppo.clip_ratio,
        entropy_coeff=cfg.ppo.entropy_coeff,
        value_loss_coeff=cfg.ppo.value_loss_coeff,
        ppo_epochs=cfg.ppo.ppo_epochs,
        batch_size=cfg.ppo.batch_size,
        device=str(device),
    )

    rng = np.random.RandomState(args.seed + 500)
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
    pbar = tqdm(
        range(args.train_iters),
        desc="train CVA-MAPPO-v2",
        unit="iter",
        dynamic_ncols=True,
        disable=args.no_progress,
    )
    try:
        for it in pbar:
            if max_train_workers > 1:
                model_state = _torch_state_to_numpy(model.state_dict())
                payloads = []
                for wi, worker_steps in enumerate(step_counts):
                    if train_payload is not None:
                        routine, dynamic = select_train_scenario(
                            train_payload, it, args.train_iters, rng
                        )
                    else:
                        routine, dynamic = mission_gen.generate_episode_missions(
                            n_routine=int(rng.choice(cfg.mission.routine_pool_sizes)),
                            n_dynamic_per_insertion=int(rng.choice(cfg.mission.dynamic_pool_sizes)),
                            n_insertions=cfg.mission.dynamic_insertions_per_day,
                        )
                    payloads.append({
                        "idx": wi,
                        "cfg": copy.deepcopy(cfg),
                        "args": args,
                        "v2_cfg": copy.deepcopy(v2_cfg),
                        "scenario": (routine, dynamic),
                        "model_state": model_state,
                        "rollout_steps": worker_steps,
                        "seed": int(rng.randint(0, 2**31 - 1)),
                    })
                worker_results = train_pool.map(_collect_rollout_worker, payloads)
                worker_results = sorted(worker_results, key=lambda row: row["idx"])
                metrics = trainer.update_many(
                    [row["buffer"] for row in worker_results],
                    [row["last_global_state"] for row in worker_results],
                )
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
                reset = env.reset(options={
                    "routine_missions": copy.deepcopy(routine),
                    "dynamic_schedule": copy.deepcopy(dynamic),
                })
                cur_obs = {aid: item[0] for aid, item in reset.items()}
                cur_info = {aid: item[1] for aid, item in reset.items()}
                buffer = MultiAgentRolloutBuffer()
                buffer.init_agents(env.agent_ids)
                _, _, reward = trainer.collect_rollout(
                    env, buffer, cfg.meta.rollout_steps, cur_obs, cur_info
                )
                metrics = trainer.update(buffer, env.get_global_state())
                rollout_steps_done = len(buffer)
            pbar.set_postfix(
                reward=f"{reward:.2f}",
                steps=rollout_steps_done,
                ploss=f"{metrics.get('policy_loss', 0.0):.3f}",
                vloss=f"{metrics.get('value_loss', 0.0):.3f}",
            )
    finally:
        if train_pool is not None:
            train_pool.close()
            train_pool.join()

    avg = _parallel_eval(
        cfg=cfg,
        args=args,
        v2_cfg=v2_cfg,
        model=model,
        scenarios=eval_scenarios,
        show_progress=not args.no_progress,
    )

    if not args.no_viz and eval_scenarios:
        routine, dynamic = eval_scenarios[-1]
        env.set_eval_mode(bool(getattr(args, "eval_use_repair", False)))
        reset = env.reset(options={
            "routine_missions": copy.deepcopy(routine),
            "dynamic_schedule": copy.deepcopy(dynamic),
        })
        cur_obs = {aid: item[0] for aid, item in reset.items()}
        cur_info = {aid: item[1] for aid, item in reset.items()}
        max_steps = _eval_step_limit(args, env)
        for _ in range(max_steps):
            actions = trainer.select_eval_actions(
                env,
                cur_obs,
                cur_info,
                deterministic=args.eval_deterministic,
            )
            step_res = env.step(actions)
            for aid, (obs, _, _, _, info) in step_res.items():
                cur_obs[aid] = obs
                cur_info[aid] = info
            if env.is_done():
                break
        _save_viz_data(env, out_dir)

    return avg


def main():
    parser = argparse.ArgumentParser(description="CVA-MAPPO v2 standalone experiment")
    parser.add_argument("--acled_path", type=str, default=None)
    parser.add_argument("--scenario_cache_dir", type=str, default=None)
    parser.add_argument("--vtw_cache_dir", type=str, default=None)
    parser.add_argument("--n_satellites", type=int, default=12)
    parser.add_argument("--train_iters", type=int, default=30)
    parser.add_argument("--eval_episodes", type=int, default=8)
    parser.add_argument("--eval_workers", type=int, default=4)
    parser.add_argument("--eval_device", type=str, default="same",
                        help="评估设备: cpu 使用多进程 CPU 并行; cuda:0/same 使用单 GPU 串行评估")
    parser.add_argument("--eval_deterministic", dest="eval_deterministic",
                        action="store_true", default=False,
                        help="评估时使用 actor argmax; 默认按 PPO 策略随机采样")
    parser.add_argument("--eval_use_repair", action="store_true",
                        help="评估时启用旧版 eval-only loser reassign/dynamic rescue 后处理")
    parser.add_argument("--eval_stochastic", dest="eval_deterministic",
                        action="store_false",
                        help="评估时按策略分布随机采样动作, 即默认口径")
    parser.add_argument("--eval_max_steps", type=int, default=0,
                        help="每个评估 episode 的最大环境步数; 0 使用 horizon/10+100 的默认防御性上限")
    parser.add_argument("--n_routine", type=int, default=1200)
    parser.add_argument("--n_dynamic", type=int, default=300)
    parser.add_argument("--train_match_eval_scale", action="store_true",
                        help="无场景缓存时直接使用 --n_routine/--n_dynamic 作为训练规模; 当前也是默认行为, 保留用于显式记录")
    parser.add_argument("--curriculum_train_scale", action="store_true",
                        help="无场景缓存时使用默认简单到复杂训练池; 不设置时训练规模默认匹配 --n_routine/--n_dynamic")
    parser.add_argument("--train_routine_pool_sizes", type=str, default=None,
                        help="无场景缓存时覆盖常规任务训练池, 例如 300,600,900,1200")
    parser.add_argument("--train_dynamic_pool_sizes", type=str, default=None,
                        help="无场景缓存时覆盖每次插入动态任务训练池, 例如 75,150,225,300")
    parser.add_argument("--n_ground_stations", type=int, default=0,
                        help="共享基站数量; 0 表示关闭基站下传约束")
    parser.add_argument("--downlink_time_s", type=float, default=0.0,
                        help="每个观测图像固定下传耗时(秒)")
    parser.add_argument("--satellite_storage_capacity", type=int, default=0,
                        help="每颗卫星最多同时存储的未交付图片数; 0 表示不限制")
    parser.add_argument("--enable_inter_satellite_transfer", action="store_true",
                        help="启用智能体显式星间转发动作")
    parser.add_argument("--inter_satellite_transfer_time_s", type=float, default=300.0,
                        help="星间转发固定耗时(秒)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out_dir", type=str, default="runs/cva_mappo_v2")
    parser.add_argument("--run_name", type=str, default="cva_mappo_v2")
    parser.add_argument("--rollout_steps", type=int, default=256)
    parser.add_argument("--train_env_workers", type=int, default=1,
                        help="训练 rollout 并行环境进程数; worker 采样在 CPU, 主进程在 --device 上更新")
    parser.add_argument("--split_rollout_steps_across_workers", dest="split_rollout_steps_across_workers",
                        action="store_true", default=True,
                        help="将 --rollout_steps 作为每轮总采样步数并平分到各 worker; v2 默认开启")
    parser.add_argument("--rollout_steps_per_worker", dest="split_rollout_steps_across_workers",
                        action="store_false",
                        help="旧语义: 每个 worker 都采集完整 --rollout_steps, 总采样步数会乘以 worker 数")
    parser.add_argument("--ppo_epochs", type=int, default=2)
    parser.add_argument("--ppo_batch_size", type=int, default=256)
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
    parser.add_argument("--w_quality", type=float, default=0.42)
    parser.add_argument("--w_priority", type=float, default=0.18)
    parser.add_argument("--w_deadline", type=float, default=0.14)
    parser.add_argument("--w_dynamic", type=float, default=0.18)
    parser.add_argument("--w_scarcity", type=float, default=0.10)
    parser.add_argument("--w_future_opportunity_loss", type=float, default=0.08)
    parser.add_argument("--w_load", type=float, default=0.16)
    parser.add_argument("--w_owner_stability", type=float, default=0.04)
    parser.add_argument("--release_before_deadline_s", type=float, default=3600.0)
    parser.add_argument("--dynamic_broadcast_window_s", type=float, default=3600.0,
                        help="动态任务到达后的短时广播窗口; 窗口内当前可执行卫星可临时接手, 0 表示关闭")
    parser.add_argument("--assignment_replan_interval_s", type=float, default=3600.0)
    parser.add_argument("--assignment_replan_horizon_s", type=float, default=21600.0)
    parser.add_argument("--assignment_replan_trigger", type=str, default="periodic,dynamic,stale_owner,deadline")
    parser.add_argument("--assignment_switch_penalty", type=float, default=0.05)
    parser.add_argument("--owner_switch_margin", type=float, default=0.08,
                        help="新 owner 分数至少超过旧 owner 的额外门槛; 用于降低 owner churn")
    parser.add_argument("--ownership_mask_mode", choices=["soft", "hard"], default="soft",
                        help="soft=CVA-guided Mixed-TopK; hard=原 hard-owner 屏蔽")
    parser.add_argument("--candidate_owner_bonus", type=float, default=0.06,
                        help="soft 模式下候选 owner 的排序加分; 0 表示不使用 owner 软引导")
    parser.add_argument("--slot_selection_mode", choices=["mixed", "typed"], default="typed",
                        help="typed=固定 routine/dynamic/flex 槽位配额; mixed=共享 Top-K 候选池")
    parser.add_argument("--assignment_lock_window_s", type=float, default=600.0)
    parser.add_argument("--assignment_max_switches_per_task", type=int, default=2)
    parser.add_argument("--torch_num_threads", type=int, default=None)
    parser.add_argument("--no_viz", action="store_true")
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
    cfg.train.vtw_time_step_s = args.vtw_time_step_s
    cfg.mission.n_ground_stations = args.n_ground_stations
    cfg.mission.downlink_time_s = args.downlink_time_s
    cfg.mission.satellite_storage_capacity = args.satellite_storage_capacity
    cfg.mission.enable_inter_satellite_transfer = args.enable_inter_satellite_transfer
    cfg.mission.inter_satellite_transfer_time_s = args.inter_satellite_transfer_time_s
    training_scale_mode = "scenario_cache"
    if args.scenario_cache_dir:
        training_scale_mode = "scenario_cache"
    elif (
        args.train_match_eval_scale
        or (
            not args.curriculum_train_scale
            and not args.train_routine_pool_sizes
            and not args.train_dynamic_pool_sizes
        )
    ):
        cfg.mission.routine_pool_sizes = [int(args.n_routine)]
        cfg.mission.dynamic_pool_sizes = [int(args.n_dynamic)]
        training_scale_mode = "match_eval"
    else:
        if args.train_routine_pool_sizes:
            cfg.mission.routine_pool_sizes = _parse_int_list(args.train_routine_pool_sizes)
        if args.train_dynamic_pool_sizes:
            cfg.mission.dynamic_pool_sizes = _parse_int_list(args.train_dynamic_pool_sizes)
        training_scale_mode = (
            "explicit_pool_sizes"
            if args.train_routine_pool_sizes or args.train_dynamic_pool_sizes
            else "default_curriculum"
        )
    required_action_dim = args.n_routine + cfg.mission.dynamic_insertions_per_day * args.n_dynamic
    cfg.mission.max_action_dim = max(cfg.mission.max_action_dim, required_action_dim)

    acled = load_acled_shapefile(args.acled_path) if args.acled_path else None
    mission_gen = MissionGenerator(acled_df=acled, seed=args.seed)
    train_payload = None
    cache_summary = None
    if args.scenario_cache_dir:
        cache = load_scenario_cache(args.scenario_cache_dir)
        train_payload = cache["train"]
        eval_scenarios = get_eval_scenarios(cache["eval"])
        cache_summary = scenario_summary(cache)
        print("场景缓存:", cache_summary)
        if len(eval_scenarios) != args.eval_episodes:
            print(
                "使用场景缓存评估集: "
                f"实际 eval episodes={len(eval_scenarios)}, "
                f"命令行 --eval_episodes={args.eval_episodes} 仅保留为运行记录"
            )
    else:
        eval_scenarios = make_test_scenarios(
            mission_gen,
            args.eval_episodes,
            args.n_routine,
            args.n_dynamic,
            n_insertions=cfg.mission.dynamic_insertions_per_day,
            seed=args.seed + 1000,
        )

    out_dir = unique_dir(args.out_dir, safe_name(args.run_name))
    out_dir.mkdir(parents=True, exist_ok=True)
    v2_cfg = _build_v2_config(args)
    plan = _runtime_plan(
        cfg=cfg,
        args=args,
        v2_cfg=v2_cfg,
        train_payload=train_payload,
        eval_scenarios=eval_scenarios,
        training_scale_mode=training_scale_mode,
    )
    _print_runtime_plan(plan)
    start = time.time()
    results = {
        "CVA-MAPPO-v2": train_and_eval(
            cfg, args, v2_cfg, train_payload, eval_scenarios, mission_gen, out_dir
        )
    }
    elapsed = time.time() - start

    with open(out_dir / "comparison_results.json", "w") as f:
        dump_json(results, f, ensure_ascii=False, indent=2)
    manifest = {
        "schema_version": 1,
        "method": "CVA-MAPPO-v2",
        "elapsed_s": elapsed,
        "args": vars(args),
        "eval_deterministic": bool(args.eval_deterministic),
        "eval_use_repair": bool(args.eval_use_repair),
        "requested_eval_episodes": int(args.eval_episodes),
        "actual_eval_episodes": int(len(eval_scenarios)),
        "scenario_cache_summary": cache_summary,
        "runtime_plan": plan,
        "v2_config": {
            "routine_slots": v2_cfg.slots.routine_slots,
            "dynamic_slots": v2_cfg.slots.dynamic_slots,
            "flex_slots": v2_cfg.slots.flex_slots,
            "routine_candidate_owners": v2_cfg.routine_candidate_owners,
            "dynamic_candidate_owners": v2_cfg.dynamic_candidate_owners,
            "urgent_candidate_owners": v2_cfg.urgent_candidate_owners,
            "stale_candidate_owners": v2_cfg.stale_candidate_owners,
            "capacity_slack_ratio": v2_cfg.capacity_slack_ratio,
            "load_penalty": v2_cfg.load_penalty,
            "switch_penalty": v2_cfg.switch_penalty,
            "owner_switch_margin": v2_cfg.owner_switch_margin,
            "ownership_mask_mode": v2_cfg.ownership_mask_mode,
            "candidate_owner_bonus": v2_cfg.candidate_owner_bonus,
            "slot_selection_mode": v2_cfg.slot_selection_mode,
            "dynamic_broadcast_window_s": v2_cfg.dynamic_broadcast_window_s,
            "score_weights": {
                "w_quality": v2_cfg.w_quality,
                "w_priority": v2_cfg.w_priority,
                "w_deadline": v2_cfg.w_deadline,
                "w_dynamic": v2_cfg.w_dynamic,
                "w_scarcity": v2_cfg.w_scarcity,
                "w_future_opportunity_loss": v2_cfg.w_future_opportunity_loss,
                "w_load": v2_cfg.w_load,
                "w_owner_stability": v2_cfg.w_owner_stability,
            },
            "triggers": list(v2_cfg.triggers),
        },
        "git": _git_metadata(),
        "results": results,
    }
    with open(out_dir / "manifest.json", "w") as f:
        dump_json(manifest, f, ensure_ascii=False, indent=2)
    print(f"结果: {out_dir / 'comparison_results.json'}")


if __name__ == "__main__":
    main()
