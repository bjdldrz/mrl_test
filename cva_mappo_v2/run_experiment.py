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
from compare_methods import (
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
        replan_interval_s=args.assignment_replan_interval_s,
        replan_horizon_s=args.assignment_replan_horizon_s,
        release_before_deadline_s=args.release_before_deadline_s,
        lock_window_s=args.assignment_lock_window_s,
        max_switches_per_task=args.assignment_max_switches_per_task,
        triggers=tuple(x for x in args.assignment_replan_trigger.split(",") if x),
    )
    cfg.validate()
    return cfg


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
    )


def _eval_worker(payload):
    cfg = payload["cfg"]
    args = payload["args"]
    v2_cfg = payload["v2_cfg"]
    routine, dynamic = payload["scenario"]
    model_state = payload["model_state"]

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

    env.set_eval_mode(True)
    reset = env.reset(options={
        "routine_missions": copy.deepcopy(routine),
        "dynamic_schedule": copy.deepcopy(dynamic),
    })
    cur_obs = {aid: item[0] for aid, item in reset.items()}
    cur_info = {aid: item[1] for aid, item in reset.items()}
    max_steps = int(env.horizon_s / 10.0) + 100
    for _ in range(max_steps):
        actions, _, _, _ = trainer.sample_actions(env, cur_obs, cur_info)
        step_res = env.step(actions)
        for aid, (obs, _, _, _, info) in step_res.items():
            cur_obs[aid] = obs
            cur_info[aid] = info
        if env.is_done():
            break
    return {"idx": payload["idx"], "metrics": env.get_metrics()}


def _parallel_eval(cfg, args, v2_cfg, model, scenarios, show_progress=True):
    from multiprocessing import get_context

    n_workers = max(1, min(args.eval_workers, len(scenarios)))
    model_state = _torch_state_to_numpy(model.state_dict())
    payloads = [
        {
            "idx": idx,
            "cfg": copy.deepcopy(cfg),
            "args": args,
            "v2_cfg": copy.deepcopy(v2_cfg),
            "scenario": scenario,
            "model_state": model_state,
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
    pbar = tqdm(
        range(args.train_iters),
        desc="train CVA-MAPPO-v2",
        unit="iter",
        dynamic_ncols=True,
        disable=args.no_progress,
    )
    for it in pbar:
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
        cur_obs, cur_info, reward = trainer.collect_rollout(
            env, buffer, cfg.meta.rollout_steps, cur_obs, cur_info
        )
        metrics = trainer.update(buffer, env.get_global_state())
        pbar.set_postfix(
            reward=f"{reward:.2f}",
            ploss=f"{metrics.get('policy_loss', 0.0):.3f}",
            vloss=f"{metrics.get('value_loss', 0.0):.3f}",
        )

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
        env.set_eval_mode(True)
        reset = env.reset(options={
            "routine_missions": copy.deepcopy(routine),
            "dynamic_schedule": copy.deepcopy(dynamic),
        })
        cur_obs = {aid: item[0] for aid, item in reset.items()}
        cur_info = {aid: item[1] for aid, item in reset.items()}
        max_steps = int(env.horizon_s / 10.0) + 100
        for _ in range(max_steps):
            actions, _, _, _ = trainer.sample_actions(env, cur_obs, cur_info)
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
    parser.add_argument("--n_routine", type=int, default=1200)
    parser.add_argument("--n_dynamic", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out_dir", type=str, default="runs/cva_mappo_v2")
    parser.add_argument("--run_name", type=str, default="cva_mappo_v2")
    parser.add_argument("--rollout_steps", type=int, default=256)
    parser.add_argument("--ppo_epochs", type=int, default=2)
    parser.add_argument("--ppo_batch_size", type=int, default=256)
    parser.add_argument("--vtw_time_step_s", type=float, default=60.0)
    parser.add_argument("--routine_slots", type=int, default=64)
    parser.add_argument("--dynamic_slots", type=int, default=32)
    parser.add_argument("--flex_slots", type=int, default=32)
    parser.add_argument("--routine_candidate_owners", type=int, default=1)
    parser.add_argument("--dynamic_candidate_owners", type=int, default=2)
    parser.add_argument("--urgent_candidate_owners", type=int, default=3)
    parser.add_argument("--stale_candidate_owners", type=int, default=3)
    parser.add_argument("--capacity_slack_ratio", type=float, default=0.05)
    parser.add_argument("--cva_load_penalty", type=float, default=0.15)
    parser.add_argument("--release_before_deadline_s", type=float, default=1800.0)
    parser.add_argument("--assignment_replan_interval_s", type=float, default=3600.0)
    parser.add_argument("--assignment_replan_horizon_s", type=float, default=7200.0)
    parser.add_argument("--assignment_replan_trigger", type=str, default="periodic,dynamic,stale_owner,deadline")
    parser.add_argument("--assignment_switch_penalty", type=float, default=0.05)
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
    required_action_dim = args.n_routine + cfg.mission.dynamic_insertions_per_day * args.n_dynamic
    cfg.mission.max_action_dim = max(cfg.mission.max_action_dim, required_action_dim)

    acled = load_acled_shapefile(args.acled_path) if args.acled_path else None
    mission_gen = MissionGenerator(acled_df=acled, seed=args.seed)
    train_payload = None
    if args.scenario_cache_dir:
        cache = load_scenario_cache(args.scenario_cache_dir)
        train_payload = cache["train"]
        eval_scenarios = get_eval_scenarios(cache["eval"])
        print("场景缓存:", scenario_summary(cache))
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
        "v2_config": {
            "routine_slots": v2_cfg.slots.routine_slots,
            "dynamic_slots": v2_cfg.slots.dynamic_slots,
            "flex_slots": v2_cfg.slots.flex_slots,
            "routine_candidate_owners": v2_cfg.routine_candidate_owners,
            "dynamic_candidate_owners": v2_cfg.dynamic_candidate_owners,
            "urgent_candidate_owners": v2_cfg.urgent_candidate_owners,
            "stale_candidate_owners": v2_cfg.stale_candidate_owners,
        },
        "git": _git_metadata(),
        "results": results,
    }
    with open(out_dir / "manifest.json", "w") as f:
        dump_json(manifest, f, ensure_ascii=False, indent=2)
    print(f"结果: {out_dir / 'comparison_results.json'}")


if __name__ == "__main__":
    main()
