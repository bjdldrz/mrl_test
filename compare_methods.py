"""
方案对比实验
============
在完全相同的条件下(任务池、种子、训练步数)对比三种调度方案,
重点体现"多星协同(MAPPO)"相对"无协同 baseline"的优势:

  1. single_ppo      : 单星 PPO                 (1 星)
  2. independent_ppo : 多星独立 PPO (无协同)    (N 星, coordinate=False, 各自独立 actor-critic)
  3. mappo           : 多星 MAPPO (协同, 本方法) (N 星, coordinate=True, 集中式 critic)

对比维度参照论文 Fig.7-10:
  - 完成率类: observation/dynamic/routine completion rate (feasible 口径)
  - 累积奖励: total_reward
  - 协同质量: 重复观测率、负载均衡、动态响应延迟、平均观测质量 (多星独有)

用法:
    python compare_methods.py --acled_path ./DynamicMission/DynamicMission.shp \
        --n_satellites 6 --train_iters 30 --eval_episodes 5 --out_dir runs/compare

结果写入 <out_dir>/comparison_results.json, 供 visualize.py 绘图。
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
import copy
import argparse
import logging
import platform
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from config import get_default_config
from config import SatelliteConfig
from data.mission_generator import MissionGenerator, load_acled_shapefile
from utils.experiment_dirs import unique_dir, safe_name
from utils.json_utils import dump_json

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("compare")


# =======================================================================
# 固定测试集: 三种方案在同一批场景上评估, 保证公平
# =======================================================================
def make_test_scenarios(mission_gen, n_episodes, n_routine, n_dynamic,
                        n_insertions=3, seed=123):
    rng = np.random.RandomState(seed)
    scenarios = []
    for _ in range(n_episodes):
        strat = "hotspot" if rng.rand() < 0.5 else "uniform"
        routine, dynamic = mission_gen.generate_episode_missions(
            n_routine=n_routine, n_dynamic_per_insertion=n_dynamic,
            n_insertions=n_insertions, sampling_strategy=strat,
        )
        scenarios.append((routine, dynamic))
    return scenarios


METHOD_ORDER = ["Single-PPO", "Indep-PPO", "MAPPO", "Greedy-Oracle"]
METHOD_ALIASES = {
    "single": "Single-PPO",
    "single_ppo": "Single-PPO",
    "single-ppo": "Single-PPO",
    "indep": "Indep-PPO",
    "independent": "Indep-PPO",
    "indep_ppo": "Indep-PPO",
    "indep-ppo": "Indep-PPO",
    "mappo": "MAPPO",
    "oracle": "Greedy-Oracle",
    "greedy_oracle": "Greedy-Oracle",
    "greedy-oracle": "Greedy-Oracle",
}


def parse_methods(text, run_oracle=False):
    """Parse a comma-separated method list while preserving the canonical order."""
    tokens = [t.strip().lower() for t in text.split(",") if t.strip()]
    if not tokens:
        tokens = ["single", "indep", "mappo"]

    requested = []
    for token in tokens:
        if token == "all":
            requested.extend(METHOD_ORDER)
            continue
        if token not in METHOD_ALIASES:
            valid = sorted([*METHOD_ALIASES.keys(), "all"])
            raise ValueError(f"未知 --methods 项: {token}; 可选: {valid}")
        requested.append(METHOD_ALIASES[token])

    if run_oracle:
        requested.append("Greedy-Oracle")

    dedup = []
    for name in METHOD_ORDER:
        if name in requested and name not in dedup:
            dedup.append(name)
    if not dedup:
        raise ValueError("--methods 至少需要包含一个方法")
    return dedup


def _configure_torch_threads(torch_num_threads):
    if torch_num_threads is None:
        return
    n_threads = max(1, int(torch_num_threads))
    torch.set_num_threads(n_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch 只允许在并行工作启动前设置 interop 线程;重复设置时忽略。
        pass
    logger.info("PyTorch CPU threads: intra_op=%s, inter_op=1", n_threads)


def expand_satellite_configs(base_satellites, n_satellites):
    """Repeat the base constellation with RAAN/phase offsets for scale tests."""
    if n_satellites <= len(base_satellites):
        return copy.deepcopy(base_satellites[:n_satellites])

    satellites = []
    base_n = len(base_satellites)
    for idx in range(n_satellites):
        base = base_satellites[idx % base_n]
        replica = idx // base_n
        n_replicas = max((n_satellites + base_n - 1) // base_n, 1)
        raan_offset = (replica * 360.0 / n_replicas) % 360.0
        phase_offset = (idx * 360.0 / n_satellites) % 360.0
        satellites.append(SatelliteConfig(
            name=f"{base.name}_r{replica + 1}" if replica else base.name,
            semi_major_axis_km=base.semi_major_axis_km,
            eccentricity=base.eccentricity,
            inclination_deg=base.inclination_deg,
            raan_deg=(base.raan_deg + raan_offset) % 360.0,
            arg_perigee_deg=base.arg_perigee_deg,
            mean_anomaly_deg=(base.mean_anomaly_deg + phase_offset) % 360.0,
            max_roll_deg=base.max_roll_deg,
            fov_deg=base.fov_deg,
            maneuver_speed_deg_s=base.maneuver_speed_deg_s,
        ))
    return satellites


def _avg_metrics(metrics_list):
    """对一批 episode 的指标取平均"""
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    return {k: float(np.mean([m.get(k, 0.0) for m in metrics_list])) for k in keys}


def _run_git(args):
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parent,
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _git_metadata():
    status = _run_git(["status", "--short"])
    return {
        "commit": _run_git(["rev-parse", "--short", "HEAD"]),
        "branch": _run_git(["branch", "--show-current"]),
        "dirty": bool(status),
        "status_short": status.splitlines(),
    }


def _write_manifest(out_dir: Path, args, results, elapsed_s: float):
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": elapsed_s,
        "command": " ".join(sys.argv),
        "args": vars(args),
        "git": _git_metadata(),
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "outputs": {
            "results_json": str(out_dir / "comparison_results.json"),
            "manifest_json": str(out_dir / "manifest.json"),
            "viz_data_json": [
                str(p) for p in sorted(out_dir.glob("*_viz_data.json"))
            ],
        },
        "results": results,
    }
    with open(out_dir / "manifest.json", "w") as f:
        dump_json(manifest, f, indent=2, ensure_ascii=False)


def _mission_to_dict(m):
    return {
        "id": int(m.id),
        "lat": float(m.lat),
        "lon": float(m.lon),
        "priority": float(m.priority),
        "duration_s": float(m.duration_s),
        "earliest_time_s": float(m.earliest_time_s),
        "deadline_s": float(m.deadline_s),
        "is_dynamic": bool(m.is_dynamic),
        "arrival_time_s": float(getattr(m, "arrival_time_s", m.earliest_time_s)),
        "event_type": getattr(m, "event_type", ""),
    }


def _record_to_dict(r):
    return {
        "mission_id": int(r.mission_id),
        "satellite_name": r.satellite_name,
        "obs_start_s": float(r.obs_start_s),
        "obs_end_s": float(r.obs_end_s),
        "reward": float(r.reward),
        "off_nadir_deg": float(r.off_nadir_deg),
        "is_dynamic": bool(r.is_dynamic),
        "earliest_time_s": float(r.earliest_time_s),
    }


def _torch_state_to_numpy(state_dict):
    return {k: v.detach().cpu().numpy() for k, v in state_dict.items()}


def _numpy_state_to_torch(state_dict):
    return {k: torch.from_numpy(v.copy()) for k, v in state_dict.items()}


def _eval_single_worker(args):
    from envs.satellite_env import SatelliteSchedulingEnv
    from models.actor_critic import ActorCritic

    idx = args["idx"]
    cfg = args["cfg"]
    routine, dynamic = args["scenario"]
    model_state = args["model_state"]

    sat = cfg.satellites[0]
    env = SatelliteSchedulingEnv(
        satellite_config=sat,
        max_action_dim=cfg.mission.max_action_dim,
        reward_config=cfg.reward,
        vtw_time_step_s=cfg.train.vtw_time_step_s,
    )
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    model = ActorCritic(
        obs_dim, act_dim, cfg.network.hidden_layers, cfg.network.activation
    ).to("cpu")
    model.load_state_dict(_numpy_state_to_torch(model_state))

    opts = {
        "routine_missions": copy.deepcopy(routine),
        "dynamic_schedule": copy.deepcopy(dynamic),
    }
    obs, info = env.reset(options=opts)
    done = False
    max_steps = int(env.horizon_s / 10.0) + 100
    for _ in range(max_steps):
        if done:
            break
        mask = info.get("action_mask", np.ones(act_dim))
        with torch.no_grad():
            a, _, _, _ = model.get_action_and_value(
                torch.FloatTensor(obs).unsqueeze(0),
                torch.FloatTensor(mask).unsqueeze(0),
            )
        obs, _, term, trunc, info = env.step(a.cpu().item())
        done = term or trunc
    return {"idx": idx, "metrics": env.get_metrics()}


def _eval_multi_worker(args):
    from envs.multi_satellite_env import MultiSatelliteEnv
    from models.mappo import MAPPOActorCritic
    from algo.mappo_trainer import MAPPOTrainer

    idx = args["idx"]
    cfg = args["cfg"]
    routine, dynamic = args["scenario"]
    model_state = args["model_state"]
    env_kwargs = args["env_kwargs"]
    trainer_kwargs = args["trainer_kwargs"]

    n_sat = min(cfg.mappo.n_satellites, len(cfg.satellites))
    env = MultiSatelliteEnv(
        satellite_configs=cfg.satellites[:n_sat],
        max_action_dim=cfg.mission.max_action_dim,
        reward_config=cfg.reward,
        vtw_time_step_s=cfg.train.vtw_time_step_s,
        **env_kwargs,
    )
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
        **trainer_kwargs,
    )

    env.set_eval_mode(True)
    opts = {
        "routine_missions": copy.deepcopy(routine),
        "dynamic_schedule": copy.deepcopy(dynamic),
    }
    res = env.reset(options=opts)
    cur_obs = {a: r[0] for a, r in res.items()}
    cur_info = {a: r[1] for a, r in res.items()}
    max_steps = int(env.horizon_s / 10.0) + 100
    for _ in range(max_steps):
        actions, _, _, _ = trainer.sample_actions(
            multi_env=env,
            current_obs=cur_obs,
            current_infos=cur_info,
            intent_broadcast=args["intent_broadcast"],
            intent_replan_rounds=args["intent_replan_rounds"],
        )
        step_res = env.step(actions)
        for aid, (obs, _, _, _, info) in step_res.items():
            cur_obs[aid] = obs
            cur_info[aid] = info
        if env.is_done():
            break
    return {"idx": idx, "metrics": env.get_metrics()}


def _collect_multi_rollout_worker(args):
    from envs.multi_satellite_env import MultiSatelliteEnv
    from models.mappo import MAPPOActorCritic
    from algo.mappo_trainer import MAPPOTrainer, MultiAgentRolloutBuffer

    seed = int(args["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    cfg = args["cfg"]
    routine, dynamic = args["scenario"]
    model_state = args["model_state"]
    env_kwargs = args["env_kwargs"]
    trainer_kwargs = args["trainer_kwargs"]
    n_steps = int(args["rollout_steps"])
    active_agent_ids = args.get("active_agent_ids")

    n_sat = min(cfg.mappo.n_satellites, len(cfg.satellites))
    env = MultiSatelliteEnv(
        satellite_configs=cfg.satellites[:n_sat],
        max_action_dim=cfg.mission.max_action_dim,
        reward_config=cfg.reward,
        vtw_time_step_s=cfg.train.vtw_time_step_s,
        **env_kwargs,
    )
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
        **trainer_kwargs,
    )

    opts = {
        "routine_missions": copy.deepcopy(routine),
        "dynamic_schedule": copy.deepcopy(dynamic),
    }
    res = env.reset(options=opts)
    cur_obs = {a: r[0] for a, r in res.items()}
    cur_info = {a: r[1] for a, r in res.items()}
    buf = MultiAgentRolloutBuffer()
    buf.init_agents(active_agent_ids or env.agent_ids)
    cur_obs, cur_info, total_reward = trainer.collect_rollout(
        env,
        buf,
        n_steps,
        cur_obs,
        cur_info,
        active_agent_ids=active_agent_ids,
        joint_explore_prob=args["joint_explore_prob"],
        intent_broadcast=args["intent_broadcast"],
        intent_replan_rounds=args["intent_replan_rounds"],
    )
    return {
        "idx": args["idx"],
        "buffer": buf,
        "last_global_state": env.get_global_state(),
        "total_reward": total_reward,
        "steps": len(buf),
    }


def _run_compare_method_worker(job):
    """Run one top-level comparison method in a separate process."""
    method_name = job["method_name"]
    cfg = job["cfg"]
    mission_gen = job["mission_gen"]
    scenarios = job["scenarios"]
    args = job["args"]
    out_dir = Path(args["out_dir"])
    device = torch.device(args["device"])
    viz_out_dir = None if args.get("no_viz", False) else out_dir
    if args.get("vtw_cache_dir"):
        os.environ["MRL_DMS_VTW_CACHE_DIR"] = str(args["vtw_cache_dir"])
    _configure_torch_threads(args.get("torch_num_threads"))

    # Each method process gets a deterministic but separate RNG stream.
    seed = int(args["seed"]) + 1009 * METHOD_ORDER.index(method_name)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if method_name == "Single-PPO":
        metrics = run_single_ppo(
            cfg, mission_gen, scenarios, args["train_iters"], device,
            eval_workers=args["eval_workers"], viz_out_dir=viz_out_dir)
    elif method_name == "Indep-PPO":
        metrics = run_multi(
            cfg, mission_gen, scenarios, args["train_iters"], device,
            coordinate=False,
            method_name="Indep-PPO",
            eval_workers=args["eval_workers"],
            viz_out_dir=viz_out_dir)
    elif method_name == "MAPPO":
        metrics = run_multi(
            cfg, mission_gen, scenarios, args["train_iters"], device,
            coordinate=True,
            episode_assignment=args["episode_assignment"],
            assign_w_load=args["assign_w_load"],
            assignment_capacity_mode=args["assignment_capacity_mode"],
            release_before_deadline_s=args["release_before_deadline_s"],
            assignment_scorer=args["assignment_scorer"],
            assignment_scorer_mix=args["assignment_scorer_mix"],
            assignment_context_encoder=args["assignment_context_encoder"],
            assignment_context_weight=args["assignment_context_weight"],
            assignment_mlp_hidden_dim=args["assignment_mlp_hidden_dim"],
            assignment_mlp_seed=args["assignment_mlp_seed"],
            assignment_sequence_hidden_dim=args["assignment_sequence_hidden_dim"],
            assignment_replan_interval_s=args["assignment_replan_interval_s"],
            assignment_replan_horizon_s=args["assignment_replan_horizon_s"],
            assignment_replan_trigger=args["assignment_replan_trigger"],
            assignment_switch_penalty=args["assignment_switch_penalty"],
            assignment_lock_window_s=args["assignment_lock_window_s"],
            assignment_max_switches_per_task=args["assignment_max_switches_per_task"],
            assignment_manager_mode=args["assignment_manager_mode"],
            team_reward_mix=args["team_reward_mix"],
            load_balance_reward_coeff=args["load_balance_reward_coeff"],
            team_completion_bonus=args["team_completion_bonus"],
            normalize_agent_rewards=args["normalize_agent_rewards"],
            global_state_mode=args["global_state_mode"],
            global_state_task_stats=args["global_state_task_stats"],
            candidate_action_top_k=args["candidate_action_top_k"],
            satellite_curriculum=args["satellite_curriculum"],
            curriculum_min_satellites=args["curriculum_min_satellites"],
            curriculum_iters=args["curriculum_iters"],
            joint_explore_prob=args["joint_explore_prob"],
            intent_broadcast=args["intent_broadcast"],
            intent_replan_rounds=args["intent_replan_rounds"],
            train_env_workers=args["train_env_workers"],
            method_name="MAPPO",
            eval_workers=args["eval_workers"],
            viz_out_dir=viz_out_dir)
    elif method_name == "Greedy-Oracle":
        metrics = run_greedy_oracle(
            cfg, scenarios, eval_workers=args["eval_workers"], viz_out_dir=viz_out_dir)
    else:
        raise ValueError(f"未知 method_name={method_name!r}")

    return method_name, metrics


def _eval_oracle_worker(args):
    from envs.multi_satellite_env import MultiSatelliteEnv

    idx = args["idx"]
    cfg = args["cfg"]
    routine, dynamic = args["scenario"]
    n_sat = min(cfg.mappo.n_satellites, len(cfg.satellites))
    env = MultiSatelliteEnv(
        satellite_configs=cfg.satellites[:n_sat],
        max_action_dim=cfg.mission.max_action_dim,
        reward_config=cfg.reward,
        vtw_time_step_s=cfg.train.vtw_time_step_s,
        coordinate=True,
        episode_assignment=False,
        reassign_losers=False,
    )
    opts = {
        "routine_missions": copy.deepcopy(routine),
        "dynamic_schedule": copy.deepcopy(dynamic),
    }
    env.reset(options=opts)
    max_steps = int(env.horizon_s / 10.0) + 100
    for _ in range(max_steps):
        if env.is_done():
            break
        env.step(_greedy_oracle_actions(env))
    return {"idx": idx, "metrics": env.get_metrics()}


def _parallel_eval(worker_fn, task_args, eval_workers, label):
    n_workers = min(eval_workers, len(task_args))
    if n_workers <= 1:
        return None
    logger.info("并行评估 %s: episodes=%s, eval_workers=%s", label, len(task_args), n_workers)
    with get_context("spawn").Pool(processes=n_workers) as pool:
        raw = pool.map(worker_fn, task_args)
    raw = sorted(raw, key=lambda r: r["idx"])
    return _avg_metrics([r["metrics"] for r in raw])


def _save_single_viz_data(env, out_dir: Path, method_name: str):
    missions = [m for m in env.missions if m is not None]
    payload = {
        "method": method_name,
        "horizon_s": float(env.horizon_s),
        "missions": [_mission_to_dict(m) for m in missions],
        "schedule": {
            env.sat_config.name: [_record_to_dict(r) for r in env.schedule_log],
        },
    }
    path = out_dir / f"{method_name}_viz_data.json"
    with open(path, "w") as f:
        dump_json(payload, f, indent=2, ensure_ascii=False)
    return path


def _save_multi_viz_data(env, out_dir: Path, method_name: str):
    first_env = list(env.envs.values())[0]
    missions = [m for m in first_env.missions if m is not None]
    payload = {
        "method": method_name,
        "horizon_s": float(env.horizon_s),
        "missions": [_mission_to_dict(m) for m in missions],
        "schedule": {
            aid: [_record_to_dict(r) for r in sub_env.schedule_log]
            for aid, sub_env in env.envs.items()
        },
    }
    path = out_dir / f"{method_name}_viz_data.json"
    with open(path, "w") as f:
        dump_json(payload, f, indent=2, ensure_ascii=False)
    return path


# =======================================================================
# 方案 1: 单星 PPO
# =======================================================================
def run_single_ppo(cfg, mission_gen, scenarios, train_iters, device,
                   eval_workers: int = 1, viz_out_dir: Path = None):
    from envs.satellite_env import SatelliteSchedulingEnv
    from models.actor_critic import ActorCritic
    from algo.ppo import PPOTrainer, RolloutBuffer

    sat = cfg.satellites[0]
    env = SatelliteSchedulingEnv(
        satellite_config=sat, max_action_dim=cfg.mission.max_action_dim,
        reward_config=cfg.reward, vtw_time_step_s=cfg.train.vtw_time_step_s,
    )
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    model = ActorCritic(obs_dim, act_dim, cfg.network.hidden_layers,
                        cfg.network.activation).to(device)
    ppo = PPOTrainer(model, lr=cfg.ppo.learning_rate, gamma=cfg.ppo.discount_factor,
                     gae_lambda=cfg.ppo.gae_lambda, clip_ratio=cfg.ppo.clip_ratio,
                     entropy_coeff=cfg.ppo.entropy_coeff,
                     value_loss_coeff=cfg.ppo.value_loss_coeff,
                     ppo_epochs=cfg.ppo.ppo_epochs, batch_size=cfg.ppo.batch_size,
                     device=str(device))

    # 训练: 在随机采样场景上跑 PPO
    for it in range(train_iters):
        routine, dynamic = mission_gen.generate_episode_missions(
            n_routine=int(np.random.choice(cfg.mission.routine_pool_sizes)),
            n_dynamic_per_insertion=int(np.random.choice(cfg.mission.dynamic_pool_sizes)),
            n_insertions=cfg.mission.dynamic_insertions_per_day,
        )
        opts = {"routine_missions": copy.deepcopy(routine),
                "dynamic_schedule": copy.deepcopy(dynamic)}
        obs, info = env.reset(options=opts)
        buf = RolloutBuffer()
        obs, info, _ = ppo.collect_rollout(env, buf, cfg.meta.rollout_steps, obs, info,
                                           reset_options=opts)
        with torch.no_grad():
            last_v = model.get_value(torch.FloatTensor(obs).unsqueeze(0).to(device)).cpu().item()
        ppo.update(buf, last_v)

    # 评估
    if eval_workers > 1:
        task_args = [
            {
                "idx": ep_idx,
                "cfg": cfg,
                "scenario": scenario,
                "model_state": _torch_state_to_numpy(model.state_dict()),
            }
            for ep_idx, scenario in enumerate(scenarios)
        ]
        avg = _parallel_eval(_eval_single_worker, task_args, eval_workers, "Single-PPO")
        if avg is not None:
            if viz_out_dir is not None and scenarios:
                # 只为可视化额外串行跑最后一集，不纳入指标，避免 worker 传回大对象。
                routine, dynamic = scenarios[-1]
                opts = {"routine_missions": copy.deepcopy(routine),
                        "dynamic_schedule": copy.deepcopy(dynamic)}
                obs, info = env.reset(options=opts)
                done = False
                max_steps = int(env.horizon_s / 10.0) + 100
                for _ in range(max_steps):
                    if done:
                        break
                    mask = info.get("action_mask", np.ones(act_dim))
                    with torch.no_grad():
                        a, _, _, _ = model.get_action_and_value(
                            torch.FloatTensor(obs).unsqueeze(0).to(device),
                            torch.FloatTensor(mask).unsqueeze(0).to(device))
                    obs, _, term, trunc, info = env.step(a.cpu().item())
                    done = term or trunc
                _save_single_viz_data(env, viz_out_dir, "Single-PPO")
            return avg

    metrics_list = []
    for ep_idx, (routine, dynamic) in enumerate(scenarios):
        opts = {"routine_missions": copy.deepcopy(routine),
                "dynamic_schedule": copy.deepcopy(dynamic)}
        obs, info = env.reset(options=opts)
        done = False
        max_steps = int(env.horizon_s / 10.0) + 100
        for _ in range(max_steps):
            if done:
                break
            mask = info.get("action_mask", np.ones(act_dim))
            with torch.no_grad():
                a, _, _, _ = model.get_action_and_value(
                    torch.FloatTensor(obs).unsqueeze(0).to(device),
                    torch.FloatTensor(mask).unsqueeze(0).to(device))
            obs, r, term, trunc, info = env.step(a.cpu().item())
            done = term or trunc
        metrics_list.append(env.get_metrics())
        if viz_out_dir is not None and ep_idx == len(scenarios) - 1:
            _save_single_viz_data(env, viz_out_dir, "Single-PPO")
    return _avg_metrics(metrics_list)


# =======================================================================
# 方案 2/3: 多星 (independent_ppo: coordinate=False; mappo: coordinate=True)
# =======================================================================
def run_multi(cfg, mission_gen, scenarios, train_iters, device, coordinate,
              episode_assignment=True, assign_w_load=0.1,
              assignment_capacity_mode="proportional",
              release_before_deadline_s=1800.0,
              assignment_scorer="heuristic",
              assignment_scorer_mix=0.25,
              assignment_context_encoder="lstm",
              assignment_context_weight=0.25,
              assignment_mlp_hidden_dim=16,
              assignment_mlp_seed=42,
              assignment_sequence_hidden_dim=16,
              assignment_replan_interval_s=0.0,
              assignment_replan_horizon_s=0.0,
              assignment_replan_trigger="none",
              assignment_switch_penalty=0.05,
              assignment_lock_window_s=600.0,
              assignment_max_switches_per_task=2,
              assignment_manager_mode="none",
              team_reward_mix=0.0,
              load_balance_reward_coeff=0.0,
              team_completion_bonus=0.0,
              normalize_agent_rewards=False,
              global_state_mode="mean",
              global_state_task_stats=False,
              candidate_action_top_k=0,
              satellite_curriculum=False,
              curriculum_min_satellites=1,
              curriculum_iters=10,
              joint_explore_prob=0.0,
              intent_broadcast=False,
              intent_replan_rounds=1,
              train_env_workers: int = 1,
              method_name="MAPPO",
              eval_workers: int = 1,
              viz_out_dir: Path = None):
    from envs.multi_satellite_env import MultiSatelliteEnv
    from models.mappo import MAPPOActorCritic
    from algo.mappo_trainer import MAPPOTrainer, MultiAgentRolloutBuffer

    n_sat = min(cfg.mappo.n_satellites, len(cfg.satellites))
    sat_cfgs = cfg.satellites[:n_sat]
    env = MultiSatelliteEnv(
        satellite_configs=sat_cfgs, max_action_dim=cfg.mission.max_action_dim,
        reward_config=cfg.reward, vtw_time_step_s=cfg.train.vtw_time_step_s,
        coordinate=coordinate,
        # 全局 episode 级指派仅对协同方法 (MAPPO) 启用
        episode_assignment=(coordinate and episode_assignment),
        assign_w_load=assign_w_load,
        assignment_capacity_mode=assignment_capacity_mode,
        release_before_deadline_s=release_before_deadline_s,
        assignment_scorer=assignment_scorer if coordinate else "heuristic",
        assignment_scorer_mix=assignment_scorer_mix,
        assignment_context_encoder=assignment_context_encoder,
        assignment_context_weight=assignment_context_weight,
        assignment_mlp_hidden_dim=assignment_mlp_hidden_dim,
        assignment_mlp_seed=assignment_mlp_seed,
        assignment_sequence_hidden_dim=assignment_sequence_hidden_dim,
        assignment_replan_interval_s=assignment_replan_interval_s if coordinate else 0.0,
        assignment_replan_horizon_s=assignment_replan_horizon_s if coordinate else 0.0,
        assignment_replan_trigger=assignment_replan_trigger if coordinate else "none",
        assignment_switch_penalty=assignment_switch_penalty,
        assignment_lock_window_s=assignment_lock_window_s,
        assignment_max_switches_per_task=assignment_max_switches_per_task,
        assignment_manager_mode=assignment_manager_mode if coordinate else "none",
        team_reward_mix=team_reward_mix if coordinate else 0.0,
        load_balance_reward_coeff=load_balance_reward_coeff if coordinate else 0.0,
        team_completion_bonus=team_completion_bonus if coordinate else 0.0,
        global_state_mode=global_state_mode if coordinate else "mean",
        global_state_task_stats=(coordinate and global_state_task_stats),
        candidate_action_top_k=candidate_action_top_k,
    )
    obs_dim = env.local_obs_dim
    act_dim = env.action_dim
    # 协同: 集中式 critic 用全局状态; 无协同: critic 仍存在但每星只用自己局部观测做全局态
    model = MAPPOActorCritic(
        local_obs_dim=obs_dim, action_dim=act_dim, global_state_dim=env.global_state_dim,
        actor_hidden_dims=cfg.network.hidden_layers,
        critic_hidden_dims=cfg.mappo.critic_hidden_dims,
    ).to(device)
    trainer = MAPPOTrainer(
        model, lr=cfg.ppo.learning_rate, gamma=cfg.ppo.discount_factor,
        gae_lambda=cfg.ppo.gae_lambda, clip_ratio=cfg.ppo.clip_ratio,
        entropy_coeff=cfg.ppo.entropy_coeff, value_loss_coeff=cfg.ppo.value_loss_coeff,
        ppo_epochs=cfg.ppo.ppo_epochs, batch_size=cfg.ppo.batch_size, device=str(device),
        normalize_agent_rewards=(coordinate and normalize_agent_rewards),
    )

    env_kwargs = {
        "coordinate": coordinate,
        "episode_assignment": (coordinate and episode_assignment),
        "assign_w_load": assign_w_load,
        "assignment_capacity_mode": assignment_capacity_mode,
        "release_before_deadline_s": release_before_deadline_s,
        "assignment_scorer": assignment_scorer if coordinate else "heuristic",
        "assignment_scorer_mix": assignment_scorer_mix,
        "assignment_context_encoder": assignment_context_encoder,
        "assignment_context_weight": assignment_context_weight,
        "assignment_mlp_hidden_dim": assignment_mlp_hidden_dim,
        "assignment_mlp_seed": assignment_mlp_seed,
        "assignment_sequence_hidden_dim": assignment_sequence_hidden_dim,
        "assignment_replan_interval_s": assignment_replan_interval_s if coordinate else 0.0,
        "assignment_replan_horizon_s": assignment_replan_horizon_s if coordinate else 0.0,
        "assignment_replan_trigger": assignment_replan_trigger if coordinate else "none",
        "assignment_switch_penalty": assignment_switch_penalty,
        "assignment_lock_window_s": assignment_lock_window_s,
        "assignment_max_switches_per_task": assignment_max_switches_per_task,
        "assignment_manager_mode": assignment_manager_mode if coordinate else "none",
        "team_reward_mix": team_reward_mix if coordinate else 0.0,
        "load_balance_reward_coeff": load_balance_reward_coeff if coordinate else 0.0,
        "team_completion_bonus": team_completion_bonus if coordinate else 0.0,
        "global_state_mode": global_state_mode if coordinate else "mean",
        "global_state_task_stats": (coordinate and global_state_task_stats),
        "candidate_action_top_k": candidate_action_top_k,
    }
    trainer_kwargs = {
        "normalize_agent_rewards": (coordinate and normalize_agent_rewards),
    }
    train_env_workers = max(1, int(train_env_workers or 1))

    for it in range(train_iters):
        active_agent_ids = None
        if coordinate and satellite_curriculum:
            min_sat = max(1, min(curriculum_min_satellites, n_sat))
            if curriculum_iters <= 1:
                active_n = n_sat
            else:
                progress = min(it / max(curriculum_iters - 1, 1), 1.0)
                span = n_sat - min_sat
                active_n = int(round(min_sat + progress * span))
                active_n = max(min_sat, min(n_sat, active_n))
            active_agent_ids = env.agent_ids[:active_n]

        n_workers = min(train_env_workers, max(int(cfg.meta.rollout_steps), 1))
        if n_workers > 1:
            step_counts = [cfg.meta.rollout_steps // n_workers] * n_workers
            for wi in range(cfg.meta.rollout_steps % n_workers):
                step_counts[wi] += 1
            model_state_np = _torch_state_to_numpy(model.state_dict())
            task_args = []
            for wi, worker_steps in enumerate(step_counts):
                routine, dynamic = mission_gen.generate_episode_missions(
                    n_routine=int(np.random.choice(cfg.mission.routine_pool_sizes)),
                    n_dynamic_per_insertion=int(np.random.choice(cfg.mission.dynamic_pool_sizes)),
                    n_insertions=cfg.mission.dynamic_insertions_per_day,
                )
                task_args.append({
                    "idx": wi,
                    "cfg": copy.deepcopy(cfg),
                    "scenario": (routine, dynamic),
                    "model_state": model_state_np,
                    "env_kwargs": env_kwargs,
                    "trainer_kwargs": trainer_kwargs,
                    "rollout_steps": worker_steps,
                    "active_agent_ids": active_agent_ids,
                    "joint_explore_prob": joint_explore_prob if coordinate else 0.0,
                    "intent_broadcast": (coordinate and intent_broadcast),
                    "intent_replan_rounds": intent_replan_rounds,
                    "seed": int(np.random.randint(0, 2**31 - 1)),
                })
            logger.info(
                "%s 训练并行 rollout: iter=%s/%s, train_env_workers=%s, steps=%s",
                method_name,
                it + 1,
                train_iters,
                n_workers,
                step_counts,
            )
            with get_context("spawn").Pool(processes=n_workers) as pool:
                worker_results = pool.map(_collect_multi_rollout_worker, task_args)
            worker_results = sorted(worker_results, key=lambda r: r["idx"])
            buffers = [r["buffer"] for r in worker_results]
            last_states = [r["last_global_state"] for r in worker_results]
            trainer.update_many(buffers, last_states)
        else:
            routine, dynamic = mission_gen.generate_episode_missions(
                n_routine=int(np.random.choice(cfg.mission.routine_pool_sizes)),
                n_dynamic_per_insertion=int(np.random.choice(cfg.mission.dynamic_pool_sizes)),
                n_insertions=cfg.mission.dynamic_insertions_per_day,
            )
            opts = {"routine_missions": copy.deepcopy(routine),
                    "dynamic_schedule": copy.deepcopy(dynamic)}
            res = env.reset(options=opts)
            cur_obs = {a: r[0] for a, r in res.items()}
            cur_info = {a: r[1] for a, r in res.items()}
            buf = MultiAgentRolloutBuffer()
            buf.init_agents(active_agent_ids or env.agent_ids)
            cur_obs, cur_info, _ = trainer.collect_rollout(
                env, buf, cfg.meta.rollout_steps, cur_obs, cur_info,
                active_agent_ids=active_agent_ids,
                joint_explore_prob=(joint_explore_prob if coordinate else 0.0),
                intent_broadcast=(coordinate and intent_broadcast),
                intent_replan_rounds=intent_replan_rounds)
            trainer.update(buf, env.get_global_state())

    # 评估
    if eval_workers > 1:
        task_args = [
            {
                "idx": ep_idx,
                "cfg": cfg,
                "scenario": scenario,
                "model_state": _torch_state_to_numpy(model.state_dict()),
                "env_kwargs": env_kwargs,
                "trainer_kwargs": trainer_kwargs,
                "intent_broadcast": (coordinate and intent_broadcast),
                "intent_replan_rounds": intent_replan_rounds,
            }
            for ep_idx, scenario in enumerate(scenarios)
        ]
        avg = _parallel_eval(_eval_multi_worker, task_args, eval_workers, method_name)
        if avg is not None:
            if viz_out_dir is not None and scenarios:
                routine, dynamic = scenarios[-1]
                opts = {"routine_missions": copy.deepcopy(routine),
                        "dynamic_schedule": copy.deepcopy(dynamic)}
                env.set_eval_mode(True)
                res = env.reset(options=opts)
                cur_obs = {a: r[0] for a, r in res.items()}
                cur_info = {a: r[1] for a, r in res.items()}
                max_steps = int(env.horizon_s / 10.0) + 100
                for _ in range(max_steps):
                    actions, _, _, _ = trainer.sample_actions(
                        multi_env=env,
                        current_obs=cur_obs,
                        current_infos=cur_info,
                        intent_broadcast=(coordinate and intent_broadcast),
                        intent_replan_rounds=intent_replan_rounds,
                    )
                    step_res = env.step(actions)
                    for aid, (obs, _, _, _, info) in step_res.items():
                        cur_obs[aid] = obs
                        cur_info[aid] = info
                    if env.is_done():
                        break
                _save_multi_viz_data(env, viz_out_dir, method_name)
            return avg

    metrics_list = []
    env.set_eval_mode(True)   # 评估期启用 A1 败者改派 (训练期关闭以保信用分配)
    for ep_idx, (routine, dynamic) in enumerate(scenarios):
        opts = {"routine_missions": copy.deepcopy(routine),
                "dynamic_schedule": copy.deepcopy(dynamic)}
        res = env.reset(options=opts)
        cur_obs = {a: r[0] for a, r in res.items()}
        cur_info = {a: r[1] for a, r in res.items()}
        max_steps = int(env.horizon_s / 10.0) + 100
        for _ in range(max_steps):
            actions, _, _, _ = trainer.sample_actions(
                multi_env=env,
                current_obs=cur_obs,
                current_infos=cur_info,
                intent_broadcast=(coordinate and intent_broadcast),
                intent_replan_rounds=intent_replan_rounds,
            )
            step_res = env.step(actions)
            for aid, (o, r, term, trunc, inf) in step_res.items():
                cur_obs[aid] = o
                cur_info[aid] = inf
            if env.is_done():
                break
        metrics_list.append(env.get_metrics())
        if viz_out_dir is not None and ep_idx == len(scenarios) - 1:
            _save_multi_viz_data(env, viz_out_dir, method_name)
    return _avg_metrics(metrics_list)


# =======================================================================
# 方案 4: Greedy Oracle (集中式启发式上界参考, 不训练)
# =======================================================================
def _oracle_action_value(env, agent_id, action):
    sub_env = env.envs[agent_id]
    mission = sub_env.missions[action]
    if mission is None or mission.is_observed:
        return None

    off_nadir = None
    for vtw in sub_env.mission_vtw.get(mission.id, []):
        if vtw.start_time <= sub_env.current_time_s <= vtw.end_time - mission.duration_s:
            off_nadir = vtw.off_nadir_deg
            break
    if off_nadir is None:
        return None

    max_roll = max(sub_env.sat_config.max_roll_deg, 1e-6)
    quality = 1.0 - min(off_nadir / max_roll, 1.0)
    priority = mission.priority / 10.0
    horizon_left = max(mission.deadline_s - mission.earliest_time_s, 1.0)
    urgency = 1.0 - max(mission.deadline_s - sub_env.current_time_s, 0.0) / horizon_left
    dynamic_bonus = 0.3 if mission.is_dynamic else 0.0
    load_penalty = 0.03 * len(sub_env.schedule_log)
    return priority + 0.5 * quality + 0.3 * urgency + dynamic_bonus - load_penalty


def _greedy_oracle_actions(env):
    idle = env.idle_action
    candidates = []
    for aid in env.agent_ids:
        mask = env.envs[aid]._build_action_mask()
        for action in np.nonzero(mask[:env.max_action_dim])[0].tolist():
            value = _oracle_action_value(env, aid, action)
            if value is not None:
                candidates.append((value, aid, action))

    actions = {aid: idle for aid in env.agent_ids}
    claimed_agents = set()
    claimed_actions = set()
    for _, aid, action in sorted(candidates, reverse=True):
        if aid in claimed_agents or action in claimed_actions:
            continue
        actions[aid] = action
        claimed_agents.add(aid)
        claimed_actions.add(action)
    return actions


def run_greedy_oracle(cfg, scenarios, eval_workers: int = 1, viz_out_dir: Path = None):
    from envs.multi_satellite_env import MultiSatelliteEnv

    if eval_workers > 1:
        task_args = [
            {"idx": ep_idx, "cfg": cfg, "scenario": scenario}
            for ep_idx, scenario in enumerate(scenarios)
        ]
        avg = _parallel_eval(_eval_oracle_worker, task_args, eval_workers, "Greedy-Oracle")
        if avg is not None and viz_out_dir is None:
            return avg
        if avg is not None and viz_out_dir is not None and not scenarios:
            return avg
        if avg is None:
            eval_workers = 1

    n_sat = min(cfg.mappo.n_satellites, len(cfg.satellites))
    sat_cfgs = cfg.satellites[:n_sat]
    env = MultiSatelliteEnv(
        satellite_configs=sat_cfgs,
        max_action_dim=cfg.mission.max_action_dim,
        reward_config=cfg.reward,
        vtw_time_step_s=cfg.train.vtw_time_step_s,
        coordinate=True,
        episode_assignment=False,
        reassign_losers=False,
    )

    if eval_workers > 1:
        eval_scenarios = [scenarios[-1]] if scenarios else []
    else:
        eval_scenarios = scenarios

    metrics_list = []
    for ep_idx, (routine, dynamic) in enumerate(eval_scenarios):
        opts = {
            "routine_missions": copy.deepcopy(routine),
            "dynamic_schedule": copy.deepcopy(dynamic),
        }
        env.reset(options=opts)
        max_steps = int(env.horizon_s / 10.0) + 100
        for _ in range(max_steps):
            if env.is_done():
                break
            actions = _greedy_oracle_actions(env)
            env.step(actions)
        metrics_list.append(env.get_metrics())
        if viz_out_dir is not None and ep_idx == len(eval_scenarios) - 1:
            _save_multi_viz_data(env, viz_out_dir, "Greedy-Oracle")
    if eval_workers > 1:
        return avg
    return _avg_metrics(metrics_list)


def main():
    parser = argparse.ArgumentParser(description="方案对比实验")
    parser.add_argument("--acled_path", type=str, default=None)
    parser.add_argument("--n_satellites", type=int, default=6)
    parser.add_argument("--train_iters", type=int, default=30)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--eval_workers", type=int, default=1,
                        help="评估 episode 并行 worker 数; 1 为串行")
    parser.add_argument("--train_env_workers", type=int, default=1,
                        help="MAPPO 训练 rollout 并行环境进程数; 1 为旧版串行采样")
    parser.add_argument("--torch_num_threads", type=int, default=None,
                        help="单个训练进程内 PyTorch CPU 线程数; 默认沿用环境设置")
    parser.add_argument("--n_routine", type=int, default=200)
    parser.add_argument("--n_dynamic", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="runs/compare")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_action_dim", type=int, default=None,
                        help="动作空间任务槽位数; 默认自动取 max(配置值, "
                             "n_routine + dynamic_insertions_per_day*n_dynamic)")
    parser.add_argument("--rollout_steps", type=int, default=None,
                        help="覆盖 PPO/MAPPO 每次训练迭代的 rollout 长度")
    parser.add_argument("--ppo_epochs", type=int, default=None,
                        help="覆盖 PPO/MAPPO 每次 update 的 epoch 数")
    parser.add_argument("--ppo_batch_size", type=int, default=None,
                        help="覆盖 PPO/MAPPO update minibatch 大小")
    parser.add_argument("--vtw_time_step_s", type=float, default=None,
                        help="覆盖 VTW 采样步长; 越小越精确但 CPU 更重")
    parser.add_argument("--vtw_cache_dir", type=str, default=None,
                        help="VTW 磁盘缓存目录; 为空则只使用进程内缓存")
    parser.add_argument("--methods", type=str, default="single,indep,mappo",
                        help="逗号分隔选择运行方法: single,indep,mappo,oracle,all; "
                             "默认运行 Single-PPO/Indep-PPO/MAPPO")
    parser.add_argument("--method_jobs", type=int, default=1,
                        help="并行训练多少个顶层方法; 例如 single,indep,mappo 可设为 3")
    parser.add_argument("--no_viz", action="store_true",
                        help="跳过 *_viz_data.json 生成, 避免并行评估后额外串行重跑 1 个 episode")
    parser.add_argument("--run_name", type=str, default=None,
                        help="本次 compare run 名称; 默认由 experiment_tag/关键参数生成")
    parser.add_argument("--flat_out_dir", action="store_true",
                        help="直接写入 --out_dir, 不自动创建唯一子目录")
    parser.add_argument("--episode_assignment", action="store_true", default=True,
                        help="MAPPO 启用全局 episode 级任务指派 (默认开)")
    parser.add_argument("--no_episode_assignment", dest="episode_assignment",
                        action="store_false", help="关闭全局指派 (退回逐时刻协同)")
    parser.add_argument("--assign_w_load", type=float, default=0.1,
                        help="全局指派的负载均衡权重 (越大越均衡, 吞吐换均衡)")
    parser.add_argument("--assignment_capacity_mode", type=str, default="proportional",
                        choices=["proportional", "equal"],
                        help="全局指派目标容量: proportional=按覆盖质量比例, equal=每星等额")
    parser.add_argument("--release_before_deadline_s", type=float, default=1800.0,
                        help="任务截止前多少秒释放所有权给非 owner 接手; 0 表示关闭")
    parser.add_argument("--assignment_scorer", type=str, default="heuristic",
                        choices=["heuristic", "mlp", "lstm", "gru", "transformer", "set_transformer", "gnn", "cva"],
                        help="episode 级任务指派打分器: heuristic/mlp/lstm/gru/transformer/set_transformer/gnn/cva")
    parser.add_argument("--assignment_scorer_mix", type=float, default=0.25,
                        help="学习式 scorer 与旧启发式分数的混合比例; 0 等价 heuristic, 1 完全使用学习式分数")
    parser.add_argument("--assignment_context_encoder", type=str, default="lstm",
                        choices=["mlp", "lstm", "gru", "transformer", "set_transformer", "gnn"],
                        help="assignment_scorer=cva 时的上下文价值编码器")
    parser.add_argument("--assignment_context_weight", type=float, default=0.25,
                        help="CVA 中上下文编码器价值项权重")
    parser.add_argument("--assignment_mlp_hidden_dim", type=int, default=16,
                        help="assignment_scorer=mlp 时的隐藏层维度")
    parser.add_argument("--assignment_mlp_seed", type=int, default=42,
                        help="assignment_scorer=mlp 时的确定性初始化种子")
    parser.add_argument("--assignment_sequence_hidden_dim", type=int, default=16,
                        help="assignment_scorer=lstm/gru/transformer/set_transformer/gnn 时的上下文隐藏维度")
    parser.add_argument("--assignment_replan_interval_s", type=float, default=0.0,
                        help="滚动重分配周期; 0 表示关闭周期重分配")
    parser.add_argument("--assignment_replan_horizon_s", type=float, default=0.0,
                        help="滚动重分配只看未来多少秒窗口; 0 表示看到剩余 horizon")
    parser.add_argument("--assignment_replan_trigger", type=str, default="none",
                        help="滚动重分配事件触发器,逗号分隔: periodic,dynamic,stale_owner,deadline,imbalance; none 关闭")
    parser.add_argument("--assignment_switch_penalty", type=float, default=0.05,
                        help="滚动重分配切换 owner 的惩罚, 越大越抑制频繁换人")
    parser.add_argument("--assignment_lock_window_s", type=float, default=600.0,
                        help="owner 下一可行窗口前多少秒锁定任务, 避免临门换人")
    parser.add_argument("--assignment_max_switches_per_task", type=int, default=2,
                        help="每个任务最多允许 owner 切换次数; 0 表示不允许切换")
    parser.add_argument("--assignment_manager_mode", type=str, default="none",
                        choices=["none", "rule"],
                        help="高层任务分配 manager: none=使用环境内 scorer, rule=规则式高层 manager")
    parser.add_argument("--team_reward_mix", type=float, default=0.0,
                        help="团队平均奖励混合比例; 0 保持个体奖励, 1 完全使用团队平均奖励")
    parser.add_argument("--load_balance_reward_coeff", type=float, default=0.0,
                        help="低负载卫星完成任务的奖励系数; 0 关闭")
    parser.add_argument("--team_completion_bonus", type=float, default=0.0,
                        help="每新增完成 1 个团队任务时给全体的 bonus; 0 关闭")
    parser.add_argument("--normalize_agent_rewards", action="store_true",
                        help="MAPPO 更新前对每颗卫星 rollout 奖励做归一化")
    parser.add_argument("--global_state_mode", type=str, default="mean",
                        choices=["mean", "concat"],
                        help="MAPPO critic 全局状态聚合: mean=旧实现, concat=拼接各星观测")
    parser.add_argument("--global_state_task_stats", action="store_true",
                        help="MAPPO critic 全局状态追加任务/负载统计")
    parser.add_argument("--candidate_action_top_k", type=int, default=0,
                        help="多星候选动作空间大小; 0=full action, >0=Top-K任务+idle")
    parser.add_argument("--run_oracle", action="store_true",
                        help="额外运行 Greedy-Oracle 集中式启发式参考")
    parser.add_argument("--satellite_curriculum", action="store_true",
                        help="MAPPO 训练期从少量活跃卫星逐步增加到全部卫星")
    parser.add_argument("--curriculum_min_satellites", type=int, default=1,
                        help="卫星数量课程的起始活跃卫星数")
    parser.add_argument("--curriculum_iters", type=int, default=10,
                        help="多少个训练迭代内从 min 卫星线性增加到全部卫星")
    parser.add_argument("--joint_explore_prob", type=float, default=0.0,
                        help="训练期联合探索概率; 随机挑选互不重复的可行动作")
    parser.add_argument("--intent_broadcast", action="store_true",
                        help="启用 E17 意图广播: 冲突败者基于广播意图重采样")
    parser.add_argument("--intent_replan_rounds", type=int, default=1,
                        help="意图广播冲突后最多重采样轮数")
    parser.add_argument("--experiment_tag", type=str, default="single_compare",
                        help="实验标签, 写入 manifest 方便批量对比")
    args = parser.parse_args()
    method_names = parse_methods(args.methods, run_oracle=args.run_oracle)
    args.methods = ",".join(method_names)
    _configure_torch_threads(args.torch_num_threads)
    if args.vtw_cache_dir:
        os.environ["MRL_DMS_VTW_CACHE_DIR"] = args.vtw_cache_dir
        Path(args.vtw_cache_dir).mkdir(parents=True, exist_ok=True)
        logger.info("VTW 磁盘缓存: %s", args.vtw_cache_dir)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    cfg = get_default_config()
    if args.rollout_steps is not None:
        cfg.meta.rollout_steps = args.rollout_steps
    if args.ppo_epochs is not None:
        cfg.ppo.ppo_epochs = args.ppo_epochs
    if args.ppo_batch_size is not None:
        cfg.ppo.batch_size = args.ppo_batch_size
    if args.vtw_time_step_s is not None:
        cfg.train.vtw_time_step_s = args.vtw_time_step_s
    base_satellite_count = len(cfg.satellites)
    cfg.satellites = expand_satellite_configs(cfg.satellites, args.n_satellites)
    cfg.mappo.n_satellites = args.n_satellites
    required_action_dim = (
        args.n_routine
        + cfg.mission.dynamic_insertions_per_day * args.n_dynamic
    )
    if args.max_action_dim is not None and args.max_action_dim < required_action_dim:
        raise ValueError(
            f"--max_action_dim={args.max_action_dim} 小于当前评估任务槽位需求 "
            f"{required_action_dim} = n_routine({args.n_routine}) + "
            f"{cfg.mission.dynamic_insertions_per_day}*n_dynamic({args.n_dynamic})"
        )
    cfg.mission.max_action_dim = max(
        cfg.mission.max_action_dim,
        args.max_action_dim or 0,
        required_action_dim,
    )
    args.max_action_dim = cfg.mission.max_action_dim
    logger.info(
        "动作空间任务槽位: max_action_dim=%s, 当前评估需求=%s",
        cfg.mission.max_action_dim,
        required_action_dim,
    )
    logger.info(
        "训练配置: train_iters=%s, rollout_steps=%s, ppo_epochs=%s, "
        "ppo_batch_size=%s, train_env_workers=%s, eval_workers=%s, "
        "vtw_time_step_s=%s, device=%s",
        args.train_iters,
        cfg.meta.rollout_steps,
        cfg.ppo.ppo_epochs,
        cfg.ppo.batch_size,
        args.train_env_workers,
        args.eval_workers,
        cfg.train.vtw_time_step_s,
        device,
    )
    logger.info(
        "任务分配: scorer=%s, mix=%s, context_encoder=%s, context_weight=%s, "
        "replan_trigger=%s, replan_horizon_s=%s",
        args.assignment_scorer,
        args.assignment_scorer_mix,
        args.assignment_context_encoder,
        args.assignment_context_weight,
        args.assignment_replan_trigger,
        args.assignment_replan_horizon_s,
    )
    if args.candidate_action_top_k > 0:
        logger.info(
            "候选动作空间: Top-K=%s + idle; 底层任务槽位 max_action_dim=%s",
            args.candidate_action_top_k,
            cfg.mission.max_action_dim,
        )
    if args.n_satellites > base_satellite_count:
        logger.info(
            "星座规模扩展: 使用 %s 颗派生卫星 (基础配置 %s 颗)",
            len(cfg.satellites),
            base_satellite_count,
        )

    acled = load_acled_shapefile(args.acled_path) if args.acled_path else None
    mission_gen = MissionGenerator(acled_df=acled, seed=args.seed)

    # 固定测试集 (三方案共用)
    scenarios = make_test_scenarios(mission_gen, args.eval_episodes,
                                    args.n_routine, args.n_dynamic,
                                    n_insertions=cfg.mission.dynamic_insertions_per_day,
                                    seed=args.seed + 1000)
    logger.info(
        "测试场景数: %s, 每个 %s routine + %s×%s dynamic",
        len(scenarios),
        args.n_routine,
        args.n_dynamic,
        cfg.mission.dynamic_insertions_per_day,
    )

    results = {}
    t0 = time.time()

    if args.flat_out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        assignment_tag = "assign_on" if args.episode_assignment else "assign_off"
        run_name = args.run_name or (
            f"{args.experiment_tag}_sat{args.n_satellites}_"
            f"iter{args.train_iters}_{assignment_tag}_seed{args.seed}"
        )
        out_dir = unique_dir(args.out_dir, safe_name(run_name))
        args.out_dir = str(out_dir)
    viz_out_dir = None if args.no_viz else out_dir

    method_jobs = max(1, min(args.method_jobs, len(method_names)))
    if method_jobs > 1:
        if str(device) != "cpu":
            logger.warning(
                "--method_jobs=%s 会并行启动多个训练进程; 单卡 GPU 场景可能抢显存, CPU 推荐使用",
                method_jobs,
            )
        logger.info(
            "并行顶层方法训练: method_jobs=%s, methods=%s",
            method_jobs,
            ",".join(method_names),
        )
        args_dict = vars(args).copy()
        args_dict["out_dir"] = str(out_dir)
        jobs = [
            {
                "method_name": method_name,
                "cfg": copy.deepcopy(cfg),
                "mission_gen": copy.deepcopy(mission_gen),
                "scenarios": copy.deepcopy(scenarios),
                "args": args_dict,
            }
            for method_name in method_names
        ]
        with ProcessPoolExecutor(
            max_workers=method_jobs,
            mp_context=get_context("spawn"),
        ) as executor:
            futures = {
                executor.submit(_run_compare_method_worker, job): job["method_name"]
                for job in jobs
            }
            for future in as_completed(futures):
                method_name, metrics = future.result()
                results[method_name] = metrics
                logger.info("完成顶层方法: %s", method_name)
    else:
        step_total = len(method_names)
        step_idx = 1
        if "Single-PPO" in method_names:
            logger.info(f"=== [{step_idx}/{step_total}] 单星 PPO ===")
            results["Single-PPO"] = run_single_ppo(
                cfg, mission_gen, scenarios, args.train_iters, device,
                eval_workers=args.eval_workers, viz_out_dir=viz_out_dir)
            step_idx += 1

        if "Indep-PPO" in method_names:
            logger.info(f"=== [{step_idx}/{step_total}] 多星独立 PPO (无协同 baseline) ===")
            results["Indep-PPO"] = run_multi(
                cfg, mission_gen, scenarios, args.train_iters, device, coordinate=False,
                method_name="Indep-PPO", eval_workers=args.eval_workers, viz_out_dir=viz_out_dir)
            step_idx += 1

        if "MAPPO" in method_names:
            logger.info(f"=== [{step_idx}/{step_total}] 多星 MAPPO (协同, 本方法) ===")
            results["MAPPO"] = run_multi(cfg, mission_gen, scenarios, args.train_iters, device,
                                         coordinate=True, episode_assignment=args.episode_assignment,
                                         assign_w_load=args.assign_w_load,
                                         assignment_capacity_mode=args.assignment_capacity_mode,
                                         release_before_deadline_s=args.release_before_deadline_s,
                                         assignment_scorer=args.assignment_scorer,
                                         assignment_scorer_mix=args.assignment_scorer_mix,
                                         assignment_context_encoder=args.assignment_context_encoder,
                                         assignment_context_weight=args.assignment_context_weight,
                                         assignment_mlp_hidden_dim=args.assignment_mlp_hidden_dim,
                                         assignment_mlp_seed=args.assignment_mlp_seed,
                                         assignment_sequence_hidden_dim=args.assignment_sequence_hidden_dim,
                                         assignment_replan_interval_s=args.assignment_replan_interval_s,
                                         assignment_replan_horizon_s=args.assignment_replan_horizon_s,
                                         assignment_replan_trigger=args.assignment_replan_trigger,
                                         assignment_switch_penalty=args.assignment_switch_penalty,
                                         assignment_lock_window_s=args.assignment_lock_window_s,
                                         assignment_max_switches_per_task=args.assignment_max_switches_per_task,
                                         assignment_manager_mode=args.assignment_manager_mode,
                                         team_reward_mix=args.team_reward_mix,
                                         load_balance_reward_coeff=args.load_balance_reward_coeff,
                                         team_completion_bonus=args.team_completion_bonus,
                                         normalize_agent_rewards=args.normalize_agent_rewards,
                                         global_state_mode=args.global_state_mode,
                                         global_state_task_stats=args.global_state_task_stats,
                                         candidate_action_top_k=args.candidate_action_top_k,
                                         satellite_curriculum=args.satellite_curriculum,
                                         curriculum_min_satellites=args.curriculum_min_satellites,
                                         curriculum_iters=args.curriculum_iters,
                                         joint_explore_prob=args.joint_explore_prob,
                                         intent_broadcast=args.intent_broadcast,
                                         intent_replan_rounds=args.intent_replan_rounds,
                                         train_env_workers=args.train_env_workers,
                                         method_name="MAPPO",
                                         eval_workers=args.eval_workers,
                                         viz_out_dir=viz_out_dir)
            step_idx += 1

        if "Greedy-Oracle" in method_names:
            logger.info(f"=== [{step_idx}/{step_total}] Greedy Oracle (集中式启发式参考) ===")
            results["Greedy-Oracle"] = run_greedy_oracle(
                cfg, scenarios, eval_workers=args.eval_workers, viz_out_dir=viz_out_dir)

    # 协同增益: 多星完成数 / (N × 单星完成数), 仅在包含 Single-PPO 时计算。
    n_sat = args.n_satellites
    single_sched = results.get("Single-PPO", {}).get("n_scheduled", 0)
    if single_sched > 0:
        for name in ["Indep-PPO", "MAPPO"]:
            if name not in results:
                continue
            multi_sched = results[name].get("n_scheduled", 0)
            results[name]["coordination_gain"] = multi_sched / (n_sat * single_sched)
    oracle_sched = results.get("Greedy-Oracle", {}).get("n_scheduled", 0)
    if oracle_sched > 0:
        for name in ["Indep-PPO", "MAPPO"]:
            if name not in results:
                continue
            results[name]["oracle_relative_completion"] = (
                results[name].get("n_scheduled", 0) / oracle_sched
            )
        results["Greedy-Oracle"]["oracle_relative_completion"] = 1.0

    with open(out_dir / "comparison_results.json", "w") as f:
        dump_json(results, f, indent=2, ensure_ascii=False)
    elapsed_s = time.time() - t0
    _write_manifest(out_dir, args, results, elapsed_s)

    # 控制台摘要
    print("\n" + "=" * 78)
    method_names = [name for name in METHOD_ORDER if name in results]
    print(f"{'指标':<32}" + "".join(f"{name:>14}" for name in method_names))
    print("-" * 78)
    show = [
        ("observation_success_rate", "观测成功率", "%"),
        ("dynamic_completion_rate", "动态完成率", "%"),
        ("routine_completion_rate", "常规完成率", "%"),
        ("total_reward", "累积奖励", ""),
        ("n_total_tasks", "全部任务数", ""),
        ("n_feasible_tasks", "可观测任务数", ""),
        ("n_feasible_routine", "可观测常规任务数", ""),
        ("n_feasible_dynamic", "可观测动态任务数", ""),
        ("n_scheduled", "完成任务数", ""),
        ("n_duplicates", "重复观测数", ""),
        ("duplicate_rate", "重复观测率", "%"),
        ("load_balance_cv", "负载变异系数", ""),
        ("avg_off_nadir_deg", "平均off-nadir", "°"),
        ("avg_dynamic_response_s", "动态响应延迟", "s"),
        ("n_replans", "重分配次数", ""),
        ("n_owner_switches", "owner切换数", ""),
        ("owner_churn_rate", "owner切换率", "%"),
        ("stale_owner_rate", "失效owner比例", "%"),
        ("deadline_rescue_rate", "deadline救援率", "%"),
        ("coordination_gain", "协同增益", ""),
        ("oracle_relative_completion", "Oracle相对完成率", "%"),
    ]
    for key, label, unit in show:
        row = f"{label:<32}"
        for name in method_names:
            v = results[name].get(key, None)
            if v is None:
                row += f"{'—':>14}"
            elif unit == "%":
                row += f"{v*100:>13.1f}%"
            else:
                row += f"{v:>14.2f}"
        print(row)
    print("=" * 78)
    print(f"总耗时: {elapsed_s:.1f}s, 结果: {out_dir/'comparison_results.json'}")
    print(f"实验记录: {out_dir/'manifest.json'}")


if __name__ == "__main__":
    main()
