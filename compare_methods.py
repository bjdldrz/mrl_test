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
import json
import time
import copy
import argparse
import logging
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from config import get_default_config
from data.mission_generator import MissionGenerator, load_acled_shapefile

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("compare")


# =======================================================================
# 固定测试集: 三种方案在同一批场景上评估, 保证公平
# =======================================================================
def make_test_scenarios(mission_gen, n_episodes, n_routine, n_dynamic, seed=123):
    rng = np.random.RandomState(seed)
    scenarios = []
    for _ in range(n_episodes):
        strat = "hotspot" if rng.rand() < 0.5 else "uniform"
        routine, dynamic = mission_gen.generate_episode_missions(
            n_routine=n_routine, n_dynamic_per_insertion=n_dynamic,
            n_insertions=3, sampling_strategy=strat,
        )
        scenarios.append((routine, dynamic))
    return scenarios


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
        },
        "results": results,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


# =======================================================================
# 方案 1: 单星 PPO
# =======================================================================
def run_single_ppo(cfg, mission_gen, scenarios, train_iters, device):
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
            n_insertions=3,
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
    metrics_list = []
    for routine, dynamic in scenarios:
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
    return _avg_metrics(metrics_list)


# =======================================================================
# 方案 2/3: 多星 (independent_ppo: coordinate=False; mappo: coordinate=True)
# =======================================================================
def run_multi(cfg, mission_gen, scenarios, train_iters, device, coordinate,
              episode_assignment=True, assign_w_load=0.1,
              assignment_capacity_mode="proportional",
              release_before_deadline_s=1800.0,
              team_reward_mix=0.0,
              load_balance_reward_coeff=0.0,
              team_completion_bonus=0.0,
              normalize_agent_rewards=False,
              global_state_mode="mean",
              global_state_task_stats=False):
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
        team_reward_mix=team_reward_mix if coordinate else 0.0,
        load_balance_reward_coeff=load_balance_reward_coeff if coordinate else 0.0,
        team_completion_bonus=team_completion_bonus if coordinate else 0.0,
        global_state_mode=global_state_mode if coordinate else "mean",
        global_state_task_stats=(coordinate and global_state_task_stats),
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

    for it in range(train_iters):
        routine, dynamic = mission_gen.generate_episode_missions(
            n_routine=int(np.random.choice(cfg.mission.routine_pool_sizes)),
            n_dynamic_per_insertion=int(np.random.choice(cfg.mission.dynamic_pool_sizes)),
            n_insertions=3,
        )
        opts = {"routine_missions": copy.deepcopy(routine),
                "dynamic_schedule": copy.deepcopy(dynamic)}
        res = env.reset(options=opts)
        cur_obs = {a: r[0] for a, r in res.items()}
        cur_info = {a: r[1] for a, r in res.items()}
        buf = MultiAgentRolloutBuffer()
        buf.init_agents(env.agent_ids)
        cur_obs, cur_info, _ = trainer.collect_rollout(
            env, buf, cfg.meta.rollout_steps, cur_obs, cur_info)
        trainer.update(buf, env.get_global_state())

    # 评估
    metrics_list = []
    env.set_eval_mode(True)   # 评估期启用 A1 败者改派 (训练期关闭以保信用分配)
    for routine, dynamic in scenarios:
        opts = {"routine_missions": copy.deepcopy(routine),
                "dynamic_schedule": copy.deepcopy(dynamic)}
        res = env.reset(options=opts)
        cur_obs = {a: r[0] for a, r in res.items()}
        cur_info = {a: r[1] for a, r in res.items()}
        max_steps = int(env.horizon_s / 10.0) + 100
        for _ in range(max_steps):
            actions = {}
            for aid in env.agent_ids:
                mask = cur_info[aid].get("action_mask", np.ones(act_dim))
                with torch.no_grad():
                    a, _, _ = model.actor.get_action(
                        torch.FloatTensor(cur_obs[aid]).unsqueeze(0).to(device),
                        torch.FloatTensor(mask).unsqueeze(0).to(device))
                actions[aid] = a.cpu().item()
            step_res = env.step(actions)
            for aid, (o, r, term, trunc, inf) in step_res.items():
                cur_obs[aid] = o
                cur_info[aid] = inf
            if env.is_done():
                break
        metrics_list.append(env.get_metrics())
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


def run_greedy_oracle(cfg, scenarios):
    from envs.multi_satellite_env import MultiSatelliteEnv

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

    metrics_list = []
    for routine, dynamic in scenarios:
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
    return _avg_metrics(metrics_list)


def main():
    parser = argparse.ArgumentParser(description="方案对比实验")
    parser.add_argument("--acled_path", type=str, default=None)
    parser.add_argument("--n_satellites", type=int, default=6)
    parser.add_argument("--train_iters", type=int, default=30)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--n_routine", type=int, default=200)
    parser.add_argument("--n_dynamic", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="runs/compare")
    parser.add_argument("--device", type=str, default="cpu")
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
    parser.add_argument("--run_oracle", action="store_true",
                        help="额外运行 Greedy-Oracle 集中式启发式参考")
    parser.add_argument("--experiment_tag", type=str, default="single_compare",
                        help="实验标签, 写入 manifest 方便批量对比")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    cfg = get_default_config()
    cfg.mappo.n_satellites = args.n_satellites

    acled = load_acled_shapefile(args.acled_path) if args.acled_path else None
    mission_gen = MissionGenerator(acled_df=acled, seed=args.seed)

    # 固定测试集 (三方案共用)
    scenarios = make_test_scenarios(mission_gen, args.eval_episodes,
                                    args.n_routine, args.n_dynamic, seed=args.seed + 1000)
    logger.info(f"测试场景数: {len(scenarios)}, 每个 {args.n_routine} routine + {args.n_dynamic}×3 dynamic")

    results = {}
    t0 = time.time()

    logger.info("=== [1/3] 单星 PPO ===")
    results["Single-PPO"] = run_single_ppo(cfg, mission_gen, scenarios, args.train_iters, device)

    logger.info("=== [2/3] 多星独立 PPO (无协同 baseline) ===")
    results["Indep-PPO"] = run_multi(cfg, mission_gen, scenarios, args.train_iters, device, coordinate=False)

    logger.info("=== [3/3] 多星 MAPPO (协同, 本方法) ===")
    results["MAPPO"] = run_multi(cfg, mission_gen, scenarios, args.train_iters, device,
                                 coordinate=True, episode_assignment=args.episode_assignment,
                                 assign_w_load=args.assign_w_load,
                                 assignment_capacity_mode=args.assignment_capacity_mode,
                                 release_before_deadline_s=args.release_before_deadline_s,
                                 team_reward_mix=args.team_reward_mix,
                                 load_balance_reward_coeff=args.load_balance_reward_coeff,
                                 team_completion_bonus=args.team_completion_bonus,
                                 normalize_agent_rewards=args.normalize_agent_rewards,
                                 global_state_mode=args.global_state_mode,
                                 global_state_task_stats=args.global_state_task_stats)

    if args.run_oracle:
        logger.info("=== [4/4] Greedy Oracle (集中式启发式参考) ===")
        results["Greedy-Oracle"] = run_greedy_oracle(cfg, scenarios)

    # 协同增益: MAPPO 完成数 / (N × 单星完成数)
    n_sat = args.n_satellites
    single_sched = results["Single-PPO"].get("n_scheduled", 0)
    for name in ["Indep-PPO", "MAPPO"]:
        multi_sched = results[name].get("n_scheduled", 0)
        gain = multi_sched / (n_sat * single_sched) if single_sched > 0 else 0.0
        results[name]["coordination_gain"] = gain
    oracle_sched = results.get("Greedy-Oracle", {}).get("n_scheduled", 0)
    if oracle_sched > 0:
        for name in ["Indep-PPO", "MAPPO"]:
            results[name]["oracle_relative_completion"] = (
                results[name].get("n_scheduled", 0) / oracle_sched
            )
        results["Greedy-Oracle"]["oracle_relative_completion"] = 1.0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    elapsed_s = time.time() - t0
    _write_manifest(out_dir, args, results, elapsed_s)

    # 控制台摘要
    print("\n" + "=" * 78)
    method_names = ["Single-PPO", "Indep-PPO", "MAPPO"]
    if "Greedy-Oracle" in results:
        method_names.append("Greedy-Oracle")
    print(f"{'指标':<32}" + "".join(f"{name:>14}" for name in method_names))
    print("-" * 78)
    show = [
        ("observation_success_rate", "观测成功率", "%"),
        ("dynamic_completion_rate", "动态完成率", "%"),
        ("routine_completion_rate", "常规完成率", "%"),
        ("total_reward", "累积奖励", ""),
        ("n_scheduled", "完成任务数", ""),
        ("n_duplicates", "重复观测数", ""),
        ("duplicate_rate", "重复观测率", "%"),
        ("load_balance_cv", "负载变异系数", ""),
        ("avg_off_nadir_deg", "平均off-nadir", "°"),
        ("avg_dynamic_response_s", "动态响应延迟", "s"),
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
