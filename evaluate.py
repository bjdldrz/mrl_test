"""
评估脚本
========
复现论文 Section 4.2 的实验:
  - 4.2.1 跨 RL 算法对比 (Table 6)
  - 4.2.2 不同动态任务规模的性能
  - 4.2.3 密集/分散空间分布对比 (Table 5)

用法:
    python evaluate.py --checkpoint checkpoints/mrl_dms_best.pt
    python evaluate.py --checkpoint checkpoints/mrl_dms_best.pt --experiment spatial
"""

import argparse
import logging
import sys
import os
import time
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_default_config
from algo.mrl_dms import MRLDMSTrainer
from envs.satellite_env import SatelliteSchedulingEnv
from data.mission_generator import MissionGenerator, load_acled_shapefile
from utils.metrics import compare_algorithms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluate")


def run_scheduling_episode(env, actor_critic, routine, dynamic, device="cpu"):
    """执行一个完整的调度 episode, 返回指标"""
    import copy
    obs, info = env.reset(options={
        "routine_missions": copy.deepcopy(routine),
        "dynamic_schedule": copy.deepcopy(dynamic),
    })

    start_time = time.time()
    done = False
    total_reward = 0.0

    while not done:
        action_mask = info.get("action_mask", np.ones(env.action_space.n))
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            mask_t = torch.FloatTensor(action_mask).unsqueeze(0).to(device)
            dist, _ = actor_critic(obs_t, mask_t)
            action = dist.sample()

        obs, reward, terminated, truncated, info = env.step(action.cpu().item())
        total_reward += reward
        done = terminated or truncated

    elapsed = time.time() - start_time
    metrics = env.get_metrics()
    metrics['computation_time'] = elapsed
    return metrics


def experiment_cross_rl(config, acled_df=None, n_episodes=10):
    """
    实验 4.2.1: 跨 RL 算法性能对比。
    对比 MRL-DMS vs PPO vs A2C vs DQN。
    """
    logger.info("=== 实验: 跨 RL 算法对比 (论文 Table 6) ===")

    mission_gen = MissionGenerator(acled_df=acled_df)
    results = defaultdict(list)

    routine_sizes = [100, 200, 300, 400, 500]
    dynamic_size = 50

    for n_routine in routine_sizes:
        for ep in range(n_episodes):
            routine, dynamic = mission_gen.generate_episode_missions(
                n_routine=n_routine,
                n_dynamic_per_insertion=dynamic_size,
            )

            # 这里应加载各算法的训练好的模型
            # 示例: 仅展示评估框架
            logger.info(
                f"  Routine={n_routine}, Episode={ep}: "
                f"生成 {len(routine)} routine + "
                f"{sum(len(d) for _, d in dynamic)} dynamic 任务"
            )

    return results


def experiment_mission_scale(config, acled_df=None, n_episodes=5):
    """
    实验 4.2.2: 不同任务规模的性能。
    """
    logger.info("=== 实验: 任务规模影响 (论文 Fig.7-10) ===")

    mission_gen = MissionGenerator(acled_df=acled_df)

    routine_sizes = [100, 200, 300, 400, 500]
    dynamic_sizes = [5, 10, 50, 100]

    results = []
    for n_r in routine_sizes:
        for n_d in dynamic_sizes:
            for _ in range(n_episodes):
                routine, dynamic = mission_gen.generate_episode_missions(
                    n_routine=n_r,
                    n_dynamic_per_insertion=n_d,
                )
                results.append({
                    'n_routine': n_r,
                    'n_dynamic': n_d * 3,
                    'n_total': n_r + n_d * 3,
                })
            logger.info(f"  Routine={n_r}, Dynamic={n_d}×3: 场景已生成")

    return results


def experiment_spatial(config, acled_df=None, n_episodes=5):
    """
    实验 4.2.3: 空间分布对比 (密集 vs 分散, 论文 Table 5)。
    """
    logger.info("=== 实验: 空间分布对比 (论文 Table 5) ===")

    mission_gen = MissionGenerator(acled_df=acled_df)

    for dist_type in ["hotspot", "uniform"]:
        for _ in range(n_episodes):
            routine, dynamic = mission_gen.generate_episode_missions(
                n_routine=300,
                n_dynamic_per_insertion=50,
                sampling_strategy=dist_type,
            )
            total_dyn = sum(len(d) for _, d in dynamic)
            logger.info(f"  {dist_type}: {len(routine)} routine + {total_dyn} dynamic")

    logger.info("空间分布实验框架准备完毕")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "cross_rl", "scale", "spatial"])
    parser.add_argument("--acled_path", type=str, default=None)
    parser.add_argument("--n_episodes", type=int, default=5)
    args = parser.parse_args()

    config = get_default_config()

    acled_df = None
    if args.acled_path:
        acled_df = load_acled_shapefile(args.acled_path)

    if args.experiment in ("all", "cross_rl"):
        experiment_cross_rl(config, acled_df, args.n_episodes)

    if args.experiment in ("all", "scale"):
        experiment_mission_scale(config, acled_df, args.n_episodes)

    if args.experiment in ("all", "spatial"):
        experiment_spatial(config, acled_df, args.n_episodes)


if __name__ == "__main__":
    main()
