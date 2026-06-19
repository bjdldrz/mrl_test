"""
主训练脚本
==========
用法:
    # MRL-DMS 元训练 (论文核心方法)
    python train.py --method mrl_dms

    # PPO baseline (论文对比实验)
    python train.py --method ppo

    # 指定 ACLED 数据路径
    python train.py --method mrl_dms --acled_path data/DynamicMission/DynamicMission.shp

    # 快速测试 (小规模)
    python train.py --method mrl_dms --fast
"""

import argparse
import logging
import sys
import os

# 将项目根目录加入 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, get_default_config
from algo.mrl_dms import MRLDMSTrainer
from algo.ppo import PPOTrainer, RolloutBuffer
from models.actor_critic import ActorCritic
from envs.satellite_env import SatelliteSchedulingEnv
from data.mission_generator import MissionGenerator, load_acled_shapefile
from utils.metrics import MetricsTracker

import numpy as np
import torch
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train")


def train_mrl_dms(config: Config, acled_df=None):
    """MRL-DMS 元训练"""
    trainer = MRLDMSTrainer(config)
    trainer.setup_data(acled_df)
    trainer.train()
    return trainer


def train_ppo_baseline(config: Config, acled_df=None):
    """PPO baseline 训练 (用于对比实验)"""
    from tqdm import tqdm

    mission_gen = MissionGenerator(acled_df=acled_df, seed=config.train.seed)

    sat_cfg = config.satellites[0]
    env = SatelliteSchedulingEnv(
        satellite_config=sat_cfg,
        max_action_dim=config.mission.max_action_dim,
        reward_config=config.reward,
    )

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    # 设备检测: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"=== PPO Baseline 训练, 设备: {device} ===")

    actor_critic = ActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=config.network.hidden_layers,
    )

    ppo = PPOTrainer(
        actor_critic=actor_critic,
        lr=config.ppo.learning_rate,
        gamma=config.ppo.discount_factor,
        gae_lambda=config.ppo.gae_lambda,
        clip_ratio=config.ppo.clip_ratio,
        entropy_coeff=config.ppo.entropy_coeff,
        value_loss_coeff=config.ppo.value_loss_coeff,
        ppo_epochs=config.ppo.ppo_epochs,
        batch_size=config.ppo.batch_size,
        device=device,
    )

    total_steps = config.train.total_training_steps
    rollout_steps = config.meta.rollout_steps
    n_updates = total_steps // rollout_steps

    routine, dynamic = mission_gen.generate_episode_missions(
        n_routine=200, n_dynamic_per_insertion=50,
    )
    obs, info = env.reset(options={
        "routine_missions": routine,
        "dynamic_schedule": dynamic,
    })

    pbar = tqdm(range(n_updates), desc="PPO Train", unit="update", dynamic_ncols=True)
    for update in pbar:
        buffer = RolloutBuffer()
        obs, info, ep_reward = ppo.collect_rollout(
            env, buffer, rollout_steps, obs, info
        )

        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(torch.device(device))
            last_value = actor_critic.get_value(obs_t).cpu().item()

        update_info = ppo.update(buffer, last_value)
        pbar.set_postfix(
            R=f"{ep_reward:.1f}",
            ploss=f"{update_info['policy_loss']:.3f}",
            ent=f"{update_info['entropy']:.2f}",
        )

        # 定期重置环境 (新的任务场景)
        if update % 10 == 0 and update > 0:
            routine, dynamic = mission_gen.generate_episode_missions(
                n_routine=np.random.choice(config.mission.routine_pool_sizes),
                n_dynamic_per_insertion=np.random.choice(config.mission.dynamic_pool_sizes),
            )
            obs, info = env.reset(options={
                "routine_missions": routine,
                "dynamic_schedule": dynamic,
            })

    logger.info("PPO baseline 训练完成")
    return actor_critic


def main():
    parser = argparse.ArgumentParser(description="MRL-DMS Training")
    parser.add_argument("--method", type=str, default="mrl_dms",
                        choices=["mrl_dms", "ppo", "a2c", "dqn"],
                        help="训练方法")
    parser.add_argument("--acled_path", type=str, default=None,
                        help="ACLED Shapefile 路径 (.shp)")
    parser.add_argument("--fast", action="store_true",
                        help="快速测试模式 (减少训练步数)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 加载配置
    config = get_default_config()
    config.train.seed = args.seed

    if args.fast:
        config.train.total_training_steps = 5000
        config.meta.rollout_steps = 256
        config.meta.inner_steps = 2
        config.meta.meta_batch_size = 2
        config.mission.routine_pool_sizes = [50]
        config.mission.dynamic_pool_sizes = [5]
        config.mission.max_action_dim = 100
        config.train.log_interval = 1
        config.train.eval_interval = 5
        logger.info("快速测试模式")

    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 加载 ACLED 数据 (可选)
    acled_df = None
    if args.acled_path:
        acled_df = load_acled_shapefile(args.acled_path)
        logger.info(f"已加载 ACLED 数据: {len(acled_df)} 条")
    else:
        logger.info("未指定 ACLED 数据, 将使用合成动态任务")

    # 训练
    if args.method == "mrl_dms":
        train_mrl_dms(config, acled_df)
    elif args.method == "ppo":
        train_ppo_baseline(config, acled_df)
    else:
        logger.info(f"Baseline {args.method} 可通过修改 PPO 训练器实现")
        # A2C / DQN 的实现可参照 PPO baseline 结构扩展
        raise NotImplementedError(f"{args.method} baseline 待实现")


if __name__ == "__main__":
    main()
