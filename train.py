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

# ⚠️ 必须在 import numpy / torch 之前覆盖线程环境变量。
# 1) 服务器环境可能把 OMP_NUM_THREADS 设成非法值（空串/带空格），
#    导致 libgomp 报 "Invalid value" 并使单线程设置失效；
# 2) 父进程设好后，spawn 出的 worker 子进程会继承到合法的 "1"，
#    从根本上避免 N 进程 × N BLAS 线程 的过度订阅。
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
import logging
import sys

# 将项目根目录加入 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, get_default_config
from algo.mrl_dms import MRLDMSTrainer
from algo.ppo import PPOTrainer, RolloutBuffer
from models.actor_critic import ActorCritic
from envs.satellite_env import SatelliteSchedulingEnv
from data.mission_generator import MissionGenerator, load_acled_shapefile
from utils.experiment_dirs import safe_name, timestamp
from utils.json_utils import dump_json

import numpy as np
import torch
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train")


def train_mrl_dms(config: Config, acled_df=None, exp_name: str = None,
                  total_meta_iterations: int = None):
    """MRL-DMS 元训练"""
    trainer = MRLDMSTrainer(config)
    trainer.setup_data(acled_df)
    trainer.train(total_meta_iterations=total_meta_iterations, exp_name=exp_name)
    return trainer


def train_ppo_baseline(config: Config, acled_df=None, exp_name: str = None):
    """PPO baseline 训练 (用于对比实验)"""
    from tqdm import tqdm
    import csv

    mission_gen = MissionGenerator(acled_df=acled_df, seed=config.train.seed)

    sat_cfg = config.satellites[0]
    env = SatelliteSchedulingEnv(
        satellite_config=sat_cfg,
        max_action_dim=config.mission.max_action_dim,
        reward_config=config.reward,
    )

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    # 设备检测: 优先用配置指定的设备, "auto" 时 CUDA > MPS > CPU
    device = config.train.device
    if device == "auto":
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

    # ---- 日志目录 ----
    run_name = exp_name or f"ppo_{int(time.time())}"
    log_dir = os.path.join(config.train.log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    train_log_path = os.path.join(log_dir, "train_log.csv")

    fieldnames = ["update", "global_step", "rollout_reward", "policy_loss", "value_loss", "entropy"]
    best_reward = -float('inf')

    routine, dynamic = mission_gen.generate_episode_missions(
        n_routine=200, n_dynamic_per_insertion=50,
    )
    obs, info = env.reset(options={
        "routine_missions": routine,
        "dynamic_schedule": dynamic,
    })

    log_csv_f = open(train_log_path, "w", newline="")
    writer = csv.DictWriter(log_csv_f, fieldnames=fieldnames)
    writer.writeheader()

    pbar = tqdm(range(n_updates), desc="PPO Train", unit="update", dynamic_ncols=True)
    try:
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

            row = {
                "update": update,
                "global_step": update * rollout_steps,
                "rollout_reward": round(ep_reward, 4),
                "policy_loss": round(update_info['policy_loss'], 6),
                "value_loss": round(update_info['value_loss'], 6),
                "entropy": round(update_info['entropy'], 6),
            }
            writer.writerow(row)
            log_csv_f.flush()

            if ep_reward > best_reward:
                best_reward = ep_reward
                ckpt_dir = config.train.checkpoint_dir
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(actor_critic.state_dict(),
                           os.path.join(ckpt_dir, "ppo_best.pt"))

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
    finally:
        log_csv_f.close()
        pbar.close()

    # 写摘要
    summary = {
        "exp_name": run_name,
        "total_updates": n_updates,
        "best_reward": best_reward,
        "train_log": train_log_path,
    }
    with open(os.path.join(log_dir, "summary.json"), "w") as f:
        dump_json(summary, f, indent=2)

    logger.info(f"PPO baseline 训练完成, 日志: {log_dir}")
    return actor_critic


def configure_action_space(config: Config, requested_max_action_dim: int = None):
    """确保动作槽位能容纳训练/评估任务规模，避免动态任务被截断。"""
    train_required = (
        max(config.mission.routine_pool_sizes)
        + config.mission.dynamic_insertions_per_day * max(config.mission.dynamic_pool_sizes)
    )
    eval_required = (
        config.train.eval_n_routine
        + config.mission.dynamic_insertions_per_day * config.train.eval_n_dynamic_per_insertion
    )
    required = max(train_required, eval_required)

    if requested_max_action_dim is not None and requested_max_action_dim < required:
        raise ValueError(
            f"--max_action_dim={requested_max_action_dim} 小于当前训练/评估任务槽位需求 "
            f"{required}; 请调大该值,否则 dynamic_completion_rate 会被污染"
        )

    config.mission.max_action_dim = max(
        config.mission.max_action_dim,
        requested_max_action_dim or 0,
        required,
    )
    logger.info(
        "动作空间任务槽位: max_action_dim=%s, 训练需求=%s, 评估需求=%s",
        config.mission.max_action_dim,
        train_required,
        eval_required,
    )


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
    parser.add_argument("--device", type=str, default="auto",
                        help="计算设备; 本地 Mac 请用 cpu (MPS 的 recurrent backward 有 bug 会崩溃)")
    parser.add_argument("--meta_encoder_type", type=str, default=None,
                        choices=["lstm", "gru", "mlp", "transformer", "set_transformer"],
                        help="MRL-DMS 外循环反馈编码器, 默认使用配置中的 lstm")
    parser.add_argument("--meta_history_len", type=int, default=None,
                        help="Transformer/Set Transformer 外循环保留的反馈历史长度")
    parser.add_argument("--meta_transformer_heads", type=int, default=None,
                        help="Transformer/Set Transformer 注意力头数")
    parser.add_argument("--meta_transformer_layers", type=int, default=None,
                        help="Transformer/Set Transformer 层数")
    parser.add_argument("--mappo_n_satellites", type=int, default=None,
                        help="MRL-DMS 使用 MAPPO 内循环时的卫星数量; >1 启用多星 MAPPO")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="实验名称, 用于命名日志目录 runs/<exp_name>/")
    parser.add_argument("--run_tag", type=str, default=None,
                        help="实验标签; 未指定 exp_name 时用于生成 runs/<method>_<tag>_<timestamp>/")
    parser.add_argument("--append_timestamp", action="store_true",
                        help="给显式指定的 exp_name 也追加时间戳, 避免覆盖同名目录")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="训练日志根目录, 默认使用配置中的 runs/")
    parser.add_argument("--meta_iterations", type=int, default=None,
                        help="覆盖 MRL-DMS 外循环迭代次数, 便于消融 smoke")
    parser.add_argument("--max_action_dim", type=int, default=None,
                        help="动作空间任务槽位数; 默认按训练/评估任务规模自动扩容")
    parser.add_argument("--eval_n_routine", type=int, default=None,
                        help="MRL-DMS evaluate() 使用的常规任务数")
    parser.add_argument("--eval_n_dynamic", type=int, default=None,
                        help="MRL-DMS evaluate() 每次动态插入的任务数")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="MRL-DMS meta batch 并行 worker 数; 0 表示等于 meta_batch_size")
    parser.add_argument("--meta_batch_size", type=int, default=None,
                        help="每次元更新采样的任务数; 通常应与 num_workers 匹配")
    parser.add_argument("--inner_steps", type=int, default=None,
                        help="每个任务内循环 PPO/MAPPO 更新步数")
    parser.add_argument("--rollout_steps", type=int, default=None,
                        help="每个内循环 step 的 rollout 长度")
    parser.add_argument("--eval_interval", type=int, default=None,
                        help="每隔多少个 meta iteration 执行 evaluate; 调大可减少 CPU 评估开销")
    parser.add_argument("--save_interval", type=int, default=None,
                        help="每隔多少个 meta iteration 保存 checkpoint")
    parser.add_argument("--vtw_time_step_s", type=float, default=None,
                        help="VTW 采样步长; 越小越精确但 CPU 更重")
    parser.add_argument("--no_profile_timing", action="store_true",
                        help="关闭 MRL-DMS 阶段耗时日志和 profile 输出")
    args = parser.parse_args()

    # 加载配置
    config = get_default_config()
    config.train.seed = args.seed
    config.train.device = args.device
    if args.meta_encoder_type is not None:
        config.network.meta_encoder_type = args.meta_encoder_type
    if args.meta_history_len is not None:
        config.network.meta_history_len = args.meta_history_len
    if args.meta_transformer_heads is not None:
        config.network.meta_transformer_heads = args.meta_transformer_heads
    if args.meta_transformer_layers is not None:
        config.network.meta_transformer_layers = args.meta_transformer_layers
    if args.mappo_n_satellites is not None:
        config.mappo.n_satellites = args.mappo_n_satellites
    if args.log_dir is not None:
        config.train.log_dir = args.log_dir
    if args.eval_n_routine is not None:
        config.train.eval_n_routine = args.eval_n_routine
    if args.eval_n_dynamic is not None:
        config.train.eval_n_dynamic_per_insertion = args.eval_n_dynamic

    if args.fast:
        config.train.total_training_steps = 5000
        config.meta.rollout_steps = 256
        config.meta.inner_steps = 2
        config.meta.meta_batch_size = 2
        config.mission.routine_pool_sizes = [20]
        config.mission.dynamic_pool_sizes = [5]
        config.mission.max_action_dim = 50
        if args.mappo_n_satellites is None:
            config.mappo.n_satellites = 1      # 单星模式, 省去 5 颗卫星的 VTW 计算
        config.train.vtw_time_step_s = 60.0    # 步长 60s: LEO 过境快, 300s 会漏采窗口
        config.train.log_interval = 1
        config.train.eval_interval = 5
        logger.info("快速测试模式")

    # 显式命令行参数放在 --fast 之后应用，便于在 fast/smoke 中单独调优瓶颈。
    if args.num_workers is not None:
        config.train.num_workers = args.num_workers
    if args.meta_batch_size is not None:
        config.meta.meta_batch_size = args.meta_batch_size
    if args.inner_steps is not None:
        config.meta.inner_steps = args.inner_steps
    if args.rollout_steps is not None:
        config.meta.rollout_steps = args.rollout_steps
    if args.eval_interval is not None:
        config.train.eval_interval = args.eval_interval
    if args.save_interval is not None:
        config.train.save_interval = args.save_interval
    if args.vtw_time_step_s is not None:
        config.train.vtw_time_step_s = args.vtw_time_step_s
    if args.no_profile_timing:
        config.train.profile_timing = False

    configure_action_space(config, requested_max_action_dim=args.max_action_dim)

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

    # 输出目录命名: 默认每次训练生成唯一 exp_name; 显式 exp_name 默认保持旧行为.
    exp_name = args.exp_name
    if exp_name is None:
        tag = f"_{safe_name(args.run_tag)}" if args.run_tag else ""
        mode = "_fast" if args.fast else ""
        exp_name = f"{args.method}{tag}{mode}_{timestamp()}"
    elif args.append_timestamp:
        exp_name = f"{safe_name(exp_name)}_{timestamp()}"

    # 训练
    if args.method == "mrl_dms":
        train_mrl_dms(
            config,
            acled_df,
            exp_name=exp_name,
            total_meta_iterations=args.meta_iterations,
        )
    elif args.method == "ppo":
        train_ppo_baseline(config, acled_df, exp_name=exp_name)
    else:
        logger.info(f"Baseline {args.method} 可通过修改 PPO 训练器实现")
        # A2C / DQN 的实现可参照 PPO baseline 结构扩展
        raise NotImplementedError(f"{args.method} baseline 待实现")


if __name__ == "__main__":
    main()
