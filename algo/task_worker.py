"""
单任务 PPO worker（顶层函数，可被 multiprocessing 序列化）
"""

import copy
import numpy as np
import torch


def run_single_task(args: dict) -> dict:
    """
    在独立进程中执行单个任务的内循环 PPO 适应 + 评估。

    参数（通过 dict 传入，避免 pickle 问题）：
        idx              : 任务索引
        routine          : list of Mission
        dynamic_schedule : list of (time, list[Mission])
        init_state       : OrderedDict，actor_critic 初始 state_dict
        sat_config       : SatelliteConfig
        reward_config    : RewardConfig
        vtw_time_step_s  : float
        max_action_dim   : int
        vtw_cache        : dict，只读的全局 VTW 缓存快照（主进程在启动前传入）
        cfg_ppo          : PPOConfig dataclass
        cfg_meta         : MetaConfig dataclass
        obs_dim          : int
        action_dim       : int
        hidden_dims      : list[int]
        activation       : str
        device           : str（子进程强制用 cpu）
    """
    from envs.satellite_env import SatelliteSchedulingEnv
    from models.actor_critic import ActorCritic
    from algo.ppo import PPOTrainer, RolloutBuffer

    idx              = args['idx']
    routine          = args['routine']
    dynamic_schedule = args['dynamic_schedule']
    init_state       = args['init_state']
    sat_config       = args['sat_config']
    reward_config    = args['reward_config']
    vtw_time_step_s  = args['vtw_time_step_s']
    max_action_dim   = args['max_action_dim']
    vtw_cache        = args['vtw_cache']        # 只读快照
    cfg_ppo          = args['cfg_ppo']
    cfg_meta         = args['cfg_meta']
    obs_dim          = args['obs_dim']
    action_dim       = args['action_dim']
    hidden_dims      = args['hidden_dims']
    activation       = args['activation']
    device           = 'cpu'                    # 子进程只用 cpu

    # 每个进程独立构建模型和环境（无 GIL 限制）
    model = ActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        activation=activation,
    ).to(device)
    # init_state 是 numpy dict（减少跨进程 pickle 体积），转回 torch
    init_state_torch = {k: torch.from_numpy(v.copy()) for k, v in init_state.items()}
    model.load_state_dict(init_state_torch)

    env = SatelliteSchedulingEnv(
        satellite_config=sat_config,
        max_action_dim=max_action_dim,
        reward_config=reward_config,
        vtw_time_step_s=vtw_time_step_s,
    )

    # 从传入的缓存快照填充 VTW（不再重新计算已知坐标）
    all_missions = list(routine)
    for _, dyn_batch in dynamic_schedule:
        all_missions.extend(dyn_batch)
    horizon_s = env.horizon_s
    step_s = env.vtw_time_step_s
    sat_name = sat_config.name
    pv = {}
    new_vtw = {}   # 本进程新计算的 VTW，结束后返回给主进程合并到缓存
    for m in all_missions:
        coord_key = (sat_name, round(m.lat, 4), round(m.lon, 4),
                     int(horizon_s), int(step_s))
        if coord_key in vtw_cache:
            pv[(sat_name, m.id)] = vtw_cache[coord_key]
        else:
            vtw = env.propagator.compute_vtw(
                m.lat, m.lon, horizon_s, time_step_s=step_s
            )
            pv[(sat_name, m.id)] = vtw
            new_vtw[coord_key] = vtw
    env.precomputed_vtw = pv

    inner_ppo = PPOTrainer(
        actor_critic=model,
        lr=cfg_ppo.learning_rate,
        gamma=cfg_ppo.discount_factor,
        gae_lambda=cfg_ppo.gae_lambda,
        clip_ratio=cfg_ppo.clip_ratio,
        entropy_coeff=cfg_ppo.entropy_coeff,
        value_loss_coeff=cfg_ppo.value_loss_coeff,
        ppo_epochs=cfg_ppo.ppo_epochs,
        batch_size=cfg_ppo.batch_size,
        device=device,
    )

    reset_options = {
        "routine_missions": copy.deepcopy(routine),
        "dynamic_schedule": copy.deepcopy(dynamic_schedule),
    }
    obs, info = env.reset(options=reset_options)
    buffer = RolloutBuffer()

    for _ in range(cfg_meta.inner_steps):
        buffer.clear()
        obs, info, _ = inner_ppo.collect_rollout(
            env, buffer, cfg_meta.rollout_steps, obs, info,
            reset_options=reset_options,
        )
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            last_value = model.get_value(obs_t).cpu().item()
        inner_ppo.update(buffer, last_value)

    # 评估
    obs, info = env.reset(options=reset_options)
    eval_reward = 0.0
    done = False
    max_steps = int(env.horizon_s / 10.0) + 100
    for _ in range(max_steps):
        if done:
            break
        action_mask = info.get("action_mask", np.ones(env.action_space.n))
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            mask_t = torch.FloatTensor(action_mask).unsqueeze(0)
            action, _, _, _ = model.get_action_and_value(obs_t, mask_t)
        obs, reward, terminated, truncated, info = env.step(action.cpu().item())
        eval_reward += reward
        done = terminated or truncated

    eval_metrics = env.get_metrics()

    adapted_state = model.state_dict()
    param_names = {name for name, _ in model.named_parameters()}
    param_diff_np = {
        name: (adapted_state[name] - init_state_torch[name]).cpu().numpy()
        for name in param_names
    }
    steps_consumed = cfg_meta.inner_steps * cfg_meta.rollout_steps

    return {
        'idx': idx,
        'param_diff_np': param_diff_np,
        'eval_reward': eval_reward,
        'eval_metrics': eval_metrics,
        'steps_consumed': steps_consumed,
        'new_vtw': new_vtw,
    }
