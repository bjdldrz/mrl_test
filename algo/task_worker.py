"""
单任务 PPO worker（顶层函数，可被 multiprocessing 序列化）
"""

import os

# 关键：限制每个 worker 进程底层线程库为单线程。
# 否则 N 个 worker 进程 × N 个 BLAS 线程 = N² 个线程争抢 N 个物理核，
# 严重 oversubscription，表现为 CPU 大部分时间低利用 + 偶发窄尖峰。
# 必须在 import numpy / torch 之前设置才能生效。
#
# 注意：用直接赋值而非 setdefault —— 服务器环境可能已把 OMP_NUM_THREADS
# 设成非法值（如空串/带空格），导致 libgomp 报 "Invalid value" 且单线程
# 设置失效。这里强制覆盖为合法的 "1"。
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # macOS Accelerate

import copy
import numpy as np
import torch

# 运行时再次确保（不依赖 import 顺序），每进程只用 1 个 intra-op 线程
torch.set_num_threads(1)


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
    n_ground_stations = args.get('n_ground_stations', 0)
    downlink_time_s  = args.get('downlink_time_s', 0.0)
    ground_station_configs = args.get('ground_station_configs')
    satellite_storage_capacity = args.get('satellite_storage_capacity', 0)
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
        n_ground_stations=n_ground_stations,
        downlink_time_s=downlink_time_s,
        ground_station_configs=ground_station_configs,
        satellite_storage_capacity=satellite_storage_capacity,
    )

    # VTW 由 env.propagator 进程内缓存（compute_vtw 内部 _vtw_cache）。
    # 坐标为随机均匀采样，跨任务命中率极低，故不再跨进程传递全局缓存
    # （那样每次都要 pickle 不断膨胀的字典，纯开销且导致主进程内存增长）。
    all_missions = list(routine)
    for _, dyn_batch in dynamic_schedule:
        all_missions.extend(dyn_batch)
    horizon_s = env.horizon_s
    step_s = env.vtw_time_step_s
    sat_name = sat_config.name
    pv = {}
    for m in all_missions:
        pv[(sat_name, m.id)] = env.propagator.compute_vtw(
            m.lat, m.lon, horizon_s, time_step_s=step_s
        )
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
    }


def run_mappo_task(args: dict) -> dict:
    """
    在独立进程中执行单个任务的多星 MAPPO 内循环适应 + 评估。

    与 run_single_task 对称：每个 worker 进程内重建 MultiSatelliteEnv(含 N 星)
    + MAPPOActorCritic，跑完整内循环，返回 actor 的参数差（FOMAML 外循环只聚合
    actor）。critic 用主进程传入的统一初始快照（不再跨任务累积）。

    参数（dict）：
        idx               : 任务索引
        routine           : list of Mission
        dynamic_schedule  : list of (time, list[Mission])
        actor_init_state  : numpy dict，已调制的 actor 初始 state_dict
        critic_init_state : numpy dict，critic 初始 state_dict（统一快照）
        sat_configs       : list[SatelliteConfig]，参与的 N 颗卫星
        reward_config     : RewardConfig
        vtw_time_step_s   : float
        max_action_dim    : int
        cfg_ppo / cfg_meta: dataclass
        obs_dim / action_dim / global_state_dim : int
        actor_hidden_dims / critic_hidden_dims  : list[int]
    """
    from envs.multi_satellite_env import MultiSatelliteEnv
    from models.mappo import MAPPOActorCritic
    from algo.mappo_trainer import MAPPOTrainer, MultiAgentRolloutBuffer

    idx               = args['idx']
    routine           = args['routine']
    dynamic_schedule  = args['dynamic_schedule']
    actor_init_state  = args['actor_init_state']
    critic_init_state = args['critic_init_state']
    sat_configs       = args['sat_configs']
    reward_config     = args['reward_config']
    vtw_time_step_s   = args['vtw_time_step_s']
    max_action_dim    = args['max_action_dim']
    n_ground_stations = args.get('n_ground_stations', 0)
    downlink_time_s   = args.get('downlink_time_s', 0.0)
    ground_station_configs = args.get('ground_station_configs')
    satellite_storage_capacity = args.get('satellite_storage_capacity', 0)
    enable_inter_satellite_transfer = args.get('enable_inter_satellite_transfer', False)
    inter_satellite_transfer_time_s = args.get('inter_satellite_transfer_time_s', 300.0)
    cfg_ppo           = args['cfg_ppo']
    cfg_meta          = args['cfg_meta']
    obs_dim           = args['obs_dim']
    action_dim        = args['action_dim']
    global_state_dim  = args['global_state_dim']
    actor_hidden_dims = args['actor_hidden_dims']
    critic_hidden_dims = args['critic_hidden_dims']
    device            = 'cpu'

    # 进程内重建多星环境与模型
    multi_env = MultiSatelliteEnv(
        satellite_configs=sat_configs,
        max_action_dim=max_action_dim,
        reward_config=reward_config,
        vtw_time_step_s=vtw_time_step_s,
        n_ground_stations=n_ground_stations,
        downlink_time_s=downlink_time_s,
        ground_station_configs=ground_station_configs,
        satellite_storage_capacity=satellite_storage_capacity,
        enable_inter_satellite_transfer=enable_inter_satellite_transfer,
        inter_satellite_transfer_time_s=inter_satellite_transfer_time_s,
    )
    model = MAPPOActorCritic(
        local_obs_dim=obs_dim,
        action_dim=action_dim,
        global_state_dim=global_state_dim,
        actor_hidden_dims=actor_hidden_dims,
        critic_hidden_dims=critic_hidden_dims,
    ).to(device)

    actor_init_torch = {k: torch.from_numpy(v.copy()) for k, v in actor_init_state.items()}
    critic_init_torch = {k: torch.from_numpy(v.copy()) for k, v in critic_init_state.items()}
    model.actor.load_state_dict(actor_init_torch)
    model.critic.load_state_dict(critic_init_torch)

    # VTW 预计算：每个子环境进程内独立计算（compute_vtw 内部按坐标缓存）
    all_missions = list(routine)
    for _, dyn_batch in dynamic_schedule:
        all_missions.extend(dyn_batch)
    for sub_env in multi_env.envs.values():
        pv = {}
        for m in all_missions:
            pv[(sub_env.sat_config.name, m.id)] = sub_env.propagator.compute_vtw(
                m.lat, m.lon, sub_env.horizon_s, time_step_s=sub_env.vtw_time_step_s
            )
        sub_env.precomputed_vtw = pv

    trainer = MAPPOTrainer(
        mappo_model=model,
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

    # ---- 内循环 MAPPO 适应 ----
    reset_result = multi_env.reset(options=reset_options)
    current_obs = {aid: r[0] for aid, r in reset_result.items()}
    current_infos = {aid: r[1] for aid, r in reset_result.items()}

    buffer = MultiAgentRolloutBuffer()
    buffer.init_agents(multi_env.agent_ids)

    for _ in range(cfg_meta.inner_steps):
        buffer.clear()
        buffer.init_agents(multi_env.agent_ids)
        current_obs, current_infos, _ = trainer.collect_rollout(
            multi_env, buffer, cfg_meta.rollout_steps,
            current_obs, current_infos,
        )
        last_gs = multi_env.get_global_state()
        trainer.update(buffer, last_gs)

    # ---- 评估适应后策略 ----
    reset_result = multi_env.reset(options=reset_options)
    current_obs = {aid: r[0] for aid, r in reset_result.items()}
    current_infos = {aid: r[1] for aid, r in reset_result.items()}

    eval_reward = 0.0
    max_steps = int(multi_env.horizon_s / 10.0) + 100
    for _ in range(max_steps):
        actions = {}
        for aid in multi_env.agent_ids:
            obs = current_obs[aid]
            mask = current_infos[aid].get("action_mask", np.ones(multi_env.action_dim))
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0)
                mask_t = torch.FloatTensor(mask).unsqueeze(0)
                action, _, _ = model.actor.get_action(obs_t, mask_t)
            actions[aid] = action.cpu().item()

        step_results = multi_env.step(actions)
        for aid, (obs, reward, term, trunc, info) in step_results.items():
            eval_reward += reward
            current_obs[aid] = obs
            current_infos[aid] = info

        if multi_env.is_done():
            break

    eval_metrics = multi_env.get_metrics()

    # actor 参数差（外循环只聚合 actor）
    adapted_actor = model.actor.state_dict()
    param_names = {name for name, _ in model.actor.named_parameters()}
    param_diff_np = {
        name: (adapted_actor[name] - actor_init_torch[name]).cpu().numpy()
        for name in param_names
    }
    steps_consumed = cfg_meta.inner_steps * cfg_meta.rollout_steps

    return {
        'idx': idx,
        'param_diff_np': param_diff_np,
        'eval_reward': eval_reward,
        'eval_metrics': eval_metrics,
        'steps_consumed': steps_consumed,
    }
