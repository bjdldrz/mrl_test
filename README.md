# DAS-CVA-MAPPO Development Branch

This branch is a cleaned working base for the DAS-CVA-MAPPO redesign.

The retained code is limited to the pieces that are still useful for the new
version:

- `das_cva_mappo/`: DAS-CVA-MAPPO implementation with an action-set-aware
  policy, PPO buffer snapshots, a learnable CVA candidate edge scorer, and
  rollout-advantage auxiliary scorer updates with hard-negative candidate
  sampling plus conflict/load target shaping. Compatibility-layer access is
  isolated behind a DAS candidate adapter, with V0.19 set-transformer action
  matching, action-type gating, idle auxiliary loss, DAS-native worker
  support, and typed candidate exposure defaults.
- `cva_mappo_v2/`: compatibility layer for candidate generation and the
  scheduling environment used by current DAS iterations.
- `envs/`: single- and multi-satellite scheduling environments.
- `data/`: mission generation and orbit/VTW utilities.
- `algo/mappo_trainer.py`: MAPPO rollout buffer and PPO update loop.
- `models/mappo.py`: shared MAPPO actor and centralized critic.
- `utils/`: scenario cache, JSON, output-directory, and experiment helpers.
- `precompute_scenarios.py`: scenario and VTW cache precomputation.
- `DynamicMission/`: local ACLED shapefile used by experiments.

Historical single-satellite MRL-DMS training code, old PPO/MAML code, tracked
experiment outputs, and obsolete experiment reports are intentionally removed
from this branch.

## Setup

Run commands from a Python 3.10+ environment with the project dependencies
installed:

```bash
python3 -m pip install -r requirements.txt
```

## DAS V0.30

Run the current DAS action-set policy with hybrid CVA edge scoring,
set-transformer action matching, action-type gating, and idle auxiliary PPO
loss. V0.30 makes candidate scoring downlink-aware before an observation is
selected, keeps post-hoc dynamic downlink priority disabled by default, and
adds per-dynamic-task diagnostics. The default runner uses parallel rollout
workers, parallel CPU evaluation, and `cuda:0` for PPO/candidate-scorer
training:

```bash
python3 -m das_cva_mappo.run_experiment \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --n_satellites 6 \
  --train_iters 30 \
  --eval_episodes 10 \
  --n_routine 600 \
  --n_dynamic 150 \
  --n_ground_stations 4 \
  --downlink_time_s 30 \
  --satellite_storage_capacity 30 \
  --enable_inter_satellite_transfer \
  --inter_satellite_transfer_time_s 300 \
  --routine_slots 64 \
  --dynamic_slots 32 \
  --flex_slots 32 \
  --routine_candidate_owners 1 \
  --dynamic_candidate_owners 6 \
  --urgent_candidate_owners 6 \
  --stale_candidate_owners 6 \
  --slot_selection_mode typed \
  --ownership_mask_mode soft \
  --candidate_owner_bonus 0.06 \
  --dynamic_broadcast_window_s 3600 \
  --dynamic_takeover_margin_s 300 \
  --candidate_wait_penalty 0.08 \
  --candidate_storage_penalty 0.08 \
  --candidate_dynamic_urgency_bonus 0.12 \
  --candidate_dynamic_response_bonus 0.24 \
  --candidate_dynamic_wait_penalty 0.20 \
  --dynamic_response_target_s 3600 \
  --dynamic_current_slot_bonus 0.65 \
  --dynamic_window_wait_weight 0.75 \
  --downlink_queue_target_s 3600 \
  --candidate_downlink_queue_penalty 0.10 \
  --candidate_downlink_miss_penalty 0.20 \
  --candidate_dynamic_delivery_bonus 0.24 \
  --candidate_dynamic_delivery_delay_penalty 0.20 \
  --no_dynamic_downlink_priority \
  --allocator_wait_penalty 0.10 \
  --allocator_stale_rescue_bonus 0.25 \
  --allocator_dynamic_urgency_bonus 0.10 \
  --allocator_dynamic_response_bonus 0.24 \
  --allocator_dynamic_wait_penalty 0.20 \
  --assignment_replan_trigger periodic,dynamic,stale_owner,deadline \
  --matcher set_transformer \
  --idle_valid_penalty 0.0 \
  --idle_aux_coeff 0.05 \
  --action_feature_mode full \
  --candidate_scorer_mode hybrid \
  --candidate_scorer_mix 0.35 \
  --candidate_warmup_edges 4096 \
  --candidate_warmup_epochs 2 \
  --candidate_aux_rank_weight 0.2 \
  --candidate_hard_negative_samples 2 \
  --candidate_hard_negative_margin 0.25 \
  --candidate_aux_conflict_penalty 0.5 \
  --candidate_aux_load_penalty 0.1 \
  --candidate_adapter_mode v2_compat \
  --candidate_dropout_prob 0.05 \
  --rollout_steps 512 \
  --train_env_workers 16 \
  --split_rollout_steps_across_workers \
  --ppo_epochs 4 \
  --ppo_batch_size 512 \
  --eval_max_steps 8000 \
  --eval_device cpu \
  --eval_workers 24 \
  --torch_num_threads 1 \
  --vtw_time_step_s 60 \
  --out_dir runs/das_cva_mappo \
  --run_name das_v0_30 \
  --device cuda:0
```

By default `--train_env_workers 16`, `--eval_workers 24`, `--eval_device cpu`,
and `--device cuda:0` are used. `--rollout_steps` is split across
`--train_env_workers`, so the total rollout budget per iteration stays at 512 in
the command above. Use
`--rollout_steps_per_worker` only when each worker should collect the full
rollout budget. If `--eval_device` is a CUDA device, DAS automatically uses one
eval worker to avoid multiple processes competing for the same GPU. On a CPU-only
machine, explicitly pass `--device cpu`.

Primary DAS ablation knobs:

- `--matcher additive|dot|set_transformer`
- `--action_feature_mode full|minimal|no_score`
- `--no_candidate_score_feature`
- `--no_set_context`
- `--no_action_type_gate`
- `--idle_valid_penalty`
- `--idle_aux_coeff`
- `--candidate_dropout_prob`
- `--candidate_scorer_mode v2_heuristic|learned|hybrid`
- `--candidate_scorer_mix`
- `--candidate_warmup_edges`
- `--candidate_warmup_epochs`
- `--no_candidate_aux_update`
- `--candidate_aux_rank_weight`
- `--candidate_aux_min_edges`
- `--candidate_hard_negative_samples`
- `--candidate_hard_negative_include_invalid`
- `--candidate_hard_negative_margin`
- `--candidate_hard_negative_value_weight`
- `--candidate_aux_conflict_penalty`
- `--candidate_aux_load_penalty`
- `--candidate_adapter_mode v2_compat`
- `--candidate_wait_penalty`
- `--candidate_storage_penalty`
- `--candidate_dynamic_urgency_bonus`
- `--candidate_dynamic_response_bonus`
- `--candidate_dynamic_wait_penalty`
- `--dynamic_current_slot_bonus`
- `--dynamic_window_wait_weight`
- `--no_downlink_aware_candidate_score`
- `--downlink_queue_target_s`
- `--candidate_downlink_queue_penalty`
- `--candidate_downlink_miss_penalty`
- `--candidate_dynamic_delivery_bonus`
- `--candidate_dynamic_delivery_delay_penalty`
- `--dynamic_downlink_priority`
- `--no_dynamic_downlink_priority`
- `--allocator_wait_penalty`
- `--allocator_stale_rescue_bonus`
- `--allocator_dynamic_urgency_bonus`
- `--allocator_dynamic_response_bonus`
- `--allocator_dynamic_wait_penalty`
- `--dynamic_takeover_margin_s`
- `--routine_candidate_owners`
- `--dynamic_candidate_owners`
- `--urgent_candidate_owners`
- `--stale_candidate_owners`
- `--slot_selection_mode mixed|typed`
- `--ownership_mask_mode soft|hard`
- `--train_env_workers`
- `--split_rollout_steps_across_workers`
- `--rollout_steps_per_worker`
- `--eval_workers`

Auxiliary scorer updates apply to `learned` and `hybrid` scorer modes. Runs
with `--candidate_scorer_mode v2_heuristic` automatically record the auxiliary
path as disabled.

## Staged Optimization Runs

Use these commands after generating the scenario cache below. Stage 1 is a
diagnostic run; stages 2-4 progressively enable the candidate/owner, dynamic,
and storage-pressure optimizations.

For V0.30 dynamic-response checks, compare `avg_dynamic_response_s`,
`avg_downlink_queue_s`, `dynamic_task_candidate_seen_rate`,
`dynamic_task_policy_selected_rate`, `dynamic_task_downlink_queue_block_rate`,
and `avg_dynamic_task_downlink_queue_s`. The per-episode task-level file
`eval_dynamic_task_diagnostics.json` records whether each arrived dynamic task
was seen in candidates, currently executable when seen, selected by the policy,
observed, downlinked, or blocked by the downlink queue.

Stage 1: slot-invalid diagnosis. Inspect `slot_invalid_*`,
`avg_filled_invalid_slots`, `eval_valid_decision_rate`, and
`stale_owner_rate` in `comparison_results.json`.

```bash
python3 -m das_cva_mappo.run_experiment \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --scenario_cache_dir runs/scenario_cache/das_cva_stress_seed42 \
  --vtw_cache_dir runs/scenario_cache/das_cva_stress_seed42/vtw_cache \
  --n_satellites 6 \
  --train_iters 0 \
  --train_env_workers 16 \
  --eval_episodes 5 \
  --n_routine 600 \
  --n_dynamic 150 \
  --n_ground_stations 4 \
  --downlink_time_s 30 \
  --satellite_storage_capacity 30 \
  --enable_inter_satellite_transfer \
  --inter_satellite_transfer_time_s 300 \
  --routine_slots 64 \
  --dynamic_slots 32 \
  --flex_slots 32 \
  --slot_selection_mode typed \
  --ownership_mask_mode soft \
  --candidate_scorer_mode v2_heuristic \
  --eval_max_steps 8000 \
  --eval_device cpu \
  --eval_workers 24 \
  --torch_num_threads 1 \
  --vtw_time_step_s 60 \
  --out_dir runs/das_cva_mappo \
  --run_name das_stage1_slot_diagnosis \
  --device cuda:0 \
  --no_progress
```

Stage 2: candidate/owner repair. This keeps the heuristic scorer so the effect
comes from candidate exposure, stale-owner rescue, and earlier feasible-window
allocation.

```bash
python3 -m das_cva_mappo.run_experiment \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --scenario_cache_dir runs/scenario_cache/das_cva_stress_seed42 \
  --vtw_cache_dir runs/scenario_cache/das_cva_stress_seed42/vtw_cache \
  --n_satellites 6 \
  --train_iters 20 \
  --eval_episodes 10 \
  --n_routine 600 \
  --n_dynamic 150 \
  --n_ground_stations 4 \
  --downlink_time_s 30 \
  --satellite_storage_capacity 30 \
  --enable_inter_satellite_transfer \
  --inter_satellite_transfer_time_s 300 \
  --routine_slots 48 \
  --dynamic_slots 48 \
  --flex_slots 32 \
  --routine_candidate_owners 1 \
  --dynamic_candidate_owners 8 \
  --urgent_candidate_owners 8 \
  --stale_candidate_owners 8 \
  --slot_selection_mode typed \
  --ownership_mask_mode soft \
  --candidate_owner_bonus 0.08 \
  --assignment_replan_interval_s 900 \
  --assignment_replan_horizon_s 21600 \
  --assignment_replan_trigger periodic,dynamic,stale_owner,deadline \
  --release_before_deadline_s 7200 \
  --dynamic_broadcast_window_s 7200 \
  --dynamic_takeover_margin_s 120 \
  --candidate_wait_penalty 0.10 \
  --candidate_dynamic_urgency_bonus 0.16 \
  --candidate_dynamic_response_bonus 0.24 \
  --candidate_dynamic_wait_penalty 0.20 \
  --dynamic_response_target_s 3600 \
  --dynamic_current_slot_bonus 0.65 \
  --dynamic_window_wait_weight 0.75 \
  --allocator_wait_penalty 0.14 \
  --allocator_stale_rescue_bonus 0.35 \
  --allocator_dynamic_urgency_bonus 0.16 \
  --allocator_dynamic_response_bonus 0.24 \
  --allocator_dynamic_wait_penalty 0.20 \
  --candidate_scorer_mode v2_heuristic \
  --rollout_steps 512 \
  --train_env_workers 16 \
  --split_rollout_steps_across_workers \
  --ppo_epochs 4 \
  --ppo_batch_size 512 \
  --eval_max_steps 8000 \
  --eval_device cpu \
  --eval_workers 24 \
  --torch_num_threads 1 \
  --vtw_time_step_s 60 \
  --out_dir runs/das_cva_mappo \
  --run_name das_stage2_candidate_owner_repair \
  --device cuda:0
```

Stage 3: dynamic-task optimization. This turns the DAS hybrid scorer back on
and lets invalid hard negatives teach the edge scorer which future-only or
stale candidates should move down.

```bash
python3 -m das_cva_mappo.run_experiment \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --scenario_cache_dir runs/scenario_cache/das_cva_stress_seed42 \
  --vtw_cache_dir runs/scenario_cache/das_cva_stress_seed42/vtw_cache \
  --n_satellites 6 \
  --train_iters 40 \
  --eval_episodes 10 \
  --n_routine 600 \
  --n_dynamic 150 \
  --n_ground_stations 4 \
  --downlink_time_s 30 \
  --satellite_storage_capacity 30 \
  --enable_inter_satellite_transfer \
  --inter_satellite_transfer_time_s 300 \
  --routine_slots 48 \
  --dynamic_slots 48 \
  --flex_slots 32 \
  --routine_candidate_owners 1 \
  --dynamic_candidate_owners 8 \
  --urgent_candidate_owners 8 \
  --stale_candidate_owners 8 \
  --slot_selection_mode typed \
  --ownership_mask_mode soft \
  --candidate_owner_bonus 0.08 \
  --assignment_replan_interval_s 900 \
  --assignment_replan_horizon_s 21600 \
  --assignment_replan_trigger periodic,dynamic,stale_owner,deadline \
  --release_before_deadline_s 7200 \
  --dynamic_broadcast_window_s 7200 \
  --dynamic_takeover_margin_s 120 \
  --candidate_wait_penalty 0.10 \
  --candidate_dynamic_urgency_bonus 0.18 \
  --allocator_wait_penalty 0.14 \
  --allocator_stale_rescue_bonus 0.35 \
  --allocator_dynamic_urgency_bonus 0.18 \
  --matcher set_transformer \
  --idle_aux_coeff 0.05 \
  --action_feature_mode full \
  --candidate_scorer_mode hybrid \
  --candidate_scorer_mix 0.45 \
  --candidate_warmup_edges 8192 \
  --candidate_warmup_epochs 3 \
  --candidate_aux_rank_weight 0.30 \
  --candidate_hard_negative_samples 4 \
  --candidate_hard_negative_include_invalid \
  --candidate_hard_negative_margin 0.30 \
  --candidate_aux_conflict_penalty 0.5 \
  --candidate_aux_load_penalty 0.1 \
  --candidate_dropout_prob 0.05 \
  --rollout_steps 512 \
  --train_env_workers 16 \
  --split_rollout_steps_across_workers \
  --ppo_epochs 4 \
  --ppo_batch_size 512 \
  --eval_max_steps 8000 \
  --eval_device cpu \
  --eval_workers 24 \
  --torch_num_threads 1 \
  --vtw_time_step_s 60 \
  --out_dir runs/das_cva_mappo \
  --run_name das_stage3_dynamic_hybrid \
  --device cuda:0
```

Stage 4: storage/downlink pressure. Compare this with stage 3 on
`n_storage_expired_drops`, `avg_downlink_queue_s`, `n_relay_storage_images`,
and raw completion rates.

```bash
python3 -m das_cva_mappo.run_experiment \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --scenario_cache_dir runs/scenario_cache/das_cva_stress_seed42 \
  --vtw_cache_dir runs/scenario_cache/das_cva_stress_seed42/vtw_cache \
  --n_satellites 6 \
  --train_iters 40 \
  --eval_episodes 10 \
  --n_routine 600 \
  --n_dynamic 150 \
  --n_ground_stations 4 \
  --downlink_time_s 30 \
  --satellite_storage_capacity 30 \
  --enable_inter_satellite_transfer \
  --inter_satellite_transfer_time_s 300 \
  --routine_slots 48 \
  --dynamic_slots 48 \
  --flex_slots 32 \
  --routine_candidate_owners 1 \
  --dynamic_candidate_owners 8 \
  --urgent_candidate_owners 8 \
  --stale_candidate_owners 8 \
  --slot_selection_mode typed \
  --ownership_mask_mode soft \
  --candidate_owner_bonus 0.08 \
  --assignment_replan_interval_s 900 \
  --assignment_replan_horizon_s 21600 \
  --assignment_replan_trigger periodic,dynamic,stale_owner,deadline \
  --release_before_deadline_s 7200 \
  --dynamic_broadcast_window_s 7200 \
  --dynamic_takeover_margin_s 120 \
  --candidate_wait_penalty 0.10 \
  --candidate_storage_penalty 0.16 \
  --candidate_dynamic_urgency_bonus 0.18 \
  --allocator_wait_penalty 0.14 \
  --allocator_stale_rescue_bonus 0.35 \
  --allocator_dynamic_urgency_bonus 0.18 \
  --matcher set_transformer \
  --idle_aux_coeff 0.05 \
  --action_feature_mode full \
  --candidate_scorer_mode hybrid \
  --candidate_scorer_mix 0.45 \
  --candidate_warmup_edges 8192 \
  --candidate_warmup_epochs 3 \
  --candidate_aux_rank_weight 0.30 \
  --candidate_hard_negative_samples 4 \
  --candidate_hard_negative_include_invalid \
  --candidate_hard_negative_margin 0.30 \
  --candidate_aux_conflict_penalty 0.5 \
  --candidate_aux_load_penalty 0.20 \
  --candidate_dropout_prob 0.05 \
  --rollout_steps 512 \
  --train_env_workers 16 \
  --split_rollout_steps_across_workers \
  --ppo_epochs 4 \
  --ppo_batch_size 512 \
  --eval_max_steps 8000 \
  --eval_device cpu \
  --eval_workers 24 \
  --torch_num_threads 1 \
  --vtw_time_step_s 60 \
  --out_dir runs/das_cva_mappo \
  --run_name das_stage4_storage_pressure \
  --device cuda:0
```

Version notes are tracked in
`docs/VERSION_HISTORY.md`.

## Scenario Cache

Generate reusable scenarios and VTW cache:

```bash
python precompute_scenarios.py \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --n_satellites 6 \
  --n_train_scenarios 200 \
  --n_eval_scenarios 10 \
  --n_routine 300 \
  --n_dynamic 100 \
  --n_ground_stations 4 \
  --curriculum_stages 300:75,600:150,900:225,1200:300 \
  --vtw_time_step_s 60 \
  --vtw_workers 12 \
  --out_dir runs/scenario_cache/das_cva_stress_seed42
```

```bash
python precompute_scenarios.py \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --n_satellites 6 \
  --n_train_scenarios 200 \
  --n_eval_scenarios 10 \
  --n_routine 600 \
  --n_dynamic 150 \
  --n_ground_stations 4 \
  --curriculum_stages 300:75,600:150 \
  --vtw_time_step_s 60 \
  --vtw_workers 12 \
  --out_dir runs/scenario_cache/das_cva_stress_seed42
```

The v2 runner is kept for compatibility checks of the candidate-generation
layer. New experiments should use `das_cva_mappo.run_experiment`.

```bash
python -m cva_mappo_v2.run_experiment \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --scenario_cache_dir runs/scenario_cache/das_cva_stress_seed42 \
  --vtw_cache_dir runs/scenario_cache/das_cva_stress_seed42/vtw_cache \
  --n_satellites 12 \
  --train_iters 30 \
  --eval_episodes 20 \
  --n_routine 1200 \
  --n_dynamic 300 \
  --n_ground_stations 4 \
  --downlink_time_s 300 \
  --satellite_storage_capacity 8 \
  --enable_inter_satellite_transfer \
  --inter_satellite_transfer_time_s 300 \
  --routine_slots 64 \
  --dynamic_slots 32 \
  --flex_slots 32 \
  --slot_selection_mode typed \
  --ownership_mask_mode soft \
  --candidate_owner_bonus 0.06 \
  --dynamic_broadcast_window_s 1800 \
  --owner_switch_margin 0.08 \
  --rollout_steps 512 \
  --ppo_epochs 4 \
  --ppo_batch_size 512 \
  --train_env_workers 8 \
  --split_rollout_steps_across_workers \
  --torch_num_threads 1 \
  --eval_device cpu \
  --eval_workers 24 \
  --vtw_time_step_s 60 \
  --out_dir runs/cva_mappo_v2 \
  --run_name cva_mappo_v2_compat \
  --no_viz \
  --device cuda:0
```

```bash
python -m cva_mappo_v2.run_experiment \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --scenario_cache_dir runs/scenario_cache/das_cva_stress_seed42 \
  --vtw_cache_dir runs/scenario_cache/das_cva_stress_seed42/vtw_cache \
  --n_satellites 6 \
  --train_iters 100 \
  --eval_episodes 10 \
  --n_routine 600 \
  --n_dynamic 150 \
  --n_ground_stations 4 \
  --downlink_time_s 30 \
  --satellite_storage_capacity 30 \
  --enable_inter_satellite_transfer \
  --inter_satellite_transfer_time_s 300 \
  --routine_slots 64 \
  --dynamic_slots 32 \
  --flex_slots 32 \
  --slot_selection_mode typed \
  --ownership_mask_mode soft \
  --candidate_owner_bonus 0.06 \
  --dynamic_broadcast_window_s 1800 \
  --owner_switch_margin 0.08 \
  --rollout_steps 512 \
  --ppo_epochs 4 \
  --ppo_batch_size 512 \
  --train_env_workers 8 \
  --split_rollout_steps_across_workers \
  --torch_num_threads 1 \
  --eval_episodes 2 \
  --eval_max_steps 8000 \
  --eval_device cpu \
  --eval_workers 24 \
  --vtw_time_step_s 60 \
  --out_dir runs/cva_mappo_v2 \
  --run_name cva_mappo_v2_compat \
  --no_viz \
  --device cuda:0
```
The DAS runner defaults to `--device cuda:0`. On a CPU-only compatibility run,
explicitly pass `--device cpu`.

## DAS-CVA-MAPPO Target

The design document in `docs/DAS_CVA_MAPPO_DESIGN_SUMMARY.md` summarizes the
Word proposal and defines the next implementation target:

1. replace rule-weighted CVA scoring with a learnable satellite-task value
   scorer;
2. expose dynamic action entities instead of fixed semantic slots;
3. replace the fixed actor output head with an action-set-aware policy;
4. extend the rollout buffer to store action-set snapshots for PPO ratios.

The current `cva_mappo_v2` code should be treated as an implementation support
layer for DAS iterations, not as the main experimental method. New method
development, logs, manifests, and ablations should be centered on
`das_cva_mappo`.

Current V0.30 short validation command:

```bash
python3 scripts/run_stage_ablation_suite.py \
  --suite_name das_v030_downlink_aware_edge_value \
  --only abl_stage2_no_dynamic_downlink_priority abl_stage2_no_downlink_aware_edge_value abl_stage2_posthoc_dynamic_downlink_priority \
  --train_iters 50 \
  --val_episodes 10 \
  --eval_workers 10 \
  --eval_device cpu \
  --train_env_workers 16 \
  --device cuda:0 \
  --no_progress
```
