# DAS-CVA-MAPPO Development Branch

This branch is a cleaned working base for the DAS-CVA-MAPPO redesign.

The retained code is limited to the pieces that are still useful for the new
version:

- `das_cva_mappo/`: DAS-CVA-MAPPO implementation with an action-set-aware
  policy, PPO buffer snapshots, a learnable CVA candidate edge scorer, and
  rollout-advantage auxiliary scorer updates with hard-negative candidate
  sampling plus conflict/load target shaping. Compatibility-layer access is
  isolated behind a DAS candidate adapter, with V0.17 set-transformer action
  matching, action-type gating, idle auxiliary loss, and typed candidate
  exposure defaults.
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

## DAS V0.17

Run the current DAS action-set policy with hybrid CVA edge scoring,
set-transformer action matching, action-type gating, and idle auxiliary PPO
loss:

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
  --ppo_epochs 4 \
  --ppo_batch_size 512 \
  --eval_max_steps 8000 \
  --eval_device cpu \
  --vtw_time_step_s 60 \
  --out_dir runs/das_cva_mappo \
  --run_name das_v0_17 \
  --device cuda:0
```

The DAS runner does not currently accept `--train_env_workers` or
`--eval_workers`; those worker flags are only available in the compatibility
runner. To evaluate on GPU, change `--eval_device cpu` to
`--eval_device cuda:0`.

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
- `--routine_candidate_owners`
- `--dynamic_candidate_owners`
- `--urgent_candidate_owners`
- `--stale_candidate_owners`
- `--slot_selection_mode mixed|typed`
- `--ownership_mask_mode soft|hard`

Auxiliary scorer updates apply to `learned` and `hybrid` scorer modes. Runs
with `--candidate_scorer_mode v2_heuristic` automatically record the auxiliary
path as disabled.

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
Use `--device cuda:0` on a CUDA machine. On local Mac CPU runs, keep
`--device cpu`.

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
