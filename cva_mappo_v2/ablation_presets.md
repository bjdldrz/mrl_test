# CVA-MAPPO v2 Ablation Presets

All commands use the standalone v2 runner.  Keep baseline experiments in the
legacy scripts, and use this folder only for the new paper method.

Base command:

```bash
python -m cva_mappo_v2.run_experiment \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --scenario_cache_dir runs/scenario_cache/cva_stress_sat12_r1200_d300_gs4_seed42 \
  --vtw_cache_dir runs/scenario_cache/cva_stress_sat12_r1200_d300_gs4_seed42/vtw_cache \
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
  --rollout_steps 512 \
  --ppo_epochs 4 \
  --ppo_batch_size 512 \
  --train_env_workers 8 \
  --torch_num_threads 1 \
  --slot_selection_mode mixed \
  --ownership_mask_mode soft \
  --candidate_owner_bonus 0.06 \
  --dynamic_broadcast_window_s 1800 \
  --owner_switch_margin 0.08 \
  --eval_device cpu \
  --eval_workers 8 \
  --vtw_time_step_s 60 \
  --no_viz \
  --device cuda:0
```

With `--enable_inter_satellite_transfer`, v2 exposes explicit relay actions:
`K` task slots + `n_satellites - 1` target-satellite relay slots + idle.  A
relay action sends all currently stored, undelivered images from the source
satellite to the selected target satellite.

## A1 Slot Scale

Purpose: verify that a fixed typed action space can replace the global task
pool without losing too much completion rate.

```bash
# K=64
... --routine_slots 32 --dynamic_slots 16 --flex_slots 16 \
    --run_name cva_v2_slots64 --out_dir runs/cva_mappo_v2_slots

# K=128
... --routine_slots 64 --dynamic_slots 32 --flex_slots 32 \
    --run_name cva_v2_slots128 --out_dir runs/cva_mappo_v2_slots

# K=256
... --routine_slots 128 --dynamic_slots 64 --flex_slots 64 \
    --run_name cva_v2_slots256 --out_dir runs/cva_mappo_v2_slots
```

## A2 Candidate Owner Count

Purpose: routine tasks should usually have one owner, while dynamic/urgent tasks
benefit from multi-candidate rescue.

```bash
# strict ownership
... --routine_candidate_owners 1 --dynamic_candidate_owners 1 \
    --urgent_candidate_owners 1 --stale_candidate_owners 1 \
    --run_name cva_v2_owner_strict --out_dir runs/cva_mappo_v2_owner

# default multi-candidate dynamic rescue
... --routine_candidate_owners 1 --dynamic_candidate_owners 2 \
    --urgent_candidate_owners 3 --stale_candidate_owners 3 \
    --run_name cva_v2_owner_default --out_dir runs/cva_mappo_v2_owner
```

## A3 Typed Slots

Purpose: test whether reserving dynamic slots improves dynamic response.

```bash
# no explicit dynamic reservation
... --slot_selection_mode typed --routine_slots 96 --dynamic_slots 0 --flex_slots 32 \
    --run_name cva_v2_no_dynamic_slots --out_dir runs/cva_mappo_v2_typed_slots

# balanced typed slots
... --slot_selection_mode typed --routine_slots 64 --dynamic_slots 32 --flex_slots 32 \
    --run_name cva_v2_typed_default --out_dir runs/cva_mappo_v2_typed_slots
```

## A4 Reassignment Triggers

Purpose: separate static candidate ownership from event-triggered repair.

```bash
# static only
... --assignment_replan_trigger "" --assignment_replan_interval_s 0 \
    --run_name cva_v2_static --out_dir runs/cva_mappo_v2_replan

# event-triggered repair
... --assignment_replan_trigger periodic,dynamic,stale_owner,deadline \
    --assignment_replan_interval_s 3600 --assignment_replan_horizon_s 7200 \
    --run_name cva_v2_event_repair --out_dir runs/cva_mappo_v2_replan
```

## A5 CVA-Guided Mixed-TopK

Purpose: verify whether CVA should be a hard owner constraint or a soft ranking
signal on top of the strong Mixed-TopK baseline.

```bash
# Mixed-TopK-like: shared Top-K, no CVA owner ranking bonus
... --slot_selection_mode mixed --ownership_mask_mode soft --candidate_owner_bonus 0 \
    --run_name cva_v2_soft_no_owner_bonus --out_dir runs/cva_mappo_v2_soft_owner

# CVA-guided Mixed-TopK: current recommended variant
... --slot_selection_mode mixed --ownership_mask_mode soft --candidate_owner_bonus 0.06 \
    --dynamic_broadcast_window_s 1800 --owner_switch_margin 0.08 \
    --run_name cva_v2_soft_owner_bonus --out_dir runs/cva_mappo_v2_soft_owner

# Hard-owner v2: earlier assignment-mask variant
... --slot_selection_mode typed --ownership_mask_mode hard --candidate_owner_bonus 0 \
    --dynamic_broadcast_window_s 1800 --owner_switch_margin 0.08 \
    --run_name cva_v2_hard_owner --out_dir runs/cva_mappo_v2_soft_owner

# Typed soft-owner ablation: tests whether fixed routine/dynamic/flex quotas hurt
... --slot_selection_mode typed --ownership_mask_mode soft --candidate_owner_bonus 0.06 \
    --dynamic_broadcast_window_s 1800 --owner_switch_margin 0.08 \
    --run_name cva_v2_typed_soft_owner --out_dir runs/cva_mappo_v2_soft_owner
```

Compare:

- completion rate and dynamic response delay;
- `avg_valid_slots` / `avg_valid_dynamic_slots`;
- duplicate rate, load balance, and owner churn.
- `eval_steps`, `eval_end_time_s`, `eval_idle_action_rate`,
  `eval_avg_valid_action_count`, and `eval_avg_raw_valid_action_count` to rule
  out early termination, idle-heavy evaluation, or candidate-slot action loss.
