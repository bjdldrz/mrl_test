# CVA-MAPPO v2

This folder contains the clean implementation of the new CVA-MAPPO design.
It is intentionally separated from the legacy `envs/multi_satellite_env.py`
implementation to keep the paper method easier to explain and ablate.

## Modules

- `config.py`: typed action slot and high-level CVA configuration.
- `scorer.py`: state-aware satellite-task pair scoring.
- `allocator.py`: task-centered capacity-aware candidate owner assignment.
- `env.py`: CVA-MAPPO v2 environment with typed local candidate slots.
- `run_experiment.py`: standalone MAPPO training/evaluation entry point.

## Main Idea

CVA-MAPPO v2 does not let the actor choose from the global task pool.
Instead:

1. score each satellite-task pair from task state, satellite state, visibility,
   quality, urgency, scarcity, owner history, and load;
2. assign each task to one or more candidate satellites under slot capacity;
3. expose fixed-size typed local action slots:
   routine slots, dynamic slots, flex slots, plus idle;
4. let low-level MAPPO choose a local slot index.

## Run

Generate scenario cache first:

```bash
python precompute_scenarios.py \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --n_satellites 12 \
  --n_train_scenarios 800 \
  --n_eval_scenarios 20 \
  --n_routine 1200 \
  --n_dynamic 300 \
  --curriculum_stages 300:75,600:150,900:225,1200:300 \
  --vtw_time_step_s 60 \
  --vtw_workers 12 \
  --out_dir runs/scenario_cache/cva_stress_sat12_r1200_d300_seed42
```

Train/evaluate v2:

```bash
python -m cva_mappo_v2.run_experiment \
  --acled_path ./DynamicMission/DynamicMission.shp \
  --scenario_cache_dir runs/scenario_cache/cva_stress_sat12_r1200_d300_seed42 \
  --vtw_cache_dir runs/scenario_cache/cva_stress_sat12_r1200_d300_seed42/vtw_cache \
  --n_satellites 12 \
  --train_iters 30 \
  --eval_episodes 20 \
  --routine_slots 64 \
  --dynamic_slots 32 \
  --flex_slots 32 \
  --ownership_mask_mode soft \
  --candidate_owner_bonus 0.06 \
  --dynamic_broadcast_window_s 1800 \
  --owner_switch_margin 0.08 \
  --assignment_switch_penalty 0.05 \
  --rollout_steps 512 \
  --ppo_epochs 4 \
  --ppo_batch_size 512 \
  --train_env_workers 8 \
  --torch_num_threads 1 \
  --eval_device cpu \
  --eval_workers 8 \
  --out_dir runs/cva_mappo_v2 \
  --run_name cva_mappo_v2_stress \
  --no_viz \
  --device cuda:0
```

For a cleaner background log, add `--no_progress`.

## CVA-Guided Mixed-TopK

The current default is `--ownership_mask_mode soft`, which turns hard owner
assignment into a soft CVA ranking signal:

- current executable tasks are kept visible, matching the strongest Mixed-TopK
  baseline behavior;
- CVA owner assignment adds a ranking bonus through `--candidate_owner_bonus`;
- future non-owner tasks are still filtered out, so slots are not filled by
  irrelevant future tasks;
- `--ownership_mask_mode hard --candidate_owner_bonus 0` restores the earlier
  hard-owner variant for ablation.

The v2 runner also exposes two knobs for the pressure-test stability issue:

- `--dynamic_broadcast_window_s`: after a dynamic task arrives, satellites that
  can execute it immediately may temporarily see it even if they are not its
  assigned owner.  This targets `avg_valid_slots` and dynamic completion.
- `--owner_switch_margin`: keeps the current primary owner unless a new owner is
  clearly better.  This targets `owner_churn_rate` and slot non-stationarity.

Recommended diagnostic comparison:

```bash
# Strong Mixed-TopK baseline inside v2: no hard owner mask, no owner bonus.
... --ownership_mask_mode soft --candidate_owner_bonus 0

# Current CVA-guided Mixed-TopK.
... --ownership_mask_mode soft --candidate_owner_bonus 0.06 \
    --dynamic_broadcast_window_s 1800 --owner_switch_margin 0.08

# Earlier hard-owner v2.
... --ownership_mask_mode hard --candidate_owner_bonus 0
```

Watch these metrics together:

- `observation_success_rate`, `dynamic_completion_rate`
- `avg_valid_slots`, `avg_valid_dynamic_slots`, `slot_valid_ratio`
- `owner_churn_rate`, `n_owner_switches`, `stale_owner_rate`
