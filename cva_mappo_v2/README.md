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

When `--enable_inter_satellite_transfer` is enabled, each satellite also gets
`n_satellites - 1` explicit relay actions between the task slots and idle.  A
relay action selects the target satellite and sends all currently stored,
undelivered images from the source satellite.

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
  --n_ground_stations 4 \
  --curriculum_stages 300:75,600:150,900:225,1200:300 \
  --vtw_time_step_s 60 \
  --vtw_workers 12 \
  --out_dir runs/scenario_cache/cva_stress_sat12_r1200_d300_gs4_seed42
```

Train/evaluate v2:

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
  --routine_slots 64 \
  --dynamic_slots 32 \
  --flex_slots 32 \
  --slot_selection_mode typed \
  --ownership_mask_mode soft \
  --candidate_owner_bonus 0.06 \
  --dynamic_broadcast_window_s 1800 \
  --owner_switch_margin 0.08 \
  --assignment_switch_penalty 0.05 \
  --rollout_steps 512 \
  --ppo_epochs 4 \
  --ppo_batch_size 512 \
  --train_env_workers 8 \
  --split_rollout_steps_across_workers \
  --torch_num_threads 1 \
  --eval_device cpu \
  --eval_workers 8 \
  --vtw_time_step_s 60 \
  --out_dir runs/cva_mappo_v2 \
  --run_name cva_mappo_v2_stress \
  --no_viz \
  --device cuda:0
```

Evaluation is stochastic by default. Add
`--eval_deterministic` only when you explicitly want actor argmax evaluation.
For a cleaner background log, add `--no_progress`.

Without `--scenario_cache_dir`, training now uses `--n_routine/--n_dynamic` by
default so quick tests run at the scale requested on the command line. Add
`--curriculum_train_scale` to restore the old generated curriculum pools. With
multiple train workers, `--rollout_steps` is treated as total samples per
iteration and split across workers; add `--rollout_steps_per_worker` to restore
the old per-worker sampling budget.

Evaluation uses a defensive per-episode step cap of `horizon/10 + 100` when
`--eval_max_steps 0` (the default). For slow pressure-test debugging, set a
smaller explicit cap such as `--eval_max_steps 2000`; this does not change the
policy, only truncates very long evaluation rollouts.

## CVA-Guided Mixed-TopK

The current default is `--slot_selection_mode typed --ownership_mask_mode soft`.
The typed path keeps routine/dynamic/flex quotas active for DAS-centered runs.
`--slot_selection_mode mixed` remains available as the shared Top-K ablation:

- current executable tasks are kept visible, matching the strongest Mixed-TopK
  reference behavior;
- in mixed mode, all candidate tasks are ranked in one shared Top-K list
  instead of being truncated by fixed routine/dynamic/flex quotas;
- CVA owner assignment adds a ranking bonus through `--candidate_owner_bonus`;
- future non-owner tasks are still filtered out, so slots are not filled by
  irrelevant future tasks;
- `--slot_selection_mode typed` uses the fixed routine/dynamic/flex slot
  layout and is now the default;
- `--ownership_mask_mode hard --candidate_owner_bonus 0` restores the earlier
  hard-owner variant for ablation.

The v2 runner also exposes two knobs for the pressure-test stability issue:

- `--dynamic_broadcast_window_s`: after a dynamic task arrives, satellites that
  can execute it immediately may temporarily see it even if they are not its
  assigned owner.  This targets `avg_valid_slots` and dynamic completion.
- `--owner_switch_margin`: keeps the current primary owner unless a new owner is
  clearly better.  This targets `owner_churn_rate` and slot non-stationarity.
- `--enable_inter_satellite_transfer`: exposes explicit relay actions.  The
  actor output dimension becomes `task_slots + n_satellites - 1 + idle`.

Recommended diagnostic comparison:

```bash
# Strong Mixed-TopK reference inside v2: no hard owner mask, no owner bonus.
... --slot_selection_mode mixed --ownership_mask_mode soft --candidate_owner_bonus 0

# Current CVA-guided Mixed-TopK.
... --slot_selection_mode mixed --ownership_mask_mode soft --candidate_owner_bonus 0.06 \
    --dynamic_broadcast_window_s 1800 --owner_switch_margin 0.08

# Earlier typed hard-owner v2.
... --slot_selection_mode typed --ownership_mask_mode hard --candidate_owner_bonus 0
```

Watch these metrics together:

- `observation_success_rate`, `dynamic_completion_rate`
- `avg_valid_slots`, `avg_valid_dynamic_slots`, `slot_valid_ratio`
- `owner_churn_rate`, `n_owner_switches`, `stale_owner_rate`
- `eval_steps`, `eval_end_time_s`, `eval_idle_action_rate`,
  `eval_avg_valid_action_count`, `eval_avg_raw_valid_action_count` to check
  whether evaluation is ending early, mostly idling, or losing valid actions
  during candidate-slot mapping.
