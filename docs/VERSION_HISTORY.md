# Version History

## DAS-CVA-MAPPO V0.30.0

Status: implemented.

Scope:

- Keeps `abl_stage2_no_dynamic_downlink_priority` as the current stronger
  Stage-2 baseline after V0.29 results showed post-hoc dynamic downlink
  replanning did not improve dynamic response or downlink queue time.
- Moves downlink pressure into candidate edge value before observation
  selection. The CVA scorer now estimates downlink queue delay, delivery delay,
  and downlink feasibility for each satellite-task candidate window.
- Extends the trainable DAS candidate scorer edge feature vector with downlink
  queue pressure, delivery delay pressure, downlink feasibility, and normalized
  queue delay.
- Adds configurable knobs:
  `--no_downlink_aware_candidate_score`, `--downlink_queue_target_s`,
  `--candidate_downlink_queue_penalty`,
  `--candidate_downlink_miss_penalty`,
  `--candidate_dynamic_delivery_bonus`,
  `--candidate_dynamic_delivery_delay_penalty`, and
  `--dynamic_downlink_priority`.
- Adds per-dynamic-task diagnostics. Evaluation now writes
  `eval_dynamic_task_diagnostics.json`, and summary metrics report whether
  arrived dynamic tasks were seen in candidates, currently executable when
  seen, selected by the policy, observed, downlinked, or blocked by the
  downlink queue.

Expected effect:

- Make dynamic response optimization part of action selection instead of
  relying on post-observation downlink reordering.
- Reduce `avg_downlink_queue_s`, `avg_dynamic_response_s`, and
  `dynamic_task_downlink_queue_block_rate` while preserving the stronger
  no-posthoc-priority Stage-2 baseline.
- Use `abl_stage2_no_downlink_aware_edge_value` to isolate the new edge value
  contribution and `abl_stage2_posthoc_dynamic_downlink_priority` to compare
  against the rejected V0.29 post-hoc downlink priority path.

## DAS-CVA-MAPPO V0.29.0

Status: implemented.

Scope:

- Adds visible-candidate idle advancement: in Top-K mode, idle mission-window
  jumps are limited to task slots the policy could actually see, while dynamic
  arrivals and storage releases remain global events.
- Makes dynamic candidate ranking response-aware by penalizing late dynamic
  windows during score-pair window selection.
- Adds a configurable current-dynamic slot boost and reports current/future
  dynamic slot exposure diagnostics.
- Adds dynamic-priority downlink replanning for not-yet-started downlinks so
  dynamic images can move ahead of routine images in the ground segment.
- Exposes `--dynamic_current_slot_bonus`, `--dynamic_window_wait_weight`, and
  `--no_dynamic_downlink_priority`.

Expected effect:

- Reduce `avg_dynamic_response_s` and improve `dynamic_completion_rate_raw`
  when dynamic observations were waiting behind routine future windows or
  routine downlink reservations.
- Use `abl_stage2_no_dynamic_downlink_priority` to isolate the ground-segment
  contribution.

## DAS-CVA-MAPPO V0.28.1

Status: implemented.

Scope:

- Restores CPU evaluation as the default for both the DAS runner and staged
  ablation suite, while keeping CUDA as the default training device.
- Keeps `--eval_device cuda:0` available for explicit diagnostic runs, but
  future model-iteration commands should use CPU evaluation for consistency.

## DAS-CVA-MAPPO V0.28.0

Status: implemented.

Scope:

- Optimizes multi-agent environment stepping after profiling showed
  `env.step()` dominated evaluation wall time.
- Adds an all-idle fast path in conflict resolution for the default
  train/eval-consistent mode, avoiding unnecessary full action-mask and auction
  work when every policy action is idle.
- Lets `MultiSatelliteEnv` call low-level `SatelliteSchedulingEnv.step()` with
  `build_observation=False` and `check_done=False`, then rebuilds final
  multi-agent observations and done flags once at the end of the step.
- Caches the latest multi-agent done result so the caller's immediate
  `is_done()` check does not repeat a full mission-feasibility scan.
- Reports `n_fast_idle_resolve_steps`, `fast_idle_resolve_rate`, and
  `n_low_level_fast_steps` in metrics and suite summaries.

Expected effect:

- Reduces repeated Python work inside evaluation-heavy runs without changing
  the final observations, action masks, rewards, or default train/eval
  environment semantics.
- Should help most when `eval_idle_action_rate` is high, which is the current
  profiled regime.

## DAS-CVA-MAPPO V0.27.0

Status: implemented.

Scope:

- Adds optional `--eval_profile` instrumentation for DAS evaluation.
- Records evaluation wall time, total evaluated steps, steps per wall second,
  and timed sections for setup, reset, valid-mask checks, feature building,
  actor forward, counter updates, environment stepping, and finalization.
- Exposes the profiling flag and the key timing columns in
  `scripts/run_stage_ablation_suite.py` summaries.

Expected effect:

- Makes it possible to verify whether slow evaluation is dominated by
  candidate/feature construction, actor forward, or Python environment stepping
  from one `summary.csv` table.
- Leaves default metrics and train/eval environment behavior unchanged unless
  `--eval_profile` is explicitly enabled.

## DAS-CVA-MAPPO V0.26.0

Status: implemented.

Scope:

- Makes DAS evaluation use the same environment action-resolution path as
  training by default. `env.eval_mode` is no longer enabled unconditionally
  during evaluation.
- Keeps the old eval-only conflict repair/rescue behavior behind
  `--eval_use_repair` for diagnostic runs.
- Records `eval_use_repair` in the runtime plan and exposes it through the
  staged experiment runner.
- Adds a lightweight static regression check to keep evaluation from
  accidentally re-enabling eval-only repair as the default.

Expected effect:

- Evaluation metrics now reflect the policy under the same conflict-resolution,
  loser handling, dynamic preemption, and idle-rescue behavior used during
  training.
- Paper-facing results avoid the previous mismatch where eval received
  rule-based post-processing that train did not use.

## DAS-CVA-MAPPO V0.25.0

Status: implemented.

Scope:

- Adds single-process batched CUDA/MPS evaluation. When `--eval_device` is not
  CPU, `--eval_workers` now controls how many evaluation environments are kept
  active in the same process and batched into one policy forward pass.
- Keeps CPU evaluation on the existing multiprocessing path.
- Updates the runtime plan to report `mode=batched_single_process` for CUDA/MPS
  evaluation instead of reducing `effective_workers` to 1.

Expected effect:

- Enable commands such as `--eval_device cuda:0 --eval_workers 8` without
  spawning 8 GPU-contending processes.
- Reduce evaluation wall time on GPU while preserving the same eval metrics and
  per-episode aggregation semantics.

## DAS-CVA-MAPPO V0.24.0

Status: implemented.

Scope:

- Adds dynamic response pressure to candidate scoring. Arrived dynamic tasks
  gain priority as their age approaches `--dynamic_response_target_s`.
- Adds dynamic wait pressure so future dynamic candidates are penalized by wait
  time relative to the response target instead of the full 24-hour horizon.
- Adds response-aware allocator repair weights and eval-time dynamic rescue
  scoring through `--allocator_dynamic_response_bonus`,
  `--allocator_dynamic_wait_penalty`, and
  `--dynamic_rescue_response_bonus`.
- Extends DAS trainable candidate-scorer edge features with dynamic response and
  dynamic wait pressure.
- Changes the staged experiment runner from hard-coded CPU evaluation to
  `--eval_device same` by default, so CUDA training runs also evaluate on CUDA
  unless explicitly overridden.
- Adds `abl_stage2_no_dynamic_response_pressure` to isolate the response-time
  optimization.

Expected effect:

- Reduce `avg_dynamic_response_s` while preserving the V0.23 dynamic completion
  recovery.
- Make GPU evaluation the default path for future sweeps. The runner still
  reports effective evaluation workers, and single-GPU CUDA evaluation is
  serialized by `run_experiment.py` to avoid multiple processes contending for
  the same GPU.

## DAS-CVA-MAPPO V0.23.0

Status: implemented.

Scope:

- Restores bounded future-task macro execution as the default behavior instead
  of V0.22's only-idle restriction. The V0.22 behavior is retained behind
  `--future_task_requires_no_current_valid`.
- Adds a separate `--future_routine_max_wait_s` cap so routine future macro
  actions cannot use the full dynamic future window by default.
- Adds `--routine_future_dynamic_guard_s` and
  `--routine_future_dynamic_penalty` to block or down-rank routine future
  actions when arrived dynamic tasks are executable or near-term feasible for
  the same satellite.
- Adds `--dynamic_future_bonus` and drops ineligible future-only candidates from
  action slots by default to reduce invalid routine context flooding.
- Records actual future macro usage with `n_future_task_executions`,
  `n_future_dynamic_task_executions`, `n_future_routine_task_executions`, and
  `avg_future_task_wait_s`.
- Adds `stage2_dynamic_priority_recovery` and matching Stage-2 dynamic-priority
  ablations to the staged experiment runner.

Expected effect:

- Improve dynamic completion relative to the open V0.22 ablation while keeping
  short-wait future macro benefits that improved total reward and routine
  throughput.
- Make dynamic-task results more suitable for a paper setting by reporting both
  raw completion and feasible-normalized completion under the fixed scenario
  feasibility ceiling.

## DAS-CVA-MAPPO V0.22.0

Status: implemented.

Scope:

- Restricts future-task macro execution to only-idle states by default. Future
  task slots are considered executable only when the satellite has no current
  non-idle executable task or transfer action.
- Lowers the default `--future_task_max_wait_s` from 7200 to 600 seconds to
  prevent long jumps over intermediate VTWs, dynamic arrivals, and resource
  events.
- Adds `--future_task_allow_with_current_valid` for ablation of the V0.21
  fully-open future macro behavior.
- Adds `abl_stage2_no_future_task_execution`,
  `abl_stage2_future_macro_with_current_valid`, and
  `abl_future_macro_with_current_valid` to the staged ablation runner.

Expected effect:

- Preserve V0.21's ability to make future-window tasks selectable in only-idle
  states while avoiding premature commitment when an immediate task or transfer
  can be executed.
- Reduce the severe throughput collapse observed when future macro actions were
  available alongside current executable actions.

## DAS-CVA-MAPPO V0.21.0

Status: implemented.

Scope:

- Adds bounded future-task macro execution. A candidate task slot can be valid
  when it is not executable at the current instant but has an earliest feasible
  observation start within `--future_task_max_wait_s`.
- Before executing such an action, the selected satellite advances to the
  earliest feasible start and then uses the original observation executor, so
  VTW, maneuver, storage, downlink, deadline, and conflict checks still apply.
- Adds `--no_future_task_execution` for ablation and keeps
  `--executable_slot_reserve_ratio` for current-action slot reservation.
- Candidate diagnostics now split valid slots into current and future macro
  validity with `avg_current_valid_slots` and `avg_future_valid_slots`.
- Adds `abl_no_future_task_execution` to the staged ablation runner.

Expected effect:

- Raise `eval_valid_decision_rate` and `avg_valid_slots` when most candidate
  slots are future-window tasks.
- Preserve the value of future context while allowing the policy to commit to a
  specific near-future executable task instead of being forced to idle.
- Make it possible to compare pure candidate reordering against true future
  execution semantics.

## DAS-CVA-MAPPO V0.20.0

Status: implemented.

Scope:

- Adds executable-aware typed candidate exposure via
  `--executable_slot_reserve_ratio`. A configurable fraction of task slots is
  filled from currently executable actions before future-only context tasks can
  occupy the remaining action set.
- Adds per-slot timing metadata to the v2 candidate interface:
  `currently_executable`, `future_executable`, `wait_norm`,
  `next_start_norm`, and `time_to_deadline_norm`.
- Feeds timing metadata into DAS action features so the actor can distinguish
  actions that are executable now from tasks that are only future context.
- Adds `abl_no_executable_slot_reserve` to the staged ablation runner.

Expected effect:

- Increase `avg_valid_slots` and `eval_valid_decision_rate` without removing
  future-window context from the action set.
- Reduce the number of filled-but-invalid slots caused by future-only tasks
  occupying typed candidate capacity.
- Give the policy an explicit temporal representation for waiting decisions
  instead of forcing it to infer future executability from VTW fields alone.

## DAS-CVA-MAPPO V0.19.0

Status: implemented.

Scope:

- Adds slot-invalid diagnostics for typed candidate exposure. Evaluation
  metrics now include `slot_invalid_*`, `slot_filled_invalid_ratio`, and
  `avg_filled_invalid_slots` so filled-but-not-executable candidate slots can
  be separated from empty slots.
- Extends the v2 candidate scorer with configurable wait penalty, storage
  pressure penalty, and dynamic urgency bonus.
- Extends the capacity-aware allocator with earlier-window ranking,
  stale-owner rescue bonus, and dynamic urgency bonus.
- Adds `--dynamic_takeover_margin_s` to tune how aggressively arrived dynamic
  tasks can be released to a non-owner with an earlier feasible window.
- Records the new candidate/allocator weights in `manifest.json`.
- Adds staged optimization commands to the README for diagnosis,
  candidate-owner repair, dynamic-task optimization, and storage/downlink
  pressure experiments.
- Sets the DAS runner defaults to parallel rollout/evaluation
  (`--train_env_workers 16`, `--eval_workers 24`) and CUDA training
  (`--device cuda:0`).

Expected effect:

- Explain why `avg_filled_slots` can be high while `avg_valid_slots` is near
  zero.
- Reduce stale-owner lock-in by favoring satellites with nearer feasible
  windows during candidate ownership repair.
- Improve dynamic-task raw completion by increasing candidate visibility and
  urgency-aware rescue opportunities before tuning PPO further.

## DAS-CVA-MAPPO V0.18.0

Status: implemented.

Scope:

- Adds native `das_cva_mappo.run_experiment` support for
  `--train_env_workers`, `--split_rollout_steps_across_workers`,
  `--rollout_steps_per_worker`, and `--eval_workers`.
- Runs rollout workers on CPU with model and candidate-scorer snapshots, then
  aggregates `ActionSetRolloutBuffer` objects in the main process for one PPO
  update.
- Merges candidate auxiliary samples across worker rollouts before updating the
  DAS candidate scorer.
- Adds parallel CPU evaluation; CUDA evaluation automatically falls back to one
  worker to avoid multi-process single-GPU contention.
- Records the effective train/eval worker plan in `manifest.json`.
- Updates the README command to the V0.18 worker-enabled DAS command.

Expected effect:

- Recover the worker-based runtime controls used in earlier experiments while
  keeping DAS as the main experimental method.
- Preserve centralized PPO and candidate-scorer updates in the main process.
- Make rollout-step semantics explicit for ablations and speed comparisons.

## DAS-CVA-MAPPO V0.17.0

Status: implemented.

Scope:

- Disables the hard `idle_valid_penalty` by default after V0.16 showed no
  improvement.
- Adds a PPO auxiliary loss controlled by `--idle_aux_coeff`: when a sampled
  action set has at least one valid non-idle action, the update penalizes the
  policy's idle probability.
- Logs `idle_aux_loss` in the training metrics.
- Keeps V0.16 eval diagnostics for `eval_idle_when_valid_rate` and related
  rates.
- Synchronizes the README run command with the current V0.17 DAS experiment
  settings and explicit idle-ablation parameters.
- Clarifies that `das_cva_mappo.run_experiment` does not accept the
  compatibility-runner worker flags `--train_env_workers` and `--eval_workers`.

Expected effect:

- Teach lower idle probability at valid decision points without forcing a fixed
  inference-time logit penalty.
- Preserve necessary waiting in only-idle states.
- Make `--idle_aux_coeff 0` and `--idle_valid_penalty 0` a clean ablation of
  idle-specific learning pressure.

## DAS-CVA-MAPPO V0.16.0

Status: implemented.

Scope:

- Adds `idle_valid_penalty`, which subtracts from the idle logit only when the
  current action mask contains at least one valid non-idle action.
- Keeps idle available when no executable action exists, so the policy is not
  punished for unavoidable waiting.
- Adds eval diagnostics: `eval_valid_decision_rate`,
  `eval_idle_when_valid_rate`, and `eval_idle_without_valid_rate`.

Expected effect:

- Reduce raw policy idle selections at decision points where a task or transfer
  action is executable.
- Preserve waiting behavior in only-idle states.
- Make future idle analysis separate action scarcity from policy over-waiting.

## DAS-CVA-MAPPO V0.15.0

Status: implemented.

Scope:

- Adds a learnable action-type gate on top of the action matcher.
- The gate predicts routine/dynamic/flex/transfer/idle mode logits from the
  current state or set context and adds the matching mode logit to each action.
- Initializes the gate to zero, so the initial policy matches V0.14 before
  PPO learns type preferences.
- Adds `--no_action_type_gate` for ablation.

Expected effect:

- Help the policy distinguish "which mode should act now" from "which entity
  within that mode is best".
- Improve dynamic/routine/idle tradeoffs without hard-coding another rule.
- Preserve V0.14's set-transformer contextual action ranking.

## DAS-CVA-MAPPO V0.14.0

Status: implemented.

Scope:

- Adds `set_transformer` as an action matcher and makes it the DAS default.
- Encodes the state as a global token and all exposed action entities as set
  tokens, then scores each action from contextualized token embeddings.
- Preserves future-task tokens as attention context even when they are not
  currently executable; the final action mask still prevents sampling invalid
  actions.
- Keeps `additive` and `dot` matchers for ablation via `--matcher`.

Expected effect:

- Improve action ranking when routine, dynamic, flex, transfer, and idle choices
  compete in the same exposed action set.
- Let the policy reason over future-window context rather than only per-action
  MLP features plus mean pooling.
- Provide a cleaner model-structure ablation target for DAS-CVA-MAPPO.

## DAS-CVA-MAPPO V0.13.0

Status: implemented.

Scope:

- Adds near-window dynamic takeover release: if an arrived dynamic task is owned
  by another satellite, but a non-owner has an earlier near-term feasible
  window and the owner does not, that non-owner may see and take over the task.
- Adds a candidate-score bonus for dynamic tasks released through this takeover
  path.
- Raises the default dynamic candidate owner count from 4 to 6 while leaving
  routine ownership narrow.
- Adds `n_dynamic_takeover_release_events`.
- Resets all rescue/takeover counters during environment reset to keep
  multi-episode metrics clean.

Expected effect:

- Improve dynamic completion and response time when V0.12 reports zero dynamic
  idle/preemption opportunities.
- Preserve the V0.11 routine rescue path instead of globally increasing dynamic
  pressure.
- Diagnose whether dynamic misses come from no near-window takeover candidates
  or from downstream action selection.

## DAS-CVA-MAPPO V0.12.0

Status: implemented.

Scope:

- Keeps the V0.11 executable idle rescue path, but tries executable dynamic
  rescues before routine rescues.
- Lowers eval-time routine-to-dynamic preemption margin to neutral so a current
  dynamic task can replace routine when its rescue value is at least comparable.
- Adds a current-executable dynamic score bonus in typed candidate exposure.
- Adds `n_dynamic_idle_rescue_opportunities` and
  `n_dynamic_preemption_opportunities` metrics.

Expected effect:

- Recover part of the V0.10/V0.9 dynamic completion rate while preserving most
  of the V0.11 routine throughput.
- Reduce `avg_dynamic_response_s` when executable dynamic opportunities exist.
- Distinguish "no dynamic opportunity existed" from "dynamic opportunity was
  ignored by the rescue layer."

## DAS-CVA-MAPPO V0.11.0

Status: implemented.

Scope:

- Keeps dynamic slots first, then lets currently executable flex tasks enter
  early before routine future-only context fills the typed action set.
- Keeps routine quota visible, and only backfills remaining flex slots after
  routine quota is considered.
- Extends eval-time rescue from dynamic-only idle rescue to executable-task
  idle rescue, while keeping dynamic and routine rescue counts separate.
- Adds `n_routine_idle_rescues` and `n_idle_executable_rescues` metrics.

Expected effect:

- Recover routine and total throughput after V0.10 without dropping dynamic
  completion sharply.
- Reduce wasted valid windows when the policy selects idle despite current
  executable tasks.
- Make it clear whether gains come from dynamic rescue or routine rescue.

## DAS-CVA-MAPPO V0.10.0

Status: implemented.

Scope:

- Keeps dynamic slots before routine slots, but lets routine slots claim their
  quota before flex fallback. This prevents flex from consuming most routine
  candidates after V0.9.
- Adds eval-time dynamic preemption: a routine action can be replaced by a
  currently executable dynamic task when the dynamic rescue value clears a
  margin.
- Adds `n_dynamic_idle_rescues` and `n_dynamic_preemptions` metrics so dynamic
  rescue behavior is auditable in experiment output.
- Adds dynamic age pressure to rescue value to reduce response delay for
  already-arrived dynamic tasks.

Expected effect:

- Preserve V0.9 dynamic completion gains while recovering routine throughput.
- Reduce `avg_dynamic_response_s`.
- Make dynamic rescue side effects easier to diagnose.

## DAS-CVA-MAPPO V0.9.0

Status: implemented.

Scope:

- Changes typed candidate exposure order to place dynamic slots before flex
  and routine slots. This reduces fixed-slot actor bias toward early routine
  indices.
- Adds eval-time dynamic rescue in the multi-agent resolver: if a satellite
  selected idle but currently has an executable dynamic task, the resolver can
  assign that dynamic task while preserving normal conflict checks.
- Keeps this rescue path evaluation-only to avoid corrupting training credit
  assignment for sampled idle actions.

Expected effect:

- Prevent currently valid dynamic windows from being skipped by idle-heavy
  policies.
- Improve `n_feasible_dynamic_done`, `dynamic_completion_rate`, and
  `dynamic_completion_rate_raw` without treating the compatibility layer as the
  main method.

## DAS-CVA-MAPPO V0.8.0

Status: implemented.

Scope:

- Fills typed `flex` slots with urgent/stale/dynamic candidates first, then
  uses the best remaining dynamic/routine candidates instead of leaving flex
  quota empty.
- Adds a typed-mode dynamic rescue pool so arrived dynamic tasks can be exposed
  to non-owner agents during broadcast, stale-owner, deadline, or currently
  executable windows.
- Raises default dynamic, urgent, and stale candidate owner counts to improve
  rescue coverage.
- Expands the default dynamic broadcast and deadline release windows to 3600s.
- Expands the default assignment replan horizon to 21600s to reduce underfilled
  candidate pools in dense stress scenarios.
- Increases the default dynamic task score weight in the compatibility scorer.
- Narrows event-triggered replans to the tasks matching the trigger reason,
  while periodic and imbalance replans still consider the full eligible set.
- Allows stale/deadline tasks to bypass the normal owner-switch cap so they can
  still be rescued when the current owner no longer has a feasible window.

Expected effect:

- Improve `avg_filled_dynamic_slots`, `avg_filled_flex_slots`, and dynamic task
  exposure.
- Reduce owner churn from event-triggered replans that previously touched the
  full task set.
- Improve feasible dynamic completion without treating `cva_mappo_v2` as the
  main experimental method.

Run:

```bash
python -m das_cva_mappo.run_experiment
```

## DAS-CVA-MAPPO V0.7.0

Status: implemented.

Scope:

- Changes idle advancement in `SatelliteEnv` to jump to the next truly
  actionable observation time instead of every raw VTW boundary.
- Uses observation duration, task arrival/deadline, attitude transition,
  storage capacity, and existing schedule conflicts when selecting the next
  idle event.
- Updates episode completion checks to ignore tasks that have no future
  feasible observation start.
- Shortens event-triggered assignment repair cooldown by using the assignment
  lock window for dynamic/stale/deadline events while keeping periodic replan
  cadence separate.
- Switches the default compatibility-layer candidate exposure from `mixed` to
  `typed`, so routine/dynamic/flex slot quotas are active unless an ablation
  explicitly selects `--slot_selection_mode mixed`.
- Records stale-owner tasks as released in CVA-MAPPO v2 diagnostics when they
  are exposed for rescue.

Expected effect:

- Reduce only-idle evaluation states.
- Improve `eval_avg_valid_action_count`, `avg_valid_slots`, and dynamic/flex
  candidate visibility.
- Make stale-owner rescue diagnostics reflect the actual soft-release behavior.

Run:

```bash
python -m das_cva_mappo.run_experiment
```

New default:

- `--slot_selection_mode typed`

## DAS-CVA-MAPPO V0.6.0

Status: implemented in `das_cva_mappo/`.

Scope:

- Adds `das_cva_mappo/env_adapter.py` as the DAS boundary around the current
  candidate-generation compatibility layer.
- Routes action-set feature construction and scorer warm-start edge collection
  through `V2CandidateAdapter`.
- Records `candidate_adapter_mode` in `manifest.json`.
- Keeps behavior unchanged while narrowing the future replacement point for a
  DAS-native allocator/adaptor.

Run:

```bash
python -m das_cva_mappo.run_experiment
```

New or expanded interface:

- `--candidate_adapter_mode v2_compat`

Known limitations:

- Only the current compatibility adapter is implemented.  A DAS-native
  candidate allocator/adaptor is the next structural step.
- Candidate allocation still reuses the existing capacity-aware allocator.
- Evaluation is serial in the DAS runner.

Next planned version:

- Add a DAS-native candidate adapter/allocator implementation behind the new
  adapter boundary.
- Add experiment orchestration presets for DAS-centered ablations.

## DAS-CVA-MAPPO V0.5.0

Status: implemented in `das_cva_mappo/`.

Scope:

- Adds coordination-aware target shaping for candidate scorer auxiliary updates.
- Penalizes selected candidate edges when multiple agents select the same task
  in the same rollout step.
- Penalizes high-load candidate edges through the scorer edge feature
  `load_pressure`.
- Records conflict/load shaping statistics in `train_log.csv`.

Run:

```bash
python -m das_cva_mappo.run_experiment
```

New or expanded ablation interfaces:

- `--candidate_aux_conflict_penalty`
- `--candidate_aux_load_penalty`

Known limitations:

- Conflict shaping relies on task ids in the exposed action-set snapshot; a
  future DAS-specific allocator/adaptor should expose richer graph-level
  conflict context.
- Candidate allocation still reuses the existing capacity-aware allocator.
- Evaluation is serial in the DAS runner.

## DAS-CVA-MAPPO V0.4.0

Status: implemented in `das_cva_mappo/`.

Scope:

- Adds hard-negative sampling from unselected candidate edges in each sampled
  action set.
- Anchors each hard negative to the selected task edge from the same decision
  point and trains it below the selected edge by a configurable margin.
- Records positive and negative auxiliary edge counts in `train_log.csv` and
  scorer checkpoint metadata.

Run:

```bash
python -m das_cva_mappo.run_experiment
```

New or expanded ablation interfaces:

- `--candidate_hard_negative_samples`
- `--candidate_hard_negative_include_invalid`
- `--candidate_hard_negative_margin`
- `--candidate_hard_negative_value_weight`

Known limitations:

- Hard negatives are sampled from edge snapshots exposed to the actor; a future
  DAS-specific allocator/adaptor should expose richer candidate graph context.
- Candidate allocation still reuses the existing capacity-aware allocator.
- Evaluation is serial in the DAS runner.

## DAS-CVA-MAPPO V0.3.0

Status: implemented in `das_cva_mappo/`.

Scope:

- Adds rollout-advantage auxiliary updates for the DAS candidate edge scorer.
- Stores candidate scorer edge-feature snapshots in the action-set rollout
  buffer.
- Fits the selected candidate edge value to normalized GAE advantages and adds
  a pairwise ranking loss between selected candidate edges.
- Saves final scorer weights after training, including optimizer state and
  auxiliary update statistics.

Run:

```bash
python -m das_cva_mappo.run_experiment
```

New or expanded ablation interfaces:

- `--no_candidate_aux_update`
- `--candidate_aux_epochs`
- `--candidate_aux_batch_size`
- `--candidate_aux_rank_weight`
- `--candidate_aux_target_clip`
- `--candidate_aux_min_edges`

The auxiliary update is active only for `learned` and `hybrid` candidate
scorers.  `v2_heuristic` runs automatically record it as disabled.

Known limitations:

- Auxiliary updates currently use only selected task edges from rollout
  snapshots; hard negative unselected edges are not yet sampled explicitly.
- Candidate allocation still reuses the existing capacity-aware allocator.
- Evaluation is serial in the DAS runner.

## DAS-CVA-MAPPO V0.2.0

Status: implemented in `das_cva_mappo/`.

Scope:

- Adds a DAS-owned learnable satellite-task edge scorer in
  `das_cva_mappo/candidate_scorer.py`.
- Keeps the existing environment and allocator as the scheduling compatibility
  layer while replacing the candidate edge value used for ranking.
- Uses the transparent heuristic scorer as a feasibility oracle and warm-start
  teacher.
- Adds `hybrid` scoring by default: the heuristic edge value is mixed with the
  learned edge value before candidate allocation.
- Saves scorer warm-start metadata in `manifest.json` and, for learned scorer
  runs, writes `candidate_scorer.pt` in the run directory.

Run:

```bash
python -m das_cva_mappo.run_experiment
```

New or expanded ablation interfaces:

- `--candidate_scorer_mode v2_heuristic|learned|hybrid`
- `--candidate_scorer_mix`
- `--candidate_scorer_hidden_dim`
- `--candidate_scorer_lr`
- `--candidate_warmup_edges`
- `--candidate_warmup_epochs`
- `--candidate_warmup_batch_size`

Known limitations:

- The learnable scorer is warm-started from the heuristic teacher, but is not
  yet updated online from rollout advantages in V0.2.
- Candidate allocation still reuses the existing capacity-aware allocator.
- Evaluation is serial in the DAS runner.

## DAS-CVA-MAPPO V0.1.0

Status: implemented in `das_cva_mappo/`.

Scope:

- Adds the first action-set-aware MAPPO path.
- Keeps `cva_mappo_v2` as the candidate generation and environment support
  layer.
- Replaces fixed slot-ID actor semantics with per-action entity scoring.
- Stores action feature snapshots in the PPO rollout buffer, so PPO ratios are
  recomputed against the action set seen at sampling time.
- Adds a standalone runner:

```bash
python -m das_cva_mappo.run_experiment
```

Implemented modules:

- `das_cva_mappo/config.py`
- `das_cva_mappo/feature_builder.py`
- `das_cva_mappo/action_set_actor.py`
- `das_cva_mappo/rollout_buffer.py`
- `das_cva_mappo/trainer.py`
- `das_cva_mappo/run_experiment.py`

Ablation interfaces:

- `--matcher additive|dot|set_transformer`
- `--action_feature_mode full|minimal|no_score`
- `--no_candidate_score_feature`
- `--no_set_context`
- `--no_action_type_gate`
- `--idle_valid_penalty`
- `--idle_aux_coeff`
- `--candidate_dropout_prob`
- `--candidate_scorer_mode v2_heuristic`
- inherited candidate-layer controls such as `--slot_selection_mode`,
  `--ownership_mask_mode`, `--candidate_owner_bonus`, and
  `--dynamic_broadcast_window_s`

Known limitations:

- Candidate generation still uses the v2 heuristic CVA scorer and allocator.
- The learnable CVA edge scorer is recorded as an interface target but is not
  trained in V0.1.
- Evaluation is serial in this first DAS runner.
