# Version History

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
