# Version History

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

- `--matcher additive|dot`
- `--action_feature_mode full|minimal|no_score`
- `--no_candidate_score_feature`
- `--no_set_context`
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
