# DAS-CVA-MAPPO Design Summary

Source document:
`/Users/zhouzidie/Downloads/DAS-CVA-MAPPO_详细方案设计.docx`

## Goal

DAS-CVA-MAPPO targets multi-satellite, multi-ground-station dynamic mission
scheduling with observation, onboard storage, downlink, and optional
inter-satellite transfer. It is intended as a redesign on top of the current
`cva_mappo_v2` environment and candidate-generation implementation.

## Diagnosis Of The Current Compatibility Layer

The existing `cva_mappo_v2` path is useful but still has three structural
limits:

1. CVA scores are hand-weighted heuristic values, not learned long-horizon
   dispatch values.
2. Different satellites have different candidate tasks, but the current code
   forces them into fixed padded slots.
3. The shared actor predicts fixed slot IDs, so the same output neuron receives
   inconsistent semantics across satellites and timesteps.

The new method should treat each action as an entity with features, not as a
global slot index.

## Proposed Architecture

The recommended first implementation is not a full two-level RL system. It is:

1. a learnable CVA scorer that estimates satellite-task edge value;
2. a constrained allocator that generates each satellite's candidate action set;
3. an action-set-aware MAPPO actor that scores the current action entities;
4. a centralized critic, initially kept as the existing global-state MLP.

This keeps candidate generation and low-level scheduling coupled but avoids the
training instability of immediately introducing a high-level PPO controller.

## Learnable CVA

Candidate scoring should consume:

- satellite state: time, position, attitude, load, storage, pending images;
- task state: priority, dynamic flag, arrival, deadline, duration, data size;
- satellite-task edge features: earliest feasible observation, wait time,
  off-nadir quality, maneuver cost, future window count;
- satellite-ground-station features: next access, queue estimate, earliest
  downlink finish;
- delivery features: storage cost, downlink delay, delivery deadline margin,
  and whether relay is likely useful.

The first practical model can be an MLP edge scorer. Cross-attention or a
heterogeneous graph model can come later after the dynamic action-set path is
working.

## Dynamic Action-Set Actor

Each satellite acts over:

- task observation actions;
- inter-satellite transfer actions;
- one idle action.

The actor should encode local state and every action entity, then compute one
logit per available action:

```text
logit(i, a) = v^T tanh(W_s h_i + W_a e_i_a + W_c c_i)
pi(a | o_i, A_i) = softmax over A_i
```

Padding may still be used inside a batch, but padding must not define action
meaning. If candidate order is shuffled, action probabilities should shuffle in
the same way.

## Training Plan

1. Warm-start CVA from the current heuristic scorer and allocator.
2. Freeze or slowly update CVA while validating the variable-length actor and
   rollout buffer.
3. Alternate MAPPO updates with CVA auxiliary updates using rollout-derived
   positive and hard-negative satellite-task edges.

Important auxiliary losses:

- edge ranking/value fitting for CVA;
- conflict probability penalty for duplicate task selection;
- candidate load-balance penalty;
- candidate-set stability penalty.

## Recommended Code Direction

Keep `cva_mappo_v2` only as the environment and candidate-generation support
layer for early DAS versions. DAS V0.12 now implements the first action-set
policy path, a DAS-owned candidate edge scorer, rollout-advantage auxiliary
updates, hard-negative candidate sampling, conflict/load target shaping, a
candidate adapter boundary, event-aware compatibility-layer iteration, and
balanced dynamic-priority exposure/rescue with dynamic-first executable idle
rescue in `das_cva_mappo/`:

```text
das_cva_mappo/
  action_set_actor.py
  candidate_scorer.py
  config.py
  env_adapter.py
  feature_builder.py
  rollout_buffer.py
  run_experiment.py
  trainer.py
```

The remaining planned layout still includes these CVA integration pieces:

```text
das_cva_mappo/
  constrained_allocator.py
  run_experiment.py
```

Future experiments and method claims should be centered on `das_cva_mappo`.
Do not present the current v2 package as a primary experimental method; it is
only a compatibility layer while the learned CVA scorer and later DAS-specific
allocator are iterated.
