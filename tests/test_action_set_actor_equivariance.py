from __future__ import annotations

try:
    import torch

    from das_cva_mappo.action_set_actor import ActionSetActor
    from das_cva_mappo.trainer import ActionSetMAPPOTrainer
except ModuleNotFoundError:  # Local lightweight checks may not install torch.
    torch = None
    ActionSetActor = None
    ActionSetMAPPOTrainer = None


def test_action_set_actor_permutation_equivariance() -> None:
    if torch is None or ActionSetActor is None:
        return

    torch.manual_seed(7)
    batch_size = 3
    state_dim = 6
    action_dim = 7
    action_feature_dim = 12
    perm = torch.tensor([3, 1, 6, 0, 5, 2, 4])
    mask = torch.tensor([
        [1, 1, 0, 1, 1, 0, 1],
        [1, 0, 1, 1, 0, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
    ], dtype=torch.float32)
    state = torch.randn(batch_size, state_dim)
    action_features = torch.randn(batch_size, action_dim, action_feature_dim)

    for matcher in ("additive", "dot", "set_transformer"):
        actor = ActionSetActor(
            state_dim=state_dim,
            action_feature_dim=action_feature_dim,
            hidden_dims=(16,),
            action_hidden_dim=16,
            matcher=matcher,
            use_set_context=True,
            use_action_type_gate=True,
            idle_valid_penalty=0.0,
        )
        actor.eval()
        with torch.no_grad():
            probs = actor(state, action_features, mask).probs
            permuted_probs = actor(state, action_features[:, perm], mask[:, perm]).probs
        assert torch.allclose(permuted_probs, probs[:, perm], atol=1e-5)


def test_dynamic_select_aux_loss_prefers_dynamic_mass() -> None:
    if torch is None or ActionSetMAPPOTrainer is None:
        return

    features = torch.zeros(1, 3, 28)
    features[0, :, 0] = 1.0
    features[0, 1, 6] = 1.0
    features[0, 2, 19] = 1.0
    mask = torch.ones(1, 3)
    low_dynamic = torch.distributions.Categorical(probs=torch.tensor([[0.8, 0.1, 0.1]]))
    high_dynamic = torch.distributions.Categorical(probs=torch.tensor([[0.1, 0.45, 0.45]]))

    low_loss = ActionSetMAPPOTrainer._dynamic_select_aux_loss(low_dynamic, features, mask)
    high_loss = ActionSetMAPPOTrainer._dynamic_select_aux_loss(high_dynamic, features, mask)
    assert high_loss < low_loss


if __name__ == "__main__":
    test_action_set_actor_permutation_equivariance()
    test_dynamic_select_aux_loss_prefers_dynamic_mass()
