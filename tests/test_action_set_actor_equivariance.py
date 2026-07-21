from __future__ import annotations

try:
    import torch

    from das_cva_mappo.action_set_actor import ActionSetActor
except ModuleNotFoundError:  # Local lightweight checks may not install torch.
    torch = None
    ActionSetActor = None


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


if __name__ == "__main__":
    test_action_set_actor_permutation_equivariance()
