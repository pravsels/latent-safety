import torch

from dino_wm.dino_models import (
    normalize_acs,
    unnormalize_acs,
    normalize_states,
    unnormalize_states,
)


def test_action_normalize_unnormalize_min_max_roundtrip():
    acs = torch.tensor([[0.0, 5.0, 10.0]])
    min_ac = torch.tensor([0.0, 0.0, 0.0])
    max_ac = torch.tensor([10.0, 10.0, 10.0])

    norm = normalize_acs(acs, min_ac=min_ac, max_ac=max_ac)
    rec = unnormalize_acs(norm, min_ac=min_ac, max_ac=max_ac)

    assert torch.allclose(rec, acs)


def test_action_normalize_unnormalize_quantile_roundtrip():
    acs = torch.tensor([[1.0, 2.0, 3.0]])
    q02 = torch.tensor([0.0, 0.0, 0.0])
    q98 = torch.tensor([4.0, 4.0, 4.0])

    norm = normalize_acs(acs, q02=q02, q98=q98)
    rec = unnormalize_acs(norm, q02=q02, q98=q98)

    assert torch.allclose(rec, acs, atol=1e-6)


def test_state_normalize_unnormalize_min_max_roundtrip():
    states = torch.tensor([[1.0, 2.0]])
    min_state = torch.tensor([0.0, 0.0])
    max_state = torch.tensor([2.0, 4.0])

    norm = normalize_states(states, min_state=min_state, max_state=max_state)
    rec = unnormalize_states(norm, min_state=min_state, max_state=max_state)

    assert torch.allclose(rec, states)
