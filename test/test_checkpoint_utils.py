import torch


def test_filter_state_dict_by_shape_filters_mismatches():
    from dino_wm.checkpoint_utils import filter_state_dict_by_shape

    model_state = {
        "layer.weight": torch.zeros(2, 2),
        "layer.bias": torch.zeros(2),
    }
    ckpt_state = {
        "layer.weight": torch.ones(2, 2),
        "layer.bias": torch.ones(3),  # shape mismatch
        "extra.weight": torch.ones(1),
    }

    filtered, missing, unexpected, mismatched = filter_state_dict_by_shape(
        model_state,
        ckpt_state,
    )

    assert list(filtered.keys()) == ["layer.weight"]
    assert missing == ["layer.bias"]
    assert unexpected == ["extra.weight"]
    assert mismatched == ["layer.bias"]
