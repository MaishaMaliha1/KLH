import pytest

from klh.models import model_entry, supported_models


def test_model_registry_is_explicit():
    assert set(supported_models()) == {"vit_b16", "convnextv2_b", "swinv2_b", "dino_vit_b16"}
    assert model_entry("vit_b16").startswith("vit_base_patch16_224")


def test_unknown_model_fails():
    with pytest.raises(ValueError):
        model_entry("unknown")
