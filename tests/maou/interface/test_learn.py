import pytest

from maou.interface.learn import (
    SUPPORTED_LR_SCHEDULERS,
    normalize_lr_scheduler_name,
)


def test_normalize_lr_scheduler_name_accepts_display_names() -> (
    None
):
    canonical = normalize_lr_scheduler_name(
        "Warmup+CosineDecay"
    )
    assert canonical == "warmup_cosine_decay"

    canonical = normalize_lr_scheduler_name("CosineAnnealingLR")
    assert canonical == "cosine_annealing_lr"


def test_normalize_lr_scheduler_name_accepts_underscored_alias() -> (
    None
):
    canonical = normalize_lr_scheduler_name(
        "cosine_annealing_lr"
    )
    assert canonical == "cosine_annealing_lr"


def test_normalize_lr_scheduler_name_rejects_unknown_scheduler() -> (
    None
):
    with pytest.raises(ValueError):
        normalize_lr_scheduler_name("linear")


def test_supported_lr_scheduler_labels_match_defaults() -> None:
    assert (
        SUPPORTED_LR_SCHEDULERS["warmup_cosine_decay"]
        == "Warmup+CosineDecay"
    )
    assert (
        SUPPORTED_LR_SCHEDULERS["cosine_annealing_lr"]
        == "CosineAnnealingLR"
    )
