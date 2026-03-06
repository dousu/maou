import pytest

from maou.interface.learn import (
    SUPPORTED_LR_SCHEDULERS,
    _resolve_stage12_scheduler,
    learn,
    learn_multi_stage,
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


class TestGradientAccumulationStepsValidation:
    """gradient_accumulation_steps バリデーションのテスト．"""

    def test_learn_rejects_zero(self) -> None:
        """learn() で gradient_accumulation_steps=0 は ValueError．"""
        with pytest.raises(
            ValueError,
            match="gradient_accumulation_steps must be >= 1",
        ):
            learn(
                datasource=None,  # type: ignore[arg-type]
                gradient_accumulation_steps=0,
            )

    def test_learn_rejects_negative(self) -> None:
        """learn() で gradient_accumulation_steps=-1 は ValueError．"""
        with pytest.raises(
            ValueError,
            match="gradient_accumulation_steps must be >= 1",
        ):
            learn(
                datasource=None,  # type: ignore[arg-type]
                gradient_accumulation_steps=-1,
            )

    def test_learn_multi_stage_rejects_zero(self) -> None:
        """learn_multi_stage() で gradient_accumulation_steps=0 は ValueError．"""
        with pytest.raises(
            ValueError,
            match="gradient_accumulation_steps must be >= 1",
        ):
            learn_multi_stage(
                stage="1",
                gradient_accumulation_steps=0,
            )

    def test_learn_multi_stage_rejects_negative(self) -> None:
        """learn_multi_stage() で gradient_accumulation_steps=-1 は ValueError．"""
        with pytest.raises(
            ValueError,
            match="gradient_accumulation_steps must be >= 1",
        ):
            learn_multi_stage(
                stage="1",
                gradient_accumulation_steps=-1,
            )


class TestResolveStage12Scheduler:
    """_resolve_stage12_scheduler()のテスト．"""

    def test_auto_enables_scheduler_for_large_batch(
        self,
    ) -> None:
        """'auto' + batch_size > 256 → 'warmup_cosine_decay'"""
        result = _resolve_stage12_scheduler(
            "auto", actual_batch_size=4096
        )
        assert result == "warmup_cosine_decay"

    def test_auto_disables_scheduler_for_small_batch(
        self,
    ) -> None:
        """'auto' + batch_size <= 256 → None"""
        result = _resolve_stage12_scheduler(
            "auto", actual_batch_size=256
        )
        assert result is None

    def test_none_disables_scheduler(self) -> None:
        """'none' → None (regardless of batch size)"""
        result = _resolve_stage12_scheduler(
            "none", actual_batch_size=4096
        )
        assert result is None

    def test_explicit_scheduler_name(self) -> None:
        """明示的なスケジューラ名はそのまま正規化される．"""
        result = _resolve_stage12_scheduler(
            "Warmup+CosineDecay", actual_batch_size=256
        )
        assert result == "warmup_cosine_decay"
