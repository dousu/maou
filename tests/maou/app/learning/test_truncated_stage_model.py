"""Tests for TruncatedStageModel layer separation in multi-stage training."""

import pytest
import torch
from torch import nn

from maou.app.learning.multi_stage_training import (
    MultiStageTrainingOrchestrator,
    TruncatedStageModel,
)
from maou.app.learning.network import (
    PIECES_IN_HAND_VECTOR_SIZE,
    HeadlessNetwork,
    ReachableSquaresHead,
)


def _make_backbone(
    *,
    hand_projection_dim: int = 0,
) -> HeadlessNetwork:
    """テスト用の小規模 HeadlessNetwork を生成する．"""
    return HeadlessNetwork(
        board_vocab_size=32,
        hand_projection_dim=hand_projection_dim,
        embedding_dim=64,
        architecture="resnet",
        out_channels=(16, 32, 64, 64),
    )


class TestTruncatedStageModelForward:
    """TruncatedStageModel の forward 出力形状テスト．"""

    def test_forward_output_shape(self) -> None:
        """forward の出力が (batch, head_out_dim) であることを検証する．"""
        backbone = _make_backbone()
        head = ReachableSquaresHead(
            input_dim=backbone.embedding_dim,
        )
        model = TruncatedStageModel(
            backbone, head, trainable_layers=2
        )

        board = torch.randint(0, 32, (4, 9, 9))
        logits, dummy_value = model((board, None))

        assert logits.shape == (4, 81)
        assert dummy_value.shape == (4, 1)

    def test_forward_with_hand_projection(self) -> None:
        """hand_projection_dim > 0 でも正しく動作することを検証する．"""
        backbone = _make_backbone(hand_projection_dim=8)
        head = ReachableSquaresHead(
            input_dim=backbone.embedding_dim,
        )
        model = TruncatedStageModel(
            backbone, head, trainable_layers=2
        )

        board = torch.randint(0, 32, (4, 9, 9))
        hand = torch.randn(4, PIECES_IN_HAND_VECTOR_SIZE)
        logits, dummy_value = model((board, hand))

        assert logits.shape == (4, 81)
        assert dummy_value.shape == (4, 1)

    def test_forward_trainable_layers_1(self) -> None:
        """trainable_layers=1 で 3 グループ使用時の出力形状を検証する．"""
        backbone = _make_backbone()
        head = ReachableSquaresHead(
            input_dim=backbone.embedding_dim,
        )
        model = TruncatedStageModel(
            backbone, head, trainable_layers=1
        )

        board = torch.randint(0, 32, (2, 9, 9))
        logits, dummy_value = model((board, None))

        assert logits.shape == (2, 81)
        assert dummy_value.shape == (2, 1)

    def test_forward_trainable_layers_3(self) -> None:
        """trainable_layers=3 で 1 グループのみ使用時の出力形状を検証する．"""
        backbone = _make_backbone()
        head = ReachableSquaresHead(
            input_dim=backbone.embedding_dim,
        )
        model = TruncatedStageModel(
            backbone, head, trainable_layers=3
        )

        board = torch.randint(0, 32, (2, 9, 9))
        logits, dummy_value = model((board, None))

        assert logits.shape == (2, 81)
        assert dummy_value.shape == (2, 1)


class TestTruncatedStageModelParameterSharing:
    """TruncatedStageModel のパラメータ共有テスト．"""

    def test_partial_backbone_shares_parameters_with_original(
        self,
    ) -> None:
        """partial_backbone のパラメータが元の backbone と同一オブジェクトであることを検証する．"""
        backbone = _make_backbone()
        head = ReachableSquaresHead(
            input_dim=backbone.embedding_dim,
        )
        model = TruncatedStageModel(
            backbone, head, trainable_layers=2
        )

        # partial_backbone[0] は backbone.backbone.layer1 と同一オブジェクト
        groups = backbone.backbone.get_freezable_groups()
        assert model.partial_backbone[0] is groups[0]
        assert model.partial_backbone[1] is groups[1]

    def test_training_updates_original_backbone(self) -> None:
        """TruncatedStageModel の訓練で元の backbone のパラメータが更新されることを検証する．"""
        backbone = _make_backbone()
        head = ReachableSquaresHead(
            input_dim=backbone.embedding_dim,
        )
        model = TruncatedStageModel(
            backbone, head, trainable_layers=2
        )

        # 訓練前のパラメータをコピー
        groups = backbone.backbone.get_freezable_groups()
        layer1_param_before = next(
            groups[0].parameters()
        ).clone()

        # 1 step 訓練
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        board = torch.randint(0, 32, (4, 9, 9))
        logits, _ = model((board, None))
        loss = logits.sum()
        loss.backward()
        optimizer.step()

        # 元の backbone の layer1 パラメータが更新されている
        layer1_param_after = next(groups[0].parameters())
        assert not torch.equal(
            layer1_param_before, layer1_param_after
        )


class TestTruncatedStageModelGradientFlow:
    """TruncatedStageModel の勾配フローテスト．"""

    def test_used_groups_receive_gradients(self) -> None:
        """使用グループに勾配が流れることを検証する．"""
        backbone = _make_backbone()
        head = ReachableSquaresHead(
            input_dim=backbone.embedding_dim,
        )
        model = TruncatedStageModel(
            backbone, head, trainable_layers=2
        )

        board = torch.randint(0, 32, (4, 9, 9))
        logits, _ = model((board, None))
        loss = logits.sum()
        loss.backward()

        # 使用グループ (layer1, layer2) に勾配あり
        groups = backbone.backbone.get_freezable_groups()
        for group in groups[:2]:
            has_grad = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in group.parameters()
            )
            assert has_grad, "Used group should have gradients"

    def test_excluded_groups_have_no_gradients(self) -> None:
        """除外グループに勾配が流れないことを検証する．"""
        backbone = _make_backbone()
        head = ReachableSquaresHead(
            input_dim=backbone.embedding_dim,
        )
        model = TruncatedStageModel(
            backbone, head, trainable_layers=2
        )

        board = torch.randint(0, 32, (4, 9, 9))
        logits, _ = model((board, None))
        loss = logits.sum()
        loss.backward()

        # 除外グループ (layer3, layer4) に勾配なし
        groups = backbone.backbone.get_freezable_groups()
        for group in groups[2:]:
            for p in group.parameters():
                assert (
                    p.grad is None or p.grad.abs().sum() == 0
                ), "Excluded group should have no gradients"

    def test_excluded_groups_requires_grad_false(self) -> None:
        """除外グループのパラメータが requires_grad=False であることを検証する．"""
        backbone = _make_backbone()
        head = ReachableSquaresHead(
            input_dim=backbone.embedding_dim,
        )
        TruncatedStageModel(backbone, head, trainable_layers=2)

        groups = backbone.backbone.get_freezable_groups()

        # 使用グループ (layer1, layer2) は requires_grad=True
        for group in groups[:2]:
            for p in group.parameters():
                assert p.requires_grad, (
                    "Used group should have requires_grad=True"
                )

        # 除外グループ (layer3, layer4) は requires_grad=False
        for group in groups[2:]:
            for p in group.parameters():
                assert not p.requires_grad, (
                    "Excluded group should have requires_grad=False"
                )


class TestTruncatedStageModelValidation:
    """TruncatedStageModel のバリデーションテスト．"""

    def test_trainable_layers_ge_total_groups_raises_value_error(
        self,
    ) -> None:
        """trainable_layers >= total_groups で ValueError が発生することを検証する．"""
        backbone = _make_backbone()
        head = ReachableSquaresHead(
            input_dim=backbone.embedding_dim,
        )

        with pytest.raises(ValueError, match="must be less"):
            TruncatedStageModel(
                backbone, head, trainable_layers=4
            )

    def test_trainable_layers_exceeds_total_groups_raises_value_error(
        self,
    ) -> None:
        """trainable_layers > total_groups でも ValueError が発生することを検証する．"""
        backbone = _make_backbone()
        head = ReachableSquaresHead(
            input_dim=backbone.embedding_dim,
        )

        with pytest.raises(ValueError, match="must be less"):
            TruncatedStageModel(
                backbone, head, trainable_layers=5
            )

    def test_mlp_mixer_architecture_accepted(
        self,
    ) -> None:
        """MLP-Mixer アーキテクチャが受け入れられることを検証する．"""
        backbone = HeadlessNetwork(
            board_vocab_size=32,
            embedding_dim=64,
            architecture="mlp-mixer",
            architecture_config={
                "embed_dim": 128,
                "depth": 4,
            },
        )
        head = ReachableSquaresHead(
            input_dim=backbone.embedding_dim,
        )

        model = TruncatedStageModel(
            backbone, head, trainable_layers=2
        )
        assert not model._is_resnet


class TestOrchestratorValidation:
    """MultiStageTrainingOrchestrator のバリデーションテスト．"""

    def test_trainable_layers_none_does_not_validate(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """trainable_layers=None で従来動作が維持されることを検証する．"""
        backbone = _make_backbone()
        orchestrator = MultiStageTrainingOrchestrator(
            backbone=backbone,
            device=torch.device("cpu"),
            model_dir=tmp_path / "checkpoints",  # type: ignore[operator]
            trainable_layers=None,
        )
        assert orchestrator.trainable_layers is None

    def test_trainable_layers_zero_does_not_validate(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """trainable_layers=0 でバリデーションエラーにならないことを検証する．"""
        backbone = _make_backbone()
        orchestrator = MultiStageTrainingOrchestrator(
            backbone=backbone,
            device=torch.device("cpu"),
            model_dir=tmp_path / "checkpoints",  # type: ignore[operator]
            trainable_layers=0,
        )
        assert orchestrator.trainable_layers == 0

    def test_trainable_layers_ge_total_raises_value_error(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """orchestrator で trainable_layers >= total_groups の ValueError を検証する．"""
        backbone = _make_backbone()

        with pytest.raises(ValueError, match="must be less"):
            MultiStageTrainingOrchestrator(
                backbone=backbone,
                device=torch.device("cpu"),
                model_dir=tmp_path / "checkpoints",  # type: ignore[operator]
                trainable_layers=4,
            )

    def test_mlp_mixer_with_trainable_layers_accepted(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """MLP-Mixer + trainable_layers > 0 で orchestrator が受け入れられることを検証する．"""
        backbone = HeadlessNetwork(
            board_vocab_size=32,
            embedding_dim=64,
            architecture="mlp-mixer",
            architecture_config={
                "embed_dim": 128,
                "depth": 4,
            },
        )

        orchestrator = MultiStageTrainingOrchestrator(
            backbone=backbone,
            device=torch.device("cpu"),
            model_dir=tmp_path / "checkpoints",  # type: ignore[operator]
            trainable_layers=2,
        )
        assert orchestrator.trainable_layers == 2


class TestComputeOutputChannels:
    """_compute_output_channels の出力チャンネル推定テスト．"""

    def test_compute_output_channels_returns_correct_value(
        self,
    ) -> None:
        """ダミー入力による出力チャンネル推定が正しいことを検証する．"""
        backbone = _make_backbone()
        groups = backbone.backbone.get_freezable_groups()

        # 2グループ使用時の出力チャンネル
        partial = nn.Sequential(*groups[:2])
        out_ch = TruncatedStageModel._compute_output_channels(
            partial,
            backbone.input_channels,
            backbone._board_size,
        )

        # out_channels=(16, 32, 64, 64) で expansion=4 (BottleneckBlock)
        # layer2 の出力: 32 * 4 = 128
        assert out_ch == 128


def _make_mixer_backbone(
    *,
    hand_projection_dim: int = 0,
) -> HeadlessNetwork:
    """テスト用の小規模 MLP-Mixer HeadlessNetwork を生成する．"""
    return HeadlessNetwork(
        board_vocab_size=32,
        hand_projection_dim=hand_projection_dim,
        embedding_dim=64,
        architecture="mlp-mixer",
        architecture_config={
            "embed_dim": 128,
            "depth": 4,
        },
    )


def _make_vit_backbone(
    *,
    hand_projection_dim: int = 0,
) -> HeadlessNetwork:
    """テスト用の小規模 ViT HeadlessNetwork を生成する．"""
    return HeadlessNetwork(
        board_vocab_size=32,
        hand_projection_dim=hand_projection_dim,
        embedding_dim=64,
        architecture="vit",
        architecture_config={
            "embed_dim": 128,
            "num_heads": 4,
            "num_layers": 4,
            "mlp_ratio": 2.0,
        },
    )


class TestMixerTruncatedStageModel:
    """MLP-Mixer での TruncatedStageModel テスト．"""

    def test_forward_output_shape(self) -> None:
        """MLP-Mixer で forward の出力形状が正しいことを検証する．"""
        backbone = _make_mixer_backbone()
        head = ReachableSquaresHead(
            input_dim=backbone.embedding_dim,
        )
        model = TruncatedStageModel(
            backbone, head, trainable_layers=2
        )

        board = torch.randint(0, 32, (4, 9, 9))
        logits, dummy_value = model((board, None))

        assert logits.shape == (4, 81)
        assert dummy_value.shape == (4, 1)

    def test_parameter_sharing(self) -> None:
        """MLP-Mixer で partial_backbone が元の backbone とパラメータを共有することを検証する．"""
        backbone = _make_mixer_backbone()
        head = ReachableSquaresHead(
            input_dim=backbone.embedding_dim,
        )
        model = TruncatedStageModel(
            backbone, head, trainable_layers=2
        )

        groups = backbone.backbone.get_freezable_groups()
        assert model.partial_backbone[0] is groups[0]
        assert model.partial_backbone[1] is groups[1]

    def test_gradient_flow(self) -> None:
        """MLP-Mixer で使用ブロックに勾配あり，除外ブロックに勾配なしを検証する．"""
        backbone = _make_mixer_backbone()
        head = ReachableSquaresHead(
            input_dim=backbone.embedding_dim,
        )
        model = TruncatedStageModel(
            backbone, head, trainable_layers=2
        )

        board = torch.randint(0, 32, (4, 9, 9))
        logits, _ = model((board, None))
        loss = logits.sum()
        loss.backward()

        groups = backbone.backbone.get_freezable_groups()
        # 使用ブロックに勾配あり
        for group in groups[:2]:
            has_grad = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in group.parameters()
            )
            assert has_grad

        # 除外ブロックは requires_grad=False
        for group in groups[2:]:
            for p in group.parameters():
                assert not p.requires_grad

    def test_truncation_norm_exists(self) -> None:
        """MLP-Mixer で truncation_norm が作成されることを検証する．"""
        backbone = _make_mixer_backbone()
        head = ReachableSquaresHead(
            input_dim=backbone.embedding_dim,
        )
        model = TruncatedStageModel(
            backbone, head, trainable_layers=2
        )

        assert hasattr(model, "truncation_norm")
        assert not hasattr(model, "projection_pool")
        assert not hasattr(model, "projection_linear")


class TestViTTruncatedStageModel:
    """ViT での TruncatedStageModel テスト．"""

    def test_forward_output_shape(self) -> None:
        """ViT で forward の出力形状が正しいことを検証する．"""
        backbone = _make_vit_backbone()
        head = ReachableSquaresHead(
            input_dim=backbone.embedding_dim,
        )
        model = TruncatedStageModel(
            backbone, head, trainable_layers=2
        )

        board = torch.randint(0, 32, (4, 9, 9))
        logits, dummy_value = model((board, None))

        assert logits.shape == (4, 81)
        assert dummy_value.shape == (4, 1)

    def test_parameter_sharing(self) -> None:
        """ViT で partial_backbone が元の backbone とパラメータを共有することを検証する．"""
        backbone = _make_vit_backbone()
        head = ReachableSquaresHead(
            input_dim=backbone.embedding_dim,
        )
        model = TruncatedStageModel(
            backbone, head, trainable_layers=2
        )

        groups = backbone.backbone.get_freezable_groups()
        assert model.partial_backbone[0] is groups[0]
        assert model.partial_backbone[1] is groups[1]

    def test_gradient_flow(self) -> None:
        """ViT で使用ブロックに勾配あり，除外ブロックに勾配なしを検証する．"""
        backbone = _make_vit_backbone()
        head = ReachableSquaresHead(
            input_dim=backbone.embedding_dim,
        )
        model = TruncatedStageModel(
            backbone, head, trainable_layers=2
        )

        board = torch.randint(0, 32, (4, 9, 9))
        logits, _ = model((board, None))
        loss = logits.sum()
        loss.backward()

        groups = backbone.backbone.get_freezable_groups()
        # 使用ブロックに勾配あり
        for group in groups[:2]:
            has_grad = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in group.parameters()
            )
            assert has_grad

        # 除外ブロックは requires_grad=False
        for group in groups[2:]:
            for p in group.parameters():
                assert not p.requires_grad
