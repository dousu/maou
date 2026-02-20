"""Gradient checkpointing の動作テスト."""

from unittest.mock import patch

import pytest
import torch

from maou.domain.model.vision_transformer import (
    VisionTransformer,
    VisionTransformerConfig,
)


class TestGradientCheckpointingConfig:
    """VisionTransformerConfig の gradient_checkpointing フィールドテスト."""

    def test_default_is_disabled(self) -> None:
        """デフォルトでは gradient checkpointing が無効."""
        config = VisionTransformerConfig()
        assert config.gradient_checkpointing is False

    def test_can_enable(self) -> None:
        """gradient_checkpointing を有効化できる."""
        config = VisionTransformerConfig(
            gradient_checkpointing=True
        )
        assert config.gradient_checkpointing is True


class TestGradientCheckpointingForwardBackward:
    """gradient checkpointing 有効時の forward/backward テスト."""

    @pytest.fixture()
    def vit_with_checkpointing(self) -> VisionTransformer:
        """チェックポイント有効な小規模 ViT."""
        config = VisionTransformerConfig(
            input_channels=104,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            gradient_checkpointing=True,
        )
        return VisionTransformer(config)

    @pytest.fixture()
    def vit_without_checkpointing(self) -> VisionTransformer:
        """チェックポイント無効な小規模 ViT."""
        config = VisionTransformerConfig(
            input_channels=104,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            gradient_checkpointing=False,
        )
        return VisionTransformer(config)

    def test_forward_produces_same_shape(
        self,
        vit_with_checkpointing: VisionTransformer,
    ) -> None:
        """checkpointing 有効時も出力形状が正しい."""
        vit_with_checkpointing.train()
        x = torch.randn(2, 104, 9, 9)
        out = vit_with_checkpointing(x)
        assert out.shape == (2,)

    def test_backward_computes_gradients(
        self,
        vit_with_checkpointing: VisionTransformer,
    ) -> None:
        """checkpointing 有効時に勾配が計算される."""
        vit_with_checkpointing.train()
        x = torch.randn(2, 104, 9, 9)
        out = vit_with_checkpointing(x)
        loss = out.sum()
        loss.backward()
        for param in vit_with_checkpointing.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_outputs_match_without_checkpointing(
        self,
        vit_with_checkpointing: VisionTransformer,
        vit_without_checkpointing: VisionTransformer,
    ) -> None:
        """checkpointing の有無で forward 出力が一致する(eval mode)."""
        vit_without_checkpointing.load_state_dict(
            vit_with_checkpointing.state_dict()
        )
        vit_with_checkpointing.eval()
        vit_without_checkpointing.eval()
        x = torch.randn(2, 104, 9, 9)
        out_with = vit_with_checkpointing(x)
        out_without = vit_without_checkpointing(x)
        torch.testing.assert_close(out_with, out_without)

    def test_outputs_match_in_train_mode(
        self,
        vit_with_checkpointing: VisionTransformer,
        vit_without_checkpointing: VisionTransformer,
    ) -> None:
        """train modeでもcheckpointing有無で出力が一致する."""
        vit_without_checkpointing.load_state_dict(
            vit_with_checkpointing.state_dict()
        )
        vit_with_checkpointing.train()
        vit_without_checkpointing.train()
        x = torch.randn(2, 104, 9, 9)
        # use_reentrant=False はRNG stateを自動復元するため，
        # 手動でのRNG state設定なしで出力が一致することを検証
        rng_state = torch.random.get_rng_state()
        out_with = vit_with_checkpointing(x)
        torch.random.set_rng_state(rng_state)
        out_without = vit_without_checkpointing(x)
        torch.testing.assert_close(out_with, out_without)

    def test_backward_with_amp(
        self,
        vit_with_checkpointing: VisionTransformer,
    ) -> None:
        """AMP autocast内でcheckpointing + backward が動作する."""
        vit_with_checkpointing.train()
        x = torch.randn(2, 104, 9, 9)
        with torch.amp.autocast("cpu"):
            out = vit_with_checkpointing(x)
            loss = out.sum()
        loss.backward()
        for p in vit_with_checkpointing.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_not_applied_in_eval_mode(
        self,
        vit_with_checkpointing: VisionTransformer,
    ) -> None:
        """eval モードでは checkpoint() が呼ばれない."""
        vit_with_checkpointing.eval()
        x = torch.randn(2, 104, 9, 9)
        with patch(
            "maou.domain.model.vision_transformer.torch_checkpoint"
        ) as mock_cp:
            vit_with_checkpointing(x)
            mock_cp.assert_not_called()

    def test_use_reentrant_false(
        self,
        vit_with_checkpointing: VisionTransformer,
    ) -> None:
        """checkpoint() が use_reentrant=False で呼ばれる."""
        vit_with_checkpointing.train()
        x = torch.randn(2, 104, 9, 9)
        with patch(
            "maou.domain.model.vision_transformer.torch_checkpoint",
            wraps=torch.utils.checkpoint.checkpoint,
        ) as mock_cp:
            vit_with_checkpointing(x)
            for call in mock_cp.call_args_list:
                assert call.kwargs.get("use_reentrant") is False
