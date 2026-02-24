"""is_tracing 関数のユニットテスト．"""

from unittest.mock import patch

from maou.domain.model.tracing import is_tracing


class TestIsTracing:
    """is_tracing がトレーシング状態を正しく検出するかを検証する．"""

    def test_is_tracing_returns_false_normally(self) -> None:
        """通常実行時は False を返す．"""
        assert is_tracing() is False

    @patch(
        "maou.domain.model.tracing.torch.compiler.is_compiling",
        return_value=True,
    )
    def test_is_tracing_detects_compiler(
        self, _mock_is_compiling: object
    ) -> None:
        """torch.compiler.is_compiling() が True のとき True を返す．"""
        assert is_tracing() is True

    @patch(
        "maou.domain.model.tracing.torch.jit.is_tracing",
        return_value=True,
    )
    def test_is_tracing_detects_jit(
        self, _mock_is_tracing: object
    ) -> None:
        """torch.jit.is_tracing() が True のとき True を返す．"""
        assert is_tracing() is True

    @patch(
        "maou.domain.model.tracing.torch.onnx.is_in_onnx_export",
        return_value=True,
    )
    def test_is_tracing_detects_onnx(
        self, _mock_is_in_onnx_export: object
    ) -> None:
        """torch.onnx.is_in_onnx_export() が True のとき True を返す．"""
        assert is_tracing() is True
