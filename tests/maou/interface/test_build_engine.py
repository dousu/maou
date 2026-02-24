"""build_engine interface層のテスト．"""

from pathlib import Path
from unittest.mock import MagicMock, patch


class TestBuildEngine:
    """build_engine 関数のテスト．"""

    def test_build_engine_returns_elapsed_time(
        self, tmp_path: Path
    ) -> None:
        """戻り値に 'built in' と 's, saved to:' が含まれること（ビルド時間表示の確認）．"""
        mock_trt_inference = MagicMock()
        mock_trt_inference.build_engine_from_onnx.return_value = b"dummy_engine"

        with patch.dict(
            "sys.modules",
            {
                "maou.app.inference.tensorrt_inference": MagicMock(
                    TensorRTInference=mock_trt_inference
                ),
            },
        ):
            from maou.interface.build_engine import (
                build_engine,
            )

            model_path = tmp_path / "model.onnx"
            model_path.touch()
            output_path = tmp_path / "model.engine"

            result = build_engine(
                model_path=model_path,
                output=output_path,
                trt_workspace_size=256,
            )

            assert "built in" in result
            assert "s, saved to:" in result
