"""TensorRT推論のエラーハンドリングテスト．

TensorRT/CUDAモジュールはGPU環境でのみ利用可能なため，
モックで差し替えてテストする．
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _create_trt_mock() -> MagicMock:
    """tensorrtモジュールのモックを生成する．"""
    trt_mock = MagicMock()
    trt_mock.NetworkDefinitionCreationFlag.EXPLICIT_BATCH = 0
    trt_mock.MemoryPoolType.WORKSPACE = 0
    trt_mock.BuilderFlag.FP16 = 0
    return trt_mock


class _DummyCUresult:
    """isinstance()で使用するためのダミー型．"""

    CUDA_SUCCESS = 0


class _DummyCudaErrorT:
    """isinstance()で使用するためのダミー型．"""

    cudaSuccess = 0


def _create_cuda_mocks() -> dict[str, MagicMock]:
    """cuda関連モジュールのモック辞書を生成する．

    Returns:
        sys.modules用のモック辞書．
        "cuda.bindings.driver" と "cuda.bindings.runtime" キーで
        個別モックにアクセス可能．
    """
    cuda_driver_mock = MagicMock()
    cuda_driver_mock.CUresult = _DummyCUresult
    cudart_mock = MagicMock()
    cudart_mock.cudaError_t = _DummyCudaErrorT
    cuda_bindings_mock = MagicMock()
    cuda_bindings_mock.driver = cuda_driver_mock
    cuda_bindings_mock.runtime = cudart_mock
    cuda_top_mock = MagicMock()
    cuda_top_mock.bindings = cuda_bindings_mock
    return {
        "cuda": cuda_top_mock,
        "cuda.bindings": cuda_bindings_mock,
        "cuda.bindings.driver": cuda_driver_mock,
        "cuda.bindings.runtime": cudart_mock,
    }


class TestBuildEngineFromOnnx:
    """_build_engine_from_onnx のエラーハンドリングテスト．"""

    def test_raises_runtime_error_when_build_returns_none(
        self,
    ) -> None:
        """build_serialized_network()がNoneを返した場合にRuntimeErrorが送出される．"""
        trt_mock = _create_trt_mock()
        cuda_mocks = _create_cuda_mocks()

        # パーサーは成功するように設定
        parser_mock = MagicMock()
        parser_mock.parse_from_file.return_value = True
        parser_mock.num_errors = 0
        trt_mock.OnnxParser.return_value = parser_mock

        # ビルダーがNoneを返すように設定
        builder_mock = MagicMock()
        builder_mock.platform_has_fast_fp16 = False
        network_mock = MagicMock()
        network_mock.num_inputs = 0
        builder_mock.create_network.return_value = network_mock
        builder_mock.build_serialized_network.return_value = (
            None
        )
        trt_mock.Builder.return_value = builder_mock

        with patch.dict(
            sys.modules,
            {"tensorrt": trt_mock, **cuda_mocks},
        ):
            # モジュールキャッシュをクリアして再インポート
            sys.modules.pop(
                "maou.app.inference.tensorrt_inference", None
            )
            from maou.app.inference.tensorrt_inference import (
                TensorRTInference,
            )

            with pytest.raises(
                RuntimeError,
                match="Failed to build TensorRT engine",
            ):
                TensorRTInference._build_engine_from_onnx(
                    Path("/dummy/model.onnx")
                )

    def test_logs_error_and_no_success_on_build_failure(
        self,
    ) -> None:
        """ビルド失敗時にエラーログが出力され，成功ログが出力されない．"""
        trt_mock = _create_trt_mock()
        cuda_mocks = _create_cuda_mocks()

        parser_mock = MagicMock()
        parser_mock.parse_from_file.return_value = True
        parser_mock.num_errors = 0
        trt_mock.OnnxParser.return_value = parser_mock

        builder_mock = MagicMock()
        builder_mock.platform_has_fast_fp16 = False
        network_mock = MagicMock()
        network_mock.num_inputs = 0
        builder_mock.create_network.return_value = network_mock
        builder_mock.build_serialized_network.return_value = (
            None
        )
        trt_mock.Builder.return_value = builder_mock

        with patch.dict(
            sys.modules,
            {"tensorrt": trt_mock, **cuda_mocks},
        ):
            sys.modules.pop(
                "maou.app.inference.tensorrt_inference", None
            )
            from maou.app.inference import tensorrt_inference
            from maou.app.inference.tensorrt_inference import (
                TensorRTInference,
            )

            with (
                patch.object(
                    tensorrt_inference.logger, "error"
                ) as mock_error,
                patch.object(
                    tensorrt_inference.logger, "info"
                ) as mock_info,
                pytest.raises(RuntimeError),
            ):
                TensorRTInference._build_engine_from_onnx(
                    Path("/dummy/model.onnx")
                )

            # エラーログが出力されたこと
            mock_error.assert_called_once()
            assert "TensorRT engine build failed for" in str(
                mock_error.call_args
            )
            # 成功ログが出力されていないこと
            for call in mock_info.call_args_list:
                assert (
                    "TensorRT engine built successfully"
                    not in str(call)
                )


class TestDeserializeEngine:
    """infer() の deserialize_cuda_engine エラーハンドリングテスト．"""

    def test_raises_runtime_error_when_deserialize_returns_none(
        self,
    ) -> None:
        """deserialize_cuda_engine()がNoneを返した場合にRuntimeErrorが送出される．"""
        trt_mock = _create_trt_mock()
        cuda_mocks = _create_cuda_mocks()
        cudart_mock = cuda_mocks["cuda.bindings.runtime"]

        # deserialize_cuda_engine がNoneを返すように設定
        runtime_mock = MagicMock()
        runtime_mock.deserialize_cuda_engine.return_value = None
        trt_mock.Runtime.return_value = runtime_mock

        # cudaStreamCreate が成功を返すように設定
        cudart_mock.cudaStreamCreate.return_value = (
            MagicMock(),
            MagicMock(),
        )

        with patch.dict(
            sys.modules,
            {"tensorrt": trt_mock, **cuda_mocks},
        ):
            sys.modules.pop(
                "maou.app.inference.tensorrt_inference", None
            )
            from maou.app.inference.tensorrt_inference import (
                TensorRTInference,
            )

            with (
                patch.object(
                    TensorRTInference,
                    "_build_engine_from_onnx",
                    return_value=b"dummy_engine_bytes",
                ),
                pytest.raises(
                    RuntimeError,
                    match="Failed to deserialize TensorRT engine",
                ),
            ):
                TensorRTInference.infer(
                    onnx_path=Path("/dummy/model.onnx"),
                    board_data=MagicMock(),
                    hand_data=MagicMock(),
                    num=5,
                    cuda_available=True,
                )
