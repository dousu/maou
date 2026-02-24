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
    """build_engine_from_onnx のエラーハンドリングテスト．"""

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
                TensorRTInference.build_engine_from_onnx(
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
                TensorRTInference.build_engine_from_onnx(
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
                    "build_engine_from_onnx",
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


class TestWorkspaceSize:
    """workspace_size_mb パラメータのテスト．"""

    def _build_with_workspace_size(
        self, workspace_size_mb: int
    ) -> MagicMock:
        """指定したworkspace_size_mbでビルドし，builder_configモックを返す．"""
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
            b"dummy_engine"
        )
        trt_mock.Builder.return_value = builder_mock

        builder_config_mock = MagicMock()
        builder_mock.create_builder_config.return_value = (
            builder_config_mock
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

            TensorRTInference.build_engine_from_onnx(
                Path("/dummy/model.onnx"),
                workspace_size_mb=workspace_size_mb,
            )

        return builder_config_mock

    def test_default_workspace_size(self) -> None:
        """デフォルト値(256MB)で正しいバイト数が設定される．"""
        config_mock = self._build_with_workspace_size(256)
        config_mock.set_memory_pool_limit.assert_called_once_with(
            0,  # trt.MemoryPoolType.WORKSPACE のモック値
            256 * (1 << 20),
        )

    def test_custom_workspace_size(self) -> None:
        """カスタム値(512MB)で正しいバイト数が設定される．"""
        config_mock = self._build_with_workspace_size(512)
        config_mock.set_memory_pool_limit.assert_called_once_with(
            0,
            512 * (1 << 20),
        )


class TestSaveEngine:
    """save_engine のテスト．"""

    def test_save_engine_creates_file(
        self, tmp_path: Path
    ) -> None:
        """save_engine でバイト列がファイルに正しく書き込まれること．"""
        trt_mock = _create_trt_mock()
        cuda_mocks = _create_cuda_mocks()

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

            engine_data = b"dummy_engine_bytes_12345"
            engine_file = tmp_path / "test.engine"
            TensorRTInference.save_engine(
                engine_data, engine_file
            )

            assert engine_file.exists()
            assert engine_file.read_bytes() == engine_data

    def test_save_engine_creates_parent_dirs(
        self, tmp_path: Path
    ) -> None:
        """存在しない親ディレクトリが自動作成されること．"""
        trt_mock = _create_trt_mock()
        cuda_mocks = _create_cuda_mocks()

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

            engine_data = b"dummy_engine_bytes"
            engine_file = (
                tmp_path / "nested" / "dir" / "test.engine"
            )
            TensorRTInference.save_engine(
                engine_data, engine_file
            )

            assert engine_file.exists()
            assert engine_file.read_bytes() == engine_data


class TestLoadEngine:
    """load_engine のテスト．"""

    def test_load_engine_returns_bytes(
        self, tmp_path: Path
    ) -> None:
        """保存済みファイルからバイト列が正しく読み込まれること．"""
        trt_mock = _create_trt_mock()
        cuda_mocks = _create_cuda_mocks()

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

            engine_data = b"dummy_engine_bytes_67890"
            engine_file = tmp_path / "test.engine"
            engine_file.write_bytes(engine_data)

            result = TensorRTInference.load_engine(engine_file)
            assert result == engine_data

    def test_load_engine_file_not_found(
        self, tmp_path: Path
    ) -> None:
        """存在しないパスで FileNotFoundError が送出されること．"""
        trt_mock = _create_trt_mock()
        cuda_mocks = _create_cuda_mocks()

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

            with pytest.raises(
                FileNotFoundError,
                match="TensorRT engine file not found",
            ):
                TensorRTInference.load_engine(
                    tmp_path / "nonexistent.engine"
                )

    def test_load_engine_empty_file(
        self, tmp_path: Path
    ) -> None:
        """空ファイルで RuntimeError が送出されること．"""
        trt_mock = _create_trt_mock()
        cuda_mocks = _create_cuda_mocks()

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

            engine_file = tmp_path / "empty.engine"
            engine_file.write_bytes(b"")

            with pytest.raises(
                RuntimeError,
                match="TensorRT engine file is empty",
            ):
                TensorRTInference.load_engine(engine_file)


class TestInferWithEnginePath:
    """engine_path を指定した infer() のテスト．"""

    def test_infer_with_engine_path_skips_build(
        self, tmp_path: Path
    ) -> None:
        """engine_path 指定時に build_engine_from_onnx が呼ばれないこと．"""
        trt_mock = _create_trt_mock()
        cuda_mocks = _create_cuda_mocks()
        cudart_mock = cuda_mocks["cuda.bindings.runtime"]

        # deserialize_cuda_engine が成功を返すように設定
        runtime_mock = MagicMock()
        runtime_mock.deserialize_cuda_engine.return_value = (
            MagicMock()
        )
        trt_mock.Runtime.return_value = runtime_mock

        # cudaStreamCreate が成功を返すように設定
        cudart_mock.cudaStreamCreate.return_value = (
            MagicMock(),
            MagicMock(),
        )

        # エンジンファイルを作成
        engine_file = tmp_path / "test.engine"
        engine_file.write_bytes(b"dummy_engine_bytes")

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

            with patch.object(
                TensorRTInference,
                "build_engine_from_onnx",
            ) as mock_build:
                try:
                    TensorRTInference.infer(
                        onnx_path=None,
                        board_data=MagicMock(),
                        hand_data=MagicMock(),
                        num=5,
                        cuda_available=True,
                        engine_path=engine_file,
                    )
                except Exception:
                    pass  # 推論処理はモックなので途中で例外は許容

                # build_engine_from_onnx が呼ばれていないことを確認
                mock_build.assert_not_called()

    def test_infer_without_engine_path_builds(self) -> None:
        """engine_path=None 時に従来通り build_engine_from_onnx が呼ばれること（後方互換性）．"""
        trt_mock = _create_trt_mock()
        cuda_mocks = _create_cuda_mocks()
        cudart_mock = cuda_mocks["cuda.bindings.runtime"]

        # deserialize_cuda_engine が成功を返すように設定
        runtime_mock = MagicMock()
        runtime_mock.deserialize_cuda_engine.return_value = (
            MagicMock()
        )
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

            with patch.object(
                TensorRTInference,
                "build_engine_from_onnx",
                return_value=b"dummy_engine_bytes",
            ) as mock_build:
                try:
                    TensorRTInference.infer(
                        onnx_path=Path("/dummy/model.onnx"),
                        board_data=MagicMock(),
                        hand_data=MagicMock(),
                        num=5,
                        cuda_available=True,
                    )
                except Exception:
                    pass  # 推論処理はモックなので途中で例外は許容

                # build_engine_from_onnx が呼ばれたことを確認
                mock_build.assert_called_once()

    def test_infer_with_engine_path_does_not_require_onnx_path(
        self, tmp_path: Path
    ) -> None:
        """engine_path 指定時に onnx_path=None を明示的に渡してもエラーにならないこと．"""
        trt_mock = _create_trt_mock()
        cuda_mocks = _create_cuda_mocks()
        cudart_mock = cuda_mocks["cuda.bindings.runtime"]

        # deserialize_cuda_engine が成功を返すように設定
        runtime_mock = MagicMock()
        runtime_mock.deserialize_cuda_engine.return_value = (
            MagicMock()
        )
        trt_mock.Runtime.return_value = runtime_mock

        # cudaStreamCreate が成功を返すように設定
        cudart_mock.cudaStreamCreate.return_value = (
            MagicMock(),
            MagicMock(),
        )

        # エンジンファイルを作成
        engine_file = tmp_path / "test.engine"
        engine_file.write_bytes(b"dummy_engine_bytes")

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

            # onnx_path=None かつ engine_path 指定で ValueError が出ないことを確認
            # (推論処理自体はモックなので途中で例外が出ても問題ない)
            try:
                TensorRTInference.infer(
                    onnx_path=None,
                    board_data=MagicMock(),
                    hand_data=MagicMock(),
                    num=5,
                    cuda_available=True,
                    engine_path=engine_file,
                )
            except ValueError:
                pytest.fail(
                    "ValueError should not be raised when engine_path is specified with onnx_path=None"
                )
            except Exception:
                pass  # 推論処理中のその他の例外は許容
