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
        parser_mock.parse.return_value = True
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

            with (
                patch.object(
                    TensorRTInference,
                    "_convert_int64_to_int32",
                    return_value=b"dummy_onnx_bytes",
                ),
                pytest.raises(
                    RuntimeError,
                    match="Failed to build TensorRT engine",
                ),
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
        parser_mock.parse.return_value = True
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
                    TensorRTInference,
                    "_convert_int64_to_int32",
                    return_value=b"dummy_onnx_bytes",
                ),
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
        parser_mock.parse.return_value = True
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

            with patch.object(
                TensorRTInference,
                "_convert_int64_to_int32",
                return_value=b"dummy_onnx_bytes",
            ):
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


def _create_onnx_model_with_int64(
    tmp_path: Path,
) -> Path:
    """Int64入力とCastノードを含むダミーONNXモデルを生成する．"""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    # Int64入力
    board_input = helper.make_tensor_value_info(
        "board", TensorProto.INT64, [1, 9, 9]
    )
    # Float32入力
    hand_input = helper.make_tensor_value_info(
        "hand", TensorProto.FLOAT, [1, 14]
    )
    # Float32出力
    output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 1]
    )

    # Cast node: ターゲット型をINT64に設定
    cast_node = helper.make_node(
        "Cast",
        inputs=["board"],
        outputs=["board_float"],
        to=TensorProto.INT64,
    )
    # Reshape for simplicity
    shape_init = numpy_helper.from_array(
        np.array([1, 81], dtype=np.int64), name="shape"
    )
    reshape_node = helper.make_node(
        "Reshape",
        inputs=["board_float", "shape"],
        outputs=["board_flat"],
    )
    # Simple matmul to produce output
    weight_init = numpy_helper.from_array(
        np.ones((81, 1), dtype=np.float32), name="weight"
    )
    matmul_node = helper.make_node(
        "MatMul",
        inputs=["board_flat", "weight"],
        outputs=["output"],
    )

    graph = helper.make_graph(
        [cast_node, reshape_node, matmul_node],
        "test_graph",
        [board_input, hand_input],
        [output],
        initializer=[shape_init, weight_init],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 20)]
    )

    model_path = tmp_path / "test_int64.onnx"
    onnx.save(model, str(model_path))
    return model_path


def _create_onnx_model_with_int32(
    tmp_path: Path,
) -> Path:
    """Int32入力のみのダミーONNXモデルを生成する．"""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    board_input = helper.make_tensor_value_info(
        "board", TensorProto.INT32, [1, 9, 9]
    )
    output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 1]
    )

    cast_node = helper.make_node(
        "Cast",
        inputs=["board"],
        outputs=["board_float"],
        to=TensorProto.FLOAT,
    )
    shape_init = numpy_helper.from_array(
        np.array([1, 81], dtype=np.int32), name="shape"
    )
    reshape_node = helper.make_node(
        "Reshape",
        inputs=["board_float", "shape"],
        outputs=["board_flat"],
    )
    weight_init = numpy_helper.from_array(
        np.ones((81, 1), dtype=np.float32), name="weight"
    )
    matmul_node = helper.make_node(
        "MatMul",
        inputs=["board_flat", "weight"],
        outputs=["output"],
    )

    graph = helper.make_graph(
        [cast_node, reshape_node, matmul_node],
        "test_graph",
        [board_input],
        [output],
        initializer=[shape_init, weight_init],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 20)]
    )

    model_path = tmp_path / "test_int32.onnx"
    onnx.save(model, str(model_path))
    return model_path


def _create_onnx_model_with_constant_int64(
    tmp_path: Path,
) -> Path:
    """Int64のConstantノードを含むダミーONNXモデルを生成する．

    Rangeノードのstart/limit/deltaがConstantノード経由で
    Int64値を持つケースを再現する．
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    # Float32入力
    x_input = helper.make_tensor_value_info(
        "x", TensorProto.FLOAT, [None, 81]
    )
    # Float32出力
    output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [None, 81]
    )

    # Rangeノードの入力をConstantノードで定義(Int64)
    start_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["range_start"],
        value=numpy_helper.from_array(
            np.array(0, dtype=np.int64), name="start_val"
        ),
    )
    limit_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["range_limit"],
        value=numpy_helper.from_array(
            np.array(81, dtype=np.int64), name="limit_val"
        ),
    )
    delta_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["range_delta"],
        value=numpy_helper.from_array(
            np.array(1, dtype=np.int64), name="delta_val"
        ),
    )

    # Rangeノード: start, limit, deltaすべてInt64
    range_node = helper.make_node(
        "Range",
        inputs=["range_start", "range_limit", "range_delta"],
        outputs=["range_out"],
    )

    # Castで Range出力(Int) → Float に変換
    cast_node = helper.make_node(
        "Cast",
        inputs=["range_out"],
        outputs=["range_float"],
        to=TensorProto.FLOAT,
    )

    # Reshape用shape
    shape_init = numpy_helper.from_array(
        np.array([1, 81], dtype=np.int64), name="shape"
    )
    reshape_node = helper.make_node(
        "Reshape",
        inputs=["range_float", "shape"],
        outputs=["range_reshaped"],
    )

    # Add: x + range_reshaped
    add_node = helper.make_node(
        "Add",
        inputs=["x", "range_reshaped"],
        outputs=["output"],
    )

    graph = helper.make_graph(
        [
            start_node,
            limit_node,
            delta_node,
            range_node,
            cast_node,
            reshape_node,
            add_node,
        ],
        "test_constant_int64_graph",
        [x_input],
        [output],
        initializer=[shape_init],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 20)]
    )

    model_path = tmp_path / "test_constant_int64.onnx"
    onnx.save(model, str(model_path))
    return model_path


def _create_onnx_model_with_shape_concat(
    tmp_path: Path,
) -> Path:
    """Shape由来Int64テンソルとConstantが混在するConcatを含むダミーONNXモデル．

    Shape→Gather→Unsqueeze(Int64)とConstant(Int64)がConcatされる
    パターンを再現する．変換前はConcatの入力型が混在してTensorRTでエラーになる．
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    # Float32入力 [batch, 32, 9, 9]
    x_input = helper.make_tensor_value_info(
        "x", TensorProto.FLOAT, [None, 32, 9, 9]
    )
    output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [None, 2592]
    )

    # Shape: x の形状を取得 → Int64 [4]
    shape_node = helper.make_node(
        "Shape",
        inputs=["x"],
        outputs=["x_shape"],
    )

    # Gather: batch_size を抽出(index=0) → scalar Int64
    gather_idx = numpy_helper.from_array(
        np.array(0, dtype=np.int64), name="gather_idx"
    )
    gather_node = helper.make_node(
        "Gather",
        inputs=["x_shape", "gather_idx"],
        outputs=["batch_size"],
        axis=0,
    )

    # Unsqueeze: scalar → [1] (Int64)
    unsqueeze_axes = numpy_helper.from_array(
        np.array([0], dtype=np.int64),
        name="unsqueeze_axes",
    )
    unsqueeze_node = helper.make_node(
        "Unsqueeze",
        inputs=["batch_size", "unsqueeze_axes"],
        outputs=["batch_1d"],
    )

    # Constant: reshape の残り次元 [2592] (Int64)
    other_dim_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["other_dim"],
        value=numpy_helper.from_array(
            np.array([2592], dtype=np.int64),
            name="other_dim_val",
        ),
    )

    # Concat: [batch_size, 2592] → Reshape 用形状テンソル
    concat_node = helper.make_node(
        "Concat",
        inputs=["batch_1d", "other_dim"],
        outputs=["new_shape"],
        axis=0,
    )

    # Reshape: [batch, 32, 9, 9] → [batch, 2592]
    reshape_node = helper.make_node(
        "Reshape",
        inputs=["x", "new_shape"],
        outputs=["output"],
    )

    graph = helper.make_graph(
        [
            shape_node,
            gather_node,
            unsqueeze_node,
            other_dim_node,
            concat_node,
            reshape_node,
        ],
        "test_shape_concat_graph",
        [x_input],
        [output],
        initializer=[gather_idx, unsqueeze_axes],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 20)]
    )

    model_path = tmp_path / "test_shape_concat.onnx"
    onnx.save(model, str(model_path))
    return model_path


def _create_onnx_model_with_shape_expand(
    tmp_path: Path,
) -> Path:
    """Shape出力がExpandのshape入力に直接接続されるダミーONNXモデル．

    Shape→Expandパターンを再現する．
    TensorRTはExpand内部でInt64のshapeテンソルを生成するため，
    Shape出力をInt32に変換すると型不一致エラーになる．
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    # Float32入力: hand projection 出力 [batch, 32]
    hand_input = helper.make_tensor_value_info(
        "hand", TensorProto.FLOAT, [None, 32]
    )
    # Float32入力: board embedding [batch, 64, 9, 9]
    board_input = helper.make_tensor_value_info(
        "board", TensorProto.FLOAT, [None, 64, 9, 9]
    )
    output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [None, 32, 9, 9]
    )

    # Reshape: hand [batch, 32] → [batch, 32, 1, 1]
    reshape_shape = numpy_helper.from_array(
        np.array([0, 32, 1, 1], dtype=np.int64),
        name="reshape_shape",
    )
    reshape_node = helper.make_node(
        "Reshape",
        inputs=["hand", "reshape_shape"],
        outputs=["hand_4d"],
    )

    # Shape: board の形状を取得 → Int64 [4] (e.g. [batch, 64, 9, 9])
    shape_node = helper.make_node(
        "Shape",
        inputs=["board"],
        outputs=["board_shape"],
    )

    # Expand: hand_4d を board_shape にブロードキャスト
    expand_node = helper.make_node(
        "Expand",
        inputs=["hand_4d", "board_shape"],
        outputs=["output"],
    )

    graph = helper.make_graph(
        [reshape_node, shape_node, expand_node],
        "test_shape_expand_graph",
        [hand_input, board_input],
        [output],
        initializer=[reshape_shape],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 20)]
    )

    model_path = tmp_path / "test_shape_expand.onnx"
    onnx.save(model, str(model_path))
    return model_path


def _create_onnx_model_with_where_expand(
    tmp_path: Path,
) -> Path:
    """Shape→Where→Expandパターンを含むダミーONNXモデル．

    PyTorchのexpand(-1, -1, h, w)エクスポートで生成される
    Shape→Where→Expandパターンを再現する．
    中間のWhereノードを経由するため，直接のShape→Expand接続よりも
    複雑なshape計算サブグラフとなる．
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    # hand: [batch, 32, 1, 1] (ブロードキャスト元)
    hand_input = helper.make_tensor_value_info(
        "hand", TensorProto.FLOAT, [None, 32, 1, 1]
    )
    # board: [batch, 64, 9, 9] (形状の参照元)
    board_input = helper.make_tensor_value_info(
        "board", TensorProto.FLOAT, [None, 64, 9, 9]
    )
    output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [None, 32, 9, 9]
    )

    # Shape: board の形状を取得 → Int64 [4]
    shape_node = helper.make_node(
        "Shape",
        inputs=["board"],
        outputs=["board_shape"],
    )

    # Constant: ターゲット形状 [-1, 32, -1, -1] (Int64)
    target_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["target_shape"],
        value=numpy_helper.from_array(
            np.array([-1, 32, -1, -1], dtype=np.int64),
            name="target_val",
        ),
    )

    # Constant: -1 (比較用)
    neg_one_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["neg_one"],
        value=numpy_helper.from_array(
            np.array(-1, dtype=np.int64),
            name="neg_one_val",
        ),
    )

    # Equal: target_shape == -1 → mask
    equal_node = helper.make_node(
        "Equal",
        inputs=["target_shape", "neg_one"],
        outputs=["mask"],
    )

    # Where: -1の要素をboard_shapeで置換
    where_node = helper.make_node(
        "Where",
        inputs=["mask", "board_shape", "target_shape"],
        outputs=["resolved_shape"],
    )

    # Expand: hand を resolved_shape にブロードキャスト
    expand_node = helper.make_node(
        "Expand",
        inputs=["hand", "resolved_shape"],
        outputs=["output"],
    )

    graph = helper.make_graph(
        [
            shape_node,
            target_node,
            neg_one_node,
            equal_node,
            where_node,
            expand_node,
        ],
        "test_where_expand_graph",
        [hand_input, board_input],
        [output],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 20)]
    )

    model_path = tmp_path / "test_where_expand.onnx"
    onnx.save(model, str(model_path))
    return model_path


def _create_onnx_model_with_value_int_attr(
    tmp_path: Path,
) -> Path:
    """value_int属性を持つConstantノードを含むダミーONNXモデル．"""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    x_input = helper.make_tensor_value_info(
        "x", TensorProto.FLOAT, [None, 81]
    )
    output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [None, 81]
    )

    # Constant with value_int (ONNX仕様でInt64固定)
    const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["const_scalar"],
        value_int=42,
    )

    # Cast to float
    cast_node = helper.make_node(
        "Cast",
        inputs=["const_scalar"],
        outputs=["const_float"],
        to=TensorProto.FLOAT,
    )

    # Reshape for broadcast
    shape_init = numpy_helper.from_array(
        np.array([1, 1], dtype=np.int64), name="shape"
    )
    reshape_node = helper.make_node(
        "Reshape",
        inputs=["const_float", "shape"],
        outputs=["const_2d"],
    )

    # Add: x + const
    add_node = helper.make_node(
        "Add",
        inputs=["x", "const_2d"],
        outputs=["output"],
    )

    graph = helper.make_graph(
        [const_node, cast_node, reshape_node, add_node],
        "test_value_int_graph",
        [x_input],
        [output],
        initializer=[shape_init],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 20)]
    )

    model_path = tmp_path / "test_value_int.onnx"
    onnx.save(model, str(model_path))
    return model_path


def _create_onnx_model_with_squeeze_mul(
    tmp_path: Path,
) -> Path:
    """Squeeze+Mulパターンを含むダミーONNXモデル．

    PyTorchのscaled_dot_product_attentionが生成する
    Split→Squeeze(unbind) + Shape→Gather→Sqrt(スケール計算) → Mul
    パターンを再現する．
    value_ints属性のSqueeze axesがInt32に不正変換されると
    Mulのブロードキャスト次元が不整合になるバグの回帰テスト．
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    # 入力: QKV結合テンソル [3, batch, num_heads, seq_len, head_dim]
    qkv_input = helper.make_tensor_value_info(
        "qkv", TensorProto.FLOAT, [3, None, 8, 81, 64]
    )
    output = helper.make_tensor_value_info(
        "output",
        TensorProto.FLOAT,
        [None, 8, 81, 64],
    )

    # Split: QKV を Q, K, V に分割 → 3つの [1, batch, 8, 81, 64]
    split_node = helper.make_node(
        "Split",
        inputs=["qkv"],
        outputs=["q_raw", "k_raw", "v_raw"],
        axis=0,
    )

    # Squeeze(axes=[0]): [1, batch, 8, 81, 64] → [batch, 8, 81, 64]
    # axes は value_ints 属性(ONNX仕様でInt64固定)
    squeeze_axes_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["squeeze_axes"],
        value_ints=[0],
    )
    squeeze_node = helper.make_node(
        "Squeeze",
        inputs=["q_raw", "squeeze_axes"],
        outputs=["q"],
    )

    # スケール計算: Shape(Q)→Gather(last_dim)→Cast→Sqrt
    shape_node = helper.make_node(
        "Shape",
        inputs=["q"],
        outputs=["q_shape"],
    )
    gather_idx = numpy_helper.from_array(
        np.array(-1, dtype=np.int64),
        name="gather_idx",
    )
    gather_node = helper.make_node(
        "Gather",
        inputs=["q_shape", "gather_idx"],
        outputs=["head_dim"],
        axis=0,
    )
    cast_to_float = helper.make_node(
        "Cast",
        inputs=["head_dim"],
        outputs=["head_dim_float"],
        to=TensorProto.FLOAT,
    )
    sqrt_node = helper.make_node(
        "Sqrt",
        inputs=["head_dim_float"],
        outputs=["sqrt_d_k"],
    )

    # Mul: Q * sqrt(d_k) (ブロードキャスト: [batch,8,81,64] * scalar)
    mul_node = helper.make_node(
        "Mul",
        inputs=["q", "sqrt_d_k"],
        outputs=["output"],
    )

    graph = helper.make_graph(
        [
            split_node,
            squeeze_axes_node,
            squeeze_node,
            shape_node,
            gather_node,
            cast_to_float,
            sqrt_node,
            mul_node,
        ],
        "test_squeeze_mul_graph",
        [qkv_input],
        [output],
        initializer=[gather_idx],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 20)]
    )

    model_path = tmp_path / "test_squeeze_mul.onnx"
    onnx.save(model, str(model_path))
    return model_path


class TestConvertInt64ToInt32:
    """_convert_int64_to_int32 のテスト．

    データ計算パス(入出力/Cast)のみInt32に変換し，
    シェイプ計算パス(Constant/initializer/Shape出力)はInt64のまま維持する．
    """

    def test_converts_int64_input_to_int32(
        self, tmp_path: Path
    ) -> None:
        """Int64入力がInt32に変換されること．"""
        onnx = pytest.importorskip("onnx")
        from onnx import TensorProto

        trt_mock = _create_trt_mock()
        cuda_mocks = _create_cuda_mocks()
        model_path = _create_onnx_model_with_int64(tmp_path)

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

            result_bytes = (
                TensorRTInference._convert_int64_to_int32(
                    model_path
                )
            )
            result_model = onnx.load_from_string(result_bytes)

            board_input = result_model.graph.input[0]
            assert (
                board_input.type.tensor_type.elem_type
                == TensorProto.INT32
            )

    def test_converts_cast_node_target_to_int32(
        self, tmp_path: Path
    ) -> None:
        """CastノードのターゲットInt64がInt32に変換されること．"""
        onnx = pytest.importorskip("onnx")
        from onnx import TensorProto

        trt_mock = _create_trt_mock()
        cuda_mocks = _create_cuda_mocks()
        model_path = _create_onnx_model_with_int64(tmp_path)

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

            result_bytes = (
                TensorRTInference._convert_int64_to_int32(
                    model_path
                )
            )
            result_model = onnx.load_from_string(result_bytes)

            cast_nodes = [
                n
                for n in result_model.graph.node
                if n.op_type == "Cast"
            ]
            for node in cast_nodes:
                for attr in node.attribute:
                    if attr.name == "to":
                        assert attr.i != TensorProto.INT64

    def test_no_change_for_int32_model(
        self, tmp_path: Path
    ) -> None:
        """Int64を含まないモデルに変更が加わらないこと．"""
        onnx = pytest.importorskip("onnx")

        trt_mock = _create_trt_mock()
        cuda_mocks = _create_cuda_mocks()
        model_path = _create_onnx_model_with_int32(tmp_path)

        original_model = onnx.load(str(model_path))

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

            result_bytes = (
                TensorRTInference._convert_int64_to_int32(
                    model_path
                )
            )
            result_model = onnx.load_from_string(result_bytes)

            assert (
                result_model.graph.input[
                    0
                ].type.tensor_type.elem_type
                == original_model.graph.input[
                    0
                ].type.tensor_type.elem_type
            )

    def test_preserves_int64_initializers(
        self, tmp_path: Path
    ) -> None:
        """Int64初期化テンソルがシェイプパスとして保持されること．"""
        onnx = pytest.importorskip("onnx")
        from onnx import TensorProto

        trt_mock = _create_trt_mock()
        cuda_mocks = _create_cuda_mocks()
        model_path = _create_onnx_model_with_int64(tmp_path)

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

            result_bytes = (
                TensorRTInference._convert_int64_to_int32(
                    model_path
                )
            )
            result_model = onnx.load_from_string(result_bytes)

            # Int64初期化テンソルが変換されずに残ること
            int64_inits = [
                init
                for init in result_model.graph.initializer
                if init.data_type == TensorProto.INT64
            ]
            assert len(int64_inits) >= 1

    def test_preserves_int64_constant_values(
        self, tmp_path: Path
    ) -> None:
        """ConstantノードのInt64値がシェイプパスとして保持されること．"""
        onnx = pytest.importorskip("onnx")
        from onnx import TensorProto

        trt_mock = _create_trt_mock()
        cuda_mocks = _create_cuda_mocks()
        model_path = _create_onnx_model_with_constant_int64(
            tmp_path
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

            result_bytes = (
                TensorRTInference._convert_int64_to_int32(
                    model_path
                )
            )
            result_model = onnx.load_from_string(result_bytes)

            # ConstantノードのInt64値が変換されずに残ること
            int64_constants = [
                n
                for n in result_model.graph.node
                if n.op_type == "Constant"
                and any(
                    a.name == "value"
                    and a.t.data_type == TensorProto.INT64
                    for a in n.attribute
                )
            ]
            assert len(int64_constants) >= 1

    def test_preserves_shape_path_int64(
        self, tmp_path: Path
    ) -> None:
        """Shape計算パス全体がInt64のまま維持されること．

        Shape→Gather→Unsqueeze→Concat→Reshape のパスで
        Cast(Int64→Int32)が挿入されないこと．
        """
        onnx = pytest.importorskip("onnx")
        from onnx import TensorProto

        trt_mock = _create_trt_mock()
        cuda_mocks = _create_cuda_mocks()
        model_path = _create_onnx_model_with_shape_concat(
            tmp_path
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

            result_bytes = (
                TensorRTInference._convert_int64_to_int32(
                    model_path
                )
            )
            result_model = onnx.load_from_string(result_bytes)

            # Shape出力後にCast(to=INT32)が挿入されていないこと
            shape_indices = [
                i
                for i, n in enumerate(result_model.graph.node)
                if n.op_type == "Shape"
            ]
            assert len(shape_indices) == 1
            next_node = result_model.graph.node[
                shape_indices[0] + 1
            ]
            # Shape直後がCast(to=INT32)でないこと
            is_cast_to_i32 = (
                next_node.op_type == "Cast"
                and any(
                    a.name == "to" and a.i == TensorProto.INT32
                    for a in next_node.attribute
                )
            )
            assert not is_cast_to_i32

            # ConstantのInt64値も変換されずに残ること
            int64_constants = [
                n
                for n in result_model.graph.node
                if n.op_type == "Constant"
                and any(
                    a.name == "value"
                    and a.t.data_type == TensorProto.INT64
                    for a in n.attribute
                )
            ]
            assert len(int64_constants) >= 1

    def test_preserves_value_int_attrs(
        self, tmp_path: Path
    ) -> None:
        """Constant(value_int)がInt64属性のまま保持されること．"""
        onnx = pytest.importorskip("onnx")

        trt_mock = _create_trt_mock()
        cuda_mocks = _create_cuda_mocks()
        model_path = _create_onnx_model_with_value_int_attr(
            tmp_path
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

            result_bytes = (
                TensorRTInference._convert_int64_to_int32(
                    model_path
                )
            )
            result_model = onnx.load_from_string(result_bytes)

            # value_int属性がそのまま残っていること
            const_with_value_int = [
                n
                for n in result_model.graph.node
                if n.op_type == "Constant"
                and any(
                    a.name == "value_int" for a in n.attribute
                )
            ]
            assert len(const_with_value_int) >= 1

    def test_squeeze_mul_no_shape_modification(
        self, tmp_path: Path
    ) -> None:
        """Squeeze+Mulパターンでシェイプパスが変更されないこと．

        回帰テスト: Squeeze axesやShape出力がInt32に不正変換されると
        TensorRTでMulノードの shape error が発生する問題の防止．
        """
        onnx = pytest.importorskip("onnx")

        trt_mock = _create_trt_mock()
        cuda_mocks = _create_cuda_mocks()
        model_path = _create_onnx_model_with_squeeze_mul(
            tmp_path
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

            result_bytes = (
                TensorRTInference._convert_int64_to_int32(
                    model_path
                )
            )
            result_model = onnx.load_from_string(result_bytes)

            # Squeeze の axes 入力が変更されていないこと
            # (Cast挿入やリネームが行われていないこと)
            squeeze_nodes = [
                n
                for n in result_model.graph.node
                if n.op_type == "Squeeze"
            ]
            assert len(squeeze_nodes) == 1
            assert squeeze_nodes[0].input[1] == "squeeze_axes"

            # Constant(value_ints)が属性のまま残っていること
            const_nodes = [
                n
                for n in result_model.graph.node
                if n.op_type == "Constant"
                and any(
                    a.name == "value_ints" for a in n.attribute
                )
            ]
            assert len(const_nodes) >= 1

            # Shape出力にCast(to=INT32)が挿入されていないこと
            shape_nodes = [
                n
                for n in result_model.graph.node
                if n.op_type == "Shape"
            ]
            assert len(shape_nodes) == 1
            assert shape_nodes[0].output[0] == "q_shape"

            # ノード数が変わっていないこと(Cast挿入なし)
            original_model = onnx.load(str(model_path))
            assert len(result_model.graph.node) == len(
                original_model.graph.node
            )

    def test_expand_shape_path_preserved(
        self, tmp_path: Path
    ) -> None:
        """ExpandのshapeパスがInt64のまま維持されること．"""
        onnx = pytest.importorskip("onnx")
        from onnx import TensorProto

        trt_mock = _create_trt_mock()
        cuda_mocks = _create_cuda_mocks()
        model_path = _create_onnx_model_with_shape_expand(
            tmp_path
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

            result_bytes = (
                TensorRTInference._convert_int64_to_int32(
                    model_path
                )
            )
            result_model = onnx.load_from_string(result_bytes)

            # Expand の shape 入力が元のまま(board_shape)であること
            expand_nodes = [
                n
                for n in result_model.graph.node
                if n.op_type == "Expand"
            ]
            assert len(expand_nodes) == 1
            assert expand_nodes[0].input[1] == "board_shape"

            # Shape出力にCastが挿入されていないこと
            shape_nodes = [
                n
                for n in result_model.graph.node
                if n.op_type == "Shape"
            ]
            assert len(shape_nodes) == 1
            assert shape_nodes[0].output[0] == "board_shape"

            # Int64初期化テンソルが保持されること
            int64_inits = [
                init
                for init in result_model.graph.initializer
                if init.data_type == TensorProto.INT64
            ]
            assert len(int64_inits) >= 1
