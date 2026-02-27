import ctypes
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import tensorrt as trt
from cuda.bindings import driver as cuda
from cuda.bindings import runtime as cudart

logger: logging.Logger = logging.getLogger(__name__)


class TensorRTInference:
    @staticmethod
    def _convert_int64_to_int32(onnx_path: Path) -> bytes:
        """TensorRT互換性のためにONNXモデルのデータパスInt64をInt32に変換する．

        シェイプ計算パス(Shape/Size/NonZero出力，Constant，initializer等)は
        ONNX仕様通りInt64のまま維持する．
        FP16 fusionが問題になるデータ計算パス(モデル入出力，Cast(to=Int64))のみ
        Int32に変換する．

        Args:
            onnx_path: ONNXモデルファイルパス．

        Returns:
            変換後のONNXモデルのシリアライズバイト列．
        """
        import onnx
        from onnx import TensorProto

        model = onnx.load(str(onnx_path))
        converted = False

        # データパス: 入力テンソルの型変換
        for inp in model.graph.input:
            if (
                inp.type.tensor_type.elem_type
                == TensorProto.INT64
            ):
                inp.type.tensor_type.elem_type = (
                    TensorProto.INT32
                )
                logger.info(
                    "Converted input '%s' from INT64 to INT32",
                    inp.name,
                )
                converted = True

        # データパス: Cast(to=Int64)のターゲット型変換
        for node in model.graph.node:
            if node.op_type == "Cast":
                for attr in node.attribute:
                    if (
                        attr.name == "to"
                        and attr.i == TensorProto.INT64
                    ):
                        attr.i = TensorProto.INT32
                        converted = True

        # データパス: 出力テンソルの型変換
        for out in model.graph.output:
            if (
                out.type.tensor_type.elem_type
                == TensorProto.INT64
            ):
                out.type.tensor_type.elem_type = (
                    TensorProto.INT32
                )
                converted = True

        # シェイプ計算パスはInt64のまま維持:
        # - Shape/Size/NonZero出力: ONNX仕様でInt64固定
        # - Constant値テンソル/initializer: シェイプパラメータ
        # - value_int/value_ints属性: ONNX仕様でInt64固定
        # これらを変換するとConcat型不一致やSqueeze/Reshape/Expand等の
        # axes/shape入力で型エラーが連鎖するため，一切変換しない

        if converted:
            logger.info(
                "ONNX model Int64 -> Int32 conversion applied "
                "for TensorRT compatibility"
            )

        return model.SerializeToString()

    @staticmethod
    def save_engine(
        serialized_engine: bytes, path: Path
    ) -> None:
        """シリアライズ済みTensorRTエンジンをファイルに保存する．

        Args:
            serialized_engine: ビルド済みのシリアライズドエンジンバイト列．
            path: 保存先ファイルパス．
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(serialized_engine)
        logger.info(
            "TensorRT engine saved to: %s (%d MB)",
            path,
            len(serialized_engine) >> 20,
        )

    @staticmethod
    def load_engine(path: Path) -> bytes:
        """ファイルからシリアライズ済みTensorRTエンジンを読み込む．

        Args:
            path: エンジンファイルパス．

        Returns:
            シリアライズドエンジンバイト列．

        Raises:
            FileNotFoundError: ファイルが存在しない場合．
            RuntimeError: ファイル読み込みに失敗した場合．
        """
        if not path.exists():
            raise FileNotFoundError(
                f"TensorRT engine file not found: {path}"
            )
        logger.info("Loading TensorRT engine from: %s", path)
        data = path.read_bytes()
        if len(data) == 0:
            raise RuntimeError(
                f"TensorRT engine file is empty: {path}"
            )
        return data

    @staticmethod
    def build_engine_from_onnx(
        onnx_path: Path,
        workspace_size_mb: int = 256,
    ) -> bytes:
        """ONNXモデルから現在のGPUに適したTensorRTエンジンを生成する．

        Args:
            onnx_path: ONNXモデルファイルパス．
            workspace_size_mb: TensorRTワークスペースサイズ(MB)．

        Returns:
            シリアライズ済みエンジンバイト列．

        Raises:
            RuntimeError: ONNXパースまたはエンジンビルドに失敗した場合．
        """

        # builder
        trt_logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(trt_logger)

        # create network definition
        # EXPLICIT_BATCHは推奨らしいので設定しておく
        network_flags = 1 << int(
            trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH
        )
        network = builder.create_network(network_flags)

        # Int64→Int32変換してからパース(TensorRTはInt64サポートが限定的)
        onnx_bytes = TensorRTInference._convert_int64_to_int32(
            onnx_path
        )
        parser = trt.OnnxParser(network, trt_logger)
        success = parser.parse(onnx_bytes)
        for idx in range(parser.num_errors):
            logger.error(parser.get_error(idx))
        if not success:
            raise RuntimeError("ONNX parse failed")

        # build engine
        builder_config = builder.create_builder_config()
        builder_config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            workspace_size_mb * (1 << 20),
        )

        # FP16最適化
        if builder.platform_has_fast_fp16:
            builder_config.set_flag(trt.BuilderFlag.FP16)

        # ONNXではバッチサイズ可変なのでプロファイルを設定する
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):  # type: ignore[attr-defined]
            input_tensor = network.get_input(i)
            non_batch_dims = tuple(input_tensor.shape[1:])
            profile.set_shape(
                input_tensor.name,
                min=(1,) + non_batch_dims,
                opt=(4000,) + non_batch_dims,
                max=(10000,) + non_batch_dims,
            )
        builder_config.add_optimization_profile(profile)

        logger.info(
            "Building TensorRT engine for current GPU..."
        )
        serialized_engine = builder.build_serialized_network(
            network, builder_config
        )
        if serialized_engine is None:
            logger.error(
                "TensorRT engine build failed for: %s",
                onnx_path,
            )
            raise RuntimeError(
                "Failed to build TensorRT engine. "
                "Check ONNX model compatibility and available GPU memory."
            )
        logger.info("TensorRT engine built successfully")
        return serialized_engine

    @staticmethod
    def infer(
        onnx_path: Optional[Path],
        board_data: np.ndarray,
        hand_data: np.ndarray,
        num: int,
        cuda_available: bool,
        workspace_size_mb: int = 256,
        engine_path: Optional[Path] = None,
    ) -> tuple[list[int], float]:
        """TensorRTエンジンで推論を実行する．

        Args:
            onnx_path: ONNXモデルファイルパス．engine_path未指定時は必須．
            board_data: 盤面特徴量．
            hand_data: 持ち駒特徴量．
            num: 上位候補手数．
            cuda_available: CUDA利用可否．
            workspace_size_mb: TensorRTワークスペースサイズ(MB)．
            engine_path: ビルド済みエンジンファイルパス．指定時はONNXビルドをスキップ．

        Returns:
            上位ラベルリストと評価値のタプル．

        Raises:
            ValueError: CUDAが無効またはonnx_pathが未指定の場合．
        """
        if not cuda_available:
            raise ValueError("TensorRT requires CUDA.")

        if engine_path is not None:
            serialized_engine = TensorRTInference.load_engine(
                engine_path
            )
        else:
            if onnx_path is None:
                raise ValueError(
                    "onnx_path is required when engine_path is not specified."
                )
            serialized_engine = (
                TensorRTInference.build_engine_from_onnx(
                    onnx_path,
                    workspace_size_mb=workspace_size_mb,
                )
            )

        # TensorRTのサンプルコードを参考に実装した
        # https://github.com/NVIDIA/TensorRT/blob/main/samples/python/common_runtime.py
        # APIリファレンス
        # https://developer.nvidia.com/docs/drive/drive-os/6.0.9/public/drive-os-tensorrt/api-reference/docs/python/infer/Core/pyCore.html
        def cuda_malloc(
            engine: Any, name: str, batch_size: int
        ) -> tuple[np.ndarray, int, int, tuple[int, ...]]:
            shape = context.get_tensor_shape(name)
            actual_shape = (batch_size,) + tuple(shape[1:])
            actual_size = trt.volume(actual_shape)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            nbytes = actual_size * np.dtype(dtype).itemsize
            err, host_mem = cudart.cudaMallocHost(nbytes)
            if isinstance(err, cuda.CUresult):
                if err != cuda.CUresult.CUDA_SUCCESS:
                    raise RuntimeError(
                        "Cuda Error: {}".format(err)
                    )
                if isinstance(err, cudart.cudaError_t):
                    if err != cudart.cudaError_t.cudaSuccess:
                        raise RuntimeError(
                            "Cuda Runtime Error: {}".format(err)
                        )
                    else:
                        raise RuntimeError(
                            "Unknown error type: {}".format(err)
                        )
            pointer_type = ctypes.POINTER(
                np.ctypeslib.as_ctypes_type(dtype)
            )
            host_ctype_array = np.ctypeslib.as_array(
                ctypes.cast(host_mem, pointer_type),
                (actual_size,),
            )
            err, cuda_mem = cudart.cudaMalloc(nbytes)
            if isinstance(err, cuda.CUresult):
                if err != cuda.CUresult.CUDA_SUCCESS:
                    raise RuntimeError(
                        "Cuda Error: {}".format(err)
                    )
                if isinstance(err, cudart.cudaError_t):
                    if err != cudart.cudaError_t.cudaSuccess:
                        raise RuntimeError(
                            "Cuda Runtime Error: {}".format(err)
                        )
                    else:
                        raise RuntimeError(
                            "Unknown error type: {}".format(err)
                        )
            return (
                host_ctype_array,
                cuda_mem,
                nbytes,
                actual_shape,
            )

        batch_size = 1
        err, stream = cudart.cudaStreamCreate()
        if isinstance(err, cuda.CUresult):
            if err != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError("Cuda Error: {}".format(err))
            if isinstance(err, cudart.cudaError_t):
                if err != cudart.cudaError_t.cudaSuccess:
                    raise RuntimeError(
                        "Cuda Runtime Error: {}".format(err)
                    )
                else:
                    raise RuntimeError(
                        "Unknown error type: {}".format(err)
                    )
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        engine = runtime.deserialize_cuda_engine(
            serialized_engine
        )
        if engine is None:
            logger.error(
                "Failed to deserialize TensorRT engine"
            )
            raise RuntimeError(
                "Failed to deserialize TensorRT engine. "
                "The serialized engine may be corrupted or incompatible."
            )
        context = engine.create_execution_context()

        # 入力テンソル: board
        board_name = engine.get_tensor_name(0)
        (
            host_board_ctype_array,
            cuda_board,
            board_nbytes,
            board_shape,
        ) = cuda_malloc(engine, board_name, batch_size)
        context.set_tensor_address(board_name, cuda_board)
        context.set_input_shape(board_name, board_shape)

        # 入力テンソル: hand
        hand_name = engine.get_tensor_name(1)
        (
            host_hand_ctype_array,
            cuda_hand,
            hand_nbytes,
            hand_shape,
        ) = cuda_malloc(engine, hand_name, batch_size)
        context.set_tensor_address(hand_name, cuda_hand)
        context.set_input_shape(hand_name, hand_shape)

        output_policy_name = engine.get_tensor_name(2)
        (
            host_output_policy_ctype_array,
            cuda_output_policy,
            output_policy_nbytes,
            _,
        ) = cuda_malloc(engine, output_policy_name, batch_size)
        context.set_tensor_address(
            output_policy_name, cuda_output_policy
        )

        output_value_name = engine.get_tensor_name(3)
        (
            host_output_value_ctype_array,
            cuda_output_value,
            output_value_nbytes,
            _,
        ) = cuda_malloc(engine, output_value_name, batch_size)
        context.set_tensor_address(
            output_value_name, cuda_output_value
        )
        # boardの転送 host -> gpu
        batched_board = np.expand_dims(board_data, axis=0)
        np.copyto(
            host_board_ctype_array,
            batched_board.astype(np.int32).ravel(),
            casting="safe",
        )
        cudart.cudaMemcpyAsync(
            cuda_board,
            host_board_ctype_array,
            board_nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            stream,
        )
        # handの転送 host -> gpu
        batched_hand = np.expand_dims(hand_data, axis=0)
        np.copyto(
            host_hand_ctype_array,
            batched_hand.astype(np.float32).ravel(),
            casting="safe",
        )
        cudart.cudaMemcpyAsync(
            cuda_hand,
            host_hand_ctype_array,
            hand_nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            stream,
        )
        # 推論
        # context.execute_async_v3(stream_handle=stream)
        context.execute_async_v3(stream_handle=stream)
        # outputの転送 gpu -> host
        # cudart.cudaMemcpyAsync(out.host, out.device, out.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
        cudart.cudaMemcpyAsync(
            host_output_policy_ctype_array,
            cuda_output_policy,
            output_policy_nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            stream,
        )
        cudart.cudaMemcpyAsync(
            host_output_value_ctype_array,
            cuda_output_value,
            output_value_nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            stream,
        )
        # 同期
        # cudart.cudaStreamSynchronize(stream)
        cudart.cudaStreamSynchronize(stream)
        # hostの出力を確認する
        policy_labels: list[int] = list(
            np.argsort(host_output_policy_ctype_array)[::-1][
                :num
            ]  # type: ignore
        )
        value: float = host_output_value_ctype_array[0].item()  # type: ignore

        # メモリ解放
        cudart.cudaFree(cuda_board)
        cudart.cudaFreeHost(host_board_ctype_array.ctypes.data)
        cudart.cudaFree(cuda_hand)
        cudart.cudaFreeHost(host_hand_ctype_array.ctypes.data)
        cudart.cudaFree(cuda_output_policy)
        cudart.cudaFreeHost(
            host_output_policy_ctype_array.ctypes.data
        )
        cudart.cudaFree(cuda_output_value)
        cudart.cudaFreeHost(
            host_output_value_ctype_array.ctypes.data
        )
        cudart.cudaStreamDestroy(stream)

        return policy_labels, value
