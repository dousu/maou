import ctypes
import logging
from pathlib import Path
from typing import Any

import numpy as np
import tensorrt as trt
from cuda.bindings import driver as cuda
from cuda.bindings import runtime as cudart

logger: logging.Logger = logging.getLogger(__name__)


class TensorRTInference:
    @staticmethod
    def _build_engine_from_onnx(
        onnx_path: Path,
    ) -> bytes:
        """ONNXモデルから現在のGPUに適したTensorRTエンジンを生成"""

        # builder
        trt_logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(trt_logger)

        # create network definition
        # EXPLICIT_BATCHは推奨らしいので設定しておく
        network_flags = 1 << int(
            trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH
        )
        network = builder.create_network(network_flags)

        # import model using the ONNX parser
        parser = trt.OnnxParser(network, trt_logger)
        success = parser.parse_from_file(
            str(onnx_path.absolute())
        )
        for idx in range(parser.num_errors):
            logger.error(parser.get_error(idx))
        if not success:
            raise RuntimeError("ONNX parse failed")

        # build engine
        builder_config = builder.create_builder_config()
        builder_config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, 1 << 30
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
        logger.info("TensorRT engine built successfully")
        return serialized_engine

    @staticmethod
    def infer(
        onnx_path: Path,
        board_data: np.ndarray,
        hand_data: np.ndarray,
        num: int,
        cuda_available: bool,
    ) -> tuple[list[int], float]:
        if not cuda_available:
            raise ValueError("TensorRT requires CUDA.")

        serialized_engine = (
            TensorRTInference._build_engine_from_onnx(onnx_path)
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
            batched_board.astype(np.int64).ravel(),
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
