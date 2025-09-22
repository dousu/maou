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
        input_tensor = network.get_input(0)
        t, h, w = input_tensor.shape[1:]
        profile.set_shape(
            input_tensor.name,
            min=(1, t, h, w),
            opt=(4000, t, h, w),
            max=(10000, t, h, w),
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
        input_data: np.ndarray,
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

        input_name = engine.get_tensor_name(0)
        (
            host_input_ctype_array,
            cuda_input,
            input_nbytes,
            input_shape,
        ) = cuda_malloc(engine, input_name, batch_size)
        context.set_tensor_address(input_name, cuda_input)
        # EXPLICIT_BATCHにしているのでshapeを渡す必要がある
        context.set_input_shape(input_name, input_shape)

        output_policy_name = engine.get_tensor_name(1)
        (
            host_output_policy_ctype_array,
            cuda_output_policy,
            output_policy_nbytes,
            _,
        ) = cuda_malloc(engine, output_policy_name, batch_size)
        context.set_tensor_address(
            output_policy_name, cuda_output_policy
        )

        output_value_name = engine.get_tensor_name(2)
        (
            host_output_value_ctype_array,
            cuda_output_value,
            output_value_nbytes,
            _,
        ) = cuda_malloc(engine, output_value_name, batch_size)
        context.set_tensor_address(
            output_value_name, cuda_output_value
        )
        # inputの転送 host -> gpu
        # np.copyto(self.host[:data.size], data.flat, casting='safe')
        # cudart.cudaMemcpyAsync(inp.device, inp.host, inp.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        np.copyto(
            host_input_ctype_array,
            input_data.astype(trt.nptype(trt.float32)).ravel(),
            casting="safe",
        )
        cudart.cudaMemcpyAsync(
            cuda_input,
            host_input_ctype_array,
            input_nbytes,
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

        # データの使用が完了したので，メモリを解放する
        cudart.cudaFree(cuda_input)
        cudart.cudaFreeHost(host_input_ctype_array.ctypes.data)
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
