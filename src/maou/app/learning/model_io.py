import logging
from pathlib import Path
from typing import Optional

import onnx
import onnxsim
import torch
from onnxruntime.transformers import float16

try:
    import tensorrt as trt

    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False


from maou.app.learning.network import Network
from maou.domain.cloud_storage import CloudStorage
from maou.domain.data.schema import (
    create_empty_preprocessing_array,
)

logger: logging.Logger = logging.getLogger(__name__)


class ModelIO:
    @staticmethod
    def save_model(
        *,
        model: Network,
        dir: Path,
        id: str,
        epoch: int,
        device: torch.device,
        cloud_storage: Optional[CloudStorage] = None,
    ) -> None:
        # Training modeを確実に解除しておく
        model.train(False)
        model_path = dir / "model_{}_{}.pt".format(id, epoch)
        logger.info("Saving model to {}".format(model_path))
        torch.save(model.state_dict(), model_path)
        if cloud_storage is not None:
            logger.info("Uploading model to cloud storage")
            cloud_storage.upload_from_local(
                local_path=model_path,
                cloud_path=str(model_path),
            )
        # AMPのような高速化をしたいので一部FP16にする
        # TensorRTに変換するときはONNXのFP32を利用してBuilderFlag.FP16を指定する

        # torch.onnx.export
        onnx_model_path = model_path.with_suffix(".onnx")
        dummy_data = create_empty_preprocessing_array(1)
        dummy_input = (
            torch.from_numpy(dummy_data["features"].copy())
            .to(torch.float32)
            .to(device)
        )
        torch.onnx.export(
            model=model,
            args=(dummy_input,),
            f=onnx_model_path,
            export_params=True,
            input_names=["input"],
            output_names=["policy", "value"],
            opset_version=20,
            dynamic_axes={
                "input": {0: "batch_size"},
                "policy": {0: "batch_size"},
                "value": {0: "batch_size"},
            },
        )

        # ONNX最適化
        onnx_model = onnx.load(f=onnx_model_path)
        onnx_model = onnx.shape_inference.infer_shapes(
            onnx_model
        )
        onnx_model_simp, check = onnxsim.simplify(onnx_model)
        if not check:
            raise RuntimeError("onnxsim.simplify failed")
        onnx.save(onnx_model_simp, onnx_model_path)

        # ONNX FP16バージョン作成
        onnx_model_fp16_path = Path(
            model_path.stem + "_fp16"
        ).with_suffix(".onnx")
        onnx_model_fp16 = float16.convert_float_to_float16(
            model=onnx_model_simp,
            keep_io_types=True,
            op_block_list=[
                "Gemm",
                "GlobalAveragePool",
                "Flatten",
            ],  # FP16にしたくない演算 (出力層とか)
        )
        onnx.save(onnx_model_fp16, onnx_model_fp16_path)
        # simplifyを挟もうとしたらエラーになったので一旦やめておく
        # onnx_model_fp16 = onnx.shape_inference.infer_shapes(
        #     onnx_model_fp16
        # )
        # onnx_model_simp_fp16, check = onnxsim.simplify(onnx_model_fp16)
        # if not check:
        #     raise RuntimeError("onnxsim.simplify failed")
        # onnx.save(onnx_model_simp_fp16, onnx_model_fp16_path)

        # TensorRTバージョン作成
        if HAS_TENSORRT:
            engine_path = Path(
                model_path.stem + "_tensorrt"
            ).with_suffix(".engine")

            # builder
            trt_logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(trt_logger)

            # create network definition
            # EXPLICIT_BATCHはdeprecatedになって無視されるので設定しない
            network_flags = 0
            network = builder.create_network(network_flags)

            # import model using the ONNX parser
            parser = trt.OnnxParser(network, trt_logger)
            success = parser.parse_from_file(
                str(onnx_model_path.absolute())
            )
            for idx in range(parser.num_errors):
                logger.error(parser.get_error(idx))
            if not success:
                raise RuntimeError("ONNX parse failed")

            # build engine
            config = builder.create_builder_config()
            config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE, 1 << 30
            )

            # FP16最適化
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)

            serialized_engine = (
                builder.build_serialized_network(
                    network, config
                )
            )
            with open(engine_path, "wb") as f:
                f.write(serialized_engine)
