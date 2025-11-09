import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from maou.app.learning.network import Network
from maou.app.learning.setup import ModelFactory
from maou.domain.cloud_storage import CloudStorage
from maou.domain.data.schema import create_empty_preprocessing_array

logger: logging.Logger = logging.getLogger(__name__)


class ModelIO:
    @staticmethod
    def save_model(
        *,
        trained_model: Network,
        dir: Path,
        id: str,
        epoch: int,
        device: torch.device,
        cloud_storage: Optional[CloudStorage] = None,
    ) -> None:
        try:
            import onnx
            import onnxsim
            from onnxruntime.transformers import float16
        except ImportError as exc:
            raise ModuleNotFoundError(
                "ONNX export dependencies are missing. "
                "Install with `poetry install -E cpu` or "
                "`poetry install -E cpu-infer` to enable model export."
            ) from exc

        model = ModelFactory.create_shogi_model(device)

        # torch.compile()で生成されたモデルのstate_dictは_orig_mod.プレフィックス付き
        # そのプレフィックスを除去して通常のモデルと互換性を保つ
        state_dict = trained_model.state_dict()
        if any(
            key.startswith("_orig_mod.")
            for key in state_dict.keys()
        ):
            # _orig_mod.プレフィックスを削除
            clean_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("_orig_mod."):
                    clean_key = key[len("_orig_mod.") :]
                    clean_state_dict[clean_key] = value
                else:
                    clean_state_dict[key] = value
            state_dict = clean_state_dict

        model.load_state_dict(state_dict)
        # Training modeを確実に解除しておく
        model.train(False)
        model_path = dir / "model_{}_{}.pt".format(id, epoch)
        logger.info("Saving model to {}".format(model_path))
        torch.save(model.state_dict(), model_path)
        if cloud_storage is not None:
            logger.info(
                f"Uploading model to cloud storage ({model_path})"
            )
            cloud_storage.upload_from_local(
                local_path=model_path,
                cloud_path=str(model_path),
            )
        # AMPのような高速化をしたいので一部FP16にする
        # TensorRTに変換するときはONNXのFP32を利用してBuilderFlag.FP16を指定する

        # torch.onnx.export
        onnx_model_path = model_path.with_suffix(".onnx")
        logger.info(
            "Saving model to {}".format(onnx_model_path)
        )
        dummy_data = create_empty_preprocessing_array(1)
        dummy_board = np.asarray(
            dummy_data[0]["boardIdPositions"], dtype=np.uint8
        )
        dummy_input = (
            torch.from_numpy(dummy_board.astype(np.int64))
            .unsqueeze(0)
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
        if cloud_storage is not None:
            logger.info(
                f"Uploading model to cloud storage ({onnx_model_path})"
            )
            cloud_storage.upload_from_local(
                local_path=onnx_model_path,
                cloud_path=str(onnx_model_path),
            )

        # ONNX FP16バージョン作成
        onnx_model_fp16_path = (
            dir / (model_path.stem + "_fp16")
        ).with_suffix(".onnx")
        logger.info(
            "Saving model to {}".format(onnx_model_fp16_path)
        )
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
        if cloud_storage is not None:
            logger.info(
                f"Uploading model to cloud storage ({onnx_model_fp16_path})"
            )
            cloud_storage.upload_from_local(
                local_path=onnx_model_fp16_path,
                cloud_path=str(onnx_model_fp16_path),
            )
