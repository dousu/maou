import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from maou.app.learning.network import (
    BackboneArchitecture,
    Network,
)
from maou.app.learning.setup import ModelFactory
from maou.domain.cloud_storage import CloudStorage
from maou.domain.data.schema import (
    create_empty_preprocessing_array,
)

logger: logging.Logger = logging.getLogger(__name__)


class ModelIO:
    @staticmethod
    def format_parameter_count(parameter_count: int) -> str:
        """Return a compact，human-friendly parameter count label.

        Args:
            parameter_count: Total number of parameters

        Returns:
            Formatted parameter count (e.g.，"1.2m"，"45k"，"123")

        Examples:
            >>> ModelIO.format_parameter_count(1_234_567)
            '1.2m'
            >>> ModelIO.format_parameter_count(45_000)
            '45k'
            >>> ModelIO.format_parameter_count(123)
            '123'
        """

        def _format(value: float) -> str:
            formatted = f"{value:.1f}"
            if formatted.endswith(".0"):
                return formatted[:-2]
            return formatted

        if parameter_count >= 1_000_000:
            return f"{_format(parameter_count / 1_000_000)}m"
        if parameter_count >= 1_000:
            return f"{_format(parameter_count / 1_000)}k"
        return str(parameter_count)

    @staticmethod
    def generate_model_tag(
        model: Network,
        architecture: BackboneArchitecture,
        *,
        trainable_layers: Optional[int] = None,
    ) -> str:
        """Generate model tag from architecture and parameter count.

        Args:
            model: The neural network model
            architecture: Backbone architecture name
            trainable_layers: Number of trainable backbone groups.
                If specified，appends '-tlN' suffix.

        Returns:
            Model tag (e.g.，"resnet-1.2m"，"vit-19m-tl2")
        """
        parameter_count = sum(
            parameter.numel()
            for parameter in model.parameters()
        )
        parameter_label = ModelIO.format_parameter_count(
            parameter_count
        )
        tag = f"{architecture}-{parameter_label}"
        if trainable_layers is not None:
            tag += f"-tl{trainable_layers}"
        return tag

    @staticmethod
    def split_state_dict(
        state_dict: dict[str, torch.Tensor],
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        """Split state_dict into all head components.

        Splits a full model state_dict into backbone and all head components，
        including the new multi-stage training heads.

        Args:
            state_dict: Complete model state_dict (may include _orig_mod. prefix)

        Returns:
            Tuple of (backbone_dict，policy_head_dict，value_head_dict，
                     reachable_head_dict，legal_moves_head_dict)

        Examples:
            >>> full_dict = model.state_dict()
            >>> backbone，policy，value，reachable，legal = ModelIO.split_state_dict(full_dict)
        """
        backbone_dict: dict[str, torch.Tensor] = {}
        policy_head_dict: dict[str, torch.Tensor] = {}
        value_head_dict: dict[str, torch.Tensor] = {}
        reachable_head_dict: dict[str, torch.Tensor] = {}
        legal_moves_head_dict: dict[str, torch.Tensor] = {}

        # _orig_mod.プレフィックスの除去を検討
        has_orig_mod_prefix = any(
            key.startswith("_orig_mod.")
            for key in state_dict.keys()
        )

        for key, value in state_dict.items():
            # _orig_mod.プレフィックスを除去
            clean_key = key
            if has_orig_mod_prefix and key.startswith(
                "_orig_mod."
            ):
                clean_key = key[len("_orig_mod.") :]

            # コンポーネントごとに分類
            if clean_key.startswith("policy_head."):
                policy_head_dict[key] = value
            elif clean_key.startswith("value_head."):
                value_head_dict[key] = value
            elif clean_key.startswith("reachable_head."):
                reachable_head_dict[key] = value
            elif clean_key.startswith("legal_moves_head."):
                legal_moves_head_dict[key] = value
            else:
                # embedding.*, backbone.*, pool.*, _hand_projection.*はすべてbackbone
                backbone_dict[key] = value

        return (
            backbone_dict,
            policy_head_dict,
            value_head_dict,
            reachable_head_dict,
            legal_moves_head_dict,
        )

    @staticmethod
    def load_backbone(
        file_path: Path, device: torch.device
    ) -> dict[str, torch.Tensor]:
        """Load backbone parameters from file.

        Args:
            file_path: Path to backbone parameter file
            device: Device to load parameters to

        Returns:
            Backbone state_dict
        """
        logger.info(
            f"Loading backbone parameters from {file_path}"
        )
        return torch.load(
            file_path, weights_only=True, map_location=device
        )

    @staticmethod
    def load_policy_head(
        file_path: Path, device: torch.device
    ) -> dict[str, torch.Tensor]:
        """Load policy head parameters from file.

        Args:
            file_path: Path to policy head parameter file
            device: Device to load parameters to

        Returns:
            Policy head state_dict
        """
        logger.info(
            f"Loading policy head parameters from {file_path}"
        )
        return torch.load(
            file_path, weights_only=True, map_location=device
        )

    @staticmethod
    def load_value_head(
        file_path: Path, device: torch.device
    ) -> dict[str, torch.Tensor]:
        """Load value head parameters from file.

        Args:
            file_path: Path to value head parameter file
            device: Device to load parameters to

        Returns:
            Value head state_dict
        """
        logger.info(
            f"Loading value head parameters from {file_path}"
        )
        return torch.load(
            file_path, weights_only=True, map_location=device
        )

    @staticmethod
    def load_reachable_head(
        file_path: Path, device: torch.device
    ) -> dict[str, torch.Tensor]:
        """Load reachable squares head parameters from file.

        Used for Stage 1 multi-stage training.

        Args:
            file_path: Path to reachable head parameter file
            device: Device to load parameters to

        Returns:
            Reachable head state_dict
        """
        logger.info(
            f"Loading reachable squares head parameters from {file_path}"
        )
        return torch.load(
            file_path, weights_only=True, map_location=device
        )

    @staticmethod
    def load_legal_moves_head(
        file_path: Path, device: torch.device
    ) -> dict[str, torch.Tensor]:
        """Load legal moves head parameters from file.

        Used for Stage 2 multi-stage training.

        Args:
            file_path: Path to legal moves head parameter file
            device: Device to load parameters to

        Returns:
            Legal moves head state_dict
        """
        logger.info(
            f"Loading legal moves head parameters from {file_path}"
        )
        return torch.load(
            file_path, weights_only=True, map_location=device
        )

    @staticmethod
    def save_model(
        *,
        trained_model: Network,
        dir: Path,
        id: str,
        epoch: int,
        device: torch.device,
        architecture: BackboneArchitecture,
        cloud_storage: Optional[CloudStorage] = None,
        verify_export: bool = False,
        architecture_config: Optional[dict[str, Any]] = None,
        hand_projection_dim: Optional[int] = None,
    ) -> None:
        try:
            import onnx
            from onnxruntime.transformers import float16
        except ImportError as exc:
            raise ModuleNotFoundError(
                "ONNX export dependencies are missing. "
                "Install with `uv sync --extra cpu` or "
                "`uv sync --extra cpu-infer` to enable model export."
            ) from exc

        # onnxsim is optional (no Python 3.12 wheels available)
        try:
            import onnxsim

            onnxsim_available = True
        except ImportError:
            onnxsim_available = False
            logger.warning(
                "onnxsim not available. "
                "ONNX model simplification will be skipped."
            )

        model = ModelFactory.create_shogi_model(
            device,
            architecture=architecture,
            architecture_config=architecture_config,
            hand_projection_dim=hand_projection_dim,
        )

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

        # パラメータ転送の検証
        if verify_export:
            from maou.app.learning.onnx_verifier import (
                ONNXExportVerifier,
            )

            logger.info(
                "Verifying parameter transfer from trained model..."
            )
            param_report = (
                ONNXExportVerifier.verify_parameter_transfer(
                    trained_model=trained_model,
                    fresh_model=model,
                    cleaned_state_dict=state_dict,
                )
            )
            logger.info(param_report.summary())

            if not param_report.success:
                logger.error(
                    "Parameter transfer verification failed!"
                )
                if param_report.missing_parameters:
                    logger.error(
                        f"Missing parameters: {param_report.missing_parameters}"
                    )
                if param_report.value_mismatches:
                    logger.error(
                        f"Value mismatches: {param_report.value_mismatches}"
                    )
                raise RuntimeError(
                    "Parameter transfer verification failed: "
                    "output head parameters may be lost"
                )

            # 出力ヘッドの検証
            head_report = ONNXExportVerifier.verify_output_head_parameters(
                model
            )
            logger.info(head_report.summary())

            if not head_report.success:
                raise RuntimeError(
                    "Output head verification failed: "
                    "policy or value head parameters are invalid"
                )

        # Training modeを確実に解除しておく
        model.train(False)

        # モデルタグの生成 (architecture-parameterCount)
        model_tag = ModelIO.generate_model_tag(
            model, architecture
        )

        # state_dictを3つのコンポーネントに分割
        full_state_dict = model.state_dict()
        (
            backbone_dict,
            policy_head_dict,
            value_head_dict,
            _reachable_head_dict,
            _legal_moves_head_dict,
        ) = ModelIO.split_state_dict(full_state_dict)

        # 3つの別ファイルに保存
        backbone_path = (
            dir
            / "model_{}_{}_{}_backbone.pt".format(
                id, model_tag, epoch
            )
        )
        policy_head_path = (
            dir
            / "model_{}_{}_{}_policy_head.pt".format(
                id, model_tag, epoch
            )
        )
        value_head_path = (
            dir
            / "model_{}_{}_{}_value_head.pt".format(
                id, model_tag, epoch
            )
        )

        logger.info(
            f"Saving model components (tag: {model_tag}):\n"
            f"  Backbone: {backbone_path}\n"
            f"  Policy Head: {policy_head_path}\n"
            f"  Value Head: {value_head_path}"
        )

        torch.save(backbone_dict, backbone_path)
        torch.save(policy_head_dict, policy_head_path)
        torch.save(value_head_dict, value_head_path)

        # クラウドストレージに3つのファイルをアップロード
        if cloud_storage is not None:
            for component_path in [
                backbone_path,
                policy_head_path,
                value_head_path,
            ]:
                logger.info(
                    f"Uploading model component to cloud storage ({component_path})"
                )
                cloud_storage.upload_from_local(
                    local_path=component_path,
                    cloud_path=str(component_path),
                )
        # AMPのような高速化をしたいので一部FP16にする
        # TensorRTに変換するときはONNXのFP32を利用してBuilderFlag.FP16を指定する

        # torch.onnx.export (統合モデルとしてエクスポート)
        onnx_model_path = dir / "model_{}_{}_{}.onnx".format(
            id, model_tag, epoch
        )
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
        if onnxsim_available:
            onnx_model_simp, check = onnxsim.simplify(
                onnx_model
            )
            if not check:
                raise RuntimeError("onnxsim.simplify failed")
            onnx.save(onnx_model_simp, onnx_model_path)
        else:
            # Skip simplification, save with shape inference only
            onnx_model_simp = onnx_model
            onnx.save(onnx_model_simp, onnx_model_path)

        # ONNX FP32の検証
        if verify_export:
            from maou.app.learning.onnx_verifier import (
                ONNXExportVerifier,
            )

            logger.info("Verifying ONNX FP32 export...")

            # グラフ構造の検証
            graph_report = (
                ONNXExportVerifier.verify_onnx_graph_structure(
                    onnx_model_path=onnx_model_path
                )
            )
            logger.info(graph_report.summary())

            if not graph_report.success:
                logger.warning(
                    f"ONNX graph structure verification failed: {graph_report.summary()}"
                )

            # 機能的等価性の検証
            func_report = ONNXExportVerifier.verify_onnx_functional_equivalence(
                pytorch_model=model,
                onnx_model_path=onnx_model_path,
                device=device,
                num_test_samples=10,
                fp16=False,
            )
            logger.info(func_report.summary())

            if not func_report.success:
                logger.warning(
                    f"ONNX FP32 functional equivalence check failed: {func_report.summary()}"
                )

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
            dir
            / "model_{}_{}_{}_fp16.onnx".format(
                id, model_tag, epoch
            )
        )
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

        # ONNX FP16の検証
        if verify_export:
            from maou.app.learning.onnx_verifier import (
                ONNXExportVerifier,
            )

            logger.info("Verifying ONNX FP16 export...")

            # 機能的等価性の検証（FP16用の緩い許容誤差）
            func_report_fp16 = ONNXExportVerifier.verify_onnx_functional_equivalence(
                pytorch_model=model,
                onnx_model_path=onnx_model_fp16_path,
                device=device,
                num_test_samples=10,
                fp16=True,
            )
            logger.info(func_report_fp16.summary())

            if not func_report_fp16.success:
                logger.warning(
                    f"ONNX FP16 functional equivalence check failed: {func_report_fp16.summary()}"
                )

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
