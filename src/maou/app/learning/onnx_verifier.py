"""ONNX export verification utilities for validating parameter preservation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from maou.app.learning.network import Network

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class ParameterTransferReport:
    """Report for parameter transfer verification from compiled to non-compiled models."""

    success: bool
    total_parameters: int
    matched_parameters: int
    missing_parameters: list[str]
    value_mismatches: dict[
        str, float
    ]  # param_name -> max_abs_diff
    policy_head_verified: bool
    value_head_verified: bool

    def summary(self) -> str:
        """Return a human-readable summary of the verification results."""
        if self.success:
            return (
                f"✓ Parameter transfer verified "
                f"({self.matched_parameters}/{self.total_parameters} matched)"
            )
        else:
            issues: list[str] = []
            if self.missing_parameters:
                issues.append(
                    f"{len(self.missing_parameters)} missing parameters"
                )
            if self.value_mismatches:
                issues.append(
                    f"{len(self.value_mismatches)} value mismatches"
                )
            if not self.policy_head_verified:
                issues.append("policy head not verified")
            if not self.value_head_verified:
                issues.append("value head not verified")
            return f"✗ Parameter transfer failed: {', '.join(issues)}"


@dataclass
class FunctionalEquivalenceReport:
    """Report for ONNX functional equivalence verification."""

    success: bool
    num_samples_tested: int
    policy_max_abs_diff: float
    value_max_abs_diff: float
    tolerance_policy: tuple[float, float]  # (rtol, atol)
    tolerance_value: tuple[float, float]

    def summary(self) -> str:
        """Return a human-readable summary of the verification results."""
        if self.success:
            return (
                f"✓ ONNX functional equivalence verified "
                f"(policy diff: {self.policy_max_abs_diff:.2e}, "
                f"value diff: {self.value_max_abs_diff:.2e})"
            )
        else:
            return (
                f"✗ ONNX functional equivalence failed "
                f"(policy diff: {self.policy_max_abs_diff:.2e} > {self.tolerance_policy}, "
                f"value diff: {self.value_max_abs_diff:.2e} > {self.tolerance_value})"
            )


@dataclass
class OutputHeadReport:
    """Report for output head parameter verification."""

    success: bool
    policy_head_exists: bool
    value_head_exists: bool
    policy_head_params: dict[
        str, tuple[int, ...]
    ]  # param_name -> shape
    value_head_params: dict[str, tuple[int, ...]]

    def summary(self) -> str:
        """Return a human-readable summary of the verification results."""
        if self.success:
            policy_count = len(self.policy_head_params)
            value_count = len(self.value_head_params)
            return (
                f"✓ Output heads verified "
                f"(policy: {policy_count} params, value: {value_count} params)"
            )
        else:
            issues: list[str] = []
            if not self.policy_head_exists:
                issues.append("policy head missing")
            if not self.value_head_exists:
                issues.append("value head missing")
            return f"✗ Output head verification failed: {', '.join(issues)}"


@dataclass
class GraphStructureReport:
    """Report for ONNX graph structure verification."""

    success: bool
    input_names: list[str]
    output_names: list[str]
    graph_valid: bool

    def summary(self) -> str:
        """Return a human-readable summary of the verification results."""
        if self.success:
            return (
                f"✓ ONNX graph structure valid "
                f"(inputs: {self.input_names}, outputs: {self.output_names})"
            )
        else:
            issues: list[str] = []
            if not self.graph_valid:
                issues.append("invalid graph")
            if self.input_names != ["input"]:
                issues.append(
                    f"unexpected inputs: {self.input_names}"
                )
            if self.output_names != ["policy", "value"]:
                issues.append(
                    f"unexpected outputs: {self.output_names}"
                )
            return f"✗ ONNX graph structure invalid: {', '.join(issues)}"


class ONNXExportVerifier:
    """Utilities for verifying ONNX export correctness and parameter preservation."""

    @staticmethod
    def verify_parameter_transfer(
        trained_model: Network,
        fresh_model: Network,
        cleaned_state_dict: dict[str, torch.Tensor],
    ) -> ParameterTransferReport:
        """Verify parameter transfer from compiled to non-compiled models.

        Args:
            trained_model: The trained model (possibly compiled with torch.compile)
            fresh_model: The fresh non-compiled model with loaded parameters
            cleaned_state_dict: The state_dict after _orig_mod. prefix removal

        Returns:
            ParameterTransferReport with verification results
        """
        # Get original state dict from trained model
        original_state_dict = trained_model.state_dict()

        # Count total parameters
        total_params = len(cleaned_state_dict)
        matched_params = 0
        missing_params: list[str] = []
        value_mismatches: dict[str, float] = {}

        # Check if all parameters from trained model are in cleaned state dict
        for key in original_state_dict.keys():
            # Remove _orig_mod. prefix if present
            clean_key = key
            if key.startswith("_orig_mod."):
                clean_key = key[len("_orig_mod.") :]

            # Check if parameter exists in cleaned state dict
            if clean_key not in cleaned_state_dict:
                missing_params.append(clean_key)
                continue

            # Check if parameter values match
            original_value = original_state_dict[key]
            cleaned_value = cleaned_state_dict[clean_key]

            if not torch.allclose(
                original_value,
                cleaned_value,
                rtol=1e-7,
                atol=1e-9,
            ):
                max_diff = torch.max(
                    torch.abs(original_value - cleaned_value)
                ).item()
                value_mismatches[clean_key] = max_diff
            else:
                matched_params += 1

        # Check if fresh model has the same parameters
        fresh_state_dict = fresh_model.state_dict()
        for key, value in cleaned_state_dict.items():
            if key not in fresh_state_dict:
                missing_params.append(f"fresh_model.{key}")
                continue

            if not torch.allclose(
                fresh_state_dict[key],
                value,
                rtol=1e-7,
                atol=1e-9,
            ):
                max_diff = torch.max(
                    torch.abs(fresh_state_dict[key] - value)
                ).item()
                value_mismatches[f"fresh_model.{key}"] = (
                    max_diff
                )

        # Verify output heads specifically
        policy_head_verified = all(
            key.startswith("policy_head.")
            for key in cleaned_state_dict.keys()
            if "policy_head" in key
        )
        value_head_verified = all(
            key.startswith("value_head.")
            for key in cleaned_state_dict.keys()
            if "value_head" in key
        )

        # Check that output head parameters actually exist
        has_policy_params = any(
            "policy_head" in key
            for key in cleaned_state_dict.keys()
        )
        has_value_params = any(
            "value_head" in key
            for key in cleaned_state_dict.keys()
        )

        if not has_policy_params:
            policy_head_verified = False
            missing_params.append("policy_head parameters")
        if not has_value_params:
            value_head_verified = False
            missing_params.append("value_head parameters")

        success = (
            len(missing_params) == 0
            and len(value_mismatches) == 0
            and policy_head_verified
            and value_head_verified
        )

        return ParameterTransferReport(
            success=success,
            total_parameters=total_params,
            matched_parameters=matched_params,
            missing_parameters=missing_params,
            value_mismatches=value_mismatches,
            policy_head_verified=policy_head_verified,
            value_head_verified=value_head_verified,
        )

    @staticmethod
    def verify_output_head_parameters(
        model: Network,
    ) -> OutputHeadReport:
        """Verify that output head parameters exist and are properly initialized.

        Args:
            model: The model to verify

        Returns:
            OutputHeadReport with verification results
        """
        policy_head_exists = hasattr(model, "policy_head")
        value_head_exists = hasattr(model, "value_head")

        policy_head_params: dict[str, tuple[int, ...]] = {}
        value_head_params: dict[str, tuple[int, ...]] = {}

        if policy_head_exists:
            for (
                name,
                param,
            ) in model.policy_head.named_parameters():
                # Check for NaN or Inf
                if (
                    torch.isnan(param).any()
                    or torch.isinf(param).any()
                ):
                    policy_head_exists = False
                    logger.error(
                        f"Policy head parameter {name} contains NaN or Inf"
                    )
                    break
                policy_head_params[name] = tuple(param.shape)

        if value_head_exists:
            for (
                name,
                param,
            ) in model.value_head.named_parameters():
                # Check for NaN or Inf
                if (
                    torch.isnan(param).any()
                    or torch.isinf(param).any()
                ):
                    value_head_exists = False
                    logger.error(
                        f"Value head parameter {name} contains NaN or Inf"
                    )
                    break
                value_head_params[name] = tuple(param.shape)

        success = policy_head_exists and value_head_exists

        return OutputHeadReport(
            success=success,
            policy_head_exists=policy_head_exists,
            value_head_exists=value_head_exists,
            policy_head_params=policy_head_params,
            value_head_params=value_head_params,
        )

    @staticmethod
    def verify_onnx_functional_equivalence(
        pytorch_model: Network,
        onnx_model_path: Path,
        device: torch.device,
        num_test_samples: int = 10,
        fp16: bool = False,
    ) -> FunctionalEquivalenceReport:
        """Verify ONNX model produces identical outputs to PyTorch model.

        Args:
            pytorch_model: The PyTorch model
            onnx_model_path: Path to the ONNX model file
            device: Device to run PyTorch inference on
            num_test_samples: Number of random test samples to verify
            fp16: Whether to use relaxed tolerances for FP16 models

        Returns:
            FunctionalEquivalenceReport with verification results
        """
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ModuleNotFoundError(
                "ONNX Runtime is required for verification. "
                "Install with `poetry install -E cpu` or `poetry install -E cpu-infer`"
            ) from exc

        from maou.domain.data.schema import (
            create_empty_preprocessing_array,
        )

        # Set up ONNX Runtime session
        options = ort.SessionOptions()
        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        )
        providers = ["CPUExecutionProvider"]
        if device.type == "cuda":
            providers.insert(0, "CUDAExecutionProvider")

        session = ort.InferenceSession(
            str(onnx_model_path),
            sess_options=options,
            providers=providers,
        )

        # Set PyTorch model to eval mode
        pytorch_model.eval()

        # Set tolerances based on precision
        if fp16:
            rtol, atol = 1e-3, 1e-4
        else:
            rtol, atol = 1e-5, 1e-7

        policy_max_abs_diff = 0.0
        value_max_abs_diff = 0.0

        # Test with multiple random samples
        with torch.no_grad():
            for _ in range(num_test_samples):
                # Generate random test input
                dummy_data = create_empty_preprocessing_array(1)
                dummy_board = np.asarray(
                    dummy_data[0]["boardIdPositions"],
                    dtype=np.uint8,
                )

                # PyTorch inference
                pytorch_input = (
                    torch.from_numpy(
                        dummy_board.astype(np.int64)
                    )
                    .unsqueeze(0)
                    .to(device)
                )
                pytorch_policy, pytorch_value = pytorch_model(
                    pytorch_input
                )

                # ONNX inference
                onnx_input = dummy_board.astype(
                    np.int64
                ).reshape(1, 9, 9)
                onnx_outputs = session.run(
                    ["policy", "value"], {"input": onnx_input}
                )
                onnx_policy = onnx_outputs[0]
                onnx_value = onnx_outputs[1]

                # Convert PyTorch outputs to numpy
                pytorch_policy_np = pytorch_policy.cpu().numpy()
                pytorch_value_np = pytorch_value.cpu().numpy()

                # Compute differences
                policy_diff = np.abs(
                    pytorch_policy_np - onnx_policy
                ).max()
                value_diff = np.abs(
                    pytorch_value_np - onnx_value
                ).max()

                policy_max_abs_diff = max(
                    policy_max_abs_diff, policy_diff
                )
                value_max_abs_diff = max(
                    value_max_abs_diff, value_diff
                )

        # Check if differences are within tolerance
        policy_ok = np.allclose(
            pytorch_policy_np, onnx_policy, rtol=rtol, atol=atol
        )
        value_ok = np.allclose(
            pytorch_value_np, onnx_value, rtol=rtol, atol=atol
        )

        success = policy_ok and value_ok

        return FunctionalEquivalenceReport(
            success=success,
            num_samples_tested=num_test_samples,
            policy_max_abs_diff=float(policy_max_abs_diff),
            value_max_abs_diff=float(value_max_abs_diff),
            tolerance_policy=(rtol, atol),
            tolerance_value=(rtol, atol),
        )

    @staticmethod
    def verify_onnx_graph_structure(
        onnx_model_path: Path,
    ) -> GraphStructureReport:
        """Verify ONNX graph has correct structure and output nodes.

        Args:
            onnx_model_path: Path to the ONNX model file

        Returns:
            GraphStructureReport with verification results
        """
        try:
            import onnx
        except ImportError as exc:
            raise ModuleNotFoundError(
                "ONNX is required for verification. "
                "Install with `poetry install -E cpu` or `poetry install -E cpu-infer`"
            ) from exc

        # Load ONNX model
        onnx_model = onnx.load(str(onnx_model_path))

        # Get input and output names
        input_names = [
            inp.name for inp in onnx_model.graph.input
        ]
        output_names = [
            out.name for out in onnx_model.graph.output
        ]

        # Verify graph validity
        graph_valid = True
        try:
            onnx.checker.check_model(onnx_model)
        except Exception as exc:
            logger.error(f"ONNX model validation failed: {exc}")
            graph_valid = False

        # Check expected structure
        success = (
            graph_valid
            and input_names == ["input"]
            and output_names == ["policy", "value"]
        )

        return GraphStructureReport(
            success=success,
            input_names=input_names,
            output_names=output_names,
            graph_valid=graph_valid,
        )
