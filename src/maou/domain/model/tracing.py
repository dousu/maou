"""Tracing detection utilities for model compilation compatibility."""

import torch


def is_tracing() -> bool:
    """Return whether the code is executing under model tracing.

    Covers ``torch.compile``, ``torch.jit.trace``, and ONNX export.
    When tracing is active, validation checks that raise exceptions
    should be skipped to avoid ``TracerWarning`` or graph breaks.

    Returns:
        ``True`` if any form of model tracing is in progress.
    """
    return (
        torch.compiler.is_compiling()
        or torch.jit.is_tracing()
        or torch.onnx.is_in_onnx_export()
    )
