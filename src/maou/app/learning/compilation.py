"""Utilities for compiling training modules with consistent options."""

from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Final,
    Optional,
    Tuple,
    Type,
    cast,
)

import torch
from torch._dynamo.exc import (
    BackendCompilerFailed,
    TorchRuntimeError,
)

if TYPE_CHECKING:
    from torch._inductor.exc import (
        InductorError as InductorErrorType,
    )
else:  # pragma: no cover - optional import for CUDA builds only
    try:
        from torch._inductor.exc import (
            InductorError as InductorErrorType,
        )
    except (
        Exception
    ):  # pragma: no cover - CPU builds omit torch._inductor

        class InductorErrorType(RuntimeError):
            """Fallback when torch._inductor is unavailable."""

            pass


InductorError = InductorErrorType


logger = logging.getLogger(__name__)

_DYNAMIC_COMPILATION: Final[bool] = False
_FALLBACK_EXCEPTIONS: Final[Tuple[Type[Exception], ...]] = (
    BackendCompilerFailed,
    TorchRuntimeError,
    InductorError,
    RuntimeError,
)


def compile_module(
    module: torch.nn.Module, *, dynamic: Optional[bool] = None
) -> torch.nn.Module:
    """Return a ``torch.compile`` wrapped module.

    Args:
        module: Module to compile with TorchDynamo.
        dynamic: Optional override for the dynamic shape flag. When ``None`` the
            module falls back to the repository-wide default.
    """

    dynamic_flag = (
        _DYNAMIC_COMPILATION if dynamic is None else dynamic
    )

    try:
        compiled = torch.compile(module, dynamic=dynamic_flag)
    except _FALLBACK_EXCEPTIONS as error:
        logger.warning(
            "torch.compile failed with %s, falling back to eager execution",
            error,
        )
        return module
    return cast(torch.nn.Module, compiled)


def warmup_compiled_model(
    model: torch.nn.Module,
    dummy_input: object,
) -> None:
    """ダミーフォワードパスで torch.compile のコンパイルを事前完了させる．

    torch.compile は遅延コンパイルのため，最初の forward パスで
    コンパイルが実行される．tqdm 開始前にこの関数を呼び出すことで，
    速度表示への影響を防ぐ．

    eval モードと ``torch.no_grad`` で実行するため，学習への影響はない．

    Args:
        model: コンパイル済みモジュール．
        dummy_input: モデルの forward に渡すダミー入力．
    """
    import time

    logger.info(
        "torch.compile warmup: triggering compilation..."
    )
    was_training = model.training
    model.eval()
    start = time.monotonic()
    try:
        with torch.no_grad():
            model(dummy_input)
    except _FALLBACK_EXCEPTIONS as error:
        logger.warning(
            "torch.compile warmup failed with %s, "
            "compilation will occur during training",
            error,
        )
        return
    finally:
        model.train(was_training)
    elapsed = time.monotonic() - start
    logger.info(
        "torch.compile warmup completed in %.1fs - "
        "model is compiled and ready",
        elapsed,
    )
