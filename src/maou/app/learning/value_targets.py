"""Value教師信号のモード定義．"""

from __future__ import annotations

from enum import Enum

import torch


class ValueTargetMode(str, Enum):
    """Value教師信号のモード．

    Attributes:
        RESULT_VALUE: 局面の勝率(resultValue = win_count / count)を使用する．
        BEST_MOVE_WIN_RATE: 最善手の勝率(moveWinRateの最大値)を使用する．
    """

    RESULT_VALUE = "result-value"
    BEST_MOVE_WIN_RATE = "best-move-win-rate"


def resolve_value_targets(
    labels_value: torch.Tensor,
    *,
    mode: ValueTargetMode,
    move_win_rate: torch.Tensor | None = None,
) -> torch.Tensor:
    """Value教師信号のモードに応じたターゲットを返す．

    RESULT_VALUEモードでは入力テンソルをそのまま返す(コピーなし)．
    呼び出し元で戻り値に対するin-place操作を行わないこと．

    BEST_MOVE_WIN_RATEモードで ``move_win_rate`` がNoneの場合は
    ``labels_value`` にフォールバックする．これはStage 1/2など
    move_win_rateを持たないデータで安全に動作するための措置である．

    Args:
        labels_value: 局面の勝率(resultValue)テンソル．shape: ``(N, 1)``．
        mode: 教師信号モード．
        move_win_rate: 指し手別勝率．BEST_MOVE_WIN_RATEモードで使用．
            shape: ``(N, MOVE_LABELS_NUM)``．Noneの場合は
            ``labels_value`` にフォールバックする．

    Returns:
        Value教師信号テンソル．shape: ``(N, 1)``．
    """
    if mode == ValueTargetMode.RESULT_VALUE:
        return labels_value

    if move_win_rate is None:
        return labels_value

    # moveWinRateの最大値を最善手勝率として使用する
    best_win_rate = move_win_rate.max(
        dim=1, keepdim=True
    ).values
    return best_win_rate.to(
        dtype=labels_value.dtype,
        device=labels_value.device,
        non_blocking=True,
    )
