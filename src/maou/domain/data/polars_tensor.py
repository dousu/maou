"""Utilities for reading rows out of Polars DataFrames．

このモジュールは，Polars DataFrame の行を後段が扱いやすい形へ
変換するユーティリティ関数を提供する．
"""

from __future__ import annotations

from typing import Any, cast


def polars_row_to_hcpe_arrays(
    row: tuple | dict,
    *,
    from_dict: bool = False,
) -> tuple[bytes, int, int, int]:
    """Convert a single Polars DataFrame HCPE row to numpy-compatible format．

    Args:
        row: Either a tuple (from df.row(idx)) or dict (from df.row(idx, named=True))
        from_dict: If True, row is a dict; if False, row is a tuple

    Returns:
        Tuple of (hcp, bestMove16, gameResult, eval)
    """
    if from_dict:
        data = cast(dict[str, Any], row)
        hcp = data["hcp"]
        best_move16 = data["bestMove16"]
        game_result = data["gameResult"]
        eval_value = data["eval"]
    else:
        # Tuple-based access (faster)
        # HCPE schema: hcp, eval, bestMove16, gameResult, id, partitioningKey, ratings, endgameStatus, moves
        hcp = row[0]
        eval_value = row[1]
        best_move16 = row[2]
        game_result = row[3]

    return (hcp, best_move16, game_result, eval_value)
