"""Utilities for converting Polars DataFrames to PyTorch tensors．

このモジュールは，Polars DataFrameの行をPyTorchテンソルに効率的に変換する
ユーティリティ関数を提供する．
"""

from __future__ import annotations

import logging
from typing import Any, Union, cast

import numpy as np
import polars as pl
import torch

logger = logging.getLogger(__name__)


def polars_row_to_preprocessing_tensors(
    row: Union[tuple, dict],
    *,
    from_dict: bool = False,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor],  # features
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],  # targets
]:
    """Convert a single Polars DataFrame row to preprocessing tensors．

    Args:
        row: Either a tuple (from df.row(idx)) or dict (from df.row(idx, named=True))
        from_dict: If True, row is a dict; if False, row is a tuple

    Returns:
        Tuple of (features, targets):
            - features: (board_tensor, pieces_in_hand_tensor)
            - targets: (move_label_tensor, result_value_tensor, legal_move_mask_tensor)
    """
    if from_dict:
        # Dict-based access (named=True)
        data = cast(dict[str, Any], row)
        board_id_positions = data["boardIdPositions"]
        pieces_in_hand = data["piecesInHand"]
        move_label = data["moveLabel"]
        result_value = data["resultValue"]
    else:
        # Tuple-based access (faster)
        # Assumes standard preprocessing schema order:
        # id, boardIdPositions, piecesInHand, moveLabel, resultValue
        board_id_positions = row[1]
        pieces_in_hand = row[2]
        move_label = row[3]
        result_value = row[4]

    # Convert to numpy arrays first (Polars lists → numpy)
    board_array = np.array(board_id_positions, dtype=np.uint8)
    pieces_array = np.array(pieces_in_hand, dtype=np.uint8)
    move_label_array = np.array(move_label, dtype=np.float32)

    # Convert to tensors (zero-copy when possible)
    board_tensor = torch.from_numpy(board_array)
    pieces_in_hand_tensor = torch.from_numpy(pieces_array)
    move_label_tensor = torch.from_numpy(move_label_array)
    result_value_tensor = torch.tensor(
        result_value, dtype=torch.float32
    ).reshape((1,))

    # Create legal move mask (all ones for preprocessing data)
    legal_move_mask_tensor = torch.ones_like(move_label_tensor)

    return (
        (board_tensor, pieces_in_hand_tensor),
        (
            move_label_tensor,
            result_value_tensor,
            legal_move_mask_tensor,
        ),
    )


def polars_row_to_hcpe_arrays(
    row: Union[tuple, dict],
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


def polars_row_to_stage1_tensors(
    row: Union[tuple, dict],
    *,
    from_dict: bool = False,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor],  # features
    torch.Tensor,  # target
]:
    """Convert a single Polars DataFrame row to Stage 1 tensors．

    Args:
        row: Either a tuple (from df.row(idx)) or dict (from df.row(idx, named=True))
        from_dict: If True, row is a dict; if False, row is a tuple

    Returns:
        Tuple of (features, target):
            - features: (board_tensor, pieces_in_hand_tensor)
            - target: reachable_squares_tensor (81-dimensional)
    """
    if from_dict:
        data = cast(dict[str, Any], row)
        board_id_positions = data["boardIdPositions"]
        pieces_in_hand = data["piecesInHand"]
        reachable_squares = data["reachableSquares"]
    else:
        # Tuple-based access
        # Stage1 schema: id, boardIdPositions, piecesInHand, reachableSquares
        board_id_positions = row[1]
        pieces_in_hand = row[2]
        reachable_squares = row[3]

    # Convert to numpy arrays
    board_array = np.array(board_id_positions, dtype=np.uint8)
    pieces_array = np.array(pieces_in_hand, dtype=np.uint8)
    reachable_array = np.array(
        reachable_squares, dtype=np.uint8
    ).flatten()

    # Convert to tensors
    board_tensor = torch.from_numpy(board_array)
    pieces_in_hand_tensor = torch.from_numpy(pieces_array)
    reachable_squares_tensor = torch.from_numpy(
        reachable_array
    ).float()

    return (
        (board_tensor, pieces_in_hand_tensor),
        reachable_squares_tensor,
    )


def polars_row_to_stage2_tensors(
    row: Union[tuple, dict],
    *,
    from_dict: bool = False,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor],  # features
    torch.Tensor,  # target
]:
    """Convert a single Polars DataFrame row to Stage 2 tensors．

    Args:
        row: Either a tuple (from df.row(idx)) or dict (from df.row(idx, named=True))
        from_dict: If True, row is a dict; if False, row is a tuple

    Returns:
        Tuple of (features, target):
            - features: (board_tensor, pieces_in_hand_tensor)
            - target: legal_moves_tensor (MOVE_LABELS_NUM-dimensional)
    """
    if from_dict:
        data = cast(dict[str, Any], row)
        board_id_positions = data["boardIdPositions"]
        pieces_in_hand = data["piecesInHand"]
        legal_moves = data["legalMovesLabel"]
    else:
        # Tuple-based access
        # Stage2 schema: id, boardIdPositions, piecesInHand, legalMovesLabel
        board_id_positions = row[1]
        pieces_in_hand = row[2]
        legal_moves = row[3]

    # Convert to numpy arrays
    board_array = np.array(board_id_positions, dtype=np.uint8)
    pieces_array = np.array(pieces_in_hand, dtype=np.uint8)
    legal_moves_array = np.array(legal_moves, dtype=np.uint8)

    # Convert to tensors
    board_tensor = torch.from_numpy(board_array)
    pieces_in_hand_tensor = torch.from_numpy(pieces_array)
    legal_moves_tensor = torch.from_numpy(
        legal_moves_array
    ).float()

    return (
        (board_tensor, pieces_in_hand_tensor),
        legal_moves_tensor,
    )


def dataframe_to_tensor_batch(
    df: pl.DataFrame,
    *,
    array_type: str,
) -> Union[
    tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ],
    tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor],
]:
    """Convert an entire Polars DataFrame to batched tensors．

    Args:
        df: Polars DataFrame with appropriate schema
        array_type: One of "preprocessing", "stage1", "stage2"

    Returns:
        Batched tensors depending on array_type
    """
    if array_type == "preprocessing":
        # Extract columns
        board_list = df["boardIdPositions"].to_list()
        pieces_list = df["piecesInHand"].to_list()
        move_label_list = df["moveLabel"].to_list()
        result_value_list = df["resultValue"].to_list()

        # Convert to batched tensors
        board_tensor = torch.tensor(
            board_list, dtype=torch.uint8
        )
        pieces_tensor = torch.tensor(
            pieces_list, dtype=torch.uint8
        )
        move_label_tensor = torch.tensor(
            move_label_list, dtype=torch.float32
        )
        result_value_tensor = torch.tensor(
            result_value_list, dtype=torch.float32
        ).reshape(-1, 1)
        legal_move_mask_tensor = torch.ones_like(
            move_label_tensor
        )

        return (
            (board_tensor, pieces_tensor),
            (
                move_label_tensor,
                result_value_tensor,
                legal_move_mask_tensor,
            ),
        )

    elif array_type == "stage1":
        board_list = df["boardIdPositions"].to_list()
        pieces_list = df["piecesInHand"].to_list()
        reachable_list = df["reachableSquares"].to_list()

        board_tensor = torch.tensor(
            board_list, dtype=torch.uint8
        )
        pieces_tensor = torch.tensor(
            pieces_list, dtype=torch.uint8
        )
        reachable_tensor = (
            torch.tensor(reachable_list, dtype=torch.uint8)
            .flatten(1)
            .float()
        )

        return (
            (board_tensor, pieces_tensor),
            reachable_tensor,
        )

    elif array_type == "stage2":
        board_list = df["boardIdPositions"].to_list()
        pieces_list = df["piecesInHand"].to_list()
        legal_moves_list = df["legalMovesLabel"].to_list()

        board_tensor = torch.tensor(
            board_list, dtype=torch.uint8
        )
        pieces_tensor = torch.tensor(
            pieces_list, dtype=torch.uint8
        )
        legal_moves_tensor = torch.tensor(
            legal_moves_list, dtype=torch.uint8
        ).float()

        return (
            (board_tensor, pieces_tensor),
            legal_moves_tensor,
        )

    else:
        raise ValueError(
            f"Unsupported array_type: {array_type}"
        )
