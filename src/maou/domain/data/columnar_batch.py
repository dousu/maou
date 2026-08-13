"""列指向(SOA)データバッチ表現モジュール．

numpy structured arrayの代わりに，各フィールドを独立したC-contiguous配列として保持する
ColumnarBatchを提供する．フィールドアクセス時のメモリコピーが不要になり，
``torch.from_numpy()`` のゼロコピー変換が可能になる．

Polars DataFrame → ColumnarBatch の変換関数も含む．
変換には ``schema.py`` の ``_explode_list_column`` を再利用し，コード重複を防ぐ．
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from maou.domain.data.schema import _explode_list_column
from maou.domain.move.label import MOVE_LABELS_NUM

if TYPE_CHECKING:
    import polars as pl


@dataclass(frozen=True)
class ColumnarBatch:
    """列指向(SOA)のデータバッチ表現．

    numpy structured arrayの代わりに，各フィールドを独立したcontiguous配列として保持する．
    フィールドアクセス時のメモリコピーが不要になり，
    ``torch.from_numpy()`` のゼロコピー変換が可能になる．

    Attributes:
        board_positions: 盤面の駒ID配列．shape: ``(N, 9, 9)``，dtype: ``uint8``，C-contiguous．
        pieces_in_hand: 持ち駒配列．shape: ``(N, 14)``，dtype: ``uint8``，C-contiguous．
        move_label: 指し手ラベル配列．shape: ``(N, MOVE_LABELS_NUM)``，dtype: ``float16``．
            preprocessing用．Stage1/Stage2では ``None``．
        result_value: 対局結果値．shape: ``(N,)``，dtype: ``float16``．
            preprocessing用．Stage1/Stage2では ``None``．
        move_win_rate: 指し手別勝率配列．shape: ``(N, MOVE_LABELS_NUM)``，dtype: ``float32``．
            moveWinRateカラムが存在する場合のみ設定される．旧データでは ``None``．
        reachable_squares: 到達可能マス配列．shape: ``(N, 9, 9)``，dtype: ``uint8``．
            Stage1用．それ以外では ``None``．
        legal_moves_label: 合法手ラベル配列．shape: ``(N, MOVE_LABELS_NUM)``，dtype: ``uint8``．
            Stage2用．それ以外では ``None``．
    """

    board_positions: np.ndarray
    pieces_in_hand: np.ndarray
    move_label: np.ndarray | None = None
    result_value: np.ndarray | None = None
    move_win_rate: np.ndarray | None = None
    reachable_squares: np.ndarray | None = None
    legal_moves_label: np.ndarray | None = None

    def __len__(self) -> int:
        """バッチ内のレコード数を返す."""
        return int(self.board_positions.shape[0])

    @staticmethod
    def concatenate(
        batches: list[ColumnarBatch],
    ) -> ColumnarBatch:
        """複数のColumnarBatchをフィールドごとに連結する．

        structured arrayの全体concatenateと異なり，各フィールドを独立して
        concatenateするため，ピークメモリ使用量を抑えられる．

        Args:
            batches: 連結するColumnarBatchのリスト

        Returns:
            連結されたColumnarBatch

        Raises:
            ValueError: batchesが空の場合
        """
        if not batches:
            raise ValueError(
                "Cannot concatenate empty list of batches"
            )

        board_positions = np.concatenate(
            [b.board_positions for b in batches]
        )
        pieces_in_hand = np.concatenate(
            [b.pieces_in_hand for b in batches]
        )

        def _concat_optional(
            field: str,
        ) -> np.ndarray | None:
            """optional 列を全バッチ一致を確認したうえで連結する．

            有無を ``batches[0]`` だけで判定すると，一部のバッチに列が
            無い場合に**短いまま**連結され，行対応が例外なしに崩れる
            (逆に先頭だけ無いと黙って全部落ちる)．全バッチで揃って
            いることを要求し，揃っていなければ明示的に落とす．
            """
            values = [getattr(b, field) for b in batches]
            present = [v is not None for v in values]
            if not any(present):
                return None
            if not all(present):
                missing = sum(1 for ok in present if not ok)
                raise ValueError(
                    f"ColumnarBatch field '{field}' is present in "
                    f"{len(batches) - missing} of {len(batches)} batches. "
                    "Concatenating batches with mismatched optional "
                    "columns would silently misalign rows."
                )
            return np.concatenate(values)

        move_label = _concat_optional("move_label")

        result_value = _concat_optional("result_value")

        move_win_rate = _concat_optional("move_win_rate")

        reachable_squares = _concat_optional(
            "reachable_squares"
        )

        legal_moves_label = _concat_optional(
            "legal_moves_label"
        )

        return ColumnarBatch(
            board_positions=board_positions,
            pieces_in_hand=pieces_in_hand,
            move_label=move_label,
            result_value=result_value,
            move_win_rate=move_win_rate,
            reachable_squares=reachable_squares,
            legal_moves_label=legal_moves_label,
        )

    def slice(self, indices: np.ndarray) -> ColumnarBatch:
        """fancy indexingによるバッチスライス．

        指定されたインデックスのレコードを含む新しい ``ColumnarBatch`` を返す．

        Args:
            indices: 取得するインデックスの配列

        Returns:
            指定インデックスのレコードを含む新しいColumnarBatch
        """
        return ColumnarBatch(
            board_positions=self.board_positions[indices],
            pieces_in_hand=self.pieces_in_hand[indices],
            move_label=(
                self.move_label[indices]
                if self.move_label is not None
                else None
            ),
            result_value=(
                self.result_value[indices]
                if self.result_value is not None
                else None
            ),
            move_win_rate=(
                self.move_win_rate[indices]
                if self.move_win_rate is not None
                else None
            ),
            reachable_squares=(
                self.reachable_squares[indices]
                if self.reachable_squares is not None
                else None
            ),
            legal_moves_label=(
                self.legal_moves_label[indices]
                if self.legal_moves_label is not None
                else None
            ),
        )


# ============================================================================
# Polars DataFrame → ColumnarBatch conversions
# ============================================================================


def convert_preprocessing_df_to_columnar(
    df: pl.DataFrame,
) -> ColumnarBatch:
    """Polars preprocessing DataFrame → ColumnarBatch 変換．

    ``schema.py`` の ``convert_preprocessing_df_to_numpy()`` と同様のロジックだが，
    structured arrayではなく独立したC-contiguous配列を返す．

    Args:
        df: preprocessing Polarsスキーマを持つDataFrame

    Returns:
        ColumnarBatch: 列指向のバッチデータ
    """
    n = len(df)

    board_positions = _explode_list_column(
        df["boardIdPositions"],
        n,
        (9, 9),
        np.dtype(np.uint8),
        nest_depth=2,
    )
    pieces_in_hand = _explode_list_column(
        df["piecesInHand"],
        n,
        (14,),
        np.dtype(np.uint8),
        nest_depth=1,
    )
    move_label = _explode_list_column(
        df["moveLabel"],
        n,
        (MOVE_LABELS_NUM,),
        np.dtype(np.float16),
        nest_depth=1,
    )
    result_value = (
        df["resultValue"].to_numpy().astype(np.float16)
    )

    move_win_rate: np.ndarray | None = None
    if "moveWinRate" in df.columns:
        move_win_rate = _explode_list_column(
            df["moveWinRate"],
            n,
            (MOVE_LABELS_NUM,),
            np.dtype(np.float32),
            nest_depth=1,
        )

    return ColumnarBatch(
        board_positions=board_positions,
        pieces_in_hand=pieces_in_hand,
        move_label=move_label,
        result_value=result_value,
        move_win_rate=move_win_rate,
    )


def convert_stage1_df_to_columnar(
    df: pl.DataFrame,
) -> ColumnarBatch:
    """Stage 1 Polars DataFrame → ColumnarBatch 変換．

    Args:
        df: Stage 1 Polarsスキーマを持つDataFrame

    Returns:
        ColumnarBatch: 列指向のバッチデータ(reachable_squaresが設定される)
    """
    n = len(df)

    board_positions = _explode_list_column(
        df["boardIdPositions"],
        n,
        (9, 9),
        np.dtype(np.uint8),
        nest_depth=2,
    )
    pieces_in_hand = _explode_list_column(
        df["piecesInHand"],
        n,
        (14,),
        np.dtype(np.uint8),
        nest_depth=1,
    )
    reachable_squares = _explode_list_column(
        df["reachableSquares"],
        n,
        (9, 9),
        np.dtype(np.uint8),
        nest_depth=2,
    )

    return ColumnarBatch(
        board_positions=board_positions,
        pieces_in_hand=pieces_in_hand,
        reachable_squares=reachable_squares,
    )


def convert_stage2_df_to_columnar(
    df: pl.DataFrame,
) -> ColumnarBatch:
    """Stage 2 Polars DataFrame → ColumnarBatch 変換．

    Args:
        df: Stage 2 Polarsスキーマを持つDataFrame

    Returns:
        ColumnarBatch: 列指向のバッチデータ(legal_moves_labelが設定される)
    """
    n = len(df)

    board_positions = _explode_list_column(
        df["boardIdPositions"],
        n,
        (9, 9),
        np.dtype(np.uint8),
        nest_depth=2,
    )
    pieces_in_hand = _explode_list_column(
        df["piecesInHand"],
        n,
        (14,),
        np.dtype(np.uint8),
        nest_depth=1,
    )
    legal_moves_label = _explode_list_column(
        df["legalMovesLabel"],
        n,
        (MOVE_LABELS_NUM,),
        np.dtype(np.uint8),
        nest_depth=1,
    )

    return ColumnarBatch(
        board_positions=board_positions,
        pieces_in_hand=pieces_in_hand,
        legal_moves_label=legal_moves_label,
    )


# ``array_type`` → ColumnarBatch 変換器の唯一の対応表．
# 以前は infra 側の 2 モジュール (``file_data_source`` と
# ``streaming_file_source``) が同内容の dict を各々持っていたため，
# 新しい array_type を足すときに片方だけ更新される事故があり得た．
# 変換器の定義と同じ場所に 1 本だけ置き，両者はこれを import する．
COLUMNAR_CONVERTERS: dict[
    str, Callable[[pl.DataFrame], ColumnarBatch]
] = {
    "preprocessing": convert_preprocessing_df_to_columnar,
    "stage1": convert_stage1_df_to_columnar,
    "stage2": convert_stage2_df_to_columnar,
}
