"""駒のマッピングと変換ユーティリティ（ドメイン層）．

将棋駒のID，名前，表記の変換を提供する純粋関数群．
"""

from typing import Tuple

from maou.domain.board.shogi import PieceId


def is_white_piece(piece_id: int) -> bool:
    """駒IDが後手（白）の駒かどうかを判定．

    Args:
        piece_id: 駒ID（0-28の範囲）

    Returns:
        後手の駒の場合True
    """
    return piece_id >= 15


def get_actual_piece_id(piece_id: int) -> int:
    """後手駒の場合，オフセットを除いた実際の駒IDを取得．

    Args:
        piece_id: 駒ID（後手の場合は +15 されている）

    Returns:
        実際のPieceId値（0-14）
    """
    if is_white_piece(piece_id):
        return piece_id - 15
    return piece_id


def split_piece_id(piece_id: int) -> Tuple[int, bool]:
    """駒IDを実際の駒種と先手/後手フラグに分解．

    Args:
        piece_id: 駒ID

    Returns:
        (actual_piece_id, is_white)のタプル
    """
    is_white = is_white_piece(piece_id)
    actual_id = get_actual_piece_id(piece_id)
    return (actual_id, is_white)


def get_piece_name_ja(piece_id: int) -> str:
    """駒IDから日本語の駒名を取得．

    Args:
        piece_id: 駒ID

    Returns:
        日本語の駒名（例: "歩"，"角"，"龍"）
    """
    actual_id = get_actual_piece_id(piece_id)

    piece_names = {
        PieceId.EMPTY: "",
        PieceId.FU: "歩",
        PieceId.KY: "香",
        PieceId.KE: "桂",
        PieceId.GI: "銀",
        PieceId.KI: "金",
        PieceId.KA: "角",
        PieceId.HI: "飛",
        PieceId.OU: "王",
        PieceId.TO: "と",
        PieceId.NKY: "杏",
        PieceId.NKE: "圭",
        PieceId.NGI: "全",
        PieceId.UMA: "馬",
        PieceId.RYU: "龍",
    }

    return piece_names.get(actual_id, "?")


def square_index_to_coords(square_idx: int) -> Tuple[int, int]:
    """マスインデックス（0-80）を行列座標に変換．

    Args:
        square_idx: マスインデックス（0=左上，80=右下）

    Returns:
        (row, col)のタプル（各0-8）
    """
    if not 0 <= square_idx < 81:
        raise ValueError(
            f"Invalid square index: {square_idx} (must be 0-80)"
        )

    row = square_idx // 9
    col = square_idx % 9
    return (row, col)


def coords_to_square_index(row: int, col: int) -> int:
    """行列座標をマスインデックス（0-80）に変換．

    Args:
        row: 行（0-8）
        col: 列（0-8）

    Returns:
        マスインデックス（0-80）
    """
    if not (0 <= row < 9 and 0 <= col < 9):
        raise ValueError(
            f"Invalid coordinates: row={row}, col={col} (must be 0-8)"
        )

    return row * 9 + col
