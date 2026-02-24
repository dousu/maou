"""Custom legal move generation for shogi pieces.

このモジュールは，cshogiに依存せずに将棋の駒の合法手を生成する．
stage1データ生成では単一の駒のみが存在するため，玉の存在確認や
王手のチェックは不要である．

座標系: (row, col) where row 0 = top, col 0 = left
盤面サイズ: 9x9 (row 0-8, col 0-8)
"""

from typing import List, Tuple

from maou.domain.board.shogi import PieceId


def _is_valid_square(row: int, col: int) -> bool:
    """盤面内の有効な座標かどうかを判定する．

    Args:
        row: 行 (0-8)
        col: 列 (0-8)

    Returns:
        bool: 盤面内であればTrue
    """
    return 0 <= row < 9 and 0 <= col < 9


def _get_pawn_moves(
    row: int, col: int
) -> List[Tuple[int, int]]:
    """歩の合法手を取得する (1マス前進)．

    Args:
        row: 駒の行
        col: 駒の列

    Returns:
        List[Tuple[int, int]]: 合法手の座標リスト [(row, col), ...]
    """
    moves = []
    # 歩は1マス前進 (先手視点では row - 1)
    new_row = row - 1
    if _is_valid_square(new_row, col):
        moves.append((new_row, col))
    return moves


def _get_lance_moves(
    row: int, col: int
) -> List[Tuple[int, int]]:
    """香車の合法手を取得する (直進)．

    Args:
        row: 駒の行
        col: 駒の列

    Returns:
        List[Tuple[int, int]]: 合法手の座標リスト
    """
    moves = []
    # 香車は前方に直進 (stage1では駒が1つだけなので障害物なし)
    for new_row in range(row - 1, -1, -1):
        moves.append((new_row, col))
    return moves


def _get_knight_moves(
    row: int, col: int
) -> List[Tuple[int, int]]:
    """桂馬の合法手を取得する (L字型: 2マス前進，1マス左右)．

    Args:
        row: 駒の行
        col: 駒の列

    Returns:
        List[Tuple[int, int]]: 合法手の座標リスト
    """
    moves = []
    # 桂馬: 2マス前進，1マス左右
    candidates = [
        (row - 2, col - 1),  # 2前，1左
        (row - 2, col + 1),  # 2前，1右
    ]
    for new_row, new_col in candidates:
        if _is_valid_square(new_row, new_col):
            moves.append((new_row, new_col))
    return moves


def _get_silver_moves(
    row: int, col: int
) -> List[Tuple[int, int]]:
    """銀将の合法手を取得する (前方3方向 + 斜め後方2方向)．

    Args:
        row: 駒の行
        col: 駒の列

    Returns:
        List[Tuple[int, int]]: 合法手の座標リスト
    """
    moves = []
    # 銀将: 前方3方向 (前，斜め前左，斜め前右) + 斜め後方2方向
    candidates = [
        (row - 1, col),  # 前
        (row - 1, col - 1),  # 斜め前左
        (row - 1, col + 1),  # 斜め前右
        (row + 1, col - 1),  # 斜め後左
        (row + 1, col + 1),  # 斜め後右
    ]
    for new_row, new_col in candidates:
        if _is_valid_square(new_row, new_col):
            moves.append((new_row, new_col))
    return moves


def _get_gold_moves(
    row: int, col: int
) -> List[Tuple[int, int]]:
    """金将の合法手を取得する (前方3方向 + 横2方向 + 後方1方向)．

    Args:
        row: 駒の行
        col: 駒の列

    Returns:
        List[Tuple[int, int]]: 合法手の座標リスト
    """
    moves = []
    # 金将: 前方3方向 + 横2方向 + 後方1方向
    candidates = [
        (row - 1, col),  # 前
        (row - 1, col - 1),  # 斜め前左
        (row - 1, col + 1),  # 斜め前右
        (row, col - 1),  # 左
        (row, col + 1),  # 右
        (row + 1, col),  # 後
    ]
    for new_row, new_col in candidates:
        if _is_valid_square(new_row, new_col):
            moves.append((new_row, new_col))
    return moves


def _get_bishop_moves(
    row: int, col: int
) -> List[Tuple[int, int]]:
    """角行の合法手を取得する (斜め4方向)．

    Args:
        row: 駒の行
        col: 駒の列

    Returns:
        List[Tuple[int, int]]: 合法手の座標リスト
    """
    moves = []
    # 角行: 斜め4方向 (stage1では駒が1つだけなので障害物なし)
    directions = [
        (-1, -1),  # 左上
        (-1, +1),  # 右上
        (+1, -1),  # 左下
        (+1, +1),  # 右下
    ]
    for d_row, d_col in directions:
        for i in range(1, 9):  # 最大8マスまで
            new_row = row + d_row * i
            new_col = col + d_col * i
            if _is_valid_square(new_row, new_col):
                moves.append((new_row, new_col))
            else:
                break
    return moves


def _get_rook_moves(
    row: int, col: int
) -> List[Tuple[int, int]]:
    """飛車の合法手を取得する (縦横4方向)．

    Args:
        row: 駒の行
        col: 駒の列

    Returns:
        List[Tuple[int, int]]: 合法手の座標リスト
    """
    moves = []
    # 飛車: 縦横4方向 (stage1では駒が1つだけなので障害物なし)
    directions = [
        (-1, 0),  # 上
        (+1, 0),  # 下
        (0, -1),  # 左
        (0, +1),  # 右
    ]
    for d_row, d_col in directions:
        for i in range(1, 9):  # 最大8マスまで
            new_row = row + d_row * i
            new_col = col + d_col * i
            if _is_valid_square(new_row, new_col):
                moves.append((new_row, new_col))
            else:
                break
    return moves


def _get_king_moves(
    row: int, col: int
) -> List[Tuple[int, int]]:
    """玉の合法手を取得する (全方向1マス)．

    Args:
        row: 駒の行
        col: 駒の列

    Returns:
        List[Tuple[int, int]]: 合法手の座標リスト
    """
    moves = []
    # 玉: 全方向1マス
    candidates = [
        (row - 1, col - 1),  # 左上
        (row - 1, col),  # 上
        (row - 1, col + 1),  # 右上
        (row, col - 1),  # 左
        (row, col + 1),  # 右
        (row + 1, col - 1),  # 左下
        (row + 1, col),  # 下
        (row + 1, col + 1),  # 右下
    ]
    for new_row, new_col in candidates:
        if _is_valid_square(new_row, new_col):
            moves.append((new_row, new_col))
    return moves


def _get_promoted_pawn_moves(
    row: int, col: int
) -> List[Tuple[int, int]]:
    """と金の合法手を取得する (金将と同じ動き)．

    Args:
        row: 駒の行
        col: 駒の列

    Returns:
        List[Tuple[int, int]]: 合法手の座標リスト
    """
    return _get_gold_moves(row, col)


def _get_promoted_lance_moves(
    row: int, col: int
) -> List[Tuple[int, int]]:
    """成香の合法手を取得する (金将と同じ動き)．

    Args:
        row: 駒の行
        col: 駒の列

    Returns:
        List[Tuple[int, int]]: 合法手の座標リスト
    """
    return _get_gold_moves(row, col)


def _get_promoted_knight_moves(
    row: int, col: int
) -> List[Tuple[int, int]]:
    """成桂の合法手を取得する (金将と同じ動き)．

    Args:
        row: 駒の行
        col: 駒の列

    Returns:
        List[Tuple[int, int]]: 合法手の座標リスト
    """
    return _get_gold_moves(row, col)


def _get_promoted_silver_moves(
    row: int, col: int
) -> List[Tuple[int, int]]:
    """成銀の合法手を取得する (金将と同じ動き)．

    Args:
        row: 駒の行
        col: 駒の列

    Returns:
        List[Tuple[int, int]]: 合法手の座標リスト
    """
    return _get_gold_moves(row, col)


def _get_promoted_bishop_moves(
    row: int, col: int
) -> List[Tuple[int, int]]:
    """竜馬 (成角) の合法手を取得する (角行の動き + 縦横1マス)．

    Args:
        row: 駒の行
        col: 駒の列

    Returns:
        List[Tuple[int, int]]: 合法手の座標リスト
    """
    moves = _get_bishop_moves(row, col)
    # 縦横1マスを追加
    additional = [
        (row - 1, col),  # 上
        (row + 1, col),  # 下
        (row, col - 1),  # 左
        (row, col + 1),  # 右
    ]
    for new_row, new_col in additional:
        if _is_valid_square(new_row, new_col):
            moves.append((new_row, new_col))
    return moves


def _get_promoted_rook_moves(
    row: int, col: int
) -> List[Tuple[int, int]]:
    """竜王 (成飛) の合法手を取得する (飛車の動き + 斜め1マス)．

    Args:
        row: 駒の行
        col: 駒の列

    Returns:
        List[Tuple[int, int]]: 合法手の座標リスト
    """
    moves = _get_rook_moves(row, col)
    # 斜め1マスを追加
    additional = [
        (row - 1, col - 1),  # 左上
        (row - 1, col + 1),  # 右上
        (row + 1, col - 1),  # 左下
        (row + 1, col + 1),  # 右下
    ]
    for new_row, new_col in additional:
        if _is_valid_square(new_row, new_col):
            moves.append((new_row, new_col))
    return moves


def get_legal_drop_squares_for_piece(
    piece_type_idx: int,
) -> List[Tuple[int, int]]:
    """持ち駒の打ち手 (drop moves) の合法な着手先を取得する．

    stage1データ生成では盤面が空なので，駒の種類に応じた
    着手可能エリアのみをチェックする (二歩や行き場のない駒は考慮しない)．

    Args:
        piece_type_idx: 駒の種類インデックス (0=FU, 1=KY, 2=KE, 3=GI, 4=KI, 5=KA, 6=HI)

    Returns:
        List[Tuple[int, int]]: 着手可能な座標リスト [(row, col), ...]

    Raises:
        ValueError: 無効な駒種別が指定された場合
    """
    drop_squares = []

    # 駒の種類によって着手可能な行を制限
    if piece_type_idx == 0:  # FU (歩)
        # 歩は1段目には打てない (行き場がない)
        for row in range(1, 9):
            for col in range(9):
                drop_squares.append((row, col))
    elif piece_type_idx == 1:  # KY (香車)
        # 香車は1段目には打てない (行き場がない)
        for row in range(1, 9):
            for col in range(9):
                drop_squares.append((row, col))
    elif piece_type_idx == 2:  # KE (桂馬)
        # 桂馬は1，2段目には打てない (行き場がない)
        for row in range(2, 9):
            for col in range(9):
                drop_squares.append((row, col))
    else:  # GI, KI, KA, HI (銀，金，角，飛)
        # その他の駒はどこでも打てる
        for row in range(9):
            for col in range(9):
                drop_squares.append((row, col))

    return drop_squares


def get_legal_moves_for_piece(
    piece_id: PieceId,
    row: int,
    col: int,
) -> List[Tuple[int, int]]:
    """指定された駒の合法手を取得する．

    このメソッドはcshogiに依存せず，純粋なPythonで合法手を計算する．
    stage1データ生成では単一の駒のみが存在するため，
    他の駒との衝突や王手のチェックは不要である．

    Args:
        piece_id: 駒の種類 (PieceId)
        row: 駒の行 (0-8)
        col: 駒の列 (0-8)

    Returns:
        List[Tuple[int, int]]: 合法手の座標リスト [(row, col), ...]

    Raises:
        ValueError: 無効な駒IDが指定された場合
    """
    if piece_id == PieceId.FU:
        return _get_pawn_moves(row, col)
    elif piece_id == PieceId.KY:
        return _get_lance_moves(row, col)
    elif piece_id == PieceId.KE:
        return _get_knight_moves(row, col)
    elif piece_id == PieceId.GI:
        return _get_silver_moves(row, col)
    elif piece_id == PieceId.KI:
        return _get_gold_moves(row, col)
    elif piece_id == PieceId.KA:
        return _get_bishop_moves(row, col)
    elif piece_id == PieceId.HI:
        return _get_rook_moves(row, col)
    elif piece_id == PieceId.OU:
        return _get_king_moves(row, col)
    elif piece_id == PieceId.TO:
        return _get_promoted_pawn_moves(row, col)
    elif piece_id == PieceId.NKY:
        return _get_promoted_lance_moves(row, col)
    elif piece_id == PieceId.NKE:
        return _get_promoted_knight_moves(row, col)
    elif piece_id == PieceId.NGI:
        return _get_promoted_silver_moves(row, col)
    elif piece_id == PieceId.UMA:
        return _get_promoted_bishop_moves(row, col)
    elif piece_id == PieceId.RYU:
        return _get_promoted_rook_moves(row, col)
    else:
        raise ValueError(f"Unknown piece ID: {piece_id}")
