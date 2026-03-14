from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from maou._rust.maou_shogi import PyBoard as _PyBoard
from maou._rust.maou_shogi import move16 as _move16
from maou._rust.maou_shogi import (
    move_drop_hand_piece as _move_drop_hand_piece,
)
from maou._rust.maou_shogi import move_from as _move_from
from maou._rust.maou_shogi import move_is_drop as _move_is_drop
from maou._rust.maou_shogi import (
    move_is_promotion as _move_is_promotion,
)
from maou._rust.maou_shogi import move_to as _move_to
from maou._rust.maou_shogi import move_to_usi as _move_to_usi

if TYPE_CHECKING:
    import polars as pl

# Eager import for performance-critical paths
try:
    import polars as _pl

    _POLARS_AVAILABLE = True
except ImportError:
    _POLARS_AVAILABLE = False
    _pl = None  # type: ignore

# Domain-level constants (not imported from cshogi)
MAX_PIECES_IN_HAND: list[int] = [
    18,
    4,
    4,
    4,
    4,
    2,
    2,
]  # 歩，香，桂，銀，金，角，飛
# 駒8種類，成駒6種類
PIECE_TYPES: int = 14  # 8 unpromoted + 6 promoted piece types

# MAX_PIECES_IN_HANDの構成
# 歩18，香車4，桂馬4，銀4，金4，角2，飛車2

# 104
FEATURES_NUM = PIECE_TYPES * 2 + sum(MAX_PIECES_IN_HAND) * 2


class Turn(IntEnum):
    BLACK = 0
    WHITE = 1


class Result(IntEnum):
    BLACK_WIN = 0
    WHITE_WIN = 1
    DRAW = 2


class PieceId(IntEnum):
    EMPTY = 0
    # 歩
    FU = auto()
    # 香車
    KY = auto()
    # 桂馬
    KE = auto()
    # 銀
    GI = auto()
    # 金
    KI = auto()
    # 角
    KA = auto()
    # 飛車
    HI = auto()
    # 王
    OU = auto()
    # と金
    TO = auto()
    # 成香
    NKY = auto()
    # 成桂
    NKE = auto()
    # 成銀
    NGI = auto()
    # 馬
    UMA = auto()
    # 龍
    RYU = auto()


# cshogi piece ID constants (for board setup)
# Black pieces: 1-14, White pieces: 17-30 (black + 16)
CSHOGI_BLACK_KING = 8  # cshogi.BKING
CSHOGI_WHITE_KING = 24  # cshogi.WKING

# ============================================================================
# Piece ID System Constants
# ============================================================================
#
# cshogi形式とdomain形式で白駒オフセットが異なる．
# マジックナンバーを排除するため，以下の定数を使用すること．
#
#   cshogi形式:  白駒 = 黒駒 + 16  (1-14 → 17-30)
#   domain形式:  白駒 = 黒駒 + 14  (0-14 → 15-28)
#

# cshogi形式の定数
CSHOGI_WHITE_OFFSET: int = 16
"""cshogi形式での白駒オフセット．白駒ID = 黒駒ID + 16．"""

CSHOGI_BLACK_MIN: int = 1
CSHOGI_BLACK_MAX: int = 14
CSHOGI_WHITE_MIN: int = 17
CSHOGI_WHITE_MAX: int = 30

# domain形式の定数
DOMAIN_WHITE_OFFSET: int = 14
"""domain形式での白駒オフセット．白駒ID = 黒駒ID + 14．"""

DOMAIN_BLACK_MIN: int = 0
DOMAIN_BLACK_MAX: int = 14
DOMAIN_WHITE_MIN: int = 15
DOMAIN_WHITE_MAX: int = 28


def is_white_cshogi(cshogi_piece: int) -> bool:
    """cshogi形式で白駒か判定．

    Args:
        cshogi_piece: cshogi駒ID (0=空, 1-14=黒駒, 17-30=白駒)

    Returns:
        True if cshogi_piece >= 17 (白駒)

    Examples:
        >>> is_white_cshogi(1)   # 黒歩
        False
        >>> is_white_cshogi(17)  # 白歩
        True
        >>> is_white_cshogi(0)   # 空
        False
    """
    return cshogi_piece >= CSHOGI_WHITE_MIN


def cshogi_to_base_piece(cshogi_piece: int) -> int:
    """cshogi形式から基本駒ID(1-14)を取得．

    白駒の場合はオフセットを減算して基本駒IDを返す．

    Args:
        cshogi_piece: cshogi駒ID (1-14=黒駒, 17-30=白駒)

    Returns:
        基本駒ID (1-14)

    Examples:
        >>> cshogi_to_base_piece(1)   # 黒歩 → 1
        1
        >>> cshogi_to_base_piece(17)  # 白歩 → 1
        1
        >>> cshogi_to_base_piece(24)  # 白王 → 8
        8
    """
    if is_white_cshogi(cshogi_piece):
        return cshogi_piece - CSHOGI_WHITE_OFFSET
    return cshogi_piece


def is_white_domain(domain_piece: int) -> bool:
    """domain形式で白駒か判定．

    Args:
        domain_piece: domain駒ID (0=空, 1-14=黒駒, 15-28=白駒)

    Returns:
        True if domain_piece >= 15 (白駒)

    Examples:
        >>> is_white_domain(1)   # 黒歩
        False
        >>> is_white_domain(15)  # 白歩
        True
        >>> is_white_domain(0)   # 空
        False
    """
    return domain_piece >= DOMAIN_WHITE_MIN


def domain_to_base_piece(domain_piece: int) -> int:
    """domain形式から基本駒ID(0-14)を取得．

    白駒の場合はオフセットを減算して基本駒IDを返す．

    Args:
        domain_piece: domain駒ID (0-14=黒駒, 15-28=白駒)

    Returns:
        基本駒ID (0-14)

    Examples:
        >>> domain_to_base_piece(1)   # 黒歩 → 1
        1
        >>> domain_to_base_piece(15)  # 白歩 → 1
        1
        >>> domain_to_base_piece(22)  # 白王 → 8
        8
    """
    if is_white_domain(domain_piece):
        return domain_piece - DOMAIN_WHITE_OFFSET
    return domain_piece


@dataclass(frozen=True, slots=True)
class ColoredPiece:
    """手番と駒種類を組み合わせた型安全な駒表現．

    cshogi形式やdomain形式との変換を一元管理し，
    オフセット計算のバグを防止する．

    Attributes:
        turn: 駒の所有者(Turn.BLACK または Turn.WHITE)
        piece_id: 駒の種類(PieceId enum)

    Examples:
        >>> cp = ColoredPiece.from_cshogi(17)  # 白歩
        >>> cp.turn
        <Turn.WHITE: 1>
        >>> cp.piece_id
        <PieceId.FU: 1>
        >>> cp.to_cshogi()
        17
    """

    turn: Turn
    piece_id: PieceId

    # cshogi形式の定数(クラス変数として参照可能)
    CSHOGI_WHITE_OFFSET: ClassVar[int] = CSHOGI_WHITE_OFFSET
    CSHOGI_BLACK_MIN: ClassVar[int] = CSHOGI_BLACK_MIN
    CSHOGI_BLACK_MAX: ClassVar[int] = CSHOGI_BLACK_MAX
    CSHOGI_WHITE_MIN: ClassVar[int] = CSHOGI_WHITE_MIN
    CSHOGI_WHITE_MAX: ClassVar[int] = CSHOGI_WHITE_MAX

    # domain形式の定数
    DOMAIN_WHITE_OFFSET: ClassVar[int] = DOMAIN_WHITE_OFFSET
    DOMAIN_BLACK_MIN: ClassVar[int] = DOMAIN_BLACK_MIN
    DOMAIN_BLACK_MAX: ClassVar[int] = DOMAIN_BLACK_MAX
    DOMAIN_WHITE_MIN: ClassVar[int] = DOMAIN_WHITE_MIN
    DOMAIN_WHITE_MAX: ClassVar[int] = DOMAIN_WHITE_MAX

    @classmethod
    def from_cshogi(cls, cshogi_piece: int) -> ColoredPiece:
        """cshogi形式(0-30)からColoredPieceを生成．

        Args:
            cshogi_piece: cshogi駒ID (0=空, 1-14=黒駒, 17-30=白駒)

        Returns:
            ColoredPiece インスタンス

        Raises:
            ValueError: 無効な駒IDの場合 (15, 16, または範囲外)

        Examples:
            >>> ColoredPiece.from_cshogi(1)   # 黒歩
            ColoredPiece(turn=<Turn.BLACK: 0>, piece_id=<PieceId.FU: 1>)
            >>> ColoredPiece.from_cshogi(17)  # 白歩
            ColoredPiece(turn=<Turn.WHITE: 1>, piece_id=<PieceId.FU: 1>)
        """
        if cshogi_piece == 0:
            return cls(Turn.BLACK, PieceId.EMPTY)

        if CSHOGI_BLACK_MIN <= cshogi_piece <= CSHOGI_BLACK_MAX:
            # 黒駒: cshogi 1-14 → PieceId 1-14(同じ)
            return cls(Turn.BLACK, PieceId(cshogi_piece))

        if CSHOGI_WHITE_MIN <= cshogi_piece <= CSHOGI_WHITE_MAX:
            # 白駒: cshogi 17-30 → PieceId 1-14
            base_piece = cshogi_piece - CSHOGI_WHITE_OFFSET
            return cls(Turn.WHITE, PieceId(base_piece))

        msg = (
            f"無効なcshogi駒ID: {cshogi_piece}．"
            f"有効範囲: 0, {CSHOGI_BLACK_MIN}-{CSHOGI_BLACK_MAX}, "
            f"{CSHOGI_WHITE_MIN}-{CSHOGI_WHITE_MAX}"
        )
        raise ValueError(msg)

    @classmethod
    def from_domain(cls, domain_piece: int) -> ColoredPiece:
        """domain形式(0-28)からColoredPieceを生成．

        Args:
            domain_piece: domain駒ID (0=空, 1-14=黒駒, 15-28=白駒)

        Returns:
            ColoredPiece インスタンス

        Raises:
            ValueError: 無効な駒IDの場合 (範囲外)

        Examples:
            >>> ColoredPiece.from_domain(1)   # 黒歩
            ColoredPiece(turn=<Turn.BLACK: 0>, piece_id=<PieceId.FU: 1>)
            >>> ColoredPiece.from_domain(15)  # 白歩
            ColoredPiece(turn=<Turn.WHITE: 1>, piece_id=<PieceId.FU: 1>)
        """
        if domain_piece == 0:
            return cls(Turn.BLACK, PieceId.EMPTY)

        if 1 <= domain_piece <= DOMAIN_BLACK_MAX:
            # 黒駒: domain 1-14 → PieceId 1-14
            return cls(Turn.BLACK, PieceId(domain_piece))

        if DOMAIN_WHITE_MIN <= domain_piece <= DOMAIN_WHITE_MAX:
            # 白駒: domain 15-28 → PieceId 1-14
            base_piece = domain_piece - DOMAIN_WHITE_OFFSET
            return cls(Turn.WHITE, PieceId(base_piece))

        msg = (
            f"無効なdomain駒ID: {domain_piece}．"
            f"有効範囲: {DOMAIN_BLACK_MIN}-{DOMAIN_BLACK_MAX}, "
            f"{DOMAIN_WHITE_MIN}-{DOMAIN_WHITE_MAX}"
        )
        raise ValueError(msg)

    def to_cshogi(self) -> int:
        """cshogi形式(0-30)に変換．

        Returns:
            cshogi駒ID (0=空, 1-14=黒駒, 17-30=白駒)

        Examples:
            >>> ColoredPiece(Turn.BLACK, PieceId.FU).to_cshogi()
            1
            >>> ColoredPiece(Turn.WHITE, PieceId.FU).to_cshogi()
            17
        """
        if self.piece_id == PieceId.EMPTY:
            return 0
        base = int(self.piece_id)
        if self.turn == Turn.WHITE:
            return base + CSHOGI_WHITE_OFFSET
        return base

    def to_domain(self) -> int:
        """domain形式(0-28)に変換．

        Returns:
            domain駒ID (0=空, 1-14=黒駒, 15-28=白駒)

        Examples:
            >>> ColoredPiece(Turn.BLACK, PieceId.FU).to_domain()
            1
            >>> ColoredPiece(Turn.WHITE, PieceId.FU).to_domain()
            15
        """
        if self.piece_id == PieceId.EMPTY:
            return 0
        base = int(self.piece_id)
        if self.turn == Turn.WHITE:
            return base + DOMAIN_WHITE_OFFSET
        return base

    @property
    def is_black(self) -> bool:
        """先手の駒か判定．"""
        return self.turn == Turn.BLACK

    @property
    def is_white(self) -> bool:
        """後手の駒か判定．"""
        return self.turn == Turn.WHITE

    @property
    def is_empty(self) -> bool:
        """空のマスか判定．"""
        return self.piece_id == PieceId.EMPTY


def move16(move: int) -> int:
    """Convert move to 16-bit representation used in HCPE format.

    The 16-bit move format is part of the HCPE binary specification
    used for efficient storage in training data.

    Args:
        move: Full 32-bit move integer

    Returns:
        16-bit compact move representation

    Note:
        16-bit encoding:
        - Bits 0-6: destination square (0-80)
        - Bits 7-13: source square (0-80) or drop piece index (81+)
        - Bit 14: promotion flag
        Drops are identified by from_field >= 81, not by a dedicated flag bit.
    """
    return _move16(move)


def move_to(move: int) -> int:
    """Extract destination square from move.

    Args:
        move: Move integer

    Returns:
        Destination square index (0-80)
    """
    return _move_to(move)


def move_from(move: int) -> int:
    """指し手から移動元マスを取得する．

    通常の指し手の場合はマス番号(0-80)を返す．
    駒打ちの場合は move_is_drop() で判定し，
    move_drop_hand_piece() で駒種を取得すること．

    Args:
        move: 指し手整数値

    Returns:
        通常手: 移動元マス番号(0-80)
        駒打ち: 内部エンコード値(マス番号ではない)
    """
    return _move_from(move)


def move_to_usi(move: int) -> str:
    """Convert move to USI (Universal Shogi Interface) string format.

    Args:
        move: Move integer

    Returns:
        USI move string (e.g., \"7g7f\", \"P*5e\")
    """
    return _move_to_usi(move)


def move_is_drop(move: int) -> bool:
    """Check if move is a drop (placing a piece from hand).

    Args:
        move: Move integer

    Returns:
        True if move is a drop, False otherwise
    """
    return _move_is_drop(move)


def move_is_promotion(move: int) -> bool:
    """Check if move includes piece promotion.

    Args:
        move: Move integer

    Returns:
        True if move promotes the piece, False otherwise
    """
    return _move_is_promotion(move)


def move_drop_hand_piece(move: int) -> int:
    """駒打ちの駒種を取得する．

    Args:
        move: 駒打ちの指し手整数値

    Returns:
        打つ駒の種類

    Note:
        move_is_drop(move) が True の場合のみ呼び出すこと．
        非駒打ちの指し手では 0 を返すが，これは歩(HPAWN)と同値であり
        区別できない．

    Raises:
        AssertionError: move_is_drop(move) が False の場合(debug モード)
    """
    assert move_is_drop(move), (
        f"move_drop_hand_piece called on non-drop move: {move}"
    )
    return _move_drop_hand_piece(move)


class Board:
    """将棋の盤面を表すドメインモデル．

    maou_shogi (Rust PyO3) の PyBoard をラップし，
    PieceId体系への変換やPolars DataFrame出力などのドメインロジックを提供する．
    """

    _CSHOGI_TO_PIECEID: dict[int, int] = {
        # Black pieces (1-14)
        0: 0,  # EMPTY
        1: 1,  # BPAWN → FU
        2: 2,  # BLANCE → KY
        3: 3,  # BKNIGHT → KE
        4: 4,  # BSILVER → GI
        5: 6,  # BBISHOP → KA (角)
        6: 7,  # BROOK → HI (飛)
        7: 5,  # BGOLD → KI (金)
        8: 8,  # BKING → OU
        9: 9,  # BPROM_PAWN → TO
        10: 10,  # BPROM_LANCE → NKY
        11: 11,  # BPROM_KNIGHT → NKE
        12: 12,  # BPROM_SILVER → NGI
        13: 13,  # BPROM_BISHOP → UMA (馬)
        14: 14,  # BPROM_ROOK → RYU (龍)
        # White pieces (17-30)
        17: 15,  # WPAWN → FU + 14
        18: 16,  # WLANCE → KY + 14
        19: 17,  # WKNIGHT → KE + 14
        20: 18,  # WSILVER → GI + 14
        21: 20,  # WBISHOP → KA + 14 (角)
        22: 21,  # WROOK → HI + 14 (飛)
        23: 19,  # WGOLD → KI + 14 (金)
        24: 22,  # WKING → OU + 14
        25: 23,  # WPROM_PAWN → TO + 14
        26: 24,  # WPROM_LANCE → NKY + 14
        27: 25,  # WPROM_KNIGHT → NKE + 14
        28: 26,  # WPROM_SILVER → NGI + 14
        29: 27,  # WPROM_BISHOP → UMA + 14 (馬)
        30: 28,  # WPROM_ROOK → RYU + 14 (龍)
    }

    @staticmethod
    def cshogi_piece_to_piece_id(cshogi_piece: int) -> int:
        """Convert cshogi piece ID to domain PieceId enum value.

        cshogi uses BISHOP=5, ROOK=6, GOLD=7 with white offset +16.
        PieceId uses KI(金)=5, KA(角)=6, HI(飛)=7 with white offset +14.

        Args:
            cshogi_piece: cshogi piece ID (0-30)

        Returns:
            PieceId enum value (0-28)

        Examples:
            >>> Board.cshogi_piece_to_piece_id(0)  # EMPTY
            0
            >>> Board.cshogi_piece_to_piece_id(5)  # cshogi.BBISHOP -> PieceId.KA
            6
            >>> Board.cshogi_piece_to_piece_id(21)  # cshogi.WBISHOP
            20
        """
        return Board._CSHOGI_TO_PIECEID.get(cshogi_piece, 0)

    @staticmethod
    def _reorder_piece_planes_cshogi_to_pieceid(
        array: np.ndarray,
    ) -> None:
        """Reorder piece planes from cshogi ordering to PieceId ordering (in-place).

        cshogi planes: [FU, KY, KE, GI, BISHOP(角), ROOK(飛), GOLD(金), OU, ...]
        PieceId planes: [FU, KY, KE, GI, KI(金), KA(角), HI(飛), OU, ...]

        This reordering is necessary because cshogi uses standard piece names
        (BISHOP=角, ROOK=飛車, GOLD=金) in a different order than PieceId enum.

        Args:
            array: Piece planes array, shape (104, 9, 9) - modified in-place

        Note:
            Promoted pieces (UMA=馬, RYU=龍) don't need reordering as their
            relative positions are consistent between cshogi and PieceId.
        """
        temp = array.copy()
        # Black pieces reordering (indices 4-6)
        array[4] = temp[6]  # GOLD (cshogi) → KI (PieceId)
        array[5] = temp[
            4
        ]  # BISHOP (cshogi) → KA (PieceId) - 角
        array[6] = temp[5]  # ROOK (cshogi) → HI (PieceId) - 飛
        # PROM_BISHOP (馬) and PROM_ROOK (龍) need no reordering

        # White pieces reordering (same pattern, offset +14)
        array[18] = temp[20]  # GOLD (cshogi) → KI (PieceId)
        array[19] = temp[
            18
        ]  # BISHOP (cshogi) → KA (PieceId) - 角
        array[20] = temp[
            19
        ]  # ROOK (cshogi) → HI (PieceId) - 飛
        # PROM_BISHOP (馬) and PROM_ROOK (龍) need no reordering

    def __init__(self) -> None:
        """初期局面(平手)でBoardを生成する．"""
        self.board = _PyBoard()

    def __copy__(self) -> Board:
        """SFENを経由した安全なコピーを返す．

        内部の_PyBoardはシャローコピーで共有されるため，
        SFENによる再構築でディープコピーを保証する．

        Note:
            コピー後は undo_stack がリセットされるため，
            コピー先での pop_move() は使用不可．
        """
        new_board = Board()
        new_board.set_sfen(self.get_sfen())
        return new_board

    def set_turn(self, turn: Turn) -> None:
        """手番を設定する．

        Args:
            turn: 設定する手番
        """
        self.board.set_turn(turn.value)

    def get_turn(self) -> Turn:
        """現在の手番を取得する．

        Returns:
            現在の手番
        """
        return Turn(self.board.turn)

    def set_sfen(self, sfen: str) -> None:
        """SFEN文字列から局面を設定する．

        Args:
            sfen: SFEN形式の局面文字列

        Raises:
            ValueError: 不正なSFEN文字列の場合
        """
        self.board.set_sfen(sfen)

    def get_sfen(self) -> str:
        """現在の局面をSFEN文字列で取得する．

        Returns:
            SFEN形式の局面文字列
        """
        return self.board.sfen()

    def set_hcp(self, hcp: np.ndarray | bytes) -> None:
        """HCPデータから局面を設定する．

        Args:
            hcp: 32バイトのHCPデータ(numpy配列またはbytes)
        """
        if isinstance(hcp, np.ndarray):
            data = hcp.tobytes()
        else:
            data = hcp
        self.board.set_hcp(data)

    def to_hcp(self) -> bytes:
        """局面をHCPバイト列にエンコードする．

        Returns:
            32バイトのHCPデータ
        """
        return bytes(self.board.to_hcp())

    def get_legal_moves(self) -> Generator[int, None, None]:
        """現在の局面で合法手を列挙する．

        Yields:
            合法手のmove整数値(32-bit)
        """
        for move in self.board.legal_moves():
            yield move

    def get_move_from_move16(self, move16: int) -> int:
        """move16形式(16-bit)からフル move(32-bit)に変換する．

        Args:
            move16: 16-bit move形式の指し手

        Returns:
            32-bit フル move
        """
        return self.board.move_from_move16(move16)

    def move_from_usi(self, usi: str) -> int:
        """USI形式の文字列から move(32-bit)に変換する．

        Args:
            usi: USI形式の指し手文字列(例: "7g7f")

        Returns:
            32-bit フル move

        Raises:
            ValueError: 不正なUSI文字列の場合
        """
        return self.board.move_from_usi(usi)

    def push_move(self, move: int) -> None:
        """指し手を実行して局面を進める．

        Args:
            move: 実行する指し手(32-bit move整数値)

        Raises:
            ValueError: 不正な指し手の場合
        """
        self.board.push(move)

    def pop_move(self) -> None:
        """直前の指し手を取り消す．"""
        self.board.pop()

    def to_piece_planes(self, array: np.ndarray) -> None:
        """先手視点の駒特徴平面を配列に書き込む(in-place)．

        PieceId順にチャネルを並べ替え，座標系を転置する．

        Args:
            array: 書き込み先の配列，shape (104, 9, 9)，dtype float32
        """
        self.board.piece_planes(array)
        # Reorder channels to match PieceId ordering using centralized method
        Board._reorder_piece_planes_cshogi_to_pieceid(array)
        # Rust feature::piece_planes fills array in column-major order (col, row).
        # Transpose axes (1,2) to convert to row-major (row, col) matching
        # get_board_id_positions(). See docs/visualization/shogi-conventions.md.
        array[:] = np.transpose(array, (0, 2, 1))

    def to_piece_planes_rotate(self, array: np.ndarray) -> None:
        """後手視点(180度回転)の駒特徴平面を配列に書き込む(in-place)．

        PieceId順にチャネルを並べ替え，座標系を転置する．

        Args:
            array: 書き込み先の配列，shape (104, 9, 9)，dtype float32
        """
        self.board.piece_planes_rotate(array)
        # Reorder channels (same as to_piece_planes) using centralized method
        Board._reorder_piece_planes_cshogi_to_pieceid(array)
        # Rust feature::piece_planes_rotate fills array in column-major order (col, row).
        # Transpose axes (1,2) to convert to row-major (row, col) matching
        # get_board_id_positions(). See docs/visualization/shogi-conventions.md.
        array[:] = np.transpose(array, (0, 2, 1))

    def get_pieces_in_hand(self) -> tuple[list[int], list[int]]:
        """持ち駒を取得する．

        手番に関係なく常に(先手, 後手)の順で返す．
        各リストは歩，香車，桂馬，銀，金，角，飛車の順番．

        Returns:
            (先手の持ち駒, 後手の持ち駒) のタプル．
            各要素は駒種ごとの所持数リスト(長さ7)．

        Examples:
            >>> board.get_pieces_in_hand()
            ([0, 1, 0, 1, 0, 0, 0], [0, 0, 1, 0, 1, 0, 0])
        """
        return self.board.pieces_in_hand()

    def get_piece_at(self, square: int) -> int:
        """指定マスのcshogi駒IDを返す．

        Args:
            square: マス番号(column-major: col * 9 + row)

        Returns:
            cshogi駒ID(0-30)．駒がない場合は0．
        """
        return self.board.piece(square)

    def get_pieces(self) -> list[int]:
        """盤面の駒配列(81要素)を返す．

        cshogiの内部表現をそのまま返す．
        値はcshogi駒ID(0-30)で，column-major順に格納される．

        Returns:
            81要素のリスト(cshogi駒ID)
        """
        return self.board.pieces()

    def to_pretty_board(self) -> str:
        """盤面を人間が読める文字列で返す．

        Returns:
            盤面の文字列表現
        """
        return str(self.board)

    def hash(self) -> int:
        """Zobristハッシュ値を取得する．

        Returns:
            現在の局面のZobristハッシュ値
        """
        return self.board.zobrist_hash()

    def is_ok(self) -> bool:
        """Check if the board state is valid.

        Returns:
            bool: True if the board is in a valid state (両方の王が存在する等)
        """
        return self.board.is_ok()

    def get_board_id_positions(self) -> list[list[int]]:
        """Get board piece positions as 9x9 nested list.

        盤面の駒配置を[row][col]形式の二次元リストで返す．
        cshogiのcolumn-major配置(square = col * 9 + row)を
        Fortran orderでreshapeして[row][col]形式に変換する．

        Returns:
            9x9のPieceId二次元リスト([row][col]形式)

        Example:
            >>> board = Board()
            >>> positions = board.get_board_id_positions()
            >>> len(positions)
            9
            >>> len(positions[0])
            9
        """
        v_map = np.vectorize(
            Board.cshogi_piece_to_piece_id,
            otypes=[np.uint8],
        )
        positions = v_map(
            np.array(
                self.board.pieces(),
                dtype=np.uint8,
            )
        ).reshape((9, 9), order="F")
        return positions.tolist()

    def get_board_id_positions_df(self) -> "pl.DataFrame":
        """Get board piece positions as 1-row Polars DataFrame．

        盤面の駒配置をPolars DataFrameで取得する．
        9x9のネストされたリストとして返す．

        Returns:
            pl.DataFrame: boardIdPositions列を持つ1行のDataFrame

        Example:
            >>> board = Board()
            >>> df = board.get_board_id_positions_df()
            >>> len(df)
            1
            >>> df.schema
            {'boardIdPositions': List(List(UInt8))}
        """
        if not _POLARS_AVAILABLE:
            raise ImportError(
                "polars is not installed. Install with: uv add polars"
            )

        positions_list = self.get_board_id_positions()

        # Use pre-imported polars for performance
        return _pl.DataFrame(
            {"boardIdPositions": [positions_list]},
            schema={
                "boardIdPositions": _pl.List(
                    _pl.List(_pl.UInt8)
                )
            },
        )

    def get_hcp_df(self) -> "pl.DataFrame":
        """Get HuffmanCodedPos as 1-row Polars DataFrame．

        HuffmanCodedPos形式の局面データをPolars DataFrameで取得する．
        32バイトのバイナリデータとして返す．

        Returns:
            pl.DataFrame: hcp列を持つ1行のDataFrame

        Example:
            >>> board = Board()
            >>> board.set_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
            >>> df = board.get_hcp_df()
            >>> len(df)
            1
            >>> df.schema
            {'hcp': Binary}
        """
        if not _POLARS_AVAILABLE:
            raise ImportError(
                "polars is not installed. Install with: uv add polars"
            )

        # Get HCP data from board
        hcp_bytes = self.to_hcp()

        # Use pre-imported polars for performance
        return _pl.DataFrame(
            {"hcp": [hcp_bytes]}, schema={"hcp": _pl.Binary()}
        )

    def get_piece_planes_df(self) -> "pl.DataFrame":
        """Get piece feature planes as 1-row Polars DataFrame．

        駒の特徴平面をPolars DataFrameで取得する．
        104x9x9のネストされたリストとして返す．

        Returns:
            pl.DataFrame: piecePlanes列を持つ1行のDataFrame

        Example:
            >>> board = Board()
            >>> df = board.get_piece_planes_df()
            >>> len(df)
            1
            >>> df.schema
            {'piecePlanes': List(List(List(Float32)))}
        """
        if not _POLARS_AVAILABLE:
            raise ImportError(
                "polars is not installed. Install with: uv add polars"
            )

        # Get piece planes from board (reorder + transpose via to_piece_planes)
        planes = np.zeros(
            (FEATURES_NUM, 9, 9), dtype=np.float32
        )
        self.to_piece_planes(planes)
        planes_list = planes.tolist()

        # Use pre-imported polars for performance
        return _pl.DataFrame(
            {"piecePlanes": [planes_list]},
            schema={
                "piecePlanes": _pl.List(
                    _pl.List(_pl.List(_pl.Float32))
                )
            },
        )

    def get_piece_planes_rotate_df(self) -> "pl.DataFrame":
        """Get rotated piece feature planes as 1-row Polars DataFrame．

        回転された駒の特徴平面をPolars DataFrameで取得する．
        後手視点の104x9x9のネストされたリストとして返す．

        Returns:
            pl.DataFrame: piecePlanes列を持つ1行のDataFrame

        Example:
            >>> board = Board()
            >>> df = board.get_piece_planes_rotate_df()
            >>> len(df)
            1
            >>> df.schema
            {'piecePlanes': List(List(List(Float32)))}
        """
        if not _POLARS_AVAILABLE:
            raise ImportError(
                "polars is not installed. Install with: uv add polars"
            )

        # Get rotated piece planes from board (reorder + transpose via to_piece_planes_rotate)
        planes = np.zeros(
            (FEATURES_NUM, 9, 9), dtype=np.float32
        )
        self.to_piece_planes_rotate(planes)
        planes_list = planes.tolist()

        # Use pre-imported polars for performance
        return _pl.DataFrame(
            {"piecePlanes": [planes_list]},
            schema={
                "piecePlanes": _pl.List(
                    _pl.List(_pl.List(_pl.Float32))
                )
            },
        )
