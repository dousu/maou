from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import TYPE_CHECKING, ClassVar

import cshogi
import numpy as np

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

# Verify domain constants match cshogi (catch upstream changes)
assert (
    cshogi.MAX_PIECES_IN_HAND == MAX_PIECES_IN_HAND  # type: ignore
), (
    f"cshogi constant changed: {cshogi.MAX_PIECES_IN_HAND} != {MAX_PIECES_IN_HAND}"
)  # type: ignore
assert (
    len(cshogi.PIECE_TYPES) == PIECE_TYPES  # type: ignore
), (
    f"cshogi constant changed: {len(cshogi.PIECE_TYPES)} != {PIECE_TYPES}"
)  # type: ignore

# MAX_PIECES_IN_HANDの構成
# 歩18，香車4，桂馬4，銀4，金4，角2，飛車2

# 104
FEATURES_NUM = PIECE_TYPES * 2 + sum(MAX_PIECES_IN_HAND) * 2


class Turn(IntEnum):
    BLACK = cshogi.BLACK  # type: ignore
    WHITE = cshogi.WHITE  # type: ignore


class Result(IntEnum):
    BLACK_WIN = cshogi.BLACK_WIN  # type: ignore
    WHITE_WIN = cshogi.WHITE_WIN  # type: ignore
    DRAW = cshogi.DRAW  # type: ignore


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
        turn: 駒の所有者（Turn.BLACK または Turn.WHITE）
        piece_id: 駒の種類（PieceId enum）

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

    # cshogi形式の定数（クラス変数として参照可能）
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
            # 黒駒: cshogi 1-14 → PieceId 1-14（同じ）
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

        if DOMAIN_BLACK_MIN < domain_piece <= DOMAIN_BLACK_MAX:
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

    This is a thin wrapper around cshogi.move16(). The 16-bit move format
    is part of the HCPE binary specification used for efficient storage
    in training data.

    Args:
        move: Full 32-bit move integer from cshogi

    Returns:
        16-bit compact move representation

    Note:
        If replacing cshogi, implement move16 encoding per HCPE spec:
        - Bits 0-6: destination square (0-80)
        - Bits 7-13: source square or drop piece type
        - Bit 14: promotion flag
        - Bit 15: drop flag
    """
    return cshogi.move16(move)  # type: ignore


def move_to(move: int) -> int:
    """Extract destination square from move.

    Args:
        move: Move integer from cshogi

    Returns:
        Destination square index (0-80)
    """
    return cshogi.move_to(move)  # type: ignore


def move_from(move: int) -> int:
    """Extract source square from move.

    Args:
        move: Move integer from cshogi

    Returns:
        Source square index (0-80) for normal moves
    """
    return cshogi.move_from(move)  # type: ignore


def move_to_usi(move: int) -> str:
    """Convert move to USI (Universal Shogi Interface) string format.

    Args:
        move: Move integer from cshogi

    Returns:
        USI move string (e.g., \"7g7f\", \"P*5e\")
    """
    return cshogi.move_to_usi(move)  # type: ignore


def move_is_drop(move: int) -> bool:
    """Check if move is a drop (placing a piece from hand).

    Args:
        move: Move integer from cshogi

    Returns:
        True if move is a drop, False otherwise
    """
    return cshogi.move_is_drop(move)  # type: ignore


def move_is_promotion(move: int) -> bool:
    """Check if move includes piece promotion.

    Args:
        move: Move integer from cshogi

    Returns:
        True if move promotes the piece, False otherwise
    """
    return cshogi.move_is_promotion(move)  # type: ignore


def move_drop_hand_piece(move: int) -> int:
    """Get which hand piece type is being dropped.

    Args:
        move: Drop move integer from cshogi

    Returns:
        Piece type being dropped (only valid for drop moves)
    """
    return cshogi.move_drop_hand_piece(move)  # type: ignore


class Board:
    @staticmethod
    def _cshogi_piece_to_piece_id(cshogi_piece: int) -> int:
        """Convert cshogi piece ID to domain PieceId enum value.

        cshogi uses BISHOP=5, ROOK=6, GOLD=7 with white offset +16.
        PieceId uses KI(金)=5, KA(角)=6, HI(飛)=7 with white offset +14.

        Args:
            cshogi_piece: cshogi piece ID (0-30)

        Returns:
            PieceId enum value (0-28)

        Examples:
            >>> Board._cshogi_piece_to_piece_id(0)  # EMPTY
            0
            >>> Board._cshogi_piece_to_piece_id(5)  # cshogi.BBISHOP -> PieceId.KA
            6
            >>> Board._cshogi_piece_to_piece_id(21)  # cshogi.WBISHOP
            20
        """
        mapping = {
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
        return mapping.get(cshogi_piece, 0)

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
        array[12] = temp[
            12
        ]  # PROM_BISHOP (cshogi) → UMA (PieceId) - 馬
        array[13] = temp[
            13
        ]  # PROM_ROOK (cshogi) → RYU (PieceId) - 龍

        # White pieces reordering (same pattern, offset +14)
        array[18] = temp[20]  # GOLD (cshogi) → KI (PieceId)
        array[19] = temp[
            18
        ]  # BISHOP (cshogi) → KA (PieceId) - 角
        array[20] = temp[
            19
        ]  # ROOK (cshogi) → HI (PieceId) - 飛
        array[26] = temp[
            26
        ]  # PROM_BISHOP (cshogi) → UMA (PieceId) - 馬
        array[27] = temp[
            27
        ]  # PROM_ROOK (cshogi) → RYU (PieceId) - 龍

    def __init__(self) -> None:
        self.board = cshogi.Board()  # type: ignore

    def set_turn(self, turn: Turn) -> None:
        self.board.turn = turn.value

    def get_turn(self) -> Turn:
        return Turn(self.board.turn)

    def set_sfen(self, sfen: str) -> None:
        self.board.set_sfen(sfen)

    def get_sfen(self) -> str:
        return self.board.sfen()

    def set_hcp(self, hcp: np.ndarray) -> None:
        self.board.set_hcp(hcp)

    def to_hcp(self, array: np.ndarray) -> None:
        self.board.to_hcp(array)

    def get_legal_moves(self) -> Generator[int, None, None]:
        for move in self.board.legal_moves:
            yield move

    def get_move_from_move16(self, move16: int) -> int:
        return self.board.move_from_move16(move16)

    def push_move(self, move: int) -> None:
        self.board.push(move)

    def to_piece_planes(self, array: np.ndarray) -> None:
        self.board.piece_planes(array)
        # Reorder channels to match PieceId ordering using centralized method
        Board._reorder_piece_planes_cshogi_to_pieceid(array)
        # Transpose to match board_id_positions coordinate system
        array[:] = np.transpose(array, (0, 2, 1))

    def to_piece_planes_rotate(self, array: np.ndarray) -> None:
        self.board.piece_planes_rotate(array)
        # Reorder channels (same as to_piece_planes) using centralized method
        Board._reorder_piece_planes_cshogi_to_pieceid(array)
        # Transpose to match board_id_positions coordinate system
        array[:] = np.transpose(array, (0, 2, 1))

    def get_pieces_in_hand(self) -> tuple[list[int], list[int]]:
        """手番関係なく常に(先手, 後手)の順にtupleにはいっている
        歩，香車，桂馬，銀，金，角，飛車の順番
        例: ([0, 1, 0, 1, 0, 0, 0], [0, 0, 1, 0, 1, 0, 0])
        """
        return self.board.pieces_in_hand

    def get_pieces(self) -> list[int]:
        """盤面の駒配列(81要素)を返す．

        cshogiの内部表現をそのまま返す．
        値はcshogi駒ID(0-30)で，column-major順に格納される．

        Returns:
            81要素のリスト(cshogi駒ID)
        """
        return self.board.pieces

    def to_pretty_board(self) -> str:
        return str(self.board)

    def hash(self) -> int:
        return self.board.zobrist_hash()

    def is_ok(self) -> bool:
        """Check if the board state is valid.

        Returns:
            bool: True if the board is in a valid state (両方の王が存在する等)
        """
        return self.board.is_ok()

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
                "polars is not installed. Install with: poetry add polars"
            )

        # Map cshogi piece IDs to PieceId enum values using centralized conversion
        v_map = np.vectorize(
            Board._cshogi_piece_to_piece_id,
            otypes=[np.uint8],
        )
        positions = v_map(
            np.array(
                self.board.pieces,
                dtype=np.uint8,
            )
        ).reshape((9, 9), order="F")
        positions_list = positions.tolist()  # Fast conversion

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
                "polars is not installed. Install with: poetry add polars"
            )

        # Get HCP data from cshogi board
        hcp_array = np.empty(1, dtype=cshogi.HuffmanCodedPos)  # type: ignore
        self.board.to_hcp(hcp_array)
        hcp_bytes = hcp_array.tobytes()  # Convert to bytes

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
                "polars is not installed. Install with: poetry add polars"
            )

        # Get piece planes from cshogi board
        planes = np.empty(
            (FEATURES_NUM, 9, 9), dtype=np.float32
        )
        planes.fill(0)
        self.board.piece_planes(planes)
        # Reorder channels to match PieceId ordering using centralized method
        Board._reorder_piece_planes_cshogi_to_pieceid(planes)
        # Transpose to match board_id_positions coordinate system
        planes = np.transpose(planes, (0, 2, 1))
        planes_list = planes.tolist()  # Fast conversion

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
                "polars is not installed. Install with: poetry add polars"
            )

        # Get rotated piece planes from cshogi board
        planes = np.empty(
            (FEATURES_NUM, 9, 9), dtype=np.float32
        )
        planes.fill(0)
        self.board.piece_planes_rotate(planes)
        # Reorder channels to match PieceId ordering using centralized method
        Board._reorder_piece_planes_cshogi_to_pieceid(planes)
        # Transpose to match get_board_id_positions_df coordinate system
        planes = np.transpose(planes, (0, 2, 1))
        planes_list = planes.tolist()  # Fast conversion

        # Use pre-imported polars for performance
        return _pl.DataFrame(
            {"piecePlanes": [planes_list]},
            schema={
                "piecePlanes": _pl.List(
                    _pl.List(_pl.List(_pl.Float32))
                )
            },
        )
