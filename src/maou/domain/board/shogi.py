from __future__ import annotations

from collections.abc import Generator
from enum import IntEnum, auto

import numpy as np

from maou._rust.maou_shogi import PyBoard as _PyBoard
from maou._rust.maou_shogi import move16 as _move16
from maou._rust.maou_shogi import (
    move_drop_hand_piece as _move_drop_hand_piece,
)
from maou._rust.maou_shogi import move_from as _move_from
from maou._rust.maou_shogi import move_is_drop as _move_is_drop
from maou._rust.maou_shogi import move_to as _move_to
from maou._rust.maou_shogi import move_to_usi as _move_to_usi

# Domain-level constants
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


class Turn(IntEnum):
    BLACK = 0
    WHITE = 1


class Result(IntEnum):
    """対局結果．

    HCPE の gameResult 列および GameRecord.win と同じ規約
    (0=引き分け, 1=先手勝ち, 2=後手勝ち)．
    旧定義 (BLACK_WIN=0, WHITE_WIN=1, DRAW=2) はこの規約とずれており，
    make_result_value が HCPE の gameResult を誤読して value head の
    教師値が全て取り違っていた．
    """

    DRAW = 0
    BLACK_WIN = 1
    WHITE_WIN = 2


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


# ============================================================================
# 駒 ID 規約の定数
# ============================================================================
#
# Rust エンジン (maou_shogi) の raw 駒 ID と，正規化後の domain PieceId で
# 白駒オフセットが異なる．マジックナンバーを排除するため以下を使用する．
#
#   raw 形式 (board.pieces() の生値): 白駒 = 黒駒 + 16  (1-14 → 17-30)
#   domain 形式 (PieceId):            白駒 = 黒駒 + 14  (0-14 → 15-28)
#

# domain PieceId (正規化後) の定数
DOMAIN_WHITE_OFFSET: int = 14
"""domain形式での白駒オフセット．白駒ID = 黒駒ID + 14．"""

DOMAIN_WHITE_MIN: int = 15
DOMAIN_WHITE_MAX: int = 28

# raw 駒 ID (Rust エンジン board.pieces() の生値, 0-30) → domain PieceId (0-28) の
# 変換テーブル (単一の真実)．raw は 金(7)/角(5)/飛(6) の順で白駒 +16，
# domain PieceId は 金(5)/角(6)/飛(7) の順で白駒 +14．15,16 は raw の未使用ギャップ．
RAW_PIECE_TO_PIECEID: np.ndarray = np.array(
    [
        0,  # 0: EMPTY
        1,
        2,
        3,
        4,  # 1-4: 歩香桂銀 → FU/KY/KE/GI
        6,
        7,
        5,  # 5-7: 角飛金 → KA/HI/KI
        8,  # 8: 玉 → OU
        9,
        10,
        11,
        12,
        13,
        14,  # 9-14: と成香成桂成銀馬龍
        0,
        0,  # 15,16: 未使用
        15,
        16,
        17,
        18,  # 17-20: 白歩香桂銀
        20,
        21,
        19,  # 21-23: 白角飛金
        22,  # 24: 白玉
        23,
        24,
        25,
        26,
        27,
        28,  # 25-30: 白成駒
    ],
    dtype=np.uint8,
)


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


def move_drop_hand_piece(move: int) -> int:
    """駒打ちの駒種を取得する．

    Args:
        move: 駒打ちの指し手整数値

    Returns:
        打つ駒の種類

    Raises:
        ValueError: move_is_drop(move) が False の場合
    """
    if not move_is_drop(move):
        msg = f"move_drop_hand_piece called on non-drop move: {move}"
        raise ValueError(msg)
    return _move_drop_hand_piece(move)


class Board:
    """将棋の盤面を表すドメインモデル．

    maou_shogi (Rust PyO3) の PyBoard をラップし，
    PieceId体系への変換やPolars DataFrame出力などのドメインロジックを提供する．
    """

    @staticmethod
    def raw_piece_to_piece_id(raw_piece: int) -> int:
        """raw 駒 ID (Rust エンジン board.pieces() の生値) を domain PieceId に変換する.

        raw は 角(5)/飛(6)/金(7)・白駒 +16，domain PieceId は 金(5)/角(6)/飛(7)・
        白駒 +14．変換表は module-level の RAW_PIECE_TO_PIECEID (単一の真実)．

        Args:
            raw_piece: raw 駒 ID (0-30)

        Returns:
            domain PieceId enum value (0-28)

        Examples:
            >>> Board.raw_piece_to_piece_id(0)  # EMPTY
            0
            >>> Board.raw_piece_to_piece_id(5)  # raw 角 -> PieceId.KA
            6
            >>> Board.raw_piece_to_piece_id(21)  # raw 白角
            20
        """
        if 0 <= raw_piece <= 30:
            return int(RAW_PIECE_TO_PIECEID[raw_piece])
        return 0

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
        """直前の指し手を取り消す．

        Raises:
            IndexError: 取り消す指し手がない場合(コピー後の盤面を含む)
        """
        self.board.pop()

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
        """指定マスのraw駒IDを返す．

        Args:
            square: マス番号(column-major: col * 9 + row)

        Returns:
            raw駒ID(0-30，Rustエンジン内部表現)．駒がない場合は0．
        """
        return self.board.piece(square)

    def get_pieces(self) -> list[int]:
        """盤面の駒配列(81要素)を返す．

        Rustエンジンの内部表現をそのまま返す．
        値はraw駒ID(0-30，白駒=黒駒+16)で，column-major順に格納される．
        domain PieceIdへの変換はRAW_PIECE_TO_PIECEIDを使う．

        Returns:
            81要素のリスト(raw駒ID)
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

    def get_board_id_positions(self) -> list[list[int]]:
        """Get board piece positions as 9x9 nested list.

        盤面の駒配置を[row][col]形式の二次元リストで返す (手番正規化なし)．
        Rust エンジンの column-major 配置 (square = col * 9 + row) を
        Fortran orderでreshapeして[row][col]形式に変換する．raw 駒 ID →
        domain PieceId 変換は RAW_PIECE_TO_PIECEID の fancy indexing で行う．

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
        raw = np.array(self.board.pieces(), dtype=np.uint8)
        positions = RAW_PIECE_TO_PIECEID[raw].reshape(
            (9, 9), order="F"
        )
        return positions.tolist()
