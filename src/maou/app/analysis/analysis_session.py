"""棋譜解析 GUI のセッション基盤ユースケース．

棋譜 (CSA / KIF) を per-ply の盤面スナップショット列 (plain data) に展開し，
`maou analyze-game` の JSON レポートとの整合検証を提供する．

スナップショットは PyO3 オブジェクト (Board 等) を保持しない plain data と
する — Gradio の ``gr.State`` はセッションごとに初期値を deepcopy するため
(docs/design/game-analysis/gui.md §11)．
"""

from dataclasses import dataclass
from typing import Any

from maou.app.analysis.game_analyzer import (
    decode_kifu_bytes,
    parse_single_game_record,
)
from maou.domain.board.shogi import (
    Board,
    Turn,
    move_drop_hand_piece,
    move_from,
    move_is_drop,
    move_to,
    move_to_usi,
)


@dataclass(frozen=True)
class MoveInfo:
    """1 手分の指し手情報 (plain data)．

    Attributes:
        usi: USI 表記 (例: "7g7f", "P*5e")．
        from_square: 移動元マス (0-80, column-major)．駒打ちは None．
        to_square: 移動先マス (0-80, column-major)．
        is_drop: 駒打ちかどうか．
        drop_piece_type: 駒打ちの駒種 (0=歩, 1=香, ...)．通常手は None．
        piece_id: 動かした駒の domain PieceId (移動前の駒種．
            駒打ちは打った駒の PieceId)．
    """

    usi: str
    from_square: int | None
    to_square: int
    is_drop: bool
    drop_piece_type: int | None
    piece_id: int


@dataclass(frozen=True)
class PositionSnapshot:
    """1 局面分の盤面スナップショット (plain data)．

    Attributes:
        ply: 局面番号 (0 = 初期局面, k = k 手目を指した後)．
        sfen: 局面の SFEN．
        board_id_positions: 9x9 の駒配置 ([row][col] 形式, domain PieceId)．
        pieces_in_hand: 持ち駒配列 (14 要素: 先手 7 種 + 後手 7 種)．
        turn: 手番 ("b" / "w")．
        last_move: この局面に至った指し手 (ply=0 は None)．
    """

    ply: int
    sfen: str
    board_id_positions: list[list[int]]
    pieces_in_hand: list[int]
    turn: str
    last_move: MoveInfo | None


@dataclass(frozen=True)
class GameDocument:
    """1 局の棋譜 (本譜) の plain data 表現．

    Attributes:
        input_format: 棋譜形式 ("csa" / "kif")．
        names: 対局者名 [先手, 後手]．
        ratings: レーティング [先手, 後手]．
        win: 勝敗 (0=引分, 1=先手勝ち, 2=後手勝ち, 不明は None)．
        endgame: 終局理由 (例: "%TORYO")．
        moves_usi: 本譜の指し手 (USI) 列．
        times: 消費時間 (秒) 列 (moves と長さ不一致があり得る)．
        scores: 棋譜記載の評価値列．
        comments: 棋譜記載のコメント列．
        snapshots: 局面スナップショット列 (長さ n_moves + 1)．
    """

    input_format: str
    names: list[str | None]
    ratings: list[float]
    win: int | None
    endgame: str | None
    moves_usi: list[str]
    times: list[int]
    scores: list[int]
    comments: list[str]
    snapshots: list[PositionSnapshot]

    @property
    def n_moves(self) -> int:
        """本譜の手数．"""
        return len(self.moves_usi)


def _turn_char(turn: Turn) -> str:
    """手番を SFEN の手番文字 ("b"/"w") にする．"""
    return "b" if turn == Turn.BLACK else "w"


def _move_info(board: Board, move: int) -> MoveInfo:
    """指し手 (32-bit move) を適用前盤面の文脈で MoveInfo にする．

    Args:
        board: 指し手適用前の盤面．
        move: 32-bit move 整数値．

    Returns:
        MoveInfo (plain data)．
    """
    usi = move_to_usi(move)
    if move_is_drop(move):
        drop_type = move_drop_hand_piece(move)
        # 持ち駒インデックス (0=歩...6=飛) は domain PieceId の 1-7 に対応
        return MoveInfo(
            usi=usi,
            from_square=None,
            to_square=move_to(move),
            is_drop=True,
            drop_piece_type=drop_type,
            piece_id=drop_type + 1,
        )
    from_sq = move_from(move)
    piece_id = Board.raw_piece_to_piece_id(
        board.get_piece_at(from_sq)
    )
    return MoveInfo(
        usi=usi,
        from_square=from_sq,
        to_square=move_to(move),
        is_drop=False,
        drop_piece_type=None,
        piece_id=piece_id,
    )


def _snapshot(
    board: Board, ply: int, last_move: MoveInfo | None
) -> PositionSnapshot:
    """現在の盤面から PositionSnapshot を作る．"""
    hand_black, hand_white = board.get_pieces_in_hand()
    return PositionSnapshot(
        ply=ply,
        sfen=board.get_sfen(),
        board_id_positions=board.get_board_id_positions(),
        pieces_in_hand=[int(c) for c in hand_black]
        + [int(c) for c in hand_white],
        turn=_turn_char(board.get_turn()),
        last_move=last_move,
    )


def load_game(data: bytes, input_format: str) -> GameDocument:
    """棋譜ファイルの bytes から GameDocument を構築する．

    Args:
        data: 棋譜ファイルの生バイト列 (UTF-8 先行 → cp932 でデコード)．
        input_format: 棋譜形式 ("csa" / "kif")．

    Returns:
        per-ply スナップショット込みの GameDocument．

    Raises:
        ValueError: パース失敗，複数局の CSA，指し手ゼロの棋譜，
            または棋譜中の不正手．
    """
    content = decode_kifu_bytes(data)
    record = parse_single_game_record(content, input_format)
    moves: list[int] = list(record.moves)
    if not moves:
        raise ValueError("棋譜に指し手がありません")

    board = Board()
    board.set_sfen(record.sfen)
    snapshots: list[PositionSnapshot] = [
        _snapshot(board, 0, None)
    ]
    moves_usi: list[str] = []
    for i, move in enumerate(moves):
        info = _move_info(board, move)
        try:
            board.push_move(move)
        except ValueError as e:
            raise ValueError(
                f"{i + 1} 手目 {info.usi} を適用できません: {e}"
            ) from e
        moves_usi.append(info.usi)
        snapshots.append(_snapshot(board, i + 1, info))

    return GameDocument(
        input_format=input_format,
        names=list(record.names),
        ratings=list(record.ratings),
        win=record.win,
        endgame=record.endgame,
        moves_usi=moves_usi,
        times=list(record.times),
        scores=list(record.scores),
        comments=list(record.comments),
        snapshots=snapshots,
    )


def validate_report(
    document: GameDocument, report: dict[str, Any]
) -> None:
    """analyze-game JSON レポートと棋譜の整合を検証する．

    検証項目 (不整合は ValueError):

    - ``positions`` の件数が棋譜の手数と一致すること
    - 各 ``positions[i].played_move`` が本譜の指し手と一致すること
    - 各 ``positions[i].sfen`` が対応する局面の SFEN と一致すること

    Args:
        document: 突き合わせる棋譜．
        report: analyze-game の JSON レポート (dict)．

    Raises:
        ValueError: レポートが棋譜と対応しない場合．
    """
    positions = report.get("positions")
    if not isinstance(positions, list):
        raise ValueError(
            "レポートに positions がありません "
            "(analyze-game の JSON 出力を指定してください)"
        )
    if len(positions) != document.n_moves:
        raise ValueError(
            f"レポートの局面数 {len(positions)} が棋譜の手数 "
            f"{document.n_moves} と一致しません"
        )
    for i, pos in enumerate(positions):
        played = pos.get("played_move")
        if played != document.moves_usi[i]:
            raise ValueError(
                f"{i + 1} 手目の指し手が一致しません: "
                f"棋譜 {document.moves_usi[i]} / レポート {played}"
            )
        sfen = pos.get("sfen")
        if (
            sfen is not None
            and sfen != document.snapshots[i].sfen
        ):
            raise ValueError(
                f"{i + 1} 手目の直前局面 SFEN が一致しません "
                f"(棋譜と別の対局のレポートの可能性があります)"
            )
