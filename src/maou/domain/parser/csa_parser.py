from datetime import datetime
from typing import Any

from maou._rust.maou_shogi import GameRecord, parse_csa_str
from maou.domain.parser import parser


class CSAParser(parser.Parser):
    """Parser for CSA format Shogi game records.

    CSA (Computer Shogi Association) is a standard format for representing
    Shogi games. This parser uses the Rust backend (maou_shogi::kifu) —
    a complete in-house implementation with no external dependency.

    **Compatibility Note**
    The Rust parser is parity-verified against the previous cshogi-based
    implementation (rust/maou_shogi/tests/kifu_parity.rs): move integers
    are bit-exact cshogi-compatible 32-bit encodings, and win/endgame/
    scores/comments semantics are preserved.
    """

    record: GameRecord

    def parse(self, content: str) -> None:
        """CSAの棋譜文字列をパースして読み取れる状態にする．

        複数対局を含む文字列は先頭の 1 局のみを対象とする (従来互換)．
        """
        self.record = parse_csa_str(content)[0]
        self._var_info = dict(self.record.var_info)

    def init_pos_sfen(self) -> str:
        """Get initial board position in SFEN notation.

        Returns:
            Initial position as SFEN string
        """
        return self.record.sfen

    def endgame(self) -> str:
        """Get game termination reason.

        Returns:
            Endgame reason (e.g. '%TORYO')，未終局なら空文字列
        """
        return self.record.endgame or ""

    def winner(self) -> int:
        """Get game result.

        Returns:
            0=引き分け，1=先手勝ち，2=後手勝ち (cshogi 互換値)
        """
        win = self.record.win
        assert win is not None  # CSA では常に決定される
        return win

    def ratings(self) -> list[int]:
        """Get player ratings.

        Returns:
            [先手, 後手] のレーティング ('black_rate:/'white_rate: 行由来，
            無指定は 0)
        """
        return self.record.ratings

    def moves(self) -> list[int]:
        """Get sequence of moves in the game.

        Returns:
            List of moves encoded as cshogi-compatible 32-bit integers
        """
        return self.record.moves

    def scores(self) -> list[int]:
        """Get evaluation scores for each position.

        Returns:
            List of position evaluation scores ('** コメント由来，
            moves と同長)
        """
        return self.record.scores

    def comments(self) -> list[str]:
        """Get comments for each move.

        Returns:
            List of move comments (moves と同長，無い手は空文字列)
        """
        return self.record.comments

    def _start_date(self) -> Any:
        """$START_TIME から日付を取得する (無ければ None)．"""
        try:
            datetime_str = self._var_info["START_TIME"]
        except KeyError:
            return None
        date_obj = datetime.strptime(
            datetime_str, "%Y/%m/%d %H:%M:%S"
        )
        return date_obj.date()

    def clustering_key_value(self) -> Any:
        """Get clustering key based on game start date.

        Returns:
            Date object for clustering, or None if START_TIME not available
        """
        return self._start_date()

    def partitioning_key_value(self) -> Any:
        """Get partitioning key based on game start date.

        Returns:
            Date object for partitioning, or None if START_TIME not available
        """
        return self._start_date()
