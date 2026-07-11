from datetime import datetime
from typing import Any, Optional

from maou._rust.maou_shogi import GameRecord, parse_kif_str
from maou.domain.parser import parser


class KifParser(parser.Parser):
    """Parser for KIF format Shogi game records.

    KIF is a human-readable format for Shogi games. Note that this format
    has limited information compared to CSA format (no ratings etc.).
    This parser uses the Rust backend (maou_shogi::kifu) — a complete
    in-house implementation with no external dependency.

    **Compatibility Note**
    The Rust parser is parity-verified against the previous cshogi-based
    implementation (rust/maou_shogi/tests/kifu_parity.rs)．従来との差分:

    - ``scores()`` は空リストでなく moves と同長の 0 列を返す (HCPE 変換が
      KIF で 1 行も出力されなかった不具合の解消)
    - ``comments()`` は moves と同長に整列される (無い手は空文字列)
    - ``開始日時`` ヘッダから clustering/partitioning キーを導出する
    - 「不成」の指し手を正しく解釈する (従来は黙って欠落し盤面が壊れた)
    - BOD (局面図) 初期局面は明示的にエラーになる (従来は黙って平手扱い)
    """

    record: GameRecord

    def parse(self, content: str) -> None:
        """KIF の棋譜文字列 (UTF-8) をパースして読み取れる状態にする．

        Shift_JIS (.kif) ファイルは呼び出し側でデコードして渡すこと．
        """
        self.record = parse_kif_str(content)
        self._var_info = dict(self.record.var_info)

    def init_pos_sfen(self) -> str:
        """Get initial board position in SFEN notation.

        Returns:
            Initial position as SFEN string (手合割ヘッダ由来，既定は平手)
        """
        return self.record.sfen

    def endgame(self) -> str:
        """Get game termination reason.

        Returns:
            Endgame reason ('%TORYO' 等)．「まで」行が無い場合は None
        """
        return self.record.endgame  # type: ignore[return-value]

    def winner(self) -> int:
        """Get game result.

        Returns:
            0=引き分け，1=先手勝ち，2=後手勝ち (cshogi 互換値)．
            「まで」行が無い場合は None
        """
        return self.record.win  # type: ignore[return-value]

    def ratings(self) -> list[int]:
        """Get player ratings.

        Returns:
            Empty list as KIF format doesn't contain rating information
        """
        return []

    def moves(self) -> list[int]:
        """Get sequence of moves in the game.

        Returns:
            List of moves encoded as cshogi-compatible 32-bit integers
        """
        return self.record.moves

    def scores(self) -> list[int]:
        """Get evaluation scores for each position.

        Returns:
            moves と同長の 0 列 (KIF には評価値情報が無いため)
        """
        return self.record.scores

    def comments(self) -> list[str]:
        """Get comments for each move.

        Returns:
            List of move comments (moves と同長，無い手は空文字列)
        """
        return self.record.comments

    def _start_date(self) -> Optional[Any]:
        """開始日時ヘッダから日付を取得する (無ければ None)．"""
        datetime_str = self._var_info.get("開始日時")
        if datetime_str is None:
            return None
        for fmt in (
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d %H:%M",
            "%Y/%m/%d",
        ):
            try:
                return datetime.strptime(
                    datetime_str, fmt
                ).date()
            except ValueError:
                continue
        return None

    def clustering_key_value(self) -> Any:
        """Get clustering key based on game start date.

        Returns:
            Date object for clustering, or None if 開始日時 not available
        """
        return self._start_date()

    def partitioning_key_value(self) -> Any:
        """Get partitioning key based on game start date.

        Returns:
            Date object for partitioning, or None if 開始日時 not available
        """
        return self._start_date()
