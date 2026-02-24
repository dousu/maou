from datetime import datetime
from typing import Any

from cshogi import CSA

from maou.domain.parser import parser


class CSAParser(parser.Parser):
    """Parser for CSA format Shogi game records.

    CSA (Computer Shogi Association) is a standard format for representing
    Shogi games. This parser extracts game information using the cshogi library.

    **Design Note: cshogi Dependency**
    This parser is intentionally coupled to cshogi.CSA.Parser as CSA parsing is
    an implementation detail. The Parser abstract base class provides abstraction
    at the domain boundary.

    If replacing cshogi with another library, create a new CSAParser implementation
    rather than wrapping the existing one.
    """

    def parse(self, content: str) -> None:
        """CSAの棋譜文字列をパースして読み取れる状態にする.
        基本的にこのクラスはcshogiの実装のラッパーでしかないが，cshogiの定義を外に出さないようにする．
        """
        # 1ファイル前提の処理
        self.kif = CSA.Parser.parse_str(content)[0]

    def init_pos_sfen(self) -> str:
        """Get initial board position in SFEN notation.

        Returns:
            Initial position as SFEN string
        """
        return self.kif.sfen

    def endgame(self) -> str:
        """Get game termination reason.

        Returns:
            Endgame reason from CSA format
        """
        return self.kif.endgame

    def winner(self) -> int:
        """Get game winner.

        Returns:
            Winner player number from CSA format
        """
        return self.kif.win

    def ratings(self) -> list[int]:
        """Get player ratings.

        Returns:
            List of player ratings from CSA format
        """
        return self.kif.ratings

    def moves(self) -> list[int]:
        """Get sequence of moves in the game.

        Returns:
            List of moves encoded as integers from CSA format
        """
        return self.kif.moves

    def scores(self) -> list[int]:
        """Get evaluation scores for each position.

        Returns:
            List of position evaluation scores from CSA format
        """
        return self.kif.scores

    def comments(self) -> list[str]:
        """Get comments for each move.

        Returns:
            List of move comments from CSA format
        """
        return self.kif.comments

    def clustering_key_value(self) -> Any:
        """Get clustering key based on game start date.

        Returns:
            Date object for clustering, or None if START_TIME not available
        """
        try:
            datetime_str = self.kif.var_info["START_TIME"]
            date_obj = datetime.strptime(
                datetime_str, "%Y/%m/%d %H:%M:%S"
            )
            clustering_key = date_obj.date()
        except KeyError:
            clustering_key = None
        return clustering_key

    def partitioning_key_value(self) -> Any:
        """Get partitioning key based on game start date.

        Returns:
            Date object for partitioning, or None if START_TIME not available
        """
        try:
            datetime_str = self.kif.var_info["START_TIME"]
            date_obj = datetime.strptime(
                datetime_str, "%Y/%m/%d %H:%M:%S"
            )
            partitioning_key = date_obj.date()
        except KeyError:
            partitioning_key = None
        return partitioning_key
