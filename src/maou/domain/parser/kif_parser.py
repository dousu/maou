from typing import Any

from cshogi import KIF

from maou.domain.parser import parser


class KifParser(parser.Parser):
    """Parser for KIF format Shogi game records.
    
    KIF is a human-readable format for Shogi games. Note that this format
    has limited information compared to CSA format (no ratings, scores, etc.).
    """
    def parse(self, content: str) -> None:
        """kifの棋譜文字列をパースして読み取れる状態にする.
        足りない情報があるがここら辺をどうするかは要検討
        """
        self.kif = KIF.Parser.parse_str(content)

    def init_pos_sfen(self) -> str:
        """Get initial board position in SFEN notation.
        
        Returns:
            Initial position as SFEN string
        """
        return self.kif.sfen  # type: ignore

    def endgame(self) -> str:
        """Get game termination reason.
        
        Returns:
            Endgame reason from KIF format
        """
        return self.kif.endgame  # type: ignore

    def winner(self) -> int:
        """Get game winner.
        
        Returns:
            Winner player number from KIF format
        """
        return self.kif.win  # type: ignore

    def ratings(self) -> list[int]:
        """Get player ratings.
        
        Returns:
            Empty list as KIF format doesn't contain rating information
        """
        return []

    def moves(self) -> list[int]:
        """Get sequence of moves in the game.
        
        Returns:
            List of moves encoded as integers from KIF format
        """
        return self.kif.moves  # type: ignore

    def scores(self) -> list[int]:
        """Get evaluation scores for each position.
        
        Returns:
            Empty list as KIF format doesn't contain score information
        """
        return []

    def comments(self) -> list[str]:
        """Get comments for each move.
        
        Returns:
            List of move comments from KIF format
        """
        return self.kif.comments  # type: ignore

    def clustering_key_value(self) -> Any:
        """Get clustering key for data partitioning.
        
        Returns:
            None as KIF format doesn't contain timestamp information
        """
        return None

    def partitioning_key_value(self) -> Any:
        """Get partitioning key for data distribution.
        
        Returns:
            None as KIF format doesn't contain timestamp information
        """
        return None
