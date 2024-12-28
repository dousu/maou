from cshogi import KIF

from maou.domain.parser import parser


class KifParser(parser.Parser):
    def parse(self, content: str) -> None:
        """kifの棋譜文字列をパースして読み取れる状態にする.
        足りない情報があるがここら辺をどうするかは要検討
        """
        self.kif = KIF.Parser.parse_str(content)

    def init_pos_sfen(self) -> str:
        return self.kif.sfen  # type: ignore

    def endgame(self) -> str:
        return self.kif.endgame  # type: ignore

    def winner(self) -> int:
        return self.kif.win  # type: ignore

    def ratings(self) -> list[int]:
        return []

    def moves(self) -> list[int]:
        return self.kif.moves  # type: ignore

    def scores(self) -> list[int]:
        return []

    def comments(self) -> list[str]:
        return self.kif.comments  # type: ignore
