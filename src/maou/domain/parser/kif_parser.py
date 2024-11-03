from maou.domain.parser import parser


class KifParser(parser.Parser):
    def parse(self, content: str) -> None:
        """kifの棋譜文字列をパースして読み取れる状態にする"""
        pass

    def init_pos_sfen(self) -> str:
        return ""

    def endgame(self) -> str:
        return ""

    def winner(self) -> int:
        return 0

    def ratings(self) -> list[int]:
        return []

    def moves(self) -> list[str]:
        return []

    def scores(self) -> list[int]:
        return []

    def comments(self) -> list[str]:
        return []
