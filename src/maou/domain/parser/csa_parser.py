from cshogi import CSA

from maou.domain.parser import parser


class CSAParser(parser.Parser):
    def parse(self, content: str) -> None:
        """CSAの棋譜文字列をパースして読み取れる状態にする"""
        # 1ファイル前提の処理
        self.kif = CSA.Parser.parse_str(content)[0]

    def init_pos_sfen(self) -> str:
        return self.kif.sfen

    def endgame(self) -> str:
        return self.kif.endgame

    def winner(self) -> int:
        return self.kif.win

    def ratings(self) -> list[int]:
        return self.kif.ratings

    def moves(self) -> list[str]:
        return self.kif.moves

    def scores(self) -> list[int]:
        return self.kif.scores

    def comments(self) -> list[str]:
        return self.kif.comments
