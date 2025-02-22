from datetime import datetime
from typing import Any

from cshogi import CSA

from maou.domain.parser import parser


class CSAParser(parser.Parser):
    def parse(self, content: str) -> None:
        """CSAの棋譜文字列をパースして読み取れる状態にする.
        基本的にこのクラスはcshogiの実装のラッパーでしかないが、cshogiの定義を外に出さないようにする。
        """
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

    def moves(self) -> list[int]:
        return self.kif.moves

    def scores(self) -> list[int]:
        return self.kif.scores

    def comments(self) -> list[str]:
        return self.kif.comments

    def clustering_key_value(self) -> Any:
        try:
            datetime_str = self.kif.var_info["START_TIME"]
            date_obj = datetime.strptime(datetime_str, "%Y/%m/%d %H:%M:%S")
            clustering_key = date_obj.date()
        except KeyError:
            clustering_key = None
        return clustering_key

    def partitioning_key_value(self) -> Any:
        try:
            datetime_str = self.kif.var_info["START_TIME"]
            date_obj = datetime.strptime(datetime_str, "%Y/%m/%d %H:%M:%S")
            partitioning_key = date_obj.date()
        except KeyError:
            partitioning_key = None
        return partitioning_key
