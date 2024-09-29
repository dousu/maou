from cshogi import CSA

from maou.domain.parser import parser


class CSAParser(parser.Parser):
    def parse(self, content: str) -> None:
        """CSAの棋譜文字列をパースして読み取れる状態にする"""
        # 1ファイル前提の処理
        self.parsed_result = CSA.Parser.parse_str(content)[0]
