from maou.domain.parser.csa_parser import CSAParser
from maou.domain.parser.kif_parser import KifParser
from maou.domain.parser.parser import Parser


class HCPEConverter:
    @staticmethod
    def convert(data: str, input_format: str) -> None:
        print("converter")
        parser: Parser
        match input_format:
            case "csa":
                parser = CSAParser()
                parser.parse(data)
            case "kif":
                parser = KifParser()
                parser.parse(data)
            case _:
                print("undefined")
        # パースした結果から一手ずつ進めていって各局面をhcpe形式で保存する
        print("converted")
