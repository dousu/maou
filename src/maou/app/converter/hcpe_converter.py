import numpy as np
from cshogi import BLACK, Board, HuffmanCodedPosAndEval, move16

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
        print(
            f"""棋譜情報
            ```
            終局状況 {parser.endgame()}
            レーティング {parser.ratings()}
            手数 {len(parser.moves())}
            ```
            """
        )
        hcpes = np.zeros(1024, HuffmanCodedPosAndEval)
        board = Board()
        board.set_sfen(parser.init_pos_sfen())
        print(board)
        try:
            for i, (move, score, comment) in enumerate(
                zip(parser.moves(), parser.scores(), parser.comments())
            ):
                print(f"{move} : {score} : {comment}")
                print(board)

                hcpe = hcpes[i]
                board.to_hcp(hcpe["hcp"])
                # 16bitに収める
                eval = min(32767, max(score, -32767))
                # 手番側の評価値にする (ここは表現の問題で前処理としてもよさそう)
                hcpe["eval"] = eval if board.turn == BLACK else -eval
                # moveは32bitになっているので16bitに変換する
                # 上位16bit削っているようだがそれで成立するのかは未検討 (たぶんCで32ビット使っているが値域的には不要だからとか？)
                hcpe["bestMove16"] = move16(move)
                hcpe["gameResult"] = parser.winner()

                board.push(move)
            hcpes[:i].tofile("test_dousu.hcpe")
        except Exception as e:
            print("Error Occured in HCPEConverter.convert")
            raise e
