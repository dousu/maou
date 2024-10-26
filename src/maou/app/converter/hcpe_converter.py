from dataclasses import dataclass
from pathlib import Path

import numpy as np
from cshogi import BLACK, Board, HuffmanCodedPosAndEval, move16

from maou.domain.parser.csa_parser import CSAParser
from maou.domain.parser.kif_parser import KifParser
from maou.domain.parser.parser import Parser


class HCPEConverter:
    @dataclass
    class ConvertOption:
        input_paths: list[Path]
        input_format: str
        output_dir: Path

    @staticmethod
    def convert(option: ConvertOption) -> None:
        """HCPEファイルを作成する.
        TODO: レーティング・手数・endgameでのフィルタリングオプションをいれる
        """
        print("converter")
        print(option.input_paths)
        for file in option.input_paths:
            parser: Parser
            match option.input_format:
                case "csa":
                    parser = CSAParser()
                    parser.parse(file.read_text())
                case "kif":
                    parser = KifParser()
                    parser.parse(file.read_text())
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
            # 1024もあれば確保しておく局面数として十分だろう
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
                hcpes[:i].tofile(option.output_dir / file.with_suffix(".hcpe").name)
            except Exception as e:
                print("Error Occured in HCPEConverter.convert")
                raise e
