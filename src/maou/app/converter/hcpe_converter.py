import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cshogi
import numpy as np

from maou.domain.parser.csa_parser import CSAParser
from maou.domain.parser.kif_parser import KifParser
from maou.domain.parser.parser import Parser


class NotApplicableFormat(Exception):
    pass


class HCPEConverter:
    logger: logging.Logger = logging.getLogger(__name__)

    @dataclass(kw_only=True, frozen=True)
    class ConvertOption:
        input_paths: list[Path]
        input_format: str
        output_dir: Path
        min_rating: Optional[int] = None
        min_moves: Optional[int] = None
        max_moves: Optional[int] = None
        allowed_endgame_status: Optional[list[str]] = None
        exclude_moves: Optional[list[int]] = None

    def convert(self, option: ConvertOption) -> Dict[str, str]:
        """HCPEファイルを作成する."""

        def game_filter(
            parser: Parser,
            min_rating: Optional[int] = None,
            min_moves: Optional[int] = None,
            max_moves: Optional[int] = None,
            allowed_endgame_status: Optional[list[str]] = None,
        ) -> bool:
            """指定された条件を満たす場合Trueを返す"""
            moves: int = len(parser.moves())
            if (
                (min_rating is not None and min(parser.ratings()) < min_rating)
                or (min_moves is not None and moves < min_moves)
                or (max_moves is not None and moves > max_moves)
                or (
                    allowed_endgame_status is not None
                    and not len(allowed_endgame_status) == 0
                    and parser.endgame() not in allowed_endgame_status
                )
            ):
                return False

            return True

        conversion_result: Dict[str, str] = {}
        self.logger.debug(f"変換対象のファイル {option.input_paths}")
        for i, file in enumerate(option.input_paths):
            self.logger.info(f"進捗: {i / len(option.input_paths) * 100}%")
            parser: Parser
            match option.input_format:
                case "csa":
                    parser = CSAParser()
                    parser.parse(file.read_text())
                case "kif":
                    parser = KifParser()
                    parser.parse(file.read_text())
                case format_str:
                    raise NotApplicableFormat(f"undefined format {format_str}")
            # パースした結果から一手ずつ進めていって各局面をhcpe形式で保存する
            self.logger.info(
                f"棋譜:{file} "
                f"終局状況:{parser.endgame()} "
                f"レーティング:{parser.ratings()} "
                f"手数:{len(parser.moves())}"
            )
            if not game_filter(
                parser,
                option.min_rating,
                option.min_moves,
                option.max_moves,
                option.allowed_endgame_status,
            ):
                conversion_result[str(file)] = "skipped"
                self.logger.info(f"skip the file {file}")
                continue
            # 1024もあれば確保しておく局面数として十分だろう
            hcpes = np.zeros(1024, cshogi.HuffmanCodedPosAndEval)  # type: ignore
            board = cshogi.Board()  # type: ignore
            board.set_sfen(parser.init_pos_sfen())
            try:
                for i, (move, score, comment) in enumerate(
                    zip(parser.moves(), parser.scores(), parser.comments())
                ):
                    self.logger.debug(f"{move} : {score} : {comment}")
                    if move < 0 or move > 16777215:
                        raise Exception(f"moveの値が想定外 path: {file}")
                    if (
                        option.exclude_moves is not None
                        and move in option.exclude_moves
                    ):
                        self.logger.info(
                            f"skip the move {move} in {file} at {i + 1}. "
                            f"exclude moves: {option.exclude_moves}"
                        )
                        continue

                    hcpe = hcpes[i]
                    board.to_hcp(hcpe["hcp"])
                    # 16bitに収める
                    eval = min(32767, max(score, -32767))
                    # 手番側の評価値にする (ここは表現の問題で前処理としてもよさそう)
                    if board.turn == cshogi.BLACK:  # type: ignore
                        hcpe["eval"] = eval
                    else:
                        hcpe["eval"] = -eval
                    # moveは32bitになっているので16bitに変換する
                    # 上位16bitを単に削っていて，上位16bitは移動する駒と取った駒の種類が入っている
                    # 特に動かす駒の種類の情報が抜けているので注意
                    hcpe["bestMove16"] = cshogi.move16(move)  # type: ignore
                    hcpe["gameResult"] = parser.winner()

                    board.push(move)
                hcpes[:i].tofile(option.output_dir / file.with_suffix(".hcpe").name)
                conversion_result[str(file)] = "success"
            except Exception as e:
                raise e

        return conversion_result
