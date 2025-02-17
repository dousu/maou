import abc
import contextlib
import logging
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ContextManager, Dict, Generator, Optional

import cshogi
import numpy as np
import pyarrow as pa
from tqdm.auto import tqdm

from maou.domain.parser.csa_parser import CSAParser
from maou.domain.parser.kif_parser import KifParser
from maou.domain.parser.parser import Parser


class FeatureStore(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def feature_store(self) -> ContextManager[None]:
        pass

    @abc.abstractmethod
    def store_features(
        self,
        *,
        key_columns: list[str],
        arrow_table: pa.Table,
        clustering_key: Optional[str] = None,
    ) -> None:
        pass


class NotApplicableFormat(Exception):
    pass


class IllegalMove(Exception):
    pass


class HCPEConverter:
    logger: logging.Logger = logging.getLogger(__name__)

    def __init__(self, *, feature_store: Optional[FeatureStore] = None):
        self.__feature_store = feature_store

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
        with self.__context():
            for file in tqdm(option.input_paths):
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
                self.logger.debug(
                    f"棋譜:{file} "
                    f"終局状況:{parser.endgame()} "
                    f"レーティング:{parser.ratings()} "
                    f"手数:{len(parser.moves())}"
                )

                # 指定された条件を満たしたら変換をスキップ
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

                # movesの数が0であればそもそもHCPEは作れないのでスキップ
                if len(parser.moves()) == 0:
                    conversion_result[str(file)] = "skipped (no moves)"
                    self.logger.info(f"skip the file {file} because of no moves")
                    continue

                # 1024もあれば確保しておく局面数として十分だろう
                hcpes = np.zeros(1024, cshogi.HuffmanCodedPosAndEval)  # type: ignore
                board = cshogi.Board()  # type: ignore
                board.set_sfen(parser.init_pos_sfen())
                arrow_features: dict[str, list[Any]] = defaultdict(list)
                try:
                    for idx, (move, score, comment) in enumerate(
                        zip(parser.moves(), parser.scores(), parser.comments())
                    ):
                        self.logger.debug(f"{move} : {score} : {comment}")

                        if move < 0 or move > 16777215:
                            raise IllegalMove(f"moveの値が想定外 path: {file}")

                        if (
                            option.exclude_moves is not None
                            and move in option.exclude_moves
                        ):
                            self.logger.info(
                                f"skip the move {move} in {file} at {idx + 1}. "
                                f"exclude moves: {option.exclude_moves}"
                            )
                            continue

                        hcpe = hcpes[idx]
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
                        if self.__feature_store is not None:
                            arrow_features["hcp"].append(pickle.dumps(hcpe["hcp"]))
                            arrow_features["eval"].append(hcpe["eval"])
                            arrow_features["bestMove16"].append(hcpe["bestMove16"])
                            arrow_features["gameResult"].append(hcpe["gameResult"])
                            # ローカルファイルには入らない情報
                            arrow_features["id"].append(
                                f"{file.with_suffix('.hcpe').name}_{idx}"
                            )
                            arrow_features["clusteringKey"].append(
                                parser.clustering_key_value()
                            )
                            arrow_features["ratings"].append(
                                pickle.dumps(np.array(parser.ratings()))
                            )
                            arrow_features["endgameStatus"].append(parser.endgame())
                            arrow_features["moves"].append(len(parser.moves()))

                        board.push(move)
                    # np.saveで保存することでメタデータをつけておく
                    # HCPEの形式から逸脱するのでCPUパフォーマンスは悪くなる
                    np.save(
                        option.output_dir / file.with_suffix(".npy").name,
                        hcpes[: idx + 1],
                    )
                    if self.__feature_store is not None:
                        self.__feature_store.store_features(
                            key_columns=["id"],
                            arrow_table=pa.table(arrow_features),
                            clustering_key="clusteringKey",
                        )
                    conversion_result[str(file)] = f"success {idx + 1} rows"
                except Exception as e:
                    raise e

        return conversion_result

    @contextlib.contextmanager
    def __context(self) -> Generator[None, None, None]:
        try:
            if self.__feature_store is not None:
                with self.__feature_store.feature_store():
                    yield
            else:
                yield
        finally:
            pass
