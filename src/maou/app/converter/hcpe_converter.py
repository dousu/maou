import abc
import contextlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ContextManager, Dict, Generator, Optional

import cshogi
import numpy as np
from tqdm.auto import tqdm

from maou.domain.parser.csa_parser import CSAParser
from maou.domain.parser.kif_parser import KifParser
from maou.domain.parser.parser import Parser


class FeatureStore(metaclass=abc.ABCMeta):
    """Abstract interface for storing game features in various backends.
    
    Defines the contract for storing processed Shogi game data
    in different storage systems (local files, cloud databases, etc.).
    """
    @abc.abstractmethod
    def feature_store(self) -> ContextManager[None]:
        pass

    @abc.abstractmethod
    def store_features(
        self,
        *,
        name: str,
        key_columns: list[str],
        structured_array: np.ndarray,
        clustering_key: Optional[str] = None,
        partitioning_key_date: Optional[str] = None,
    ) -> None:
        pass


class NotApplicableFormat(Exception):
    pass


class IllegalMove(Exception):
    pass


class HCPEConverter:
    """Converts Shogi game records to HCPE (HuffmanCodedPosAndEval) format.
    
    Processes CSA and KIF format game files, extracts positions and evaluations,
    and converts them to the HCPE format used for neural network training.
    Supports quality filtering based on game ratings and move counts.
    """
    logger: logging.Logger = logging.getLogger(__name__)

    def __init__(self, *, feature_store: Optional[FeatureStore] = None):
        """Initialize HCPE converter.
        
        Args:
            feature_store: Optional storage backend for converted features
        """
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
                # これ以上ある場合は無駄な局面が大量にありそうなので枝刈りした方がよさそう
                # HCPEと同じ部分は同じdtypeを利用しているが
                # 実は独自データフォーマットなのでHCPEではない
                # https://github.com/TadaoYamaoka/cshogi/blob/b13c3b248f870b218cb71e7f9c17dfae7bf0f6e9/cshogi/_cshogi.pyx#L19
                hcpes = np.zeros(
                    1024,
                    dtype=[
                        ("hcp", (np.uint8, 32)),
                        ("eval", np.int16),
                        ("bestMove16", np.int16),
                        ("gameResult", np.int8),
                        ("id", (np.unicode_, 128)),  # type: ignore[attr-defined]
                        ("partitioningKey", np.dtype("datetime64[D]")),
                        ("ratings", (np.uint16, 2)),
                        ("endgameStatus", (np.unicode_, 16)),  # type: ignore[attr-defined] # noqa: E501
                        ("moves", np.int16),
                    ],
                )
                board = cshogi.Board()  # type: ignore
                board.set_sfen(parser.init_pos_sfen())
                try:
                    # 棋譜共通情報を取得する
                    partitioning_key_value = parser.partitioning_key_value()
                    ratings = parser.ratings()
                    endgame = parser.endgame()
                    moves = len(parser.moves())
                    # 1手毎に代わる情報を取得する
                    for idx, (move, score, comment) in enumerate(
                        zip(
                            parser.moves(),
                            parser.scores(),
                            parser.comments(),
                        )
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
                        hcpe["id"] = f"{file.with_suffix('.hcpe').name}_{idx}"
                        # 棋譜共通情報を記録
                        hcpe["partitioningKey"] = np.datetime64(
                            partitioning_key_value.isoformat()
                        )
                        hcpe["ratings"] = ratings
                        hcpe["endgameStatus"] = endgame
                        hcpe["moves"] = moves

                        # 局面に指し手を反映させる
                        board.push(move)
                    # np.saveで保存することでメタデータをつけておく
                    # HCPEの形式から逸脱するのでCPUパフォーマンスは悪くなる
                    np.save(
                        option.output_dir / file.with_suffix(".npy").name,
                        hcpes[: idx + 1],
                    )
                    if self.__feature_store is not None:
                        self.__feature_store.store_features(
                            name=file.with_suffix(".hcpe").name,
                            key_columns=["id"],
                            structured_array=hcpes[: idx + 1],
                            clustering_key=None,
                            partitioning_key_date="partitioningKey",
                        )
                    conversion_result[str(file)] = f"success {idx + 1} rows"
                except Exception:
                    raise

        return conversion_result

    @contextlib.contextmanager
    def __context(self) -> Generator[None, None, None]:
        try:
            if self.__feature_store is not None:
                with self.__feature_store.feature_store():
                    yield
            else:
                yield
        except Exception:
            raise
