import abc
import contextlib
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import ContextManager, Dict, Generator, Optional

import numpy as np
import polars as pl
from tqdm.auto import tqdm

from maou.domain.board import shogi
from maou.domain.data.array_io import save_hcpe_df
from maou.domain.data.rust_io import load_hcpe_df
from maou.domain.data.schema import (
    get_hcpe_polars_schema,
)
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

    def __init__(
        self, *, feature_store: Optional[FeatureStore] = None
    ):
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
        max_workers: int

    @staticmethod
    def _process_single_file(
        file: Path,
        input_format: str,
        output_dir: Path,
        min_rating: Optional[int],
        min_moves: Optional[int],
        max_moves: Optional[int],
        allowed_endgame_status: Optional[list[str]],
        exclude_moves: Optional[list[int]],
    ) -> tuple[str, str]:
        """Process a single file using Polars DataFrames (outputs .feather files)．"""
        logger = logging.getLogger(__name__)

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
                (
                    min_rating is not None
                    and min(parser.ratings()) < min_rating
                )
                or (min_moves is not None and moves < min_moves)
                or (max_moves is not None and moves > max_moves)
                or (
                    allowed_endgame_status is not None
                    and not len(allowed_endgame_status) == 0
                    and parser.endgame()
                    not in allowed_endgame_status
                )
            ):
                return False
            return True

        try:
            parser: Parser
            if input_format == "csa":
                parser = CSAParser()
                parser.parse(file.read_text())
            elif input_format == "kif":
                parser = KifParser()
                parser.parse(file.read_text())
            else:
                raise NotApplicableFormat(
                    f"undefined format {input_format}"
                )

            logger.debug(
                f"棋譜:{file} "
                f"終局状況:{parser.endgame()} "
                f"レーティング:{parser.ratings()} "
                f"手数:{len(parser.moves())}"
            )

            # 指定された条件を満たしたら変換をスキップ
            if not game_filter(
                parser,
                min_rating,
                min_moves,
                max_moves,
                allowed_endgame_status,
            ):
                logger.debug(f"skip the file {file}")
                return (str(file), "skipped")

            # movesの数が0であればそもそもHCPEは作れないのでスキップ
            if len(parser.moves()) == 0:
                logger.debug(
                    f"skip the file {file} because of no moves"
                )
                return (str(file), "skipped (no moves)")

            # Polars用にデータをリストで収集
            hcpe_data: dict[str, list] = {
                "hcp": [],
                "eval": [],
                "bestMove16": [],
                "gameResult": [],
                "id": [],
                "partitioningKey": [],
                "ratings": [],
                "endgameStatus": [],
                "moves": [],
            }

            board = shogi.Board()
            board.set_sfen(parser.init_pos_sfen())

            # 棋譜共通情報を取得する
            partitioning_key_value = (
                parser.partitioning_key_value()
            )
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
                logger.debug(f"{move} : {score} : {comment}")

                if move < 0 or move > 16777215:
                    raise IllegalMove(
                        f"moveの値が想定外 path: {file}"
                    )

                if (
                    exclude_moves is not None
                    and move in exclude_moves
                ):
                    logger.info(
                        f"skip the move {move} in {file} at {idx + 1}. "
                        f"exclude moves: {exclude_moves}"
                    )
                    continue

                # HCP (Huffman Coded Position)
                # to_hcp() expects numpy array, not bytearray
                hcp_array = np.zeros(32, dtype=np.uint8)
                board.to_hcp(hcp_array)
                # Convert to bytes for Polars Binary type
                hcpe_data["hcp"].append(hcp_array.tobytes())

                # 評価値（16bitに収める）
                eval = min(32767, max(score, -32767))
                # 手番側の評価値にする
                if board.get_turn() == shogi.Turn.BLACK:
                    hcpe_data["eval"].append(eval)
                else:
                    hcpe_data["eval"].append(-eval)

                # moveは32bitになっているので16bitに変換する
                hcpe_data["bestMove16"].append(
                    shogi.move16(move)
                )
                hcpe_data["gameResult"].append(parser.winner())
                hcpe_data["id"].append(
                    f"{file.with_suffix('.hcpe').name}_{idx}"
                )

                # 棋譜共通情報を記録
                hcpe_data["partitioningKey"].append(
                    datetime.fromisoformat(
                        partitioning_key_value.isoformat()
                    ).date()
                )
                # Convert ratings to uint16
                hcpe_data["ratings"].append(
                    [int(r) for r in ratings]
                )
                hcpe_data["endgameStatus"].append(endgame)
                hcpe_data["moves"].append(moves)

                # 局面に指し手を反映させる
                board.push_move(move)

            # Polars DataFrameを作成
            df = pl.DataFrame(
                hcpe_data, schema=get_hcpe_polars_schema()
            )

            # ファイルを保存（.feather形式）
            save_hcpe_df(
                df,
                output_dir / file.with_suffix(".feather").name,
            )
            return (str(file), f"success {len(df)} rows")

        except Exception as e:
            logger.error(f"Error processing file {file}: {e}")
            return (str(file), f"error: {str(e)}")

    def convert(self, option: ConvertOption) -> Dict[str, str]:
        """HCPEファイルを作成する (並列処理版)."""
        conversion_result: Dict[str, str] = {}
        self.logger.debug(
            f"変換対象のファイル {option.input_paths}"
        )

        # Determine number of workers
        max_workers = option.max_workers

        self.logger.info(
            f"Using {max_workers} workers for parallel processing"
        )

        with self.__context():
            if max_workers == 1 or len(option.input_paths) == 1:
                # Sequential processing for single worker or single file
                for file in tqdm(
                    option.input_paths, desc="HCPE (single)"
                ):
                    file_path, result = (
                        self._process_single_file(
                            file,
                            option.input_format,
                            option.output_dir,
                            option.min_rating,
                            option.min_moves,
                            option.max_moves,
                            option.allowed_endgame_status,
                            option.exclude_moves,
                        )
                    )

                    # For sequential processing, re-raise exceptions for compatibility
                    if result.startswith("error:"):
                        error_msg = result[
                            7:
                        ]  # Remove "error: " prefix
                        if (
                            "No such file or directory"
                            in error_msg
                        ):
                            raise FileNotFoundError(error_msg)
                        elif "undefined format" in error_msg:
                            raise NotApplicableFormat(error_msg)
                        else:
                            raise Exception(error_msg)

                    conversion_result[file_path] = result

                    # Handle feature store for sequential processing
                    if (
                        self.__feature_store is not None
                        and result.startswith("success")
                    ):
                        # Load the saved .feather file and convert to numpy array for feature store
                        feather_file = (
                            option.output_dir
                            / file.with_suffix(".feather").name
                        )
                        if feather_file.exists():
                            df = load_hcpe_df(feather_file)
                            # Convert Polars DataFrame to numpy structured array
                            from maou.domain.data.schema import (
                                convert_hcpe_df_to_numpy,
                            )

                            structured_array = (
                                convert_hcpe_df_to_numpy(df)
                            )
                            self.__feature_store.store_features(
                                name=file.name,
                                key_columns=["id"],
                                structured_array=structured_array,
                                clustering_key=None,
                                partitioning_key_date="partitioningKey",
                            )
            else:
                # Parallel processing
                with ProcessPoolExecutor(
                    max_workers=max_workers
                ) as executor:
                    # Submit all jobs
                    future_to_file = {
                        executor.submit(
                            self._process_single_file,
                            file,
                            option.input_format,
                            option.output_dir,
                            option.min_rating,
                            option.min_moves,
                            option.max_moves,
                            option.allowed_endgame_status,
                            option.exclude_moves,
                        ): file
                        for file in option.input_paths
                    }

                    # Process completed futures with progress bar
                    for future in tqdm(
                        as_completed(future_to_file),
                        total=len(option.input_paths),
                        desc=f"HCPE (parallel {max_workers} workers)",
                    ):
                        file = future_to_file[future]
                        try:
                            file_path, result = future.result()
                            conversion_result[file_path] = (
                                result
                            )

                            # Handle feature store for parallel processing
                            if (
                                self.__feature_store is not None
                                and result.startswith("success")
                            ):
                                # Load the saved .feather file and convert to numpy array for feature store
                                feather_file = (
                                    option.output_dir
                                    / file.with_suffix(
                                        ".feather"
                                    ).name
                                )
                                if feather_file.exists():
                                    df = load_hcpe_df(
                                        feather_file
                                    )
                                    # Convert Polars DataFrame to numpy structured array
                                    from maou.domain.data.schema import (
                                        convert_hcpe_df_to_numpy,
                                    )

                                    structured_array = convert_hcpe_df_to_numpy(
                                        df
                                    )
                                    self.__feature_store.store_features(
                                        name=file.with_suffix(
                                            ".hcpe"
                                        ).name,
                                        key_columns=["id"],
                                        structured_array=structured_array,
                                        clustering_key=None,
                                        partitioning_key_date="partitioningKey",
                                    )
                        except Exception as exc:
                            self.logger.error(
                                f"File {file} generated an exception: {exc}"
                            )
                            conversion_result[str(file)] = (
                                f"error: {str(exc)}"
                            )

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
