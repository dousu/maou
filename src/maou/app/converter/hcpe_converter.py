import abc
import contextlib
import logging
from collections.abc import Generator
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from tqdm.auto import tqdm

from maou.domain.data.rust_io import (
    load_hcpe_df,
    merge_hcpe_feather_files,
)


class FeatureStore(metaclass=abc.ABCMeta):
    """Abstract interface for storing game features in various backends.

    Defines the contract for storing processed Shogi game data
    in different storage systems (local files, cloud databases, etc.).
    """

    @abc.abstractmethod
    def feature_store(self) -> AbstractContextManager[None]:
        pass

    @abc.abstractmethod
    def store_features(
        self,
        *,
        name: str,
        key_columns: list[str],
        dataframe: pl.DataFrame,
        clustering_key: str | None = None,
        partitioning_key_date: str | None = None,
    ) -> None:
        pass


class NotApplicableFormat(Exception):
    pass


class HCPEConverter:
    """Converts Shogi game records to HCPE (HuffmanCodedPosAndEval) format.

    Processes CSA and KIF format game files, extracts positions and evaluations,
    and converts them to the HCPE format used for neural network training.
    Supports quality filtering based on game ratings and move counts.
    """

    logger: logging.Logger = logging.getLogger(__name__)

    def __init__(
        self, *, feature_store: FeatureStore | None = None
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
        min_rating: int | None = None
        min_moves: int | None = None
        max_moves: int | None = None
        allowed_endgame_status: list[str] | None = None
        exclude_moves: list[int] | None = None
        max_workers: int
        chunk_size: int = 500_000

    # Rust 一括変換に渡すファイル数のバッチ粒度 (tqdm 進捗の更新単位)．
    _FILE_BATCH_SIZE = 200

    def convert(self, option: ConvertOption) -> dict[str, str]:
        """HCPEファイルを作成する (Rust 一括変換パイプライン版)．

        処理フロー:
        1. maou._rust.maou_convert で各ファイルを .feather に一括変換
           (ファイル直読み + UTF-8→cp932 fallback + rayon 並列)
        2. chunk_size > 0 の場合，個別ファイルをチャンクにマージ
        3. feature_store が設定されている場合，チャンクをアップロード

        cshogi 時代の per-move PyO3 往復 (~3N/局) と ProcessPoolExecutor は
        Rust 側の一括処理に置き換えられた．複数局 CSA は全局変換され，
        cp932 の .kif も読める (従来は先頭 1 局のみ / UTF-8 固定だった)．
        """
        from maou._rust.maou_convert import convert_hcpe_files

        # 入力フォーマット検証 (Rust も検証するが，直接 convert を呼ぶ
        # 消費者のために NotApplicableFormat を明示送出する)
        if option.input_format not in ("csa", "kif"):
            raise NotApplicableFormat(
                f"undefined format {option.input_format}"
            )

        conversion_result: dict[str, str] = {}
        self.logger.debug(
            f"変換対象のファイル {option.input_paths}"
        )

        # 出力先を作成 (従来は save_hcpe_df の mkdir に依存していた)
        option.output_dir.mkdir(parents=True, exist_ok=True)

        # max_workers を rayon スレッド数にマップ (0/None はグローバルプール)
        threads = (
            option.max_workers
            if option.max_workers and option.max_workers >= 1
            else None
        )
        # 単一ワーカー / 単一ファイルのとき，欠損ファイル等を例外で
        # 再送出する (従来の sequential 互換規約)
        raise_on_error = (
            option.max_workers == 1
            or len(option.input_paths) == 1
        )

        with self.__context():
            # Phase 1: Rust 一括変換 (tqdm 進捗のためバッチ分割して呼ぶ)
            paths = [str(p) for p in option.input_paths]
            with tqdm(
                total=len(paths), desc="HCPE (rust)"
            ) as pbar:
                for start in range(
                    0, len(paths), self._FILE_BATCH_SIZE
                ):
                    batch = paths[
                        start : start + self._FILE_BATCH_SIZE
                    ]
                    results = convert_hcpe_files(
                        batch,
                        option.input_format,
                        str(option.output_dir),
                        min_rating=option.min_rating,
                        min_moves=option.min_moves,
                        max_moves=option.max_moves,
                        allowed_endgame_status=option.allowed_endgame_status,
                        exclude_moves=option.exclude_moves,
                        threads=threads,
                    )
                    for file_path, result in results:
                        if raise_on_error and result.startswith(
                            "error:"
                        ):
                            self._reraise_error(result[7:])
                        conversion_result[file_path] = result
                    pbar.update(len(batch))

            # Phase 2: チャンキングとアップロード
            self._chunk_and_upload(
                conversion_result=conversion_result,
                output_dir=option.output_dir,
                chunk_size=option.chunk_size,
            )

        return conversion_result

    @staticmethod
    def _reraise_error(error_msg: str) -> None:
        """Rust の error status を従来の例外型に再送出する (sequential 互換)．"""
        if (
            "No such file or directory" in error_msg
            or "os error 2" in error_msg
        ):
            raise FileNotFoundError(error_msg)
        elif "undefined format" in error_msg:
            raise NotApplicableFormat(error_msg)
        else:
            raise Exception(error_msg)

    def _chunk_and_upload(
        self,
        *,
        conversion_result: dict[str, str],
        output_dir: Path,
        chunk_size: int,
    ) -> None:
        """個別 .feather ファイルをチャンクにまとめ，feature_store にアップロードする．

        Args:
            conversion_result: 変換結果のマッピング(ファイルパス→ステータス)
            output_dir: .feather ファイルの出力ディレクトリ
            chunk_size: 1チャンクあたりの行数(0の場合チャンキングしない)
        """
        # 成功した .feather ファイルを収集
        # NOTE: _process_single_file が
        # output_dir / file.with_suffix(".feather").name で保存するため，
        # ここでも同じ変換で出力パスを再構築している
        successful_feather_files = [
            output_dir / Path(fp).with_suffix(".feather").name
            for fp, result in conversion_result.items()
            if result.startswith("success")
        ]
        existing_feather_files: list[Path] = []
        for f in successful_feather_files:
            if f.exists():
                existing_feather_files.append(f)
            else:
                self.logger.warning(
                    f"Expected feather file missing: {f}"
                )

        if not existing_feather_files:
            return

        if chunk_size > 0:
            self.logger.info(
                f"Merging {len(existing_feather_files)} files "
                f"into chunks (chunk_size={chunk_size})"
            )
            chunked_paths = merge_hcpe_feather_files(
                existing_feather_files,
                output_dir,
                rows_per_chunk=chunk_size,
                output_prefix="hcpe",
            )

            self._upload_to_feature_store(chunked_paths)

            # 個別ファイルを削除(チャンクファイルのみ残す)
            # merge_hcpe_feather_files は常に新規ファイルを作成するため
            # chunked_paths と existing_feather_files は重複しない
            # NOTE: symlink を含むパスでは absolute() の方が安全だが，
            # 本プロジェクトでは output_dir 配下に symlink は想定しない
            chunked_set = {p.resolve() for p in chunked_paths}
            for f in existing_feather_files:
                if f.resolve() not in chunked_set:
                    f.unlink()

            self.logger.info(
                f"Created {len(chunked_paths)} chunked files"
            )
        elif self.__feature_store is not None:
            self._upload_to_feature_store(
                existing_feather_files
            )

    def _upload_to_feature_store(
        self, paths: list[Path]
    ) -> None:
        """feather ファイルを feature_store にアップロードする．

        Args:
            paths: アップロード対象の .feather ファイルパスリスト
        """
        if self.__feature_store is None:
            return
        for p in paths:
            df = load_hcpe_df(p)
            self.__feature_store.store_features(
                name=p.name,
                key_columns=["id"],
                dataframe=df,
                clustering_key=None,
                partitioning_key_date="partitioningKey",
            )

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
