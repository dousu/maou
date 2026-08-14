"""Benchmark Polars + Rust I/O performance．

Polars + Rust I/O パイプラインの保存・読込・イテレーションを実測する．

以前は ``.npy`` (numpy) との比較ベンチマークだったが，``.npy`` は
データパイプラインのどの経路からも使われなくなったため 2026-08-13 に
削除した (`/audit-backlog` backlog 行 N-2)．DataSource イテレーションの
``.npy`` 側は ``FileDataSource`` に ``.npy`` を渡しており，
``Only .feather files are supported`` で **必ず ValueError になっていた**
— つまり ``python -m maou.infra.utility.benchmark_polars_io`` は
docs/performance.md が案内する形では最後まで走らなかった．
"""

import logging
import time
from collections.abc import Mapping
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import psutil

from maou.infra.file_system.file_data_source import (
    FileDataSource,
)
from maou.interface.data_io import (
    load_hcpe_df,
    load_preprocessing_df,
    save_hcpe_df,
    save_preprocessing_df,
)
from maou.interface.data_schema import (
    get_hcpe_dtype,
    get_hcpe_polars_schema,
    get_preprocessing_dtype,
    get_preprocessing_polars_schema,
)

logger = logging.getLogger(__name__)


def _field_dtype(
    np_dtype: np.dtype, column: str
) -> np.dtype | None:
    """structured dtype 側が宣言する列の dtype を引く．

    polars スキーマは ``pl.List(pl.UInt8)`` や ``pl.Utf8`` のように
    **長さを持たない** (固定長は Arrow / numpy 側の情報)．一方
    structured dtype は ``("piecesInHand", np.uint8, (14,))`` や
    ``("endgameStatus", (np.str_, 16))`` と長さまで宣言する．
    ベンチのテストデータが必要とする長さは常に後者なので，そこから引く．

    Args:
        np_dtype: 対応する structured dtype
        column: 列名

    Returns:
        列の dtype．structured dtype にその列が無ければ ``None``．
    """
    fields = np_dtype.fields
    if fields is None or column not in fields:
        return None
    field: tuple = fields[column]
    return field[0]


def _synthesize_column(
    column: str,
    pl_dtype: "pl.DataType",
    np_field: np.dtype | None,
    num_records: int,
) -> list:
    """1 列分のベンチ用ダミー値を polars dtype と structured dtype から作る．

    値の中身は I/O 性能の計測にしか使われないので意味は問わない．
    重要なのは **スキーマが変わったら自動で追従する**ことで，
    ここに長さや列名をハードコードしないこと自体が目的である．

    Args:
        column: 列名 (文字列列の値づくりに使う)
        pl_dtype: polars の列 dtype
        np_field: structured dtype 側の宣言 (shape と文字幅の出所)．
            structured dtype に無い列なら ``None``
        num_records: 生成する行数

    Returns:
        行数分の値のリスト

    Raises:
        ValueError: 対応していない polars dtype のとき
    """
    base = pl_dtype.base_type()
    shape: tuple[int, ...] = (
        np_field.shape if np_field is not None else ()
    )

    if base == pl.Binary:
        # hcp のような固定長バイト列．shape が無ければ 1 バイト．
        width = shape[0] if shape else 1
        return [
            bytes((i + j) % 256 for j in range(width))
            for i in range(num_records)
        ]

    if base == pl.Utf8:
        # 宣言幅いっぱいに埋めると id 列だけで 128 文字になり，計測
        # 対象のデータ量が従来と大きく変わってしまう．従来どおり
        # 短い可読値を作り，**宣言幅で切り詰めるだけ**にする
        # (structured array へ戻すときの黙った切り詰めを防ぐ)．
        width = _str_width(np_field)
        return [
            f"{column[:8]}_{i:08d}"[:width]
            for i in range(num_records)
        ]

    if base == pl.Date:
        return [date(2025, 12, 25)] * num_records

    if base == pl.List:
        inner = pl_dtype.inner  # type: ignore[attr-defined]
        if not shape:
            raise ValueError(
                f"列 {column} は polars では List だが structured dtype に"
                f"長さの宣言が無い．benchmark_polars_io は shape を "
                f"structured dtype から引くので，dtype 側に列を足すか，"
                f"_synthesize_column に例外を書くこと．"
            )
        return [
            _nested(shape, inner, i) for i in range(num_records)
        ]

    if base.is_integer():
        return [i % 128 for i in range(num_records)]

    if base.is_float():
        return [float(i % 2) for i in range(num_records)]

    raise ValueError(
        f"列 {column} の polars dtype {pl_dtype} に対応していない．"
        f"benchmark_polars_io._synthesize_column を更新すること．"
    )


def _str_width(np_field: np.dtype | None) -> int:
    """文字列列の宣言幅 (文字数) を返す．

    numpy の unicode dtype は 1 文字 4 バイトなので ``itemsize // 4``．
    structured dtype に列が無い / 幅が取れない場合は 16 文字に倒す．
    """
    if np_field is None or np_field.kind != "U":
        return 16
    return max(1, np_field.itemsize // 4)


def _nested(
    shape: tuple[int, ...], inner: "pl.DataType", seed: int
) -> list:
    """``shape`` の入れ子リストを作る (9x9 なら 9 要素 × 9 要素)．"""
    if len(shape) == 1:
        if inner.is_float():
            return [
                float((seed + k) % 7) / 7.0
                for k in range(shape[0])
            ]
        return [(seed + k) % 128 for k in range(shape[0])]
    return [
        _nested(shape[1:], inner, seed + k)
        for k in range(shape[0])
    ]


def _create_test_data(
    schema: Mapping[str, "pl.DataType"],
    np_dtype: np.dtype,
    num_records: int,
    label: str,
) -> "pl.DataFrame":
    """polars スキーマ + structured dtype からベンチ用データを組み立てる．

    以前は列名も長さ (9x9 / 14 / ``MOVE_LABELS_NUM`` / 32 バイト) も
    ハードコードした dict を手で維持していた．スキーマに列が増えると
    polars 内部の ``KeyError`` として現れるだけで原因が読み取れず，
    実際 ``moveWinRate`` と ``bestMoveWinRate`` の 2 列が欠けたまま
    気付かれずにいた (backlog 行 N5-1)．

    列集合は polars スキーマ，長さは structured dtype を唯一の出所と
    することで，どちらが変わってもテストデータが自動で追従する．
    追従できない組み合わせ (List なのに長さの出所が無い，未対応の
    dtype) は列名を名指しして落とす．

    Args:
        schema: polars スキーマ
        np_dtype: 対応する structured dtype (shape の出所)
        num_records: 生成する行数
        label: エラーメッセージ用のデータ種別名

    Returns:
        スキーマどおりの DataFrame

    Raises:
        ValueError: 列を生成できないとき
    """
    data: dict[str, list] = {}
    for column, pl_dtype in schema.items():
        np_field = _field_dtype(np_dtype, column)
        if np_field is None and pl_dtype.base_type() in (
            pl.List,
            pl.Binary,
        ):
            # 長さの出所が無い列は生成できない．polars スキーマだけが
            # 先に育って structured dtype が追い付いていない状態なので，
            # 列名を名指しして落とす (実際 bestMoveWinRate は polars
            # スキーマにしか無いが，スカラなのでここには来ない)．
            raise ValueError(
                f"{label} の列 {column} は長さを要するが structured "
                f"dtype に宣言が無い．structured dtype 側に列を"
                f"足すこと．"
            )
        data[column] = _synthesize_column(
            column, pl_dtype, np_field, num_records
        )

    return pl.DataFrame(data, schema=dict(schema))


class PerformanceBenchmark:
    """Performance benchmark for Polars + Rust I/O pipeline．"""

    def __init__(self, num_records: int = 10000):
        """Initialize benchmark．

        Args:
            num_records: Number of records to generate for testing
        """
        self.num_records = num_records
        self.process = psutil.Process()

    def _get_memory_mb(self) -> float:
        """Get current memory usage in MB．"""
        return self.process.memory_info().rss / 1024 / 1024

    def _create_hcpe_test_data_polars(self) -> pl.DataFrame:
        """Create HCPE test data as Polars DataFrame．"""
        return _create_test_data(
            get_hcpe_polars_schema(),
            get_hcpe_dtype(),
            self.num_records,
            "hcpe",
        )

    def _create_preprocessing_test_data_polars(
        self,
    ) -> pl.DataFrame:
        """Create preprocessing test data as Polars DataFrame．"""
        return _create_test_data(
            get_preprocessing_polars_schema(),
            get_preprocessing_dtype(),
            self.num_records,
            "preprocessing",
        )

    def benchmark_hcpe_io(self, output_dir: Path) -> dict:
        """Benchmark HCPE I/O performance．

        Args:
            output_dir: Directory to save test files

        Returns:
            Dict with benchmark results
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        results = {}

        logger.info(
            f"Benchmarking HCPE I/O with {self.num_records} records"
        )

        # Polars + Rust
        polars_data = self._create_hcpe_test_data_polars()
        polars_path = output_dir / "hcpe_test.feather"

        mem_before = self._get_memory_mb()
        start = time.perf_counter()
        save_hcpe_df(polars_data, polars_path)
        polars_save_time = time.perf_counter() - start

        start = time.perf_counter()
        _ = load_hcpe_df(polars_path)
        polars_load_time = time.perf_counter() - start
        mem_after = self._get_memory_mb()

        polars_file_size = polars_path.stat().st_size
        polars_memory = mem_after - mem_before

        results["polars"] = {
            "save_time": polars_save_time,
            "load_time": polars_load_time,
            "file_size_mb": polars_file_size / 1024 / 1024,
            "memory_mb": polars_memory,
        }

        logger.info(
            f"Polars: save={polars_save_time:.4f}s, load={polars_load_time:.4f}s, "
            f"size={polars_file_size / 1024 / 1024:.2f}MB"
        )

        return results

    def benchmark_preprocessing_io(
        self, output_dir: Path
    ) -> dict:
        """Benchmark preprocessing I/O performance．

        Args:
            output_dir: Directory to save test files

        Returns:
            Dict with benchmark results
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        results = {}

        logger.info(
            f"Benchmarking preprocessing I/O with {self.num_records} records"
        )

        # Polars + Rust
        polars_data = (
            self._create_preprocessing_test_data_polars()
        )
        polars_path = output_dir / "preprocessing_test.feather"

        mem_before = self._get_memory_mb()
        start = time.perf_counter()
        save_preprocessing_df(polars_data, polars_path)
        polars_save_time = time.perf_counter() - start

        start = time.perf_counter()
        _ = load_preprocessing_df(polars_path)
        polars_load_time = time.perf_counter() - start
        mem_after = self._get_memory_mb()

        polars_file_size = polars_path.stat().st_size
        polars_memory = mem_after - mem_before

        results["polars"] = {
            "save_time": polars_save_time,
            "load_time": polars_load_time,
            "file_size_mb": polars_file_size / 1024 / 1024,
            "memory_mb": polars_memory,
        }

        logger.info(
            f"Polars: save={polars_save_time:.4f}s, load={polars_load_time:.4f}s, "
            f"size={polars_file_size / 1024 / 1024:.2f}MB"
        )

        return results

    def benchmark_datasource_iteration(
        self, output_dir: Path
    ) -> dict:
        """Benchmark DataSource iteration performance．

        Args:
            output_dir: Directory with test files

        Returns:
            Dict with benchmark results
        """
        results = {}

        logger.info("Benchmarking DataSource iteration")

        # Create test files
        polars_path = output_dir / "iteration_test.feather"

        if not polars_path.exists():
            polars_data = (
                self._create_preprocessing_test_data_polars()
            )
            save_preprocessing_df(polars_data, str(polars_path))

        # Benchmark Polars iteration
        polars_datasource = FileDataSource(
            file_paths=[polars_path],
            array_type="preprocessing",
        )

        start = time.perf_counter()
        count = 0
        for name, df in polars_datasource.iter_batches_df():
            count += len(df)
        polars_iter_time = time.perf_counter() - start

        results["polars"] = {
            "iteration_time": polars_iter_time,
            "records_processed": count,
            "throughput": count / polars_iter_time,
        }

        logger.info(
            f"Polars iteration: {polars_iter_time:.4f}s, "
            f"throughput={count / polars_iter_time:.0f} records/s"
        )

        return results

    def print_summary(
        self,
        hcpe_results: dict,
        preprocessing_results: dict,
        iteration_results: dict,
    ) -> None:
        """Print benchmark summary．

        Args:
            hcpe_results: HCPE I/O benchmark results
            preprocessing_results: Preprocessing I/O benchmark results
            iteration_results: DataSource iteration benchmark results
        """
        print("\n" + "=" * 80)
        print("POLARS + RUST I/O PERFORMANCE BENCHMARK SUMMARY")
        print("=" * 80)

        print("\n### HCPE Data I/O ###")
        print(f"Records: {self.num_records:,}")
        print("\nPolars + Rust (.feather):")
        print(
            f"  Save: {hcpe_results['polars']['save_time']:.4f}s"
        )
        print(
            f"  Load: {hcpe_results['polars']['load_time']:.4f}s"
        )
        print(
            f"  Size: {hcpe_results['polars']['file_size_mb']:.2f} MB"
        )

        print("\n### Preprocessing Data I/O ###")
        print(f"Records: {self.num_records:,}")
        print("\nPolars + Rust (.feather):")
        print(
            f"  Save: {preprocessing_results['polars']['save_time']:.4f}s"
        )
        print(
            f"  Load: {preprocessing_results['polars']['load_time']:.4f}s"
        )
        print(
            f"  Size: {preprocessing_results['polars']['file_size_mb']:.2f} MB"
        )

        print("\n### DataSource Iteration ###")
        print("\nPolars:")
        print(
            f"  Time: {iteration_results['polars']['iteration_time']:.4f}s"
        )
        print(
            f"  Throughput: {iteration_results['polars']['throughput']:.0f} records/s"
        )

        print("\n" + "=" * 80)


def main(
    output_dir: Path = Path("/tmp/benchmark_polars"),
    num_records: int = 10000,
) -> None:
    """Run all benchmarks．

    Args:
        output_dir: Directory to save test files
        num_records: Number of records to generate
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    benchmark = PerformanceBenchmark(num_records=num_records)

    hcpe_results = benchmark.benchmark_hcpe_io(output_dir)
    preprocessing_results = (
        benchmark.benchmark_preprocessing_io(output_dir)
    )
    iteration_results = (
        benchmark.benchmark_datasource_iteration(output_dir)
    )

    benchmark.print_summary(
        hcpe_results, preprocessing_results, iteration_results
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark Polars + Rust I/O performance"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/benchmark_polars"),
        help="Directory to save test files",
    )
    parser.add_argument(
        "--num-records",
        type=int,
        default=10000,
        help="Number of records to generate",
    )

    args = parser.parse_args()
    main(
        output_dir=args.output_dir, num_records=args.num_records
    )
