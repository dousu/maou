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
    MOVE_LABELS_NUM,
    get_hcpe_polars_schema,
    get_preprocessing_polars_schema,
)

logger = logging.getLogger(__name__)


def _assert_covers_schema(
    data: Mapping[str, object],
    schema: Mapping[str, object],
    label: str,
) -> None:
    """テストデータがスキーマの全列を持つことを確認する．

    ``pl.DataFrame(data, schema=schema)`` は列が欠けていると
    ``KeyError: '<column>'`` を polars の内部フレームで投げるだけで，
    「ベンチマークのテストデータがスキーマに追い付いていない」とは
    読み取れない．実際 ``moveWinRate`` (2026-08-12 追加) と
    ``bestMoveWinRate`` の 2 列が欠けたまま気付かれずにいた．
    どこを直せばよいかを名指しして落とす．

    Args:
        data: 列名 → 値リスト
        schema: polars スキーマ
        label: エラーメッセージ用のデータ種別名

    Raises:
        ValueError: スキーマにあって data に無い列があるとき
    """
    missing = [c for c in schema if c not in data]
    if missing:
        raise ValueError(
            f"{label} のベンチマーク用テストデータがスキーマに追い付いて"
            f"いない: {missing} が欠けている．"
            f"benchmark_polars_io._create_{label}_test_data_polars を"
            f"更新すること．"
        )


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
        schema = get_hcpe_polars_schema()

        data = {
            "hcp": [
                bytes([i % 256 for _ in range(32)])
                for i in range(self.num_records)
            ],
            "eval": [
                (i % 1000) - 500
                for i in range(self.num_records)
            ],
            "bestMove16": [
                i % 10000 for i in range(self.num_records)
            ],
            "gameResult": [
                i % 3 for i in range(self.num_records)
            ],
            "id": [
                f"id_{i:08d}" for i in range(self.num_records)
            ],
            "partitioningKey": [
                date(2025, 12, 25)
                for _ in range(self.num_records)
            ],
            "ratings": [
                [1500 + (i % 500), 1500 - (i % 500)]
                for i in range(self.num_records)
            ],
            "endgameStatus": [
                "Toryo" for _ in range(self.num_records)
            ],
            "moves": [
                100 + (i % 100) for i in range(self.num_records)
            ],
        }

        _assert_covers_schema(data, schema, "hcpe")
        return pl.DataFrame(data, schema=schema)

    def _create_preprocessing_test_data_polars(
        self,
    ) -> pl.DataFrame:
        """Create preprocessing test data as Polars DataFrame．"""
        schema = get_preprocessing_polars_schema()

        data = {
            "id": list(range(self.num_records)),
            "boardIdPositions": [
                np.arange(81, dtype=np.uint8)
                .reshape(9, 9)
                .tolist()
                for _ in range(self.num_records)
            ],
            "piecesInHand": [
                np.arange(14, dtype=np.uint8).tolist()
                for _ in range(self.num_records)
            ],
            "moveLabel": [
                np.random.rand(MOVE_LABELS_NUM)
                .astype(np.float32)
                .tolist()
                for _ in range(self.num_records)
            ],
            # moveWinRate は 2026-08-12 に preprocessing スキーマへ
            # 追加された．ここで作らないと ``pl.DataFrame(data,
            # schema=schema)`` が KeyError で落ちるため，
            # スキーマに列を足すときはこの辞書も必ず更新すること．
            "moveWinRate": [
                np.random.rand(MOVE_LABELS_NUM)
                .astype(np.float32)
                .tolist()
                for _ in range(self.num_records)
            ],
            "bestMoveWinRate": [
                float(i % 2) for i in range(self.num_records)
            ],
            "resultValue": [
                float(i % 2) for i in range(self.num_records)
            ],
        }

        _assert_covers_schema(data, schema, "preprocessing")
        return pl.DataFrame(data, schema=schema)

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
            cache_mode="memory",
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
