"""Benchmark Polars + Rust I/O performance．

このモジュールは，新しいPolars + Rust I/Oパイプラインと従来のnumpy実装の
パフォーマンスを比較するベンチマークを提供する．
"""

import logging
import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import psutil

from maou.domain.data.rust_io import (
    load_hcpe_df,
    load_preprocessing_df,
    save_hcpe_df,
    save_preprocessing_df,
)
from maou.domain.data.schema import (
    get_hcpe_dtype,
    get_hcpe_polars_schema,
    get_preprocessing_dtype,
    get_preprocessing_polars_schema,
)
from maou.domain.move.label import MOVE_LABELS_NUM
from maou.infra.file_system.file_data_source import (
    FileDataSource,
)


# Numpy I/O functions for benchmarking
def save_hcpe_array(array: np.ndarray, path: Path) -> None:
    """Save HCPE numpy array to .npy file."""
    np.save(path, array)


def load_hcpe_array(path: Path) -> np.ndarray:
    """Load HCPE numpy array from .npy file."""
    return np.load(path)


def save_preprocessing_array(
    array: np.ndarray, path: Path, bit_pack: bool = False
) -> None:
    """Save preprocessing numpy array to .npy file."""
    np.save(path, array)


def load_preprocessing_array(
    path: Path, bit_pack: bool = False
) -> np.ndarray:
    """Load preprocessing numpy array from .npy file."""
    return np.load(path)


logger = logging.getLogger(__name__)


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

    def _create_hcpe_test_data_numpy(self) -> np.ndarray:
        """Create HCPE test data as numpy array．"""
        dtype = get_hcpe_dtype()
        data = np.zeros(self.num_records, dtype=dtype)

        for i in range(self.num_records):
            data[i]["hcp"] = np.frombuffer(
                bytes([i % 256 for _ in range(32)]),
                dtype=np.uint8,
            )
            data[i]["eval"] = (i % 1000) - 500
            data[i]["bestMove16"] = i % 10000
            data[i]["gameResult"] = i % 3
            data[i]["id"] = f"id_{i:08d}"
            data[i]["partitioningKey"] = np.datetime64(
                "2025-12-25"
            )
            data[i]["ratings"] = [
                1500 + (i % 500),
                1500 - (i % 500),
            ]
            data[i]["endgameStatus"] = "Toryo"
            data[i]["moves"] = 100 + (i % 100)

        return data

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

        return pl.DataFrame(data, schema=schema)

    def _create_preprocessing_test_data_numpy(
        self,
    ) -> np.ndarray:
        """Create preprocessing test data as numpy array．"""
        dtype = get_preprocessing_dtype()
        data = np.zeros(self.num_records, dtype=dtype)

        for i in range(self.num_records):
            data[i]["id"] = i
            data[i]["boardIdPositions"] = np.arange(
                81, dtype=np.uint8
            ).reshape(9, 9)
            data[i]["piecesInHand"] = np.arange(
                14, dtype=np.uint8
            )
            data[i]["moveLabel"] = np.random.rand(
                MOVE_LABELS_NUM
            ).astype(np.float32)
            data[i]["resultValue"] = float(i % 2)

        return data

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
            "resultValue": [
                float(i % 2) for i in range(self.num_records)
            ],
        }

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

        # Numpy baseline
        numpy_data = self._create_hcpe_test_data_numpy()
        numpy_path = output_dir / "hcpe_test.npy"

        mem_before = self._get_memory_mb()
        start = time.perf_counter()
        save_hcpe_array(numpy_data, numpy_path)
        numpy_save_time = time.perf_counter() - start

        start = time.perf_counter()
        _ = load_hcpe_array(numpy_path)
        numpy_load_time = time.perf_counter() - start
        mem_after = self._get_memory_mb()

        numpy_file_size = numpy_path.stat().st_size
        numpy_memory = mem_after - mem_before

        results["numpy"] = {
            "save_time": numpy_save_time,
            "load_time": numpy_load_time,
            "file_size_mb": numpy_file_size / 1024 / 1024,
            "memory_mb": numpy_memory,
        }

        logger.info(
            f"Numpy: save={numpy_save_time:.4f}s, load={numpy_load_time:.4f}s, "
            f"size={numpy_file_size / 1024 / 1024:.2f}MB"
        )

        # Polars + Rust
        polars_data = self._create_hcpe_test_data_polars()
        polars_path = output_dir / "hcpe_test.feather"

        mem_before = self._get_memory_mb()
        start = time.perf_counter()
        save_hcpe_df(polars_data, str(polars_path))
        polars_save_time = time.perf_counter() - start

        start = time.perf_counter()
        _ = load_hcpe_df(str(polars_path))
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

        # Calculate improvements
        results["improvement"] = {
            "save_speedup": numpy_save_time / polars_save_time,
            "load_speedup": numpy_load_time / polars_load_time,
            "compression_ratio": numpy_file_size
            / polars_file_size,
        }

        logger.info(
            f"Speedup: save={results['improvement']['save_speedup']:.2f}x, "
            f"load={results['improvement']['load_speedup']:.2f}x, "
            f"compression={results['improvement']['compression_ratio']:.2f}x"
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

        # Numpy baseline
        numpy_data = (
            self._create_preprocessing_test_data_numpy()
        )
        numpy_path = output_dir / "preprocessing_test.npy"

        mem_before = self._get_memory_mb()
        start = time.perf_counter()
        save_preprocessing_array(
            numpy_data, numpy_path, bit_pack=True
        )
        numpy_save_time = time.perf_counter() - start

        start = time.perf_counter()
        _ = load_preprocessing_array(numpy_path, bit_pack=True)
        numpy_load_time = time.perf_counter() - start
        mem_after = self._get_memory_mb()

        numpy_file_size = numpy_path.stat().st_size
        numpy_memory = mem_after - mem_before

        results["numpy"] = {
            "save_time": numpy_save_time,
            "load_time": numpy_load_time,
            "file_size_mb": numpy_file_size / 1024 / 1024,
            "memory_mb": numpy_memory,
        }

        logger.info(
            f"Numpy: save={numpy_save_time:.4f}s, load={numpy_load_time:.4f}s, "
            f"size={numpy_file_size / 1024 / 1024:.2f}MB"
        )

        # Polars + Rust
        polars_data = (
            self._create_preprocessing_test_data_polars()
        )
        polars_path = output_dir / "preprocessing_test.feather"

        mem_before = self._get_memory_mb()
        start = time.perf_counter()
        save_preprocessing_df(polars_data, str(polars_path))
        polars_save_time = time.perf_counter() - start

        start = time.perf_counter()
        _ = load_preprocessing_df(str(polars_path))
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

        # Calculate improvements
        results["improvement"] = {
            "save_speedup": numpy_save_time / polars_save_time,
            "load_speedup": numpy_load_time / polars_load_time,
            "compression_ratio": numpy_file_size
            / polars_file_size,
        }

        logger.info(
            f"Speedup: save={results['improvement']['save_speedup']:.2f}x, "
            f"load={results['improvement']['load_speedup']:.2f}x, "
            f"compression={results['improvement']['compression_ratio']:.2f}x"
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
        numpy_path = output_dir / "iteration_test.npy"
        polars_path = output_dir / "iteration_test.feather"

        if not numpy_path.exists():
            numpy_data = (
                self._create_preprocessing_test_data_numpy()
            )
            save_preprocessing_array(
                numpy_data, numpy_path, bit_pack=True
            )

        if not polars_path.exists():
            polars_data = (
                self._create_preprocessing_test_data_polars()
            )
            save_preprocessing_df(polars_data, str(polars_path))

        # Benchmark numpy iteration
        numpy_datasource = FileDataSource(
            file_paths=[numpy_path],
            array_type="preprocessing",
            cache_mode="memory",
        )

        start = time.perf_counter()
        count = 0
        for name, arr in numpy_datasource.iter_batches():
            count += len(arr)
        numpy_iter_time = time.perf_counter() - start

        results["numpy"] = {
            "iteration_time": numpy_iter_time,
            "records_processed": count,
            "throughput": count / numpy_iter_time,
        }

        logger.info(
            f"Numpy iteration: {numpy_iter_time:.4f}s, "
            f"throughput={count / numpy_iter_time:.0f} records/s"
        )

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

        # Calculate improvement
        results["improvement"] = {
            "speedup": numpy_iter_time / polars_iter_time,
        }

        logger.info(
            f"Speedup: {results['improvement']['speedup']:.2f}x"
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
        print("\nNumpy (.npy):")
        print(
            f"  Save: {hcpe_results['numpy']['save_time']:.4f}s"
        )
        print(
            f"  Load: {hcpe_results['numpy']['load_time']:.4f}s"
        )
        print(
            f"  Size: {hcpe_results['numpy']['file_size_mb']:.2f} MB"
        )

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

        print("\nImprovement:")
        print(
            f"  Save speedup: {hcpe_results['improvement']['save_speedup']:.2f}x"
        )
        print(
            f"  Load speedup: {hcpe_results['improvement']['load_speedup']:.2f}x"
        )
        print(
            f"  Compression: {hcpe_results['improvement']['compression_ratio']:.2f}x"
        )

        print("\n### Preprocessing Data I/O ###")
        print(f"Records: {self.num_records:,}")
        print("\nNumpy (.npy):")
        print(
            f"  Save: {preprocessing_results['numpy']['save_time']:.4f}s"
        )
        print(
            f"  Load: {preprocessing_results['numpy']['load_time']:.4f}s"
        )
        print(
            f"  Size: {preprocessing_results['numpy']['file_size_mb']:.2f} MB"
        )

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

        print("\nImprovement:")
        print(
            f"  Save speedup: {preprocessing_results['improvement']['save_speedup']:.2f}x"
        )
        print(
            f"  Load speedup: {preprocessing_results['improvement']['load_speedup']:.2f}x"
        )
        print(
            f"  Compression: {preprocessing_results['improvement']['compression_ratio']:.2f}x"
        )

        print("\n### DataSource Iteration ###")
        print("\nNumpy:")
        print(
            f"  Time: {iteration_results['numpy']['iteration_time']:.4f}s"
        )
        print(
            f"  Throughput: {iteration_results['numpy']['throughput']:.0f} records/s"
        )

        print("\nPolars:")
        print(
            f"  Time: {iteration_results['polars']['iteration_time']:.4f}s"
        )
        print(
            f"  Throughput: {iteration_results['polars']['throughput']:.0f} records/s"
        )

        print("\nImprovement:")
        print(
            f"  Speedup: {iteration_results['improvement']['speedup']:.2f}x"
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
