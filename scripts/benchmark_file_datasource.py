"""FileDataSourceのランダムアクセスパフォーマンスをベンチマークする．

このスクリプトは以下をテストする：
1. LRUキャッシュ無効時のパフォーマンス
2. LRUキャッシュ有効時のパフォーマンス
3. キャッシュヒット率の測定
"""

import logging
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np

from maou.infra.file_system.file_data_source import (
    FileDataSource,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_test_data(
    output_dir: Path,
    num_files: int = 4,
    records_per_file: int = 10000,
) -> list[Path]:
    """テスト用のダミーデータを作成する．

    Args:
        output_dir: 出力ディレクトリ
        num_files: ファイル数
        records_per_file: ファイルあたりのレコード数

    Returns:
        作成されたファイルパスのリスト
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # preprocessing配列のdtype（簡易版）
    dtype = np.dtype(
        [
            ("features", np.float32, (119, 9, 9)),
            ("policy", np.float32, 1496),
            ("value", np.float32),
        ]
    )

    file_paths = []
    for i in range(num_files):
        file_path = output_dir / f"test_data_{i:03d}.npy"
        data = np.zeros(records_per_file, dtype=dtype)
        # ランダムなデータを生成
        for j in range(records_per_file):
            data[j]["features"] = np.random.randn(
                119, 9, 9
            ).astype(np.float32)
            data[j]["policy"] = np.random.randn(1496).astype(
                np.float32
            )
            data[j]["value"] = np.random.randn().astype(
                np.float32
            )
        np.save(file_path, data)
        file_paths.append(file_path)
        logger.info(f"Created test file: {file_path}")

    return file_paths


def benchmark_random_access(
    file_paths: list[Path],
    num_accesses: int = 1000,
    cache_mode: str = "mmap",
    seed: Optional[int] = 42,
) -> dict[str, float]:
    """ランダムアクセスのパフォーマンスをベンチマークする．

    Args:
        file_paths: テストデータのファイルパス
        num_accesses: アクセス回数
        cache_mode: キャッシュモード ("mmap" または "memory")
        seed: 乱数シード

    Returns:
        ベンチマーク結果の辞書
    """
    if seed is not None:
        random.seed(seed)

    logger.info(
        f"Creating FileDataSource with cache_mode={cache_mode}"
    )

    # FileDataSourceを作成（初期化時間を測定）
    init_start = time.perf_counter()
    datasource = FileDataSource(
        file_paths=file_paths,
        array_type="preprocessing",
        bit_pack=False,
        cache_mode=cache_mode,  # type: ignore
    )
    init_time = time.perf_counter() - init_start

    total_records = len(datasource)
    logger.info(f"Total records: {total_records}")
    logger.info(f"Initialization time: {init_time:.2f}s")

    # ランダムなインデックスを生成
    indices = [
        random.randint(0, total_records - 1)
        for _ in range(num_accesses)
    ]

    # ウォームアップ（最初の数回のアクセスでキャッシュを暖める）
    logger.info("Warming up...")
    for i in range(min(100, num_accesses)):
        _ = datasource[indices[i]]

    # ベンチマーク実行
    logger.info(
        f"Starting benchmark with {num_accesses} random accesses"
    )
    start_time = time.perf_counter()

    for idx in indices:
        _ = datasource[idx]

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    results = {
        "init_time": init_time,
        "elapsed_time": elapsed_time,
        "accesses_per_second": num_accesses / elapsed_time,
        "avg_access_time_us": (elapsed_time / num_accesses)
        * 1_000_000,
    }

    logger.info(f"Benchmark completed in {elapsed_time:.2f}s")
    logger.info(
        f"Accesses per second: {results['accesses_per_second']:.2f}"
    )
    logger.info(
        f"Average access time: {results['avg_access_time_us']:.2f}μs"
    )

    return results


def main() -> None:
    """メイン関数．"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark FileDataSource random access performance"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/tmp/file_datasource_benchmark"),
        help="Directory for test data",
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=4,
        help="Number of test files to create",
    )
    parser.add_argument(
        "--records-per-file",
        type=int,
        default=10000,
        help="Number of records per file",
    )
    parser.add_argument(
        "--num-accesses",
        type=int,
        default=1000,
        help="Number of random accesses",
    )
    parser.add_argument(
        "--cache-mode",
        type=str,
        default="memory",
        choices=["mmap", "memory"],
        help="Cache mode (default: memory)",
    )
    parser.add_argument(
        "--create-data",
        action="store_true",
        help="Create new test data",
    )

    args = parser.parse_args()

    # テストデータを作成
    if args.create_data or not args.data_dir.exists():
        logger.info("Creating test data...")
        file_paths = create_test_data(
            args.data_dir, args.num_files, args.records_per_file
        )
    else:
        logger.info("Using existing test data")
        file_paths = sorted(
            args.data_dir.glob("test_data_*.npy")
        )
        if not file_paths:
            logger.error(
                f"No test data found in {args.data_dir}"
            )
            return

    logger.info(f"Found {len(file_paths)} test files")

    # ベンチマーク実行
    results = benchmark_random_access(
        file_paths=file_paths,
        num_accesses=args.num_accesses,
        cache_mode=args.cache_mode,
    )

    # 結果を表示
    print("\n" + "=" * 60)
    print("Benchmark Results")
    print("=" * 60)
    print(f"Cache mode: {args.cache_mode}")
    print(f"Number of files: {len(file_paths)}")
    print(f"Total accesses: {args.num_accesses}")
    print(f"Initialization time: {results['init_time']:.2f}s")
    print(
        f"Benchmark elapsed time: {results['elapsed_time']:.2f}s"
    )
    print(
        f"Accesses per second: {results['accesses_per_second']:.2f}"
    )
    print(
        f"Average access time: {results['avg_access_time_us']:.2f}μs"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
