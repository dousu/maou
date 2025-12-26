#!/usr/bin/env python3
"""HCPEデータからユニーク局面数とメモリ使用量を推定するスクリプト．"""

import logging
import sys
from pathlib import Path

from tqdm.auto import tqdm

# maouモジュールのimport
from maou.app.pre_process.label import MOVE_LABELS_NUM
from maou.domain.board import shogi
from maou.domain.data.rust_io import load_hcpe_df
from maou.domain.data.schema import convert_hcpe_df_to_numpy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def count_unique_positions(
    input_path: Path,
) -> tuple[int, int, int]:
    """ユニーク局面数を計算する．

    Args:
        input_path: HCPEデータのディレクトリパス

    Returns:
        (ユニーク局面数, 総局面数, 推定メモリ使用量バイト数)
    """
    # 全ての.featherファイルを検索
    feather_files = sorted(input_path.glob("**/*.feather"))
    logger.info(
        f"Found {len(feather_files)} feather files in {input_path}"
    )

    if not feather_files:
        logger.error("No feather files found")
        return 0, 0, 0

    # ハッシュ値を収集（重複を許可）
    all_hashes = []
    total_positions = 0
    board = shogi.Board()

    for feather_file in tqdm(
        feather_files, desc="Processing files"
    ):
        try:
            # Load .feather file and convert to numpy for processing
            df = load_hcpe_df(feather_file)
            data = convert_hcpe_df_to_numpy(df)
            n = len(data)
            total_positions += n

            # 各局面のハッシュ値を計算
            for i in range(n):
                board.set_hcp(data["hcp"][i])
                all_hashes.append(board.hash())
        except Exception as e:
            logger.error(
                f"Error processing {feather_file}: {e}"
            )
            continue

    # ユニーク局面数を計算
    unique_hashes = set(all_hashes)
    unique_count = len(unique_hashes)

    logger.info(f"Total positions: {total_positions:,}")
    logger.info(f"Unique positions: {unique_count:,}")
    if unique_count > 0:
        logger.info(
            f"Duplication rate: {total_positions / unique_count:.2f}x"
        )
    else:
        logger.warning("No positions found")

    # メモリ使用量を推定
    # intermediate_dictの各エントリのサイズを計算
    # - hash_id (key): 8 bytes (uint64)
    # - count: 4 bytes (int32)
    # - winCount: 4 bytes (float32)
    # - moveLabelCount: MOVE_LABELS_NUM * 8 bytes (int64 array)
    # - features: 104 * 9 * 9 * 1 bytes (uint8 array)

    per_entry_size = (
        8  # hash_id
        + 4  # count
        + 4  # winCount
        + MOVE_LABELS_NUM
        * 8  # moveLabelCount (numpy defaults to int64 for bincount)
        + 104 * 9 * 9 * 1  # features
    )

    # Pythonの辞書オーバーヘッド（概算）
    # - dict entry overhead: ~200 bytes per entry (CPython 3.x)
    dict_overhead = 200

    total_per_entry = per_entry_size + dict_overhead
    estimated_memory = unique_count * total_per_entry

    logger.info("\n=== Memory Estimation ===")
    logger.info(f"Per entry size: {per_entry_size:,} bytes")
    logger.info(f"Dict overhead: {dict_overhead:,} bytes")
    logger.info(f"Total per entry: {total_per_entry:,} bytes")
    logger.info(
        f"Estimated memory: {estimated_memory / (1024**3):.2f} GB"
    )

    return unique_count, total_positions, estimated_memory


def main() -> None:
    """メインエントリポイント．"""
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input_path>")
        print(f"Example: {sys.argv[0]} hcpe/floodgate/2020")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    if not input_path.exists():
        logger.error(f"Path does not exist: {input_path}")
        sys.exit(1)

    if not input_path.is_dir():
        logger.error(f"Path is not a directory: {input_path}")
        sys.exit(1)

    unique_count, total_positions, estimated_memory = (
        count_unique_positions(input_path)
    )

    print("\n" + "=" * 60)
    print(f"Total positions: {total_positions:,}")
    print(f"Unique positions: {unique_count:,}")
    if unique_count > 0:
        print(
            f"Duplication rate: {total_positions / unique_count:.2f}x"
        )
        print(
            f"Estimated memory: {estimated_memory / (1024**3):.2f} GB"
        )
    else:
        print("No positions found")
    print("=" * 60)


if __name__ == "__main__":
    main()
