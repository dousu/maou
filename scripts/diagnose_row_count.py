#!/usr/bin/env python3
"""Diagnose row count discrepancy between scan_row_count and load_hcpe_df.

Usage:
    python scripts/diagnose_row_count.py /path/to/hcpe/feather/dir

This script compares three different row count methods for each feather file:
1. scan_row_count (pl.scan_ipc metadata) - used by StreamingHcpeDataSource
2. load_hcpe_df (Rust load_feather) - used by actual data loading
3. pyarrow direct read - to check number of RecordBatches in each file
"""

from __future__ import annotations

import sys
from pathlib import Path


def diagnose_files(directory: Path) -> None:
    """Run diagnostics on all feather files in directory."""
    import polars as pl
    import pyarrow.ipc as ipc

    from maou.domain.data.rust_io import load_hcpe_df
    from maou.infra.file_system.streaming_file_source import (
        is_arrow_ipc_file_format,
        scan_row_count,
    )

    feather_files = sorted(directory.rglob("*.feather"))
    if not feather_files:
        print(f"No .feather files found in {directory}")
        return

    print(f"Found {len(feather_files)} feather files in {directory}\n")
    print(
        f"{'File':<40} {'Format':<8} {'Batches':>8} "
        f"{'scan_row':>10} {'load_row':>10} {'pyarrow_row':>12} {'Match':>6}"
    )
    print("-" * 100)

    total_scan = 0
    total_load = 0
    total_pyarrow = 0
    misstatus_count = 0
    multi_batch_count = 0

    for fp in feather_files:
        # 1. Check format
        is_file_fmt = is_arrow_ipc_file_format(fp)
        fmt = "File" if is_file_fmt else "Stream"

        # 2. scan_row_count (metadata-based)
        scan_rows = scan_row_count(fp)

        # 3. load_hcpe_df (Rust load_feather)
        try:
            df = load_hcpe_df(fp)
            load_rows = len(df)
            del df
        except Exception as e:
            load_rows = -1
            print(f"  ERROR loading {fp.name}: {e}")

        # 4. PyArrow direct read - count batches and total rows
        try:
            if is_file_fmt:
                with open(fp, "rb") as f:
                    reader = ipc.open_file(f)
                    num_batches = reader.num_record_batches
                    pyarrow_rows = 0
                    batch_sizes = []
                    for i in range(num_batches):
                        batch = reader.get_batch(i)
                        pyarrow_rows += batch.num_rows
                        batch_sizes.append(batch.num_rows)
            else:
                with open(fp, "rb") as f:
                    reader = ipc.open_stream(f)
                    num_batches = 0
                    pyarrow_rows = 0
                    batch_sizes = []
                    for batch in reader:
                        num_batches += 1
                        pyarrow_rows += batch.num_rows
                        batch_sizes.append(batch.num_rows)
        except Exception as e:
            num_batches = -1
            pyarrow_rows = -1
            batch_sizes = []
            print(f"  ERROR pyarrow read {fp.name}: {e}")

        status = "OK" if scan_rows == load_rows == pyarrow_rows else "DIFF"
        if status == "DIFF":
            misstatus_count += 1
        if num_batches > 1:
            multi_batch_count += 1

        total_scan += scan_rows
        total_load += max(load_rows, 0)
        total_pyarrow += max(pyarrow_rows, 0)

        print(
            f"{fp.name:<40} {fmt:<8} {num_batches:>8} "
            f"{scan_rows:>10,} {load_rows:>10,} {pyarrow_rows:>12,} {status:>6}"
        )

        # Show batch size details for multi-batch files
        if num_batches > 1:
            print(f"  -> Batch sizes: {batch_sizes}")

    print("-" * 100)
    print(
        f"{'TOTAL':<40} {'':8} {'':>8} "
        f"{total_scan:>10,} {total_load:>10,} {total_pyarrow:>12,}"
    )
    print(f"\nFiles with misstatuses: {misstatus_count}")
    print(f"Files with multiple batches: {multi_batch_count}")

    # Also check to_batches behavior
    print("\n\n=== to_batches() behavior check ===")
    print("Testing whether Polars to_batches() produces multiple batches...\n")

    sample_file = feather_files[0]
    try:
        df = load_hcpe_df(sample_file)
        arrow_table = df.to_arrow()
        batches = arrow_table.to_batches()
        print(f"File: {sample_file.name}")
        print(f"  DataFrame rows: {len(df)}")
        print(f"  Arrow Table rows: {arrow_table.num_rows}")
        print(f"  Number of chunks per column: {arrow_table.column(0).num_chunks}")
        print(f"  to_batches() count: {len(batches)}")
        for i, b in enumerate(batches):
            print(f"    Batch {i}: {b.num_rows} rows")

        # Test with concat (simulating what hcpe_converter might do)
        if len(feather_files) >= 2:
            df2 = load_hcpe_df(feather_files[1])
            combined = pl.concat([df, df2])
            arrow_combined = combined.to_arrow()
            batches_combined = arrow_combined.to_batches()
            print(f"\n  After pl.concat of 2 files:")
            print(f"    Combined DataFrame rows: {len(combined)}")
            print(f"    Arrow Table rows: {arrow_combined.num_rows}")
            print(
                f"    Number of chunks per column: "
                f"{arrow_combined.column(0).num_chunks}"
            )
            print(f"    to_batches() count: {len(batches_combined)}")
            for i, b in enumerate(batches_combined):
                print(f"      Batch {i}: {b.num_rows} rows")
            print(
                f"    to_batches()[0] rows: {batches_combined[0].num_rows} "
                f"(would lose {arrow_combined.num_rows - batches_combined[0].num_rows} rows!)"
                if len(batches_combined) > 1
                else f"    to_batches()[0] rows: {batches_combined[0].num_rows} (no data loss)"
            )
    except Exception as e:
        print(f"  ERROR: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/diagnose_row_count.py /path/to/hcpe/feather/dir")
        sys.exit(1)

    target = Path(sys.argv[1])
    if not target.exists():
        print(f"Path does not exist: {target}")
        sys.exit(1)

    diagnose_files(target)
