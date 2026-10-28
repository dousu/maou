#!/usr/bin/env python
"""探索値を新しい手数帯へ広げる前に，その帯の**局面数と探索速度**を実測する．

## 何のためのスクリプトか

探索値の蓄積は 1 帯で数十 GPU 時間〜数百 GPU 時間かかる．投入の前に
次の 2 つが分からないと，セッション数も `POSITIONS_PER_SESSION`
(`scripts/colab_search_values_job.py`) も決められない．

1. **対象局面数** — `search-values` は同一局面 (Zobrist hash) を 1 回しか
   探索しないので，生の行数ではなく**一意な hash の数**が効く．序盤ほど
   多数の対局で同じ局面を共有するので，生の行数からの比例では過大になる．
   さらに既存の蓄積 (別の帯の出力) と hash が重なる分は，前処理で後勝ちに
   なるだけで探索としては重複する．
2. **探索速度** — compass § TRIPWIRES「帯をまたいで速度を流用しない」．
   局面の性質 (詰みの有無，合法手数) で 1 局面あたりのコストが変わるので，
   その帯の局面を実際に探索して測る．

`search-values` 本体は `--max-positions` を適用した後の数しか知らないので，
ここで走査だけを独立に行う．選定は `select_positions` (本体と同じ関数) を
そのまま使う．

## 使い方

    # 局面数だけ (CPU で動く．GPU は要らない)
    scripts/probe_search_band.py --hcpe HCPE_TRAIN --min-ply 20 --max-ply 60 \\
        --split 40 --existing SV_DIR_A --existing SV_DIR_B

    # 速度も測る (search-values を N 局面だけ一時ディレクトリへ走らせる)
    scripts/probe_search_band.py --hcpe HCPE_TRAIN --min-ply 20 --max-ply 60 \\
        --split 40 --speed-positions 4000 --speed-out /content/shogi/tmp/probe \\
        --model teacher.onnx -- --tensorrt --cuda --trt-cache-dir /content/trt

`--` 以降は `maou utility search-values` へそのまま渡す．探索の設定
(`--playouts` / `--batch-size` / `--threads`) は本番と同じ既定値を
このスクリプトが付ける (`docs/performance.md` の決着済み値)．

最後に `PROBE_OK` (成功) または `PROBE_FAILED` を出す．
"""

from __future__ import annotations

import argparse
import glob
import itertools
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

#: 本番 (`scripts/colab_search_values_job.py`) と同じ探索設定．
PLAYOUTS = 800
BATCH_SIZE = 64
THREADS = 1

#: tqdm の `12.6pos/s` を拾う．
_RATE = re.compile(r"([0-9.]+)pos/s")


def _hcpe_files(root: Path) -> list[Path]:
    """HCPE feather を安定した順序で列挙する．"""
    if root.is_file():
        return [root]
    return sorted(
        Path(p)
        for p in glob.glob(
            str(root / "**" / "*.feather"), recursive=True
        )
    )


def _existing_ids(dirs: list[Path]) -> dict[Path, np.ndarray]:
    """既存の探索値ディレクトリごとに，昇順・一意な id を返す．"""
    out: dict[Path, np.ndarray] = {}
    for d in dirs:
        files = sorted(glob.glob(str(d / "*.feather")))
        if not files:
            raise SystemExit(f"no .feather under {d}")
        ids = np.concatenate(
            [
                pl.read_ipc(
                    f, columns=["id"], memory_map=False
                )["id"].to_numpy()
                for f in files
            ]
        )
        out[d] = np.unique(ids.astype(np.uint64))
        print(f"existing {d}: {len(out[d]):,} ids", flush=True)
    return out


def count_targets(
    hcpe: Path, min_ply: int, max_ply: int, splits: list[int]
) -> tuple[
    dict[tuple[int, int], int],
    dict[tuple[int, int], np.ndarray],
]:
    """帯ごとの生の行数と一意な hash を返す．

    Args:
        hcpe: HCPE のディレクトリ (train のみを指すこと)．
        min_ply: 帯の下限 (含む)．
        max_ply: 帯の上限 (含まない)．
        splits: 帯を細分する境界 (例: ``[40]`` で 20-39 / 40-59)．

    Returns:
        ``(生の行数, 昇順・一意な hash)`` を細分帯ごとに．
    """
    from maou._rust.maou_search import hcp_hashes
    from maou.app.pre_process.search_value import (
        select_positions,
    )

    edges = [
        min_ply,
        *sorted(s for s in splits if min_ply < s < max_ply),
        max_ply,
    ]
    bands = list(itertools.pairwise(edges))
    raw = {b: 0 for b in bands}
    chunks: dict[tuple[int, int], list[np.ndarray]] = {
        b: [] for b in bands
    }
    files = _hcpe_files(hcpe)
    t0 = time.time()
    for i, path in enumerate(files, 1):
        df = pl.read_ipc(path, memory_map=False)
        hcp = np.ascontiguousarray(
            np.stack(
                [
                    np.frombuffer(b, np.uint8)
                    for b in df["hcp"].to_numpy()
                ]
            )
        )
        hashes = np.asarray(hcp_hashes(hcp)).astype(np.uint64)
        ply = (
            df["id"]
            .str.split(".hcpe_")
            .list.last()
            .cast(pl.Int64, strict=False)
            .to_numpy()
        )
        for lo, hi in bands:
            raw[(lo, hi)] += int(
                ((ply >= lo) & (ply < hi)).sum()
            )
            rows = select_positions(
                df,
                hashes,
                min_ply=lo,
                max_ply=hi,
                max_positions=0,
                seed=0,
            )
            chunks[(lo, hi)].append(hashes[rows])
        print(
            f"scan {i}/{len(files)} {path.name} rows={len(df):,} ({time.time() - t0:.0f}s)",
            flush=True,
        )
    uniq = {
        b: np.unique(np.concatenate(c))
        if c
        else np.empty(0, np.uint64)
        for b, c in chunks.items()
    }
    return raw, uniq


def report_counts(
    raw: dict[tuple[int, int], int],
    uniq: dict[tuple[int, int], np.ndarray],
    existing: dict[Path, np.ndarray],
) -> None:
    """帯ごとの局面数と既存蓄積との重なりを出す．"""
    allu = np.unique(np.concatenate(list(uniq.values())))
    rows: list[tuple[str, int, np.ndarray]] = [
        (f"ply {lo}-{hi - 1}", raw[(lo, hi)], u)
        for (lo, hi), u in uniq.items()
    ]
    rows.append(("total (band union)", sum(raw.values()), allu))
    print("\n== targets ==", flush=True)
    for name, n_raw, u in rows:
        line = f"{name:<20} raw {n_raw:>12,}  unique {len(u):>12,}  ({len(u) / max(n_raw, 1):.3f})"
        new = u
        for d, ids in existing.items():
            ov = int(np.isin(u, ids, assume_unique=True).sum())
            line += f"  ∩{d.name} {ov:,}"
            new = new[~np.isin(new, ids, assume_unique=True)]
        if existing:
            line += f"  not-yet-searched {len(new):,}"
        print(line, flush=True)


def measure_speed(
    hcpe: Path,
    min_ply: int,
    max_ply: int,
    n: int,
    out: Path,
    model: Path,
    extra: list[str],
) -> bool:
    """`search-values` を `n` 局面だけ走らせ，速度と停止理由を出す．

    Returns:
        探索が rc=0 で終わり，`n` 局面ぶんの行が出たら True．
    """
    if out.exists():
        shutil.rmtree(out)
    cmd = [
        "maou",
        "utility",
        "search-values",
        "--input-path",
        str(hcpe),
        "--output-path",
        str(out),
        "--model-path",
        str(model),
        "--min-ply",
        str(min_ply),
        "--max-ply",
        str(max_ply),
        "--max-positions",
        str(n),
        "--seed",
        "0",
        "--playouts",
        str(PLAYOUTS),
        "--batch-size",
        str(BATCH_SIZE),
        "--threads",
        str(THREADS),
        *extra,
    ]
    print("\n$ " + " ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    rates: list[float] = []
    search_start: float | None = None
    assert proc.stdout is not None
    buf = ""
    while True:
        ch = proc.stdout.read(1)
        if not ch:
            break
        if ch not in "\r\n":
            buf += ch
            continue
        if "Searching positions" in buf:
            search_start = search_start or time.time()
            m = _RATE.search(buf)
            if m:
                rates.append(float(m.group(1)))
        elif buf.strip():
            print(buf, flush=True)
        buf = ""
    rc = proc.wait()
    wall = time.time() - t0
    print(f"search-values rc={rc} wall={wall:.0f}s", flush=True)
    files = sorted(glob.glob(str(out / "*.feather")))
    if rc != 0 or not files:
        return False
    d = pl.concat(
        [pl.read_ipc(f, memory_map=False) for f in files],
        how="diagonal",
    )
    searching = (
        time.time() - search_start
        if search_start
        else float("nan")
    )
    print("\n== speed ==", flush=True)
    print(f"rows {len(d):,} / requested {n:,}", flush=True)
    print(
        f"wall total {wall:.0f}s / searching phase {searching:.0f}s",
        flush=True,
    )
    if rates:
        tail = rates[len(rates) // 2 :]
        print(
            f"tqdm rate (second half median) {np.median(tail):.2f} pos/s = {np.median(tail) * 3600:,.0f} pos/h",
            flush=True,
        )
    print(
        f"searching-phase rate {len(d) / searching:.2f} pos/s = {len(d) / searching * 3600:,.0f} pos/h",
        flush=True,
    )
    elapsed = d["elapsedMs"].to_numpy().astype(np.float64)
    warmup = d["warmupMs"].to_numpy().astype(np.float64)
    playouts = d["playouts"].to_numpy()
    print(
        f"elapsedMs median {np.median(elapsed):.1f}"
        f" / warmupMs median {np.median(warmup):.1f}"
        f" / (elapsed+warmup) mean"
        f" {(elapsed + warmup).mean():.1f} ms",
        flush=True,
    )
    print(
        f"playouts median {np.median(playouts):.0f}"
        f" / <{PLAYOUTS} share"
        f" {(playouts < PLAYOUTS).mean():.3f}",
        flush=True,
    )
    print(
        f"stop {d.group_by('stop').len().sort('len', descending=True).to_dicts()}",
        flush=True,
    )
    print(
        "searchWinRate mean"
        f" {d['searchWinRate'].to_numpy().mean():.3f}",
        flush=True,
    )
    return len(d) == n


def main() -> int:
    """エントリポイント．"""
    argv = sys.argv[1:]
    extra: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1 :]
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--hcpe",
        type=Path,
        required=True,
        help="HCPE のディレクトリ (train のみ)",
    )
    ap.add_argument("--min-ply", type=int, required=True)
    ap.add_argument("--max-ply", type=int, required=True)
    ap.add_argument(
        "--split",
        type=int,
        action="append",
        default=[],
        help="帯を細分する境界 (複数可)",
    )
    ap.add_argument(
        "--existing",
        type=Path,
        action="append",
        default=[],
        help="既存の探索値ディレクトリ (重なりを数える)",
    )
    ap.add_argument(
        "--speed-positions",
        type=int,
        default=0,
        help="速度を測る局面数 (0 で測らない)",
    )
    ap.add_argument(
        "--speed-out",
        type=Path,
        help="速度測定の出力先 (毎回消して作り直す．本番の蓄積を指さないこと)",
    )
    ap.add_argument(
        "--model",
        type=Path,
        help="teacher ONNX (速度測定に必要)",
    )
    args = ap.parse_args(argv)
    if args.min_ply >= args.max_ply:
        ap.error("--min-ply must be < --max-ply")
    if args.speed_positions and (
        args.speed_out is None or args.model is None
    ):
        ap.error(
            "--speed-positions needs --speed-out and --model"
        )
    if args.speed_out is not None and any(
        args.speed_out.resolve() == e.resolve()
        for e in args.existing
    ):
        ap.error(
            "--speed-out must not be an existing accumulation"
        )

    existing = _existing_ids(args.existing)
    raw, uniq = count_targets(
        args.hcpe, args.min_ply, args.max_ply, args.split
    )
    report_counts(raw, uniq, existing)
    ok = True
    if args.speed_positions:
        ok = measure_speed(
            args.hcpe,
            args.min_ply,
            args.max_ply,
            args.speed_positions,
            args.speed_out,
            args.model,
            extra,
        )
    print("PROBE_OK" if ok else "PROBE_FAILED", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
