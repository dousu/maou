#!/usr/bin/env python3
"""Benchmark script: legal move generation speed comparison (maou vs cshogi).

All SFENs use both kings (双玉) for cshogi compatibility.
"""

import time

import cshogi

from maou._rust.maou_shogi import PyBoard

# 双玉SFEN (cshogiは片玉だと合法手生成にバグが出るため)
POSITIONS = {
    "startpos": "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
    "midgame": "4+P2kl/7s1/5R3/7B1/9/9/9/9/K8 b GNrb3g3s3n3l17p 1",
    "endgame": "7nl/9/7kp/4r1N2/8P/6LG+p/9/9/K8 b R2b3g4s2n2l15p 1",
    "tsume": "9/5Pk2/9/8R/8B/9/9/9/K8 b 2Srb4g2s4n4l17p 1",
}

N = 100_000


def bench_legal_moves() -> None:
    """合法手生成の速度を比較する．"""
    print("=== Legal Move Generation Speed ===")
    print(f"  iterations: {N:,}")
    print()
    header = (
        f"{'Position':<12} {'maou_us':>10} {'cshogi_us':>10}"
        f" {'ratio':>8} {'moves':>7}"
    )
    print(header)
    print("-" * len(header))

    for name, sfen in POSITIONS.items():
        mb = PyBoard()
        mb.set_sfen(sfen)
        cb = cshogi.Board(sfen)

        # warmup
        for _ in range(1000):
            mb.legal_moves()
        for _ in range(1000):
            list(cb.legal_moves)

        # maou
        t0 = time.perf_counter()
        for _ in range(N):
            moves_m = mb.legal_moves()
        maou_us = (time.perf_counter() - t0) / N * 1_000_000

        # cshogi
        t0 = time.perf_counter()
        for _ in range(N):
            moves_c = list(cb.legal_moves)
        cshogi_us = (time.perf_counter() - t0) / N * 1_000_000

        ratio = maou_us / cshogi_us if cshogi_us > 0 else float("inf")
        print(
            f"{name:<12} {maou_us:>10.2f} {cshogi_us:>10.2f}"
            f" {ratio:>7.1f}x {len(moves_m):>7}"
        )

    print()


def bench_do_undo_move() -> None:
    """do_move + undo_move の速度を比較する．"""
    print("=== do_move + undo_move Speed ===")
    print(f"  iterations: {N:,}")
    print()
    header = f"{'Position':<12} {'maou_us':>10} {'cshogi_us':>10} {'ratio':>8}"
    print(header)
    print("-" * len(header))

    for name, sfen in POSITIONS.items():
        mb = PyBoard()
        mb.set_sfen(sfen)
        cb = cshogi.Board(sfen)

        first_maou = mb.legal_moves()[0]
        first_cshogi = list(cb.legal_moves)[0]

        # warmup
        for _ in range(1000):
            mb.push(first_maou)
            mb.pop()
        for _ in range(1000):
            cb.push(first_cshogi)
            cb.pop()

        # maou
        t0 = time.perf_counter()
        for _ in range(N):
            mb.push(first_maou)
            mb.pop()
        maou_us = (time.perf_counter() - t0) / N * 1_000_000

        # cshogi
        t0 = time.perf_counter()
        for _ in range(N):
            cb.push(first_cshogi)
            cb.pop()
        cshogi_us = (time.perf_counter() - t0) / N * 1_000_000

        ratio = maou_us / cshogi_us if cshogi_us > 0 else float("inf")
        print(f"{name:<12} {maou_us:>10.3f} {cshogi_us:>10.3f} {ratio:>7.1f}x")

    print()


def main() -> None:
    bench_legal_moves()
    bench_do_undo_move()


if __name__ == "__main__":
    main()
