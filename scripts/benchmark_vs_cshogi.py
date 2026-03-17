#!/usr/bin/env python3
"""Comprehensive benchmark: maou_shogi (Rust/PyO3) vs cshogi (C/Cython).

Measures:
  1. Legal move generation
  2. do_move / undo_move (push/pop)
  3. SFEN parse & serialize
  4. Perft (recursive move generation tree)
  5. Tsume-shogi solver (Df-Pn)
"""

import time
from typing import Callable

import cshogi
from cshogi import DfPn

from maou._rust.maou_shogi import PyBoard, solve_tsume

# ---------------------------------------------------------------------------
# Test positions (双玉SFEN: cshogi requires both kings)
# ---------------------------------------------------------------------------
POSITIONS = {
    "startpos": "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
    "midgame": "4+P2kl/7s1/5R3/7B1/9/9/9/9/K8 b GNrb3g3s3n3l17p 1",
    "endgame": "7nl/9/7kp/4r1N2/8P/6LG+p/9/9/K8 b R2b3g4s2n2l15p 1",
    "tsume": "9/5Pk2/9/8R/8B/9/9/9/K8 b 2Srb4g2s4n4l17p 1",
}

N_ITER = 100_000
N_WARMUP = 1_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _timeit(fn: Callable[[], object], n: int, warmup: int = N_WARMUP) -> float:
    """Run *fn* n times after warmup，return average microseconds."""
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1_000_000


def _ratio_str(maou_us: float, cshogi_us: float) -> str:
    if cshogi_us <= 0:
        return "N/A"
    r = maou_us / cshogi_us
    return f"{r:.2f}x"


def _print_header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _fmt_nps(nodes: int, ms: float) -> str:
    if ms <= 0:
        return "N/A"
    nps = nodes / (ms / 1000)
    if nps >= 1_000_000:
        return f"{nps / 1_000_000:.1f}M"
    if nps >= 1_000:
        return f"{nps / 1_000:.1f}K"
    return f"{nps:.0f}"


# ---------------------------------------------------------------------------
# 1. Legal move generation
# ---------------------------------------------------------------------------
def bench_legal_moves() -> None:
    _print_header(f"Legal Move Generation  (N={N_ITER:,})")
    header = (
        f"{'Position':<14}{'maou(us)':>10}{'cshogi(us)':>12}"
        f"{'ratio':>8}{'moves':>7}"
    )
    print(header)
    print("-" * len(header))
    for name, sfen in POSITIONS.items():
        mb = PyBoard()
        mb.set_sfen(sfen)
        cb = cshogi.Board(sfen)

        maou_us = _timeit(mb.legal_moves, N_ITER)
        cshogi_us = _timeit(lambda: list(cb.legal_moves), N_ITER)

        n_moves = len(mb.legal_moves())
        print(
            f"{name:<14}{maou_us:>10.2f}{cshogi_us:>12.2f}"
            f"{_ratio_str(maou_us, cshogi_us):>8}{n_moves:>7}"
        )


# ---------------------------------------------------------------------------
# 2. push / pop (do_move / undo_move)
# ---------------------------------------------------------------------------
def bench_push_pop() -> None:
    _print_header(f"push + pop (do/undo move)  (N={N_ITER:,})")
    header = (
        f"{'Position':<14}{'maou(us)':>10}{'cshogi(us)':>12}{'ratio':>8}"
    )
    print(header)
    print("-" * len(header))
    for name, sfen in POSITIONS.items():
        mb = PyBoard()
        mb.set_sfen(sfen)
        cb = cshogi.Board(sfen)

        m_move = mb.legal_moves()[0]
        c_move = list(cb.legal_moves)[0]

        def _maou_push_pop() -> None:
            mb.push(m_move)
            mb.pop()

        def _cshogi_push_pop() -> None:
            cb.push(c_move)
            cb.pop()

        maou_us = _timeit(_maou_push_pop, N_ITER)
        cshogi_us = _timeit(_cshogi_push_pop, N_ITER)

        print(
            f"{name:<14}{maou_us:>10.3f}{cshogi_us:>12.3f}"
            f"{_ratio_str(maou_us, cshogi_us):>8}"
        )


# ---------------------------------------------------------------------------
# 3. SFEN parse & serialize
# ---------------------------------------------------------------------------
def bench_sfen() -> None:
    _print_header(f"SFEN parse + serialize  (N={N_ITER:,})")
    header = (
        f"{'Position':<14}{'maou(us)':>10}{'cshogi(us)':>12}{'ratio':>8}"
    )
    print(header)
    print("-" * len(header))
    for name, sfen in POSITIONS.items():
        mb = PyBoard()
        cb = cshogi.Board()

        def _maou_sfen() -> None:
            mb.set_sfen(sfen)
            mb.sfen()

        def _cshogi_sfen() -> None:
            cb.set_sfen(sfen)
            cb.sfen()

        maou_us = _timeit(_maou_sfen, N_ITER)
        cshogi_us = _timeit(_cshogi_sfen, N_ITER)

        print(
            f"{name:<14}{maou_us:>10.2f}{cshogi_us:>12.2f}"
            f"{_ratio_str(maou_us, cshogi_us):>8}"
        )


# ---------------------------------------------------------------------------
# 4. Perft (recursive move generation tree)
# ---------------------------------------------------------------------------
def _perft_maou(board: PyBoard, depth: int) -> int:
    if depth == 0:
        return 1
    nodes = 0
    for m in board.legal_moves():
        board.push(m)
        nodes += _perft_maou(board, depth - 1)
        board.pop()
    return nodes


def _perft_cshogi(board: cshogi.Board, depth: int) -> int:  # type: ignore[type-arg]
    if depth == 0:
        return 1
    nodes = 0
    for m in board.legal_moves:
        board.push(m)
        nodes += _perft_cshogi(board, depth - 1)
        board.pop()
    return nodes


def bench_perft() -> None:
    depth = 3
    print(f"\n{'=' * 60}")
    print(f"  Perft depth={depth}  (single run)")
    print(f"{'=' * 60}")
    header = (
        f"{'Position':<14}{'maou(ms)':>10}{'cshogi(ms)':>12}"
        f"{'ratio':>8}{'nodes':>10}"
    )
    print(header)
    print("-" * len(header))

    for name, sfen in POSITIONS.items():
        mb = PyBoard()
        mb.set_sfen(sfen)

        t0 = time.perf_counter()
        m_nodes = _perft_maou(mb, depth)
        maou_ms = (time.perf_counter() - t0) * 1000

        cb = cshogi.Board(sfen)
        t0 = time.perf_counter()
        c_nodes = _perft_cshogi(cb, depth)
        cshogi_ms = (time.perf_counter() - t0) * 1000

        assert m_nodes == c_nodes, f"Node count mismatch: {m_nodes} vs {c_nodes}"
        ratio = _ratio_str(maou_ms, cshogi_ms)

        print(
            f"{name:<14}{maou_ms:>10.1f}{cshogi_ms:>12.1f}"
            f"{ratio:>8}{m_nodes:>10,}"
        )


# ---------------------------------------------------------------------------
# 5. Tsume-shogi solver (Df-Pn)
# ---------------------------------------------------------------------------
TSUME_PROBLEMS = {
    "tsume_5te": (
        "7nk/9/5R3/8p/6P2/9/9/9/K8 b SNPr2b4g3s2n4l15p 1",
        31,
    ),
    "tsume_9te": (
        "6s2/6l2/9/6BBk/9/9/9/9/K8 b RPr4g3s4n3l17p 1",
        31,
    ),
    "tsume_11te": (
        "4+P2kl/7s1/5R3/7B1/9/9/9/9/K8 b GNrb3g3s3n3l17p 1",
        31,
    ),
    "tsume_17te": (
        "9/5Pk2/9/8R/8B/9/9/9/K8 b 2Srb4g2s4n4l17p 1",
        31,
    ),
    "tsume_29te": (
        "l2+P5/2k4+L1/2n1p2B1/p1pp1spN1/4Ps3/PlPP2P2/1P1Sb4/1KG2+p3/LN7 w R2GPrgsn4p 1",
        31,
    ),
}


def bench_tsume() -> None:
    max_nodes = 10_000_000
    print(f"\n{'=' * 60}")
    print(f"  Tsume Solver (Df-Pn)  max_nodes={max_nodes:,}")
    print(f"{'=' * 60}")
    header = (
        f"{'Problem':<14}{'maou(ms)':>9}{'cshogi(ms)':>11}{'ratio':>7}"
        f"{'maou_nps':>10}{'cs_nps':>10}"
        f"{'m_nodes':>9}{'c_nodes':>9}"
        f"{'m_pv':>5}{'c_pv':>5}"
    )
    print(header)
    print("-" * len(header))

    for name, (sfen, depth) in TSUME_PROBLEMS.items():
        # maou (find_shortest=False — cshogi相当)
        t0 = time.perf_counter()
        result_fast = solve_tsume(
            sfen, depth=depth, nodes=max_nodes, find_shortest=False
        )
        maou_fast_ms = (time.perf_counter() - t0) * 1000
        mf_nodes = result_fast.nodes_searched
        mf_pv = len(result_fast.moves) if result_fast.status == "checkmate" else 0

        # maou (find_shortest=True — 最短手数探索)
        t0 = time.perf_counter()
        result = solve_tsume(sfen, depth=depth, nodes=max_nodes)
        maou_ms = (time.perf_counter() - t0) * 1000
        m_nodes = result.nodes_searched
        m_pv = len(result.moves) if result.status == "checkmate" else 0

        # cshogi
        dfpn = DfPn()
        dfpn.set_max_depth(depth)
        dfpn.set_max_search_node(max_nodes)
        board = cshogi.Board(sfen)
        t0 = time.perf_counter()
        found = dfpn.search(board)
        cshogi_ms = (time.perf_counter() - t0) * 1000
        c_nodes = dfpn.searched_node

        c_pv_list: list[str] = []
        if found:
            for m in dfpn.get_pv(board):
                c_pv_list.append(cshogi.move_to_usi(m))
        c_pv = len(c_pv_list)

        # maou (find_shortest=False) vs cshogi
        ratio_fast = _ratio_str(maou_fast_ms, cshogi_ms)
        print(
            f"{name:<14}{maou_fast_ms:>9.1f}{cshogi_ms:>11.1f}{ratio_fast:>7}"
            f"{_fmt_nps(mf_nodes, maou_fast_ms):>10}"
            f"{_fmt_nps(c_nodes, cshogi_ms):>10}"
            f"{mf_nodes:>9}{c_nodes:>9}"
            f"{mf_pv:>5}{c_pv:>5}"
        )
        # maou (find_shortest=True) — 参考
        ratio_full = _ratio_str(maou_ms, cshogi_ms)
        print(
            f"  (shortest)  {maou_ms:>9.1f}{'':>11}{ratio_full:>7}"
            f"{_fmt_nps(m_nodes, maou_ms):>10}{'':>10}"
            f"{m_nodes:>9}{'':>9}"
            f"{m_pv:>5}"
        )

        # Show PV
        if result_fast.status == "checkmate":
            print(f"  maou(fast): {' '.join(result_fast.moves)}")
        if result.status == "checkmate" and result.moves != result_fast.moves:
            print(f"  maou(short): {' '.join(result.moves)}")
        if found:
            print(f"  cshogi    : {' '.join(c_pv_list)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Benchmark: maou_shogi (Rust/PyO3) vs cshogi (C/Cython)")
    print(f"Python iterations per micro-bench: {N_ITER:,}")

    bench_legal_moves()
    bench_push_pop()
    bench_sfen()
    bench_perft()
    bench_tsume()

    print(f"\n{'=' * 60}")
    print("  Done.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
