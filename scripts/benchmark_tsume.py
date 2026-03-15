#!/usr/bin/env python3
"""Benchmark script: maou DfPn vs cshogi DfPn for tsume-shogi problems."""

import time

import cshogi
from cshogi import DfPn

from maou._rust.maou_shogi import solve_tsume

# 攻め方の玉を9九に配置(cshogiは片玉SFENの合法手生成にバグがあるため)
PROBLEMS = {
    "kosaka_9te": (
        "6s2/6l2/9/6BBk/9/9/9/9/K8 b RPr4g3s4n3l17p 1",
        31,
    ),
    "image2_11te": (
        "4+P2kl/7s1/5R3/7B1/9/9/9/9/K8 b GNrb3g3s3n3l17p 1",
        31,
    ),
    "17te": (
        "9/5Pk2/9/8R/8B/9/9/9/K8 b 2Srb4g2s4n4l17p 1",
        31,
    ),
    "image3_7te": (
        "7nl/9/7kp/4r1N2/8P/6LG+p/9/9/K8 b R2b3g4s2n2l15p 1",
        31,
    ),
    "tsume4_5te": (
        "7nk/9/5R3/8p/6P2/9/9/9/K8 b SNPr2b4g3s2n4l15p 1",
        31,
    ),
}


def fmt_nps(nodes: int, ms: float) -> str:
    """Format nodes/sec as human-readable string."""
    if ms <= 0:
        return "N/A"
    nps = nodes / (ms / 1000)
    if nps >= 1_000_000:
        return f"{nps / 1_000_000:.1f}M"
    if nps >= 1_000:
        return f"{nps / 1_000:.1f}K"
    return f"{nps:.0f}"


def main() -> None:
    max_nodes = 10_000_000

    print("=== Performance Comparison: maou vs cshogi ===")
    header = (
        f"{'Problem':<14} {'maou_ms':>9} {'cshogi_ms':>9} {'ratio':>7}"
        f" {'maou_nps':>10} {'cshogi_nps':>10}"
        f" {'maou_N':>8} {'cshogi_N':>8}"
        f" {'m_len':>5} {'c_len':>5}"
    )
    print(header)
    print("-" * len(header))

    for name, (sfen, depth) in PROBLEMS.items():
        # --- maou ---
        t0 = time.perf_counter()
        result = solve_tsume(sfen, depth=depth, nodes=max_nodes)
        maou_ms = (time.perf_counter() - t0) * 1000
        maou_nodes = result.nodes_searched
        maou_n_moves = len(result.moves) if result.status == "checkmate" else 0

        # --- cshogi ---
        dfpn = DfPn()
        dfpn.set_max_depth(depth)
        dfpn.set_max_search_node(max_nodes)
        board = cshogi.Board(sfen)
        t0 = time.perf_counter()
        found = dfpn.search(board)
        cshogi_ms = (time.perf_counter() - t0) * 1000
        cshogi_nodes = dfpn.searched_node

        cshogi_moves_list: list[str] = []
        if found:
            for m in dfpn.get_pv(board):
                cshogi_moves_list.append(cshogi.move_to_usi(m))
        cshogi_n_moves = len(cshogi_moves_list)

        ratio = maou_ms / cshogi_ms if cshogi_ms > 0 else float("inf")

        print(
            f"{name:<14} {maou_ms:>9.2f} {cshogi_ms:>9.2f} {ratio:>6.1f}x"
            f" {fmt_nps(maou_nodes, maou_ms):>10}"
            f" {fmt_nps(cshogi_nodes, cshogi_ms):>10}"
            f" {maou_nodes:>8} {cshogi_nodes:>8}"
            f" {maou_n_moves:>5} {cshogi_n_moves:>5}"
        )

        if result.status == "checkmate":
            print(f"  maou  : {' '.join(result.moves)}")
        if found:
            print(f"  cshogi: {' '.join(cshogi_moves_list)}")
        print()


if __name__ == "__main__":
    main()
