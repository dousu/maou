#!/usr/bin/env python3
"""39手詰めの必要予算を推定する実験スクリプト.

正解PVの終盤からさかのぼり，各中間局面で詰みを解くのに必要なノード数を測定する．
残り手数と必要ノード数の関係から，全体を解くのに必要な予算を推定する．
"""

import time

from maou._rust.maou_shogi import PyBoard, solve_tsume

# 39手詰め問題
SFEN = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1"

# 正解PV (39手)
PV_MOVES = [
    "7b6b", "5b4c", "8b9c", "4c3d", "1b2c", "3d2c",
    "N*1e", "2c3b", "N*2d", "3b2b", "2d1b+", "2b3b",
    "1b2b", "3b2b", "4f1c", "2b1c", "9c3c", "1c1d",
    "3c2c", "1d1e", "P*1f", "1e1f", "P*1g", "1f1g",
    "5g6f", "1g1h", "2c2g", "1h1i", "8g8i", "S*6i",
    "8i6i", "6h6i+", "S*2h", "1i2i", "2h3g", "2i3i",
    "2g2h", "3i4i", "2h4h",
]

# テストする予算レベル
BUDGETS = [
    100_000,
    500_000,
    1_000_000,
    2_000_000,
    5_000_000,
    10_000_000,
    20_000_000,
    50_000_000,
    100_000_000,
]


def make_position_after_n_moves(n: int) -> str:
    """正解PVのn手目まで進めた局面のSFENを返す."""
    board = PyBoard()
    board.set_sfen(SFEN)
    for i in range(n):
        m = board.move_from_usi(PV_MOVES[i])
        board.push(m)
    return board.sfen()


def find_min_budget(sfen: str, remaining_moves: int) -> tuple[str, int, float]:
    """最小の解ける予算を見つける. (status, nodes, time_ms) を返す."""
    depth = remaining_moves + 4  # 余裕を持たせる
    result = None
    elapsed_ms = 0.0
    for budget in BUDGETS:
        t0 = time.perf_counter()
        result = solve_tsume(
            sfen,
            depth=depth,
            nodes=budget,
            timeout_secs=120,
            find_shortest=False,
            tt_gc_threshold=max(budget // 2, 1_000_000),
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if result.is_proven:
            return result.status, result.nodes_searched, elapsed_ms
    # 最大予算でも解けなかった場合，最後の結果を返す
    if result is None:
        msg = "BUDGETS is empty"
        raise ValueError(msg)
    return result.status, result.nodes_searched, elapsed_ms


def main() -> None:
    """メイン実行."""
    total_moves = len(PV_MOVES)
    print(f"39手詰め予算推定実験")
    print(f"SFEN: {SFEN}")
    print(f"正解PV: {total_moves}手")
    print()

    # 終盤から逆向きに探索: 残り1手, 3手, 5手, ... (攻め方手番のみ)
    print(f"{'残り手数':>8} {'状態':<16} {'ノード数':>12} {'時間(ms)':>10} {'局面SFEN'}")
    print("-" * 100)

    results = []

    # 残り手数を奇数ずつ (攻め方手番 = 王手側) で測定
    # 偶数手進めた局面 = 攻め方手番
    for moves_played in range(total_moves - 1, -1, -2):
        remaining = total_moves - moves_played
        sfen = make_position_after_n_moves(moves_played)
        status, nodes, elapsed_ms = find_min_budget(sfen, remaining)
        results.append((remaining, status, nodes, elapsed_ms))
        print(f"{remaining:>8} {status:<16} {nodes:>12,} {elapsed_ms:>10.1f} {sfen[:60]}...")

        # 解けなくなったら数回分追加してから終了
        if status == "unknown":
            # もう少し前も確認
            unsolved_count = sum(1 for _, s, _, _ in results if s == "unknown")
            if unsolved_count >= 3:
                break

    print()
    print("=" * 80)
    print("結果サマリ")
    print("=" * 80)

    # 解けた最大の残り手数
    solved = [(r, n, t) for r, s, n, t in results if s in ("checkmate", "checkmate_no_pv")]
    if solved:
        max_solved = max(solved, key=lambda x: x[0])
        print(f"解けた最長: 残り{max_solved[0]}手 (ノード数: {max_solved[1]:,}, 時間: {max_solved[2]:.0f}ms)")

    # 解けなかった最短の残り手数
    unsolved = [(r, n) for r, s, n, _ in results if s == "unknown"]
    if unsolved:
        min_unsolved = min(unsolved, key=lambda x: x[0])
        print(f"解けない最短: 残り{min_unsolved[0]}手 (最大予算{BUDGETS[-1]:,}ノードで不足)")

    # ノード数の増加傾向を分析
    print()
    print("ノード数の増加傾向 (解けた問題):")
    prev_nodes = None
    for remaining, status, nodes, _ in sorted(results):
        if status in ("checkmate", "checkmate_no_pv"):
            ratio = f"x{nodes / prev_nodes:.1f}" if prev_nodes and prev_nodes > 0 else "-"
            print(f"  残り{remaining:>3}手: {nodes:>12,} ノード ({ratio})")
            prev_nodes = nodes

    # 簡易外挿
    if len(solved) >= 3:
        import math

        solved_sorted = sorted(solved)
        # 最後の3点で指数フィット
        last3 = solved_sorted[-3:]
        # log(nodes) vs remaining で線形回帰
        xs = [r for r, _, _ in last3]
        ys = [math.log(n) if n > 0 else 0 for _, n, _ in last3]
        n = len(xs)
        sx = sum(xs)
        sy = sum(ys)
        sxy = sum(x * y for x, y in zip(xs, ys))
        sxx = sum(x * x for x in xs)
        denom = n * sxx - sx * sx
        if denom != 0:
            slope = (n * sxy - sx * sy) / denom
            intercept = (sy - slope * sx) / n
            # 39手の推定
            est_log_nodes = slope * 39 + intercept
            est_nodes = math.exp(est_log_nodes)
            print()
            print(f"指数外挿 (最後の3データ点):")
            print(f"  log(nodes) ≈ {slope:.3f} * remaining + {intercept:.3f}")
            print(f"  推定: 残り39手 → {est_nodes:,.0f} ノード ({est_nodes:.2e})")
            print(f"  (2手増えるごとにノード数が約 {math.exp(2 * slope):.1f}倍)")


if __name__ == "__main__":
    main()
