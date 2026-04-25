"""39手詰め問題における pn/dn 分布の分析スクリプト．

100M ノード探索後に Unknown となる 39手詰め問題の WorkingTT における
pn/dn 値の分布をグラフ化する．IDS 各 depth 反復終了時点の分布も出力する．
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import japanize_matplotlib  # noqa: F401

from maou._rust.maou_shogi import solve_tsume_pn_dn_dist

SFEN_39TE = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1"
MAX_NODES = 50_000_000
PN_UNIT = 16


def bucket_label_short(k: int) -> str:
    if k == 0:
        return "0"
    if k == 1:
        return "1"
    if k == 31:
        return "INF"
    return f"2^{k-1}"


def val_to_bucket(val: int) -> int:
    if val == 0:
        return 0
    if val == 0xFFFFFFFF:
        return 31
    bit = val.bit_length()
    return min(bit, 30)


def plot_final_dist(
    result: "TsumeResult",  # type: ignore[name-defined]
    pn_hist: list[int],
    dn_hist: list[int],
    joint_hist_flat: list[int],
    elapsed: float,
    out_path: str,
) -> None:
    pn_arr = np.array(pn_hist, dtype=np.int64)
    dn_arr = np.array(dn_hist, dtype=np.int64)
    joint_arr = np.array(joint_hist_flat, dtype=np.int64).reshape(32, 32)
    total_entries = int(pn_arr.sum())

    labels = [bucket_label_short(k) for k in range(32)]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"39手詰め pn/dn 分布 (最終) — {result.status} ({result.nodes_searched:,} nodes, {elapsed:.1f}s)",
        fontsize=13,
    )

    # --- pn ヒストグラム ---
    ax = axes[0, 0]
    ax.bar(range(32), pn_arr, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xticks(range(32))
    ax.set_xticklabels(labels, rotation=70, fontsize=7)
    ax.set_xlabel("pn バケット (log2 スケール)")
    ax.set_ylabel("エントリ数 (対数軸)")
    ax.set_title("pn 値の分布")
    ax.axvline(val_to_bucket(PN_UNIT), color="red", linestyle="--", linewidth=1,
               label=f"PN_UNIT={PN_UNIT} (bucket {val_to_bucket(PN_UNIT)})")
    ax.axvline(31, color="orange", linestyle="--", linewidth=1, label="INF (bucket 31)")
    ax.legend(fontsize=8)
    for i, v in enumerate(pn_arr):
        if v > 0:
            ax.text(i, v * 1.05, f"{v:,}", ha="center", va="bottom", fontsize=5, rotation=90)

    # --- dn ヒストグラム ---
    ax = axes[0, 1]
    ax.bar(range(32), dn_arr, color="coral", edgecolor="white", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xticks(range(32))
    ax.set_xticklabels(labels, rotation=70, fontsize=7)
    ax.set_xlabel("dn バケット (log2 スケール)")
    ax.set_ylabel("エントリ数 (対数軸)")
    ax.set_title("dn 値の分布")
    ax.axvline(val_to_bucket(PN_UNIT), color="red", linestyle="--", linewidth=1, label=f"PN_UNIT={PN_UNIT}")
    ax.axvline(31, color="orange", linestyle="--", linewidth=1, label="INF (bucket 31)")
    ax.legend(fontsize=8)
    for i, v in enumerate(dn_arr):
        if v > 0:
            ax.text(i, v * 1.05, f"{v:,}", ha="center", va="bottom", fontsize=5, rotation=90)

    # --- pn vs dn 2D ヒストグラム ---
    ax = axes[1, 0]
    masked = np.ma.masked_where(joint_arr == 0, joint_arr)
    im = ax.imshow(
        masked,
        origin="lower",
        aspect="auto",
        norm=mcolors.LogNorm(vmin=1, vmax=max(joint_arr.max(), 1)),
        cmap="viridis",
    )
    plt.colorbar(im, ax=ax, label="エントリ数 (対数)")
    ax.set_xticks(range(0, 32, 2))
    ax.set_xticklabels([labels[k] for k in range(0, 32, 2)], rotation=70, fontsize=7)
    ax.set_yticks(range(0, 32, 2))
    ax.set_yticklabels([labels[k] for k in range(0, 32, 2)], fontsize=7)
    ax.set_xlabel("dn バケット")
    ax.set_ylabel("pn バケット")
    ax.set_title("pn vs dn 2D 分布 (対数カラーマップ)")

    # --- 累積分布 ---
    ax = axes[1, 1]
    pn_cumsum = np.cumsum(pn_arr)
    dn_cumsum = np.cumsum(dn_arr)
    if total_entries > 0:
        pn_cdf = pn_cumsum / total_entries
        dn_cdf = dn_cumsum / total_entries
    else:
        pn_cdf = pn_cumsum
        dn_cdf = dn_cumsum
    ax.plot(range(32), pn_cdf, "b-o", markersize=4, label="pn CDF")
    ax.plot(range(32), dn_cdf, "r-o", markersize=4, label="dn CDF")
    ax.set_xticks(range(32))
    ax.set_xticklabels(labels, rotation=70, fontsize=7)
    ax.set_xlabel("バケット (log2 スケール)")
    ax.set_ylabel("累積割合")
    ax.set_title("pn/dn 累積分布関数")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(val_to_bucket(PN_UNIT), color="gray", linestyle="--", linewidth=1, alpha=0.7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"グラフを保存: {out_path}")
    plt.close(fig)


def plot_per_depth(
    per_depth: list[tuple[int, int, list[int], list[int], list[int]]],
    out_path: str,
) -> None:
    """IDS 各 depth 反復終了時点の pn/dn 分布を1枚のグラフにまとめる．"""
    if not per_depth:
        print("per_depth データなし: グラフをスキップ")
        return

    n = len(per_depth)
    labels = [bucket_label_short(k) for k in range(32)]

    # 2列レイアウト: 各行が1 IDS depth (pn/dn 並べて表示)
    fig, axes = plt.subplots(n, 2, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]  # shape を (1, 2) に統一

    fig.suptitle("IDS 各 depth 終了時点の pn/dn 分布 (TT 遷移前)", fontsize=14)

    for row, (ids_depth, nodes, pn_hist, dn_hist, _joint_flat) in enumerate(per_depth):
        pn_arr = np.array(pn_hist, dtype=np.int64)
        dn_arr = np.array(dn_hist, dtype=np.int64)
        total = int(pn_arr.sum())

        for col, (arr, color, label) in enumerate([
            (pn_arr, "steelblue", "pn"),
            (dn_arr, "coral", "dn"),
        ]):
            ax = axes[row, col]
            ax.bar(range(32), arr, color=color, edgecolor="white", linewidth=0.3)
            if arr.max() > 0:
                ax.set_yscale("log")
            ax.set_xticks(range(0, 32, 2))
            ax.set_xticklabels([labels[k] for k in range(0, 32, 2)], rotation=60, fontsize=6)
            ax.set_title(
                f"depth={ids_depth} | {label} 分布 | total={total:,} | nodes={nodes:,}",
                fontsize=9,
            )
            ax.set_ylabel("エントリ数", fontsize=7)
            ax.axvline(val_to_bucket(PN_UNIT), color="red", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.axvline(31, color="orange", linestyle="--", linewidth=0.8, alpha=0.7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"IDS per-depth グラフを保存: {out_path}")
    plt.close(fig)


def plot_per_depth_total_trend(
    per_depth: list[tuple[int, int, list[int], list[int], list[int]]],
    out_path: str,
) -> None:
    """IDS depth ごとの total entries 推移と pn/dn バケット構成の変化を可視化する．"""
    if not per_depth:
        return

    depths = [d for d, *_ in per_depth]
    nodes_list = [n for _, n, *_ in per_depth]
    totals = [int(sum(pn)) for _, _, pn, _, _ in per_depth]

    # 各 depth で INF (bucket 31) の割合
    inf_pn_frac = [pn[31] / max(sum(pn), 1) for _, _, pn, _, _ in per_depth]
    inf_dn_frac = [dn[31] / max(sum(dn), 1) for _, _, _, dn, _ in per_depth]
    zero_dn_frac = [dn[0] / max(sum(dn), 1) for _, _, _, dn, _ in per_depth]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("IDS depth 推移: WorkingTT エントリ数・pn/dn 構成変化", fontsize=13)

    x = range(len(depths))
    xlabels = [str(d) for d in depths]

    # エントリ総数の推移
    ax = axes[0]
    ax.bar(x, totals, color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("IDS depth")
    ax.set_ylabel("非空エントリ総数")
    ax.set_title("WorkingTT エントリ総数")
    for i, (v, nd) in enumerate(zip(totals, nodes_list)):
        ax.text(i, v * 1.01, f"{v:,}\n({nd//1000}K)", ha="center", va="bottom", fontsize=7)

    # pn INF 割合の推移
    ax = axes[1]
    ax.plot(x, [f * 100 for f in inf_pn_frac], "b-o", label="pn=INF %")
    ax.plot(x, [f * 100 for f in inf_dn_frac], "r-o", label="dn=INF %")
    ax.plot(x, [f * 100 for f in zero_dn_frac], "g-^", label="dn=0 %")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("IDS depth")
    ax.set_ylabel("割合 (%)")
    ax.set_title("特殊バケット割合の推移")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ノード消費の推移
    ax = axes[2]
    node_deltas = [nodes_list[0]] + [
        nodes_list[i] - nodes_list[i - 1] for i in range(1, len(nodes_list))
    ]
    ax.bar(x, [d / 1000 for d in node_deltas], color="purple", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("IDS depth")
    ax.set_ylabel("消費ノード数 (K)")
    ax.set_title("各 depth のノード消費量")
    for i, v in enumerate(node_deltas):
        ax.text(i, v / 1000 * 1.01, f"{v//1000}K", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"IDS trend グラフを保存: {out_path}")
    plt.close(fig)


def print_text_summary(
    pn_hist: list[int],
    dn_hist: list[int],
    per_depth: list[tuple[int, int, list[int], list[int], list[int]]],
) -> None:
    pn_arr = np.array(pn_hist, dtype=np.int64)
    dn_arr = np.array(dn_hist, dtype=np.int64)
    total_entries = int(pn_arr.sum())

    print(f"\n=== 最終スナップショット: pn 分布 ===")
    for k in range(32):
        if pn_arr[k] > 0:
            pct = 100.0 * pn_arr[k] / max(total_entries, 1)
            print(f"  [{k:2d}] {bucket_label_short(k):>8s}: {pn_arr[k]:>12,} ({pct:5.1f}%)")

    print(f"\n=== 最終スナップショット: dn 分布 ===")
    for k in range(32):
        if dn_arr[k] > 0:
            pct = 100.0 * dn_arr[k] / max(total_entries, 1)
            print(f"  [{k:2d}] {bucket_label_short(k):>8s}: {dn_arr[k]:>12,} ({pct:5.1f}%)")

    print("\n=== pn パーセンタイル ===")
    for pct_target in [50, 75, 90, 95, 99]:
        threshold = total_entries * pct_target / 100
        cum = 0
        for k, v in enumerate(pn_arr):
            cum += v
            if cum >= threshold:
                print(f"  p{pct_target:3d}: bucket {k:2d} ({bucket_label_short(k)}), cum={cum:,}")
                break

    print("\n=== dn パーセンタイル ===")
    for pct_target in [50, 75, 90, 95, 99]:
        threshold = total_entries * pct_target / 100
        cum = 0
        for k, v in enumerate(dn_arr):
            cum += v
            if cum >= threshold:
                print(f"  p{pct_target:3d}: bucket {k:2d} ({bucket_label_short(k)}), cum={cum:,}")
                break

    if per_depth:
        print("\n=== IDS depth ごとの WorkingTT 分布サマリ ===")
        print(f"  {'depth':>6}  {'nodes':>12}  {'total':>12}  {'pn=INF%':>8}  {'dn=INF%':>8}  {'dn=0%':>7}")
        for ids_depth, nodes, pn_h, dn_h, _ in per_depth:
            total = sum(pn_h)
            inf_pn = 100.0 * pn_h[31] / max(total, 1)
            inf_dn = 100.0 * dn_h[31] / max(total, 1)
            zero_dn = 100.0 * dn_h[0] / max(total, 1)
            print(f"  {ids_depth:>6}  {nodes:>12,}  {total:>12,}  {inf_pn:>7.1f}%  {inf_dn:>7.1f}%  {zero_dn:>6.1f}%")


def main() -> None:
    print(f"39手詰め pn/dn 分布分析 (max_nodes={MAX_NODES:,})")
    print(f"SFEN: {SFEN_39TE}")
    print("探索中...", flush=True)

    t0 = time.time()
    result, pn_hist, dn_hist, joint_hist_flat, per_depth = solve_tsume_pn_dn_dist(
        SFEN_39TE,
        depth=41,
        nodes=MAX_NODES,
        draw_ply=32767,
        timeout_secs=600,
        find_shortest=False,
    )
    elapsed = time.time() - t0

    print(f"結果: status={result.status}, nodes={result.nodes_searched:,}, elapsed={elapsed:.1f}s")
    print(f"WorkingTT 非空エントリ数: {sum(pn_hist):,}")
    print(f"IDS depth スナップショット数: {len(per_depth)}")

    plot_final_dist(result, pn_hist, dn_hist, joint_hist_flat, elapsed,
                    "/tmp/pn_dn_dist_39te.png")
    plot_per_depth(per_depth, "/tmp/pn_dn_dist_39te_per_depth.png")
    plot_per_depth_total_trend(per_depth, "/tmp/pn_dn_dist_39te_trend.png")
    print_text_summary(pn_hist, dn_hist, per_depth)


if __name__ == "__main__":
    main()
