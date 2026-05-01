"""39手詰め問題における pn/dn 分布の分析スクリプト．

50M ノード探索後に Unknown となる 39手詰め問題の WorkingTT における
pn/dn 値の分布をグラフ化する．IDS 各 depth 反復終了時点の分布も出力する．

分布フィット: バケット空間での対数正規分布 (k ~ LogNormal(μ_ln, σ_ln))．
  PN_UNIT が下限として機能するため，バケット空間での正規分布は左裾が切り取られ
  右裾型 (right-skewed) になる．対数正規分布がより自然な目標分布である．
"""

import time
import numpy as np
import scipy.stats as stats
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


def _fit_lognormal_to_histogram(
    counts_1_30: np.ndarray,
) -> tuple[float, float, float, np.ndarray] | None:
    """bucket 1-30 の 30 要素カウント配列に対数正規分布をフィット．

    バケット値 k (1-30) に対して k ~ LogNormal(μ_ln, σ_ln) を仮定．
    ln(k) の重み付き平均・分散からパラメータを推定する．

    Returns (mu_ln, sigma_ln, kl_div, curve) or None if too few data.
    curve は bucket 1-30 上の連続フィットカーブ (total スケール，300 点)．
    """
    xs = np.arange(1, 31, dtype=float)  # バケット値 1-30
    ys = counts_1_30.astype(float)
    total = ys.sum()
    if total < 2:
        return None

    eps = 1e-10
    ln_xs = np.log(xs)

    # ln(bucket) の重み付き平均・分散
    mu_ln = float(np.dot(ln_xs, ys) / total)
    var_ln = float(np.dot((ln_xs - mu_ln) ** 2, ys) / total)
    sigma_ln = np.sqrt(max(var_ln, 0.01))

    # 離散確率質量 (正規化)
    pdf_at_xs = stats.lognorm.pdf(xs, s=sigma_ln, scale=np.exp(mu_ln))
    p_fit = pdf_at_xs / (pdf_at_xs.sum() + eps)
    p_data = ys / (total + eps)

    # KL divergence D(data || lognormal)
    kl = float(np.sum(p_data[p_data > 0] * np.log(p_data[p_data > 0] / (p_fit[p_data > 0] + eps))))

    # 連続フィットカーブ (total にスケール，bucket 1-30 の範囲)
    xs_cont = np.linspace(1.0, 30.0, 300)
    pdf_cont = stats.lognorm.pdf(xs_cont, s=sigma_ln, scale=np.exp(mu_ln))
    curve = pdf_cont / (pdf_at_xs.sum() + eps) * total

    return mu_ln, sigma_ln, kl, curve


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


def plot_intermediate_dist(
    pn_hist: list[int],
    dn_hist: list[int],
    per_depth: list[tuple[int, int, list[int], list[int], list[int]]],
    out_path: str,
    snapshot_label: str = "最終",
) -> None:
    """中間エントリ (0 < pn/dn < INF) のみの分布と対数正規分布フィットを可視化する．

    - bucket 0 (pn=0 or dn=0) と bucket 31 (INF) を除外
    - バケット 1-30 のみ対象
    - 対数正規分布フィット (k ~ LogNormal(μ_ln, σ_ln)) を重ね表示
    """
    pn_arr = np.array(pn_hist, dtype=np.int64)
    dn_arr = np.array(dn_hist, dtype=np.int64)

    # 中間エントリのみ抽出 (bucket 1-30)
    interm_range = range(1, 31)
    pn_interm = pn_arr[list(interm_range)]
    dn_interm = dn_arr[list(interm_range)]
    pn_total = int(pn_interm.sum())
    dn_total = int(dn_interm.sum())

    labels = [bucket_label_short(k) for k in interm_range]
    xs = np.arange(len(interm_range))

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        f"WorkingTT 中間エントリ (0 < pn/dn < INF) 分布と対数正規分布フィット [{snapshot_label}]",
        fontsize=14,
    )

    for col, (arr, total, color, name) in enumerate([
        (pn_interm, pn_total, "steelblue", "pn"),
        (dn_interm, dn_total, "coral", "dn"),
    ]):
        ax = axes[col]
        ax.bar(xs, arr, color=color, alpha=0.7, label=f"{name} (中間のみ, total={total:,})")

        if total > 0:
            result = _fit_lognormal_to_histogram(arr)
            if result is not None:
                mu_ln, sigma_ln, kl, curve = result
                median_bucket = np.exp(mu_ln)
                xs_cont = np.linspace(1, 30, 300)
                ax.plot(xs_cont - 1, curve, "k-", linewidth=2,
                        label=f"対数正規分布フィット\n(μ_ln={mu_ln:.2f}, σ_ln={sigma_ln:.2f})\n"
                              f"中央値={median_bucket:.1f} bucket")
                ax.set_title(
                    f"{name} 中間エントリ分布\n"
                    f"(bucket 1-30, KL={kl:.3f}, 中央値={median_bucket:.1f}bucket, σ_ln={sigma_ln:.2f})"
                )
            else:
                ax.set_title(f"{name} 中間エントリ分布 (フィット不可)")
        else:
            ax.set_title(f"{name} 中間エントリ分布 (データなし)")

        ax.set_xticks(xs[::2])
        ax.set_xticklabels([labels[k] for k in range(0, len(labels), 2)], rotation=60, fontsize=7)
        ax.set_xlabel("バケット (log2: bucket k = 2^(k-1) <= val < 2^k)")
        ax.set_ylabel("エントリ数")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"中間エントリ分布グラフを保存: {out_path}")
    plt.close(fig)


def plot_intermediate_per_depth(
    per_depth: list[tuple[int, int, list[int], list[int], list[int]]],
    out_path: str,
) -> None:
    """IDS 各 depth で中間エントリの分布がどう変化するかを可視化する．"""
    # depth が単調増加するスナップショットのみを使用する
    outer = []
    prev_d = -1
    for snap in per_depth:
        d = snap[0]
        if d > prev_d:
            outer.append(snap)
            prev_d = d
        else:
            break

    if not outer:
        print("IDS per-depth データなし")
        return

    n = len(outer)
    labels = [bucket_label_short(k) for k in range(1, 31)]
    fig, axes = plt.subplots(n, 2, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle("IDS 各 depth での中間エントリ (0 < pn/dn < INF) 分布と対数正規フィット", fontsize=13)

    for row, (ids_depth, nodes, pn_h, dn_h, _) in enumerate(outer):
        pn_arr = np.array(pn_h, dtype=np.int64)
        dn_arr = np.array(dn_h, dtype=np.int64)

        # 中間のみ (bucket 1-30)
        pn_interm = pn_arr[1:31]
        dn_interm = dn_arr[1:31]

        xs = np.arange(len(pn_interm))

        for col, (arr, color, name) in enumerate([
            (pn_interm, "steelblue", "pn"),
            (dn_interm, "coral", "dn"),
        ]):
            ax = axes[row, col]
            total = int(arr.sum())
            ax.bar(xs, arr, color=color, alpha=0.7, linewidth=0.3)
            title_stats = "N/A"
            if total > 0:
                result = _fit_lognormal_to_histogram(arr)
                if result is not None:
                    mu_ln, sigma_ln, kl, curve = result
                    median_bucket = np.exp(mu_ln)
                    xs_cont = np.linspace(1, 30, 300)
                    ax.plot(xs_cont - 1, curve, "k-", linewidth=1.5)
                    title_stats = f"KL={kl:.3f} | 中央値={median_bucket:.1f} | σ_ln={sigma_ln:.2f}"

            ax.set_title(
                f"depth={ids_depth} | {name} 中間 | total={total:,} | nodes={nodes:,} | {title_stats}",
                fontsize=8,
            )
            ax.set_xticks(xs[::2])
            ax.set_xticklabels([labels[k] for k in range(0, 30, 2)], rotation=60, fontsize=6)
            ax.set_ylabel("エントリ数", fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"IDS per-depth 中間エントリグラフを保存: {out_path}")
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


def analyze_kl_by_bucket(
    per_depth: list[tuple[int, int, list[int], list[int], list[int]]],
    out_path: str,
) -> None:
    """課題 D: depth ごとの per-bucket KL 寄与度を可視化する．

    各 depth の中間エントリ (bucket 1-30) に対し KL(P||Q) = Σ_k p_k * log(p_k/q_k) を
    bucket k ごとに分解する．Q は対数正規フィット，正値は実分布の過剰，負値は不足を示す．
    特定 bucket へのスパイクが KL 劣化の主因かを判定する (課題 D 診断)．
    """
    outer: list[tuple[int, int, list[int], list[int], list[int]]] = []
    prev_d = -1
    for snap in per_depth:
        d = snap[0]
        if d > prev_d:
            outer.append(snap)
            prev_d = d
        else:
            break

    if not outer:
        print("per-bucket KL 分析: データなし")
        return

    n = len(outer)
    bucket_xs = np.arange(1, 31, dtype=float)
    labels = [bucket_label_short(k) for k in range(1, 31)]
    cmap = plt.colormaps.get_cmap("viridis")
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]
    eps = 1e-10

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        "課題 D — per-bucket KL 寄与度 (depth 増加に伴う KL 劣化の分析)\n"
        "正値: 実分布 > 対数正規フィット (過剰)，負値: 実分布 < 対数正規 (不足)",
        fontsize=12,
    )

    print("\n=== per-bucket KL 寄与: 最大寄与 bucket (depth 別) ===")

    for col, name in enumerate(["pn", "dn"]):
        ax = axes[col]

        for i, (ids_depth, _nodes, pn_h, dn_h, _jh) in enumerate(outer):
            raw_h = pn_h if name == "pn" else dn_h
            arr = np.array(raw_h[1:31], dtype=np.float64)
            total = arr.sum()
            if total < 2:
                continue

            result = _fit_lognormal_to_histogram(arr.astype(np.int64))
            if result is None:
                continue

            mu_ln, sigma_ln, kl_total, _ = result
            pdf_at_xs = stats.lognorm.pdf(bucket_xs, s=sigma_ln, scale=np.exp(mu_ln))
            q_k = pdf_at_xs / (pdf_at_xs.sum() + eps)
            p_k = arr / (total + eps)

            kl_by_bucket = np.zeros(30)
            mask = p_k > 0
            kl_by_bucket[mask] = p_k[mask] * np.log(p_k[mask] / (q_k[mask] + eps))

            ax.plot(
                np.arange(30), kl_by_bucket,
                color=colors[i], linewidth=1.5, marker="o", markersize=3,
                label=f"depth={ids_depth} (KL={kl_total:.3f})",
            )

            max_idx = int(np.argmax(np.abs(kl_by_bucket)))
            print(f"  {name} depth={ids_depth:>3}: KL={kl_total:.3f}, "
                  f"max寄与 bucket {max_idx + 1} ({bucket_label_short(max_idx + 1)}) = {kl_by_bucket[max_idx]:.4f}")

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xticks(range(0, 30, 2))
        ax.set_xticklabels([labels[k] for k in range(0, 30, 2)], rotation=60, fontsize=7)
        ax.set_xlabel("bucket k (log2 スケール, 1-30)")
        ax.set_ylabel("KL 寄与 = p_k · log(p_k / q_k)")
        ax.set_title(f"{name}: per-bucket KL 寄与度 (bucket 1-30)")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"per-bucket KL 寄与グラフを保存: {out_path}")
    plt.close(fig)


def analyze_pn_tail(
    joint_hist_flat: list[int],
    per_depth: list[tuple[int, int, list[int], list[int], list[int]]],
    out_path: str,
    high_pn_threshold: int = 20,
) -> None:
    """課題 C: pn 右テール (bucket >= high_pn_threshold) の dn 分布を分析する．

    joint_hist から高 pn エントリの dn 分布を抽出し，ノードの性質を推定する:
      高 pn + 低 dn (bucket <= 6) → AND ノード候補 (WPN sum 累積が主因)
      高 pn + 高 dn (bucket >= 12) → OR ノード候補 (全子が高コスト)
    """
    joint_final = np.array(joint_hist_flat, dtype=np.int64).reshape(32, 32)
    labels = [bucket_label_short(k) for k in range(32)]

    dn_of_high_pn = joint_final[high_pn_threshold:, :].sum(axis=0)
    total_high_pn = int(dn_of_high_pn.sum())
    and_count = int(dn_of_high_pn[:7].sum())
    or_count = int(dn_of_high_pn[12:].sum())
    and_pct = 100.0 * and_count / max(total_high_pn, 1)
    or_pct = 100.0 * or_count / max(total_high_pn, 1)

    outer: list[tuple[int, int, np.ndarray]] = []
    prev_d = -1
    for snap in per_depth:
        d = snap[0]
        if d > prev_d:
            jh = np.array(snap[4], dtype=np.int64).reshape(32, 32)
            dn_dist = jh[high_pn_threshold:, :].sum(axis=0)
            outer.append((d, snap[1], dn_dist))
            prev_d = d
        else:
            break

    print(f"\n=== pn 右テール分析 (bucket >= {high_pn_threshold}) ===")
    print(f"  最終スナップショット: 高 pn エントリ = {total_high_pn:,}")
    print(f"    AND候補 (dn<=2^6):  {and_count:,} ({and_pct:.1f}%)")
    print(f"    OR候補  (dn>=2^12): {or_count:,} ({or_pct:.1f}%)")
    if outer:
        print("  depth 別内訳:")
        for d, _, dn_d in outer:
            tot = int(dn_d.sum())
            a = int(dn_d[:7].sum())
            o = int(dn_d[12:].sum())
            print(f"    depth={d:>3}: total={tot:>8,}, AND候補={a:>7,} ({100*a/max(tot,1):.1f}%), "
                  f"OR候補={o:>7,} ({100*o/max(tot,1):.1f}%)")

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        f"課題 C: pn 右テール (bucket ≥ {high_pn_threshold}, pn ≥ {2**(high_pn_threshold-1):,}) の dn 分布分析\n"
        f"高 pn + 低 dn → AND ノード候補，高 pn + 高 dn → OR ノード候補",
        fontsize=12,
    )

    # --- (0,0): 最終スナップショットの高 pn エントリの dn 分布 ---
    ax = axes[0, 0]
    bar_colors = [
        "#d73027" if k <= 6 else "#4575b4" if k >= 12 else "#74add1"
        for k in range(32)
    ]
    ax.bar(range(32), dn_of_high_pn, color=bar_colors, edgecolor="white", linewidth=0.3)
    if dn_of_high_pn.max() > 0:
        ax.set_yscale("log")
    ax.set_xticks(range(0, 32, 2))
    ax.set_xticklabels([labels[k] for k in range(0, 32, 2)], rotation=60, fontsize=7)
    ax.set_xlabel("dn バケット")
    ax.set_ylabel("エントリ数 (対数軸)")
    ax.set_title(
        f"高 pn エントリの dn 分布 [最終]\n"
        f"total={total_high_pn:,} | 赤=AND候補(dn≤64) | 青=OR候補(dn≥4096)",
    )
    ax.axvline(6.5, color="red", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.axvline(11.5, color="blue", linestyle="--", linewidth=1.2, alpha=0.8)
    ymax = dn_of_high_pn.max() if dn_of_high_pn.max() > 0 else 1
    ax.text(3, ymax * 0.3, f"AND候補\n{and_count:,}\n({and_pct:.1f}%)",
            ha="center", fontsize=9, color="darkred",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    ax.text(20, ymax * 0.3, f"OR候補\n{or_count:,}\n({or_pct:.1f}%)",
            ha="center", fontsize=9, color="darkblue",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    # --- (0,1): joint heatmap (右テール領域を赤線で強調) ---
    ax = axes[0, 1]
    masked = np.ma.masked_where(joint_final == 0, joint_final)
    im = ax.imshow(
        masked,
        origin="lower",
        aspect="auto",
        norm=mcolors.LogNorm(vmin=1, vmax=max(joint_final.max(), 1)),
        cmap="viridis",
    )
    plt.colorbar(im, ax=ax, label="エントリ数 (対数)")
    ax.axhline(high_pn_threshold - 0.5, color="red", linewidth=2, linestyle="--",
               label=f"pn bucket ≥ {high_pn_threshold}")
    ax.set_xticks(range(0, 32, 2))
    ax.set_xticklabels([labels[k] for k in range(0, 32, 2)], rotation=60, fontsize=7)
    ax.set_yticks(range(0, 32, 2))
    ax.set_yticklabels([labels[k] for k in range(0, 32, 2)], fontsize=7)
    ax.set_xlabel("dn バケット")
    ax.set_ylabel("pn バケット")
    ax.set_title("pn vs dn 2D 分布 [最終] (赤線以上が右テール)")
    ax.legend(fontsize=8)

    # --- (1,0): per-depth の高 pn エントリ数 ---
    ax = axes[1, 0]
    if outer:
        depths = [d for d, _, _ in outer]
        counts = [int(dn_d.sum()) for _, _, dn_d in outer]
        ax.bar(range(len(depths)), counts, color="purple", alpha=0.8)
        ax.set_xticks(range(len(depths)))
        ax.set_xticklabels([str(d) for d in depths])
        ax.set_xlabel("IDS depth")
        ax.set_ylabel(f"pn bucket ≥ {high_pn_threshold} のエントリ数")
        ax.set_title("各 depth での右テールエントリ数")
        for i, c in enumerate(counts):
            ax.text(i, c * 1.01, f"{c:,}", ha="center", va="bottom", fontsize=7)

    # --- (1,1): per-depth の AND/OR 構成割合 ---
    ax = axes[1, 1]
    if outer:
        depths = [d for d, _, _ in outer]
        and_fracs = [100.0 * dn_d[:7].sum() / max(dn_d.sum(), 1) for _, _, dn_d in outer]
        or_fracs = [100.0 * dn_d[12:].sum() / max(dn_d.sum(), 1) for _, _, dn_d in outer]
        mid_fracs = [100.0 - a - o for a, o in zip(and_fracs, or_fracs)]

        x = range(len(depths))
        ax.bar(x, and_fracs, label="AND候補 (dn≤2^6=64)", color="#d73027", alpha=0.85)
        ax.bar(x, mid_fracs, bottom=and_fracs, label="中間 (2^6<dn<2^12)", color="#74add1", alpha=0.85)
        ax.bar(
            x, or_fracs,
            bottom=[a + m for a, m in zip(and_fracs, mid_fracs)],
            label="OR候補 (dn≥2^12=4096)", color="#4575b4", alpha=0.85,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([str(d) for d in depths])
        ax.set_xlabel("IDS depth")
        ax.set_ylabel("割合 (%)")
        ax.set_title("各 depth での右テールエントリの AND/OR 構成割合")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"pn 右テール分析グラフを保存: {out_path}")
    plt.close(fig)


def _lognormal_stats(arr: np.ndarray) -> tuple[float, float, float, float] | None:
    """bucket 1-30 の中間エントリから対数正規分布パラメータを計算する．

    Returns (mu_ln, sigma_ln, kl_div, median_bucket) or None.
    """
    total = float(arr.sum())
    if total < 2:
        return None
    result = _fit_lognormal_to_histogram(arr)
    if result is None:
        return None
    mu_ln, sigma_ln, kl, _ = result
    return mu_ln, sigma_ln, kl, float(np.exp(mu_ln))


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

    # 中間エントリ (bucket 1-30) の統計
    pn_interm = pn_arr[1:31]
    dn_interm = dn_arr[1:31]
    pn_interm_total = int(pn_interm.sum())
    dn_interm_total = int(dn_interm.sum())

    print(f"\n=== 中間エントリ統計 (bucket 1-30, INF/0 除外) ===")
    print(f"  pn 中間エントリ数: {pn_interm_total:,} ({100.0 * pn_interm_total / max(total_entries, 1):.1f}%)")
    print(f"  dn 中間エントリ数: {dn_interm_total:,} ({100.0 * dn_interm_total / max(total_entries, 1):.1f}%)")

    for name, arr, total in [("pn", pn_interm, pn_interm_total), ("dn", dn_interm, dn_interm_total)]:
        if total > 0:
            result = _lognormal_stats(arr)
            if result is not None:
                mu_ln, sigma_ln, kl, median_bucket = result
                xs_float = np.array(range(1, 31), dtype=float)
                ys = arr.astype(float)
                # 参考: バケット空間での算術平均・分散
                mean_bkt = float(np.dot(xs_float, ys) / total)
                var_bkt = float(np.dot((xs_float - mean_bkt) ** 2, ys) / total)
                std_bkt = np.sqrt(max(var_bkt, 0.1))
                # 歪度
                skew_w = float(np.dot((xs_float - mean_bkt) ** 3, ys) / total) / (std_bkt ** 3)
                print(f"  {name}: KL(対数正規)={kl:.3f}, μ_ln={mu_ln:.3f}, σ_ln={sigma_ln:.3f}, "
                      f"中央値={median_bucket:.1f}bucket")
                print(f"       (参考: 算術 μ={mean_bkt:.1f}bucket, σ={std_bkt:.1f}bucket, 歪度={skew_w:.3f})")
            # バケット 1-30 を表示 (非ゼロのみ)
            print(f"  {name} 非ゼロバケット:")
            for k in range(1, 31):
                v = arr[k - 1]
                if v > 0:
                    pct = 100.0 * v / total
                    print(f"    [{k:2d}] {bucket_label_short(k):>8s}: {v:>10,} ({pct:5.1f}%)")

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
        print(f"  {'depth':>6}  {'nodes':>12}  {'total':>12}  {'pn=INF%':>8}  {'dn=0%':>7}"
              f"  {'pn中央値':>9}  {'pn_σ_ln':>8}  {'pn_KL':>7}"
              f"  {'dn中央値':>9}  {'dn_σ_ln':>8}  {'dn_KL':>7}")
        for ids_depth, nodes, pn_h, dn_h, _ in per_depth:
            total = sum(pn_h)
            inf_pn = 100.0 * pn_h[31] / max(total, 1)
            zero_dn = 100.0 * dn_h[0] / max(total, 1)

            # pn 中間エントリの対数正規統計
            pn_interm_arr = np.array(pn_h[1:31], dtype=np.int64)
            pn_res = _lognormal_stats(pn_interm_arr)
            if pn_res is not None:
                _, pn_sig, pn_kl, pn_med = pn_res
                pn_med_str = f"{pn_med:.1f}"
                pn_sig_str = f"{pn_sig:.3f}"
                pn_kl_str  = f"{pn_kl:.3f}"
            else:
                pn_med_str = pn_sig_str = pn_kl_str = "N/A"

            # dn 中間エントリの対数正規統計
            dn_interm_arr = np.array(dn_h[1:31], dtype=np.int64)
            dn_res = _lognormal_stats(dn_interm_arr)
            if dn_res is not None:
                _, dn_sig, dn_kl, dn_med = dn_res
                dn_med_str = f"{dn_med:.1f}"
                dn_sig_str = f"{dn_sig:.3f}"
                dn_kl_str  = f"{dn_kl:.3f}"
            else:
                dn_med_str = dn_sig_str = dn_kl_str = "N/A"

            print(f"  {ids_depth:>6}  {nodes:>12,}  {total:>12,}  {inf_pn:>7.1f}%  {zero_dn:>6.1f}%"
                  f"  {pn_med_str:>9}  {pn_sig_str:>8}  {pn_kl_str:>7}"
                  f"  {dn_med_str:>9}  {dn_sig_str:>8}  {dn_kl_str:>7}")


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
        pv_nodes_per_child=0,
        tt_gc_threshold=0,
    )
    elapsed = time.time() - t0

    print(f"結果: status={result.status}, nodes={result.nodes_searched:,}, elapsed={elapsed:.1f}s")
    print(f"WorkingTT 非空エントリ数: {sum(pn_hist):,}")
    print(f"IDS depth スナップショット数: {len(per_depth)}")

    plot_final_dist(result, pn_hist, dn_hist, joint_hist_flat, elapsed,
                    "/tmp/pn_dn_dist_39te.png")
    plot_per_depth(per_depth, "/tmp/pn_dn_dist_39te_per_depth.png")
    plot_per_depth_total_trend(per_depth, "/tmp/pn_dn_dist_39te_trend.png")
    # 最後の IDS スナップショットを使用
    last_snap = per_depth[-1] if per_depth else None
    if last_snap is not None:
        _, _, last_pn_h, last_dn_h, _ = last_snap
        snap_label = f"IDS depth={last_snap[0]}"
    else:
        last_pn_h, last_dn_h = pn_hist, dn_hist
        snap_label = "最終"
    plot_intermediate_dist(last_pn_h, last_dn_h, per_depth,
                           "/tmp/pn_dn_dist_39te_intermediate.png",
                           snapshot_label=snap_label)
    plot_intermediate_per_depth(per_depth, "/tmp/pn_dn_dist_39te_intermediate_depth.png")
    analyze_kl_by_bucket(per_depth, "/tmp/pn_dn_dist_39te_kl_by_bucket.png")
    analyze_pn_tail(joint_hist_flat, per_depth, "/tmp/pn_dn_dist_39te_pn_tail.png")
    print_text_summary(pn_hist, dn_hist, per_depth)


if __name__ == "__main__":
    main()
