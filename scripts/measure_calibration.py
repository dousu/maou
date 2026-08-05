#!/usr/bin/env python
"""value head の較正を held-out 棋譜で測る．

## 何を何と比べているか

**ビンの境界 (``BINS``) はモデルの「予測勝率」を区切るものであり，
対局結果を区切るものではない．**

1. 各局面にモデルの予測勝率 ``p`` を出す
2. **``p`` の値でビンに振り分ける** (例: ``p`` が 0.05〜0.10 の局面を集める)
3. そのビンに入った局面群について，**実際の対局結果 ``y`` (0/1) を平均**する

出力表の 2 列は別々の配列から来る::

    bin           = 予測勝率 p の区間           <- 振り分けの軸
    pred   = p.mean()   区間内の予測の平均      <- モデル由来
    actual = y.mean()   同じ局面群の勝敗の平均  <- 対局結果由来

較正が取れていれば ``pred == actual`` (対角線)．過信していれば S 字
(予測が両端に張り付き，実測が中央へ寄る)．

## 個々の局面は 0/1 しか持たないのに，なぜ連続値が出るのか

同一局面を繰り返す必要はない．中終盤の局面はほぼユニークで 0/1 しか
持たないが，**「モデルが同程度の確率を出した別々の局面」を数百〜数千集めて
その 0/1 を平均する**ので連続値になる．必要なのは「同じ局面が何度も出る」
ことではなく「**同じ予測値を持つ局面が大量にある**」こと．

較正の定義 ``E[Y | p(X) = q] = q`` を，``p(X) ≈ q`` の局面で ``Y`` を
標本平均して推定していることに相当する．

## ラベルの出所

``y`` は HCPE の ``gameResult`` (対局の勝敗) のみから算出する
(手番側が勝ったか = ``preprocess.rs`` の ``result_value``)．
棋譜中のエンジン評価値 (``eval`` 列) は対局しているクライアントの仕様に
依存するため使わない．

測っているのは「**入力した棋譜を指した集団のもとでの勝率**」に対する較正で
あって，ゲーム理論的な真の勝率ではない．本スクリプト自体は棋譜の出所を問わず，
HCPE ならなんでも受け取る:

- floodgate 棋譜を渡す → floodgate のエンジン集団に対する較正
- 自己対局の棋譜を渡す → maou 自身の指し手のもとでの較正

**2 つの測定を比べるときは同じ集団から取ること** (学習期間内 vs held-out の
比較が成立するのは，どちらも floodgate という同一集団だからである)．

同一対局の局面は同じ勝敗ラベルを共有するので独立標本ではない．
信頼区間は**対局を単位とした bootstrap** で求める (実効的な標本サイズは
局面数ではなく対局数に近い)．

使い方::

    # 学習期間外の棋譜を取得して HCPE 化する
    maou utility fetch-floodgate --start-date 2026-03-02 --end-date 2026-03-03 \\
        --output-dir holdout_csa --strategy daily
    maou hcpe-convert --input-path holdout_csa --input-format csa \\
        --output-dir holdout_hcpe

    # 較正を測る
    python scripts/measure_calibration.py model.onnx holdout_hcpe

    # 学習期間内のデータでも測り，汎化ギャップを見る
    python scripts/measure_calibration.py model.onnx train_hcpe

背景と目標値は docs/design/training-quality/index.md を参照．
"""

from __future__ import annotations

import argparse
import glob
import itertools
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import polars as pl

# **モデルの予測勝率** を区切る境界 (対局結果を区切るものではない)．
# 両端は分布が集中するので細かく刻む．
BINS = np.array(
    [
        0,
        0.05,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        0.95,
        1.0,
    ]
)


def load_positions(
    hcpe_dir: Path, sample: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HCPE から (予測入力用 hcp, 手番視点の勝敗, 対局 ID) を取り出す．"""
    from maou._rust.maou_search import preprocess_hcpes

    paths = sorted(
        glob.glob(
            str(hcpe_dir / "**" / "*.feather"), recursive=True
        )
    )
    if not paths:
        raise SystemExit(f"no .feather under {hcpe_dir}")
    df = pl.concat(
        [pl.read_ipc(p, memory_map=False) for p in paths],
        how="vertical",
    )
    if sample and len(df) > sample:
        rng = np.random.default_rng(seed)
        df = df[
            np.sort(rng.choice(len(df), sample, replace=False))
        ]

    hcp = np.ascontiguousarray(
        np.stack(
            [
                np.frombuffer(b, np.uint8)
                for b in df["hcp"].to_numpy()
            ]
        )
    )
    _, _, wins = preprocess_hcpes(
        hcp,
        np.ascontiguousarray(
            df["bestMove16"].to_numpy().astype(np.int16)
        ),
        np.ascontiguousarray(
            df["gameResult"].to_numpy().astype(np.int8)
        ),
    )
    # id は "{棋譜ファイル名}.hcpe_{ply}"．接頭辞が対局 ID．
    games = np.array(
        [s.rsplit(".hcpe_", 1)[0] for s in df["id"].to_list()]
    )
    return hcp, np.asarray(wins), games


def predict(
    model_path: Path, hcp: np.ndarray, batch: int
) -> np.ndarray:
    """value head の logit を返す．"""
    from maou._rust.maou_search import encode_hcp_features

    boards, hands = encode_hcp_features(hcp)
    sess = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    n0, n1 = (i.name for i in sess.get_inputs())
    out = []
    for i in range(0, len(hcp), batch):
        out.append(
            sess.run(
                ["value"],
                {
                    n0: boards[i : i + batch].astype(np.int32),
                    n1: hands[i : i + batch].astype(np.float32),
                },
            )[0].reshape(-1)
        )
    return np.concatenate(out).astype(np.float64)


def _bin_mask(
    pred: np.ndarray, lo: float, hi: float
) -> np.ndarray:
    """**予測勝率**が ``[lo, hi)`` に入る局面を選ぶマスクを返す．

    最終ビンだけ右端を含める (``p == 1.0`` を落とさないため)．

    Args:
        pred: 局面ごとのモデル予測勝率．**対局結果ではない**．
        lo: 区間の下限 (含む)．
        hi: 区間の上限 (最終ビン以外は含まない)．
    """
    return (
        (pred >= lo) & (pred < hi)
        if hi < 1.0
        else (pred >= lo) & (pred <= hi)
    )


def expected_calibration_error(
    p: np.ndarray, y: np.ndarray
) -> float:
    """ビンごとの |平均予測 − 実測勝敗| を件数で重み付けした平均．

    Args:
        p: 局面ごとの予測勝率 (ビン分けの軸でもある)．
        y: 局面ごとの実際の勝敗 (0/1)．
    """
    ece = 0.0
    for lo, hi in itertools.pairwise(BINS):
        m = _bin_mask(p, lo, hi)
        if m.sum():
            ece += (
                m.sum()
                / len(p)
                * abs(p[m].mean() - y[m].mean())
            )
    return ece


def bootstrap_by_game(
    p: np.ndarray,
    y: np.ndarray,
    gidx: np.ndarray,
    n_games: int,
    rounds: int,
    seed: int,
) -> np.ndarray:
    """対局単位で再標本し，ビンごとの**実測勝敗平均**の分布を返す．

    同一対局の局面は同じ勝敗ラベルを共有するため独立標本ではない．
    局面ではなく**対局ごと丸ごと**再標本することで，この相関を
    区間幅に反映させる (実効標本サイズは局面数でなく対局数に近い)．
    """
    rng = np.random.default_rng(seed)
    by_game = [np.where(gidx == i)[0] for i in range(n_games)]
    boot = np.full((rounds, len(BINS) - 1), np.nan)
    for b in range(rounds):
        pick = np.concatenate(
            [
                by_game[i]
                for i in rng.integers(0, n_games, n_games)
            ]
        )
        pp, yy = p[pick], y[pick]
        for k in range(len(BINS) - 1):
            m = _bin_mask(pp, BINS[k], BINS[k + 1])
            if m.sum() >= 10:
                boot[b, k] = yy[m].mean()
    return boot


def fit_temperature(
    z: np.ndarray, y: np.ndarray
) -> tuple[float, float]:
    """NLL を最小化する温度 T と，そのときの ECE を返す．

    T は較正の**診断値**として使う (1 から離れているほど過信).
    後処理として適用する方針は採らない (docs/design/training-quality)．
    """

    def nll(t: float) -> float:
        p = np.clip(1 / (1 + np.exp(-z / t)), 1e-9, 1 - 1e-9)
        return float(
            -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        )

    grid = np.linspace(0.5, 6.0, 111)
    best = float(min(grid, key=nll))
    return best, expected_calibration_error(
        1 / (1 + np.exp(-z / best)), y
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("model", type=Path, help="ONNX model path")
    ap.add_argument(
        "hcpe_dir",
        type=Path,
        help="directory with .feather HCPE",
    )
    ap.add_argument(
        "--sample",
        type=int,
        default=20000,
        help="max positions to evaluate (0 = all). "
        "CPU inference is slow and large runs can be OOM-killed.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="game-clustered bootstrap rounds (0 to skip)",
    )
    args = ap.parse_args()

    hcp, wins, games = load_positions(
        args.hcpe_dir, args.sample, args.seed
    )
    z = predict(args.model, hcp, args.batch)
    p = 1 / (1 + np.exp(-z))

    # 引き分けは較正の定義が曖昧なので除外する
    keep = (wins == 0.0) | (wins == 1.0)
    z, p, y, g = (
        z[keep],
        p[keep],
        wins[keep].astype(np.float64),
        games[keep],
    )
    _, gidx = np.unique(g, return_inverse=True)
    n_games = int(gidx.max()) + 1
    print(
        f"positions {len(y):,} / games {n_games:,} "
        f"(dropped {int((~keep).sum()):,} draws) / "
        f"actual win rate {y.mean():.3f}\n"
    )

    boot = (
        bootstrap_by_game(
            p, y, gidx, n_games, args.bootstrap, args.seed
        )
        if args.bootstrap
        else None
    )

    print(
        "bin      = predicted win rate interval (the axis positions are "
        "grouped by)\n"
        "pred     = mean of the model's predictions inside that interval\n"
        "actual   = mean of the real game outcomes (0/1) of the SAME "
        "positions\n"
        "         -> calibrated means pred == actual\n"
    )
    print(
        f"{'predicted bin':>15}{'positions':>11}{'games':>7}"
        f"{'pred':>8}{'actual':>8}{'diff':>8}"
        + ("       95% CI" if boot is not None else "")
    )
    print("-" * (57 + (13 if boot is not None else 0)))
    for k in range(len(BINS) - 1):
        m = _bin_mask(p, BINS[k], BINS[k + 1])
        if not m.sum():
            continue
        pm, ym = p[m].mean(), y[m].mean()
        line = (
            f"{f'[{BINS[k]:.2f},{BINS[k + 1]:.2f})':>15}"
            f"{m.sum():>11,}{len(np.unique(gidx[m])):>7,}"
            f"{pm:>8.3f}{ym:>8.3f}{pm - ym:>+8.3f}"
        )
        if boot is not None:
            col = boot[:, k]
            col = col[~np.isnan(col)]
            if len(col) > 50:
                lo, hi = np.percentile(col, [2.5, 97.5])
                line += f"  [{lo:.3f}, {hi:.3f}]"
        print(line)

    ece = expected_calibration_error(p, y)
    brier = float(np.mean((p - y) ** 2))
    t, ece_t = fit_temperature(z, y)
    print("-" * (57 + (13 if boot is not None else 0)))
    print(f"ECE   = {ece:.4f}")
    print(
        f"Brier = {brier:.4f}   (0.25 = always answering 0.5)"
    )
    print(
        f"diagnostic temperature T = {t:.2f} (ECE would be {ece_t:.4f}); "
        "T far from 1 means over-confidence"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
