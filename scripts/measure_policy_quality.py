#!/usr/bin/env python
"""policy head の質を held-out 棋譜で測る．

## なぜ単一局面ではなくこれを見るのか

policy の欠陥はこれまで 1 局面の探索結果 (「探索が 80% 訪問した手の prior が
5.9%」) でしか示されていなかった．単一局面は**探針であって指標ではない**ので，
モデル間の優劣を判定する根拠にはならない．本スクリプトは同じ性質を
**棋譜コーパス全体の統計**として測る．

## 何を測るか

各局面について，モデルの policy を**合法手だけに制限して softmax** し
(推論経路 ``onnx.rs`` と同じ扱い)，実際に指された手 (``bestMove16``) を
正解ラベルとして次を集計する．

| 指標 | 意味 |
|---|---|
| ``top1`` / ``top5`` | 指された手を 1 位 / 上位 5 位に入れた割合 |
| ``played_mass`` | 指された手に置いた確率の平均 |
| ``ce`` | 指された手の負の対数尤度 (交差エントロピー) |
| ``entropy`` | 合法手分布のエントロピー (nat)．分布の尖り具合 |
| ``top1_mass`` | 1 位の手に置いた確率の平均．過信の指標 |
| ``legal_moves`` | 平均合法手数 (エントロピーの上限は ``ln(legal_moves)``) |

``entropy`` と ``top1_mass`` はラベルを使わないので，**教師の質が変わった
ときに分布の形がどう動いたか**を勝敗ラベルと独立に見られる．

## 何と比べるか

較正 (``measure_calibration.py``) と同じく，**学習期間内**と**held-out** の
両方で測って比を見る．policy にも汎化ギャップがあるのか，
それとも value 側だけの問題なのかを切り分けるのが目的である．

「指された手」は floodgate のエンジン集団が選んだ手であって最善手ではない．
したがって top1 は「その集団の再現度」であり，棋力そのものではない．
**2 つのモデルを比べるときに同じコーパスを使うこと**が成立条件になる．

使い方::

    python scripts/measure_policy_quality.py model.onnx holdout_hcpe
    python scripts/measure_policy_quality.py model.onnx train_hcpe

背景と目標値は docs/design/training-quality/index.md を参照．
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import polars as pl


def load_positions(
    hcpe_dir: Path, sample: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HCPE から (hcp, 指された手の move label, 対局 ID) を取り出す．

    Args:
        hcpe_dir: ``.feather`` を含むディレクトリ．
        sample: 評価する最大局面数 (0 で全件)．
        seed: 標本抽出の乱数種．

    Returns:
        ``(hcp (N,32) uint8, move_labels (N,) int64, games (N,) str)``．
    """
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
    _, labels, _ = preprocess_hcpes(
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
    return hcp, np.asarray(labels).astype(np.int64), games


def predict_policy(
    model_path: Path, hcp: np.ndarray, batch: int
) -> np.ndarray:
    """policy head の logit を返す．

    Args:
        model_path: ONNX モデルのパス．
        hcp: HCP 配列 (N, 32)．
        batch: 推論バッチサイズ．

    Returns:
        ``(N, 1496)`` の logit 配列．
    """
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
                ["policy"],
                {
                    n0: boards[i : i + batch].astype(np.int32),
                    n1: hands[i : i + batch].astype(np.float32),
                },
            )[0]
        )
    return np.concatenate(out).astype(np.float64)


def legal_softmax(
    logits: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """合法手だけに制限した softmax を返す．

    推論経路 (``onnx.rs``) は合法手のみを gather してから softmax するので，
    測定でも同じ扱いに揃える．非合法手の確率は 0 になる．

    Args:
        logits: ``(N, 1496)`` の policy logit．
        mask: ``(N, 1496)`` の合法手マスク (1 = 合法)．

    Returns:
        ``(N, 1496)`` の確率分布．
    """
    z = np.where(mask > 0, logits, -np.inf)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def bootstrap_by_game(
    values: np.ndarray,
    gidx: np.ndarray,
    n_games: int,
    rounds: int,
    seed: int,
) -> tuple[float, float]:
    """対局単位で再標本した平均の 95% 区間を返す．

    同一対局の局面は独立ではないため，局面ではなく対局ごと丸ごと
    再標本する (``measure_calibration.py`` と同じ扱い)．

    Args:
        values: 局面ごとの値 (top1 なら 0/1)．
        gidx: 局面ごとの対局インデックス．
        n_games: 対局数．
        rounds: bootstrap 回数．
        seed: 乱数種．

    Returns:
        ``(下側 2.5%, 上側 97.5%)``．
    """
    rng = np.random.default_rng(seed)
    by_game = [np.where(gidx == i)[0] for i in range(n_games)]
    boot = np.empty(rounds)
    for b in range(rounds):
        pick = np.concatenate(
            [
                by_game[i]
                for i in rng.integers(0, n_games, n_games)
            ]
        )
        boot[b] = values[pick].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> int:
    """CLI エントリポイント．"""
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

    from maou._rust.maou_search import legal_move_masks

    hcp, labels, games = load_positions(
        args.hcpe_dir, args.sample, args.seed
    )
    logits = predict_policy(args.model, hcp, args.batch)
    mask = np.asarray(legal_move_masks(hcp))
    probs = legal_softmax(logits, mask)

    n = len(labels)
    rows = np.arange(n)
    played = probs[rows, labels]
    # 全体を argsort すると (N, 1496) の索引配列が要るので上位のみ取る
    best = probs.argmax(axis=1)
    top5_idx = np.argpartition(-probs, 5, axis=1)[:, :5]
    top1 = (best == labels).astype(np.float64)
    top5 = (
        (top5_idx == labels[:, None])
        .any(axis=1)
        .astype(np.float64)
    )
    top1_mass = probs[rows, best]
    legal_count = mask.sum(axis=1).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.sum(
            np.where(probs > 0, probs * np.log(probs), 0.0),
            axis=1,
        )
        ce = -np.log(np.clip(played, 1e-12, None))

    _, gidx = np.unique(games, return_inverse=True)
    n_games = int(gidx.max()) + 1
    print(
        f"positions {n:,} / games {n_games:,} / "
        f"mean legal moves {legal_count.mean():.1f}\n"
    )

    ci = ""
    if args.bootstrap:
        lo, hi = bootstrap_by_game(
            top1, gidx, n_games, args.bootstrap, args.seed
        )
        ci = f"   95% CI [{lo:.4f}, {hi:.4f}] (game-clustered)"

    print(f"top1          = {top1.mean():.4f}{ci}")
    print(f"top5          = {top5.mean():.4f}")
    print(f"played_mass   = {played.mean():.4f}")
    print(f"ce            = {ce.mean():.4f}")
    print(
        f"entropy       = {entropy.mean():.4f} nat "
        f"(uniform over legal = {np.log(legal_count).mean():.4f})"
    )
    print(
        f"top1_mass     = {top1_mass.mean():.4f} "
        "(higher = more peaked prior)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
