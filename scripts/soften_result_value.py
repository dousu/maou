#!/usr/bin/env python
"""対照実験 (Arm 0) 用に，勝敗ラベルを機械的に鈍らせた教師を作る．

## 何のためのスクリプトか

探索値を投与すると，狙った帯の**過信 gap** が有意に下がることが測れた
(ply>=120 で Δ = −0.0177 [−0.0254, −0.0106])．しかし探索値がしていたのは
**軟化だけ**である — 実測で訂正率 0.1%，証明済み詰み 1,692 件は実際の勝敗と
100% 一致していた．

そうすると次の問いが残る:

    効いているのは「軟化」そのものか，「探索値の中身」か．

**もし軟化だけで同じ効果が出るなら，探索値の生成に GPU を投じる理由が無い．**
本スクリプトは，位置情報を一切持たない機械的な軟化を作り，探索値と同じ
経路で投与するための対照を用意する．

**これは採用候補ではない．**ラベルを鈍らせるのは compass § VETOES の
「教師信号は根本解決のみ / 温度スケーリングは却下」に触れる．出荷するため
ではなく，**探索値の中身が必要かどうかを 1 本の再学習で判定するため**にだけ
使う．

## なぜ前処理出力を書き換えないのか

前処理出力は 1496 要素のリスト列を 2 本持ち，1 ファイル約 99 万行で全読み
すると 12GB になる．代わりに ``(id, searchWinRate)`` の 2 列だけを書き出し，
``pre-process --search-value-path`` に渡す．置換は既存の実装済み経路が行う
ので，**Arm 0 と Arm 1 が完全に同じ経路を通り，違いが値だけになる**．

## 対象行の選び方

``maou utility search-values`` に合わせる:

- **手数**: その局面が **1 度でも**指定帯に現れれば対象 (search-values は
  生の HCPE 行を手数で絞るので，同じ局面が帯の内外に現れる場合は帯の中の
  行が拾われる)．
- **出現回数**: 絞らない (search-values も ``--position-count-threshold`` と
  独立に，出現回数によらず適用する)．
- **値**: 集約後の ``resultValue`` が**厳密に 0.0 か 1.0** の行のみ．
  引き分け (0.5) と，複数対局で既に平均されている行は既に統計なので触らない．

## 使い方

    # 探索値と同じ条件分布から標本化する (推奨)
    scripts/soften_result_value.py HCPE_DIR out.feather \\
        --min-ply 60 --max-ply 100 --mode empirical \\
        --reference scratchpad/measure/sv_vs_outcome.feather

    # 一様に鈍らせる
    scripts/soften_result_value.py HCPE_DIR out.feather \\
        --min-ply 60 --max-ply 100 --mode uniform --epsilon 0.15

    # そのまま pre-process へ
    maou pre-process --input-path HCPE_DIR --output-dir OUT \\
        --search-value-path out.feather
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import polars as pl

CONFIDENT = (0.0, 1.0)
"""軟化の対象にする集約後 ``resultValue``．引き分け 0.5 と平均済みは除く．"""


def load_positions(
    hcpe_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HCPE から ``(zobrist id, 手番視点の勝敗, 手数)`` を取り出す．

    メモリを抑えるためファイル単位で処理し，``hcp`` は保持しない．

    Args:
        hcpe_dir: ``.feather`` を含むディレクトリ (再帰的に走査する)．

    Returns:
        ``(id (N,) uint64, result (N,) float64, ply (N,) int64)``．
    """
    from maou._rust.maou_search import preprocess_hcpes

    paths = sorted(
        glob.glob(
            str(hcpe_dir / "**" / "*.feather"), recursive=True
        )
    )
    if not paths:
        raise SystemExit(f"no .feather under {hcpe_dir}")
    ids: list[np.ndarray] = []
    results: list[np.ndarray] = []
    plies: list[np.ndarray] = []
    for path in paths:
        df = pl.read_ipc(path, memory_map=False)
        hcp = np.ascontiguousarray(
            np.stack(
                [
                    np.frombuffer(b, np.uint8)
                    for b in df["hcp"].to_numpy()
                ]
            )
        )
        zid, _, res = preprocess_hcpes(
            hcp,
            np.ascontiguousarray(
                df["bestMove16"].to_numpy().astype(np.int16)
            ),
            np.ascontiguousarray(
                df["gameResult"].to_numpy().astype(np.int8)
            ),
        )
        ids.append(np.asarray(zid).astype(np.uint64))
        results.append(np.asarray(res).astype(np.float64))
        plies.append(
            np.array(
                [
                    int(s.rsplit(".hcpe_", 1)[1])
                    for s in df["id"].to_list()
                ],
                dtype=np.int64,
            )
        )
    return (
        np.concatenate(ids),
        np.concatenate(results),
        np.concatenate(plies),
    )


def aggregate_positions(
    ids: np.ndarray, results: np.ndarray, plies: np.ndarray
) -> pl.DataFrame:
    """局面 (zobrist id) 単位に集約する．``pre-process`` と同じ値になる．

    ``resultValue`` は同一局面を含む全対局の勝敗の平均 — これは
    ``maou pre-process`` の出力と厳密に一致することを実測で確認してある
    (in-period 41,542 行で最大誤差 2.9e-08 = float32 の精度)．

    Args:
        ids: 局面ごとの zobrist id．
        results: 局面ごとの手番視点の勝敗 (0 / 0.5 / 1)．
        plies: 局面ごとの手数．

    Returns:
        ``id`` / ``resultValue`` / ``count`` / ``min_ply`` / ``max_ply``．
    """
    return (
        pl.DataFrame({"id": ids, "res": results, "ply": plies})
        .group_by("id")
        .agg(
            pl.col("res").mean().alias("resultValue"),
            pl.len().alias("count"),
            pl.col("ply").min().alias("min_ply"),
            pl.col("ply").max().alias("max_ply"),
        )
    )


def select_targets(
    agg: pl.DataFrame, min_ply: int, max_ply: int
) -> pl.DataFrame:
    """軟化の対象になる行を選ぶ．

    帯は ``[min_ply, max_ply)``．**1 度でも帯に現れた局面**を対象にする
    (``search-values`` が生の HCPE 行を手数で絞るのに合わせる)．

    Args:
        agg: :func:`aggregate_positions` の出力．
        min_ply: 帯の下限 (含む)．
        max_ply: 帯の上限 (含まない)．

    Returns:
        ``agg`` の部分集合．
    """
    if max_ply <= min_ply:
        raise ValueError(
            f"--max-ply ({max_ply}) は --min-ply ({min_ply}) より大きいこと"
        )
    return agg.filter(
        (pl.col("max_ply") >= min_ply)
        & (pl.col("min_ply") < max_ply)
        & pl.col("resultValue").is_in(list(CONFIDENT))
    )


def soften_uniform(
    values: np.ndarray, epsilon: float
) -> np.ndarray:
    """``0 -> epsilon`` / ``1 -> 1 - epsilon`` に鈍らせる．

    Args:
        values: 厳密に 0.0 か 1.0 の配列．
        epsilon: 0 側へ寄せる量 (``0 < epsilon < 0.5``)．
    """
    if not 0.0 < epsilon < 0.5:
        raise ValueError(
            f"--epsilon は 0 と 0.5 の間 (got {epsilon})"
        )
    return np.where(
        values >= 0.5, 1.0 - epsilon, epsilon
    ).astype(np.float32)


def empirical_pools(
    reference: pl.DataFrame, min_ply: int, max_ply: int
) -> tuple[np.ndarray, np.ndarray]:
    """参照ファイルから，元ラベル別の探索値の経験分布を取り出す．

    参照は ``searchWinRate`` と (置換前の) ``resultValue`` を持つこと．
    ``ply`` があれば帯で絞る．

    Returns:
        ``(元が 0 だった行の探索値, 元が 1 だった行の探索値)``．
    """
    for col in ("searchWinRate", "resultValue"):
        if col not in reference.columns:
            raise ValueError(f"参照に {col} 列が無い")
    ref = reference
    if "ply" in ref.columns:
        ref = ref.filter(
            (pl.col("ply") >= min_ply)
            & (pl.col("ply") < max_ply)
        )
    lo = ref.filter(pl.col("resultValue") == 0.0)[
        "searchWinRate"
    ].to_numpy()
    hi = ref.filter(pl.col("resultValue") == 1.0)[
        "searchWinRate"
    ].to_numpy()
    if len(lo) < 100 or len(hi) < 100:
        raise ValueError(
            f"参照の標本が少なすぎる (0 側 {len(lo)} / 1 側 {len(hi)})．"
            "帯を広げるか別の参照を使う"
        )
    return lo.astype(np.float32), hi.astype(np.float32)


def soften_empirical(
    values: np.ndarray,
    pool_lo: np.ndarray,
    pool_hi: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """元ラベル別の経験分布から標本化して鈍らせる．

    探索値と**同じ周辺分布**を持ちながら，局面ごとの情報を一切持たない
    教師になる — これが対照として要る性質．

    参照の偏りもそのまま再現する点に注意 (実測で探索値は実際の勝敗より
    +0.029 楽観だった)．偏りを分離したいときは :func:`debias` を通す．
    """
    out = np.empty(len(values), dtype=np.float32)
    is_hi = values >= 0.5
    n_hi = int(is_hi.sum())
    n_lo = len(values) - n_hi
    if n_lo:
        out[~is_hi] = pool_lo[
            rng.integers(0, len(pool_lo), n_lo)
        ]
    if n_hi:
        out[is_hi] = pool_hi[
            rng.integers(0, len(pool_hi), n_hi)
        ]
    return out


def debias(
    softened: np.ndarray, original: np.ndarray
) -> np.ndarray:
    """軟化後の平均を元ラベルの平均へ揃える．

    ``empirical`` は参照の偏りごと再現するので，そのままだと「軟化の効果」と
    「偏りの効果」が同居する．平行移動して 1 次モーメントだけ合わせることで，
    偏りを持たない対照を作れる．``[0, 1]`` へ丸めるため完全には一致しないので，
    残差は呼び出し側で報告すること．

    Args:
        softened: 軟化後の値．
        original: 対応する元ラベル (厳密に 0.0 か 1.0)．
    """
    shift = float(original.mean() - softened.mean())
    return np.clip(softened + shift, 0.0, 1.0).astype(
        np.float32
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "hcpe_dir",
        type=Path,
        help="HCPE (.feather) のディレクトリ",
    )
    ap.add_argument(
        "output",
        type=Path,
        help="出力 .feather (id / searchWinRate)",
    )
    ap.add_argument("--min-ply", type=int, default=60)
    ap.add_argument("--max-ply", type=int, default=100)
    ap.add_argument(
        "--mode",
        choices=("uniform", "empirical"),
        default="empirical",
    )
    ap.add_argument(
        "--epsilon",
        type=float,
        default=0.15,
        help="--mode uniform のときの軟化量",
    )
    ap.add_argument(
        "--reference",
        type=Path,
        help="--mode empirical のときの参照 "
        "(searchWinRate と置換前 resultValue を持つ feather)",
    )
    ap.add_argument(
        "--debias",
        action="store_true",
        help="--mode empirical の偏りを打ち消し，軟化後の平均を元ラベルの"
        "平均へ揃える．「軟化の効果」と「偏りの効果」を分離したいとき",
    )
    ap.add_argument(
        "--max-positions",
        type=int,
        default=0,
        help="出力する局面数の上限 (0 で無制限)．投与量の制御に使う",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.debias and args.mode != "empirical":
        raise SystemExit(
            "--debias は --mode empirical のときだけ使える"
        )

    ids, results, plies = load_positions(args.hcpe_dir)
    print(f"HCPE {len(ids):,} 行を読んだ", file=sys.stderr)

    agg = aggregate_positions(ids, results, plies)
    targets = select_targets(agg, args.min_ply, args.max_ply)
    print(
        f"局面 {len(agg):,} 件 -> ply [{args.min_ply},{args.max_ply}) かつ "
        f"確信ラベルの対象 {len(targets):,} 件",
        file=sys.stderr,
    )
    if not len(targets):
        raise SystemExit("対象が 0 件．帯を確認すること")

    rng = np.random.default_rng(args.seed)
    if args.max_positions and len(targets) > args.max_positions:
        keep = np.sort(
            rng.choice(
                len(targets), args.max_positions, replace=False
            )
        )
        targets = targets[keep]
        print(
            f"--max-positions で {len(targets):,} 件へ絞った",
            file=sys.stderr,
        )

    original = targets["resultValue"].to_numpy()
    if args.mode == "uniform":
        softened = soften_uniform(original, args.epsilon)
    else:
        if args.reference is None:
            raise SystemExit(
                "--mode empirical には --reference が要る"
            )
        pool_lo, pool_hi = empirical_pools(
            pl.read_ipc(args.reference, memory_map=False),
            args.min_ply,
            args.max_ply,
        )
        print(
            f"参照の標本: 0 側 {len(pool_lo):,} / 1 側 {len(pool_hi):,}",
            file=sys.stderr,
        )
        softened = soften_empirical(
            original, pool_lo, pool_hi, rng
        )
        if args.debias:
            before = float(softened.mean())
            softened = debias(softened, original)
            print(
                f"--debias: mean {before:.4f} -> {softened.mean():.4f} "
                f"(目標 {original.mean():.4f}，残差 "
                f"{softened.mean() - original.mean():+.4f})",
                file=sys.stderr,
            )

    out = pl.DataFrame(
        {
            "id": targets["id"].cast(pl.UInt64),
            "searchWinRate": pl.Series(
                softened, dtype=pl.Float32
            ),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.write_ipc(args.output, compression="lz4")

    conf = float(((softened < 0.1) | (softened > 0.9)).mean())
    print(
        f"\n書き出し {len(out):,} 行 -> {args.output}\n"
        f"  元 mean   = {original.mean():.4f}\n"
        f"  軟化後 mean = {softened.mean():.4f} "
        f"(差 {softened.mean() - original.mean():+.4f})\n"
        f"  軟化後もほぼ確信 (<0.1 or >0.9) = {100 * conf:.1f}%"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
