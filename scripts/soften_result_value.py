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

    # 探索値と同じ条件分布から標本化する (推奨)．参照は埋め込み済みなので
    # このファイル 1 枚と maou wheel だけで動く
    scripts/soften_result_value.py HCPE_DIR out.feather \\
        --min-ply 60 --max-ply 100 --mode empirical

    # 偏りを打ち消した対照 (軟化の効果と偏りの効果を分離する)
    scripts/soften_result_value.py HCPE_DIR out_debias.feather \\
        --min-ply 60 --max-ply 100 --mode empirical --debias

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


# --- 埋め込み参照 -----------------------------------------------------------
# ``--reference`` を渡さなくても ``--mode empirical`` が使えるようにするため，
# 測定済みの条件分布を分位点表として持つ．Colab などへ**このファイル 1 枚だけ**
# 持っていける形にするのが目的 (参照 feather は scratchpad 配下で git 管理外)．
#
# 出所: floodgate 2025-03-02 の 394 局 (学習期間内) の ply 60-99 について，
#   1. maou utility search-values --min-ply 40 --model-path <teacher>
#   2. maou pre-process (--search-value-path なし) で置換前の resultValue
#   3. zobrist id で join
# 元ラベル別に 11,939 行 (0 側 5,953 / 1 側 5,986)．101 分位点で平均は 4 桁目
# まで，「ほぼ確信 (<0.1 or >0.9)」の割合は 28.6% vs 28.8% で再現する．
#
# **帯を変えるなら再測定すること．**分布は手数に強く依存し，ply>=120 では
# ほぼ確信が 58.6% まで上がる (ply 60-99 は 28.6%)．

EMBEDDED_BAND = (60, 100)
"""埋め込み分位点表を測った手数帯 ``[min_ply, max_ply)``．"""

EMBEDDED_QUANTILES_LO = np.array(
    [
        0.0000,
        0.0000,
        0.0000,
        0.0008,
        0.0029,
        0.0065,
        0.0099,
        0.0135,
        0.0176,
        0.0218,
        0.0265,
        0.0316,
        0.0354,
        0.0392,
        0.0434,
        0.0479,
        0.0532,
        0.0577,
        0.0625,
        0.0685,
        0.0746,
        0.0814,
        0.0863,
        0.0928,
        0.0985,
        0.1044,
        0.1090,
        0.1150,
        0.1202,
        0.1263,
        0.1314,
        0.1369,
        0.1426,
        0.1484,
        0.1550,
        0.1613,
        0.1660,
        0.1716,
        0.1784,
        0.1843,
        0.1902,
        0.1960,
        0.2025,
        0.2070,
        0.2128,
        0.2196,
        0.2250,
        0.2310,
        0.2385,
        0.2444,
        0.2524,
        0.2603,
        0.2663,
        0.2729,
        0.2783,
        0.2838,
        0.2899,
        0.2959,
        0.3023,
        0.3092,
        0.3170,
        0.3235,
        0.3302,
        0.3369,
        0.3471,
        0.3556,
        0.3638,
        0.3726,
        0.3847,
        0.3943,
        0.4033,
        0.4102,
        0.4201,
        0.4280,
        0.4380,
        0.4486,
        0.4589,
        0.4709,
        0.4812,
        0.4916,
        0.5031,
        0.5162,
        0.5320,
        0.5441,
        0.5562,
        0.5731,
        0.5872,
        0.5999,
        0.6147,
        0.6308,
        0.6493,
        0.6656,
        0.6820,
        0.6984,
        0.7117,
        0.7340,
        0.7520,
        0.7686,
        0.7997,
        0.8371,
        0.9474,
    ],
    dtype=np.float32,
)
"""元ラベルが 0 だった局面の探索値の 101 分位点 (0, 0.01, ..., 1)．"""

EMBEDDED_QUANTILES_HI = np.array(
    [
        0.0557,
        0.2325,
        0.2728,
        0.3096,
        0.3448,
        0.3730,
        0.4028,
        0.4289,
        0.4497,
        0.4645,
        0.4884,
        0.5063,
        0.5210,
        0.5361,
        0.5529,
        0.5710,
        0.5839,
        0.5970,
        0.6114,
        0.6222,
        0.6325,
        0.6430,
        0.6524,
        0.6600,
        0.6685,
        0.6777,
        0.6841,
        0.6912,
        0.6974,
        0.7056,
        0.7129,
        0.7197,
        0.7249,
        0.7304,
        0.7356,
        0.7429,
        0.7475,
        0.7540,
        0.7584,
        0.7640,
        0.7709,
        0.7764,
        0.7828,
        0.7883,
        0.7932,
        0.7978,
        0.8029,
        0.8074,
        0.8119,
        0.8174,
        0.8235,
        0.8278,
        0.8329,
        0.8375,
        0.8423,
        0.8476,
        0.8527,
        0.8573,
        0.8605,
        0.8654,
        0.8692,
        0.8733,
        0.8780,
        0.8823,
        0.8868,
        0.8911,
        0.8950,
        0.8987,
        0.9036,
        0.9076,
        0.9115,
        0.9158,
        0.9204,
        0.9241,
        0.9288,
        0.9331,
        0.9369,
        0.9406,
        0.9440,
        0.9480,
        0.9521,
        0.9556,
        0.9586,
        0.9615,
        0.9652,
        0.9686,
        0.9717,
        0.9745,
        0.9774,
        0.9806,
        0.9833,
        0.9864,
        0.9888,
        0.9911,
        0.9936,
        0.9955,
        0.9979,
        0.9995,
        1.0000,
        1.0000,
        1.0000,
    ],
    dtype=np.float32,
)
"""元ラベルが 1 だった局面の探索値の 101 分位点．"""


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


def embedded_pools(
    n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """埋め込み分位点表から逆関数法で標本を作る．

    ``--reference`` が無いときの既定．外部ファイルを持ち運ばずに
    ``--mode empirical`` を使えるようにするためにある．
    """
    q = np.linspace(0.0, 1.0, len(EMBEDDED_QUANTILES_LO))
    u = rng.random(n)
    return (
        np.interp(u, q, EMBEDDED_QUANTILES_LO).astype(
            np.float32
        ),
        np.interp(u, q, EMBEDDED_QUANTILES_HI).astype(
            np.float32
        ),
    )


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
        help="--mode empirical の参照 (searchWinRate と置換前 resultValue を"
        "持つ feather)．省略すると埋め込み分位点表を使うので，"
        "このファイル 1 枚だけで動く",
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
            if (args.min_ply, args.max_ply) != EMBEDDED_BAND:
                print(
                    f"警告: 埋め込み分位点表は ply {EMBEDDED_BAND[0]}-"
                    f"{EMBEDDED_BAND[1] - 1} で測ったもので，指定された "
                    f"ply {args.min_ply}-{args.max_ply - 1} とは違う．"
                    "分布は手数に強く依存するので --reference で"
                    "測り直した参照を渡すこと",
                    file=sys.stderr,
                )
            pool_lo, pool_hi = embedded_pools(
                len(original), rng
            )
            print(
                "参照: 埋め込み分位点表 "
                f"(ply {EMBEDDED_BAND[0]}-{EMBEDDED_BAND[1] - 1})",
                file=sys.stderr,
            )
        else:
            pool_lo, pool_hi = empirical_pools(
                pl.read_ipc(args.reference, memory_map=False),
                args.min_ply,
                args.max_ply,
            )
            print(
                f"参照: {args.reference} "
                f"(0 側 {len(pool_lo):,} / 1 側 {len(pool_hi):,})",
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
