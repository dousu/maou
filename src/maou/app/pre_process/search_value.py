"""floodgate 局面を探索して value 教師を作る．

## なぜこれが要るのか

現在の value 教師は HCPE の `gameResult` から作られるので，
**1 対局に属する ~110 局面が全部同じ 0/1 を持つ**．学習の最小化戦略として
「この局面はどの対局のものかを思い出す」が有効になってしまい，
その近道は未知の対局では 1 ビットも稼げない．

実測でも記憶の痕跡は手数とともに増える (学習期間内と held-out の Brier 比が
序盤 1.08 に対し 120 手以降 2.31)．中終盤の局面はほぼ一意なので
対局の同定が容易であり，機序と一致する．

**探索値は同一対局の中でも局面ごとに異なる**ので，この近道が教師の予測に
使えなくなる．本モジュールは floodgate の既存局面をそのまま探索して
`resultValue` を差し替えるための値を作る (`docs/design/training-quality/` §3.3)．

policy 教師 (floodgate の実指し手) には手を触れない．局面の分布も変わらない．

## 選定はラベルと独立でなければならない

対象の絞り込みには**手数と出現回数しか使わない**．
「モデルが外している局面を選ぶ」は能動学習として魅力的だが，
学習分布がモデルの誤りへ偏り較正測定の前提が壊れるので採らない．

## 千日手履歴について

HCPE は局面のみを持ち指し手履歴を持たないため，SFEN へ復元した時点で
**千日手の文脈は失われる**．元の対局で千日手絡みだった局面は，
その文脈なしに評価される．
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from tqdm.auto import tqdm

logger: logging.Logger = logging.getLogger(__name__)

#: 出力 feather のスキーマ．``id`` は前処理出力の ``id`` と同じ Zobrist hash．
#:
#: ``playouts`` / ``stop`` / ``elapsedMs`` / ``warmupMs`` は**律速の切り分け用**
#: に持つ．本番実行そのものが計測になるので，別途 A/B を組まなくても「時間が
#: playout に行っているのか詰み探索に行っているのか」を後から追える．
#: ``elapsedMs`` は 0.82.0 で，``warmupMs`` は 0.97.0 で追加したため**古い出力
#: には無い** (`_with_current_schema` が null で補う)．
#:
#: ``warmupMs`` を別に持つのは，**``elapsedMs`` が探索本体だけで 1 局面の総
#: コストではない**ためである．root の同期評価とノードプールの確保は計測区間
#: の外 (Rust 側 ``warmup_ms``) にあり，この経路は保持木を引き継がないので
#: **局面ごとに払われる**．``elapsedMs`` だけで所要時間を外挿すると下限を
#: 見積もることになる．
SEARCH_VALUE_SCHEMA: dict[str, pl.DataType] = {
    "id": pl.UInt64(),
    "searchWinRate": pl.Float32(),
    "playouts": pl.Int32(),
    "stop": pl.String(),
    "elapsedMs": pl.Int32(),
    "warmupMs": pl.Int32(),
}

#: 前処理が ``resultValue`` の差し替えに実際に使う列．
#:
#: 診断列 (``playouts`` / ``stop`` / ``elapsedMs`` / ``warmupMs``) は前処理では
#: 読まない．
#: **読み込み時点でこの 2 列へ射影する**ので，診断列の構成が違うファイル
#: (``elapsedMs`` を持たない 0.82.0 より前の出力など) 同士でも union できる．
SEARCH_VALUE_REQUIRED_SCHEMA: dict[str, pl.DataType] = {
    "id": pl.UInt64(),
    "searchWinRate": pl.Float32(),
}


@dataclass(kw_only=True, frozen=True)
class SearchValueOption:
    """探索による value 教師生成のオプション．

    Attributes:
        input_path: HCPE (`.feather`) を含むディレクトリまたはファイル．
        output_path: シャード (`part_NNNNNNNN.feather`) を書き出す
            ディレクトリ．flush ごとに 1 枚増える．
        model_path: ONNX モデルのパス．None なら mock 評価器 (API 検証用)．
        min_ply: この手数以上の局面のみ対象にする．記憶は中終盤に集中する
            ので既定は 60．
        max_positions: 対象局面数の上限 (0 で無制限)．GPU 予算に合わせる．
        seed: 上限を超えたときの標本抽出の乱数種．
        max_playouts: 1 局面あたりの playout 上限．
        time_ms: 1 局面あたりの時間上限 (ミリ秒)．
        threads: 探索スレッド数．
        batch_size: 評価バッチサイズ．GPU では 64 以上．
        node_capacity: 探索木のノードプール容量．None なら playout 予算から
            決める (`_node_capacity`)．Rust 既定の 2^20 は予算に対して 3 桁
            過剰で，確保が 1 局面あたりの固定費になる．
        root_dfpn: ルート並行 dfpn 詰み探索を行うか．
        root_dfpn_nodes: ルート dfpn のノード予算 (None で Rust 既定 2,000,000)．
        root_dfpn_depth: ルート dfpn の深さ上限 (None で Rust 既定)．
        leaf_mate: 葉の短手詰み探索を行うか．
        leaf_mate_nodes: leaf-mate 1 回あたりのノード予算 (None で Rust 既定)．
        leaf_mate_threads: leaf-mate 専用スレッド数 (None で Rust 既定)．
        defensive_mate: 受け方向の詰み探索 (root 敗着フィルタ)．None で Rust 既定．
        defensive_mate_threads: root 敗着フィルタの並列度 (None で Rust 既定)．
        pad_buckets: TensorRT の padding を `batch_size` 固定でなく 2 冪バケットへ
            切り上げる．**この用途は 1 局面 1 探索で毎回 root から立ち上げるので
            葉が少なく，固定 padding だと 1 件の評価が `batch_size` 件分のコストを
            払う**．None で Rust 既定 (固定 padding)．
        cuda: CUDA Execution Provider を使うか．
        tensorrt: TensorRT Execution Provider を使うか．
        trt_engine_cache_dir: TensorRT エンジンキャッシュ保存先．
        resume: 出力が既にあるとき，未計算の局面だけを追加で探索する．
        overwrite: 出力ディレクトリを削除して作り直す．``resume`` と
            どちらも指定しなければ既存出力があるとエラーにする．
        flush_interval: 途中結果を書き出す局面数の間隔．中断しても
            ここまでの結果が残り ``resume`` で再開できる．
        shard_rows: 確定シャード 1 枚に収める目標行数．flush は小さな
            ``pending_*.feather`` を足すだけで，累積がこの行数に達したら
            1 枚の ``part_*.feather`` へまとめて pending を消す．
            **1 行あたり実測 19.4 B** なので 5,000,000 行 ≒ 97MB．
            大きくするほどファイル数は減るが，中断時に pending として
            残る量も増える．
    """

    input_path: Path
    output_path: Path
    model_path: Path | None = None
    min_ply: int = 60
    max_positions: int = 0
    seed: int = 0
    max_playouts: int = 800
    time_ms: int | None = None
    threads: int = 1
    batch_size: int = 8
    node_capacity: int | None = None
    root_dfpn: bool = True
    root_dfpn_nodes: int | None = None
    root_dfpn_depth: int | None = None
    leaf_mate: bool = True
    leaf_mate_nodes: int | None = None
    leaf_mate_threads: int | None = None
    defensive_mate: bool | None = None
    defensive_mate_threads: int | None = None
    pad_buckets: bool | None = None
    cuda: bool = False
    tensorrt: bool = False
    trt_engine_cache_dir: Path | None = None
    resume: bool = False
    overwrite: bool = False
    flush_interval: int = 500
    shard_rows: int = 5_000_000


#: `--node-capacity` を省いたときにノードプールへ足す余裕 (ノード数)．
#:
#: ノードは「未展開の子へ降りた playout」1 回につき 1 個しか確保されない
#: ので，必要数は playout 予算で上から押さえられる．この余裕は root の
#: 1 個と，CAS 競合に負けて捨てられるノード (Rust 側 `leaked_nodes`) を
#: まとめて呑むためのもの．
NODE_CAPACITY_MARGIN: int = 4096


def _node_capacity(option: SearchValueOption) -> int:
    """1 局面あたりのノードプール容量を決める．

    Rust 既定の 2^20 (ノード約 48 B なので約 50MB) は playout 予算に対して
    3 桁過剰である．**この経路は保持木を引き継がない**ため確保は局面ごとに
    払われ，しかも計測区間の外 (`warmupMs`) にあるので `elapsedMs` には
    出ないまま壁時計に乗る．

    容量を絞っても探索は変わらない — ノードプールの GC はプールが**枯渇
    したときにしか**走らないので，必要数を上回っている限り木は同一になる．
    上回っているかは実行後の `gc_runs` で観測する (発火したら警告する)．

    Args:
        option: 実行オプション．

    Returns:
        ノードプール容量 (ノード数)．
    """
    if option.node_capacity is not None:
        return option.node_capacity
    return 2 * option.max_playouts + NODE_CAPACITY_MARGIN


#: 探索値のディレクトリ指定で拾う拡張子．
#:
#: 書き出しは `.feather` だが，Arrow IPC を `.arrow` で書いた古い出力や手で
#: 組んだ入力もあり得る．**拾えなかったファイルは無言で部分適用になる**ので，
#: 実際に読める形式は拾っておく．`_write` の中間ファイル (`*.feather.tmp`) は
#: どちらにも一致しない．
SEARCH_VALUE_SUFFIXES: tuple[str, ...] = (".feather", ".arrow")

#: 確定シャードのファイル名 (`--output-path` 配下)．
#:
#: 桁を固定するのは，`_feather_paths` の `sorted()` が**辞書順**だからである．
#: `part_10.feather` が `part_2.feather` より前に来ると「パス順で後勝ち」の
#: 重複解決が探索順とずれ，**古い値が新しい値を上書きする**．
SHARD_FORMAT: str = "part_{index:08d}.feather"

#: 未確定シャードのファイル名 (flush ごとに 1 枚，確定時にまとめて消える)．
#:
#: `part_` < `pending_` という辞書順が，そのまま「確定 → 未確定」の順序に
#: なる．未確定分は常に後に読まれるので「後勝ち」の向きが正しくなる．
PENDING_FORMAT: str = "pending_{index:08d}.feather"

SHARD_PATTERN: re.Pattern[str] = re.compile(
    r"^part_(\d{8})\.feather$"
)
PENDING_PATTERN: re.Pattern[str] = re.compile(
    r"^pending_(\d{8})\.feather$"
)


def _indexed(
    path: Path, pattern: re.Pattern[str]
) -> int | None:
    """ファイル名から連番を取り出す (該当しなければ None)．

    Args:
        path: 判定するパス．
        pattern: 連番を 1 つ捕獲する正規表現．

    Returns:
        連番．該当しなければ None．
    """
    matched = pattern.match(path.name)
    return int(matched.group(1)) if matched else None


def _next_index(
    output_dir: Path, pattern: re.Pattern[str]
) -> int:
    """次に書く連番を返す．

    resume では既存の続きから振る．**個数ではなく最大値の次を取る**: 途中を
    手で消した運用でも既存を上書きしない．

    Args:
        output_dir: 出力ディレクトリ．
        pattern: 対象のファイル名パターン．

    Returns:
        次の連番 (既存が無ければ 1)．
    """
    if not output_dir.is_dir():
        return 1
    indices = [
        i
        for i in (
            _indexed(p, pattern) for p in output_dir.iterdir()
        )
        if i is not None
    ]
    return max(indices, default=0) + 1


def _ordered_value_paths(paths: Sequence[Path]) -> list[Path]:
    """探索値ファイルを「古い → 新しい」順に並べる．

    重複 id は後勝ちで解決するので，**並び順がそのまま新旧の判定になる**．
    素の辞書順では旧形式の単一ファイル (`search_values.feather` など) が
    `part_*` より後ろに来てしまい，移行直後に**旧い値が新しいシャードを
    上書きする**．明示的に「旧形式 → 確定シャード → 未確定シャード」の順にする．

    Args:
        paths: 並べ替える対象．

    Returns:
        古い順に並べたパス．
    """

    def key(p: Path) -> tuple[int, int, str]:
        part = _indexed(p, SHARD_PATTERN)
        if part is not None:
            return (1, part, p.name)
        pending = _indexed(p, PENDING_PATTERN)
        if pending is not None:
            return (2, pending, p.name)
        # 旧形式や手で置いたファイル．連番が無いので名前順で安定させる
        return (0, 0, str(p))

    return sorted(paths, key=key)


def _feather_paths(
    path: Path, suffixes: tuple[str, ...] = (".feather",)
) -> list[Path]:
    """ファイルならそれ自身，ディレクトリなら配下の該当ファイルを列挙する．

    パス指定オプションはこのプロジェクト全体でディレクトリを受け付けるので，
    探索値の入出力も同じ規約に揃える．

    Args:
        path: ディレクトリまたは単一ファイル．
        suffixes: ディレクトリ配下で拾う拡張子．

    Returns:
        該当パスの一覧 (パス順で安定)．
    """
    if path.is_file():
        return [path]
    found: list[Path] = []
    for suffix in suffixes:
        found.extend(path.glob(f"**/*{suffix}"))
    return sorted(found)


def _collect_feather_paths(
    paths: Sequence[Path],
    suffixes: tuple[str, ...] = (".feather",),
) -> list[Path]:
    """複数の入力パスを列挙して重複を畳む．

    `--search-value-path` は複数回指定できる．同じディレクトリを 2 回渡したり，
    ディレクトリとその配下のファイルを両方渡したりすると同じファイルが 2 度
    現れるが，**探索値の重複は前処理の join で行を増やしかねない**ので入口で
    畳んでおく (`load_search_values` の id 一意化より手前で効かせる)．

    Args:
        paths: ディレクトリまたはファイルの一覧．
        suffixes: ディレクトリ配下で拾う拡張子．

    Returns:
        実体で重複排除したパス一覧 (与えられた順序を保つ)．
    """
    seen: set[Path] = set()
    found: list[Path] = []
    for path in paths:
        for p in _feather_paths(path, suffixes):
            resolved = p.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            found.append(p)
    return found


def _hcpe_paths(
    input_path: Path, exclude: Path | None = None
) -> list[Path]:
    """入力パス配下の HCPE feather を列挙する．

    `exclude` は出力先を想定している．出力を入力ディレクトリ配下へ置く運用は
    あり得るが，それを HCPE として読むと `hcp` カラムが無くて落ちる．
    **出力はシャードのディレクトリなので，配下を丸ごと外す**必要がある
    (単一パス比較では `part_0001.feather` を HCPE として読んでしまう)．

    Args:
        input_path: ディレクトリまたは単一ファイル．
        exclude: 列挙から外すパス．ディレクトリならその配下すべて．

    Returns:
        `.feather` のパス一覧 (安定した順序)．
    """
    paths = _feather_paths(input_path)
    if exclude is None:
        return paths
    skipped = exclude.resolve()
    return [
        p
        for p in paths
        if p.resolve() != skipped
        and skipped not in p.resolve().parents
    ]


def _ply_of(record_id: str) -> int:
    """HCPE の ``id`` (``{棋譜名}.hcpe_{ply}``) から手数を取り出す．

    Args:
        record_id: HCPE の ``id`` カラムの値．

    Returns:
        手数．接尾辞を解釈できない場合は -1．
    """
    _, sep, tail = record_id.rpartition(".hcpe_")
    if not sep:
        return -1
    try:
        return int(tail)
    except ValueError:
        return -1


def select_positions(
    df: pl.DataFrame,
    hashes: np.ndarray,
    *,
    min_ply: int,
    max_positions: int,
    seed: int,
    already_done: Sequence[int] | np.ndarray = (),
) -> np.ndarray:
    """探索対象の行番号を返す．

    絞り込みは**手数と重複のみ**で行い，ラベルは一切見ない．
    同一局面 (同じ Zobrist hash) は 1 回だけ探索すれば足りるので
    最初の 1 行に代表させる．

    Args:
        df: HCPE の DataFrame (``id`` カラムが必要)．
        hashes: 行ごとの Zobrist hash．
        min_ply: この手数以上を対象にする．
        max_positions: 上限 (0 で無制限)．
        seed: 上限超過時の標本抽出の乱数種．
        already_done: 既に計算済みの hash (resume 用)．

    Returns:
        `df` に対する行番号の配列 (昇順)．
    """
    ply = np.array(
        [_ply_of(s) for s in df["id"].to_list()], dtype=np.int64
    )
    keep = ply >= min_ply
    if len(already_done):
        keep &= ~np.isin(hashes, np.asarray(already_done))
    rows = np.where(keep)[0]
    # 同一 hash は 1 回でよい (前処理は hash で集約するため)
    _, first = np.unique(hashes[rows], return_index=True)
    rows = np.sort(rows[first])
    if max_positions and len(rows) > max_positions:
        rng = np.random.default_rng(seed)
        rows = np.sort(
            rng.choice(rows, max_positions, replace=False)
        )
    return rows


def _check_search_value_schema(
    path: Path, schema: Mapping[str, pl.DataType]
) -> None:
    """1 ファイルのスキーマが前処理で使える形かを検査する．

    Args:
        path: 検査対象のパス (エラーメッセージ用)．
        schema: そのファイルのスキーマ．

    Raises:
        ValueError: 必要な列が無い，または型が数値でない場合．
    """
    missing = [
        c
        for c in SEARCH_VALUE_REQUIRED_SCHEMA
        if c not in schema
    ]
    if missing:
        raise ValueError(
            f"{path} is not a search value file: it has no "
            f"{', '.join(missing)} column "
            f"(its columns are {', '.join(schema) or '<none>'}). "
            "--search-value-path takes the output of "
            "`maou utility search-values`; when it is a directory, every "
            f"{'/'.join(SEARCH_VALUE_SUFFIXES)} under it must be such an "
            "output -- point it at the search value directory, not at the "
            "HCPE directory the values were searched from."
        )
    for (
        column,
        expected,
    ) in SEARCH_VALUE_REQUIRED_SCHEMA.items():
        actual = schema[column]
        if not actual.is_numeric():
            raise ValueError(
                f"{path}: column '{column}' has type {actual}, which cannot "
                f"be used as the {expected} that pre-processing needs. "
                "The file does not look like the output of "
                "`maou utility search-values`."
            )


def validate_search_value_source(
    paths: Sequence[Path],
) -> list[Path]:
    """``--search-value-path`` が読める形かを**データを読まずに**検査する．

    スキーマ検査は IPC のフッタしか触らないので，実行の一番手前で全ファイル
    分をまとめて呼べる．前処理の本体はここを通ってから，入力のダウンロードや
    リサイズといった高価な準備に進む．

    Args:
        paths: ディレクトリまたはファイルの一覧 (`--search-value-path` は
            複数回指定できる)．

    Returns:
        検査を通ったパスの一覧 (与えられた順序，重複排除済み)．

    Raises:
        ValueError: 対象ファイルが 1 つも無い，Arrow IPC として読めない，
            前処理に必要な列が無い，または型が使えない場合．
    """
    found = _collect_feather_paths(paths, SEARCH_VALUE_SUFFIXES)
    if not found:
        given = ", ".join(str(p) for p in paths) or "(none)"
        raise ValueError(
            f"{given} contains no "
            f"{' or '.join(SEARCH_VALUE_SUFFIXES)} file. "
            "--search-value-path takes the output of "
            "`maou utility search-values`: the shard directory, a single "
            "file, or any mix of them (the option can be repeated)."
        )

    for p in found:
        try:
            schema = pl.scan_ipc(p).collect_schema()
        except Exception as exc:
            raise ValueError(
                f"{p} could not be read as a feather (Arrow IPC) file: {exc}"
            ) from exc
        _check_search_value_schema(p, dict(schema))
    return found


def load_search_values(paths: Sequence[Path]) -> pl.DataFrame:
    """``--search-value-path`` を読み，前処理が使う 2 列の表にまとめる．

    ディレクトリを渡すと配下の対象ファイルを全て union する (行方向の連結)．
    シャード出力 (`part_NNNNNNNN.feather`) はそのままディレクトリを渡せばよい．

    読み出しは **Rust バックエンド** ([`load_generic_df`]) を通す．書き出し側も
    Rust なので，Arrow IPC の File/Stream 判定を書き手と読み手で揃えられる．

    **重複 id はここで 1 回だけ解決する** (パス順で後勝ち)．resume を重ねた
    出力が同じディレクトリに並ぶと重複は普通に起こるが，`apply_search_values`
    は出力チャンクごとに呼ばれるので，そちらに任せると union 全体の重複排除と
    同一の警告をチャンク数だけ繰り返すことになる．

    Args:
        paths: ディレクトリまたはファイルの一覧．

    Returns:
        `SEARCH_VALUE_REQUIRED_SCHEMA` の DataFrame (id で一意)．

    Raises:
        ValueError: `validate_search_value_source` と同じ条件．
    """
    from maou.domain.data.rust_io import load_generic_df

    found = validate_search_value_source(paths)

    columns = list(SEARCH_VALUE_REQUIRED_SCHEMA)
    frames: list[pl.DataFrame] = []
    for p in found:
        try:
            # 列射影は Rust 側に無いので読んでから落とす．診断列 (playouts /
            # stop / elapsedMs) の構成が違うファイル同士でも，ここで必要 2 列へ
            # 揃うので union できる．
            frames.append(
                load_generic_df(p)
                .select(columns)
                .cast(SEARCH_VALUE_REQUIRED_SCHEMA)  # type: ignore[arg-type]
            )
        except Exception as exc:
            raise ValueError(
                f"{p}: could not read {', '.join(columns)} as "
                f"{', '.join(str(t) for t in SEARCH_VALUE_REQUIRED_SCHEMA.values())}"
                f": {exc}"
            ) from exc

    unioned = (
        frames[0]
        if len(frames) == 1
        else pl.concat(frames, how="vertical")
    )
    # maintain_order: 既定の unique は行順を保たない．join は左の順序で
    # 出るので前処理出力は変わらないが，ここが実行ごとに入れ替わると
    # 「パス順で後勝ち」が結果から確かめられなくなる．並べ直しは union 1 回分
    values = unioned.unique(
        subset=["id"], keep="last", maintain_order=True
    )
    if len(values) != len(unioned):
        logger.warning(
            "dropped %d duplicate id(s) while unioning search values; "
            "kept the occurrence from the last file in path order",
            len(unioned) - len(values),
        )
    # 拾えなかったファイルは無言の部分適用になるので，何を読んだか残す
    given = ", ".join(str(p) for p in paths)
    logger.info(
        "Loaded %d search values from %d file(s) under %s: %s",
        len(values),
        len(found),
        given,
        ", ".join(p.name for p in found),
    )
    if values.is_empty():
        logger.warning(
            "%s holds no search value; every resultValue will keep its "
            "game-result value",
            given,
        )
    return values


def apply_search_values(
    df: pl.DataFrame, values: pl.DataFrame
) -> tuple[pl.DataFrame, int]:
    """前処理出力の ``resultValue`` を探索値へ差し替える．

    `values` に無い局面はそのまま (対局結果由来の値) 残る．
    したがって floodgate の全局面を探索しなくても部分適用できる．

    **`values` は id で一意化してから join する．** 左 join は右側に同じキーが
    2 行あると左の行を複製するため，重複を許すと**前処理出力の行数が増えて
    学習データが静かに壊れる**．行数不変はここで保証する．

    Args:
        df: 前処理出力の DataFrame (``id`` / ``resultValue`` が必要)．
        values: `SEARCH_VALUE_SCHEMA` の DataFrame．

    Returns:
        `(差し替え後の DataFrame, 差し替えた行数)`．

    Raises:
        RuntimeError: join で行数が変わった場合 (起こらないはずの保険)．
    """
    if values.is_empty():
        return df, 0
    unique_values = values.unique(subset=["id"], keep="last")
    if len(unique_values) != len(values):
        logger.warning(
            "search value input has %d duplicate id(s); keeping the last "
            "occurrence of each (with a directory, that is the last file "
            "in path order)",
            len(values) - len(unique_values),
        )
    joined = df.join(
        unique_values.select(["id", "searchWinRate"]),
        on="id",
        how="left",
    )
    if len(joined) != len(df):
        raise RuntimeError(
            f"join changed the row count ({len(df)} -> {len(joined)}); "
            "the search value file is not unique by id"
        )
    applied = int(joined["searchWinRate"].is_not_null().sum())
    return (
        joined.with_columns(
            pl.coalesce(
                pl.col("searchWinRate"), pl.col("resultValue")
            )
            .cast(pl.Float32)
            .alias("resultValue")
        ).drop("searchWinRate"),
        applied,
    )


def _with_current_schema(df: pl.DataFrame) -> pl.DataFrame:
    """古い出力を現行スキーマへ揃える．

    `elapsedMs` は 0.82.0 で，`warmupMs` は 0.97.0 で追加した．**既に走って
    いる実行の出力を捨てさせない**ため，欠けている列は null で補って
    `--resume` を継続できるようにする．

    Args:
        df: 読み込んだ既存出力．

    Returns:
        `SEARCH_VALUE_SCHEMA` の列を揃えた DataFrame．
    """
    missing = [
        c for c in SEARCH_VALUE_SCHEMA if c not in df.columns
    ]
    if missing:
        logger.info(
            "Backfilling %s from an older output format",
            ", ".join(missing),
        )
        df = df.with_columns(
            [
                pl.lit(
                    None, dtype=SEARCH_VALUE_SCHEMA[c]
                ).alias(c)
                for c in missing
            ]
        )
    return df.select(list(SEARCH_VALUE_SCHEMA))


def _frame(
    ids: list[int],
    win_rates: list[float],
    playouts: list[int],
    stops: list[str],
    elapsed_ms: list[int],
    warmup_ms: list[int],
) -> pl.DataFrame:
    """探索結果の列から `SEARCH_VALUE_SCHEMA` の DataFrame を作る．

    Args:
        ids: Zobrist hash．
        win_rates: 手番側から見た探索の勝率．
        playouts: 消化した playout 数．
        stops: 停止理由．
        elapsed_ms: 1 局面あたりの探索時間 (ミリ秒)．
        warmup_ms: 1 局面あたりの計測区間外コスト (ミリ秒)．root の同期
            評価とノードプール確保．

    Returns:
        `SEARCH_VALUE_SCHEMA` の DataFrame．
    """
    return pl.DataFrame(
        {
            "id": pl.Series("id", ids, dtype=pl.UInt64),
            "searchWinRate": pl.Series(
                "searchWinRate", win_rates, dtype=pl.Float32
            ),
            "playouts": pl.Series(
                "playouts", playouts, dtype=pl.Int32
            ),
            "stop": pl.Series("stop", stops, dtype=pl.String),
            "elapsedMs": pl.Series(
                "elapsedMs", elapsed_ms, dtype=pl.Int32
            ),
            "warmupMs": pl.Series(
                "warmupMs", warmup_ms, dtype=pl.Int32
            ),
        },
        schema=SEARCH_VALUE_SCHEMA,
    )


def _merge(
    done: pl.DataFrame, fresh: pl.DataFrame
) -> pl.DataFrame:
    """既存の結果と今回の結果を連結し，id で一意化する．

    通常の経路では走査時に既探索の hash を除外しているので重複は出ない．
    それでも一意化するのは，**重複した出力を前処理に渡すと行数が増えて
    学習データが壊れる**ためで，ここが最後の防波堤になる
    (手で結合したファイルなど経路外の入力もあり得る)．

    Args:
        done: 既存の結果 (resume 時のみ非空)．
        fresh: 今回探索した結果．

    Returns:
        id で一意な DataFrame (重複時は後勝ち = 新しい探索を残す)．
    """
    merged = (
        fresh
        if done.is_empty()
        else pl.concat([done, fresh], how="vertical")
    )
    deduped = merged.unique(subset=["id"], keep="last")
    if len(deduped) != len(merged):
        logger.warning(
            "dropped %d duplicate id(s) while merging search values",
            len(merged) - len(deduped),
        )
    return deduped


class SearchValueCollector:
    """HCPE の局面を探索して value 教師を作るユースケース．

    モデルのロードと TensorRT エンジンの構築は 1 回だけ行い
    (`maou._rust.maou_search.SearchEngine`)，以降は局面ごとに探索する．
    """

    logger: logging.Logger = logging.getLogger(__name__)

    def _load_done(
        self, option: SearchValueOption
    ) -> pl.DataFrame:
        """resume 用に既存の出力を読む．

        Args:
            option: 実行オプション．

        Returns:
            既存の出力 (無ければ空の DataFrame)．
        """
        empty = pl.DataFrame(schema=SEARCH_VALUE_SCHEMA)
        if not option.resume or not option.output_path.exists():
            return empty

        from maou.domain.data.rust_io import load_generic_df

        paths = _ordered_value_paths(
            _feather_paths(
                option.output_path, SEARCH_VALUE_SUFFIXES
            )
        )
        if not paths:
            return empty
        # 旧形式 (単一ファイル) をディレクトリへ移した運用も拾えるよう，
        # シャード名に限らず配下の feather を全て読む．
        frames = [
            _with_current_schema(load_generic_df(p))
            for p in paths
        ]
        done = (
            frames[0]
            if len(frames) == 1
            else pl.concat(frames, how="vertical")
        )
        done = done.unique(subset=["id"], keep="last")
        self.logger.info(
            "Resuming: %d positions already searched in %d shard(s)",
            len(done),
            len(paths),
        )
        return done

    def _read_hashes(
        self, path: Path
    ) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
        """1 ファイルを読んで `(df, hcp, hash)` を返す．

        Args:
            path: HCPE feather のパス．

        Returns:
            `(DataFrame, hcp 配列 (N,32), Zobrist hash 配列 (N,))`．
        """
        from maou._rust.maou_search import hcp_hashes

        df = pl.read_ipc(path, memory_map=False)
        hcp = np.ascontiguousarray(
            np.stack(
                [
                    np.frombuffer(b, np.uint8)
                    for b in df["hcp"].to_numpy()
                ]
            )
        )
        return df, hcp, np.asarray(hcp_hashes(hcp))

    def _scan_targets(
        self, option: SearchValueOption, done: pl.DataFrame
    ) -> np.ndarray:
        """探索対象の hash を集める (SFEN は作らない)．

        SFEN まで先に作ると数百万局面で GB 単位のメモリを食うので，
        走査は hash (8 バイト) だけに留め，SFEN は探索の直前に作る．

        Args:
            option: 実行オプション．
            done: 既に計算済みの出力．

        Returns:
            昇順・重複なしの hash 配列 (`--max-positions` 適用後)．
        """
        done_ids = done["id"].to_numpy()
        paths = _hcpe_paths(
            option.input_path, option.output_path
        )
        chunks: list[np.ndarray] = []
        for path in tqdm(
            paths, desc="Scanning HCPE", unit="file"
        ):
            df, _, hashes = self._read_hashes(path)
            rows = select_positions(
                df,
                hashes,
                min_ply=option.min_ply,
                max_positions=0,
                seed=option.seed,
                already_done=done_ids,
            )
            if len(rows):
                chunks.append(hashes[rows])
        if not chunks:
            return np.empty(0, dtype=np.uint64)
        # ファイルをまたぐ重複もここで落とす (np.unique は昇順で返す)
        selected = np.unique(np.concatenate(chunks))
        if (
            option.max_positions
            and len(selected) > option.max_positions
        ):
            rng = np.random.default_rng(option.seed)
            selected = np.sort(
                rng.choice(
                    selected,
                    option.max_positions,
                    replace=False,
                )
            )
        return selected

    def _iter_targets(
        self, option: SearchValueOption, selected: np.ndarray
    ) -> Iterator[tuple[int, str]]:
        """選ばれた hash の `(hash, sfen)` をファイル順に返す．

        Args:
            option: 実行オプション．
            selected: `_scan_targets` が返した昇順の hash 配列．

        Yields:
            `(Zobrist hash, SFEN)`．同じ hash は 1 回だけ返す．
        """
        from maou._rust.maou_shogi import PyBoard

        emitted = np.zeros(len(selected), dtype=bool)
        board = PyBoard()
        for path in _hcpe_paths(
            option.input_path, option.output_path
        ):
            if emitted.all():
                break
            _, hcp, hashes = self._read_hashes(path)
            pos = np.searchsorted(selected, hashes)
            np.clip(pos, 0, max(len(selected) - 1, 0), out=pos)
            hit = selected[pos] == hashes
            for i in np.where(hit)[0]:
                j = pos[i]
                if emitted[j]:
                    continue
                emitted[j] = True
                board.set_hcp(bytes(hcp[i]))
                yield int(hashes[i]), board.sfen()

    def collect(
        self, option: SearchValueOption
    ) -> dict[str, str]:
        """対象局面を探索し，結果を feather へ書き出す．

        Args:
            option: 実行オプション．

        Returns:
            表示用の要約 dict．

        Raises:
            ValueError: `resume` と `overwrite` を同時に指定した場合，
                または出力が既にあるのにどちらも指定されていない場合．
        """
        from maou._rust.maou_search import SearchEngine

        if option.output_path.is_file():
            # 旧形式 (単一ファイル) を黙ってディレクトリ扱いすると，既存の
            # 数日分をどう扱ったのか利用者から見えない．移行の手順を明示する
            raise ValueError(
                f"--output-path is now a directory of shards, but "
                f"{option.output_path} is a file. Create a directory and "
                f"move the old output into it "
                f"(`mkdir -p DIR && mv {option.output_path} DIR/`), then "
                "pass DIR with --resume; every feather under it is read."
            )

        if option.resume and option.overwrite:
            # 「続きから」と「作り直し」は両立しない．暗黙の優先順位を持たせると
            # 作り直したつもりで古い値が残る (実測で確認した)
            raise ValueError(
                "--resume and --overwrite are mutually exclusive. Use "
                "--overwrite once to start a new run, then --resume for "
                "every following run."
            )

        existing = _feather_paths(
            option.output_path, SEARCH_VALUE_SUFFIXES
        )
        if (
            existing
            and not option.resume
            and not option.overwrite
        ):
            # 出力は数日分の GPU 時間そのものなので黙って捨てさせない
            raise ValueError(
                f"{option.output_path} already holds {len(existing)} "
                "search value file(s). Pass --resume to continue "
                "accumulating into it (already-searched positions are "
                "skipped), or --overwrite to discard it and start over."
            )

        if option.overwrite and option.output_path.exists():
            # シャードは複数ファイルなので，作り直しはディレクトリごと消す．
            # 残骸が 1 枚でも生き残ると，次の resume が古い値を拾う
            self.logger.info(
                "Discarding %d existing search value file(s) in %s",
                len(existing),
                option.output_path,
            )
            shutil.rmtree(option.output_path)

        done = self._load_done(option)
        selected = self._scan_targets(option, done)

        # **前回の実行が残した未確定シャードを引き継ぐ．** ここを空で始めると
        # 前回の pending が永久に確定されず，蓄積カウンタも実行ごとに 0 へ戻る
        # ため，中断を繰り返すほど小さなファイルが溜まり続ける．
        shard_index = _next_index(
            option.output_path, SHARD_PATTERN
        )
        pending_index = _next_index(
            option.output_path, PENDING_PATTERN
        )
        pending_paths = [
            p
            for p in _ordered_value_paths(
                _feather_paths(
                    option.output_path, SEARCH_VALUE_SUFFIXES
                )
            )
            if _indexed(p, PENDING_PATTERN) is not None
        ]
        carry = self._read_pending(pending_paths)
        empty_frame = pl.DataFrame(schema=SEARCH_VALUE_SCHEMA)

        def compact(
            extra: pl.DataFrame, *, quiet: bool
        ) -> None:
            """引き継ぎ分 + `extra` を 1 枚の確定シャードにまとめる．"""
            nonlocal shard_index, pending_paths, carry
            if carry.is_empty():
                frame = extra
            elif extra.is_empty():
                frame = carry
            else:
                frame = pl.concat(
                    [carry, extra], how="vertical"
                )
            if frame.is_empty():
                return
            self._write_value_file(
                frame,
                option.output_path,
                SHARD_FORMAT.format(index=shard_index),
                quiet=quiet,
            )
            # **確定シャードを書いてから pending を消す**．逆順だと間で落ちた
            # ときに行が失われる．この順なら重複が残るだけで，読み手の
            # id 一意化が吸収する
            for path in pending_paths:
                path.unlink(missing_ok=True)
            pending_paths = []
            carry = empty_frame
            shard_index += 1

        self.logger.info(
            "Searching %d positions (min_ply=%d, playouts=%d)",
            len(selected),
            option.min_ply,
            option.max_playouts,
        )
        if not len(selected):
            # 探索するものは無いが，**前回の pending は確定させる**．
            # そうしないと「resume したのに小さいファイルが残ったまま」に
            # なり，何度 resume しても片付かない
            compact(empty_frame, quiet=False)
            return {
                "searched": "0",
                "total": str(len(done)),
                "output": str(option.output_path),
            }

        engine = SearchEngine(
            model_path=(
                str(option.model_path)
                if option.model_path
                else None
            ),
            threads=option.threads,
            batch_size=option.batch_size,
            use_cuda=option.cuda,
            use_tensorrt=option.tensorrt,
            pad_buckets=option.pad_buckets,
            trt_engine_cache_dir=(
                str(option.trt_engine_cache_dir)
                if option.trt_engine_cache_dir
                else None
            ),
        )

        ids: list[int] = []
        win_rates: list[float] = []
        playouts: list[int] = []
        stops: list[str] = []
        elapsed: list[int] = []
        warmup: list[int] = []
        # プール容量は探索を変えない範囲で絞る (`_node_capacity`)．正しく
        # 上回れているかは観測で確かめる: GC が 1 度でも走ったら木が刈られて
        # おり，刈られなかった実行と値が比較できなくなる．
        node_capacity = _node_capacity(option)
        gc_runs = 0
        max_nodes_used = 0
        # 2 段構え．flush は小さな `pending_*` を足すだけで既存には触れず
        # (クラッシュ保護は flush_interval 粒度のまま)，累積が `shard_rows` に
        # 達したら**メモリ上の行から** `part_*` を 1 枚書いて pending を消す．
        # 各行はちょうど 2 回書かれるだけなので総 I/O は行数に比例する．
        # 「毎回 現在のシャードを書き直す」方式はシャードを大きくするほど
        # 二乗で重くなる (300MB シャード / 18.7M 行で約 10 時間) ので採らない．
        shard_start = 0
        written = 0

        def _slice(lo: int, hi: int) -> pl.DataFrame:
            return _frame(
                ids[lo:hi],
                win_rates[lo:hi],
                playouts[lo:hi],
                stops[lo:hi],
                elapsed[lo:hi],
                warmup[lo:hi],
            )

        def flush(*, final: bool) -> None:
            """未書き出し行を pending にし，貯まっていれば確定させる．

            **実行の終わりでは目標行数に届かなくても確定させる** (`final`)．
            端数を pending のまま残して次回に育てさせる案もあるが，
            `--resume` のたびに `--shard-rows` が同じとは限らず，値が変われば
            「既存の確定シャードも分割し直すのか」という決着のつかない問題に
            なる．実行ごとに閉じておけばその問いが発生しない．
            代償として**1 回の実行につき最低 1 枚**の確定シャードができる．
            """
            nonlocal pending_index, shard_start, written
            if len(ids) > written:
                pending_paths.append(
                    self._write_value_file(
                        _slice(written, len(ids)),
                        option.output_path,
                        PENDING_FORMAT.format(
                            index=pending_index
                        ),
                        quiet=True,
                    )
                )
                pending_index += 1
                written = len(ids)
            # 引き継いだ未確定分も 1 枚の目標行数に数える
            outstanding = len(carry) + written - shard_start
            if outstanding <= 0:
                return
            if not final and outstanding < option.shard_rows:
                return
            compact(
                _slice(shard_start, written), quiet=not final
            )
            shard_start = written

        progress = tqdm(
            self._iter_targets(option, selected),
            total=len(selected),
            desc="Searching positions",
            unit="pos",
            smoothing=0.05,
        )
        for n, (hash_id, sfen) in enumerate(progress, start=1):
            result = engine.search(
                sfen,
                max_playouts=option.max_playouts,
                time_ms=option.time_ms,
                node_capacity=node_capacity,
                root_dfpn=option.root_dfpn,
                root_dfpn_nodes=option.root_dfpn_nodes,
                root_dfpn_depth=option.root_dfpn_depth,
                leaf_mate=option.leaf_mate,
                leaf_mate_nodes=option.leaf_mate_nodes,
                leaf_mate_threads=option.leaf_mate_threads,
                defensive_mate=option.defensive_mate,
                defensive_mate_threads=option.defensive_mate_threads,
            )
            ids.append(hash_id)
            win_rates.append(float(result.winrate))
            playouts.append(int(result.playouts))
            stops.append(str(result.stop))
            elapsed.append(int(result.elapsed_ms))
            warmup.append(int(result.warmup_ms))
            gc_runs += int(result.gc_runs)
            max_nodes_used = max(
                max_nodes_used, int(result.nodes_used)
            )
            # 途中経過を落とさない: 実運用では数十万局面を数日かけて回すので，
            # 最後にしか書かないと中断で全損する (--resume はここに依存する)
            if n % option.flush_interval == 0:
                flush(final=False)
                progress.set_postfix(
                    mean_wr=f"{np.mean(win_rates):.3f}",
                    ms=f"{np.mean(elapsed):.0f}",
                    flushed=n,
                )
        progress.close()

        flush(final=True)

        if gc_runs:
            # 容量の見積りが外れている．GC は木を刈るので，刈られなかった
            # 実行と探索値を並べられない (教師データとしては混ぜられない)
            logger.warning(
                "Node pool GC ran %d time(s) at capacity %d "
                "(max nodes used %d). GC prunes the tree, so these "
                "values are not comparable with a run that never "
                "collected. Raise --node-capacity and redo them.",
                gc_runs,
                node_capacity,
                max_nodes_used,
            )

        fresh = _frame(
            ids, win_rates, playouts, stops, elapsed, warmup
        )
        # `merged` は集計にしか使わない (書き出しはシャード済み)．`_merge` は
        # id 重複を最後の防波堤として潰すので，件数の整合もここで取れる．
        merged = _merge(done, fresh)
        return {
            "searched": str(len(fresh)),
            "total": str(len(merged)),
            "mean_win_rate": f"{np.mean(win_rates):.4f}",
            "mean_elapsed_ms": f"{np.mean(elapsed):.0f}",
            "mean_warmup_ms": f"{np.mean(warmup):.0f}",
            "node_capacity": str(node_capacity),
            "max_nodes_used": str(max_nodes_used),
            "gc_runs": str(gc_runs),
            "output": str(option.output_path),
        }

    def _read_pending(
        self, paths: Sequence[Path]
    ) -> pl.DataFrame:
        """未確定シャードを読み，次の確定に持ち越す行として返す．

        `--resume` で前回の pending を引き継ぐために要る．引き継がないと
        前回の未確定分が確定されないまま残り，中断のたびにファイルが増える．

        Args:
            paths: 未確定シャードのパス (古い順)．

        Returns:
            `SEARCH_VALUE_SCHEMA` の DataFrame (無ければ空)．
        """
        empty = pl.DataFrame(schema=SEARCH_VALUE_SCHEMA)
        if not paths:
            return empty
        from maou.domain.data.rust_io import load_generic_df

        frames = [
            _with_current_schema(load_generic_df(p))
            for p in paths
        ]
        carry = (
            frames[0]
            if len(frames) == 1
            else pl.concat(frames, how="vertical")
        )
        self.logger.info(
            "Carrying over %d row(s) from %d unfinished shard(s)",
            len(carry),
            len(paths),
        )
        return carry

    def _write_value_file(
        self,
        df: pl.DataFrame,
        output_dir: Path,
        name: str,
        *,
        quiet: bool = False,
    ) -> Path:
        """**新規行だけ**を 1 枚のシャードとして書き出す．

        累積全体を毎回書き直すと書き込み量が行数の二乗で伸びる (実測: 1.25M 行
        で 1 回 76ms，flush 間隔 500 局面なら 18.7M 行の運用で合計 6 時間近く)．
        シャードなら 1 回のコストが flush 間隔ぶんで一定になる．読み手
        (`load_search_values` / `_load_done`) はディレクトリを union するので，
        分割されていても意味は変わらない．

        Args:
            df: 書き出す新規行．
            output_dir: 出力ディレクトリ．
            name: 書き出すファイル名 (`part_*` / `pending_*`)．
            quiet: 途中フラッシュではログを出さない．

        Returns:
            書き出したシャードのパス．
        """
        from maou.domain.data.rust_io import save_generic_df

        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / name
        # 途中フラッシュはクラッシュと同時に起こりうる．直接書くと壊れた
        # ファイルが残り，数日分の探索が resume 不能になる．**中間名は
        # `.feather` で終わらせない** — 読み手が拾ってしまう
        tmp = path.with_name(path.name + ".tmp")
        save_generic_df(df, tmp)
        os.replace(tmp, path)
        if not quiet:
            self.logger.info(
                "Wrote %d search values to %s", len(df), path
            )
        return path
