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
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from tqdm.auto import tqdm

logger: logging.Logger = logging.getLogger(__name__)

#: 出力 feather のスキーマ．``id`` は前処理出力の ``id`` と同じ Zobrist hash．
#:
#: ``playouts`` / ``stop`` / ``elapsedMs`` は**律速の切り分け用**に持つ．
#: 本番実行そのものが計測になるので，別途 A/B を組まなくても「時間が playout に
#: 行っているのか詰み探索に行っているのか」を後から追える．
#: ``elapsedMs`` は 0.82.0 で追加したため**古い出力には無い**
#: (`_with_current_schema` が null で補う)．
SEARCH_VALUE_SCHEMA: dict[str, pl.DataType] = {
    "id": pl.UInt64(),
    "searchWinRate": pl.Float32(),
    "playouts": pl.Int32(),
    "stop": pl.String(),
    "elapsedMs": pl.Int32(),
}


@dataclass(kw_only=True, frozen=True)
class SearchValueOption:
    """探索による value 教師生成のオプション．

    Attributes:
        input_path: HCPE (`.feather`) を含むディレクトリまたはファイル．
        output_path: 出力する feather のパス．
        model_path: ONNX モデルのパス．None なら mock 評価器 (API 検証用)．
        min_ply: この手数以上の局面のみ対象にする．記憶は中終盤に集中する
            ので既定は 60．
        max_positions: 対象局面数の上限 (0 で無制限)．GPU 予算に合わせる．
        seed: 上限を超えたときの標本抽出の乱数種．
        max_playouts: 1 局面あたりの playout 上限．
        time_ms: 1 局面あたりの時間上限 (ミリ秒)．
        threads: 探索スレッド数．
        batch_size: 評価バッチサイズ．GPU では 64 以上．
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
        overwrite: 出力が既にあっても破棄して作り直す．``resume`` と
            どちらも指定しなければ既存出力があるとエラーにする．
        flush_interval: 途中結果を書き出す局面数の間隔．中断しても
            ここまでの結果が残り ``resume`` で再開できる．
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


def _hcpe_paths(
    input_path: Path, exclude: Path | None = None
) -> list[Path]:
    """入力パス配下の HCPE feather を列挙する．

    `exclude` は出力ファイルを想定している．出力を入力ディレクトリ配下へ
    置く運用はあり得るが，それを HCPE として読むと `hcp` カラムが無くて落ちる．

    Args:
        input_path: ディレクトリまたは単一ファイル．
        exclude: 列挙から外すパス (出力ファイル)．

    Returns:
        `.feather` のパス一覧 (安定した順序)．
    """
    paths = (
        [input_path]
        if input_path.is_file()
        else sorted(input_path.glob("**/*.feather"))
    )
    if exclude is None:
        return paths
    skipped = exclude.resolve()
    return [p for p in paths if p.resolve() != skipped]


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
            "search value file has %d duplicate id(s); keeping the last "
            "occurrence of each",
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

    `elapsedMs` は 0.82.0 で追加した．**既に走っている実行の出力を捨てさせない**
    ため，欠けている列は null で補って `--resume` を継続できるようにする．

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
) -> pl.DataFrame:
    """探索結果の列から `SEARCH_VALUE_SCHEMA` の DataFrame を作る．

    Args:
        ids: Zobrist hash．
        win_rates: 手番側から見た探索の勝率．
        playouts: 消化した playout 数．
        stops: 停止理由．
        elapsed_ms: 1 局面あたりの探索時間 (ミリ秒)．

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
        if option.resume and option.output_path.exists():
            done = _with_current_schema(
                pl.read_ipc(
                    option.output_path, memory_map=False
                )
            )
            self.logger.info(
                "Resuming: %d positions already searched",
                len(done),
            )
            return done
        return pl.DataFrame(schema=SEARCH_VALUE_SCHEMA)

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

        if option.resume and option.overwrite:
            # 「続きから」と「作り直し」は両立しない．暗黙の優先順位を持たせると
            # 作り直したつもりで古い値が残る (実測で確認した)
            raise ValueError(
                "--resume and --overwrite are mutually exclusive. Use "
                "--overwrite once to start a new run, then --resume for "
                "every following run."
            )

        if (
            option.output_path.exists()
            and not option.resume
            and not option.overwrite
        ):
            # 出力は数日分の GPU 時間そのものなので黙って捨てさせない
            raise ValueError(
                f"{option.output_path} already exists. Pass --resume to "
                "continue accumulating into it (already-searched positions "
                "are skipped), or --overwrite to discard it and start over."
            )

        done = self._load_done(option)
        selected = self._scan_targets(option, done)

        self.logger.info(
            "Searching %d positions (min_ply=%d, playouts=%d)",
            len(selected),
            option.min_ply,
            option.max_playouts,
        )
        if not len(selected):
            self._write(done, option.output_path)
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
            # 途中経過を落とさない: 実運用では数十万局面を数日かけて回すので，
            # 最後にしか書かないと中断で全損する (--resume はここに依存する)
            if n % option.flush_interval == 0:
                self._write(
                    _merge(
                        done,
                        _frame(
                            ids,
                            win_rates,
                            playouts,
                            stops,
                            elapsed,
                        ),
                    ),
                    option.output_path,
                    quiet=True,
                )
                progress.set_postfix(
                    mean_wr=f"{np.mean(win_rates):.3f}",
                    ms=f"{np.mean(elapsed):.0f}",
                    flushed=n,
                )
        progress.close()

        fresh = _frame(ids, win_rates, playouts, stops, elapsed)
        merged = _merge(done, fresh)
        self._write(merged, option.output_path)
        return {
            "searched": str(len(fresh)),
            "total": str(len(merged)),
            "mean_win_rate": f"{np.mean(win_rates):.4f}",
            "mean_elapsed_ms": f"{np.mean(elapsed):.0f}",
            "output": str(option.output_path),
        }

    def _write(
        self,
        df: pl.DataFrame,
        path: Path,
        *,
        quiet: bool = False,
    ) -> None:
        """結果を feather へ書き出す．

        Args:
            df: 書き出す DataFrame．
            path: 出力先．
            quiet: 途中フラッシュではログを出さない．
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        # 途中フラッシュはクラッシュと同時に起こりうる．直接書くと壊れた
        # ファイルが残り，数日分の探索が resume 不能になる
        tmp = path.with_name(path.name + ".tmp")
        df.write_ipc(tmp, compression="lz4")
        os.replace(tmp, path)
        if not quiet:
            self.logger.info(
                "Wrote %d search values to %s", len(df), path
            )
