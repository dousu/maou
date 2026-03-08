"""ゲームツリー構築ロジック(BFS)."""

from __future__ import annotations

import logging
import resource
from collections import deque
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from maou.domain.board import shogi
from maou.domain.game_tree.model import (
    GameTreeEdge,
    GameTreeNode,
)
from maou.domain.move.label import (
    make_usi_move_from_label,
)

if TYPE_CHECKING:
    import polars as pl

logger = logging.getLogger(__name__)

_UINT16_MAX = 65535
_BFS_LOG_INTERVAL = 10_000


def _get_rss_mb() -> int:
    """現在のRSS(Resident Set Size)をMB単位で返す．"""
    # Linux: ru_maxrss は KB 単位
    return (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        // 1024
    )


class GameTreeBuilder:
    """preprocessデータからゲームツリーを構築する．"""

    def build(
        self,
        preprocess_df: pl.DataFrame,
        max_depth: int = 30,
        min_probability: float = 0.001,
        progress_callback: Callable[[int, int], None]
        | None = None,
        initial_hash: int | None = None,
        initial_sfen: str | None = None,
        list_column_fn: Callable[
            [int], tuple[np.ndarray, np.ndarray]
        ]
        | None = None,
    ) -> tuple[list[GameTreeNode], list[GameTreeEdge]]:
        """BFSでツリーを構築する．

        Args:
            preprocess_df: preprocessデータのDataFrame
                (カラム: id, resultValue, bestMoveWinRate．
                list_column_fnがNoneの場合はmoveLabel, moveWinRateも必要)
            max_depth: 最大探索深さ
            min_probability: 指し手の最小確率閾値
            progress_callback: プログレスコールバック(処理済み局面数, 発見済み局面数)
            initial_hash: 開始局面のZobrist hash(Noneの場合は平手初期局面)
            initial_sfen: 開始局面のSFEN文字列．initial_hash指定時は必須
            list_column_fn: List型カラムの遅延アクセス関数．
                row_idxを受け取り(moveLabel, moveWinRate)のNumPy配列タプルを返す．
                Noneの場合はpreprocess_dfから直接読み込む(後方互換)

        Returns:
            (nodes, edges) のタプル．
            エッジの child_hash に対応するノードが存在しない場合がある
            (is_leaf=True のエッジ: 子局面がpreprocessデータに存在しない)．
            下流処理では LEFT JOIN 時に null-safe な処理が必要

        Raises:
            ValueError: 開始局面がpreprocessデータに見つからない場合，
                またはinitial_hash指定時にinitial_sfenが未指定の場合，
                またはmax_depth/min_probabilityが範囲外の場合
        """
        if max_depth > _UINT16_MAX:
            raise ValueError(
                f"max_depth={max_depth} は UInt16 の"
                f"最大値({_UINT16_MAX})を超えています．"
            )
        if not (0.0 <= min_probability <= 1.0):
            raise ValueError(
                f"min_probability={min_probability} は"
                "0.0〜1.0の範囲でなければなりません．"
            )

        # 1. ルックアップテーブル構築: id → 行インデックス(後勝ち)
        logger.info(
            "ルックアップテーブル構築開始: %s 行, RSS=%d MB",
            f"{len(preprocess_df):,}",
            _get_rss_mb(),
        )
        id_list = preprocess_df["id"].to_list()
        duplicate_count = 0
        lookup: dict[int, int] = {}
        for idx, hash_val in enumerate(id_list):
            if hash_val in lookup:
                duplicate_count += 1
            lookup[hash_val] = idx
        if duplicate_count > 0:
            logger.warning(
                "入力データに %d 件のハッシュ重複があります．"
                "(後勝ちで最後の行を使用)",
                duplicate_count,
            )
        logger.info(
            "ルックアップテーブル構築完了: %s エントリ, RSS=%d MB",
            f"{len(lookup):,}",
            _get_rss_mb(),
        )

        # 2. 開始局面のZobrist hashとSFENを決定
        if initial_hash is None:
            init_board = shogi.Board()
            initial_hash = init_board.hash()
            initial_sfen = init_board.get_sfen()
        elif initial_sfen is None:
            raise ValueError(
                "initial_hash を指定する場合は initial_sfen も"
                "指定してください．"
            )

        assert initial_sfen is not None  # if/elif で確定済み

        if initial_hash not in lookup:
            raise ValueError(
                f"開始局面(hash={initial_hash})がpreprocessデータに見つかりません．"
            )

        # スカラーカラムはNumPy配列に変換(メモリ効率が良い)
        result_value_arr = preprocess_df[
            "resultValue"
        ].to_numpy()
        best_move_win_rate_arr = preprocess_df[
            "bestMoveWinRate"
        ].to_numpy()

        # List型カラム(moveLabel, moveWinRate)のアクセス方法を決定
        # list_column_fn が指定されている場合は遅延アクセス(省メモリ)
        # 指定されていない場合はDataFrameから直接アクセス(後方互換)
        move_label_series: pl.Series | None = None
        move_win_rate_series: pl.Series | None = None
        if list_column_fn is None:
            move_label_series = preprocess_df["moveLabel"]
            move_win_rate_series = preprocess_df["moveWinRate"]
        logger.info(
            "NumPy配列変換完了: list_column_fn=%s, RSS=%d MB",
            "あり" if list_column_fn is not None else "なし",
            _get_rss_mb(),
        )

        # 3. BFS (Board インスタンスはループ外で1回だけ生成し再利用)
        board = shogi.Board()
        nodes: list[GameTreeNode] = []
        edges: list[GameTreeEdge] = []
        # visited: hash → depth(BFS最短距離を記録)
        visited: dict[int, int] = {initial_hash: 0}
        # キューは (hash, sfen) を保持し，盤面を O(1) で復元する
        queue: deque[tuple[int, str]] = deque(
            [(initial_hash, initial_sfen)]
        )
        processed = 0
        prev_depth = 0

        logger.info(
            "BFS開始: max_depth=%d, min_probability=%f, RSS=%d MB",
            max_depth,
            min_probability,
            _get_rss_mb(),
        )

        while queue:
            current_hash, current_sfen = queue.popleft()
            current_depth = visited[current_hash]
            row_idx = lookup[current_hash]

            # 深さが変わった時にログ出力
            if current_depth != prev_depth:
                logger.info(
                    "BFS depth=%d 完了: "
                    "処理済み=%s, ノード=%s, エッジ=%s, "
                    "キュー=%s, 訪問済み=%s, RSS=%d MB",
                    prev_depth,
                    f"{processed:,}",
                    f"{len(nodes):,}",
                    f"{len(edges):,}",
                    f"{len(queue):,}",
                    f"{len(visited):,}",
                    _get_rss_mb(),
                )
                prev_depth = current_depth

            # スカラー値はNumPy配列から直接取得
            result_value = float(result_value_arr[row_idx])
            best_move_win_rate = float(
                best_move_win_rate_arr[row_idx]
            )

            # List型カラムはオンデマンドで取得
            if list_column_fn is not None:
                move_labels, move_win_rates = list_column_fn(
                    row_idx
                )
            else:
                assert move_label_series is not None
                assert move_win_rate_series is not None
                move_labels = np.array(
                    move_label_series[row_idx],
                    dtype=np.float32,
                )
                move_win_rates = np.array(
                    move_win_rate_series[row_idx],
                    dtype=np.float32,
                )

            # min_probability以上の指し手を取得(NumPyで高速フィルタリング)
            candidate_indices = np.where(
                move_labels >= min_probability
            )[0]

            # max_depth に達したら展開しない
            # NOTE: num_branches はフィルタ後の候補手数(未展開のため
            # ラベル変換失敗分も含む．通常ノードとは意味が異なる)
            if current_depth >= max_depth:
                nodes.append(
                    GameTreeNode(
                        position_hash=current_hash,
                        result_value=result_value,
                        best_move_win_rate=best_move_win_rate,
                        num_branches=len(candidate_indices),
                        depth=current_depth,
                        is_depth_cutoff=True,
                    )
                )
                processed += 1
                if progress_callback:
                    progress_callback(processed, len(visited))
                continue

            # SFENから盤面を復元(O(1)，Boardインスタンスを再利用)
            board.set_sfen(current_sfen)

            # 各候補手を処理
            edges_before = len(edges)
            for label_idx in candidate_indices:
                label_idx_int = int(label_idx)
                probability = float(move_labels[label_idx_int])
                win_rate = float(move_win_rates[label_idx_int])

                # ラベルからUSI指し手に変換
                try:
                    usi_move = make_usi_move_from_label(
                        board, label_idx_int
                    )
                except ValueError:
                    logger.debug(
                        "ラベル %d の変換に失敗(hash=%d)",
                        label_idx_int,
                        current_hash,
                    )
                    continue

                # USIからmoveに変換
                try:
                    move = board.move_from_usi(usi_move)
                except (ValueError, RuntimeError) as e:
                    logger.warning(
                        "USI %s のmove変換に失敗 (hash=%d): %s",
                        usi_move,
                        current_hash,
                        e,
                    )
                    continue

                # 明示的に16-bit move形式に変換
                move16_val = shogi.move16(move)

                # 子局面のハッシュとSFENを取得
                board.push_move(move)
                child_hash = board.hash()
                child_sfen = board.get_sfen()
                board.pop_move()

                # エッジ追加
                child_in_lookup = child_hash in lookup
                edges.append(
                    GameTreeEdge(
                        parent_hash=current_hash,
                        child_hash=child_hash,
                        move16=move16_val,
                        move_label=label_idx_int,
                        probability=probability,
                        win_rate=win_rate,
                        is_leaf=not child_in_lookup,
                    )
                )

                # 子局面がルックアップテーブルにあり，未訪問の場合はキューに追加
                # BFSは等コストのため，最初に到達した経路が最短経路となる
                if (
                    child_in_lookup
                    and child_hash not in visited
                ):
                    visited[child_hash] = current_depth + 1
                    queue.append((child_hash, child_sfen))

            # ノードを追加(実際に生成されたエッジ数をnum_branchesに使用．
            # ラベル変換失敗分は除外されるため，max_depthノードとは意味が異なる)
            actual_branches = len(edges) - edges_before
            nodes.append(
                GameTreeNode(
                    position_hash=current_hash,
                    result_value=result_value,
                    best_move_win_rate=best_move_win_rate,
                    num_branches=actual_branches,
                    depth=current_depth,
                    is_depth_cutoff=False,
                )
            )

            processed += 1
            if progress_callback:
                progress_callback(processed, len(visited))

            # 定期的にBFS進捗をログ出力
            if processed % _BFS_LOG_INTERVAL == 0:
                logger.info(
                    "BFS進捗: depth=%d, 処理済み=%s, "
                    "ノード=%s, エッジ=%s, "
                    "キュー=%s, RSS=%d MB",
                    current_depth,
                    f"{processed:,}",
                    f"{len(nodes):,}",
                    f"{len(edges):,}",
                    f"{len(queue):,}",
                    _get_rss_mb(),
                )

        logger.info(
            "BFS完了: ノード=%s, エッジ=%s, "
            "最大depth=%d, RSS=%d MB",
            f"{len(nodes):,}",
            f"{len(edges):,}",
            prev_depth,
            _get_rss_mb(),
        )

        return nodes, edges
