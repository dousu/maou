"""ゲームツリー構築ロジック(BFS)."""

from __future__ import annotations

import logging
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
    ) -> tuple[list[GameTreeNode], list[GameTreeEdge]]:
        """BFSでツリーを構築する．

        Args:
            preprocess_df: preprocessデータのDataFrame
                (カラム: id, moveLabel, moveWinRate, resultValue, bestMoveWinRate)
            max_depth: 最大探索深さ
            min_probability: 指し手の最小確率閾値
            progress_callback: プログレスコールバック(処理済み局面数, 発見済み局面数)
            initial_hash: 開始局面のZobrist hash(Noneの場合は平手初期局面)

        Returns:
            (nodes, edges) のタプル

        Raises:
            ValueError: 開始局面がpreprocessデータに見つからない場合
        """
        # 1. ルックアップテーブル構築: id → 行インデックス(後勝ち)
        id_list = preprocess_df["id"].to_list()
        seen: set[int] = set()
        duplicate_count = 0
        lookup: dict[int, int] = {}
        for idx, hash_val in enumerate(id_list):
            if hash_val in seen:
                duplicate_count += 1
            seen.add(hash_val)
            lookup[hash_val] = idx
        if duplicate_count > 0:
            logger.warning(
                f"入力データに {duplicate_count} 件のハッシュ重複があります．"
                f"(後勝ちで最後の行を使用)"
            )

        # 2. 開始局面のZobrist hashを決定
        if initial_hash is None:
            board = shogi.Board()
            initial_hash = board.hash()

        if initial_hash not in lookup:
            raise ValueError(
                f"開始局面(hash={initial_hash})がpreprocessデータに見つかりません．"
            )

        # スカラーカラムはNumPy配列に変換(メモリ効率が良い)
        # List型カラム(moveLabel, moveWinRate)はPolars Seriesのまま保持し，
        # BFSで到達した行のみオンデマンドでアクセスする
        # (40Mレコード × 1496要素 × 4bytes × 2列 ≈ 477GB のため全展開不可)
        result_value_arr = preprocess_df[
            "resultValue"
        ].to_numpy()
        best_move_win_rate_arr = preprocess_df[
            "bestMoveWinRate"
        ].to_numpy()
        move_label_series = preprocess_df["moveLabel"]
        move_win_rate_series = preprocess_df["moveWinRate"]

        # 3. BFS
        nodes: list[GameTreeNode] = []
        edges: list[GameTreeEdge] = []
        # parent_info: hash → (depth, parent_hash, move)
        # パスをメモリに保持せず，必要時に遡って盤面を復元する
        parent_info: dict[
            int, tuple[int, int | None, int | None]
        ] = {initial_hash: (0, None, None)}
        queue: deque[int] = deque([initial_hash])
        processed = 0

        while queue:
            current_hash = queue.popleft()
            current_depth, _, _ = parent_info[current_hash]
            row_idx = lookup[current_hash]

            # スカラー値はNumPy配列から直接取得
            result_value = float(result_value_arr[row_idx])
            best_move_win_rate = float(
                best_move_win_rate_arr[row_idx]
            )

            # List型カラムはオンデマンドでNumPy配列に変換
            move_labels = np.array(
                move_label_series[row_idx].to_list(),
                dtype=np.float32,
            )
            move_win_rates = np.array(
                move_win_rate_series[row_idx].to_list(),
                dtype=np.float32,
            )

            # min_probability以上の指し手を取得(NumPyで高速フィルタリング)
            candidate_indices = np.where(
                move_labels >= min_probability
            )[0]

            # max_depth に達したら展開しない
            if current_depth >= max_depth:
                nodes.append(
                    GameTreeNode(
                        position_hash=current_hash,
                        result_value=result_value,
                        best_move_win_rate=best_move_win_rate,
                        num_branches=len(candidate_indices),
                        depth=current_depth,
                    )
                )
                processed += 1
                if progress_callback:
                    progress_callback(
                        processed, len(parent_info)
                    )
                continue

            # 盤面を復元(parent_infoを遡ってパスを再構成)
            board = self._reconstruct_board(
                current_hash, parent_info
            )

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
                        f"ラベル {label_idx_int} の変換に失敗"
                        f"(hash={current_hash})"
                    )
                    continue

                # USIからmoveに変換
                try:
                    move = board.move_from_usi(usi_move)
                except (ValueError, RuntimeError) as e:
                    logger.warning(
                        f"USI {usi_move} のmove変換に失敗 "
                        f"(hash={current_hash}): {e}"
                    )
                    continue

                # 明示的に16-bit move形式に変換
                move16_val = shogi.move16(move)

                # 子局面のハッシュを取得
                board.push_move(move)
                child_hash = board.hash()
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
                    and child_hash not in parent_info
                ):
                    parent_info[child_hash] = (
                        current_depth + 1,
                        current_hash,
                        move,
                    )
                    queue.append(child_hash)

            # ノードを追加(実際に生成されたエッジ数をnum_branchesに使用)
            actual_branches = len(edges) - edges_before
            nodes.append(
                GameTreeNode(
                    position_hash=current_hash,
                    result_value=result_value,
                    best_move_win_rate=best_move_win_rate,
                    num_branches=actual_branches,
                    depth=current_depth,
                )
            )

            processed += 1
            if progress_callback:
                progress_callback(processed, len(parent_info))

        return nodes, edges

    @staticmethod
    def _reconstruct_board(
        target_hash: int,
        parent_info: dict[
            int, tuple[int, int | None, int | None]
        ],
    ) -> shogi.Board:
        """parent_infoを遡って盤面を復元する．

        Args:
            target_hash: 復元対象の局面ハッシュ
            parent_info: hash → (depth, parent_hash, move)

        Returns:
            復元された盤面
        """
        # ルートまでのパスを逆順に構築
        moves: list[int] = []
        h = target_hash
        while True:
            _, parent_hash, move = parent_info[h]
            if parent_hash is None:
                break
            assert move is not None
            moves.append(move)
            h = parent_hash
        moves.reverse()

        board = shogi.Board()
        for move in moves:
            board.push_move(move)
        return board
