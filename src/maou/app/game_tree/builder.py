"""ゲームツリー構築ロジック(BFS)."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable
from typing import TYPE_CHECKING

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
    ) -> tuple[list[GameTreeNode], list[GameTreeEdge]]:
        """BFSでツリーを構築する．

        Args:
            preprocess_df: preprocessデータのDataFrame
                (カラム: id, moveLabel, moveWinRate, resultValue, bestMoveWinRate)
            max_depth: 最大探索深さ
            min_probability: 指し手の最小確率閾値
            progress_callback: プログレスコールバック(処理済み局面数, 全局面数)

        Returns:
            (nodes, edges) のタプル

        Raises:
            ValueError: 初期局面がpreprocessデータに見つからない場合
        """
        # 1. ルックアップテーブル構築: id → 行インデックス
        id_list = preprocess_df["id"].to_list()
        lookup: dict[int, int] = {
            hash_val: idx
            for idx, hash_val in enumerate(id_list)
        }
        total_positions = len(lookup)

        # 2. 初期局面のZobrist hashを取得
        board = shogi.Board()
        initial_hash = board.hash()

        if initial_hash not in lookup:
            raise ValueError(
                f"初期局面(hash={initial_hash})がpreprocessデータに見つかりません．"
            )

        # moveLabel, moveWinRate を事前にリストとして取得(ランダムアクセス用)
        move_label_list = preprocess_df["moveLabel"].to_list()
        move_win_rate_list = preprocess_df[
            "moveWinRate"
        ].to_list()
        result_value_list = preprocess_df[
            "resultValue"
        ].to_list()
        best_move_win_rate_list = preprocess_df[
            "bestMoveWinRate"
        ].to_list()

        # 3. BFS
        nodes: list[GameTreeNode] = []
        edges: list[GameTreeEdge] = []
        # visited: hash → depth(最短距離を管理)
        visited: dict[int, int] = {initial_hash: 0}
        # キュー: (zobrist_hash, move_path)
        queue: deque[tuple[int, list[int]]] = deque(
            [(initial_hash, [])]
        )
        processed = 0

        while queue:
            current_hash, move_path = queue.popleft()
            current_depth = visited[current_hash]
            row_idx = lookup[current_hash]

            # ノード情報の取得
            move_labels = move_label_list[row_idx]
            move_win_rates = move_win_rate_list[row_idx]
            result_value = float(result_value_list[row_idx])
            best_move_win_rate = float(
                best_move_win_rate_list[row_idx]
            )

            # min_probability以上の指し手を取得
            candidate_indices = [
                i
                for i in range(len(move_labels))
                if move_labels[i] >= min_probability
            ]

            # ノードを追加
            nodes.append(
                GameTreeNode(
                    position_hash=current_hash,
                    result_value=result_value,
                    best_move_win_rate=best_move_win_rate,
                    num_branches=len(candidate_indices),
                    depth=current_depth,
                )
            )

            # max_depth に達したら展開しない
            if current_depth >= max_depth:
                processed += 1
                if progress_callback:
                    progress_callback(
                        processed, total_positions
                    )
                continue

            # 盤面を復元
            board = shogi.Board()
            for move in move_path:
                board.push_move(move)

            # 各候補手を処理
            for label_idx in candidate_indices:
                probability = float(move_labels[label_idx])
                win_rate = float(move_win_rates[label_idx])

                # ラベルからUSI指し手に変換
                try:
                    usi_move = make_usi_move_from_label(
                        board, label_idx
                    )
                except ValueError:
                    logger.debug(
                        f"ラベル {label_idx} の変換に失敗(hash={current_hash})"
                    )
                    continue

                # USIからmove16に変換
                try:
                    move16 = board.board.move_from_usi(usi_move)
                except Exception:
                    logger.debug(
                        f"USI {usi_move} のmove16変換に失敗(hash={current_hash})"
                    )
                    continue

                # 子局面のハッシュを取得
                board.push_move(move16)
                child_hash = board.hash()
                board.board.pop()

                # エッジ追加
                edges.append(
                    GameTreeEdge(
                        parent_hash=current_hash,
                        child_hash=child_hash,
                        move16=move16,
                        move_label=label_idx,
                        probability=probability,
                        win_rate=win_rate,
                    )
                )

                # 子局面がルックアップテーブルにあり，未訪問または
                # より短い経路で到達した場合はキューに追加
                if child_hash in lookup:
                    if child_hash not in visited:
                        visited[child_hash] = current_depth + 1
                        new_path = move_path + [move16]
                        queue.append((child_hash, new_path))
                    elif (
                        current_depth + 1 < visited[child_hash]
                    ):
                        # より短い経路が見つかった場合，depthを更新
                        visited[child_hash] = current_depth + 1

            processed += 1
            if progress_callback:
                progress_callback(processed, total_positions)

        return nodes, edges
