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
            progress_callback: プログレスコールバック(処理済み局面数, 発見済み局面数)

        Returns:
            (nodes, edges) のタプル

        Raises:
            ValueError: 初期局面がpreprocessデータに見つからない場合
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
                f"入力データに {duplicate_count} 件のハッシュ重複があります"
                f"(後勝ちで最後の行を使用)"
            )

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
        # parent_info: hash → (depth, parent_hash, move16)
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
                except (ValueError, RuntimeError) as e:
                    logger.warning(
                        f"USI {usi_move} のmove16変換に失敗"
                        f"(hash={current_hash}): {e}"
                    )
                    continue

                # 子局面のハッシュを取得
                board.push_move(move16)
                child_hash = board.hash()
                board.pop_move()

                # エッジ追加
                child_in_lookup = child_hash in lookup
                edges.append(
                    GameTreeEdge(
                        parent_hash=current_hash,
                        child_hash=child_hash,
                        move16=move16,
                        move_label=label_idx,
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
                        move16,
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
            parent_info: hash → (depth, parent_hash, move16)

        Returns:
            復元された盤面
        """
        # ルートまでのパスを逆順に構築
        moves: list[int] = []
        h = target_hash
        while True:
            _, parent_hash, move16 = parent_info[h]
            if parent_hash is None:
                break
            assert move16 is not None
            moves.append(move16)
            h = parent_hash
        moves.reverse()

        board = shogi.Board()
        for move in moves:
            board.push_move(move)
        return board
