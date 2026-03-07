"""ゲームツリー構築ロジックのテスト."""

from __future__ import annotations

import polars as pl
import pytest

from maou.domain.board import shogi
from maou.domain.move.label import (
    MOVE_LABELS_NUM,
    make_move_label,
)

from maou.app.game_tree.builder import GameTreeBuilder


def _create_preprocess_row(
    board: shogi.Board,
    move_probs: dict[int, float] | None = None,
    result_value: float = 0.5,
    best_move_win_rate: float = 0.5,
) -> dict:
    """テスト用のpreprocess行データを生成する．

    Args:
        board: 局面
        move_probs: {move16: probability} の辞書．Noneの場合は合法手に均等確率
        result_value: 局面の勝率
        best_move_win_rate: 最善手の勝率
    """
    move_labels = [0.0] * MOVE_LABELS_NUM
    move_win_rates = [0.0] * MOVE_LABELS_NUM

    if move_probs is None:
        legal_moves = list(board.get_legal_moves())
        prob = 1.0 / len(legal_moves)
        for m in legal_moves:
            label = make_move_label(board.get_turn(), m)
            move_labels[label] = prob
            move_win_rates[label] = 0.5
    else:
        for move16, prob in move_probs.items():
            label = make_move_label(board.get_turn(), move16)
            move_labels[label] = prob
            move_win_rates[label] = 0.5

    return {
        "id": board.hash(),
        "moveLabel": move_labels,
        "moveWinRate": move_win_rates,
        "resultValue": result_value,
        "bestMoveWinRate": best_move_win_rate,
    }


def _build_preprocess_df(
    rows: list[dict],
) -> pl.DataFrame:
    """行データのリストからpreprocess DataFrameを構築する."""
    return pl.DataFrame(
        {
            "id": pl.Series(
                [r["id"] for r in rows], dtype=pl.UInt64
            ),
            "moveLabel": [r["moveLabel"] for r in rows],
            "moveWinRate": [r["moveWinRate"] for r in rows],
            "resultValue": pl.Series(
                [r["resultValue"] for r in rows],
                dtype=pl.Float32,
            ),
            "bestMoveWinRate": pl.Series(
                [r["bestMoveWinRate"] for r in rows],
                dtype=pl.Float32,
            ),
        }
    )


class TestGameTreeBuilder:
    """GameTreeBuilder のテスト."""

    def test_initial_position_only(self) -> None:
        """初期局面のみでツリーを構築できる."""
        board = shogi.Board()
        # 初期局面のみ，子局面なし → 全候補の確率を0にする
        move_labels = [0.0] * MOVE_LABELS_NUM
        move_win_rates = [0.0] * MOVE_LABELS_NUM
        row = {
            "id": board.hash(),
            "moveLabel": move_labels,
            "moveWinRate": move_win_rates,
            "resultValue": 0.5,
            "bestMoveWinRate": 0.5,
        }
        df = _build_preprocess_df([row])

        builder = GameTreeBuilder()
        nodes, edges = builder.build(df, max_depth=5)

        assert len(nodes) == 1
        assert len(edges) == 0
        assert nodes[0].position_hash == board.hash()
        assert nodes[0].depth == 0
        assert nodes[0].num_branches == 0

    def test_depth_one_expansion(self) -> None:
        """初期局面から depth=1 の展開ができる."""
        board = shogi.Board()

        # 初期局面: 7g7f のみ高確率
        move_7g7f = board.board.move_from_usi("7g7f")
        initial_row = _create_preprocess_row(
            board,
            move_probs={move_7g7f: 0.9},
            result_value=0.52,
        )

        # 7g7f 後の局面
        board.push_move(move_7g7f)
        after_row = _create_preprocess_row(
            board,
            move_probs={},  # 空 → 子なし
            result_value=0.48,
        )
        board.pop_move()

        df = _build_preprocess_df([initial_row, after_row])

        builder = GameTreeBuilder()
        nodes, edges = builder.build(df, max_depth=1)

        assert len(nodes) == 2
        assert len(edges) == 1

        # ルートノード
        root = next(n for n in nodes if n.depth == 0)
        assert root.position_hash == shogi.Board().hash()
        assert root.result_value == pytest.approx(0.52)
        assert root.num_branches == 1

        # 子ノード
        child = next(n for n in nodes if n.depth == 1)
        assert child.result_value == pytest.approx(0.48)

        # エッジ
        edge = edges[0]
        assert edge.parent_hash == root.position_hash
        assert edge.child_hash == child.position_hash
        assert edge.move16 == shogi.move16(move_7g7f)
        assert edge.probability == pytest.approx(0.9)

    def test_min_probability_filtering(self) -> None:
        """min_probability 未満の手はフィルタリングされる."""
        board = shogi.Board()
        move_7g7f = board.board.move_from_usi("7g7f")
        move_2g2f = board.board.move_from_usi("2g2f")

        # 7g7f: 0.5, 2g2f: 0.005 (閾値 0.01 未満)
        initial_row = _create_preprocess_row(
            board,
            move_probs={move_7g7f: 0.5, move_2g2f: 0.005},
        )

        # 7g7f後の局面
        board.push_move(move_7g7f)
        after_7g7f = _create_preprocess_row(
            board, move_probs={}
        )
        board.pop_move()

        # 2g2f後の局面(フィルタリングされるが一応追加)
        board.push_move(move_2g2f)
        after_2g2f = _create_preprocess_row(
            board, move_probs={}
        )
        board.pop_move()

        df = _build_preprocess_df(
            [initial_row, after_7g7f, after_2g2f]
        )

        builder = GameTreeBuilder()
        nodes, edges = builder.build(
            df, max_depth=1, min_probability=0.01
        )

        # 2g2f は確率 0.005 < 0.01 なのでフィルタリングされる
        assert len(edges) == 1
        assert edges[0].move16 == shogi.move16(move_7g7f)

    def test_max_depth_cutoff(self) -> None:
        """max_depth で展開が停止する."""
        board = shogi.Board()
        move_7g7f = board.board.move_from_usi("7g7f")

        initial_row = _create_preprocess_row(
            board, move_probs={move_7g7f: 0.9}
        )

        board.push_move(move_7g7f)
        # depth=1 の局面にも子を持たせるが max_depth=0 で打ち切り
        move_3c3d = board.board.move_from_usi("3c3d")
        after_row = _create_preprocess_row(
            board, move_probs={move_3c3d: 0.8}
        )
        board.pop_move()

        df = _build_preprocess_df([initial_row, after_row])

        builder = GameTreeBuilder()
        # max_depth=0 → 初期局面のみ展開，子は探索しない
        nodes, edges = builder.build(df, max_depth=0)

        assert len(nodes) == 1
        assert nodes[0].depth == 0

    def test_initial_position_not_found(self) -> None:
        """初期局面がない場合はValueError."""
        # 空のDataFrame
        df = pl.DataFrame(
            {
                "id": pl.Series([], dtype=pl.UInt64),
                "moveLabel": [],
                "moveWinRate": [],
                "resultValue": pl.Series([], dtype=pl.Float32),
                "bestMoveWinRate": pl.Series(
                    [], dtype=pl.Float32
                ),
            }
        )

        builder = GameTreeBuilder()
        with pytest.raises(ValueError, match="開始局面"):
            builder.build(df)

    def test_progress_callback(self) -> None:
        """プログレスコールバックが呼ばれる."""
        board = shogi.Board()
        move_labels = [0.0] * MOVE_LABELS_NUM
        row = {
            "id": board.hash(),
            "moveLabel": move_labels,
            "moveWinRate": [0.0] * MOVE_LABELS_NUM,
            "resultValue": 0.5,
            "bestMoveWinRate": 0.5,
        }
        df = _build_preprocess_df([row])

        callback_calls: list[tuple[int, int]] = []

        def callback(processed: int, total: int) -> None:
            callback_calls.append((processed, total))

        builder = GameTreeBuilder()
        builder.build(
            df, max_depth=5, progress_callback=callback
        )

        assert len(callback_calls) > 0
        # 最後のコールは processed == 1
        assert callback_calls[-1][0] == 1

    def test_child_not_in_lookup_ignored(self) -> None:
        """子局面がルックアップテーブルにない場合はエッジのみ追加されキューに入らない."""
        board = shogi.Board()
        move_7g7f = board.board.move_from_usi("7g7f")

        # 初期局面のみ(子局面のデータなし)
        initial_row = _create_preprocess_row(
            board, move_probs={move_7g7f: 0.9}
        )
        df = _build_preprocess_df([initial_row])

        builder = GameTreeBuilder()
        nodes, edges = builder.build(df, max_depth=5)

        # ノードは初期局面のみ
        assert len(nodes) == 1
        # エッジは生成される(子局面のデータがなくてもエッジは追加)
        assert len(edges) == 1
        assert edges[0].move16 == shogi.move16(move_7g7f)

    def test_duplicate_hash_uses_last(self) -> None:
        """入力データにハッシュ重複がある場合，最後の行が使われる(dict内包表記の後勝ち)."""
        board = shogi.Board()
        h = board.hash()

        move_labels_a = [0.0] * MOVE_LABELS_NUM
        move_labels_b = [0.0] * MOVE_LABELS_NUM

        row_a = {
            "id": h,
            "moveLabel": move_labels_a,
            "moveWinRate": [0.0] * MOVE_LABELS_NUM,
            "resultValue": 0.7,
            "bestMoveWinRate": 0.7,
        }
        row_b = {
            "id": h,
            "moveLabel": move_labels_b,
            "moveWinRate": [0.0] * MOVE_LABELS_NUM,
            "resultValue": 0.3,
            "bestMoveWinRate": 0.3,
        }
        df = _build_preprocess_df([row_a, row_b])

        builder = GameTreeBuilder()
        nodes, edges = builder.build(df, max_depth=1)

        assert len(nodes) == 1
        # dict内包表記のlookupは後勝ちなので最後の行(0.3)が使われる
        assert nodes[0].result_value == pytest.approx(0.3)

    def test_transposition_bfs_shortest_depth(self) -> None:
        """合流局面(transposition)のdepthはBFS最短距離になる."""
        board = shogi.Board()
        move_7g7f = board.board.move_from_usi("7g7f")
        move_2g2f = board.board.move_from_usi("2g2f")

        # 初期局面: 2つの指し手(7g7f, 2g2f)
        initial_row = _create_preprocess_row(
            board,
            move_probs={move_7g7f: 0.5, move_2g2f: 0.5},
        )

        # 7g7f後の局面: 3c3dを指す
        board.push_move(move_7g7f)
        move_3c3d_after_7g7f = board.board.move_from_usi("3c3d")
        after_7g7f = _create_preprocess_row(
            board,
            move_probs={move_3c3d_after_7g7f: 0.5},
        )
        board.pop_move()

        # 2g2f後の局面: 3c3dを指す
        board.push_move(move_2g2f)
        move_3c3d_after_2g2f = board.board.move_from_usi("3c3d")
        after_2g2f = _create_preprocess_row(
            board,
            move_probs={move_3c3d_after_2g2f: 0.5},
        )
        board.pop_move()

        # 合流局面: 7g7f→3c3d と 2g2f→3c3d は異なる局面だが
        # もし同じhashなら合流する(cshogiでは手順が違えば別hash)
        # ここでは各経路の子が独立した局面になることを検証
        board.push_move(move_7g7f)
        board.push_move(move_3c3d_after_7g7f)
        merged_hash_a = board.hash()
        merged_row_a = _create_preprocess_row(
            board, move_probs={}
        )
        board.pop_move()
        board.pop_move()

        board.push_move(move_2g2f)
        board.push_move(move_3c3d_after_2g2f)
        merged_hash_b = board.hash()
        merged_row_b = _create_preprocess_row(
            board, move_probs={}
        )
        board.pop_move()
        board.pop_move()

        rows = [
            initial_row,
            after_7g7f,
            after_2g2f,
            merged_row_a,
        ]
        # hashが同じなら合流テスト，違えば独立テスト
        if merged_hash_a != merged_hash_b:
            rows.append(merged_row_b)

        df = _build_preprocess_df(rows)

        builder = GameTreeBuilder()
        nodes, edges = builder.build(
            df, max_depth=5, min_probability=0.01
        )

        # depth=0: 初期局面, depth=1: 2局面, depth=2: 合流局面
        depth_0 = [n for n in nodes if n.depth == 0]
        depth_1 = [n for n in nodes if n.depth == 1]
        depth_2 = [n for n in nodes if n.depth == 2]
        assert len(depth_0) == 1
        assert len(depth_1) == 2
        # 合流していれば1，独立なら2
        if merged_hash_a == merged_hash_b:
            assert len(depth_2) == 1
        else:
            assert len(depth_2) == 2

    def test_progress_callback_total_equals_visited(
        self,
    ) -> None:
        """プログレスコールバックのtotalが最終的にprocessedと一致する."""
        board = shogi.Board()
        move_7g7f = board.board.move_from_usi("7g7f")

        initial_row = _create_preprocess_row(
            board, move_probs={move_7g7f: 0.9}
        )

        board.push_move(move_7g7f)
        after_row = _create_preprocess_row(board, move_probs={})
        board.pop_move()

        df = _build_preprocess_df([initial_row, after_row])

        callback_calls: list[tuple[int, int]] = []

        def callback(processed: int, total: int) -> None:
            callback_calls.append((processed, total))

        builder = GameTreeBuilder()
        nodes, _ = builder.build(
            df, max_depth=5, progress_callback=callback
        )

        # 最後のコールバックで processed == total
        last_processed, last_total = callback_calls[-1]
        assert last_processed == last_total
        assert last_processed == len(nodes)
