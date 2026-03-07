"""ゲームツリーデータモデルのテスト."""

from __future__ import annotations

import pytest

from maou.domain.game_tree.model import (
    GameTreeEdge,
    GameTreeNode,
)


class TestGameTreeNode:
    """GameTreeNode のテスト."""

    def test_create_node(self) -> None:
        """ノードを生成できる."""
        node = GameTreeNode(
            position_hash=12345,
            result_value=0.523,
            best_move_win_rate=0.531,
            num_branches=30,
            depth=0,
        )
        assert node.position_hash == 12345
        assert node.result_value == pytest.approx(0.523)
        assert node.best_move_win_rate == pytest.approx(0.531)
        assert node.num_branches == 30
        assert node.depth == 0

    def test_frozen(self) -> None:
        """frozen dataclass は変更不可."""
        node = GameTreeNode(
            position_hash=1,
            result_value=0.5,
            best_move_win_rate=0.5,
            num_branches=1,
            depth=0,
        )
        with pytest.raises(AttributeError):
            node.depth = 1  # type: ignore[misc]

    def test_equality(self) -> None:
        """同じフィールド値のノードは等価."""
        node1 = GameTreeNode(1, 0.5, 0.5, 1, 0)
        node2 = GameTreeNode(1, 0.5, 0.5, 1, 0)
        assert node1 == node2


class TestGameTreeEdge:
    """GameTreeEdge のテスト."""

    def test_create_edge(self) -> None:
        """エッジを生成できる."""
        edge = GameTreeEdge(
            parent_hash=100,
            child_hash=200,
            move16=7654,
            move_label=42,
            probability=0.452,
            win_rate=0.521,
        )
        assert edge.parent_hash == 100
        assert edge.child_hash == 200
        assert edge.move16 == 7654
        assert edge.move_label == 42
        assert edge.probability == pytest.approx(0.452)
        assert edge.win_rate == pytest.approx(0.521)

    def test_frozen(self) -> None:
        """frozen dataclass は変更不可."""
        edge = GameTreeEdge(
            parent_hash=1,
            child_hash=2,
            move16=100,
            move_label=0,
            probability=0.5,
            win_rate=0.5,
        )
        with pytest.raises(AttributeError):
            edge.probability = 0.9  # type: ignore[misc]
