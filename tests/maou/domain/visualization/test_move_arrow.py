"""MoveArrowデータクラスのテスト．"""

import pytest

from maou.domain.visualization.board_renderer import MoveArrow


class TestMoveArrow:
    """MoveArrowデータクラスのテストケース．"""

    def test_normal_move_with_defaults(self) -> None:
        """通常の移動手: デフォルト値の確認．"""
        arrow = MoveArrow(from_square=76, to_square=77)

        assert arrow.from_square == 76
        assert arrow.to_square == 77
        assert arrow.is_drop is False
        assert arrow.drop_piece_type is None

    def test_drop_move(self) -> None:
        """駒打ち: 全フィールドの明示的指定．"""
        arrow = MoveArrow(
            from_square=None,
            to_square=40,
            is_drop=True,
            drop_piece_type=0,
        )

        assert arrow.from_square is None
        assert arrow.to_square == 40
        assert arrow.is_drop is True
        assert arrow.drop_piece_type == 0

    def test_frozen_immutability(self) -> None:
        """frozen=Trueによる不変性の確認．"""
        arrow = MoveArrow(from_square=76, to_square=77)

        with pytest.raises(AttributeError):
            arrow.to_square = 50  # type: ignore[misc]
