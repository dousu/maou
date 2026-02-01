"""piece_mappingユーティリティのテスト．

座標系はcshogiに準拠: square = col * 9 + row
- col: 筋（0=1筋，8=9筋）
- row: 段（0=1段目，8=9段目）

詳細は docs/visualization/shogi-conventions.md を参照．
"""

from maou.domain.visualization.piece_mapping import (
    coords_to_square_index,
    get_actual_piece_id,
    is_white_piece,
    square_index_to_coords,
)


class TestPieceMappingFunctions:
    """駒マッピング関数のテスト．"""

    def test_is_white_piece_black(self) -> None:
        """先手の駒（0-14）はFalseを返す．"""
        for piece_id in range(15):
            assert not is_white_piece(piece_id)

    def test_is_white_piece_white(self) -> None:
        """後手の駒（15-28）はTrueを返す．"""
        for piece_id in range(15, 29):
            assert is_white_piece(piece_id)

    def test_get_actual_piece_id_black(self) -> None:
        """先手の駒はそのままのIDを返す．"""
        assert get_actual_piece_id(1) == 1
        assert get_actual_piece_id(8) == 8
        assert get_actual_piece_id(14) == 14

    def test_get_actual_piece_id_white(self) -> None:
        """後手の駒は14を引いた値を返す．"""
        assert get_actual_piece_id(15) == 1  # 後手のFU
        assert get_actual_piece_id(22) == 8  # 後手のOU
        assert get_actual_piece_id(28) == 14  # 後手のRYU

    def test_square_index_to_coords(self) -> None:
        """マスインデックスを(row, col)座標に変換できる．

        cshogi座標系: square = col * 9 + row
        """
        # square=0: col=0, row=0 → 1筋1段
        assert square_index_to_coords(0) == (0, 0)

        # square=8: col=0, row=8 → 1筋9段
        assert square_index_to_coords(8) == (8, 0)

        # square=72: col=8, row=0 → 9筋1段
        assert square_index_to_coords(72) == (0, 8)

        # square=80: col=8, row=8 → 9筋9段
        assert square_index_to_coords(80) == (8, 8)

        # square=40: col=4, row=4 → 5筋5段（中央）
        assert square_index_to_coords(40) == (4, 4)

    def test_square_index_to_coords_all_squares(self) -> None:
        """全81マスのインデックス変換が正しい．

        cshogi座標系: square = col * 9 + row
        """
        for col in range(9):
            for row in range(9):
                index = col * 9 + row
                coords = square_index_to_coords(index)
                assert coords == (row, col)

    def test_coords_to_square_index(self) -> None:
        """(row, col)座標をマスインデックスに変換できる．

        cshogi座標系: square = col * 9 + row
        """
        # 1筋1段 → square=0
        assert coords_to_square_index(0, 0) == 0

        # 1筋9段 → square=8
        assert coords_to_square_index(8, 0) == 8

        # 9筋1段 → square=72
        assert coords_to_square_index(0, 8) == 72

        # 9筋9段 → square=80
        assert coords_to_square_index(8, 8) == 80

        # 5筋5段（中央） → square=40
        assert coords_to_square_index(4, 4) == 40

    def test_coords_and_square_index_roundtrip(self) -> None:
        """座標変換の往復が正しい．"""
        for square_idx in range(81):
            row, col = square_index_to_coords(square_idx)
            assert (
                coords_to_square_index(row, col) == square_idx
            )
