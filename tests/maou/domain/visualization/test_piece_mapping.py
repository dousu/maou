"""piece_mappingユーティリティのテスト．"""

from maou.domain.visualization.piece_mapping import (
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
        """マスインデックスを(row, col)座標に変換できる．"""
        # 左上 (0,0)
        assert square_index_to_coords(0) == (0, 0)

        # 右上 (0,8)
        assert square_index_to_coords(8) == (0, 8)

        # 左下 (8,0)
        assert square_index_to_coords(72) == (8, 0)

        # 右下 (8,8)
        assert square_index_to_coords(80) == (8, 8)

        # 中央 (4,4)
        assert square_index_to_coords(40) == (4, 4)

    def test_square_index_to_coords_all_squares(self) -> None:
        """全81マスのインデックス変換が正しい．"""
        for row in range(9):
            for col in range(9):
                index = row * 9 + col
                coords = square_index_to_coords(index)
                assert coords == (row, col)
