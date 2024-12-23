import logging
import typing

import cshogi
import numpy as np
import pytest

from maou.app.learning import feature

logger: logging.Logger = logging.getLogger(__name__)


class TestTransform:
    @pytest.fixture
    def default_fixture(self) -> None:
        self.test_class = feature.Transform()

    def test_make_feature(
        self, default_fixture: typing.Annotated[None, pytest.fixture]
    ) -> None:
        """特徴量作成のテスト.
        以下の局面の特徴量を計算してみてチェックする
        '  9  8  7  6  5  4  3  2  1
        P1-KY *  *  *  *  * +KA * -KY
        P2 *  *  *  * +TO *  * +KI *
        P3 *  * -KE *  * -GI-KE *  *
        P4-FU * -FU-HI * -FU-OU-FU-FU
        P5 *  *  *  *  *  *  * +FU *
        P6+FU * +FU * +HI *  *  * +FU
        P7 * +FU+KE-TO * +FU+KE *  *
        P8 *  * +KI *  * -GI *  *  *
        P9+KY-TO * +OU *  *  *  * +KY
        P+00FU00FU00FU00FU
        P+00KI
        P-00GI00GI
        P-00KI
        P-00KA
        +
        """
        board = cshogi.Board()
        board.set_sfen(
            "l5B1l/4+P2G1/2n2sn2/p1pr1pkpp/7P1/P1P1R3P/1PN+p1PN2/2G2s3/L+p1K4L b 1G4P1b1g2s 99"
        )
        features = self.test_class._Transform__make_feature(board)
        logger.info(features.shape)
        # shapeの確認
        assert features.shape == (feature.FEATURES_NUM, 9, 9)
        # 0または1しか値がないことを確認
        assert np.all(np.logical_or(features == 0, features == 1))

        # ここからは値を具体的にテストする

        # BLACKの駒配置
        black_pos = np.stack(
            [
                # 歩
                np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    ]
                ),
                # 香
                np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    ]
                ),
                # 桂
                np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ),
                # 銀
                np.zeros((9, 9)),
                # 角
                np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ),
                # 飛
                np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ),
                # 金
                np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ),
                # 王
                np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ),
                # と金
                np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ),
                # 成香
                np.zeros((9, 9)),
                # 成桂
                np.zeros((9, 9)),
                # 成銀
                np.zeros((9, 9)),
                # 竜馬
                np.zeros((9, 9)),
                # 竜王
                np.zeros((9, 9)),
            ]
        )

        # WHITEの駒配置
        white_pos = np.stack(
            [
                # 歩
                np.array(
                    [
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ),
                # 香
                np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ),
                # 桂
                np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ),
                # 銀
                np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ),
                # 角
                np.zeros((9, 9)),
                # 飛
                np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ),
                # 金
                np.zeros((9, 9)),
                # 王
                np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ),
                # と金
                np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ),
                # 成香
                np.zeros((9, 9)),
                # 成桂
                np.zeros((9, 9)),
                # 成銀
                np.zeros((9, 9)),
                # 竜馬
                np.zeros((9, 9)),
                # 竜王
                np.zeros((9, 9)),
            ]
        )

        # BLACKの持ち駒
        black_pieces_in_hand = np.concatenate(
            [
                # 歩4枚
                np.ones((4, 9, 9)),
                np.zeros((26, 9, 9)),
                # 金1枚
                np.ones((1, 9, 9)),
                np.zeros((7, 9, 9)),
            ]
        )

        # WHITEの持ち駒
        white_pieces_in_hand = np.concatenate(
            [
                np.zeros((26, 9, 9)),
                # 銀2枚
                np.ones((2, 9, 9)),
                np.zeros((2, 9, 9)),
                # 金1枚
                np.ones((1, 9, 9)),
                np.zeros((3, 9, 9)),
                # 角1枚
                np.ones((1, 9, 9)),
                np.zeros((3, 9, 9)),
            ]
        )

        # 駒配置チェック
        assert np.array_equal(features[0:14], black_pos)
        assert np.array_equal(features[14:28], white_pos)

        # 持ち駒チェック
        assert np.array_equal(features[28:66], black_pieces_in_hand)
        assert np.array_equal(features[66:], white_pieces_in_hand)

        # 手番変えてチェック
        board.turn = cshogi.WHITE
        features_rotated = self.test_class._Transform__make_feature(board)

        # 駒配置チェック
        # 盤を回転させるので駒配置をsliceで上下左右逆転させる
        assert np.array_equal(features_rotated[0:14], white_pos[:, ::-1, ::-1])
        assert np.array_equal(features_rotated[14:28], black_pos[:, ::-1, ::-1])

        # 持ち駒チェック
        assert np.array_equal(features_rotated[28:66], white_pieces_in_hand)
        assert np.array_equal(features_rotated[66:], black_pieces_in_hand)

    def test_make_move_label(
        self, default_fixture: typing.Annotated[None, pytest.fixture]
    ) -> None:
        """指し手用のラベル作成のテスト.
        指し手をいくつか指定して想定通りの値が返ってくるかテストする
        成りや打ちも少なくとも1種類テストしておく
        moveとmove16の違いは以下を実行してみると大体わかる
        parser = cshogi.CSA.Parser.parse_file("tests/maou/app/converter/resources/test_dir/input/test_data_1.csa")[0]
        move16s = [move16(move) for move in parser.moves]
        for i, move in enumerate(parser.moves):
            print(i, format(move, '024b'))
            print(i, format(move16s[i], '024b'))

        テスト用のmove16はテストしたい指し手が指せる局面をsfenで作って以下のように出した
        board.set_sfen('sfen')
        for move in board.legal_moves:
            print(cshogi.move_to_usi(move), cshogi.move16(move))

        テスト項目
        - 1b1a (11と想定) 128
        - R*9i (99飛打想定) 11088
        - 7g7f (76歩想定) 7739
        - 2h7h (78飛想定) 2109
        - 4e5c (53桂成らず想定) 4006
        - 4e5c+ (53桂成想定) 20390
        - 5i5h (58玉想定) 5675
        - 4i5h (58金右想定) 4523
        - 6i5h (58金左想定) 6827
        - 8f8g (87金想定) 8773
        - P*5b (52歩打想定) 10405
        - L*5b (52香打想定) 10533
        - N*5b (52桂打想定) 10661
        - S*5b (52銀打想定) 10789
        - G*5b (52金打想定) 11173
        - B*5b (52角打想定) 10917
        - R*5b (52飛打想定) 11045
        - 1a1b (12龍想定) 1
        - 7i9i+ (99飛成想定) 24400
        - 8f8g+ (87歩成想定) 25157
        """
        # 1b1a (11と想定) 128
        test1b1a = self.test_class._Transform__make_move_label(cshogi.BLACK, 128)
        assert test1b1a < feature.MOVE_LABELS_NUM
        assert test1b1a == 0
        # R*9i (99飛打想定) 11088
        testR_9i = self.test_class._Transform__make_move_label(cshogi.BLACK, 11088)
        assert testR_9i < feature.MOVE_LABELS_NUM
        assert testR_9i == 2186  # feature.MOVE_LABELS_NUM - 1
        # 7g7f (76歩想定) 7739
        test7g7f = self.test_class._Transform__make_move_label(cshogi.BLACK, 7739)
        assert test7g7f < feature.MOVE_LABELS_NUM
        assert test7g7f == 59
        # 2h7h (78飛想定) 7997
        test2h7h = self.test_class._Transform__make_move_label(cshogi.BLACK, 2109)
        assert test2h7h < feature.MOVE_LABELS_NUM
        assert test2h7h == 304
        # 4e5c (53桂成らず想定) 4006
        test4e5c = self.test_class._Transform__make_move_label(cshogi.BLACK, 4006)
        assert test4e5c < feature.MOVE_LABELS_NUM
        assert test4e5c == 686
        # 4e5c+ (53桂成想定) 20390
        test4e5c_ = self.test_class._Transform__make_move_label(cshogi.BLACK, 20390)
        assert test4e5c_ < feature.MOVE_LABELS_NUM
        assert test4e5c_ == 1496
        # 5i5h (58玉想定) 5675
        test5i5h = self.test_class._Transform__make_move_label(cshogi.BLACK, 5675)
        assert test5i5h < feature.MOVE_LABELS_NUM
        assert test5i5h == 43
        # 4i5h (58金右想定) 4523
        test4i5h = self.test_class._Transform__make_move_label(cshogi.BLACK, 4523)
        assert test4i5h < feature.MOVE_LABELS_NUM
        assert test4i5h == 124
        # 6i5h (58金左想定) 6827
        test6i5h = self.test_class._Transform__make_move_label(cshogi.BLACK, 6827)
        assert test6i5h < feature.MOVE_LABELS_NUM
        assert test6i5h == 205
        # 8f8g (87金想定) 8773
        test8f8g = self.test_class._Transform__make_move_label(cshogi.BLACK, 8773)
        assert test8f8g < feature.MOVE_LABELS_NUM
        assert test8f8g == 474
        # P*5b (52歩打想定) 10405
        testP_5b = self.test_class._Transform__make_move_label(cshogi.BLACK, 10405)
        assert testP_5b < feature.MOVE_LABELS_NUM
        assert testP_5b == 20 * 81 + 37
        # L*5b (52香打想定) 10533
        testL_5b = self.test_class._Transform__make_move_label(cshogi.BLACK, 10533)
        assert testL_5b < feature.MOVE_LABELS_NUM
        assert testL_5b == 21 * 81 + 37
        # N*5b (52桂打想定) 10661
        testN_5b = self.test_class._Transform__make_move_label(cshogi.BLACK, 10661)
        assert testN_5b < feature.MOVE_LABELS_NUM
        assert testN_5b == 22 * 81 + 37
        # S*5b (52銀打想定) 10789
        testS_5b = self.test_class._Transform__make_move_label(cshogi.BLACK, 10789)
        assert testS_5b < feature.MOVE_LABELS_NUM
        assert testS_5b == 23 * 81 + 37
        # G*5b (52金打想定) 11173
        testG_5b = self.test_class._Transform__make_move_label(cshogi.BLACK, 11173)
        assert testG_5b < feature.MOVE_LABELS_NUM
        assert testG_5b == 24 * 81 + 37
        # B*5b (52角打想定) 10917
        testB_5b = self.test_class._Transform__make_move_label(cshogi.BLACK, 10917)
        assert testB_5b < feature.MOVE_LABELS_NUM
        assert testB_5b == 25 * 81 + 37
        # R*5b (52飛打想定) 11045
        testR_5b = self.test_class._Transform__make_move_label(cshogi.BLACK, 11045)
        assert testR_5b < feature.MOVE_LABELS_NUM
        assert testR_5b == 26 * 81 + 37
        # 1a1b (12龍想定) 1
        test1a1b = self.test_class._Transform__make_move_label(cshogi.BLACK, 1)
        assert test1a1b < feature.MOVE_LABELS_NUM
        assert test1a1b == 406
        # 7i9i+ (99飛成想定) 24400
        test7i9i_ = self.test_class._Transform__make_move_label(cshogi.BLACK, 24400)
        assert test7i9i_ < feature.MOVE_LABELS_NUM
        assert test7i9i_ == 1133

        # 手番入れ替え確認
        # 1a1b (12香想定) 1
        test1a1bwhite = self.test_class._Transform__make_move_label(cshogi.WHITE, 1)
        assert test1a1bwhite < feature.MOVE_LABELS_NUM
        assert test1a1bwhite == 79
        # 8f8g+ (87歩成想定) 25157
        test8f8g_ = self.test_class._Transform__make_move_label(cshogi.WHITE, 25157)
        assert test8f8g_ < feature.MOVE_LABELS_NUM
        assert test8f8g_ == 821
        # G*5b (52金打想定) 11173
        testG_5b = self.test_class._Transform__make_move_label(cshogi.WHITE, 11173)
        assert testG_5b < feature.MOVE_LABELS_NUM
        assert testG_5b == 24 * 81 + 43

    def test_make_result_value(
        self, default_fixture: typing.Annotated[None, pytest.fixture]
    ) -> None:
        """勝率の教師データ作成のテスト.
        結果の種類を全種類用意してテストする
        手番変えるのは1種類だけテストしておく
        """
        assert (
            self.test_class._Transform__make_result_value(
                cshogi.BLACK, cshogi.BLACK_WIN
            )
            == 1
        )
        assert (
            self.test_class._Transform__make_result_value(cshogi.BLACK, cshogi.DRAW)
            == 0.5
        )
        assert (
            self.test_class._Transform__make_result_value(
                cshogi.BLACK, cshogi.WHITE_WIN
            )
            == 0
        )
        assert (
            self.test_class._Transform__make_result_value(
                cshogi.WHITE, cshogi.BLACK_WIN
            )
            == 0
        )
        assert (
            self.test_class._Transform__make_result_value(cshogi.WHITE, cshogi.DRAW)
            == 0.5
        )
        assert (
            self.test_class._Transform__make_result_value(
                cshogi.WHITE, cshogi.WHITE_WIN
            )
            == 1
        )
