import logging

import cshogi

from maou.app.learning import label

logger: logging.Logger = logging.getLogger("TEST")


class TestLabel:
    def test_make_move_label(self) -> None:
        """指し手用のラベル作成のテスト.
        指し手をいくつか指定して想定通りの値が返ってくるかテストする
        成りや打ちも少なくとも1種類テストしておく
        moveとmove16の違いは以下を実行してみると大体わかる
        parser = cshogi.CSA.Parser.parse_file(
            "tests/maou/app/converter/resources/test_dir/input/test_data_1.csa"
        )[0]
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
        - 1c3e+ (35角成想定) 16662
        - 1c7i+ (79角成想定) 16702
        - 9c7e+ (75角成想定) 25914
        - 9c3i+ (39角成想定) 25882
        - P*5b (52歩打想定) 10405
        - L*5b (52香打想定) 10533
        - N*5h (58桂打想定) 10667
        - S*5b (52銀打想定) 10789
        - G*5b (52金打想定) 11173
        - B*5b (52角打想定) 10917
        - R*5b (52飛打想定) 11045
        - 1a1b (12龍想定) 1
        - 7i9i+ (99飛成想定) 24400
        - 8f8g+ (87歩成想定) 25157
        - G*5b (52金打想定) 11173
        """
        # 1b1a (11と想定) 128
        test1b1a = label.make_move_label(cshogi.BLACK, 128)
        assert test1b1a < label.MOVE_LABELS_NUM
        assert test1b1a == 0
        # R*9i (99飛打想定) 11088
        testR_9i = label.make_move_label(cshogi.BLACK, 11088)
        assert testR_9i < label.MOVE_LABELS_NUM
        assert testR_9i == label.MOVE_LABELS_NUM - 1
        # 7g7f (76歩想定) 7739
        test7g7f = label.make_move_label(cshogi.BLACK, 7739)
        assert test7g7f < label.MOVE_LABELS_NUM
        assert test7g7f == label.MoveCategoryStartLabel.UP + 53
        # 2h7h (78飛想定) 7997
        test2h7h = label.make_move_label(cshogi.BLACK, 2109)
        assert test2h7h < label.MOVE_LABELS_NUM
        assert test2h7h == label.MoveCategoryStartLabel.LEFT + 61 - 9
        # 4e5c (53桂成らず想定) 4006
        test4e5c = label.make_move_label(cshogi.BLACK, 4006)
        assert test4e5c < label.MOVE_LABELS_NUM
        assert test4e5c == label.MoveCategoryStartLabel.KEIMA_LEFT + 38 - 10 - 8 - 5
        # 4e5c+ (53桂成想定) 20390
        test4e5c_ = label.make_move_label(cshogi.BLACK, 20390)
        assert test4e5c_ < label.MOVE_LABELS_NUM
        assert (
            test4e5c_ == label.MoveCategoryStartLabel.KEIMA_LEFT_PROMOTION + 38 - 24 - 3
        )
        # 5i5h (58玉想定) 5675
        test5i5h = label.make_move_label(cshogi.BLACK, 5675)
        assert test5i5h < label.MOVE_LABELS_NUM
        assert test5i5h == label.MoveCategoryStartLabel.UP + 43 - 4
        # 4i5h (58金右想定) 4523
        test4i5h = label.make_move_label(cshogi.BLACK, 4523)
        assert test4i5h < label.MOVE_LABELS_NUM
        assert test4i5h == label.MoveCategoryStartLabel.UP_LEFT + 43 - 4 - 8
        # 6i5h (58金左想定) 6827
        test6i5h = label.make_move_label(cshogi.BLACK, 6827)
        assert test6i5h < label.MOVE_LABELS_NUM
        assert test6i5h == label.MoveCategoryStartLabel.UP_RIGHT + 43 - 4
        # 8f8g (87金想定) 8773
        test8f8g = label.make_move_label(cshogi.BLACK, 8773)
        assert test8f8g < label.MOVE_LABELS_NUM
        assert test8f8g == label.MoveCategoryStartLabel.DOWN + 69 - 8
        # 1c3e+ (35角成想定) 16662
        test1c3e_ = label.make_move_label(cshogi.BLACK, 16662)
        assert test1c3e_ < label.MOVE_LABELS_NUM
        assert test1c3e_ == label.MoveCategoryStartLabel.DOWN_LEFT_PROMOTION + 6
        # 1c7i+ (79角成想定) 16702
        test1c7i_ = label.make_move_label(cshogi.BLACK, 16702)
        assert test1c7i_ < label.MOVE_LABELS_NUM
        assert test1c7i_ == label.MoveCategoryStartLabel.DOWN_LEFT_PROMOTION + 32
        # 9c7e+ (75角成想定) 25914
        test9c7e_ = label.make_move_label(cshogi.BLACK, 25914)
        assert test9c7e_ < label.MOVE_LABELS_NUM
        assert test9c7e_ == label.MoveCategoryStartLabel.DOWN_RIGHT_PROMOTION + 45
        # 9c3i+ (39角成想定) 25882
        test9c3i_ = label.make_move_label(cshogi.BLACK, 25882)
        assert test9c3i_ < label.MOVE_LABELS_NUM
        assert test9c3i_ == label.MoveCategoryStartLabel.DOWN_RIGHT_PROMOTION + 23
        # P*5b (52歩打想定) 10405
        testP_5b = label.make_move_label(cshogi.BLACK, 10405)
        assert testP_5b < label.MOVE_LABELS_NUM
        assert testP_5b == label.MoveCategoryStartLabel.FU + 37 - 5
        # L*5b (52香打想定) 10533
        testL_5b = label.make_move_label(cshogi.BLACK, 10533)
        assert testL_5b < label.MOVE_LABELS_NUM
        assert testL_5b == label.MoveCategoryStartLabel.KY + 37 - 5
        # N*5b (58桂打想定) 10667
        testN_5b = label.make_move_label(cshogi.BLACK, 10667)
        assert testN_5b < label.MOVE_LABELS_NUM
        assert testN_5b == label.MoveCategoryStartLabel.KE + 43 - 10
        # S*5b (52銀打想定) 10789
        testS_5b = label.make_move_label(cshogi.BLACK, 10789)
        assert testS_5b < label.MOVE_LABELS_NUM
        assert testS_5b == label.MoveCategoryStartLabel.GI + 37
        # G*5b (52金打想定) 11173
        testG_5b = label.make_move_label(cshogi.BLACK, 11173)
        assert testG_5b < label.MOVE_LABELS_NUM
        assert testG_5b == label.MoveCategoryStartLabel.KI + 37
        # B*5b (52角打想定) 10917
        testB_5b = label.make_move_label(cshogi.BLACK, 10917)
        assert testB_5b < label.MOVE_LABELS_NUM
        assert testB_5b == label.MoveCategoryStartLabel.KA + 37
        # R*5b (52飛打想定) 11045
        testR_5b = label.make_move_label(cshogi.BLACK, 11045)
        assert testR_5b < label.MOVE_LABELS_NUM
        assert testR_5b == label.MoveCategoryStartLabel.HI + 37
        # 1a1b (12龍想定) 1
        test1a1b = label.make_move_label(cshogi.BLACK, 1)
        assert test1a1b < label.MOVE_LABELS_NUM
        assert test1a1b == label.MoveCategoryStartLabel.DOWN + 2 - 2

        # 手番入れ替え確認
        # 1a1b (12香想定) 1
        test1a1bwhite = label.make_move_label(cshogi.WHITE, 1)
        assert test1a1bwhite < label.MOVE_LABELS_NUM
        assert test1a1bwhite == label.MoveCategoryStartLabel.UP + 79 - 8
        # 7i9i+ (99飛成想定) 24400
        test7i9i_ = label.make_move_label(cshogi.WHITE, 24400)
        assert test7i9i_ < label.MOVE_LABELS_NUM
        assert test7i9i_ == label.MoveCategoryStartLabel.RIGHT_PROMOTION + 0
        # 8f8g+ (87歩成想定) 25157
        test8f8g_ = label.make_move_label(cshogi.WHITE, 25157)
        assert test8f8g_ < label.MOVE_LABELS_NUM
        assert test8f8g_ == label.MoveCategoryStartLabel.UP_PROMOTION + 11 - 6
        # G*5b (52金打想定) 11173
        testG_5b = label.make_move_label(cshogi.WHITE, 11173)
        assert testG_5b < label.MOVE_LABELS_NUM
        assert testG_5b == label.MoveCategoryStartLabel.KI + 43

        # ラベル変換のロジックとMoveCategoryStartLabelの整合性取れているかテスト
        # 最初と最後をチェック
        # UP
        up_first = label.make_move_label(cshogi.BLACK, 128)
        assert up_first == label.MoveCategoryStartLabel.UP
        up_last = label.make_move_label(cshogi.BLACK, 10319)
        assert up_last == label.MoveCategoryStartLabel.UP_LEFT - 1
        # UP_LEFT
        up_left_first = label.make_move_label(cshogi.BLACK, 137)
        assert up_left_first == label.MoveCategoryStartLabel.UP_LEFT
        up_left_last = label.make_move_label(cshogi.BLACK, 9167)
        assert up_left_last == label.MoveCategoryStartLabel.UP_RIGHT - 1
        # UP_RIGHT
        up_right_first = label.make_move_label(cshogi.BLACK, 1280)
        assert up_right_first == label.MoveCategoryStartLabel.UP_RIGHT
        up_right_last = label.make_move_label(cshogi.BLACK, 10310)
        assert up_right_last == label.MoveCategoryStartLabel.LEFT - 1
        # LEFT
        left_first = label.make_move_label(cshogi.BLACK, 9)
        assert left_first == label.MoveCategoryStartLabel.LEFT
        left_last = label.make_move_label(cshogi.BLACK, 9168)
        assert left_last == label.MoveCategoryStartLabel.RIGHT - 1
        # RIGHT
        right_first = label.make_move_label(cshogi.BLACK, 1152)
        assert right_first == label.MoveCategoryStartLabel.RIGHT
        right_last = label.make_move_label(cshogi.BLACK, 10311)
        assert right_last == label.MoveCategoryStartLabel.DOWN - 1
        # DOWN
        down_first = label.make_move_label(cshogi.BLACK, 1)
        assert down_first == label.MoveCategoryStartLabel.DOWN
        down_last = label.make_move_label(cshogi.BLACK, 10192)
        assert down_last == label.MoveCategoryStartLabel.DOWN_LEFT - 1
        # DOWN_LEFT
        down_left_first = label.make_move_label(cshogi.BLACK, 10)
        assert down_left_first == label.MoveCategoryStartLabel.DOWN_LEFT
        down_left_last = label.make_move_label(cshogi.BLACK, 9040)
        assert down_left_last == label.MoveCategoryStartLabel.DOWN_RIGHT - 1
        # DOWN_RIGHT
        down_right_first = label.make_move_label(cshogi.BLACK, 1153)
        assert down_right_first == label.MoveCategoryStartLabel.DOWN_RIGHT
        down_right_last = label.make_move_label(cshogi.BLACK, 10183)
        assert down_right_last == label.MoveCategoryStartLabel.KEIMA_LEFT - 1
        # KEIMA_LEFT
        keima_left_first = label.make_move_label(cshogi.BLACK, 523)
        assert keima_left_first == label.MoveCategoryStartLabel.KEIMA_LEFT
        keima_left_last = label.make_move_label(cshogi.BLACK, 9166)
        assert keima_left_last == label.MoveCategoryStartLabel.KEIMA_RIGHT - 1
        # KEIMA_RIGHT
        keima_right_first = label.make_move_label(cshogi.BLACK, 1666)
        assert keima_right_first == label.MoveCategoryStartLabel.KEIMA_RIGHT
        keima_right_last = label.make_move_label(cshogi.BLACK, 10309)
        assert keima_right_last == label.MoveCategoryStartLabel.UP_PROMOTION - 1
        # UP_PROMOTION
        up_promotion_first = label.make_move_label(cshogi.BLACK, 16512)
        assert up_promotion_first == label.MoveCategoryStartLabel.UP_PROMOTION
        up_promotion_last = label.make_move_label(cshogi.BLACK, 26698)
        assert up_promotion_last == label.MoveCategoryStartLabel.UP_LEFT_PROMOTION - 1
        # UP_LEFT_PROMOTION
        up_left_promotion_first = label.make_move_label(cshogi.BLACK, 16521)
        assert up_left_promotion_first == label.MoveCategoryStartLabel.UP_LEFT_PROMOTION
        up_left_promotion_last = label.make_move_label(cshogi.BLACK, 21834)
        assert (
            up_left_promotion_last
            == label.MoveCategoryStartLabel.UP_RIGHT_PROMOTION - 1
        )
        # UP_RIGHT_PROMOTION
        up_right_promotion_first = label.make_move_label(cshogi.BLACK, 26624)
        assert (
            up_right_promotion_first == label.MoveCategoryStartLabel.UP_RIGHT_PROMOTION
        )
        up_right_promotion_last = label.make_move_label(cshogi.BLACK, 26049)
        assert (
            up_right_promotion_last == label.MoveCategoryStartLabel.LEFT_PROMOTION - 1
        )
        # LEFT_PROMOTION
        left_promotion_first = label.make_move_label(cshogi.BLACK, 16393)
        assert left_promotion_first == label.MoveCategoryStartLabel.LEFT_PROMOTION
        left_promotion_last = label.make_move_label(cshogi.BLACK, 16714)
        assert left_promotion_last == label.MoveCategoryStartLabel.RIGHT_PROMOTION - 1
        # RIGHT_PROMOTION
        right_promotion_first = label.make_move_label(cshogi.BLACK, 17536)
        assert right_promotion_first == label.MoveCategoryStartLabel.RIGHT_PROMOTION
        right_promotion_last = label.make_move_label(cshogi.BLACK, 25921)
        assert right_promotion_last == label.MoveCategoryStartLabel.DOWN_PROMOTION - 1
        # DOWN_PROMOTION
        down_promotion_first = label.make_move_label(cshogi.BLACK, 16385)
        assert down_promotion_first == label.MoveCategoryStartLabel.DOWN_PROMOTION
        down_promotion_last = label.make_move_label(cshogi.BLACK, 25936)
        assert (
            down_promotion_last == label.MoveCategoryStartLabel.DOWN_LEFT_PROMOTION - 1
        )
        # DOWN_LEFT_PROMOTION
        down_left_promotion_first = label.make_move_label(cshogi.BLACK, 16394)
        assert (
            down_left_promotion_first
            == label.MoveCategoryStartLabel.DOWN_LEFT_PROMOTION
        )
        down_left_promotion_last = label.make_move_label(cshogi.BLACK, 16464)
        assert (
            down_left_promotion_last
            == label.MoveCategoryStartLabel.DOWN_RIGHT_PROMOTION - 1
        )
        # DOWN_RIGHT_PROMOTION
        down_right_promotion_first = label.make_move_label(cshogi.BLACK, 17537)
        assert (
            down_right_promotion_first
            == label.MoveCategoryStartLabel.DOWN_RIGHT_PROMOTION
        )
        down_right_promotion_last = label.make_move_label(cshogi.BLACK, 25922)
        assert (
            down_right_promotion_last
            == label.MoveCategoryStartLabel.KEIMA_LEFT_PROMOTION - 1
        )
        # KEIMA_LEFT_PROMOTION
        keima_left_promotion_first = label.make_move_label(cshogi.BLACK, 16649)
        assert (
            keima_left_promotion_first
            == label.MoveCategoryStartLabel.KEIMA_LEFT_PROMOTION
        )
        keima_left_promotion_last = label.make_move_label(cshogi.BLACK, 25034)
        assert (
            keima_left_promotion_last
            == label.MoveCategoryStartLabel.KEIMA_RIGHT_PROMOTION - 1
        )
        # KEIMA_RIGHT_PROMOTION
        keima_right_promotion_first = label.make_move_label(cshogi.BLACK, 17792)
        assert (
            keima_right_promotion_first
            == label.MoveCategoryStartLabel.KEIMA_RIGHT_PROMOTION
        )
        keima_right_promotion_last = label.make_move_label(cshogi.BLACK, 26177)
        assert keima_right_promotion_last == label.MoveCategoryStartLabel.FU - 1
        # FU
        fu_first = label.make_move_label(cshogi.BLACK, 10369)
        assert fu_first == label.MoveCategoryStartLabel.FU
        fu_last = label.make_move_label(cshogi.BLACK, 10448)
        assert fu_last == label.MoveCategoryStartLabel.KY - 1
        # KY
        ky_first = label.make_move_label(cshogi.BLACK, 10497)
        assert ky_first == label.MoveCategoryStartLabel.KY
        ky_last = label.make_move_label(cshogi.BLACK, 10576)
        assert ky_last == label.MoveCategoryStartLabel.KE - 1
        # KE
        ke_first = label.make_move_label(cshogi.BLACK, 10626)
        assert ke_first == label.MoveCategoryStartLabel.KE
        ke_last = label.make_move_label(cshogi.BLACK, 10704)
        assert ke_last == label.MoveCategoryStartLabel.GI - 1
        # GI
        gi_first = label.make_move_label(cshogi.BLACK, 10752)
        assert gi_first == label.MoveCategoryStartLabel.GI
        gi_last = label.make_move_label(cshogi.BLACK, 10832)
        assert gi_last == label.MoveCategoryStartLabel.KI - 1
        # KI
        ki_first = label.make_move_label(cshogi.BLACK, 11136)
        assert ki_first == label.MoveCategoryStartLabel.KI
        ki_last = label.make_move_label(cshogi.BLACK, 11216)
        assert ki_last == label.MoveCategoryStartLabel.KA - 1
        # KA
        ka_first = label.make_move_label(cshogi.BLACK, 10880)
        assert ka_first == label.MoveCategoryStartLabel.KA
        ka_last = label.make_move_label(cshogi.BLACK, 10960)
        assert ka_last == label.MoveCategoryStartLabel.HI - 1
        # HI
        hi_first = label.make_move_label(cshogi.BLACK, 11008)
        assert hi_first == label.MoveCategoryStartLabel.HI
        hi_last = label.make_move_label(cshogi.BLACK, 11088)
        assert hi_last == label.MoveCategoryStartLabel.HI + 80

    def test_make_result_value(self) -> None:
        """勝率の教師データ作成のテスト.
        結果の種類を全種類用意してテストする
        手番変えるのは1種類だけテストしておく
        """
        assert label.make_result_value(cshogi.BLACK, cshogi.BLACK_WIN) == 1
        assert label.make_result_value(cshogi.BLACK, cshogi.DRAW) == 0.5
        assert label.make_result_value(cshogi.BLACK, cshogi.WHITE_WIN) == 0
        assert label.make_result_value(cshogi.WHITE, cshogi.BLACK_WIN) == 0
        assert label.make_result_value(cshogi.WHITE, cshogi.DRAW) == 0.5
        assert label.make_result_value(cshogi.WHITE, cshogi.WHITE_WIN) == 1
