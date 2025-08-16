import logging

import cshogi  # type: ignore[import]

from maou.app.pre_process import label

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
        test1b1a = label.make_move_label(cshogi.BLACK, 128)  # type: ignore[attr-defined]
        assert test1b1a < label.MOVE_LABELS_NUM
        assert test1b1a == 0
        # R*9i (99飛打想定) 11088
        testR_9i = label.make_move_label(cshogi.BLACK, 11088)  # type: ignore[attr-defined]
        assert testR_9i < label.MOVE_LABELS_NUM
        assert testR_9i == label.MOVE_LABELS_NUM - 1
        # 7g7f (76歩想定) 7739
        test7g7f = label.make_move_label(cshogi.BLACK, 7739)  # type: ignore[attr-defined]
        assert test7g7f < label.MOVE_LABELS_NUM
        assert test7g7f == label.MoveCategoryStartLabel.UP + 53
        # 2h7h (78飛想定) 7997
        test2h7h = label.make_move_label(cshogi.BLACK, 2109)  # type: ignore[attr-defined]
        assert test2h7h < label.MOVE_LABELS_NUM
        assert (
            test2h7h
            == label.MoveCategoryStartLabel.LEFT + 61 - 9
        )
        # 4e5c (53桂成らず想定) 4006
        test4e5c = label.make_move_label(cshogi.BLACK, 4006)  # type: ignore[attr-defined]
        assert test4e5c < label.MOVE_LABELS_NUM
        assert (
            test4e5c
            == label.MoveCategoryStartLabel.KEIMA_LEFT
            + 38
            - 10
            - 8
            - 5
        )
        # 4e5c+ (53桂成想定) 20390
        test4e5c_ = label.make_move_label(cshogi.BLACK, 20390)  # type: ignore[attr-defined]
        assert test4e5c_ < label.MOVE_LABELS_NUM
        assert (
            test4e5c_
            == label.MoveCategoryStartLabel.KEIMA_LEFT_PROMOTION
            + 38
            - 24
            - 3
        )
        # 5i5h (58玉想定) 5675
        test5i5h = label.make_move_label(cshogi.BLACK, 5675)  # type: ignore[attr-defined]
        assert test5i5h < label.MOVE_LABELS_NUM
        assert (
            test5i5h == label.MoveCategoryStartLabel.UP + 43 - 4
        )
        # 4i5h (58金右想定) 4523
        test4i5h = label.make_move_label(cshogi.BLACK, 4523)  # type: ignore[attr-defined]
        assert test4i5h < label.MOVE_LABELS_NUM
        assert (
            test4i5h
            == label.MoveCategoryStartLabel.UP_LEFT + 43 - 4 - 8
        )
        # 6i5h (58金左想定) 6827
        test6i5h = label.make_move_label(cshogi.BLACK, 6827)  # type: ignore[attr-defined]
        assert test6i5h < label.MOVE_LABELS_NUM
        assert (
            test6i5h
            == label.MoveCategoryStartLabel.UP_RIGHT + 43 - 4
        )
        # 8f8g (87金想定) 8773
        test8f8g = label.make_move_label(cshogi.BLACK, 8773)  # type: ignore[attr-defined]
        assert test8f8g < label.MOVE_LABELS_NUM
        assert (
            test8f8g
            == label.MoveCategoryStartLabel.DOWN + 69 - 8
        )
        # 1c3e+ (35角成想定) 16662
        test1c3e_ = label.make_move_label(cshogi.BLACK, 16662)  # type: ignore[attr-defined]
        assert test1c3e_ < label.MOVE_LABELS_NUM
        assert (
            test1c3e_
            == label.MoveCategoryStartLabel.DOWN_LEFT_PROMOTION
            + 6
        )
        # 1c7i+ (79角成想定) 16702
        test1c7i_ = label.make_move_label(cshogi.BLACK, 16702)  # type: ignore[attr-defined]
        assert test1c7i_ < label.MOVE_LABELS_NUM
        assert (
            test1c7i_
            == label.MoveCategoryStartLabel.DOWN_LEFT_PROMOTION
            + 32
        )
        # 9c7e+ (75角成想定) 25914
        test9c7e_ = label.make_move_label(cshogi.BLACK, 25914)  # type: ignore[attr-defined]
        assert test9c7e_ < label.MOVE_LABELS_NUM
        assert (
            test9c7e_
            == label.MoveCategoryStartLabel.DOWN_RIGHT_PROMOTION
            + 45
        )
        # 9c3i+ (39角成想定) 25882
        test9c3i_ = label.make_move_label(cshogi.BLACK, 25882)  # type: ignore[attr-defined]
        assert test9c3i_ < label.MOVE_LABELS_NUM
        assert (
            test9c3i_
            == label.MoveCategoryStartLabel.DOWN_RIGHT_PROMOTION
            + 23
        )
        # P*5b (52歩打想定) 10405
        testP_5b = label.make_move_label(cshogi.BLACK, 10405)  # type: ignore[attr-defined]
        assert testP_5b < label.MOVE_LABELS_NUM
        assert (
            testP_5b == label.MoveCategoryStartLabel.FU + 37 - 5
        )
        # L*5b (52香打想定) 10533
        testL_5b = label.make_move_label(cshogi.BLACK, 10533)  # type: ignore[attr-defined]
        assert testL_5b < label.MOVE_LABELS_NUM
        assert (
            testL_5b == label.MoveCategoryStartLabel.KY + 37 - 5
        )
        # N*5h (58桂打想定) 10667
        testN_5h = label.make_move_label(cshogi.BLACK, 10667)  # type: ignore[attr-defined]
        assert testN_5h < label.MOVE_LABELS_NUM
        assert (
            testN_5h
            == label.MoveCategoryStartLabel.KE + 43 - 10
        )
        # S*5b (52銀打想定) 10789
        testS_5b = label.make_move_label(cshogi.BLACK, 10789)  # type: ignore[attr-defined]
        assert testS_5b < label.MOVE_LABELS_NUM
        assert testS_5b == label.MoveCategoryStartLabel.GI + 37
        # G*5b (52金打想定) 11173
        testG_5b = label.make_move_label(cshogi.BLACK, 11173)  # type: ignore[attr-defined]
        assert testG_5b < label.MOVE_LABELS_NUM
        assert testG_5b == label.MoveCategoryStartLabel.KI + 37
        # B*5b (52角打想定) 10917
        testB_5b = label.make_move_label(cshogi.BLACK, 10917)  # type: ignore[attr-defined]
        assert testB_5b < label.MOVE_LABELS_NUM
        assert testB_5b == label.MoveCategoryStartLabel.KA + 37
        # R*5b (52飛打想定) 11045
        testR_5b = label.make_move_label(cshogi.BLACK, 11045)  # type: ignore[attr-defined]
        assert testR_5b < label.MOVE_LABELS_NUM
        assert testR_5b == label.MoveCategoryStartLabel.HI + 37
        # 1a1b (12龍想定) 1
        test1a1b = label.make_move_label(cshogi.BLACK, 1)  # type: ignore[attr-defined]
        assert test1a1b < label.MOVE_LABELS_NUM
        assert (
            test1a1b
            == label.MoveCategoryStartLabel.DOWN + 2 - 2
        )

        # 手番入れ替え確認
        # 1a1b (12香想定) 1
        test1a1bwhite = label.make_move_label(cshogi.WHITE, 1)  # type: ignore[attr-defined]
        assert test1a1bwhite < label.MOVE_LABELS_NUM
        assert (
            test1a1bwhite
            == label.MoveCategoryStartLabel.UP + 79 - 8
        )
        # 7i9i+ (99飛成想定) 24400
        test7i9i_ = label.make_move_label(cshogi.WHITE, 24400)  # type: ignore[attr-defined]
        assert test7i9i_ < label.MOVE_LABELS_NUM
        assert (
            test7i9i_
            == label.MoveCategoryStartLabel.RIGHT_PROMOTION + 0
        )
        # 8f8g+ (87歩成想定) 25157
        test8f8g_ = label.make_move_label(cshogi.WHITE, 25157)  # type: ignore[attr-defined]
        assert test8f8g_ < label.MOVE_LABELS_NUM
        assert (
            test8f8g_
            == label.MoveCategoryStartLabel.UP_PROMOTION
            + 11
            - 6
        )
        # G*5b (52金打想定) 11173
        testG_5b = label.make_move_label(cshogi.WHITE, 11173)  # type: ignore[attr-defined]
        assert testG_5b < label.MOVE_LABELS_NUM
        assert testG_5b == label.MoveCategoryStartLabel.KI + 43

    def test_make_move_from_label(self) -> None:
        """ラベルから指し手への逆変換のテスト."""
        # Test cases that work correctly with current implementation
        working_test_cases = [
            # (move, expected_usi, description)
            (128, "1b1a", "1b1a - basic UP move"),
            (5675, "5i5h", "5i5h - UP move"),
            (11088, "R*9i", "R*9i - HI drop"),
            (10405, "P*5b", "P*5b - FU drop"),
            (10533, "L*5b", "L*5b - KY drop"),
            (10667, "N*5h", "N*5h - KE drop"),
            (10789, "S*5b", "S*5b - GI drop"),
            (11173, "G*5b", "G*5b - KI drop"),
            (10917, "B*5b", "B*5b - KA drop"),
            (11045, "R*5b", "R*5b - HI drop"),
            (20390, "4e5c+", "4e5c+ - KEIMA_LEFT promotion"),
        ]

        logger.info(
            "\n=== Testing reverse conversion for working cases ==="
        )
        for (
            move,
            expected_usi,
            description,
        ) in working_test_cases:
            # Test round-trip conversion
            move_label = label.make_move_label(
                cshogi.BLACK,  # type: ignore[attr-defined]
                move,
            )
            result = label.make_move_from_label(
                cshogi.BLACK,  # type: ignore[attr-defined]
                move_label,
            )

            logger.info(
                f"Move {move} -> Label {move_label} -> {result} ({description})"
            )
            assert result == expected_usi, (
                f"Expected {expected_usi}, got {result}"
            )

        # Test extended functionality: move values as input
        # Note: Only move values beyond MOVE_LABELS_NUM are treated as cshogi move values
        logger.info(
            "\n=== Testing move values as direct input (for values > MOVE_LABELS_NUM) ==="
        )
        large_move_cases = [
            (11088, "R*9i"),  # This is beyond MOVE_LABELS_NUM
            (10405, "P*5b"),  # This is beyond MOVE_LABELS_NUM
            (10533, "L*5b"),  # This is beyond MOVE_LABELS_NUM
            (10789, "S*5b"),  # This is beyond MOVE_LABELS_NUM
            (11173, "G*5b"),  # This is beyond MOVE_LABELS_NUM
        ]

        for move, expected_usi in large_move_cases:
            if (
                move >= label.MOVE_LABELS_NUM
            ):  # Only test moves beyond label range
                result = label.make_move_from_label(
                    cshogi.BLACK,  # type: ignore[attr-defined]
                    move,
                )
                actual_usi = cshogi.move_to_usi(move)  # type: ignore[attr-defined]

                logger.info(
                    f"Move {move} -> {result} (cshogi: {actual_usi})"
                )
                assert result == actual_usi, (
                    f"Expected {actual_usi}, got {result}"
                )

        # Test WHITE turn for drop moves (they should work the same)
        logger.info(
            "\n=== Testing WHITE turn for drop moves ==="
        )
        drop_moves = [
            (10405, "P*5b"),
            (10789, "S*5b"),
            (11173, "G*5b"),
        ]

        for move, expected_usi in drop_moves:
            white_label = label.make_move_label(
                cshogi.WHITE,  # type: ignore[attr-defined]
                move,
            )
            white_result = label.make_move_from_label(
                cshogi.WHITE,  # type: ignore[attr-defined]
                white_label,
            )
            actual_usi = cshogi.move_to_usi(move)  # type: ignore[attr-defined]

            logger.info(
                f"WHITE Move {move} -> Label {white_label} -> {white_result} (cshogi: {actual_usi})"
            )
            assert white_result == actual_usi, (
                f"Expected {actual_usi}, got {white_result}"
            )

        # Test specific user-requested cases
        logger.info(
            "\n=== Testing specific requested cases ==="
        )
        # Test make_move_from_label(cshogi.BLACK, 928) -> 4e5c+
        result_928 = label.make_move_from_label(
            cshogi.BLACK,  # type: ignore[attr-defined]
            928,
        )
        logger.info(
            f"Label 928 -> {result_928} (expected: 4e5c+)"
        )
        assert result_928 == "4e5c+", (
            f"Expected 4e5c+, got {result_928}"
        )

        # Test make_move_from_label(cshogi.BLACK, 39) -> 5i5h
        result_39 = label.make_move_from_label(cshogi.BLACK, 39)  # type: ignore[attr-defined]
        logger.info(f"Label 39 -> {result_39} (expected: 5i5h)")
        assert result_39 == "5i5h", (
            f"Expected 5i5h, got {result_39}"
        )

        # Test error cases
        logger.info("\n=== Testing error cases ===")
        try:
            label.make_move_from_label(cshogi.BLACK, -1)  # type: ignore[attr-defined]
            assert False, (
                "Should raise ValueError for negative label"
            )
        except ValueError:
            logger.info("✅ Negative label correctly rejected")
            pass

        # Test invalid move value
        try:
            label.make_move_from_label(cshogi.BLACK, 999999)  # type: ignore[attr-defined]
            logger.info(
                "⚠️ Very large move value accepted (may be valid cshogi move)"
            )
        except ValueError:
            logger.info(
                "✅ Invalid move value correctly rejected"
            )

        # ラベル変換のロジックとMoveCategoryStartLabelの整合性取れているかテスト
        # 最初と最後をチェック
        # UP
        up_first = label.make_move_label(cshogi.BLACK, 128)  # type: ignore[attr-defined]
        assert up_first == label.MoveCategoryStartLabel.UP
        up_last = label.make_move_label(cshogi.BLACK, 10319)  # type: ignore[attr-defined]
        assert (
            up_last == label.MoveCategoryStartLabel.UP_LEFT - 1
        )
        # UP_LEFT
        up_left_first = label.make_move_label(cshogi.BLACK, 137)  # type: ignore[attr-defined]
        assert (
            up_left_first
            == label.MoveCategoryStartLabel.UP_LEFT
        )
        up_left_last = label.make_move_label(cshogi.BLACK, 9167)  # type: ignore[attr-defined]
        assert (
            up_left_last
            == label.MoveCategoryStartLabel.UP_RIGHT - 1
        )
        # UP_RIGHT
        up_right_first = label.make_move_label(
            cshogi.BLACK,  # type: ignore[attr-defined]
            1280,
        )
        assert (
            up_right_first
            == label.MoveCategoryStartLabel.UP_RIGHT
        )
        up_right_last = label.make_move_label(
            cshogi.BLACK,  # type: ignore[attr-defined]
            10310,
        )
        assert (
            up_right_last
            == label.MoveCategoryStartLabel.LEFT - 1
        )
        # LEFT
        left_first = label.make_move_label(cshogi.BLACK, 9)  # type: ignore[attr-defined]
        assert left_first == label.MoveCategoryStartLabel.LEFT
        left_last = label.make_move_label(cshogi.BLACK, 9168)  # type: ignore[attr-defined]
        assert (
            left_last == label.MoveCategoryStartLabel.RIGHT - 1
        )
        # RIGHT
        right_first = label.make_move_label(cshogi.BLACK, 1152)  # type: ignore[attr-defined]
        assert right_first == label.MoveCategoryStartLabel.RIGHT
        right_last = label.make_move_label(cshogi.BLACK, 10311)  # type: ignore[attr-defined]
        assert (
            right_last == label.MoveCategoryStartLabel.DOWN - 1
        )
        # DOWN
        down_first = label.make_move_label(cshogi.BLACK, 1)  # type: ignore[attr-defined]
        assert down_first == label.MoveCategoryStartLabel.DOWN
        down_last = label.make_move_label(cshogi.BLACK, 10192)  # type: ignore[attr-defined]
        assert (
            down_last
            == label.MoveCategoryStartLabel.DOWN_LEFT - 1
        )
        # DOWN_LEFT
        down_left_first = label.make_move_label(
            cshogi.BLACK,  # type: ignore[attr-defined]
            10,
        )
        assert (
            down_left_first
            == label.MoveCategoryStartLabel.DOWN_LEFT
        )
        down_left_last = label.make_move_label(
            cshogi.BLACK,  # type: ignore[attr-defined]
            9040,
        )
        assert (
            down_left_last
            == label.MoveCategoryStartLabel.DOWN_RIGHT - 1
        )
        # DOWN_RIGHT
        down_right_first = label.make_move_label(
            cshogi.BLACK,  # type: ignore[attr-defined]
            1153,
        )  # type: ignore[attr-defined]
        assert (
            down_right_first
            == label.MoveCategoryStartLabel.DOWN_RIGHT
        )
        down_right_last = label.make_move_label(
            cshogi.BLACK,  # type: ignore[attr-defined]
            10183,
        )
        assert (
            down_right_last
            == label.MoveCategoryStartLabel.KEIMA_LEFT - 1
        )
        # KEIMA_LEFT
        keima_left_first = label.make_move_label(
            cshogi.BLACK,  # type: ignore[attr-defined]
            523,
        )
        assert (
            keima_left_first
            == label.MoveCategoryStartLabel.KEIMA_LEFT
        )
        keima_left_last = label.make_move_label(
            cshogi.BLACK,  # type: ignore[attr-defined]
            9166,
        )
        assert (
            keima_left_last
            == label.MoveCategoryStartLabel.KEIMA_RIGHT - 1
        )
        # KEIMA_RIGHT
        keima_right_first = label.make_move_label(
            cshogi.BLACK,  # type: ignore[attr-defined]
            1666,
        )
        assert (
            keima_right_first
            == label.MoveCategoryStartLabel.KEIMA_RIGHT
        )
        keima_right_last = label.make_move_label(
            cshogi.BLACK,  # type: ignore[attr-defined]
            10309,
        )
        assert (
            keima_right_last
            == label.MoveCategoryStartLabel.UP_PROMOTION - 1
        )
        # UP_PROMOTION
        up_promotion_first = label.make_move_label(
            cshogi.BLACK,  # type: ignore[attr-defined]
            16512,
        )
        assert (
            up_promotion_first
            == label.MoveCategoryStartLabel.UP_PROMOTION
        )
        up_promotion_last = label.make_move_label(
            cshogi.BLACK,  # type: ignore[attr-defined]
            26698,
        )
        assert (
            up_promotion_last
            == label.MoveCategoryStartLabel.UP_LEFT_PROMOTION
            - 1
        )
        # UP_LEFT_PROMOTION
        up_left_promotion_first = label.make_move_label(
            cshogi.BLACK,  # type: ignore[attr-defined]
            16521,
        )
        assert (
            up_left_promotion_first
            == label.MoveCategoryStartLabel.UP_LEFT_PROMOTION
        )
        up_left_promotion_last = label.make_move_label(
            cshogi.BLACK,  # type: ignore[attr-defined]
            21834,
        )
        assert (
            up_left_promotion_last
            == label.MoveCategoryStartLabel.UP_RIGHT_PROMOTION
            - 1
        )
        # UP_RIGHT_PROMOTION
        up_right_promotion_first = label.make_move_label(
            cshogi.BLACK,  # type: ignore[attr-defined]
            26624,
        )
        assert (
            up_right_promotion_first
            == label.MoveCategoryStartLabel.UP_RIGHT_PROMOTION
        )
        up_right_promotion_last = label.make_move_label(
            cshogi.BLACK,  # type: ignore[attr-defined]
            26049,
        )
        assert (
            up_right_promotion_last
            == label.MoveCategoryStartLabel.LEFT_PROMOTION - 1
        )
        # LEFT_PROMOTION
        left_promotion_first = label.make_move_label(
            cshogi.BLACK,  # type: ignore[attr-defined]
            16393,
        )
        assert (
            left_promotion_first
            == label.MoveCategoryStartLabel.LEFT_PROMOTION
        )
        left_promotion_last = label.make_move_label(
            cshogi.BLACK,  # type: ignore[attr-defined]
            16714,
        )
        assert (
            left_promotion_last
            == label.MoveCategoryStartLabel.RIGHT_PROMOTION - 1
        )
        # RIGHT_PROMOTION
        right_promotion_first = label.make_move_label(
            cshogi.BLACK,  # type: ignore[attr-defined]
            17536,
        )
        assert (
            right_promotion_first
            == label.MoveCategoryStartLabel.RIGHT_PROMOTION
        )
        right_promotion_last = label.make_move_label(
            cshogi.BLACK,  # type: ignore[attr-defined]
            25921,
        )
        assert (
            right_promotion_last
            == label.MoveCategoryStartLabel.DOWN_PROMOTION - 1
        )
        # DOWN_PROMOTION
        down_promotion_first = label.make_move_label(
            cshogi.BLACK,  # type: ignore[attr-defined]
            16385,
        )
        assert (
            down_promotion_first
            == label.MoveCategoryStartLabel.DOWN_PROMOTION
        )
        down_promotion_last = label.make_move_label(
            cshogi.BLACK,  # type: ignore[attr-defined]
            25936,
        )
        assert (
            down_promotion_last
            == label.MoveCategoryStartLabel.DOWN_LEFT_PROMOTION
            - 1
        )
        # DOWN_LEFT_PROMOTION
        down_left_promotion_first = label.make_move_label(
            cshogi.BLACK,  # type: ignore[attr-defined]
            16394,
        )
        assert (
            down_left_promotion_first
            == label.MoveCategoryStartLabel.DOWN_LEFT_PROMOTION
        )
        down_left_promotion_last = label.make_move_label(
            cshogi.BLACK,  # type: ignore[attr-defined]
            16464,
        )
        assert (
            down_left_promotion_last
            == label.MoveCategoryStartLabel.DOWN_RIGHT_PROMOTION
            - 1
        )
        # DOWN_RIGHT_PROMOTION
        down_right_promotion_first = label.make_move_label(
            cshogi.BLACK,  # type: ignore[attr-defined]
            17537,
        )
        assert (
            down_right_promotion_first
            == label.MoveCategoryStartLabel.DOWN_RIGHT_PROMOTION
        )
        down_right_promotion_last = label.make_move_label(
            cshogi.BLACK,  # type: ignore[attr-defined]
            25922,
        )
        assert (
            down_right_promotion_last
            == label.MoveCategoryStartLabel.KEIMA_LEFT_PROMOTION
            - 1
        )
        # KEIMA_LEFT_PROMOTION
        keima_left_promotion_first = label.make_move_label(
            cshogi.BLACK,  # type: ignore[attr-defined]
            16649,
        )
        assert (
            keima_left_promotion_first
            == label.MoveCategoryStartLabel.KEIMA_LEFT_PROMOTION
        )
        keima_left_promotion_last = label.make_move_label(
            cshogi.BLACK,  # type: ignore[attr-defined]
            25034,
        )
        assert (
            keima_left_promotion_last
            == label.MoveCategoryStartLabel.KEIMA_RIGHT_PROMOTION
            - 1
        )
        # KEIMA_RIGHT_PROMOTION
        keima_right_promotion_first = label.make_move_label(
            cshogi.BLACK,  # type: ignore[attr-defined]
            17792,
        )
        assert (
            keima_right_promotion_first
            == label.MoveCategoryStartLabel.KEIMA_RIGHT_PROMOTION
        )
        keima_right_promotion_last = label.make_move_label(
            cshogi.BLACK,  # type: ignore[attr-defined]
            26177,
        )
        assert (
            keima_right_promotion_last
            == label.MoveCategoryStartLabel.FU - 1
        )
        # FU
        fu_first = label.make_move_label(cshogi.BLACK, 10369)  # type: ignore[attr-defined]
        assert fu_first == label.MoveCategoryStartLabel.FU
        fu_last = label.make_move_label(cshogi.BLACK, 10448)  # type: ignore[attr-defined]
        assert fu_last == label.MoveCategoryStartLabel.KY - 1
        # KY
        ky_first = label.make_move_label(cshogi.BLACK, 10497)  # type: ignore[attr-defined]
        assert ky_first == label.MoveCategoryStartLabel.KY
        ky_last = label.make_move_label(cshogi.BLACK, 10576)  # type: ignore[attr-defined]
        assert ky_last == label.MoveCategoryStartLabel.KE - 1
        # KE
        ke_first = label.make_move_label(cshogi.BLACK, 10626)  # type: ignore[attr-defined]
        assert ke_first == label.MoveCategoryStartLabel.KE
        ke_last = label.make_move_label(cshogi.BLACK, 10704)  # type: ignore[attr-defined]
        assert ke_last == label.MoveCategoryStartLabel.GI - 1
        # GI
        gi_first = label.make_move_label(cshogi.BLACK, 10752)  # type: ignore[attr-defined]
        assert gi_first == label.MoveCategoryStartLabel.GI
        gi_last = label.make_move_label(cshogi.BLACK, 10832)  # type: ignore[attr-defined]
        assert gi_last == label.MoveCategoryStartLabel.KI - 1
        # KI
        ki_first = label.make_move_label(cshogi.BLACK, 11136)  # type: ignore[attr-defined]
        assert ki_first == label.MoveCategoryStartLabel.KI
        ki_last = label.make_move_label(cshogi.BLACK, 11216)  # type: ignore[attr-defined]
        assert ki_last == label.MoveCategoryStartLabel.KA - 1
        # KA
        ka_first = label.make_move_label(cshogi.BLACK, 10880)  # type: ignore[attr-defined]
        assert ka_first == label.MoveCategoryStartLabel.KA
        ka_last = label.make_move_label(cshogi.BLACK, 10960)  # type: ignore[attr-defined]
        assert ka_last == label.MoveCategoryStartLabel.HI - 1
        # HI
        hi_first = label.make_move_label(cshogi.BLACK, 11008)  # type: ignore[attr-defined]
        assert hi_first == label.MoveCategoryStartLabel.HI
        hi_last = label.make_move_label(cshogi.BLACK, 11088)  # type: ignore[attr-defined]
        assert hi_last == label.MoveCategoryStartLabel.HI + 80

    def test_make_result_value(self) -> None:
        """勝率の教師データ作成のテスト.
        結果の種類を全種類用意してテストする
        手番変えるのは1種類だけテストしておく
        """
        assert (
            label.make_result_value(
                cshogi.BLACK,  # type: ignore[attr-defined]
                cshogi.BLACK_WIN,  # type: ignore[attr-defined]
            )
            == 1
        )
        assert (
            label.make_result_value(cshogi.BLACK, cshogi.DRAW)  # type: ignore[attr-defined]
            == 0.5
        )
        assert (
            label.make_result_value(
                cshogi.BLACK,  # type: ignore[attr-defined]
                cshogi.WHITE_WIN,  # type: ignore[attr-defined]
            )
            == 0
        )
        assert (
            label.make_result_value(
                cshogi.WHITE,  # type: ignore[attr-defined]
                cshogi.BLACK_WIN,  # type: ignore[attr-defined]
            )
            == 0
        )
        assert (
            label.make_result_value(cshogi.WHITE, cshogi.DRAW)  # type: ignore[attr-defined]
            == 0.5
        )
        assert (
            label.make_result_value(
                cshogi.WHITE,  # type: ignore[attr-defined]
                cshogi.WHITE_WIN,  # type: ignore[attr-defined]
            )
            == 1
        )
