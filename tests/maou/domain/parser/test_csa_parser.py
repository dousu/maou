"""CSA パーサ (Rust backend maou._rust.maou_shogi.parse_csa_str) のテスト．

golden fixture (rust/maou_shogi/tests/fixtures/kifu/) は cshogi (oracle) の
出力から生成されており，Rust 側 parity は rust/maou_shogi/tests/
kifu_parity.rs が検証する．本テストは PyO3 表面 (GameRecord) の契約と，
cshogi がインストールされている場合の直接交差検証を行う．
"""

from pathlib import Path

import pytest

from maou._rust.maou_shogi import parse_csa_str

FIXTURE_DIR = (
    Path(__file__).resolve().parents[4]
    / "rust/maou_shogi/tests/fixtures/kifu"
)

HIRATE_SFEN = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"


def _var_info_dict(record: object) -> dict[str, str]:
    return dict(record.var_info)  # type: ignore[attr-defined]


class TestCsaParser:
    def test_parse_floodgate_game(self) -> None:
        (record,) = _parse("test_data_1.csa")
        assert record.sfen == HIRATE_SFEN
        assert record.endgame == "%TORYO"
        assert record.win == 1
        assert list(record.ratings) == [3495.0, 3300.0]
        assert len(record.moves) == 163
        # +7776FU (cshogi 互換 32-bit エンコーディング)
        assert record.moves[0] == 0x00011E3B
        # scores / comments は moves と同長に整列される
        assert len(record.scores) == len(record.moves)
        assert len(record.comments) == len(record.moves)

    def test_start_time_in_var_info(self) -> None:
        (record,) = _parse("test_data_1.csa")
        # partitioningKey の日付導出は maou_convert 側 (date.rs) に移り，
        # パーサは START_TIME を var_info として保持する
        assert (
            _var_info_dict(record)["START_TIME"]
            == "2020/01/31 10:00:03"
        )

    def test_no_start_time(self) -> None:
        (record,) = parse_csa_str("V2.2\nPI\n+\n%CHUDAN\n")
        assert "START_TIME" not in _var_info_dict(record)

    def test_handicap_pi(self) -> None:
        (record,) = _parse("csa_handicap_pi.csa")
        # 二枚落ち相当 (82HI 22KA 除去，上手番)
        assert record.sfen == (
            "lnsgkgsnl/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1"
        )

    def test_multi_game_returns_all_games(self) -> None:
        # 複数対局は全て返る (従来 Python 版は先頭のみ利用していた)
        records = _parse("csa_multi_game.csa")
        assert len(records) == 2
        assert len(records[0].moves) == 1
        assert records[0].endgame == "%TORYO"
        assert len(records[1].moves) == 2

    def test_no_moves_game(self) -> None:
        (record,) = _parse("test_data_no_moves.csa")
        assert list(record.moves) == []
        assert record.endgame == "%TORYO"
        assert record.win == 2

    def test_invalid_content_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_csa_str("V2.2\nPI\n+\nGARBAGE LINE\n")

    def test_individual_placement_and_hand(self) -> None:
        # cshogi は P+/P-/AL で segfault していた領域 (spec 準拠の独自拡張)
        (record,) = parse_csa_str(
            "V2.2\n"
            "P-51OU\n"
            "P+59OU28HI\n"
            "P+00KI\n"
            "P-00AL\n"
            "+\n"
            "%CHUDAN\n"
        )
        assert record.sfen == (
            "4k4/9/9/9/9/9/9/7R1/4K4 b Gr2b3g4s4n4l18p 1"
        )


class TestCshogiCrossCheck:
    """cshogi (dev 依存) が使える場合の直接交差検証．"""

    def test_all_csa_fixtures_match_cshogi(self) -> None:
        cshogi_csa = pytest.importorskip("cshogi.CSA")
        for path in sorted(FIXTURE_DIR.glob("*.csa")):
            content = path.read_text()
            (ours,) = _first_game_only(parse_csa_str(content))
            ref = cshogi_csa.Parser.parse_str(content)[0]
            assert ours.sfen == ref.sfen, path.name
            assert ours.endgame == ref.endgame, path.name
            assert ours.win == ref.win, path.name
            assert list(ours.moves) == list(ref.moves), (
                path.name
            )
            assert list(ours.scores) == list(ref.scores), (
                path.name
            )
            assert list(ours.comments) == list(ref.comments), (
                path.name
            )
            assert [float(r) for r in ours.ratings] == list(
                ref.ratings
            ), path.name


def _parse(name: str) -> list:
    return parse_csa_str((FIXTURE_DIR / name).read_text())


def _first_game_only(records: list) -> tuple:
    # cshogi の parse_str(...)[0] は先頭 1 局のみなので先頭で比較する
    return (records[0],)
