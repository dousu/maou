"""CSAParser (Rust backend maou_shogi::kifu) のテスト．

golden fixture (rust/maou_shogi/tests/fixtures/kifu/) は cshogi (oracle) の
出力から生成されており，Rust 側 parity は rust/maou_shogi/tests/
kifu_parity.rs が検証する．本テストは Python 表面の契約と，cshogi が
インストールされている場合の直接交差検証を行う．
"""

from datetime import date
from pathlib import Path

import pytest

from maou.domain.parser.csa_parser import CSAParser

FIXTURE_DIR = (
    Path(__file__).resolve().parents[4]
    / "rust/maou_shogi/tests/fixtures/kifu"
)

HIRATE_SFEN = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"


class TestCSAParser:
    def test_parse_floodgate_game(self) -> None:
        parser = CSAParser()
        parser.parse(
            (FIXTURE_DIR / "test_data_1.csa").read_text()
        )
        assert parser.init_pos_sfen() == HIRATE_SFEN
        assert parser.endgame() == "%TORYO"
        assert parser.winner() == 1
        assert [float(r) for r in parser.ratings()] == [
            3495.0,
            3300.0,
        ]
        moves = parser.moves()
        assert len(moves) == 163
        # +7776FU (cshogi 互換 32-bit エンコーディング)
        assert moves[0] == 0x00011E3B
        # scores / comments は moves と同長に整列される
        assert len(parser.scores()) == len(moves)
        assert len(parser.comments()) == len(moves)

    def test_partitioning_key_from_start_time(self) -> None:
        parser = CSAParser()
        parser.parse(
            (FIXTURE_DIR / "test_data_1.csa").read_text()
        )
        assert parser.partitioning_key_value() == date(
            2020, 1, 31
        )
        assert parser.clustering_key_value() == date(
            2020, 1, 31
        )

    def test_no_start_time_returns_none(self) -> None:
        parser = CSAParser()
        parser.parse("V2.2\nPI\n+\n%CHUDAN\n")
        assert parser.partitioning_key_value() is None
        assert parser.clustering_key_value() is None

    def test_handicap_pi(self) -> None:
        parser = CSAParser()
        parser.parse(
            (FIXTURE_DIR / "csa_handicap_pi.csa").read_text()
        )
        # 二枚落ち相当 (82HI 22KA 除去，上手番)
        assert parser.init_pos_sfen() == (
            "lnsgkgsnl/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1"
        )

    def test_multi_game_uses_first_game(self) -> None:
        # 従来 (cshogi 版) 互換: 複数対局は先頭のみ
        parser = CSAParser()
        parser.parse(
            (FIXTURE_DIR / "csa_multi_game.csa").read_text()
        )
        assert len(parser.moves()) == 1
        assert parser.endgame() == "%TORYO"

    def test_no_moves_game(self) -> None:
        parser = CSAParser()
        parser.parse(
            (FIXTURE_DIR / "test_data_no_moves.csa").read_text()
        )
        assert parser.moves() == []
        assert parser.endgame() == "%TORYO"
        assert parser.winner() == 2

    def test_invalid_content_raises(self) -> None:
        parser = CSAParser()
        with pytest.raises(ValueError):
            parser.parse("V2.2\nPI\n+\nGARBAGE LINE\n")

    def test_individual_placement_and_hand(self) -> None:
        # cshogi は P+/P-/AL で segfault していた領域 (spec 準拠の独自拡張)
        parser = CSAParser()
        parser.parse(
            "V2.2\n"
            "P-51OU\n"
            "P+59OU28HI\n"
            "P+00KI\n"
            "P-00AL\n"
            "+\n"
            "%CHUDAN\n"
        )
        assert parser.init_pos_sfen() == (
            "4k4/9/9/9/9/9/9/7R1/4K4 b Gr2b3g4s4n4l18p 1"
        )


class TestCshogiCrossCheck:
    """cshogi (dev 依存) が使える場合の直接交差検証．"""

    def test_all_csa_fixtures_match_cshogi(self) -> None:
        cshogi_csa = pytest.importorskip("cshogi.CSA")
        for path in sorted(FIXTURE_DIR.glob("*.csa")):
            content = path.read_text()
            ours = CSAParser()
            ours.parse(content)
            ref = cshogi_csa.Parser.parse_str(content)[0]
            assert ours.init_pos_sfen() == ref.sfen, path.name
            assert ours.endgame() == ref.endgame, path.name
            assert ours.winner() == ref.win, path.name
            assert list(ours.moves()) == list(ref.moves), (
                path.name
            )
            assert list(ours.scores()) == list(ref.scores), (
                path.name
            )
            assert list(ours.comments()) == list(
                ref.comments
            ), path.name
            assert [float(r) for r in ours.ratings()] == list(
                ref.ratings
            ), path.name
