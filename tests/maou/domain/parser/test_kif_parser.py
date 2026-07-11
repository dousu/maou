"""KifParser (Rust backend maou_shogi::kifu) のテスト．

golden fixture (rust/maou_shogi/tests/fixtures/kifu/) は cshogi (oracle) の
出力から生成されており，Rust 側 parity は rust/maou_shogi/tests/
kifu_parity.rs が検証する．本テストは Python 表面の契約 (従来からの
改善点含む) と，cshogi がインストールされている場合の直接交差検証を行う．
"""

from datetime import date
from pathlib import Path

import pytest

from maou.domain.parser.csa_parser import CSAParser
from maou.domain.parser.kif_parser import KifParser

FIXTURE_DIR = (
    Path(__file__).resolve().parents[4]
    / "rust/maou_shogi/tests/fixtures/kifu"
)

HIRATE_SFEN = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"


class TestKifParser:
    def test_parse_basic_game(self) -> None:
        parser = KifParser()
        parser.parse(
            (FIXTURE_DIR / "kif_edge_basic.kifu").read_text()
        )
        assert parser.init_pos_sfen() == HIRATE_SFEN
        assert parser.endgame() == "%TORYO"
        assert parser.winner() == 1
        assert len(parser.moves()) == 7
        assert parser.moves()[0] == 0x00011E3B  # ７六歩(77)

    def test_same_game_as_csa_yields_same_moves(self) -> None:
        # 同一対局の CSA と KIF は同じ move int 列になる
        csa = CSAParser()
        csa.parse((FIXTURE_DIR / "test_data_1.csa").read_text())
        kif = KifParser()
        kif.parse(
            (
                FIXTURE_DIR / "kif_gen_test_data_1.kifu"
            ).read_text()
        )
        assert list(kif.moves()) == list(csa.moves())

    def test_scores_and_comments_aligned_with_moves(
        self,
    ) -> None:
        # 従来 (cshogi 版) は scores() が [] で HCPE 変換が 1 行も
        # 出力されなかった．moves と同長の 0 列を返すのが本実装の契約
        parser = KifParser()
        parser.parse(
            (
                FIXTURE_DIR / "kif_gen_test_data_1.kifu"
            ).read_text()
        )
        n = len(parser.moves())
        assert n == 163
        assert parser.scores() == [0] * n
        assert len(parser.comments()) == n

    def test_partitioning_key_from_start_datetime_header(
        self,
    ) -> None:
        parser = KifParser()
        parser.parse(
            (
                FIXTURE_DIR / "kif_gen_test_data_1.kifu"
            ).read_text()
        )
        assert parser.partitioning_key_value() == date(
            2020, 1, 31
        )
        assert parser.clustering_key_value() == date(
            2020, 1, 31
        )

    def test_start_datetime_without_seconds(self) -> None:
        parser = KifParser()
        parser.parse(
            (FIXTURE_DIR / "kif_edge_basic.kifu").read_text()
        )
        assert parser.partitioning_key_value() == date(
            2026, 7, 11
        )

    def test_no_header_returns_none_keys(self) -> None:
        parser = KifParser()
        parser.parse(
            (
                FIXTURE_DIR / "kif_edge_sennichite.kifu"
            ).read_text()
        )
        assert parser.partitioning_key_value() is None
        assert parser.clustering_key_value() is None

    def test_ratings_empty(self) -> None:
        parser = KifParser()
        parser.parse(
            (FIXTURE_DIR / "kif_edge_basic.kifu").read_text()
        )
        assert parser.ratings() == []

    def test_handicap_game(self) -> None:
        parser = KifParser()
        parser.parse(
            (FIXTURE_DIR / "kif_edge_handicap.kifu").read_text()
        )
        # 六枚落ち (上手番開始)
        assert parser.init_pos_sfen() == (
            "2sgkgs2/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1"
        )
        assert parser.endgame() == "%CHUDAN"
        assert parser.winner() is None

    def test_no_result_line_yields_none(self) -> None:
        parser = KifParser()
        parser.parse(
            (
                FIXTURE_DIR / "kif_edge_no_result.kifu"
            ).read_text()
        )
        assert parser.endgame() is None
        assert parser.winner() is None

    def test_funari_move_is_parsed(self) -> None:
        # cshogi は「不成」を黙って読み飛ばし盤面が壊れていた (改善点)
        parser = KifParser()
        parser.parse(
            "手合割：平手\n"
            "   1 ７六歩(77)\n"
            "   2 ３四歩(33)\n"
            "   3 ２二角不成(88)\n"
        )
        moves = parser.moves()
        assert len(moves) == 3
        assert (
            moves[2] >> 14
        ) & 1 == 0  # 不成 (promotion flag なし)

    def test_bod_initial_position_raises(self) -> None:
        # BOD (局面図) は黙って平手扱いにせず明示エラー (改善点)
        parser = KifParser()
        with pytest.raises(ValueError):
            parser.parse(
                "後手の持駒：なし\n"
                "+---------------------------+\n"
                "|v香v桂v銀v金v玉v金v銀v桂v香|一\n"
            )

    def test_unknown_handicap_raises(self) -> None:
        parser = KifParser()
        with pytest.raises(ValueError):
            parser.parse("手合割：その他\n")


class TestCshogiCrossCheck:
    """cshogi (dev 依存) が使える場合の直接交差検証．"""

    def test_all_kifu_fixtures_match_cshogi(self) -> None:
        cshogi_kif = pytest.importorskip("cshogi.KIF")
        for path in sorted(FIXTURE_DIR.glob("*.kifu")):
            content = path.read_text()
            ours = KifParser()
            ours.parse(content)
            ref = cshogi_kif.Parser.parse_str(content)
            assert ours.init_pos_sfen() == ref.sfen, path.name
            assert ours.endgame() == ref.endgame, path.name
            assert ours.winner() == ref.win, path.name
            assert list(ours.moves()) == list(ref.moves), (
                path.name
            )
