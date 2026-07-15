"""KIF パーサ (Rust backend maou._rust.maou_shogi.parse_kif_str) のテスト．

golden fixture (rust/maou_shogi/tests/fixtures/kifu/) は cshogi (oracle) の
出力から生成されており，Rust 側 parity は rust/maou_shogi/tests/
kifu_parity.rs が検証する．本テストは PyO3 表面 (GameRecord) の契約 (従来
からの改善点含む) と，cshogi がインストールされている場合の直接交差検証を行う．
"""

from pathlib import Path
from typing import Any

import pytest

from maou._rust.maou_shogi import parse_csa_str, parse_kif_str

FIXTURE_DIR = (
    Path(__file__).resolve().parents[4]
    / "rust/maou_shogi/tests/fixtures/kifu"
)

HIRATE_SFEN = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"


def _kif(name: str) -> Any:
    return parse_kif_str((FIXTURE_DIR / name).read_text())


def _var_info_dict(record: object) -> dict[str, str]:
    return dict(record.var_info)  # type: ignore[attr-defined]


class TestKifParser:
    def test_parse_basic_game(self) -> None:
        record = _kif("kif_edge_basic.kifu")
        assert record.sfen == HIRATE_SFEN
        assert record.endgame == "%TORYO"
        assert record.win == 1
        assert len(record.moves) == 7
        assert record.moves[0] == 0x00011E3B  # ７六歩(77)

    def test_same_game_as_csa_yields_same_moves(self) -> None:
        # 同一対局の CSA と KIF は同じ move int 列になる
        (csa,) = parse_csa_str(
            (FIXTURE_DIR / "test_data_1.csa").read_text()
        )
        kif = _kif("kif_gen_test_data_1.kifu")
        assert list(kif.moves) == list(csa.moves)

    def test_scores_and_comments_aligned_with_moves(
        self,
    ) -> None:
        # 従来 (cshogi 版) は scores() が [] で HCPE 変換が 1 行も
        # 出力されなかった．moves と同長の 0 列を返すのが本実装の契約
        record = _kif("kif_gen_test_data_1.kifu")
        n = len(record.moves)
        assert n == 163
        assert list(record.scores) == [0] * n
        assert len(record.comments) == n

    def test_start_datetime_in_var_info(self) -> None:
        # partitioningKey の日付導出は maou_convert 側 (date.rs) に移り，
        # パーサは開始日時ヘッダを var_info として保持する
        record = _kif("kif_gen_test_data_1.kifu")
        assert (
            _var_info_dict(record)["開始日時"]
            == "2020/01/31 10:00:03"
        )

    def test_no_header_returns_no_start_datetime(self) -> None:
        record = _kif("kif_edge_sennichite.kifu")
        assert "開始日時" not in _var_info_dict(record)

    def test_ratings_empty(self) -> None:
        # KIF はレーティング情報を持たない (record.ratings は 0 埋め)
        record = _kif("kif_edge_basic.kifu")
        assert list(record.ratings) == [0.0, 0.0]

    def test_handicap_game(self) -> None:
        record = _kif("kif_edge_handicap.kifu")
        # 六枚落ち (上手番開始)
        assert record.sfen == (
            "2sgkgs2/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1"
        )
        assert record.endgame == "%CHUDAN"
        assert record.win is None

    def test_no_result_line_yields_none(self) -> None:
        record = _kif("kif_edge_no_result.kifu")
        assert record.endgame is None
        assert record.win is None

    def test_funari_move_is_parsed(self) -> None:
        # cshogi は「不成」を黙って読み飛ばし盤面が壊れていた (改善点)
        record = parse_kif_str(
            "手合割：平手\n"
            "   1 ７六歩(77)\n"
            "   2 ３四歩(33)\n"
            "   3 ２二角不成(88)\n"
        )
        assert len(record.moves) == 3
        assert (
            record.moves[2] >> 14
        ) & 1 == 0  # 不成 (promotion flag なし)

    def test_bod_initial_position_raises(self) -> None:
        # BOD (局面図) は黙って平手扱いにせず明示エラー (改善点)
        with pytest.raises(ValueError):
            parse_kif_str(
                "後手の持駒：なし\n"
                "+---------------------------+\n"
                "|v香v桂v銀v金v玉v金v銀v桂v香|一\n"
            )

    def test_unknown_handicap_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_kif_str("手合割：その他\n")


class TestCshogiCrossCheck:
    """cshogi (dev 依存) が使える場合の直接交差検証．"""

    def test_all_kifu_fixtures_match_cshogi(self) -> None:
        cshogi_kif = pytest.importorskip("cshogi.KIF")
        for path in sorted(FIXTURE_DIR.glob("*.kifu")):
            content = path.read_text()
            ours = parse_kif_str(content)
            ref = cshogi_kif.Parser.parse_str(content)
            assert ours.sfen == ref.sfen, path.name
            assert ours.endgame == ref.endgame, path.name
            assert ours.win == ref.win, path.name
            assert list(ours.moves) == list(ref.moves), (
                path.name
            )
