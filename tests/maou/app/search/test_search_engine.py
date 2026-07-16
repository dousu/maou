"""maou._rust.maou_search.SearchEngine (永続エンジン binding) のテスト．"""

import pytest

from maou._rust.maou_search import SearchEngine, search

SFEN_INITIAL = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
MATE_IN_1 = "4k4/9/4P4/9/9/9/9/9/9 b G 1"


class TestSearchEngine:
    def test_engine_reuse_is_deterministic(self) -> None:
        engine = SearchEngine()
        r1 = engine.search(SFEN_INITIAL, max_playouts=200)
        r2 = engine.search(SFEN_INITIAL, max_playouts=200)
        # mock 評価器は決定論的なので同一エンジンの再利用で結果が一致する
        assert r1.best_move == r2.best_move
        assert r1.winrate == r2.winrate
        assert r1.stop == "playout_limit"

    def test_engine_matches_one_shot_search(self) -> None:
        # 同一条件なら関数 search (毎回評価器を構築) と結果が一致する
        engine_result = SearchEngine().search(
            SFEN_INITIAL, max_playouts=200
        )
        function_result = search(SFEN_INITIAL, max_playouts=200)
        assert (
            engine_result.best_move == function_result.best_move
        )
        assert engine_result.winrate == function_result.winrate
        assert (
            engine_result.playouts == function_result.playouts
        )

    def test_moves_history_is_applied(self) -> None:
        engine = SearchEngine()
        result = engine.search(
            SFEN_INITIAL,
            moves=["7g7f", "3c3d"],
            max_playouts=100,
        )
        assert result.best_move is not None

    def test_mate_in_1_is_proven_by_root_dfpn(self) -> None:
        engine = SearchEngine()
        result = engine.search(MATE_IN_1, max_playouts=2000)
        assert result.stop == "root_proven"
        assert result.best_move == "G*5b"
        assert result.winrate == 1.0

    def test_illegal_move_raises(self) -> None:
        engine = SearchEngine()
        with pytest.raises(ValueError):
            engine.search(
                SFEN_INITIAL, moves=["9a9b"], max_playouts=10
            )

    def test_invalid_sfen_raises(self) -> None:
        engine = SearchEngine()
        with pytest.raises(ValueError):
            engine.search("not-a-sfen", max_playouts=10)
