"""interactive_analyzer (GUI 常駐の対話解析) のテスト．

mock 評価器 (model_path=None) の SearchEngine を実際に使う
(決定論的・GPU/モデル不要)．
"""

import threading
from pathlib import Path

import pytest

from maou.app.analysis.analysis_session import (
    GameDocument,
    advance_move,
    build_variation_tree,
    load_game,
)
from maou.app.analysis.interactive_analyzer import (
    DEFAULT_TIME_MS,
    EngineSettings,
    InteractiveAnalyzer,
    _normalize_budget,
)

RESOURCES = Path(__file__).parent / "resources"

# 高速化のためルート dfpn / leaf-mate は無効化
FAST_SETTINGS = EngineSettings(
    root_dfpn=False, leaf_mate=False, num_candidates=5
)

POSITION_KEYS = {
    "ply",
    "side_to_move",
    "sfen",
    "played_move",
    "best_move",
    "match",
    "winrate",
    "eval_cp",
    "played_move_winrate",
    "winrate_loss",
    "pv",
    "candidates",
    "mate_found",
    "playouts",
    "elapsed_ms",
    "stop",
    "record_time_s",
    "record_score",
    "record_comment",
}


@pytest.fixture
def mini_document() -> GameDocument:
    """mini.csa (4 手 + 投了) から GameDocument を構築する．"""
    data = (RESOURCES / "mini.csa").read_bytes()
    return load_game(data, "csa")


class TestEngineSettings:
    """EngineSettings のテスト．"""

    def test_describe_matches_analyze_game_schema(self) -> None:
        """describe が analyze-game の engine セクション互換．"""
        meta = FAST_SETTINGS.describe()
        assert set(meta.keys()) == {
            "model_path",
            "threads",
            "batch_size",
            "cuda",
            "tensorrt",
            "root_dfpn",
            "root_dfpn_nodes",
            "root_dfpn_depth",
            "leaf_mate",
            "leaf_mate_nodes",
            "leaf_mate_threads",
        }
        assert meta["model_path"] is None
        assert meta["root_dfpn"] is False

    def test_is_mock(self) -> None:
        """model_path 未指定は mock 評価器扱い．"""
        analyzer = InteractiveAnalyzer(FAST_SETTINGS)
        assert analyzer.is_mock


class TestNormalizeBudget:
    """予算正規化のテスト．"""

    def test_default(self) -> None:
        assert _normalize_budget(None, None) == (
            DEFAULT_TIME_MS,
            None,
        )

    def test_passthrough(self) -> None:
        assert _normalize_budget(500, None) == (500, None)
        assert _normalize_budget(None, 32) == (None, 32)


class TestAnalyzePosition:
    """analyze_position のテスト (mock 評価器 integration)．"""

    def test_mainline_node_has_played_move(
        self, mini_document: GameDocument
    ) -> None:
        """本譜ノードは実戦手比較付きの記録になる．"""
        analyzer = InteractiveAnalyzer(FAST_SETTINGS)
        tree = build_variation_tree(mini_document)
        record = analyzer.analyze_position(
            mini_document, tree, max_playouts=8
        )
        assert set(record.keys()) == POSITION_KEYS
        assert record["ply"] == 1
        assert record["side_to_move"] == "b"
        assert record["played_move"] == "7g7f"
        assert record["sfen"] == (
            mini_document.snapshots[0].sfen
        )
        assert record["candidates"]

    def test_branch_node_has_no_played_move(
        self, mini_document: GameDocument
    ) -> None:
        """分岐ノードは played_move なし (match=False)．"""
        analyzer = InteractiveAnalyzer(FAST_SETTINGS)
        tree = build_variation_tree(mini_document)
        advance_move(tree, "2g2f")
        record = analyzer.analyze_position(
            mini_document, tree, max_playouts=8
        )
        assert record["ply"] == 2
        assert record["side_to_move"] == "w"
        assert record["played_move"] is None
        assert record["match"] is False
        assert record["winrate_loss"] is None

    def test_engine_is_reused(
        self, mini_document: GameDocument
    ) -> None:
        """SearchEngine は 1 回だけ構築される．"""
        analyzer = InteractiveAnalyzer(FAST_SETTINGS)
        tree = build_variation_tree(mini_document)
        analyzer.analyze_position(
            mini_document, tree, max_playouts=8
        )
        engine = analyzer._engine
        assert engine is not None
        analyzer.analyze_position(
            mini_document, tree, max_playouts=8
        )
        assert analyzer._engine is engine


class TestAnalyzeMainline:
    """analyze_mainline / build_report のテスト．"""

    def test_full_run_and_report(
        self, mini_document: GameDocument
    ) -> None:
        """全局面を解析して analyze-game 互換レポートを作れる．"""
        analyzer = InteractiveAnalyzer(FAST_SETTINGS)
        positions = []
        for i, n, record in analyzer.analyze_mainline(
            mini_document, max_playouts=8
        ):
            assert n == 4
            assert record["ply"] == i + 1
            positions.append(record)
        assert len(positions) == 4
        assert [p["played_move"] for p in positions] == (
            mini_document.moves_usi
        )
        report = analyzer.build_report(
            mini_document,
            positions,
            source_name="mini.csa",
            max_playouts=8,
        )
        assert set(report.keys()) == {
            "input",
            "engine",
            "budget",
            "positions",
            "summary",
        }
        assert report["input"]["path"] == "mini.csa"
        assert report["input"]["n_moves"] == 4
        assert report["budget"]["mode"] == "fixed_playouts"
        assert report["budget"]["per_position"] == {
            "max_playouts": 8,
            "time_ms": None,
        }
        assert "match_rate" in report["summary"]

    def test_cancel_stops_early(
        self, mini_document: GameDocument
    ) -> None:
        """キャンセルイベントで途中終了する．"""
        analyzer = InteractiveAnalyzer(FAST_SETTINGS)
        cancel = threading.Event()
        results = []
        for i, _n, record in analyzer.analyze_mainline(
            mini_document, max_playouts=8, cancel=cancel
        ):
            results.append(record)
            if i == 1:
                cancel.set()
        assert len(results) == 2

    def test_build_report_count_mismatch_raises(
        self, mini_document: GameDocument
    ) -> None:
        """positions の件数不一致はエラー．"""
        analyzer = InteractiveAnalyzer(FAST_SETTINGS)
        with pytest.raises(ValueError, match="件数"):
            analyzer.build_report(
                mini_document, [], source_name="x.csa"
            )
