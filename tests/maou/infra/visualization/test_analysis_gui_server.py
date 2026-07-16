"""analysis_gui_server (棋譜解析 GUI サーバー) のテスト．"""

import json
from pathlib import Path

import pytest

pytest.importorskip("gradio")

import gradio as gr  # noqa: E402

from maou.app.analysis.game_analyzer import (  # noqa: E402
    FixedPlayoutsAllocator,
    GameAnalyzer,
)
from maou.infra.visualization.analysis_gui_server import (  # noqa: E402
    AnalysisGuiServer,
    _clamp_ply,
    _file_path,
)

RESOURCES = (
    Path(__file__).parents[2] / "app" / "analysis" / "resources"
)
MINI_CSA = RESOURCES / "mini.csa"


@pytest.fixture(scope="module")
def real_report_path(
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """mock 評価器で mini.csa の実レポートを生成する (integration)．"""
    analyzer = GameAnalyzer()
    result = analyzer.analyze(
        GameAnalyzer.AnalyzeOption(
            input_path=MINI_CSA,
            input_format="csa",
            allocator=FixedPlayoutsAllocator(playouts=16),
            root_dfpn=False,
            leaf_mate=False,
        )
    )
    path = tmp_path_factory.mktemp("report") / "report.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    return path


class TestAnalysisGuiServer:
    """AnalysisGuiServer のテスト．"""

    def test_demo_builds_empty(self) -> None:
        """棋譜なしでも demo を構築できる．"""
        demo = AnalysisGuiServer().create_demo()
        assert len(demo.blocks) > 0

    def test_demo_builds_with_kifu(self) -> None:
        """棋譜を初期ロードして demo を構築できる．"""
        server = AnalysisGuiServer(kifu_path=MINI_CSA)
        assert server.initial_view is not None
        assert server.initial_view.document.n_moves == 4
        demo = server.create_demo()
        assert len(demo.blocks) > 0

    def test_demo_builds_with_report(
        self, real_report_path: Path
    ) -> None:
        """analyze-game の実レポートを初期ロードできる (integration)．"""
        server = AnalysisGuiServer(
            kifu_path=MINI_CSA,
            report_path=real_report_path,
        )
        assert server.initial_view is not None
        assert server.initial_view.report is not None
        demo = server.create_demo()
        assert len(demo.blocks) > 0

    def test_report_without_kifu_raises(
        self, real_report_path: Path
    ) -> None:
        """レポートのみの指定はエラー．"""
        with pytest.raises(ValueError, match="input-path"):
            AnalysisGuiServer(report_path=real_report_path)

    def test_render_without_view(self) -> None:
        """棋譜未読込時はプレースホルダを返す．"""
        server = AnalysisGuiServer()
        board, _fig, candidates, sfen, position_str, note = (
            server._render(None, 0, True, False, 5, "winrate")
        )
        assert "棋譜が読み込まれていません" in board
        assert candidates == []
        assert sfen == ""
        assert position_str == ""
        assert note == ""

    def test_render_with_view(
        self, real_report_path: Path
    ) -> None:
        """読込済みの状態で盤面 SVG と候補手を返す．"""
        server = AnalysisGuiServer(
            kifu_path=MINI_CSA,
            report_path=real_report_path,
        )
        board, fig, candidates, sfen, position_str, _note = (
            server._render(
                server.initial_view,
                0,
                True,
                False,
                5,
                "winrate",
            )
        )
        assert "<svg" in board
        assert fig.data
        assert candidates
        assert sfen.endswith("b - 1")
        assert position_str.startswith("position sfen")

    def test_render_clamps_ply(self) -> None:
        """範囲外の ply は末尾に丸められる．"""
        server = AnalysisGuiServer(kifu_path=MINI_CSA)
        board, *_ = server._render(
            server.initial_view, 999, True, False, 5, "winrate"
        )
        assert "<svg" in board

    def test_on_load(self, real_report_path: Path) -> None:
        """_on_load が状態と全出力を返す．"""
        server = AnalysisGuiServer()
        outputs = server._on_load(
            str(MINI_CSA),
            str(real_report_path),
            True,
            False,
            5,
            "winrate",
        )
        assert len(outputs) == 11
        view = outputs[0]
        assert view is not None
        assert view.document.n_moves == 4
        assert view.report is not None
        # 読み込みステータス
        assert "4 手" in outputs[4]

    def test_on_load_without_kifu_raises(self) -> None:
        """棋譜ファイル未指定の読み込みはエラー．"""
        server = AnalysisGuiServer()
        with pytest.raises(gr.Error):
            server._on_load(
                None, None, True, False, 5, "winrate"
            )


class TestHelpers:
    """モジュールヘルパーのテスト．"""

    def test_clamp_ply_without_view(self) -> None:
        assert _clamp_ply(None, 5) == 0

    def test_file_path(self) -> None:
        assert _file_path(None) is None
        assert _file_path("/tmp/a.csa") == Path("/tmp/a.csa")
