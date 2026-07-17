"""analysis_gui_server (棋譜解析 GUI サーバー) のテスト．"""

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("gradio")

import gradio as gr  # noqa: E402

from maou.app.analysis.game_analyzer import (  # noqa: E402
    FixedPlayoutsAllocator,
    GameAnalyzer,
)
from maou.app.analysis.interactive_analyzer import (  # noqa: E402
    EngineSettings,
)
from maou.infra.visualization.analysis_gui_server import (  # noqa: E402
    AnalysisGuiServer,
    _clamp_ply,
    _file_path,
)
from maou.interface.analysis_gui import (  # noqa: E402
    ClickState,
)

RESOURCES = (
    Path(__file__).parents[2] / "app" / "analysis" / "resources"
)
MINI_CSA = RESOURCES / "mini.csa"

# mock 評価器で高速に回すテスト用エンジン設定
FAST_ENGINE = EngineSettings(
    root_dfpn=False, leaf_mate=False, num_candidates=5
)

# ハンドラ共通の表示オプション (arrows, pv, top_n, y_mode)
OPTS: tuple[bool, bool, int, str] = (True, False, 5, "winrate")


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


def _make_server(**kwargs: Any) -> AnalysisGuiServer:
    """テスト用の共通エンジン設定でサーバーを作る．"""
    kwargs.setdefault("engine_settings", FAST_ENGINE)
    return AnalysisGuiServer(**kwargs)


class TestAnalysisGuiServer:
    """AnalysisGuiServer のテスト．"""

    def test_demo_builds_empty(self) -> None:
        """棋譜なしでも demo を構築できる．"""
        demo = _make_server().create_demo()
        assert len(demo.blocks) > 0

    def test_demo_builds_with_kifu(self) -> None:
        """棋譜を初期ロードして demo を構築できる．"""
        server = _make_server(kifu_path=MINI_CSA)
        assert server.initial_view is not None
        assert server.initial_view.document.n_moves == 4
        assert server.initial_tree is not None
        assert len(server.initial_tree.mainline_ids) == 5
        demo = server.create_demo()
        assert len(demo.blocks) > 0

    def test_demo_builds_with_report(
        self, real_report_path: Path
    ) -> None:
        """analyze-game の実レポートを初期ロードできる (integration)．"""
        server = _make_server(
            kifu_path=MINI_CSA,
            report_path=real_report_path,
        )
        assert server.initial_view is not None
        assert server.initial_view.report is not None
        # レポートが本譜ノードのキャッシュに取り込まれる
        assert server.initial_tree is not None
        assert server.initial_tree.nodes[0].analysis is not None
        demo = server.create_demo()
        assert len(demo.blocks) > 0

    def test_report_without_kifu_raises(
        self, real_report_path: Path
    ) -> None:
        """レポートのみの指定はエラー．"""
        with pytest.raises(ValueError, match="input-path"):
            _make_server(report_path=real_report_path)

    def test_render_node_without_view(self) -> None:
        """棋譜未読込時はプレースホルダを返す．"""
        server = _make_server()
        outputs = server._render_node(
            None, None, ClickState(), *OPTS
        )
        assert len(outputs) == 13
        board = outputs[2]
        assert "棋譜が読み込まれていません" in board
        assert outputs[4] == []  # candidates

    def test_render_node_with_view(
        self, real_report_path: Path
    ) -> None:
        """読込済みの状態で盤面 SVG (クリック標的付き) と候補手を返す．"""
        server = _make_server(
            kifu_path=MINI_CSA,
            report_path=real_report_path,
        )
        outputs = server._render_node(
            server.initial_view,
            server.initial_tree,
            ClickState(),
            *OPTS,
        )
        board = outputs[2]
        assert "<svg" in board
        assert 'data-click="sq:' in board  # interactive
        assert outputs[4]  # candidates
        assert outputs[5].endswith("b - 1")  # sfen
        assert outputs[6].startswith("position sfen")
        assert "本譜" in outputs[8]  # breadcrumb

    def test_on_slider_clamps(self) -> None:
        """範囲外の ply は末尾に丸められる．"""
        server = _make_server(kifu_path=MINI_CSA)
        tree = server.initial_tree
        outputs = server._on_slider(
            server.initial_view,
            tree,
            ClickState(),
            *OPTS,
            999,
        )
        assert tree is not None
        assert tree.current_id == tree.mainline_ids[-1]
        assert "<svg" in outputs[2]

    def test_play_dropdown_creates_branch(self) -> None:
        """本譜と違う手を指すと分岐が生まれる．"""
        server = _make_server(kifu_path=MINI_CSA)
        view, tree = server.initial_view, server.initial_tree
        assert view is not None and tree is not None
        # mini.csa の初手は 7g7f — 別の合法手 2g2f で分岐
        outputs = server._on_play_dropdown(
            view, tree, ClickState(), *OPTS, "2g2f"
        )
        node = tree.nodes[tree.current_id]
        assert node.move_usi == "2g2f"
        assert not node.is_mainline
        assert "▶" in outputs[8]  # breadcrumb に分岐
        # スライダーは分岐中は更新しない (gr.update() no-op)
        # (最後の要素がスライダー更新)
        assert len(outputs) == 14

    def test_play_dropdown_reuses_mainline(self) -> None:
        """本譜と同じ手は本譜ノードを再利用する．"""
        server = _make_server(kifu_path=MINI_CSA)
        view, tree = server.initial_view, server.initial_tree
        assert view is not None and tree is not None
        server._on_play_dropdown(
            view, tree, ClickState(), *OPTS, "7g7f"
        )
        node = tree.nodes[tree.current_id]
        assert node.is_mainline
        assert node.node_id == tree.mainline_ids[1]

    def test_play_illegal_move_raises(self) -> None:
        """非合法手はエラー．"""
        server = _make_server(kifu_path=MINI_CSA)
        view, tree = server.initial_view, server.initial_tree
        with pytest.raises(gr.Error):
            server._on_play_dropdown(
                view, tree, ClickState(), *OPTS, "1a1b"
            )

    def test_board_click_two_step(self) -> None:
        """盤面クリック 2 回で 1 手進む (7g 選択 → 7f 確定)．"""
        server = _make_server(kifu_path=MINI_CSA)
        view, tree = server.initial_view, server.initial_tree
        assert view is not None and tree is not None
        # 7g = column-major (7-1)*9 + (7-1) = 60
        outputs = server._apply_board_click(
            view, tree, ClickState(), "sq:60", *OPTS
        )
        click = outputs[1]
        assert click.selected == "sq:60"
        assert "選択中" in outputs[10]
        # 7f = (7-1)*9 + (6-1) = 59
        outputs = server._apply_board_click(
            view, tree, click, "sq:59", *OPTS
        )
        node = tree.nodes[tree.current_id]
        assert node.move_usi == "7g7f"
        assert outputs[1].selected is None

    def test_back_to_mainline(self) -> None:
        """本譜へ戻るで分岐点の本譜ノードに復帰する．"""
        server = _make_server(kifu_path=MINI_CSA)
        view, tree = server.initial_view, server.initial_tree
        assert view is not None and tree is not None
        server._on_play_dropdown(
            view, tree, ClickState(), *OPTS, "2g2f"
        )
        assert not tree.nodes[tree.current_id].is_mainline
        server._on_back_mainline(
            view, tree, ClickState(), *OPTS
        )
        assert tree.current_id == tree.mainline_ids[0]

    def test_prev_next_walk_tree(self) -> None:
        """前/次は分岐中も木構造を辿る．"""
        server = _make_server(kifu_path=MINI_CSA)
        view, tree = server.initial_view, server.initial_tree
        assert view is not None and tree is not None
        server._on_play_dropdown(
            view, tree, ClickState(), *OPTS, "2g2f"
        )
        branch_id = tree.current_id
        server._on_prev(view, tree, ClickState(), *OPTS)
        assert tree.current_id == tree.mainline_ids[0]
        # 次は本譜の子を優先する
        server._on_next(view, tree, ClickState(), *OPTS)
        assert tree.current_id == tree.mainline_ids[1]
        assert tree.current_id != branch_id

    def test_analyze_current_and_cache(self) -> None:
        """1 局面解析がキャッシュされ，再解析なしでは再実行しない．"""
        server = _make_server(kifu_path=MINI_CSA)
        view, tree = server.initial_view, server.initial_tree
        assert view is not None and tree is not None
        outputs = server._analyze_current(
            view,
            tree,
            ClickState(),
            *OPTS,
            "playouts",
            8,
            force=False,
        )
        node = tree.nodes[tree.current_id]
        assert node.analysis is not None
        assert node.analysis["played_move"] == "7g7f"
        assert "解析完了" in outputs[-1]
        assert "mock" in outputs[-1]  # mock 明示
        # 2 回目はキャッシュ利用
        cached = node.analysis
        outputs = server._analyze_current(
            view,
            tree,
            ClickState(),
            *OPTS,
            "playouts",
            8,
            force=False,
        )
        assert node.analysis is cached
        assert "解析済み" in outputs[-1]
        # force で上書き
        outputs = server._analyze_current(
            view,
            tree,
            ClickState(),
            *OPTS,
            "playouts",
            8,
            force=True,
        )
        assert node.analysis is not cached

    def test_analyze_branch_node(self) -> None:
        """分岐ノードの解析は played_move なしの記録になる．"""
        server = _make_server(kifu_path=MINI_CSA)
        view, tree = server.initial_view, server.initial_tree
        assert view is not None and tree is not None
        server._on_play_dropdown(
            view, tree, ClickState(), *OPTS, "2g2f"
        )
        server._analyze_current(
            view,
            tree,
            ClickState(),
            *OPTS,
            "playouts",
            8,
            force=False,
        )
        node = tree.nodes[tree.current_id]
        assert node.analysis is not None
        assert node.analysis["played_move"] is None
        assert node.analysis["match"] is False
        assert node.analysis["candidates"]

    def test_analyze_all_generates_report(self) -> None:
        """全局面解析が analyze-game 互換レポートを生成する．"""
        server = _make_server(kifu_path=MINI_CSA)
        view, tree = server.initial_view, server.initial_tree
        assert view is not None and tree is not None
        results = list(
            server._on_analyze_all(
                view,
                tree,
                ClickState(),
                *OPTS,
                "playouts",
                8,
            )
        )
        # 開始 + 4 局面 + 最終 = 6 yield
        assert len(results) == 6
        final = results[-1]
        new_view = final[0]
        assert new_view.report is not None
        assert (
            len(new_view.report["positions"])
            == view.document.n_moves
        )
        assert "summary" in new_view.report
        assert new_view.report["input"]["path"] == "mini.csa"
        # 本譜ノードにもキャッシュが載る
        assert all(
            tree.nodes[nid].analysis is not None
            for nid in tree.mainline_ids[:-1]
        )
        # ダウンロードファイルが生成される
        download_update = final[-2]
        report_path = Path(download_update["value"])
        assert report_path.exists()
        saved = json.loads(
            report_path.read_text(encoding="utf-8")
        )
        assert len(saved["positions"]) == 4
        assert "完了" in final[-1]

    def test_analyze_all_cancel(self) -> None:
        """キャンセルで全局面解析が途中終了する．"""
        server = _make_server(kifu_path=MINI_CSA)
        view, tree = server.initial_view, server.initial_tree
        assert view is not None and tree is not None
        gen = server._on_analyze_all(
            view, tree, ClickState(), *OPTS, "playouts", 8
        )
        next(gen)  # 開始
        next(gen)  # 1 局面目
        server._cancel_event.set()
        results = list(gen)
        assert "キャンセル" in results[-1][-1]
        # レポートは生成されない (view 更新なし)
        assert view.report is None

    def test_on_load(self, real_report_path: Path) -> None:
        """_on_load が状態と全出力を返す．"""
        server = _make_server()
        outputs = server._on_load(
            str(MINI_CSA),
            str(real_report_path),
            *OPTS,
        )
        assert len(outputs) == 18
        view = outputs[0]
        assert view is not None
        assert view.document.n_moves == 4
        assert view.report is not None
        tree = outputs[1]
        assert tree is not None
        assert len(tree.mainline_ids) == 5
        # 読み込みステータス (末尾)
        assert "4 手" in outputs[-1]

    def test_on_load_without_kifu_raises(self) -> None:
        """棋譜ファイル未指定の読み込みはエラー．"""
        server = _make_server()
        with pytest.raises(gr.Error):
            server._on_load(None, None, *OPTS)


class TestHelpers:
    """モジュールヘルパーのテスト．"""

    def test_clamp_ply_without_view(self) -> None:
        assert _clamp_ply(None, 5) == 0

    def test_file_path(self) -> None:
        assert _file_path(None) is None
        assert _file_path("/tmp/a.csa") == Path("/tmp/a.csa")

    def test_budget_conversion(self) -> None:
        server = _make_server(default_time_ms=500)
        assert server._budget("time", 800) == (800, None)
        assert server._budget("playouts", 32) == (None, 32)
        # 不正値はデフォルト時間予算
        assert server._budget("time", None) == (500, None)
        assert server._budget("playouts", 0) == (500, None)
