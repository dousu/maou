"""analysis_gui_server (棋譜解析 GUI サーバー) のテスト．

ワークベンチは gr.HTML 1 枚なので，ハンドラの検証は「分岐木の状態が
どう動いたか」と「返る HTML に何が出ているか」の 2 点で行う．
"""

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("gradio")

import gradio as gr

from maou.app.analysis.game_analyzer import (
    FixedPlayoutsAllocator,
    GameAnalyzer,
)
from maou.app.analysis.interactive_analyzer import (
    EngineSettings,
)
from maou.infra.visualization.analysis_gui_server import (
    AnalysisGuiServer,
    _clamp_ply,
    _file_path,
    _head_scripts,
    _make_bridge,
)
from maou.interface.analysis_gui import ClickState
from maou.interface.analysis_workbench import (
    AnalysisProgress,
    WorkbenchOptions,
)

RESOURCES = (
    Path(__file__).parents[2] / "app" / "analysis" / "resources"
)
MINI_CSA = RESOURCES / "mini.csa"

# mock 評価器で高速に回すテスト用エンジン設定
FAST_ENGINE = EngineSettings(
    root_dfpn=False, leaf_mate=False, num_candidates=5
)


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


def _act(
    server: AnalysisGuiServer,
    action: str,
    *,
    click: ClickState | None = None,
    options: WorkbenchOptions | None = None,
    progress: AnalysisProgress | None = None,
) -> tuple[Any, ...]:
    """初期セッションに対してアクションを 1 つ適用する．"""
    return server._on_action(
        action,
        server.initial_view,
        server.initial_tree,
        click,
        options or server.initial_options,
        progress or AnalysisProgress(),
    )


class TestConstruction:
    """初期化と demo 構築のテスト．"""

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
        assert len(server.create_demo().blocks) > 0

    def test_demo_builds_with_report(
        self, real_report_path: Path
    ) -> None:
        """analyze-game の実レポートを初期ロードできる (integration)．"""
        server = _make_server(
            kifu_path=MINI_CSA, report_path=real_report_path
        )
        assert server.initial_view is not None
        assert server.initial_view.report is not None
        # レポートが本譜ノードのキャッシュに取り込まれる
        assert server.initial_tree is not None
        assert server.initial_tree.nodes[0].analysis is not None
        assert len(server.create_demo().blocks) > 0

    def test_report_without_kifu_raises(
        self, real_report_path: Path
    ) -> None:
        """レポートのみの指定はエラー．"""
        with pytest.raises(ValueError, match="input-path"):
            _make_server(report_path=real_report_path)

    def test_playouts_default_selects_playout_budget(
        self,
    ) -> None:
        """--playouts 指定時は playouts 予算が初期選択される．"""
        server = _make_server(default_playouts=32)
        assert server.initial_options.budget_mode == "playouts"
        assert server.initial_options.budget_value == 32
        assert server._budget(server.initial_options) == (
            None,
            32,
        )


class TestRendering:
    """ワークベンチ HTML のテスト．"""

    def test_render_without_view(self) -> None:
        """棋譜未読込時はプレースホルダを返す．"""
        server = _make_server()
        html = server._render(
            None,
            None,
            None,
            server.initial_options,
            AnalysisProgress(),
        )
        assert "棋譜が読み込まれていません" in html
        assert 'id="maou-workbench"' in html

    def test_render_with_report(
        self, real_report_path: Path
    ) -> None:
        """読込済みの状態で盤面・棋譜・グラフ・候補手が出る．"""
        server = _make_server(
            kifu_path=MINI_CSA, report_path=real_report_path
        )
        html = server._render(
            server.initial_view,
            server.initial_tree,
            ClickState(),
            server.initial_options,
            AnalysisProgress(total=4, done=4),
        )
        assert "<svg" in html
        assert 'data-click="sq:' in html  # 盤面クリック標的
        assert "mw-kifu-row" in html
        assert "polyline" in html  # 評価値グラフ
        assert "mw-candidate" in html
        assert "position sfen" in html

    def test_render_stamp_advances(self) -> None:
        """再描画ごとに data-render の刻印が変わる (JS の追従用)．"""
        server = _make_server(kifu_path=MINI_CSA)
        first = server._render(
            server.initial_view,
            server.initial_tree,
            None,
            server.initial_options,
            AnalysisProgress(),
        )
        second = server._render(
            server.initial_view,
            server.initial_tree,
            None,
            server.initial_options,
            AnalysisProgress(),
        )
        assert first != second

    def test_head_scripts_include_assets(self) -> None:
        """head 注入に CSS と JS が含まれる．"""
        head = _head_scripts()
        assert "#maou-workbench" in head
        assert "__maou_wb" in head


class TestNavigationActions:
    """ナビゲーション系アクションのテスト．"""

    def test_goto_and_bounds(self) -> None:
        """goto は範囲外を末尾に丸める．"""
        server = _make_server(kifu_path=MINI_CSA)
        tree = server.initial_tree
        assert tree is not None
        _act(server, "goto:999")
        assert tree.current_id == tree.mainline_ids[-1]
        _act(server, "goto:0")
        assert tree.current_id == tree.mainline_ids[0]

    def test_prev_next_walk_tree(self) -> None:
        """前/次は分岐中も木構造を辿る．"""
        server = _make_server(kifu_path=MINI_CSA)
        tree = server.initial_tree
        assert tree is not None
        _act(server, "play:2g2f")
        branch_id = tree.current_id
        _act(server, "nav:prev")
        assert tree.current_id == tree.mainline_ids[0]
        # 次は本譜の子を優先する
        _act(server, "nav:next")
        assert tree.current_id == tree.mainline_ids[1]
        assert tree.current_id != branch_id

    def test_first_and_last(self) -> None:
        """最初へ / 最後へが本譜の端に移動する．"""
        server = _make_server(kifu_path=MINI_CSA)
        tree = server.initial_tree
        assert tree is not None
        _act(server, "nav:last")
        assert tree.current_id == tree.mainline_ids[-1]
        _act(server, "nav:first")
        assert tree.current_id == tree.mainline_ids[0]

    def test_back_to_mainline(self) -> None:
        """本譜へ戻るで分岐点の本譜ノードに復帰する．"""
        server = _make_server(kifu_path=MINI_CSA)
        tree = server.initial_tree
        assert tree is not None
        _act(server, "play:2g2f")
        assert not tree.nodes[tree.current_id].is_mainline
        _act(server, "nav:mainline")
        assert tree.current_id == tree.mainline_ids[0]

    def test_blunder_jump_reports_when_absent(self) -> None:
        """悪手がないときはステータスで知らせる．"""
        server = _make_server(kifu_path=MINI_CSA)
        _, _, _, progress, _ = _act(server, "blunder:next")
        assert "悪手はありません" in progress.status
        assert progress.is_error


class TestMoveInput:
    """指し手入力 (分岐) のテスト．"""

    def test_play_creates_branch(self) -> None:
        """本譜と違う手を指すと分岐が生まれる．"""
        server = _make_server(kifu_path=MINI_CSA)
        tree = server.initial_tree
        assert tree is not None
        # mini.csa の初手は 7g7f — 別の合法手 2g2f で分岐
        _, _, _, _, html = _act(server, "play:2g2f")
        node = tree.nodes[tree.current_id]
        assert node.move_usi == "2g2f"
        assert not node.is_mainline
        assert "分岐" in html

    def test_play_reuses_mainline(self) -> None:
        """本譜と同じ手は本譜ノードを再利用する．"""
        server = _make_server(kifu_path=MINI_CSA)
        tree = server.initial_tree
        assert tree is not None
        _act(server, "play:7g7f")
        node = tree.nodes[tree.current_id]
        assert node.is_mainline
        assert node.node_id == tree.mainline_ids[1]

    def test_illegal_move_sets_error_status(self) -> None:
        """非合法手はステータスに出す (画面は落とさない)．"""
        server = _make_server(kifu_path=MINI_CSA)
        _, _, _, progress, _ = _act(server, "play:1a1b")
        assert progress.is_error
        assert "指せません" in progress.status

    def test_board_click_two_step(self) -> None:
        """盤面クリック 2 回で 1 手進む (7g 選択 → 7f 確定)．"""
        server = _make_server(kifu_path=MINI_CSA)
        tree = server.initial_tree
        assert tree is not None
        # 7g = column-major (7-1)*9 + (7-1) = 60
        _, click, _, _, html = _act(server, "board:sq:60")
        assert click.selected == "sq:60"
        assert "選択中" in html
        # 7f = (7-1)*9 + (6-1) = 59
        _, click, _, _, _ = _act(
            server, "board:sq:59", click=click
        )
        assert tree.nodes[tree.current_id].move_usi == "7g7f"
        assert click.selected is None

    def test_candidate_branch_without_analysis(self) -> None:
        """未解析局面の候補手ショートカットは理由を返す．"""
        server = _make_server(kifu_path=MINI_CSA)
        _, _, _, progress, _ = _act(server, "cand:0")
        assert "候補手はありません" in progress.status

    def test_pv_without_analysis(self) -> None:
        """解析 PV がない局面の PV 再生は理由を返す．"""
        server = _make_server(kifu_path=MINI_CSA)
        _, _, _, progress, _ = _act(server, "pv")
        assert "解析 PV がありません" in progress.status


class TestOptionActions:
    """表示・解析オプションのテスト．"""

    def test_option_updates(self) -> None:
        """opt:* が表示オプションに反映される．"""
        server = _make_server(kifu_path=MINI_CSA)
        _, _, options, _, _ = _act(server, "opt:threshold:0.25")
        assert options.threshold == pytest.approx(0.25)
        _, _, options, _, _ = _act(
            server, "opt:ymode:eval_cp", options=options
        )
        assert options.y_mode == "eval_cp"
        _, _, options, _, _ = _act(
            server, "opt:budgetmode:playouts", options=options
        )
        assert options.budget_mode == "playouts"
        _, _, options, _, _ = _act(
            server, "opt:arrows:0", options=options
        )
        assert options.show_arrows is False

    def test_topn_is_clamped_to_num_candidates(self) -> None:
        """候補手数は CLI の上限を超えない．"""
        server = _make_server(
            kifu_path=MINI_CSA, num_candidates=3
        )
        _, _, options, _, _ = _act(server, "opt:topn:99")
        assert options.top_n == 3

    def test_invalid_option_value_is_ignored(self) -> None:
        """不正な値は無視して現状を保つ．"""
        server = _make_server(kifu_path=MINI_CSA)
        _, _, options, _, _ = _act(server, "opt:budget:abc")
        assert (
            options.budget_value
            == server.initial_options.budget_value
        )

    def test_toggle_legal_panel(self) -> None:
        """L トグルで合法手一覧が開閉する．"""
        server = _make_server(kifu_path=MINI_CSA)
        _, _, options, _, html = _act(server, "toggle:legal")
        assert options.legal_open
        assert "mw-legal" in html


class TestAnalysis:
    """解析レーンのテスト．"""

    def _analyze(
        self,
        server: AnalysisGuiServer,
        action: str,
        progress: Any,
    ) -> tuple[Any, ...]:
        return server._on_analyze(
            action,
            server.initial_view,
            server.initial_tree,
            None,
            server.initial_options,
            progress,
        )

    def test_analyze_current_and_cache(self) -> None:
        """1 局面解析がキャッシュされ，再解析なしでは再実行しない．"""
        server = _make_server(
            kifu_path=MINI_CSA, default_playouts=8
        )
        tree = server.initial_tree
        assert tree is not None
        _, progress, _ = self._analyze(
            server, "analyze", AnalysisProgress()
        )
        node = tree.nodes[tree.current_id]
        assert node.analysis is not None
        assert node.analysis["played_move"] == "7g7f"
        assert "解析完了" in progress.status
        # 2 回目はキャッシュ利用
        cached = node.analysis
        _, progress, _ = self._analyze(
            server, "analyze", progress
        )
        assert node.analysis is cached
        assert "解析済み" in progress.status
        # 再解析で上書き
        _, progress, html = self._analyze(
            server, "reanalyze", progress
        )
        assert node.analysis is not cached
        assert "mw-candidate" in html

    def test_analyze_without_view(self) -> None:
        """棋譜未読込の解析はステータスで知らせる．"""
        server = _make_server()
        _, progress, _ = server._on_analyze(
            "analyze",
            None,
            None,
            None,
            server.initial_options,
            AnalysisProgress(),
        )
        assert progress.is_error
        assert "棋譜を読み込んでください" in progress.status

    def test_analyze_branch_node(self) -> None:
        """分岐ノードの解析は played_move なしの記録になる．"""
        server = _make_server(
            kifu_path=MINI_CSA, default_playouts=8
        )
        tree = server.initial_tree
        assert tree is not None
        _act(server, "play:2g2f")
        self._analyze(server, "analyze", AnalysisProgress())
        node = tree.nodes[tree.current_id]
        assert node.analysis is not None
        assert node.analysis["played_move"] is None
        assert node.analysis["match"] is False
        assert node.analysis["candidates"]

    def test_analyze_shows_evaluation_line(self) -> None:
        """解析後は現局面のエンジン評価行が候補手パネルに出る．"""
        server = _make_server(
            kifu_path=MINI_CSA, default_playouts=8
        )
        _, _, html = self._analyze(
            server, "analyze", AnalysisProgress()
        )
        assert "この局面のエンジン評価" in html
        assert "先手視点" in html

    def test_candidate_tooltip_carries_prior(self) -> None:
        """prior と確定値は列ではなく行のホバーで拾える．"""
        server = _make_server(
            kifu_path=MINI_CSA, default_playouts=8
        )
        _, _, html = self._analyze(
            server, "analyze", AnalysisProgress()
        )
        assert "prior " in html
        assert 'class="mw-candidate' in html

    def test_analyze_all_generates_report(self) -> None:
        """全局面解析が analyze-game 互換レポートを生成する．"""
        server = _make_server(
            kifu_path=MINI_CSA, default_playouts=8
        )
        view, tree = server.initial_view, server.initial_tree
        assert view is not None and tree is not None
        results = list(
            server._on_analyze_all(
                view,
                tree,
                None,
                server.initial_options,
                AnalysisProgress(),
            )
        )
        # 開始 + 4 局面 + 最終 = 6 yield
        assert len(results) == 6
        new_view, progress, html, report_path = results[-1]
        assert new_view.report is not None
        assert (
            len(new_view.report["positions"])
            == view.document.n_moves
        )
        assert new_view.report["input"]["path"] == "mini.csa"
        assert not progress.running
        assert "完了" in progress.status
        assert "polyline" in html  # グラフが出るようになる
        # 本譜ノードにもキャッシュが載る
        assert all(
            tree.nodes[nid].analysis is not None
            for nid in tree.mainline_ids[:-1]
        )
        saved = json.loads(
            Path(report_path).read_text(encoding="utf-8")
        )
        assert len(saved["positions"]) == 4

    def test_analyze_all_cancel(self) -> None:
        """キャンセルで全局面解析が途中終了する．"""
        server = _make_server(
            kifu_path=MINI_CSA, default_playouts=8
        )
        view, tree = server.initial_view, server.initial_tree
        assert view is not None and tree is not None
        gen = server._on_analyze_all(
            view,
            tree,
            None,
            server.initial_options,
            AnalysisProgress(),
        )
        next(gen)  # 開始
        next(gen)  # 1 局面目
        server._cancel_event.set()
        results = list(gen)
        assert "キャンセル" in results[-1][1].status
        # レポートは生成されない (view 更新なし)
        assert view.report is None

    def test_analyze_all_failure_clears_running(self) -> None:
        """解析が例外で落ちても running を戻し，やり直せる状態にする．"""
        server = _make_server(
            kifu_path=MINI_CSA, default_playouts=8
        )
        view, tree = server.initial_view, server.initial_tree
        assert view is not None and tree is not None

        def _boom(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("engine died")
            yield  # pragma: no cover - ジェネレータにするためだけ

        server._analyzer.analyze_mainline = _boom  # type: ignore[method-assign]
        results = list(
            server._on_analyze_all(
                view,
                tree,
                None,
                server.initial_options,
                AnalysisProgress(),
            )
        )
        _, progress, html, _ = results[-1]
        assert not progress.running
        assert progress.is_error
        assert "engine died" in progress.status
        # 「キャンセル」ではなく「開始」に戻っている
        assert 'data-action="analyzeall"' in html

    def test_cancel_sets_event(self) -> None:
        """キャンセルレーンはフラグを立ててステータスを返す．"""
        server = _make_server(kifu_path=MINI_CSA)
        progress, html = server._on_cancel(
            server.initial_view,
            server.initial_tree,
            None,
            server.initial_options,
            AnalysisProgress(),
        )
        assert server._cancel_event.is_set()
        assert "キャンセル" in progress.status
        assert 'id="maou-workbench"' in html

    def test_save_without_report(self) -> None:
        """レポートがなければ保存できないと知らせる．"""
        server = _make_server(kifu_path=MINI_CSA)
        progress, _, _ = server._on_save(
            server.initial_view,
            server.initial_tree,
            None,
            server.initial_options,
            AnalysisProgress(),
        )
        assert progress.is_error
        assert "保存できるレポート" in progress.status

    def test_save_writes_report(
        self, real_report_path: Path
    ) -> None:
        """読込済みレポートを JSON に書き出せる．"""
        server = _make_server(
            kifu_path=MINI_CSA, report_path=real_report_path
        )
        progress, _, path = server._on_save(
            server.initial_view,
            server.initial_tree,
            None,
            server.initial_options,
            AnalysisProgress(),
        )
        assert not progress.is_error
        saved = json.loads(
            Path(path).read_text(encoding="utf-8")
        )
        assert saved["positions"]


class TestLoad:
    """読み込みのテスト．"""

    def test_on_load(self, real_report_path: Path) -> None:
        """_on_load が状態とワークベンチを作り直す．"""
        server = _make_server()
        view, tree, click, progress, html = server._on_load(
            str(MINI_CSA),
            str(real_report_path),
            server.initial_options,
        )
        assert view is not None
        assert view.document.n_moves == 4
        assert view.report is not None
        assert tree is not None
        assert len(tree.mainline_ids) == 5
        assert click.selected is None
        assert "4 手" in progress.status
        assert "mw-kifu-row" in html

    def test_on_load_without_kifu_raises(self) -> None:
        """棋譜ファイル未指定の読み込みはエラー．"""
        server = _make_server()
        with pytest.raises(gr.Error):
            server._on_load(None, None, server.initial_options)


class TestHelpers:
    """モジュールヘルパーのテスト．"""

    def test_clamp_ply_without_view(self) -> None:
        assert _clamp_ply(None, 5) == 0

    def test_clamp_ply_with_garbage(self) -> None:
        server = _make_server(kifu_path=MINI_CSA)
        assert _clamp_ply(server.initial_view, "x") == 0
        assert _clamp_ply(server.initial_view, "99") == 4

    def test_file_path(self) -> None:
        assert _file_path(None) is None
        assert _file_path("/tmp/a.csa") == Path("/tmp/a.csa")

    def test_budget_conversion(self) -> None:
        server = _make_server(default_time_ms=500)
        options = WorkbenchOptions(
            budget_mode="time", budget_value=800
        )
        assert server._budget(options) == (800, None)
        assert server._budget(
            WorkbenchOptions(
                budget_mode="playouts", budget_value=32
            )
        ) == (None, 32)
        # 非正の値はデフォルト時間予算
        assert server._budget(
            WorkbenchOptions(budget_mode="time", budget_value=0)
        ) == (500, None)

    def test_bridge_buffers_action(self) -> None:
        """ブリッジはアクションをバッファに積んで True を返す．"""
        buffer: dict[str, str] = {}
        handle = _make_bridge(buffer)
        assert handle("nav:next") is True
        assert buffer["value"] == "nav:next"
        assert handle("") is False
