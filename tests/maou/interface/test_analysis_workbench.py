"""analysis_workbench (ワークベンチ HTML 組み立て) のテスト．"""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from maou.app.analysis.analysis_session import (
    advance_move,
    build_variation_tree,
    goto_node,
)
from maou.interface.analysis_gui import (
    ClickState,
    SessionView,
    load_session,
)
from maou.interface.analysis_workbench import (
    AnalysisProgress,
    WorkbenchOptions,
    blunder_target_ply,
    blunders,
    neighbouring_blunder,
    render_workbench,
)

RESOURCES = (
    Path(__file__).parents[1] / "app" / "analysis" / "resources"
)
MINI_CSA = RESOURCES / "mini.csa"

OPTIONS = WorkbenchOptions()
PROGRESS = AnalysisProgress()
CLICK = ClickState()

# mini.csa の本譜 (▲7六歩 △3四歩 ▲2六歩 △8四歩)
MINI_MOVES = ["7g7f", "3c3d", "2g2f", "8c8d"]


def _view(report: dict[str, Any] | None = None) -> SessionView:
    """mini.csa の SessionView を作る (任意でレポート付き)．"""
    return load_session(
        MINI_CSA.read_bytes(),
        MINI_CSA.name,
        json.dumps(report) if report is not None else None,
    )


def _position(
    ply: int,
    *,
    side: str,
    loss: float | None,
    mate: bool = False,
) -> dict[str, Any]:
    """レポートの positions 要素を最小構成で作る．

    ``played_move`` は棋譜と一致していないと ``validate_report`` が
    弾くため，mini.csa の実際の指し手を入れる．
    """
    return {
        "ply": ply,
        "side_to_move": side,
        "played_move": MINI_MOVES[ply - 1],
        "best_move": None,
        "match": False,
        "winrate": 0.5,
        "eval_cp": 0.0,
        "winrate_loss": loss,
        "pv": [],
        "candidates": [],
        "mate_found": mate,
        "playouts": 1,
        "elapsed_ms": 1,
    }


@pytest.fixture
def report() -> dict[str, Any]:
    """4 手ぶんの合成レポート (2 手目と 4 手目が大きく損している)．"""
    return {
        "input": {"path": "mini.csa"},
        "positions": [
            _position(1, side="b", loss=0.01),
            _position(2, side="w", loss=0.30),
            _position(3, side="b", loss=None),
            _position(4, side="w", loss=0.12, mate=True),
        ],
        "summary": {
            "match_rate": {"black": 0.5, "white": 0.25},
            "mean_winrate_loss": {"black": 0.01, "white": 0.21},
            "worst_moves": [],
            "mates_found": [{"ply": 4, "side": "w"}],
        },
    }


class TestBlunders:
    """悪手抽出のテスト．"""

    def test_without_report(self) -> None:
        """レポート未読込では空リスト．"""
        assert blunders(_view(), 0.1) == []

    def test_filters_and_sorts_by_loss(
        self, report: dict[str, Any]
    ) -> None:
        """しきい値以上を損失の降順で返す．"""
        found = blunders(_view(report), 0.1)
        assert [ply for ply, _, _ in found] == [2, 4]
        assert found[0][2] == pytest.approx(0.30)
        # 日本語表記が付く
        assert found[0][1].startswith("△")

    def test_threshold_is_inclusive(
        self, report: dict[str, Any]
    ) -> None:
        """しきい値ちょうどの手も含む．"""
        assert len(blunders(_view(report), 0.30)) == 1
        assert blunders(_view(report), 0.31) == []

    def test_none_loss_is_skipped(
        self, report: dict[str, Any]
    ) -> None:
        """winrate_loss が null の手は対象外．"""
        assert 3 not in [
            ply for ply, _, _ in blunders(_view(report), 0.0)
        ]

    def test_target_ply_is_before_the_move(self) -> None:
        """ジャンプ先は «その手を指す直前» の局面．"""
        assert blunder_target_ply(2) == 1
        assert blunder_target_ply(0) == 0

    def test_neighbouring_blunder(
        self, report: dict[str, Any]
    ) -> None:
        """現在位置から前/次の悪手を辿れる．"""
        view = _view(report)
        # 悪手は 2 手目 (target 1) と 4 手目 (target 3)
        assert (
            neighbouring_blunder(view, 0.1, 0, forward=True)
            == 1
        )
        assert (
            neighbouring_blunder(view, 0.1, 1, forward=True)
            == 3
        )
        assert (
            neighbouring_blunder(view, 0.1, 3, forward=True)
            is None
        )
        assert (
            neighbouring_blunder(view, 0.1, 3, forward=False)
            == 1
        )
        assert (
            neighbouring_blunder(view, 0.1, 0, forward=False)
            is None
        )


class TestRender:
    """ワークベンチ HTML のテスト．"""

    def _render(
        self,
        view: SessionView | None,
        *,
        options: WorkbenchOptions = OPTIONS,
        click: ClickState = CLICK,
        progress: AnalysisProgress = PROGRESS,
    ) -> str:
        tree = (
            build_variation_tree(view.document, view.report)
            if view is not None
            else None
        )
        return render_workbench(
            view,
            tree,
            click,
            options,
            progress,
            engine_label="mock 評価器",
            is_mock=True,
        )

    def test_placeholder_without_view(self) -> None:
        """未読込はプレースホルダ (ヘッダーは出す)．"""
        html = self._render(None)
        assert "棋譜が読み込まれていません" in html
        assert "mw-header" in html
        assert "mw-main" not in html

    def test_sections_present(
        self, report: dict[str, Any]
    ) -> None:
        """3 カラムの各セクションが揃う．"""
        html = self._render(_view(report))
        for marker in (
            "mw-header",
            "mw-summary",
            "mw-main",
            "mw-rail-left",
            "mw-center",
            "mw-rail-right",
            "mw-kifu-row",
            "mw-blunder",
            "mw-graph",
            "mw-analysis",
        ):
            assert marker in html, marker

    def test_summary_uses_report(
        self, report: dict[str, Any]
    ) -> None:
        """サマリ帯に一致率・平均損失・詰みが出る．"""
        html = self._render(_view(report))
        assert "50.0%" in html  # 先手一致率
        assert "0.210" in html  # 後手平均勝率損失
        assert "詰み発見" in html

    def test_summary_without_report(self) -> None:
        """レポート未読込は案内を出す．"""
        html = self._render(_view())
        assert "レポート未読込" in html

    def test_threshold_controls_blunder_list(
        self, report: dict[str, Any]
    ) -> None:
        """しきい値を上げると悪手リストが減る．"""
        view = _view(report)
        loose = self._render(
            view, options=WorkbenchOptions(threshold=0.1)
        )
        strict = self._render(
            view, options=WorkbenchOptions(threshold=0.5)
        )
        assert loose.count('class="mw-blunder"') > strict.count(
            'class="mw-blunder"'
        )
        assert "しきい値以上の手はありません" in strict

    def test_graph_only_with_report(
        self, report: dict[str, Any]
    ) -> None:
        """グラフはレポートがあるときだけ折れ線を描く．"""
        assert "polyline" in self._render(_view(report))
        assert "polyline" not in self._render(_view())

    def test_escapes_player_names(self) -> None:
        """対局者名などの外部由来文字列をエスケープする．"""
        view = _view()
        hostile = replace(
            view.document,
            names=['<img src=x onerror="alert(1)">', None],
        )
        html = render_workbench(
            SessionView(
                document=hostile,
                report=None,
                move_labels=view.move_labels,
                source_name='"><script>bad()</script>',
            ),
            build_variation_tree(hostile, None),
            ClickState(),
            OPTIONS,
            PROGRESS,
            engine_label="mock",
            is_mock=True,
        )
        assert "<script>bad()" not in html
        assert "<img" not in html
        assert "&lt;script&gt;" in html
        assert "&lt;img" in html

    def test_current_move_is_marked(
        self, report: dict[str, Any]
    ) -> None:
        """現在の手数の棋譜行に is-current が付く．"""
        view = _view(report)
        tree = build_variation_tree(view.document, view.report)
        goto_node(tree, tree.mainline_ids[2])
        html = render_workbench(
            view,
            tree,
            ClickState(),
            OPTIONS,
            PROGRESS,
            engine_label="mock",
            is_mock=True,
        )
        assert "mw-kifu-row is-current" in html

    def test_branch_shows_breadcrumb(self) -> None:
        """分岐中はパンくずと本譜へ戻るを出す．"""
        view = _view()
        tree = build_variation_tree(view.document, None)
        advance_move(tree, "2g2f")
        html = render_workbench(
            view,
            tree,
            ClickState(),
            OPTIONS,
            PROGRESS,
            engine_label="mock",
            is_mock=True,
        )
        assert "分岐" in html
        assert "本譜へ戻る" in html
        assert 'data-action="nav:mainline"' in html

    def test_click_selection_shown(self) -> None:
        """選択中はヒット帯に選択状態を出す．"""
        html = self._render(
            _view(), click=ClickState(selected="sq:60")
        )
        assert "選択中" in html
        assert "is-active" in html

    def test_promotion_prompt(self) -> None:
        """成/不成の保留中は確認ボタンを出す．"""
        html = self._render(
            _view(),
            click=ClickState(
                selected="sq:60", pending_usis=("7g7f+", "7g7f")
            ),
        )
        assert 'data-action="promote:1"' in html
        assert 'data-action="promote:0"' in html

    def test_legal_panel_toggle(self) -> None:
        """合法手一覧はオプションで出し分ける．"""
        assert "mw-legal" not in self._render(_view())
        assert "mw-legal" in self._render(
            _view(), options=WorkbenchOptions(legal_open=True)
        )

    def test_progress_bar(self, report: dict[str, Any]) -> None:
        """全局面解析の進捗が幅とラベルに出る．"""
        html = self._render(
            _view(report),
            progress=AnalysisProgress(
                running=True, done=2, total=4, status="解析中"
            ),
        )
        assert "width:50.0%" in html
        assert "2 / 4 局面" in html
        # 実行中はキャンセルに変わる
        assert 'data-action="cancel"' in html

    def test_error_status_styling(self) -> None:
        """エラーステータスは専用クラスで出す．"""
        html = self._render(
            _view(),
            progress=AnalysisProgress(
                status="指せません", is_error=True
            ),
        )
        assert "mw-status is-error" in html
