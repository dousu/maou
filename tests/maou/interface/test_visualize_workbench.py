"""ワークベンチ HTML 生成のテスト．

画面は gr.HTML 1 枚なので，ここが見た目とアクション符号化の唯一の
定義元になる．エスケープ・選択行・空状態・ヒストグラムの階級分けを固定する．
"""

from maou.app.visualization.record_renderer import Distribution
from maou.interface.visualize_workbench import (
    GraphData,
    RecordData,
    StatusView,
    WorkbenchState,
    render_workbench,
)


class TestRenderWorkbench:
    """render_workbench のテスト．"""

    def test_record_screen_has_the_three_rails(self) -> None:
        """レコード画面は左レール・ステージ・右レールを持つ．"""
        html = render_workbench(
            WorkbenchState(), StatusView(), record=RecordData()
        )

        assert 'id="viz-workbench"' in html
        assert html.count('class="vz-rail left"') == 1
        assert html.count('class="vz-stage"') == 1
        assert html.count('class="vz-rail right"') == 1

    def test_graph_screen_is_chosen_by_array_type(
        self,
    ) -> None:
        """array_type が game-graph ならグラフ画面を出す．"""
        html = render_workbench(
            WorkbenchState(array_type="game-graph"),
            StatusView(),
            graph=GraphData(graph_html="<div id='g'></div>"),
        )

        assert 'class="vz-graph"' in html
        assert "凡例" in html
        assert 'class="vz-winbar"' not in html

    def test_selected_type_is_marked_in_the_segment(
        self,
    ) -> None:
        """選択中のデータ型だけ on が付く．"""
        html = render_workbench(
            WorkbenchState(array_type="stage1"),
            StatusView(),
            record=RecordData(),
        )

        assert html.count('class="on" data-action="type:') == 1
        assert 'data-action="type:stage1"' in html

    def test_types_can_be_rendered_disabled(self) -> None:
        """types_enabled=False ならデータ型は押せない．"""
        html = render_workbench(
            WorkbenchState(array_type="game-graph"),
            StatusView(),
            graph=GraphData(),
            types_enabled=False,
        )

        assert 'data-action="type:' not in html
        assert html.count(" disabled>") >= 5

    def test_render_stamp_lands_on_the_root(self) -> None:
        """再描画の識別子はルート要素に載る (JS が差し替えを検出する)．"""
        html = render_workbench(
            WorkbenchState(),
            StatusView(),
            record=RecordData(),
            render_stamp="42",
        )

        assert 'data-render="42"' in html


class TestResultTable:
    """結果一覧のテスト．"""

    def test_rows_carry_click_actions(self) -> None:
        """各行に 0 始まりの data-action が付く．"""
        html = render_workbench(
            WorkbenchState(),
            StatusView(),
            record=RecordData(
                headers=["Index", "ID", "Eval"],
                rows=[[1, "a", 10], [2, "b", -3]],
            ),
        )

        assert 'data-action="row:0"' in html
        assert 'data-action="row:1"' in html

    def test_selected_row_is_marked(self) -> None:
        """selected の行だけ on が付く．"""
        html = render_workbench(
            WorkbenchState(selected=1),
            StatusView(),
            record=RecordData(
                headers=["Index", "ID"],
                rows=[[1, "a"], [2, "b"], [3, "c"]],
            ),
        )

        assert html.count('class="vz-row on"') == 1

    def test_cells_are_escaped(self) -> None:
        """セルの HTML はエスケープされる．"""
        html = render_workbench(
            WorkbenchState(),
            StatusView(),
            record=RecordData(
                headers=["ID"], rows=[["<img onerror=x>"]]
            ),
        )

        assert "<img onerror" not in html
        assert "&lt;img onerror=x&gt;" in html

    def test_extra_cells_are_truncated_to_header_width(
        self,
    ) -> None:
        """見出しより多いセルは捨てる (列ずれを防ぐ)．"""
        html = render_workbench(
            WorkbenchState(),
            StatusView(),
            record=RecordData(
                headers=["A", "B"], rows=[[1, 2, 3]]
            ),
        )

        assert ">3<" not in html


class TestEvalSearch:
    """評価値範囲検索のテスト．"""

    def test_enabled_for_hcpe(self) -> None:
        """hcpe では評価値レンジが有効になる．"""
        html = render_workbench(
            WorkbenchState(),
            StatusView(),
            record=RecordData(supports_eval_search=True),
        )

        assert "評価値範囲検索は hcpe のみ対応" not in html
        assert 'data-action-input="mineval"' in html

    def test_disabled_elsewhere_with_a_reason(self) -> None:
        """非対応の型では無効化して理由を出す．"""
        html = render_workbench(
            WorkbenchState(array_type="stage1"),
            StatusView(),
            record=RecordData(supports_eval_search=False),
        )

        assert "評価値範囲検索は hcpe のみ対応" in html
        assert "vz-in vz-m off" in html


class TestWinBar:
    """先手勝率バーのテスト．"""

    def test_absent_without_a_result_value(self) -> None:
        """result_value が無ければバーを出さない．"""
        html = render_workbench(
            WorkbenchState(),
            StatusView(),
            record=RecordData(result_value=None),
        )

        assert 'class="vz-winbar"' not in html

    def test_fill_width_follows_the_value(self) -> None:
        """バーの幅は勝率に比例する．"""
        html = render_workbench(
            WorkbenchState(array_type="preprocessing"),
            StatusView(),
            record=RecordData(result_value=0.664),
        )

        assert "width:66.4%" in html
        assert ">66%<" in html

    def test_out_of_range_values_are_clamped(self) -> None:
        """範囲外の値は 0〜100% に丸める．"""
        html = render_workbench(
            WorkbenchState(array_type="preprocessing"),
            StatusView(),
            record=RecordData(result_value=-1.0),
        )

        assert "width:0.0%" in html


class TestHistogram:
    """ヒストグラムのテスト．"""

    @staticmethod
    def _html(
        values: tuple[float, ...], current: float | None
    ) -> str:
        return render_workbench(
            WorkbenchState(),
            StatusView(),
            record=RecordData(
                rows=[[v] for v in values],
                distribution=Distribution(
                    title="評価値の分布",
                    axis_label="評価値",
                    values=values,
                ),
                current_value=current,
            ),
        )

    def test_draws_one_bar_per_bin(self) -> None:
        """階級数ぶんのバーを描く．"""
        html = self._html(
            tuple(float(i) for i in range(100)), None
        )

        assert html.count('class="vz-bar"') == 20

    def test_current_record_bin_is_accented(self) -> None:
        """現在レコードが属する階級だけ accent 色にする．"""
        html = self._html(
            tuple(float(i) for i in range(100)), 5.0
        )

        assert html.count('class="vz-bar on"') == 1
        assert html.count('class="vz-bar"') == 19

    def test_axis_labels_show_the_range(self) -> None:
        """両端に分布の下限・上限を出す．"""
        html = self._html((0.0, 100.0), None)

        assert ">0<" in html
        assert ">100<" in html

    def test_constant_series_still_renders(self) -> None:
        """全部同じ値でも幅ゼロの階級を作らない．"""
        html = self._html((5.0, 5.0, 5.0), 5.0)

        assert "vz-bar" in html
        assert "分布を出せるデータがありません" not in html

    def test_empty_distribution_shows_a_message(self) -> None:
        """分布が無いときは空状態を出す．"""
        html = render_workbench(
            WorkbenchState(),
            StatusView(),
            record=RecordData(distribution=None),
        )

        assert "分布を出せるデータがありません" in html


class TestGraphScreen:
    """グラフ画面のテスト．"""

    def test_breadcrumb_links_all_but_the_last(self) -> None:
        """末尾以外のパンくずはクリックできる．"""
        html = render_workbench(
            WorkbenchState(array_type="game-graph"),
            StatusView(),
            graph=GraphData(
                breadcrumb=[("root", "1"), ("7g7f", "2")]
            ),
        )

        assert 'data-action="node:1"' in html
        assert 'data-action="node:2"' not in html
        assert 'class="vz-crumb on vz-m"' in html

    def test_moves_carry_click_actions(self) -> None:
        """指し手の行はクリックで子局面へ移る．"""
        html = render_workbench(
            WorkbenchState(array_type="game-graph"),
            StatusView(),
            graph=GraphData(moves=[["7g7f", ".41", "51.9%"]]),
        )

        assert 'data-action="move:0"' in html

    def test_canvas_reads_sliders_and_root_from_dom(
        self,
    ) -> None:
        """Canvas レンダラーが読む id を出しておく．"""
        html = render_workbench(
            WorkbenchState(array_type="game-graph", node="99"),
            StatusView(),
            graph=GraphData(),
        )

        assert 'id="gt-depth-slider"' in html
        assert 'id="gt-min-prob-slider"' in html
        assert 'id="current-root"' in html
        assert 'value="99"' in html
