"""ワークベンチ用 HTML レンダラーのテスト．

gr.JSON / gr.Dataframe を廃してサーバー側で HTML を組むようにしたため，
エスケープ・選択行・空状態の扱いを固定する．
"""

from maou.infra.visualization.game_graph_shared import (
    build_kv_html,
    build_row_table_html,
    build_stats_grid_html,
)


class TestBuildKvHtml:
    """build_kv_html のテスト．"""

    def test_renders_each_pair_as_a_row(self) -> None:
        """辞書の各要素が vz-kv 行になる．"""
        html = build_kv_html({"id": "abc", "eval": 120})

        assert html.count('class="vz-kv"') == 2
        assert "<span>id</span>" in html
        assert ">abc<" in html
        assert ">120<" in html

    def test_escapes_markup_in_keys_and_values(self) -> None:
        """キーと値の HTML はエスケープされる．"""
        html = build_kv_html(
            {"<k>": "<script>alert(1)</script>"}
        )

        assert "<script>" not in html
        assert "&lt;script&gt;" in html
        assert "&lt;k&gt;" in html

    def test_empty_input_returns_empty_state(self) -> None:
        """None と空辞書は空状態を返す．"""
        assert 'class="vz-empty"' in build_kv_html(None)
        assert 'class="vz-empty"' in build_kv_html({})

    def test_none_value_is_shown_as_dash(self) -> None:
        """None の値はダッシュで表示する．"""
        assert "—" in build_kv_html({"missing": None})


class TestBuildStatsGridHtml:
    """build_stats_grid_html のテスト．"""

    def test_flat_values_go_into_a_grid(self) -> None:
        """平坦な値は 1 つのグリッドにまとまる．"""
        html = build_stats_grid_html({"mean": 1.5, "max": 9})

        assert html.count('class="vz-kv-grid"') == 1
        assert html.count('class="vz-kv"') == 2

    def test_nested_dict_gets_its_own_section(self) -> None:
        """入れ子の辞書は見出し付きの別グリッドになる．"""
        html = build_stats_grid_html(
            {"total": 10, "eval": {"mean": 1.0}}
        )

        assert html.count('class="vz-kv-grid"') == 2
        assert 'class="vz-sub-lbl"' in html
        assert ">eval<" in html

    def test_empty_input_returns_empty_state(self) -> None:
        """統計が無いときは案内を返す．"""
        assert 'class="vz-empty"' in build_stats_grid_html(None)


class TestBuildRowTableHtml:
    """build_row_table_html のテスト．"""

    def test_renders_header_and_rows(self) -> None:
        """見出し行と各データ行が出る．"""
        html = build_row_table_html(
            ["Index", "ID", "Eval"],
            [[1, "a", 10], [2, "b", -3]],
        )

        assert html.count('class="vz-h"') == 1
        assert html.count("vz-row") == 2
        assert ">Index<" in html
        assert ">-3<" in html

    def test_selected_row_is_marked(self) -> None:
        """selected で指定した行だけに on が付く．"""
        html = build_row_table_html(
            ["Index", "ID"],
            [[1, "a"], [2, "b"], [3, "c"]],
            selected=1,
        )

        assert html.count('class="vz-row on"') == 1

    def test_clickable_adds_data_row_indices(self) -> None:
        """clickable のとき各行に 0 始まりの data-row が付く．"""
        html = build_row_table_html(
            ["Index", "ID"],
            [[1, "a"], [2, "b"]],
            clickable=True,
        )

        assert 'data-row="0"' in html
        assert 'data-row="1"' in html

    def test_not_clickable_omits_data_row(self) -> None:
        """clickable でなければ data-row は付かない．"""
        html = build_row_table_html(
            ["Index"], [[1]], clickable=False
        )

        assert "data-row=" not in html

    def test_escapes_cell_markup(self) -> None:
        """セルの HTML はエスケープされる．"""
        html = build_row_table_html(
            ["ID"], [["<img onerror=x>"]]
        )

        assert "<img" not in html
        assert "&lt;img" in html

    def test_empty_rows_keep_header_and_show_message(
        self,
    ) -> None:
        """行が無くても見出しは残し，空状態の文言を出す．"""
        html = build_row_table_html(
            ["Index", "ID"], [], empty_message="なし"
        )

        assert 'class="vz-h"' in html
        assert "なし" in html

    def test_no_columns_returns_empty_state(self) -> None:
        """列が決まらないときは空状態だけを返す．"""
        html = build_row_table_html([], [])

        assert 'class="vz-empty"' in html
        assert 'class="vz-h"' not in html

    def test_extra_cells_are_truncated_to_header_width(
        self,
    ) -> None:
        """見出しより多いセルは切り捨てる (列ずれを防ぐ)．"""
        html = build_row_table_html(["A", "B"], [[1, 2, 3]])

        assert ">3<" not in html
