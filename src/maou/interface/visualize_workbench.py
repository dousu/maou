"""データ可視化ワークベンチの HTML 生成 (インターフェース層)．

``maou visualize`` の画面は gr.HTML 1 枚である．本モジュールは plain data
(状態と表示データ) だけを受け取り，ワークベンチ全体の HTML 文字列を返す．
Gradio には依存しない．

UI 操作はすべて ``data-action`` 文字列として符号化し，クライアント側
(static/visualize_workbench.js) がブリッジ経由で infra 層へ渡す．
analyze-gui の :mod:`maou.interface.analysis_workbench` と同じ方式．

配色・寸法は Modernist デザインシステム (Archivo・赤単色・角丸ゼロ・
2px 罫) に従う．クラス名は ``vz-`` 接頭辞を持ち，
static/visualize_workbench.css が唯一の定義元である．
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass, field
from typing import Any

from maou.app.visualization.record_renderer import Distribution

ARRAY_TYPES: tuple[str, ...] = (
    "hcpe",
    "preprocessing",
    "stage1",
    "stage2",
    "game-graph",
)
"""セグメンテッドコントロールに並べるデータ型 (CLI の --array-type と同順)．"""

_HIST_BINS = 20
"""ヒストグラムの階級数 (デザインモックのバー本数に合わせる)．"""


@dataclass(frozen=True)
class WorkbenchState:
    """ブラウザセッションごとの操作状態．

    Attributes:
        array_type: 表示中のデータ型
        source_mode: データソースの指定方法 ("Directory" / "File list")
        path_text: データソースのパス入力
        id_query: ID 検索の入力
        sfen_query: SFEN 検索の入力
        min_eval: 評価値範囲の下限入力 (空文字で無制限)
        max_eval: 評価値範囲の上限入力 (空文字で無制限)
        page: 現在のページ番号 (1 始まり)
        page_size: 1 ページあたりの件数
        selected: ページ内で選択中の行番号 (0 始まり)
        depth: ゲームグラフの表示深さ
        min_prob: ゲームグラフのエッジ最小確率
        node: ゲームグラフで選択中のノード (position_hash の文字列)
        mode: 結果一覧の内容 ("page" = ページ送り / "search" = 検索ヒット)
    """

    array_type: str = "hcpe"
    source_mode: str = "Directory"
    path_text: str = ""
    id_query: str = ""
    sfen_query: str = ""
    min_eval: str = ""
    max_eval: str = ""
    page: int = 1
    page_size: int = 20
    selected: int = 0
    depth: int = 3
    min_prob: float = 0.01
    node: str = ""
    mode: str = "page"


@dataclass(frozen=True)
class StatusView:
    """トップバーに出すサーバー状態．

    Attributes:
        badge: モードバッジの文言 (例 "REAL" / "MOCK" / "NO DATA")
        tone: バッジの色調 ("ok" / "mock" / "busy" / "none" / "error")
        count_main: 主要な件数 (カンマ区切り済み)
        count_unit: 件数の単位 (例 "records")
        path_label: 読み込み中のパス表示
        message: 詳細メッセージ (エラーや進捗)．空なら出さない．
    """

    badge: str = "NO DATA"
    tone: str = "none"
    count_main: str = "0"
    count_unit: str = "records"
    path_label: str = "—"
    message: str = ""


@dataclass(frozen=True)
class RecordData:
    """レコードブラウザの表示データ．

    Attributes:
        headers: 結果一覧の列見出し
        rows: 結果一覧の行 (セル値の並び)
        board_svg: 中央に出す盤面 SVG
        sfen: 選択中レコードの SFEN (無ければ空)
        record_id: 選択中レコードの ID
        details: レコード詳細のキーと値
        stats: データセット統計のキーと値
        distribution: 分布のもとになる数値列
        current_value: 分布上で強調する現在レコードの値
        total_pages: 総ページ数
        total_records: 総レコード数
        result_value: 先手勝率 (preprocessing のみ)．None で非表示．
        supports_eval_search: 評価値範囲検索に対応するか
        highlight_label: 盤面ハイライトの凡例文言
    """

    headers: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    board_svg: str = ""
    sfen: str = ""
    record_id: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    distribution: Distribution | None = None
    current_value: float | None = None
    total_pages: int = 1
    total_records: int = 0
    result_value: float | None = None
    supports_eval_search: bool = False
    highlight_label: str = ""


@dataclass(frozen=True)
class GraphData:
    """ゲームグラフの表示データ．

    Attributes:
        graph_html: グラフ本体 (Canvas レンダラーのマウント先を含む HTML)
        breadcrumb: パンくず [(ラベル, ノード hash), ...]
        board_svg: 選択ノードの盤面 SVG
        node_stats: 局面統計のキーと値
        moves: 指し手一覧の行 (指し手, 確率, 勝率)
        usi_line: エクスポート用の USI position 文字列
        node_count: 表示中のノード数
        edge_count: 総エッジ数
        total_nodes: 総ノード数
    """

    graph_html: str = ""
    breadcrumb: list[tuple[str, str]] = field(
        default_factory=list
    )
    board_svg: str = ""
    node_stats: dict[str, Any] = field(default_factory=dict)
    moves: list[list[Any]] = field(default_factory=list)
    usi_line: str = ""
    node_count: int = 0
    edge_count: int = 0
    total_nodes: int = 0


# ============================================================
# 小さな整形ヘルパー
# ============================================================


def _esc(value: Any) -> str:
    """HTML エスケープした文字列を返す．

    Args:
        value: 任意の値

    Returns:
        エスケープ済み文字列
    """
    return html.escape(str(value), quote=True)


def _fmt(value: Any, precision: int = 0) -> str:
    """表示用に値を整形する (エスケープ込み)．

    Args:
        value: 任意の値
        precision: 小数桁数 (0 なら整数として扱う)

    Returns:
        エスケープ済みの表示文字列
    """
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if precision:
            return _esc(f"{value:.{precision}f}")
        if value.is_integer():
            return _esc(f"{int(value):,}")
        return _esc(f"{value:.6g}")
    if isinstance(value, int):
        return _esc(f"{value:,}")
    return _esc(value)


def _grid_template(n_cols: int) -> str:
    """列数から grid-template-columns を決める．

    先頭を通し番号，2 列目を可変幅の主キー，残りを数値列とみなす
    (デザインモックの 44px / 1fr / 62px に合わせる)．

    Args:
        n_cols: 列数

    Returns:
        grid-template-columns の値
    """
    if n_cols <= 1:
        return "1fr"
    if n_cols == 2:
        return "minmax(0,1fr) 72px"
    return "42px minmax(0,1fr)" + " 58px" * (n_cols - 2)


def _cell_align(col: int, n_cols: int) -> str:
    """数値列を右寄せにする class 属性を返す．

    Args:
        col: 列番号 (0 始まり)
        n_cols: 列数

    Returns:
        class 属性に足す文字列 (不要なら空)
    """
    if n_cols >= 3 and col >= 2:
        return " vz-right"
    if n_cols == 2 and col == 1:
        return " vz-right"
    return ""


def _label(text: str) -> str:
    """セクション見出し (大文字ラベル) の HTML を返す．

    Args:
        text: 見出し文言

    Returns:
        HTML 文字列
    """
    return f'<div class="vz-lbl">{_esc(text)}</div>'


def _kv_rows(data: dict[str, Any]) -> str:
    """キーと値の一覧を vz-kv 行として組み立てる．

    Args:
        data: 表示する辞書

    Returns:
        HTML 文字列
    """
    if not data:
        return '<div class="vz-empty">データがありません</div>'
    return "".join(
        '<div class="vz-kv">'
        f"<span>{_esc(key)}</span>"
        f'<span class="vz-m">{_fmt(value)}</span>'
        "</div>"
        for key, value in data.items()
    )


def _table(
    headers: list[str],
    rows: list[list[Any]],
    *,
    selected: int | None = None,
    action: str = "",
    empty_message: str = "該当するレコードがありません",
) -> str:
    """罫線だけの表 (vz-h ヘッダー + vz-row 行) を組み立てる．

    Args:
        headers: 列見出し
        rows: 行データ
        selected: 強調表示する行番号 (0 始まり)
        action: 行クリックで送る data-action の接頭辞．空ならクリック不可．
        empty_message: 行が無いときの文言

    Returns:
        HTML 文字列
    """
    n_cols = len(headers) or (len(rows[0]) if rows else 0)
    if n_cols == 0:
        return (
            f'<div class="vz-empty">{_esc(empty_message)}</div>'
        )
    tmpl = _grid_template(n_cols)

    head = (
        f'<div class="vz-h" style="grid-template-columns:{tmpl}">'
        + "".join(
            f'<span class="vz-cell{_cell_align(i, n_cols)}">'
            f"{_esc(h)}</span>"
            for i, h in enumerate(headers)
        )
        + "</div>"
    )
    if not rows:
        body = (
            f'<div class="vz-empty">{_esc(empty_message)}</div>'
        )
    else:
        items = []
        for i, row in enumerate(rows):
            cls = "vz-row" + (" on" if selected == i else "")
            attr = (
                f' data-action="{_esc(action)}:{i}"'
                if action
                else ""
            )
            cells = "".join(
                f'<span class="vz-cell vz-m'
                f'{_cell_align(j, n_cols)}">{_fmt(cell)}</span>'
                for j, cell in enumerate(row[:n_cols])
            )
            items.append(
                f'<div class="{cls}"{attr} '
                f'style="grid-template-columns:{tmpl}">{cells}</div>'
            )
        body = "".join(items)
    return (
        '<div class="vz-table">'
        + head
        + f'<div class="vz-table-body">{body}</div>'
        + "</div>"
    )


def _text_input(
    action: str,
    value: str,
    placeholder: str,
    *,
    disabled: bool = False,
    mono: bool = True,
) -> str:
    """1 行テキスト入力を返す．

    Args:
        action: 変更時に送る data-action-input の名前
        value: 現在値
        placeholder: プレースホルダ
        disabled: 無効表示にするか
        mono: 等幅フォントにするか

    Returns:
        HTML 文字列
    """
    cls = (
        "vz-in"
        + (" vz-m" if mono else "")
        + (" off" if disabled else "")
    )
    dis = " disabled" if disabled else ""
    return (
        f'<input class="{cls}" type="text" '
        f'data-action-input="{_esc(action)}" '
        f'value="{_esc(value)}" '
        f'placeholder="{_esc(placeholder)}"{dis}>'
    )


def _button(
    action: str,
    text: str,
    *,
    variant: str = "",
    grow: bool = False,
) -> str:
    """ボタンを返す．

    Args:
        action: クリック時に送る data-action
        text: 表示文言
        variant: "pri" (赤の主ボタン) / "gh" (細罫) / 空 (既定)
        grow: 横幅いっぱいに伸ばすか

    Returns:
        HTML 文字列
    """
    cls = "vz-b"
    if variant:
        cls += f" {variant}"
    if grow:
        cls += " grow"
    return (
        f'<button type="button" class="{cls}" '
        f'data-action="{_esc(action)}">{_esc(text)}</button>'
    )


def _slider(
    action: str,
    value: float,
    low: float,
    high: float,
    step: float,
    label: str,
    *,
    precision: int = 0,
    elem_id: str = "",
) -> str:
    """ラベル + 値 + トラックのスライダーを返す．

    Args:
        action: 変更時に送る data-action-input の名前
        value: 現在値
        low: 最小値
        high: 最大値
        step: 刻み
        label: 表示ラベル
        precision: 値表示の小数桁数
        elem_id: 外枠に付ける id．Canvas レンダラー
            (static/game_graph_canvas.js) が DOM から値を読むために使う．

    Returns:
        HTML 文字列
    """
    shown = (
        f"{value:.{precision}f}"
        if precision
        else f"{int(value)}"
    )
    lo = f"{low:.{precision}f}" if precision else f"{int(low)}"
    hi = (
        f"{high:.{precision}f}" if precision else f"{int(high)}"
    )
    ident = f' id="{_esc(elem_id)}"' if elem_id else ""
    return (
        f'<div class="vz-slider"{ident}>'
        '<div class="vz-slider-head">'
        f"<span>{_esc(label)}</span>"
        f'<span class="vz-m">{_esc(shown)}</span>'
        "</div>"
        f'<input type="range" data-action-input="{_esc(action)}" '
        f'min="{low}" max="{high}" step="{step}" value="{value}">'
        '<div class="vz-slider-foot">'
        f"<span>{_esc(lo)}</span><span>{_esc(hi)}</span>"
        "</div></div>"
    )


# ============================================================
# ヒストグラム (デザインモックの SVG をそのまま数値で駆動する)
# ============================================================
#
# モックのバーは幅 16px・間隔 20px・ベースライン y=100・最大高 94px．
# 20 本並べて viewBox="0 0 400 110"．現在レコードが属する階級だけ
# accent 色にし，残りは neutral-500 に落とす．


def _histogram(
    distribution: Distribution | None,
    current: float | None,
) -> str:
    """分布の SVG ヒストグラムを返す．

    Args:
        distribution: 分布データ．None なら空状態を返す．
        current: 強調する現在レコードの値．None なら強調しない．

    Returns:
        HTML 文字列
    """
    if distribution is None or not distribution.values:
        return '<div class="vz-empty">分布を出せるデータがありません</div>'

    values = distribution.values
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        # 全部同じ値なら 1 本だけ立てる (幅ゼロの階級を作らない)
        low -= 0.5
        high += 0.5
    width = (high - low) / _HIST_BINS

    counts = [0] * _HIST_BINS
    for value in values:
        index = int((value - low) / width)
        counts[min(max(index, 0), _HIST_BINS - 1)] += 1
    peak = max(counts) or 1

    hot = -1
    if current is not None:
        hot = min(
            max(int((current - low) / width), 0),
            _HIST_BINS - 1,
        )

    bars = []
    for i, count in enumerate(counts):
        height = round(94 * count / peak)
        if height == 0 and count:
            height = 1
        x = i * 20 + 2
        y = 100 - height
        cls = "vz-bar on" if i == hot else "vz-bar"
        bars.append(
            f'<rect class="{cls}" x="{x}" y="{y}" '
            f'width="16" height="{height}"></rect>'
        )

    precision = distribution.precision
    left = (
        f"{low:.{precision}f}"
        if precision
        else f"{round(low):,}"
    )
    right = (
        f"{high:.{precision}f}"
        if precision
        else f"{round(high):,}"
    )

    return (
        '<div class="vz-hist">'
        '<svg viewBox="0 0 400 110" width="100%" '
        'preserveAspectRatio="none" role="img" '
        f'aria-label="{_esc(distribution.title)}">'
        + "".join(bars)
        + '<line class="vz-axis" x1="0" y1="100" x2="400" y2="100"></line>'
        + "</svg>"
        '<div class="vz-hist-foot">'
        f"<span>{_esc(left)}</span>"
        f'<span class="vz-hist-axis">{_esc(distribution.axis_label)}</span>'
        f"<span>{_esc(right)}</span>"
        "</div></div>"
    )


# ============================================================
# トップバー
# ============================================================


def _topbar(
    state: WorkbenchState,
    status: StatusView,
    *,
    types_enabled: bool = True,
) -> str:
    """ブランド・データ型切替・サーバー状態を並べたトップバーを返す．

    Args:
        state: 操作状態
        status: サーバー状態
        types_enabled: データ型を切り替えられるか．スタンドアロンの
            ゲームグラフサーバーはグラフ専用なので False にして，
            押せないコントロールを出さない．

    Returns:
        HTML 文字列
    """
    segs = "".join(
        '<button type="button" '
        f'class="{"on" if t == state.array_type else ""}"'
        + (
            f' data-action="type:{t}"'
            if types_enabled
            else " disabled"
        )
        + f">{_esc(t)}</button>"
        for t in ARRAY_TYPES
    )
    message = (
        f'<span class="vz-status-msg">{_esc(status.message)}</span>'
        if status.message
        else ""
    )
    return (
        '<header class="vz-top">'
        '<div class="vz-brand">'
        '<span class="vz-brand-name">MAOU</span>'
        '<span class="vz-brand-sub">VISUALIZE</span>'
        "</div>"
        f'<div class="vz-seg">{segs}</div>'
        '<div class="vz-spacer"></div>'
        '<div class="vz-status">'
        f'<span class="vz-dot {_esc(status.tone)}"></span>'
        f'<span class="vz-badge">{_esc(status.badge)}</span>'
        f'<span class="vz-m vz-count">{_esc(status.count_main)}</span>'
        f'<span class="vz-unit">{_esc(status.count_unit)}</span>'
        "</div>"
        '<span class="vz-vr"></span>'
        f'<span class="vz-m vz-path" title="{_esc(status.path_label)}">'
        f"{_esc(status.path_label)}</span>"
        + message
        + _button("rebuild", "再構築", variant="gh")
        + "</header>"
    )


# ============================================================
# レコードブラウザ
# ============================================================


def _source_section(state: WorkbenchState) -> str:
    """データソースの指定セクションを返す．

    Args:
        state: 操作状態

    Returns:
        HTML 文字列
    """
    modes = "".join(
        f'<button type="button" class="{"on" if m == state.source_mode else ""}" '
        f'data-action="srcmode:{m}">{_esc(m)}</button>'
        for m in ("Directory", "File list")
    )
    placeholder = (
        "./data/hcpe/"
        if state.source_mode == "Directory"
        else "a.feather, b.feather"
    )
    return (
        '<section class="vz-sec">'
        + _label("データソース")
        + f'<div class="vz-seg start">{modes}</div>'
        + _text_input("path", state.path_text, placeholder)
        + '<div class="vz-btn-row">'
        + _button("load", "読み込み", variant="pri", grow=True)
        + _button("refresh", "更新", variant="gh")
        + "</div></section>"
    )


def _search_section(
    state: WorkbenchState, data: RecordData
) -> str:
    """ID / SFEN / 評価値範囲の検索セクションを返す．

    Args:
        state: 操作状態
        data: 表示データ (評価値検索の可否を見る)

    Returns:
        HTML 文字列
    """
    if data.supports_eval_search:
        eval_row = (
            '<div class="vz-range">'
            + _text_input("mineval", state.min_eval, "-300")
            + '<span class="vz-range-sep">〜</span>'
            + _text_input("maxeval", state.max_eval, "+300")
            + "</div>"
        )
    else:
        eval_row = (
            '<div class="vz-range is-off">'
            + _text_input("mineval", "", "-300", disabled=True)
            + '<span class="vz-range-sep">〜</span>'
            + _text_input("maxeval", "", "+300", disabled=True)
            + "</div>"
            '<div class="vz-hint">評価値範囲検索は hcpe のみ対応</div>'
        )
    return (
        '<section class="vz-sec">'
        + _label("検索")
        + _text_input(
            "id", state.id_query, "ID — 0x… (2文字で候補)"
        )
        + _text_input(
            "sfen",
            state.sfen_query,
            "SFEN — lnsgkgsnl/…",
        )
        + eval_row
        + '<div class="vz-btn-row">'
        + _button("search", "検索", variant="pri", grow=True)
        + _button("clear", "クリア", variant="gh")
        + "</div></section>"
    )


def _result_section(
    state: WorkbenchState, data: RecordData
) -> str:
    """結果一覧とページ送りのセクションを返す．

    Args:
        state: 操作状態
        data: 表示データ

    Returns:
        HTML 文字列
    """
    return (
        '<section class="vz-sec grow">'
        '<div class="vz-sec-head">'
        f"{_label(f'結果 — {len(data.rows)} 件')}"
        '<div class="vz-pager">'
        + _button("page:prev", "‹", variant="gh")
        + f'<span class="vz-m vz-pageno">{state.page} / '
        f"{max(data.total_pages, 1)}</span>"
        + _button("page:next", "›", variant="gh")
        + "</div></div>"
        + _table(
            data.headers,
            data.rows,
            selected=state.selected,
            action="row",
            empty_message="データソースを読み込んでください",
        )
        + "</section>"
    )


def _result_bar(data: RecordData) -> str:
    """先手勝率のバー (preprocessing のみ) を返す．

    Args:
        data: 表示データ

    Returns:
        HTML 文字列．対象外なら空文字．
    """
    if data.result_value is None:
        return ""
    pct = max(0.0, min(1.0, data.result_value)) * 100
    return (
        '<div class="vz-winbar">'
        '<span class="vz-lbl">result_value</span>'
        '<div class="vz-winbar-track">'
        f'<div class="vz-winbar-fill" style="width:{pct:.1f}%"></div>'
        '<div class="vz-winbar-mid"></div>'
        "</div>"
        f'<span class="vz-m vz-winbar-pct">{pct:.0f}%</span>'
        '<span class="vz-unit">先手勝率</span>'
        "</div>"
    )


def _stage(state: WorkbenchState, data: RecordData) -> str:
    """中央ステージ (局面・盤面・レコード送り) を返す．

    Args:
        state: 操作状態
        data: 表示データ

    Returns:
        HTML 文字列
    """
    sfen = data.sfen or data.record_id or "—"
    total = len(data.rows)
    position = state.selected + 1 if total else 0
    return (
        '<section class="vz-stage">'
        '<div class="vz-stage-head">'
        + _label("局面")
        + f'<span class="vz-m vz-sfen" title="{_esc(sfen)}">'
        f"{_esc(sfen)}</span>"
        + f'<button type="button" class="vz-b gh" '
        f'data-copy="{_esc(sfen)}">コピー</button>' + "</div>"
        '<div class="vz-board">'
        + (
            data.board_svg
            or '<div class="vz-empty">盤面がありません</div>'
        )
        + "</div>"
        + _result_bar(data)
        + '<div class="vz-rule"></div>'
        '<div class="vz-nav">'
        + _button("rec:prev", "◀ 前")
        + '<div class="vz-nav-label">'
        f'<span class="vz-m vz-recno">Record {position} / {total}</span>'
        '<span class="vz-hint">K / J で送り，Ctrl+← → でページ</span>'
        "</div>"
        + _button("rec:next", "次 ▶")
        + '<span class="vz-vr"></span>'
        '<label class="vz-legend-inline">'
        '<span class="vz-swatch-box"></span>'
        f"{_esc(data.highlight_label or '選択中のレコード')}"
        "</label>"
        "</div></section>"
    )


def _detail_rail(data: RecordData) -> str:
    """右レール (レコード詳細・分布・統計) を返す．

    Args:
        data: 表示データ

    Returns:
        HTML 文字列
    """
    title = (
        data.distribution.title
        if data.distribution is not None
        else "分布"
    )
    return (
        '<aside class="vz-rail right">'
        '<section class="vz-sec">'
        + _label("レコード詳細")
        + f'<div class="vz-kv-list">{_kv_rows(data.details)}</div>'
        + "</section>"
        '<section class="vz-sec">'
        '<div class="vz-sec-head">'
        + _label(title)
        + f'<span class="vz-m vz-hint">n={len(data.rows)}</span>'
        + "</div>"
        + _histogram(data.distribution, data.current_value)
        + "</section>"
        '<section class="vz-sec">'
        '<div class="vz-sec-head">'
        + _label("データセット統計")
        + _button("stats", "更新", variant="gh")
        + "</div>"
        + (
            f'<div class="vz-kv-grid">{_kv_rows(data.stats)}</div>'
            if data.stats
            else '<div class="vz-empty">「更新」で統計を取得します</div>'
        )
        + "</section></aside>"
    )


def _record_screen(
    state: WorkbenchState, data: RecordData
) -> str:
    """レコードブラウザ画面 (3 カラム) を返す．

    Args:
        state: 操作状態
        data: 表示データ

    Returns:
        HTML 文字列
    """
    return (
        '<div class="vz-body">'
        '<aside class="vz-rail left">'
        + _source_section(state)
        + _search_section(state, data)
        + _result_section(state, data)
        + "</aside>"
        + _stage(state, data)
        + _detail_rail(data)
        + "</div>"
    )


# ============================================================
# ゲームグラフ
# ============================================================

GRAPH_LEGEND: tuple[tuple[str, str, str], ...] = (
    ("dot", "rgb(25,118,210)", "先手有利 — 最善手勝率 > 55%"),
    ("dot", "#9E9E9E", "互角 — 45% 〜 55%"),
    ("dot", "rgb(211,47,47)", "後手有利 — 最善手勝率 < 45%"),
    ("ring", "#0070f3", "選択中のノード"),
    ("dash", "#ff9800", "深さ打ち切り (未展開の子を持つ)"),
)
"""凡例の (形, 色, 説明)．static/game_graph_canvas.js の描画と一致させる．"""


def _graph_legend() -> str:
    """ゲームグラフの凡例を返す．

    Returns:
        HTML 文字列
    """
    items = []
    for shape, color, text in GRAPH_LEGEND:
        if shape == "dot":
            style = f"background:{color}"
        elif shape == "ring":
            style = f"background:transparent;border:3px solid {color}"
        else:
            style = f"background:transparent;border:2px dashed {color}"
        items.append(
            '<div class="vz-legend-item">'
            f'<span class="vz-swatch" style="{style}"></span>'
            f"{_esc(text)}</div>"
        )
    return (
        '<div class="vz-legend">'
        + "".join(items)
        + '<p class="vz-hint">ノード径は親エッジ確率の平方根に比例し，'
        "エッジ線幅は確率に比例する．</p></div>"
    )


def _breadcrumb(crumbs: list[tuple[str, str]]) -> str:
    """パンくずリストを返す．

    Args:
        crumbs: [(ラベル, ノード hash), ...]

    Returns:
        HTML 文字列
    """
    if not crumbs:
        return '<div class="vz-empty">—</div>'
    parts = []
    for i, (label, node) in enumerate(crumbs):
        last = i == len(crumbs) - 1
        if last:
            parts.append(
                f'<span class="vz-crumb on vz-m">{_esc(label)}</span>'
            )
        else:
            parts.append(
                f'<button type="button" class="vz-crumb vz-m" '
                f'data-action="node:{_esc(node)}">{_esc(label)}</button>'
                '<span class="vz-crumb-sep">/</span>'
            )
    return '<nav class="vz-crumbs">' + "".join(parts) + "</nav>"


def _graph_screen(
    state: WorkbenchState, data: GraphData
) -> str:
    """ゲームグラフ画面 (3 カラム) を返す．

    Args:
        state: 操作状態
        data: 表示データ

    Returns:
        HTML 文字列
    """
    return (
        '<div class="vz-body">'
        '<aside class="vz-rail left">'
        '<section class="vz-sec">'
        + _label("パンくず")
        + _breadcrumb(data.breadcrumb)
        + '<div class="vz-btn-row">'
        + _button(
            "node:root", "ルートに戻る", variant="gh", grow=True
        )
        + _button(
            "setroot", "ルートに設定", variant="gh", grow=True
        )
        + "</div></section>"
        '<section class="vz-sec">'
        + _label("表示コントロール")
        + _slider(
            "depth",
            state.depth,
            1,
            20,
            1,
            "表示深さ",
            elem_id="gt-depth-slider",
        )
        + _slider(
            "minprob",
            state.min_prob,
            0.001,
            0.3,
            0.001,
            "最小確率",
            precision=3,
            elem_id="gt-min-prob-slider",
        )
        + _button("redraw", "再描画", variant="pri", grow=True)
        + "</section>"
        '<section class="vz-sec">'
        + _label("凡例")
        + _graph_legend()
        + "</section>"
        '<section class="vz-sec">'
        + _label("エクスポート")
        + f'<div class="vz-in vz-m wrap">{_esc(data.usi_line or "—")}</div>'
        + '<div class="vz-btn-row">'
        + f'<button type="button" class="vz-b gh grow" '
        f'data-copy="{_esc(data.usi_line)}">USI コピー</button>'
        + _button("csv", "CSV 出力", variant="gh", grow=True)
        + "</div></section></aside>"
        '<section class="vz-stage">'
        '<div class="vz-stage-head">'
        + _label("グラフ")
        + '<span class="vz-hint">クリックで詳細更新 / '
        "ダブルクリックで展開 / ホイールでズーム / ドラッグで移動</span>"
        + f'<span class="vz-m vz-hint">表示 {data.node_count:,} / '
        f"{data.total_nodes:,}</span>" + "</div>"
        f'<div class="vz-graph">{data.graph_html}</div>'
        # Canvas レンダラーは深さ・最小確率・現在ルートを DOM から読む
        # (static/game_graph_canvas.js の readSlider / readCurrentRoot)．
        f'<div id="current-root" class="maou-hidden">'
        f'<input type="hidden" value="{_esc(state.node)}"></div>'
        "</section>"
        '<aside class="vz-rail right">'
        '<section class="vz-sec">'
        + _label("選択局面")
        + '<div class="vz-board small">'
        + (
            data.board_svg
            or '<div class="vz-empty">ノードを選択してください</div>'
        )
        + "</div></section>"
        '<section class="vz-sec">'
        + _label("局面統計")
        + f'<div class="vz-kv-list">{_kv_rows(data.node_stats)}</div>'
        + "</section>"
        '<section class="vz-sec grow">'
        + _label("指し手一覧 — 確率降順")
        + _table(
            ["MOVE", "P", "WIN"],
            data.moves,
            action="move",
            empty_message="指し手がありません",
        )
        + "</section></aside></div>"
    )


# ============================================================
# エントリポイント
# ============================================================


def render_workbench(
    state: WorkbenchState,
    status: StatusView,
    *,
    record: RecordData | None = None,
    graph: GraphData | None = None,
    render_stamp: str = "",
    types_enabled: bool = True,
) -> str:
    """ワークベンチ全体の HTML を返す．

    Args:
        state: 操作状態
        status: サーバー状態 (トップバー)
        record: レコードブラウザの表示データ (game-graph 以外で必須)
        graph: ゲームグラフの表示データ (game-graph で必須)
        render_stamp: 再描画を JS に知らせる任意の識別子
        types_enabled: データ型セグメントを押せるようにするか

    Returns:
        ``#viz-workbench`` をルートとする HTML 文字列
    """
    if state.array_type == "game-graph":
        screen = _graph_screen(state, graph or GraphData())
    else:
        screen = _record_screen(state, record or RecordData())
    return (
        f'<div id="viz-workbench" data-render="{_esc(render_stamp)}">'
        + _topbar(state, status, types_enabled=types_enabled)
        + screen
        + "</div>"
    )
