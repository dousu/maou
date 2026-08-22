"""棋譜解析 GUI (analyze-gui) ワークベンチの HTML 組み立て (interface 層)．

3 カラム (悪手/棋譜・盤面・グラフ/候補手/解析) を 1 枚の HTML として
組み立てる．Gradio 側は :func:`render_workbench` の戻り値を gr.HTML に
流し込むだけで，レイアウトと表示整形はすべて本モジュールが担う．

操作は ``data-action`` 属性に符号化し，static/analysis_workbench.js が
ブリッジ経由で infra 層のハンドラに渡す (アクション名の定義は
:data:`ACTION_*` 群とハンドラ側で対にする)．
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Any

from maou.domain.visualization.board_renderer import (
    MODERNIST_BOARD_THEME,
)
from maou.interface import analysis_gui
from maou.interface.analysis_gui import (
    ClickState,
    SessionView,
    VariationTree,
    sente_eval_cp,
    sente_winrate,
)

# 評価値グラフの描画領域 (viewBox 座標)
_GRAPH_W = 306
_GRAPH_H = 156
_GRAPH_LEFT = 4.0
_GRAPH_RIGHT = 300.0
_GRAPH_TOP = 6.0
_GRAPH_BOTTOM = 150.0

# 評価値表示の上下クリップ (cp)．これを超える値は端に貼り付く
_EVAL_CLIP_CP = 2000.0

# 候補手のキーボードショートカット (1-5) が届く範囲
_SHORTCUT_CANDIDATES = 5


@dataclass(frozen=True)
class WorkbenchOptions:
    """ワークベンチの表示・解析オプション (gr.State 保持用 plain data)．

    Attributes:
        threshold: 悪手と見なす勝率損失のしきい値．
        show_arrows: 盤面に候補手矢印を描くか．
        show_pv: 最善手の PV 連鎖矢印を描くか．
        top_n: 候補手の表示件数．
        y_mode: グラフ縦軸 ("winrate" / "eval_cp")．
        budget_mode: 解析予算の種類 ("time" / "playouts")．
        budget_value: 解析予算の値．
        legal_open: 合法手一覧 (クリック代替入力) を開いているか．
    """

    threshold: float = 0.10
    show_arrows: bool = True
    show_pv: bool = False
    top_n: int = 5
    y_mode: str = "winrate"
    budget_mode: str = "time"
    budget_value: int = 1000
    legal_open: bool = False


@dataclass(frozen=True)
class AnalysisProgress:
    """全局面解析の進捗と直近の解析ステータス．

    Attributes:
        running: 全局面解析が実行中か．
        done: 解析済み局面数．
        total: 全局面数．
        status: 直近の解析メッセージ (1 局面解析を含む)．
        is_error: status をエラー色で出すか．
    """

    running: bool = False
    done: int = 0
    total: int = 0
    status: str = ""
    is_error: bool = False


def _esc(value: Any) -> str:
    """HTML エスケープした文字列にする．"""
    return html.escape(str(value), quote=True)


def _fmt_rate(value: Any) -> str:
    """割合を百分率にする (None は "-")．"""
    if value is None:
        return "-"
    return f"{float(value) * 100:.1f}%"


def _fmt_loss(value: Any) -> str:
    """勝率損失を 3 桁小数にする (None は空文字列)．"""
    if value is None:
        return ""
    return f"{float(value):.3f}"


# ----------------------------------------------------------------------
# データ抽出
# ----------------------------------------------------------------------


def blunders(
    view: SessionView, threshold: float
) -> list[tuple[int, str, float]]:
    """勝率損失がしきい値以上の手を損失の大きい順に返す．

    Args:
        view: セッション状態．
        threshold: 勝率損失のしきい値．

    Returns:
        ``(手数 (1 始まり), 日本語表記, 勝率損失)`` のリスト．
        レポート未読込のときは空リスト．
    """
    if view.report is None:
        return []
    found: list[tuple[int, str, float]] = []
    for i, label in enumerate(view.move_labels):
        analysis = analysis_gui.position_analysis(view, i)
        if analysis is None:
            continue
        loss = analysis.get("winrate_loss")
        if loss is None or float(loss) < threshold:
            continue
        found.append((i + 1, label, float(loss)))
    found.sort(key=lambda item: item[2], reverse=True)
    return found


def blunder_target_ply(ply: int) -> int:
    """悪手の手数から «その手を指す直前» のスナップショット番号を返す．

    悪手は「指す前の局面」で候補手矢印とともに見比べたいので，棋譜行
    クリック (指した後の局面へ移動) とは 1 つずれる．
    """
    return max(0, ply - 1)


def neighbouring_blunder(
    view: SessionView,
    threshold: float,
    current_ply: int,
    *,
    forward: bool,
) -> int | None:
    """現在位置から前/次の悪手のスナップショット番号を返す．

    Args:
        view: セッション状態．
        threshold: 勝率損失のしきい値．
        current_ply: 現在のスナップショット番号．
        forward: True で次の悪手，False で前の悪手．

    Returns:
        移動先のスナップショット番号．該当がなければ None．
    """
    targets = sorted(
        blunder_target_ply(ply)
        for ply, _, _ in blunders(view, threshold)
    )
    if forward:
        return next(
            (p for p in targets if p > current_ply), None
        )
    return next(
        (p for p in reversed(targets) if p < current_ply), None
    )


# ----------------------------------------------------------------------
# 部分描画
# ----------------------------------------------------------------------


def _header(
    source_label: str, engine_label: str, is_mock: bool
) -> str:
    """ヘッダー帯 (ブランド・読込元・エンジン・ショートカット) を作る．"""
    engine_tag = (
        f'<span class="mw-tag mw-tag-outline">{_esc(engine_label)}</span>'
        if is_mock
        else f'<span class="mw-tag mw-tag-neutral">{_esc(engine_label)}</span>'
    )
    shortcuts = (
        '<span class="mw-kbd">←</span> <span class="mw-kbd">→</span> 手送り'
        '　<span class="mw-kbd">⇧←</span> 前の悪手'
        '　<span class="mw-kbd">B</span> 本譜'
        '　<span class="mw-kbd">Space</span> 解析'
    )
    return (
        '<div class="mw-header">'
        '<div class="mw-brand">maou <span>棋譜解析</span></div>'
        f'<div class="mw-source">{_esc(source_label)}</div>'
        f"{engine_tag}"
        '<div class="mw-header-right">'
        f'<span class="mw-shortcuts">{shortcuts}</span>'
        '<button type="button" class="mw-btn mw-btn-secondary" '
        'data-open-file="1">ファイル…</button>'
        "</div></div>"
    )


def _summary_strip(view: SessionView) -> str:
    """対局・一致率・平均勝率損失・詰み発見の 4 セルを作る．"""
    doc = view.document
    names = [n or "?" for n in doc.names]
    win_text = {
        0: "引き分け",
        1: "先手勝ち",
        2: "後手勝ち",
    }.get(doc.win if doc.win is not None else -1, "結果不明")
    result = f"{doc.n_moves}手・{win_text}"
    if doc.endgame:
        result += f"（{doc.endgame}）"

    cells = [
        (
            "対局",
            f"▲{_esc(names[0])} vs △{_esc(names[1])}",
            _esc(result),
        )
    ]

    if view.report is None:
        cells.append(
            (
                "解析",
                "レポート未読込",
                "盤面再生と分岐のみ利用できます",
            )
        )
        cells.append(
            ("一致率", "-", "レポートを読み込むと表示されます")
        )
        cells.append(("詰み発見", "-", ""))
        return _summary_html(cells)

    summary = view.report.get("summary", {})
    match_rate = summary.get("match_rate", {})
    mean_loss = summary.get("mean_winrate_loss", {})
    cells.append(
        (
            "一致率",
            f"先手 {_fmt_rate(match_rate.get('black'))} ／ "
            f"後手 {_fmt_rate(match_rate.get('white'))}",
            "エンジン最善との一致",
        )
    )
    cells.append(
        (
            "平均勝率損失",
            f"先手 {_fmt_loss(mean_loss.get('black')) or '-'} ／ "
            f"後手 {_fmt_loss(mean_loss.get('white')) or '-'}",
            "1手あたり",
        )
    )
    mates = summary.get("mates_found") or []
    if mates:
        plies = "・".join(f"{m['ply']}" for m in mates[:6])
        suffix = "手目" if len(mates) <= 6 else "手目 ほか"
        cells.append(
            (
                "詰み発見",
                f"{plies}{suffix}",
                f"{len(mates)}局面",
            )
        )
    else:
        cells.append(("詰み発見", "なし", "0局面"))
    return _summary_html(cells)


def _summary_html(cells: list[tuple[str, str, str]]) -> str:
    """サマリ帯のセル列を HTML にする (値は組み立て済み)．"""
    parts = ['<div class="mw-summary">']
    for kicker, value, sub in cells:
        parts.append(
            '<div class="mw-summary-cell">'
            f'<div class="mw-kicker">{_esc(kicker)}</div>'
            f'<div class="mw-summary-value">{value}</div>'
            f'<div class="mw-summary-sub">{sub}</div>'
            "</div>"
        )
    parts.append("</div>")
    return "".join(parts)


def _blunder_rail(
    view: SessionView, options: WorkbenchOptions
) -> str:
    """悪手ジャンプリストを作る．"""
    hits = blunders(view, options.threshold)
    head = (
        '<div class="mw-section-head">'
        '<span class="mw-label">悪手（損失 ≥ '
        f'<input class="mw-threshold" type="number" min="0.01" max="0.99" '
        f'step="0.01" value="{options.threshold:.2f}" '
        'data-action-input="opt:threshold">）</span>'
        f'<span class="mw-tag mw-tag-accent">{len(hits)}</span>'
        "</div>"
    )
    if view.report is None:
        return head + (
            '<div class="mw-empty">解析レポートを読み込むか，'
            "全局面解析を実行すると表示されます．</div>"
        )
    if not hits:
        return head + (
            '<div class="mw-empty">しきい値以上の手はありません．</div>'
        )
    rows = []
    for ply, label, loss in hits:
        target = blunder_target_ply(ply)
        rows.append(
            f'<button type="button" class="mw-blunder" '
            f'data-action="goto:{target}">'
            f'<span class="mw-blunder-ply">{ply}手目</span>'
            f'<span class="mw-move">{_esc(label)}</span>'
            f'<span class="mw-blunder-loss">{loss:.3f}</span>'
            "</button>"
        )
    return head + "".join(rows)


def _kifu_list(
    view: SessionView,
    tree: VariationTree,
    options: WorkbenchOptions,
) -> str:
    """棋譜リスト (手数/指し手/損失/一致) を作る．"""
    node = analysis_gui.current_node(tree)
    current_ply = (
        node.snapshot.ply
        if node.is_mainline
        else analysis_gui.mainline_ply(tree)
    )
    parts = [
        '<div class="mw-section-head mw-section-head-split">'
        '<span class="mw-label">棋譜</span>'
        '<span class="mw-note">行クリックで移動</span></div>',
        '<div class="mw-kifu-head"><span>手数</span><span>指し手</span>'
        '<span style="text-align:right">損失</span>'
        '<span style="text-align:center">一致</span></div>',
        '<div class="mw-kifu-list">',
    ]
    for i, label in enumerate(view.move_labels):
        ply = i + 1
        analysis = analysis_gui.position_analysis(view, i)
        loss_value = (
            analysis.get("winrate_loss") if analysis else None
        )
        loss_text = _fmt_loss(loss_value)
        is_blunder = (
            loss_value is not None
            and float(loss_value) >= options.threshold
        )
        match_text = (
            "✓" if analysis and analysis.get("match") else ""
        )
        if analysis and analysis.get("mate_found"):
            match_text = f"★{match_text}"
        classes = "mw-kifu-row" + (
            " is-current" if ply == current_ply else ""
        )
        loss_classes = "mw-kifu-loss" + (
            " is-blunder" if is_blunder else ""
        )
        parts.append(
            f'<button type="button" class="{classes}" '
            f'data-action="goto:{ply}">'
            f'<span class="mw-kifu-ply">{ply}</span>'
            f'<span class="mw-move">{_esc(label)}</span>'
            f'<span class="{loss_classes}">{loss_text}</span>'
            f'<span class="mw-kifu-match">{match_text}</span>'
            "</button>"
        )
    parts.append("</div>")
    return "".join(parts)


def _position_bar(
    view: SessionView, tree: VariationTree
) -> str:
    """盤面上の現在局面帯 (本譜/分岐・手数・消費時間) を作る．"""
    node = analysis_gui.current_node(tree)
    turn_text = (
        "先手番" if node.snapshot.turn == "b" else "後手番"
    )

    if node.is_mainline:
        tag = '<span class="mw-tag mw-tag-neutral">本譜</span>'
        ply = node.snapshot.ply
        if ply == 0:
            body = f"初期局面・{turn_text}"
        else:
            label = view.move_labels[ply - 1]
            body = (
                f"{ply}手目 "
                f'<span class="mw-move">{_esc(label)}</span> '
                f"のあと・{turn_text}"
            )
        elapsed = ""
        if 0 < ply <= len(view.document.times):
            elapsed = (
                f'<span class="mw-elapsed">消費 '
                f"{view.document.times[ply - 1]}s</span>"
            )
        return (
            f'<div class="mw-position-bar">{tag}<span>{body}</span>'
            f"{elapsed}</div>"
        )

    ancestor = analysis_gui.mainline_ancestor(tree)
    depth = node.snapshot.ply - ancestor.snapshot.ply
    breadcrumb = analysis_gui.breadcrumb_markdown(tree)
    breadcrumb = breadcrumb.replace("**", "")
    return (
        '<div class="mw-position-bar">'
        '<span class="mw-tag mw-tag-accent">分岐</span>'
        f'<span class="mw-move">{_esc(breadcrumb)}</span>'
        f"<span>（{depth}手）・{turn_text}</span>"
        '<span class="mw-elapsed">'
        '<button type="button" class="mw-btn mw-btn-secondary" '
        'data-action="nav:mainline">本譜へ戻る '
        '<span class="mw-kbd">B</span></button></span>'
        "</div>"
    )


def _nav_bar(view: SessionView, tree: VariationTree) -> str:
    """手送りボタンと手数スクラバを作る．"""
    ply = analysis_gui.mainline_ply(tree)
    total = view.document.n_moves
    ratio = (ply / total * 100) if total else 0.0
    buttons = "".join(
        f'<button type="button" class="mw-btn mw-btn-secondary" '
        f'data-action="{action}">{_esc(label)}</button>'
        for action, label in (
            ("nav:first", "|◀"),
            ("nav:prev", "◀ 前"),
            ("nav:next", "次 ▶"),
            ("nav:last", "▶|"),
        )
    )
    return (
        f'<div class="mw-navbar">{buttons}'
        '<div class="mw-scrubber">'
        f'<div class="mw-scrubber-track" data-scrub="{total}">'
        f'<div class="mw-scrubber-fill" style="width:{ratio:.2f}%"></div>'
        f'<div class="mw-scrubber-thumb" style="left:{ratio:.2f}%"></div>'
        "</div>"
        f'<span class="mw-scrubber-count">{ply} / {total}</span>'
        "</div></div>"
    )


def _hint_bar(
    view: SessionView,
    tree: VariationTree,
    click: ClickState,
    options: WorkbenchOptions,
) -> str:
    """盤面クリックの状態表示と成/不成の確認ボタンを作る．"""
    node = analysis_gui.current_node(tree)
    status = analysis_gui.click_status_text(
        node.snapshot, click
    )
    active = bool(click.selected or click.pending_usis)
    classes = "mw-hint-bar" + (" is-active" if active else "")

    if click.pending_usis:
        right = (
            '<span class="mw-promotion">'
            '<button type="button" class="mw-btn mw-btn-primary" '
            'data-action="promote:1">成る</button>'
            '<button type="button" class="mw-btn mw-btn-secondary" '
            'data-action="promote:0">成らず</button></span>'
        )
    else:
        legal_label = (
            "合法手一覧を閉じる"
            if options.legal_open
            else "合法手一覧"
        )
        right = (
            '<span class="mw-hint-right">代替入力 '
            '<button type="button" class="mw-btn mw-btn-ghost" '
            f'data-action="toggle:legal">{legal_label} '
            '<span class="mw-kbd">L</span></button></span>'
        )
    if not active:
        status = (
            "盤面クリック: 動かす駒 → 行き先。"
            "別の手を指すと自動で分岐します。"
        )
    return (
        f'<div class="{classes}"><span>{_esc(status)}</span>'
        f"{right}</div>"
    )


def _legal_moves(tree: VariationTree) -> str:
    """合法手一覧 (盤面クリックの代替入力) を作る．"""
    node = analysis_gui.current_node(tree)
    legal = analysis_gui.node_legal_moves(tree)
    choices = analysis_gui.legal_move_choices(
        node.snapshot, legal
    )
    if not choices:
        return (
            '<div class="mw-legal">'
            '<span class="mw-note">合法手がありません（詰み）．</span>'
            "</div>"
        )
    buttons = "".join(
        f'<button type="button" class="mw-legal-move" '
        f'data-action="play:{_esc(usi)}" title="{_esc(usi)}">'
        f"{_esc(label.split(' (')[0])}</button>"
        for label, usi in choices
    )
    return f'<div class="mw-legal">{buttons}</div>'


def _position_strings(
    view: SessionView, tree: VariationTree
) -> str:
    """SFEN と position 文字列のエクスポート行を作る．"""
    sfen, position_str, _ = analysis_gui.node_position_info(
        view, tree
    )
    return (
        '<div class="mw-strings">'
        f"<span>SFEN <code>{_esc(sfen)}</code></span>"
        '<button type="button" class="mw-btn mw-btn-ghost" '
        f'data-copy="{_esc(sfen)}">コピー</button>'
        '<span class="mw-strings-right">position 文字列</span>'
        '<button type="button" class="mw-btn mw-btn-ghost" '
        f'data-copy="{_esc(position_str)}">コピー</button>'
        "</div>"
    )


def _graph_points(
    view: SessionView, y_mode: str
) -> list[tuple[int, float]]:
    """グラフの (手数, 先手視点の値) 列を返す．"""
    if view.report is None:
        return []
    points: list[tuple[int, float]] = []
    for position in view.report["positions"]:
        side = position["side_to_move"]
        if y_mode == "eval_cp":
            value = sente_eval_cp(
                float(position["eval_cp"]), side
            )
            value = max(
                -_EVAL_CLIP_CP, min(_EVAL_CLIP_CP, value)
            )
            value = (value + _EVAL_CLIP_CP) / (
                2 * _EVAL_CLIP_CP
            )
        else:
            value = sente_winrate(
                float(position["winrate"]), side
            )
        points.append((int(position["ply"]), value))
    return points


def _eval_graph(
    view: SessionView,
    tree: VariationTree,
    options: WorkbenchOptions,
) -> str:
    """評価値グラフ (SVG) を作る．

    縦軸は 0-1 に正規化した先手視点の値 (評価値表示は ±2000cp で
    クリップして正規化する)．悪手・詰み発見を重ね，現在位置に縦線を
    引く．プロット領域のクリックでその手数へジャンプする．
    """
    label = (
        "勝率（先手）"
        if options.y_mode == "winrate"
        else "評価値（先手）"
    )
    seg = (
        '<div class="mw-seg">'
        f'<button type="button" class="mw-seg-opt" '
        f'aria-pressed="{str(options.y_mode == "winrate").lower()}" '
        'data-action="opt:ymode:winrate">勝率</button>'
        f'<button type="button" class="mw-seg-opt" '
        f'aria-pressed="{str(options.y_mode == "eval_cp").lower()}" '
        'data-action="opt:ymode:eval_cp">評価値</button>'
        "</div>"
    )
    head = (
        '<div class="mw-section-head">'
        f'<span class="mw-label">{_esc(label)}</span>{seg}</div>'
    )

    points = _graph_points(view, options.y_mode)
    if not points:
        return head + (
            '<div class="mw-empty">解析レポート未読込 — '
            "全局面解析を実行するとここに推移が出ます．</div>"
        )

    total = view.document.n_moves
    span = max(1, total)

    def px(ply: int) -> float:
        return _GRAPH_LEFT + (_GRAPH_RIGHT - _GRAPH_LEFT) * (
            ply / span
        )

    def py(value: float) -> float:
        return (
            _GRAPH_BOTTOM - (_GRAPH_BOTTOM - _GRAPH_TOP) * value
        )

    polyline = " ".join(
        f"{px(ply):.1f},{py(value):.1f}"
        for ply, value in points
    )
    value_by_ply = dict(points)

    marks: list[str] = []
    for ply, _, _ in blunders(view, options.threshold):
        value = value_by_ply.get(ply)
        if value is None:
            continue
        target = blunder_target_ply(ply)
        marks.append(
            f'<circle class="mw-graph-point" cx="{px(ply):.1f}" '
            f'cy="{py(value):.1f}" r="4" fill="var(--mw-accent)" '
            f'data-action="goto:{target}"><title>{ply}手目 悪手</title>'
            "</circle>"
        )
    for position in (
        view.report["positions"] if view.report else []
    ):
        if not position.get("mate_found"):
            continue
        ply = int(position["ply"])
        value = value_by_ply.get(ply)
        if value is None:
            continue
        marks.append(
            f'<text class="mw-graph-point" x="{px(ply):.1f}" '
            f'y="{py(value) - 6:.1f}" text-anchor="middle" '
            f'font-size="11" fill="var(--mw-accent-700)" '
            f'data-action="goto:{blunder_target_ply(ply)}">★'
            f"<title>{ply}手目 詰み発見</title></text>"
        )

    current = analysis_gui.mainline_ply(tree)
    cursor_x = px(current)
    centre_y = py(0.5)
    near_end = cursor_x > (_GRAPH_LEFT + _GRAPH_RIGHT) / 2
    label_anchor = "end" if near_end else "start"
    label_x = cursor_x - 4 if near_end else cursor_x + 4
    return head + (
        '<div class="mw-graph">'
        f'<svg viewBox="0 0 {_GRAPH_W} {_GRAPH_H}" '
        'preserveAspectRatio="none" role="img">'
        f'<rect x="{_GRAPH_LEFT}" y="{_GRAPH_TOP}" '
        f'width="{_GRAPH_RIGHT - _GRAPH_LEFT}" '
        f'height="{_GRAPH_BOTTOM - _GRAPH_TOP}" fill="transparent" '
        f'data-scrub="{total}" style="cursor:pointer"></rect>'
        f'<line x1="{_GRAPH_LEFT}" y1="{centre_y:.1f}" '
        f'x2="{_GRAPH_RIGHT}" y2="{centre_y:.1f}" '
        'stroke="var(--mw-divider)" stroke-width="1" '
        'stroke-dasharray="3 3"></line>'
        f'<polyline fill="none" stroke="var(--mw-text)" '
        f'stroke-width="1.5" points="{polyline}"></polyline>'
        + "".join(marks)
        + f'<line x1="{cursor_x:.1f}" y1="{_GRAPH_TOP}" '
        f'x2="{cursor_x:.1f}" y2="{_GRAPH_BOTTOM}" '
        'stroke="var(--mw-accent)" stroke-width="2"></line>'
        f'<text x="{label_x:.1f}" y="16" font-size="9" '
        f'text-anchor="{label_anchor}" '
        f'fill="var(--mw-accent)">{current}手目</text>'
        f'<text x="{_GRAPH_LEFT}" y="{_GRAPH_H - 3}" font-size="9" '
        'fill="var(--mw-neutral-700)">0</text>'
        f'<text x="{_GRAPH_RIGHT}" y="{_GRAPH_H - 3}" font-size="9" '
        'text-anchor="end" '
        f'fill="var(--mw-neutral-700)">{total}</text>'
        "</svg>"
        '<div class="mw-graph-legend">● 悪手をクリックでジャンプ　'
        "★ 詰み発見　グラフのクリックで手数移動</div></div>"
    )


def _candidates(
    tree: VariationTree,
    options: WorkbenchOptions,
    max_candidates: int,
) -> str:
    """候補手リストを作る．"""
    head = (
        '<div class="mw-section-head">'
        '<span class="mw-label">候補手 — 現局面</span>'
        '<span class="mw-candidate-opts">'
        '<span class="mw-seg">'
        '<button type="button" class="mw-seg-opt" '
        f'aria-pressed="{str(options.show_arrows).lower()}" '
        f'data-action="opt:arrows:{0 if options.show_arrows else 1}"'
        ">矢印</button>"
        '<button type="button" class="mw-seg-opt" '
        f'aria-pressed="{str(options.show_pv).lower()}" '
        f'data-action="opt:pv:{0 if options.show_pv else 1}"'
        ">PV</button></span>"
        '<input class="mw-topn" type="number" min="1" '
        f'max="{max_candidates}" step="1" value="{options.top_n}" '
        'title="候補手の表示件数" data-action-input="opt:topn">'
        "</span></div>"
    )
    rows = analysis_gui.node_candidates_table(
        tree, options.top_n
    )
    if not rows:
        return head + (
            '<div class="mw-empty">この局面は未解析です．'
            "「この局面を解析」で候補手が出ます．</div>"
        )
    items = []
    for index, row in enumerate(rows):
        rank, label, visits, winrate = (
            row[0],
            row[1],
            row[2],
            row[3],
        )
        classes = "mw-candidate" + (
            " is-best" if index == 0 else ""
        )
        items.append(
            f'<button type="button" class="{classes}" '
            f'data-action="cand:{index}">'
            f'<span class="mw-candidate-rank">{_esc(rank)}</span>'
            f'<span class="mw-move">{_esc(label)}</span>'
            f'<span class="mw-candidate-visits">{_esc(visits)}</span>'
            f'<span class="mw-candidate-winrate">{_esc(winrate)}</span>'
            "</button>"
        )
    shortcut_max = min(len(rows), _SHORTCUT_CANDIDATES)
    hint = (
        '<div class="mw-candidate-hint">'
        '<span class="mw-kbd">1</span>–'
        f'<span class="mw-kbd">{shortcut_max}</span> でその手に分岐</div>'
    )
    return head + "".join(items) + hint


def _analysis_panel(
    options: WorkbenchOptions,
    progress: AnalysisProgress,
    *,
    loaded: bool,
) -> str:
    """解析パネル (予算・1 局面解析・全局面解析) を作る．"""
    disabled = "" if loaded else " disabled"
    budget_seg = (
        '<div class="mw-seg">'
        '<button type="button" class="mw-seg-opt" '
        f'aria-pressed="{str(options.budget_mode == "time").lower()}" '
        'data-action="opt:budgetmode:time">時間 ms</button>'
        '<button type="button" class="mw-seg-opt" '
        f'aria-pressed="{str(options.budget_mode == "playouts").lower()}" '
        'data-action="opt:budgetmode:playouts">playouts</button>'
        "</div>"
    )
    percent = (
        (progress.done / progress.total * 100)
        if progress.total
        else 0.0
    )
    if progress.running:
        progress_label = (
            f"{progress.done} / {progress.total} 局面"
        )
    elif progress.total and progress.done >= progress.total:
        progress_label = f"完了（{progress.total} 局面）"
    else:
        progress_label = "未実行"
    all_label = "キャンセル" if progress.running else "開始"
    all_action = "cancel" if progress.running else "analyzeall"
    status_classes = "mw-status" + (
        " is-error" if progress.is_error else ""
    )
    return (
        '<div class="mw-analysis">'
        f'<div class="mw-budget">{budget_seg}'
        f'<input type="number" min="1" step="1" '
        f'value="{int(options.budget_value)}" '
        f'data-action-input="opt:budget"{disabled}></div>'
        '<button type="button" class="mw-btn mw-btn-primary mw-btn-block" '
        f'data-action="analyze"{disabled}>この局面を解析'
        '<span style="opacity:.75;margin-left:auto">'
        '<span class="mw-kbd" style="border-color:currentColor">Space</span>'
        "</span></button>"
        '<div class="mw-btn-row">'
        '<button type="button" class="mw-btn mw-btn-secondary" '
        f'data-action="reanalyze"{disabled}>再解析</button>'
        '<button type="button" class="mw-btn mw-btn-secondary" '
        f'data-action="pv"{disabled}>PVを分岐で再生</button></div>'
        f'<div class="{status_classes}">{_esc(progress.status)}</div>'
        '<div class="mw-analyze-all">'
        '<div class="mw-analyze-all-head"><strong>全局面解析</strong>'
        f'<span class="mw-note">{_esc(progress_label)}</span></div>'
        '<div class="mw-progress">'
        f'<div class="mw-progress-fill" style="width:{percent:.1f}%">'
        "</div></div>"
        '<div class="mw-btn-row">'
        '<button type="button" class="mw-btn mw-btn-secondary" '
        f'data-action="{all_action}"{disabled}>{all_label}</button>'
        '<button type="button" class="mw-btn mw-btn-secondary" '
        f'data-action="save"{disabled}>JSON保存</button></div>'
        "</div></div>"
    )


def _board(
    tree: VariationTree,
    click: ClickState,
    options: WorkbenchOptions,
) -> str:
    """盤面 SVG (Modernist テーマ・クリック標的付き) を作る．"""
    svg = analysis_gui.node_board_svg(
        tree,
        show_candidates=options.show_arrows,
        show_pv=options.show_pv,
        top_n=options.top_n,
        click_state=click,
        legal=analysis_gui.node_legal_moves(tree),
        interactive=True,
        theme=MODERNIST_BOARD_THEME,
    )
    return f'<div class="mw-board">{svg}</div>'


# ----------------------------------------------------------------------
# 全体組み立て
# ----------------------------------------------------------------------


def _placeholder(
    source_label: str, engine_label: str, is_mock: bool
) -> str:
    """棋譜未読込のワークベンチを作る．"""
    return (
        '<div id="maou-workbench" data-render="empty">'
        + _header(source_label, engine_label, is_mock)
        + '<div class="mw-placeholder"><div>'
        "<h2>棋譜が読み込まれていません</h2>"
        "<p>下の「棋譜/レポートの読み込み」から CSA / KIF を"
        "読み込んでください．</p>"
        '<button type="button" class="mw-btn mw-btn-primary" '
        'data-open-file="1">ファイルを選ぶ</button>'
        "</div></div></div>"
    )


def render_workbench(
    view: SessionView | None,
    tree: VariationTree | None,
    click: ClickState,
    options: WorkbenchOptions,
    progress: AnalysisProgress,
    *,
    engine_label: str,
    is_mock: bool,
    max_candidates: int = 5,
    render_stamp: str = "",
) -> str:
    """ワークベンチ全体の HTML を返す．

    Args:
        view: セッション状態．None で未読込プレースホルダ．
        tree: 分岐木．None で未読込プレースホルダ．
        click: 盤面クリックの進行状態．
        options: 表示・解析オプション．
        progress: 全局面解析の進捗と解析ステータス．
        engine_label: エンジン表示名 (mock 明示を含む)．
        is_mock: mock 評価器か (タグの見た目を変える)．
        max_candidates: 候補手表示数の上限 (CLI の --num-candidates)．
        render_stamp: 再描画を JS 側に知らせる任意の識別子．

    Returns:
        ``#maou-workbench`` をルートとする HTML 文字列．
    """
    source_label = (
        view.source_name if view is not None else "未読込"
    )
    if view is None or tree is None:
        return _placeholder(source_label, engine_label, is_mock)

    legal_panel = (
        _legal_moves(tree) if options.legal_open else ""
    )
    return (
        f'<div id="maou-workbench" data-render="{_esc(render_stamp)}">'
        + _header(source_label, engine_label, is_mock)
        + _summary_strip(view)
        + '<div class="mw-main">'
        + '<div class="mw-rail-left">'
        + _blunder_rail(view, options)
        + _kifu_list(view, tree, options)
        + "</div>"
        + '<div class="mw-center">'
        + _position_bar(view, tree)
        + _board(tree, click, options)
        + _nav_bar(view, tree)
        + '<div class="mw-center-foot">'
        + _hint_bar(view, tree, click, options)
        + legal_panel
        + _position_strings(view, tree)
        + "</div></div>"
        + '<div class="mw-rail-right">'
        + _eval_graph(view, tree, options)
        + _candidates(tree, options, max_candidates)
        + _analysis_panel(options, progress, loaded=True)
        + "</div></div></div>"
    )
