"""棋譜解析 GUI (analyze-gui) の interface アダプタ．

App 層の :mod:`maou.app.analysis.analysis_session` と Infra 層の Gradio UI
を接続する．盤面 SVG (候補手矢印付き)・評価値グラフ (Plotly)・棋譜/候補手
テーブル・局面情報の整形を担う．

analyze-game JSON の ``winrate`` / ``eval_cp`` は手番視点のため，グラフ表示
時に先手視点へ変換する (:func:`sente_winrate` / :func:`sente_eval_cp`)．
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import plotly.graph_objects as go

# infra 層 (analysis_gui_server) が本モジュール経由で使う app 層 API の
# 再エクスポート (サーバーは interface 層のみを import する)
from maou.app.analysis.analysis_session import (
    GameDocument,
    LegalMoveInfo,
    PositionSnapshot,
    VariationTree,
)
from maou.app.analysis.analysis_session import (  # noqa: F401
    advance_move as advance_move,
)
from maou.app.analysis.analysis_session import (
    build_variation_tree as build_variation_tree,
)
from maou.app.analysis.analysis_session import (
    current_node,
)
from maou.app.analysis.analysis_session import (
    goto_node as goto_node,
)
from maou.app.analysis.analysis_session import (
    legal_move_infos,
    load_game,
    mainline_ancestor,
    path_moves_usi,
    validate_report,
)
from maou.app.analysis.interactive_analyzer import (  # noqa: F401
    DEFAULT_TIME_MS as DEFAULT_TIME_MS,
)
from maou.app.analysis.interactive_analyzer import (
    EngineSettings as EngineSettings,
)
from maou.app.analysis.interactive_analyzer import (
    InteractiveAnalyzer as InteractiveAnalyzer,
)
from maou.domain.board.shogi import (
    HAND_PIECE_SFEN_CHARS,
    Turn,
)
from maou.domain.visualization.board_renderer import (
    ArrowSpec,
    BoardPosition,
    MoveArrow,
    SVGBoardRenderer,
)
from maou.domain.visualization.piece_mapping import (
    get_piece_name_ja,
)
from maou.interface.analyzer import resolve_input_format
from maou.interface.game_graph_visualization import (
    GameGraphVisualizationInterface,
)

logger: logging.Logger = logging.getLogger(__name__)

# 手番の先頭記号
_SIDE_MARK = {"b": "▲", "w": "△"}

# 矢印スタイル: 最善手は濃い赤で太く，次善以下は青 (訪問数比で不透明度)
BEST_ARROW_COLOR = "rgb(214, 69, 65)"
CANDIDATE_ARROW_COLOR = "rgb(0, 100, 200)"
BEST_ARROW_WIDTH = 6
CANDIDATE_ARROW_WIDTH = 4

# 棋譜テーブルの列 (レポート未読込時は解析列が空文字列)
MOVE_TABLE_HEADERS = [
    "手数",
    "指し手",
    "一致",
    "勝率(先手)",
    "評価値(先手)",
    "勝率損失",
    "詰み",
]

CANDIDATES_TABLE_HEADERS = [
    "順位",
    "指し手",
    "訪問数",
    "勝率(手番)",
    "prior",
    "確定値",
]


@dataclass(frozen=True)
class SessionView:
    """GUI 1 セッション分の閲覧状態 (plain data)．

    Attributes:
        document: 棋譜の per-ply スナップショット表現．
        report: analyze-game の JSON レポート (dict)．未読込は None．
        move_labels: 本譜の日本語表記 (▲/△ 付き，手数分)．
        source_name: 読み込んだ棋譜のファイル名 (GUI 内で生成する
            レポートの ``input.path`` に記録する)．
    """

    document: GameDocument
    report: dict[str, Any] | None
    move_labels: list[str]
    source_name: str = ""


def sente_winrate(winrate: float, side_to_move: str) -> float:
    """手番視点の勝率を先手視点に変換する．"""
    return winrate if side_to_move == "b" else 1.0 - winrate


def sente_eval_cp(eval_cp: float, side_to_move: str) -> float:
    """手番視点の評価値 (cp) を先手視点に変換する．"""
    return eval_cp if side_to_move == "b" else -eval_cp


def _usi_square(text: str) -> int:
    """USI のマス表記 (例: "7g") を column-major マス番号にする．"""
    file = int(text[0])
    rank = ord(text[1]) - ord("a") + 1
    if not (1 <= file <= 9 and 1 <= rank <= 9):
        raise ValueError(f"不正な USI マス表記です: {text}")
    return (file - 1) * 9 + (rank - 1)


def usi_to_move_arrow(usi: str) -> MoveArrow:
    """USI 指し手文字列を盤面描画用の MoveArrow にする．

    Args:
        usi: USI 表記 (例: "7g7f", "P*5e")．

    Returns:
        MoveArrow (駒打ちは持ち駒エリア始点)．

    Raises:
        ValueError: USI として解釈できない場合．
    """
    if len(usi) < 4:
        raise ValueError(f"不正な USI 指し手です: {usi}")
    if usi[1] == "*":
        if usi[0] not in HAND_PIECE_SFEN_CHARS:
            raise ValueError(f"不正な駒打ち USI です: {usi}")
        return MoveArrow(
            from_square=None,
            to_square=_usi_square(usi[2:4]),
            is_drop=True,
            drop_piece_type=HAND_PIECE_SFEN_CHARS.index(usi[0]),
        )
    return MoveArrow(
        from_square=_usi_square(usi[0:2]),
        to_square=_usi_square(usi[2:4]),
    )


def _highlight_index(square: int) -> int:
    """column-major マス番号を renderer のハイライト索引にする．

    :meth:`SVGBoardRenderer.render` の ``highlight_squares`` は
    [row][col] 行優先 (row * 9 + col) 索引を取る (矢印とは索引系が
    異なる — 既存仕様)．
    """
    row, col = square % 9, square // 9
    return row * 9 + col


def _snapshot_piece_name(
    snapshot: PositionSnapshot, usi: str
) -> str:
    """スナップショット盤面から USI 指し手の移動元駒名を得る．"""
    if usi[1] == "*":
        return ""
    square = _usi_square(usi[0:2])
    row, col = square % 9, square // 9
    return get_piece_name_ja(
        snapshot.board_id_positions[row][col]
    )


def _move_label(
    snapshot_before: PositionSnapshot, usi: str
) -> str:
    """指し手を「▲7六歩」形式の日本語表記にする．"""
    japanese = GameGraphVisualizationInterface.usi_to_japanese(
        usi,
        piece_name=_snapshot_piece_name(snapshot_before, usi),
    )
    return f"{_SIDE_MARK[snapshot_before.turn]}{japanese}"


def load_session(
    kifu_bytes: bytes,
    filename: str,
    report_json: str | None = None,
) -> SessionView:
    """棋譜 (+ 任意で解析レポート) から SessionView を構築する．

    Args:
        kifu_bytes: 棋譜ファイルの生バイト列．
        filename: 棋譜ファイル名 (拡張子から形式を判定)．
        report_json: analyze-game の JSON レポート文字列．None なら
            レポートなし (盤面再生のみ)．

    Returns:
        SessionView (gr.State 保持用の plain data)．

    Raises:
        ValueError: 棋譜のパース失敗，またはレポートの不整合．
    """
    input_format = resolve_input_format(Path(filename), None)
    document = load_game(kifu_bytes, input_format)
    report: dict[str, Any] | None = None
    if report_json:
        report = json.loads(report_json)
        assert report is not None
        validate_report(document, report)
    move_labels = [
        _move_label(document.snapshots[i], usi)
        for i, usi in enumerate(document.moves_usi)
    ]
    return SessionView(
        document=document,
        report=report,
        move_labels=move_labels,
        source_name=filename,
    )


def position_analysis(
    view: SessionView, ply: int
) -> dict[str, Any] | None:
    """局面 ply (スナップショット番号) の解析記録を返す．

    レポートの ``positions[i]`` は「i+1 手目を指す直前の局面」=
    スナップショット i の解析なので，そのまま索引する．最終局面
    (ply = n_moves) の解析は存在しないため None．
    """
    if view.report is None:
        return None
    if not 0 <= ply < view.document.n_moves:
        return None
    positions = view.report["positions"]
    result: dict[str, Any] = positions[ply]
    return result


def board_svg(
    view: SessionView,
    ply: int,
    *,
    show_candidates: bool = True,
    show_pv: bool = False,
    top_n: int = 5,
) -> str:
    """局面 ply の盤面 SVG (最終手ハイライト + 候補手矢印) を返す．

    Args:
        view: セッション状態．
        ply: スナップショット番号 (0 = 初期局面)．
        show_candidates: 候補手矢印を描画するか．
        show_pv: 最善手の PV 先頭 3 手を連鎖矢印で描画するか．
        top_n: 描画する候補手の数．

    Returns:
        SVG 文字列．
    """
    return snapshot_board_svg(
        view.document.snapshots[ply],
        position_analysis(view, ply),
        show_candidates=show_candidates,
        show_pv=show_pv,
        top_n=top_n,
    )


def snapshot_board_svg(
    snapshot: PositionSnapshot,
    analysis: dict[str, Any] | None,
    *,
    show_candidates: bool = True,
    show_pv: bool = False,
    top_n: int = 5,
    selected_squares: list[int] | None = None,
    destination_squares: list[int] | None = None,
    interactive: bool = False,
) -> str:
    """スナップショット局面の盤面 SVG を返す (分岐局面にも使える)．

    Args:
        snapshot: 対象局面のスナップショット．
        analysis: この局面の解析記録 (候補手矢印の元．None で矢印なし)．
        show_candidates: 候補手矢印を描画するか．
        show_pv: 最善手の PV 先頭 3 手を連鎖矢印で描画するか．
        top_n: 描画する候補手の数．
        selected_squares: クリック選択中として塗るマス (row-major)．
        destination_squares: 行き先候補として塗るマス (row-major)．
        interactive: クリック標的 rect を重ねるか．

    Returns:
        SVG 文字列．
    """
    position = BoardPosition(
        board_id_positions=snapshot.board_id_positions,
        pieces_in_hand=snapshot.pieces_in_hand,
    )

    highlights: list[int] = []
    if snapshot.last_move is not None:
        highlights.append(
            _highlight_index(snapshot.last_move.to_square)
        )
        if snapshot.last_move.from_square is not None:
            highlights.append(
                _highlight_index(snapshot.last_move.from_square)
            )

    arrows: list[ArrowSpec] = []
    if analysis is not None and show_candidates:
        arrows = _candidate_arrows(
            analysis, top_n=top_n, show_pv=show_pv
        )

    turn = Turn.BLACK if snapshot.turn == "b" else Turn.WHITE
    return SVGBoardRenderer().render(
        position,
        highlight_squares=highlights,
        turn=turn,
        move_arrows=arrows,
        selected_squares=selected_squares,
        destination_squares=destination_squares,
        interactive=interactive,
    )


def _candidate_arrows(
    analysis: dict[str, Any], *, top_n: int, show_pv: bool
) -> list[ArrowSpec]:
    """解析記録から候補手矢印 (+ 任意で PV 連鎖矢印) を作る．"""
    arrows: list[ArrowSpec] = []
    candidates = list(analysis.get("candidates") or [])[
        : max(top_n, 0)
    ]
    best_usi = analysis.get("best_move")
    max_visits = max(
        (int(c.get("visits", 0)) for c in candidates),
        default=0,
    )
    for rank, candidate in enumerate(candidates, start=1):
        usi = candidate.get("usi")
        if not usi:
            continue
        try:
            move = usi_to_move_arrow(usi)
        except ValueError:
            logger.warning(
                "候補手の USI を解釈できません: %s", usi
            )
            continue
        is_best = usi == best_usi
        if is_best:
            opacity = 0.95
        elif max_visits > 0:
            visits = int(candidate.get("visits", 0))
            opacity = 0.25 + 0.5 * (visits / max_visits)
        else:
            opacity = 0.5
        arrows.append(
            ArrowSpec(
                move=move,
                color=(
                    BEST_ARROW_COLOR
                    if is_best
                    else CANDIDATE_ARROW_COLOR
                ),
                width=(
                    BEST_ARROW_WIDTH
                    if is_best
                    else CANDIDATE_ARROW_WIDTH
                ),
                opacity=round(opacity, 3),
                label=str(rank),
            )
        )

    if show_pv:
        pv = list(analysis.get("pv") or [])
        # pv[0] は最善手 (候補手矢印と重複) なので 2 手目以降を描く
        for depth, usi in enumerate(pv[1:3], start=1):
            try:
                move = usi_to_move_arrow(usi)
            except ValueError:
                break
            arrows.append(
                ArrowSpec(
                    move=move,
                    color=BEST_ARROW_COLOR,
                    width=CANDIDATE_ARROW_WIDTH,
                    opacity=round(0.45 - 0.15 * depth, 3),
                )
            )
    return arrows


def eval_figure(
    view: SessionView,
    current_ply: int,
    y_mode: str = "winrate",
) -> go.Figure:
    """評価値グラフ (先手視点) を作る．

    x 軸は手数 (positions[i].ply = 「その手を指す直前の局面」)．
    現在表示中のスナップショット p は x = p + 1 に対応する縦線で示す．

    Args:
        view: セッション状態．
        current_ply: 表示中のスナップショット番号．
        y_mode: "winrate" (先手勝率) または "eval_cp" (先手評価値)．

    Returns:
        Plotly Figure (レポート未読込時は案内文のみ)．
    """
    fig = go.Figure()
    if view.report is None:
        fig.add_annotation(
            text=(
                "解析レポート未読込 "
                "(analyze-game の JSON を読み込むと表示されます)"
            ),
            showarrow=False,
            font={"size": 14},
        )
        fig.update_layout(
            xaxis={"visible": False},
            yaxis={"visible": False},
            height=320,
            margin={"l": 20, "r": 20, "t": 30, "b": 20},
        )
        return fig

    positions = view.report["positions"]
    x = [int(p["ply"]) for p in positions]
    if y_mode == "eval_cp":
        y = [
            sente_eval_cp(
                float(p["eval_cp"]), p["side_to_move"]
            )
            for p in positions
        ]
        y_title = "評価値 (先手視点, cp)"
        center = 0.0
    else:
        y = [
            sente_winrate(
                float(p["winrate"]), p["side_to_move"]
            )
            for p in positions
        ]
        y_title = "勝率 (先手視点)"
        center = 0.5

    hover = [
        f"{p['ply']}手目 {label}"
        for p, label in zip(positions, view.move_labels)
    ]
    fig.add_hline(y=center, line_width=1, line_color="#bbbbbb")
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            name=y_title,
            text=hover,
            hovertemplate="%{text}<br>%{y:.3f}<extra></extra>",
            line={"color": "#2b6cb0", "width": 2},
            marker={"size": 5},
        )
    )

    mate_x = [
        int(p["ply"]) for p in positions if p["mate_found"]
    ]
    if mate_x:
        mate_y = [
            y[i]
            for i, p in enumerate(positions)
            if p["mate_found"]
        ]
        fig.add_trace(
            go.Scatter(
                x=mate_x,
                y=mate_y,
                mode="markers",
                name="詰み発見",
                marker={
                    "symbol": "star",
                    "size": 12,
                    "color": "#d69e2e",
                },
                hovertemplate="詰み発見 (%{x}手目)<extra></extra>",
            )
        )

    if 0 <= current_ply < view.document.n_moves:
        fig.add_vline(
            x=current_ply + 1,
            line_width=2,
            line_color="rgba(214, 69, 65, 0.6)",
        )

    if y_mode != "eval_cp":
        fig.update_yaxes(range=[0.0, 1.0])
    fig.update_layout(
        xaxis_title="手数 (指す直前の局面)",
        yaxis_title=y_title,
        height=320,
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
        showlegend=False,
    )
    return fig


def _fmt(value: Any, spec: str = ".3f") -> str:
    """None を空文字列にする表示用フォーマッタ．"""
    if value is None:
        return ""
    return format(float(value), spec)


def move_table(view: SessionView) -> list[list[str]]:
    """棋譜テーブル (手数/指し手/解析列) の行リストを返す．

    レポート未読込時は解析列が空文字列になる．
    """
    rows: list[list[str]] = []
    for i, label in enumerate(view.move_labels):
        analysis = position_analysis(view, i)
        if analysis is None:
            rows.append([str(i + 1), label, "", "", "", "", ""])
            continue
        side = analysis["side_to_move"]
        rows.append(
            [
                str(i + 1),
                label,
                "✓" if analysis["match"] else "",
                _fmt(
                    sente_winrate(
                        float(analysis["winrate"]), side
                    )
                ),
                _fmt(
                    sente_eval_cp(
                        float(analysis["eval_cp"]), side
                    ),
                    ".0f",
                ),
                _fmt(analysis["winrate_loss"]),
                "★" if analysis["mate_found"] else "",
            ]
        )
    return rows


def candidates_table(
    view: SessionView, ply: int, top_n: int = 5
) -> list[list[str]]:
    """局面 ply の候補手テーブル (上位 top_n) の行リストを返す．"""
    return _candidate_rows(
        view.document.snapshots[ply],
        position_analysis(view, ply),
        top_n,
    )


def _candidate_rows(
    snapshot: PositionSnapshot,
    analysis: dict[str, Any] | None,
    top_n: int,
) -> list[list[str]]:
    """解析記録から候補手テーブルの行リストを作る．"""
    if analysis is None:
        return []
    rows: list[list[str]] = []
    candidates = list(analysis.get("candidates") or [])[
        : max(top_n, 0)
    ]
    for rank, candidate in enumerate(candidates, start=1):
        usi = candidate.get("usi", "")
        try:
            label = _move_label(snapshot, usi)
        except (ValueError, IndexError):
            label = usi
        proven = candidate.get("proven")
        rows.append(
            [
                str(rank),
                label,
                str(int(candidate.get("visits", 0))),
                _fmt(candidate.get("winrate")),
                _fmt(candidate.get("prior")),
                "" if proven is None else f"{float(proven):g}",
            ]
        )
    return rows


def summary_markdown(view: SessionView) -> str:
    """対局・解析サマリの Markdown を返す．"""
    doc = view.document
    names = [n or "?" for n in doc.names]
    win_text = {
        0: "引き分け",
        1: "先手勝ち",
        2: "後手勝ち",
    }.get(doc.win if doc.win is not None else -1, "不明")
    lines = [
        f"**対局**: ▲{names[0]} vs △{names[1]} "
        f"({doc.n_moves} 手, {win_text}"
        + (f", {doc.endgame}" if doc.endgame else "")
        + ")",
    ]
    if view.report is None:
        lines.append(
            "解析レポート未読込 — 盤面再生のみ利用できます．"
        )
        return "\n\n".join(lines)

    summary = view.report.get("summary", {})
    match_rate = summary.get("match_rate", {})
    mean_loss = summary.get("mean_winrate_loss", {})

    def _rate(value: Any) -> str:
        return (
            f"{float(value) * 100:.1f}%"
            if value is not None
            else "-"
        )

    lines.append(
        "**一致率**: 先手 "
        f"{_rate(match_rate.get('black'))} / 後手 "
        f"{_rate(match_rate.get('white'))}　"
        "**平均勝率損失**: 先手 "
        f"{_fmt(mean_loss.get('black'))} / 後手 "
        f"{_fmt(mean_loss.get('white'))}"
    )
    worst = summary.get("worst_moves") or []
    if worst:
        items = ", ".join(
            f"{w['ply']}手目 ({_fmt(w['winrate_loss'])})"
            for w in worst[:5]
        )
        lines.append(f"**勝率損失の大きい手**: {items}")
    mates = summary.get("mates_found") or []
    if mates:
        items = ", ".join(f"{m['ply']}手目" for m in mates)
        lines.append(f"**詰み発見**: {items}")
    return "\n\n".join(lines)


def position_info(
    view: SessionView, ply: int
) -> tuple[str, str, str]:
    """局面情報 (SFEN / position 文字列 / 注記 Markdown) を返す．

    注記には直前の手の解析比較 (実戦手 vs 最善手) と棋譜側の記録
    (消費時間・コメント) を含める．
    """
    doc = view.document
    snapshot = doc.snapshots[ply]
    sfen = snapshot.sfen
    if ply == 0:
        position_str = f"position sfen {doc.snapshots[0].sfen}"
    else:
        moves = " ".join(doc.moves_usi[:ply])
        position_str = (
            f"position sfen {doc.snapshots[0].sfen} "
            f"moves {moves}"
        )

    notes: list[str] = []
    if ply == 0:
        notes.append("初期局面")
    else:
        notes.append(
            f"**{ply}手目**: {view.move_labels[ply - 1]}"
        )
        analysis = position_analysis(view, ply - 1)
        if analysis is not None:
            best = analysis.get("best_move")
            if best:
                best_label = _move_label(
                    doc.snapshots[ply - 1], best
                )
                if analysis["match"]:
                    notes.append(
                        f"エンジン最善と一致 ({best_label})"
                    )
                else:
                    loss = _fmt(analysis["winrate_loss"])
                    notes.append(
                        f"エンジン最善: {best_label}"
                        + (
                            f" (勝率損失 {loss})"
                            if loss
                            else " (実戦手は未訪問)"
                        )
                    )
        if ply - 1 < len(doc.times):
            notes.append(f"消費時間: {doc.times[ply - 1]}s")
        if (
            ply - 1 < len(doc.comments)
            and doc.comments[ply - 1]
        ):
            notes.append(f"コメント: {doc.comments[ply - 1]}")
    return sfen, position_str, "\n\n".join(notes)


# ----------------------------------------------------------------------
# 対話解析 (分岐木 + 盤面クリック入力)
# ----------------------------------------------------------------------

# 段の漢数字 (指し手・マス表記用)
_RANK_KANJI = "一二三四五六七八九"


@dataclass(frozen=True)
class ClickState:
    """盤面クリック入力の進行状態 (plain data)．

    2 クリック方式 (docs/design/game-analysis/gui.md §10) の状態機械:
    選択なし → 自駒/持ち駒クリックで選択 → 行き先クリックで確定．
    成/不成が両方合法な行き先では ``pending_usis`` に両候補を保持し，
    UI の確認ボタンで確定する．

    Attributes:
        selected: 選択中の起点 (``data-click`` 値: "sq:60" /
            "hand:b:0")．未選択は None．
        pending_usis: 成/不成の選択待ち USI (成る手, 成らない手)．
    """

    selected: str | None = None
    pending_usis: tuple[str, ...] = ()


def node_legal_moves(
    tree: VariationTree,
) -> list[LegalMoveInfo]:
    """現在ノードの局面の合法手を列挙する．"""
    return legal_move_infos(current_node(tree).snapshot)


def _moves_from_origin(
    legal: list[LegalMoveInfo], origin: str, turn: str
) -> list[LegalMoveInfo]:
    """クリック値 (起点) に対応する合法手を返す．

    "sq:N" は移動元マス N の手，"hand:{side}:{type}" は手番側の
    その駒種の駒打ち (相手側の持ち駒は空リスト)．
    """
    parts = origin.split(":")
    try:
        if parts[0] == "sq" and len(parts) == 2:
            square = int(parts[1])
            return [
                m
                for m in legal
                if not m.is_drop and m.from_square == square
            ]
        if parts[0] == "hand" and len(parts) == 3:
            if parts[1] != turn:
                return []
            piece_type = int(parts[2])
            return [
                m
                for m in legal
                if m.is_drop and m.drop_piece_type == piece_type
            ]
    except ValueError:
        logger.warning("不正なクリック値です: %s", origin)
    return []


def handle_board_click(
    legal: list[LegalMoveInfo],
    state: ClickState,
    value: str,
    turn: str,
) -> tuple[ClickState, str | None]:
    """盤面クリック 1 回分の状態遷移を計算する (純関数)．

    Args:
        legal: 現局面の合法手．
        state: 現在のクリック状態．
        value: クリックされた標的の ``data-click`` 値．
        turn: 手番 ("b" / "w")．

    Returns:
        ``(新しい ClickState, 指す USI or None)``．成/不成の選択待ちに
        入った場合は USI は None で ``pending_usis`` に両候補が入る．
    """
    value = (value or "").strip()
    if not value:
        return ClickState(), None
    if state.pending_usis:
        # 確認ボタン以外のクリックはキャンセルし，新規クリックとして解釈
        state = ClickState()

    clicked_origin_moves = _moves_from_origin(
        legal, value, turn
    )
    if state.selected is None:
        if clicked_origin_moves:
            return ClickState(selected=value), None
        return ClickState(), None

    if value == state.selected:
        # 同じ起点の再クリックで選択解除
        return ClickState(), None

    selected_moves = _moves_from_origin(
        legal, state.selected, turn
    )
    if value.startswith("sq:"):
        try:
            to_square = int(value.split(":")[1])
        except ValueError:
            return ClickState(), None
        matches = [
            m
            for m in selected_moves
            if m.to_square == to_square
        ]
        if len(matches) == 1:
            return ClickState(), matches[0].usi
        if len(matches) >= 2:
            promo = next(
                (m.usi for m in matches if m.is_promotion),
                None,
            )
            nonpromo = next(
                (m.usi for m in matches if not m.is_promotion),
                None,
            )
            if promo is not None and nonpromo is not None:
                return (
                    ClickState(
                        selected=state.selected,
                        pending_usis=(promo, nonpromo),
                    ),
                    None,
                )

    # 行き先ではない → 有効な起点なら選択切替，そうでなければ解除
    if clicked_origin_moves:
        return ClickState(selected=value), None
    return ClickState(), None


def click_overlays(
    legal: list[LegalMoveInfo],
    state: ClickState,
    turn: str,
) -> tuple[list[int], list[int]]:
    """クリック状態から盤面の塗り分け (選択/行き先) を作る．

    Returns:
        ``(selected_squares, destination_squares)``．いずれも
        renderer の row-major 索引．
    """
    if state.selected is None:
        return [], []
    selected: list[int] = []
    if state.selected.startswith("sq:"):
        try:
            selected.append(
                _highlight_index(
                    int(state.selected.split(":")[1])
                )
            )
        except ValueError:
            return [], []
    destinations = sorted(
        {
            _highlight_index(m.to_square)
            for m in _moves_from_origin(
                legal, state.selected, turn
            )
        }
    )
    return selected, destinations


def click_status_text(
    snapshot: PositionSnapshot, state: ClickState
) -> str:
    """クリック状態の説明テキストを返す (手入力状態の表示用)．"""
    if state.pending_usis:
        return "成 / 不成を選択してください"
    if state.selected is None:
        return (
            "盤面クリック: 動かす駒 (または持ち駒) をクリック"
        )
    parts = state.selected.split(":")
    if parts[0] == "sq":
        try:
            square = int(parts[1])
        except ValueError:
            return ""
        row, col = square % 9, square // 9
        piece = get_piece_name_ja(
            snapshot.board_id_positions[row][col]
        )
        return (
            f"選択中: {col + 1}{_RANK_KANJI[row]}{piece} — "
            "行き先をクリック"
        )
    if parts[0] == "hand" and len(parts) == 3:
        try:
            piece = get_piece_name_ja(int(parts[2]) + 1)
        except ValueError:
            return ""
        return f"選択中: 持ち駒の{piece} — 打つマスをクリック"
    return ""


def legal_move_choices(
    snapshot: PositionSnapshot,
    legal: list[LegalMoveInfo],
) -> list[tuple[str, str]]:
    """合法手 Dropdown の選択肢 (日本語表記, USI) を返す．

    盤面クリックが使えない環境向けのフォールバック入力
    (docs/design/game-analysis/gui.md §10)．
    """
    choices: list[tuple[str, str]] = []
    for move in legal:
        try:
            label = _move_label(snapshot, move.usi)
        except (ValueError, IndexError):
            label = move.usi
        choices.append((f"{label} ({move.usi})", move.usi))
    return choices


def breadcrumb_markdown(tree: VariationTree) -> str:
    """分岐パンくず (本譜 N手目 ▶ △8四飛 ▶ …) の Markdown を返す．"""
    node = current_node(tree)
    if node.is_mainline:
        if node.snapshot.ply == 0:
            return "**本譜** 初期局面"
        return f"**本譜** {node.snapshot.ply}手目"
    chain = []
    cursor = node
    while not cursor.is_mainline:
        assert cursor.parent_id is not None
        parent = tree.nodes[cursor.parent_id]
        assert cursor.move_usi is not None
        try:
            label = _move_label(
                parent.snapshot, cursor.move_usi
            )
        except (ValueError, IndexError):
            label = cursor.move_usi
        chain.append(label)
        cursor = parent
    chain.reverse()
    base = (
        f"**本譜 {cursor.snapshot.ply}手目**"
        if cursor.snapshot.ply > 0
        else "**本譜 初期局面**"
    )
    return base + " ▶ " + " ▶ ".join(chain)


def mainline_ply(tree: VariationTree) -> int:
    """現在ノードに最も近い本譜局面の ply を返す．

    分岐中はスライダー・評価値グラフの現在位置線を分岐点に保つ．
    """
    return mainline_ancestor(tree).snapshot.ply


def node_board_svg(
    tree: VariationTree,
    *,
    show_candidates: bool = True,
    show_pv: bool = False,
    top_n: int = 5,
    click_state: ClickState | None = None,
    legal: list[LegalMoveInfo] | None = None,
    interactive: bool = False,
) -> str:
    """現在ノードの盤面 SVG (解析キャッシュの候補手矢印付き) を返す．"""
    node = current_node(tree)
    selected: list[int] = []
    destinations: list[int] = []
    if click_state is not None and legal is not None:
        selected, destinations = click_overlays(
            legal, click_state, node.snapshot.turn
        )
    return snapshot_board_svg(
        node.snapshot,
        node.analysis,
        show_candidates=show_candidates,
        show_pv=show_pv,
        top_n=top_n,
        selected_squares=selected,
        destination_squares=destinations,
        interactive=interactive,
    )


def node_candidates_table(
    tree: VariationTree, top_n: int = 5
) -> list[list[str]]:
    """現在ノードの候補手テーブル (上位 top_n) の行リストを返す．"""
    node = current_node(tree)
    return _candidate_rows(node.snapshot, node.analysis, top_n)


def candidate_usi(
    analysis: dict[str, Any] | None,
    row_index: int,
    top_n: int,
) -> str | None:
    """候補手テーブルの行番号から USI を引く (行クリック → 分岐用)．

    行の並びは :func:`_candidate_rows` と同じ
    (``candidates`` の先頭 top_n 件)．
    """
    if analysis is None:
        return None
    candidates = list(analysis.get("candidates") or [])[
        : max(top_n, 0)
    ]
    if 0 <= row_index < len(candidates):
        usi = candidates[row_index].get("usi")
        return str(usi) if usi else None
    return None


def node_position_info(
    view: SessionView, tree: VariationTree
) -> tuple[str, str, str]:
    """現在ノードの局面情報 (SFEN / position 文字列 / 注記) を返す．

    本譜ノードは :func:`position_info` に委譲し，分岐ノードは分岐点と
    分岐手順の注記を作る．
    """
    node = current_node(tree)
    if node.is_mainline:
        return position_info(view, node.snapshot.ply)
    doc = view.document
    path = path_moves_usi(tree, node.node_id)
    position_str = (
        f"position sfen {doc.snapshots[0].sfen} "
        f"moves {' '.join(path)}"
    )
    ancestor = mainline_ancestor(tree, node.node_id)
    notes = [
        f"**分岐** (本譜 {ancestor.snapshot.ply}手目から "
        f"{node.snapshot.ply - ancestor.snapshot.ply} 手)"
    ]
    assert node.parent_id is not None
    assert node.move_usi is not None
    parent = tree.nodes[node.parent_id]
    try:
        label = _move_label(parent.snapshot, node.move_usi)
    except (ValueError, IndexError):
        label = node.move_usi
    notes.append(f"**{node.snapshot.ply}手目**: {label}")
    if node.analysis is None:
        notes.append(
            "この局面は未解析です "
            "(「この局面を解析」で解析できます)"
        )
    return node.snapshot.sfen, position_str, "\n\n".join(notes)
