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

from maou.app.analysis.analysis_session import (
    GameDocument,
    PositionSnapshot,
    load_game,
    validate_report,
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
    """

    document: GameDocument
    report: dict[str, Any] | None
    move_labels: list[str]


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
    snapshot = view.document.snapshots[ply]
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
    analysis = position_analysis(view, ply)
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
    analysis = position_analysis(view, ply)
    if analysis is None:
        return []
    snapshot = view.document.snapshots[ply]
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
