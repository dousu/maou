"""棋譜解析 (analyze-game) の interface アダプタ．

CLI 入力の検証・予算配分戦略の構築・結果の整形 (JSON / 人間向けサマリ)
を担う．解析本体は :class:`maou.app.analysis.game_analyzer.GameAnalyzer`．
"""

import json
import logging
from pathlib import Path
from typing import Any

from maou.app.analysis.game_analyzer import (
    BudgetAllocator,
    EqualDivisionAllocator,
    FixedPlayoutsAllocator,
    FixedTimeAllocator,
    GameAnalyzer,
)

logger: logging.Logger = logging.getLogger(__name__)

_SUFFIX_TO_FORMAT: dict[str, str] = {
    ".csa": "csa",
    ".kif": "kif",
    ".kifu": "kif",
}


def resolve_input_format(
    input_path: Path, input_format: str | None
) -> str:
    """棋譜形式を確定する (明示指定を優先し，無ければ拡張子から判定)．

    Args:
        input_path: 棋譜ファイルのパス．
        input_format: 明示指定された形式 ("csa" / "kif")．None なら自動判定．

    Returns:
        "csa" または "kif"．

    Raises:
        ValueError: 不正な明示指定，または拡張子から判定できない場合．
    """
    if input_format is not None:
        if input_format not in ("csa", "kif"):
            raise ValueError(
                f"未対応の棋譜形式です: {input_format} (csa / kif のみ)"
            )
        return input_format
    fmt = _SUFFIX_TO_FORMAT.get(input_path.suffix.lower())
    if fmt is None:
        raise ValueError(
            f"拡張子 '{input_path.suffix}' から棋譜形式を判定できません．"
            "--input-format csa|kif を明示してください"
        )
    return fmt


def build_allocator(
    *,
    time_ms: int | None,
    total_time_ms: int | None,
    playouts: int | None,
) -> BudgetAllocator:
    """予算オプション (相互排他) から配分戦略を構築する．

    全て未指定の場合は 1 局面 1000ms (``maou search`` の既定と同じ)．

    Args:
        time_ms: 1 局面あたりの時間 (ミリ秒)．
        total_time_ms: 対局全体の時間 (ミリ秒，局面数で等分)．
        playouts: 1 局面あたりの playout 数．

    Returns:
        構築した配分戦略．

    Raises:
        ValueError: 2 つ以上の同時指定，または正でない値．
    """
    specified = [
        v
        for v in (time_ms, total_time_ms, playouts)
        if v is not None
    ]
    if len(specified) > 1:
        raise ValueError(
            "--time-ms / --total-time-ms / --playouts は同時に 1 つまで"
            "しか指定できません"
        )
    if specified and specified[0] <= 0:
        raise ValueError(
            f"探索予算は正の値が必要です: {specified[0]}"
        )
    if time_ms is not None:
        return FixedTimeAllocator(time_ms=time_ms)
    if total_time_ms is not None:
        return EqualDivisionAllocator(
            total_time_ms=total_time_ms
        )
    if playouts is not None:
        return FixedPlayoutsAllocator(playouts=playouts)
    return FixedTimeAllocator(time_ms=1000)


def analyze_game(
    *,
    input_path: Path,
    input_format: str | None = None,
    model_path: Path | None = None,
    time_ms: int | None = None,
    total_time_ms: int | None = None,
    playouts: int | None = None,
    num_candidates: int = 5,
    threads: int = 1,
    batch_size: int = 8,
    root_dfpn: bool = True,
    root_dfpn_nodes: int = 2_000_000,
    root_dfpn_depth: int = 2047,
    leaf_mate: bool = True,
    leaf_mate_nodes: int = 50,
    leaf_mate_threads: int = 1,
    cuda: bool = False,
    tensorrt: bool = False,
    trt_engine_cache_dir: Path | None = None,
) -> tuple[str, str]:
    """棋譜 1 局を解析して (JSON 文字列, サマリ文字列) を返す．

    Args:
        input_path: 棋譜ファイル (CSA / KIF) のパス．
        input_format: 棋譜形式．None なら拡張子から自動判定．
        model_path: ONNX モデルのパス．None なら mock 評価器 (開発用)．
        time_ms: 1 局面あたりの時間予算 (ミリ秒，排他)．
        total_time_ms: 対局全体の時間予算 (ミリ秒，等分，排他)．
        playouts: 1 局面あたりの playout 予算 (排他)．
        num_candidates: JSON に記録する候補手数．
        threads: 探索スレッド数．
        batch_size: 評価バッチサイズ．
        root_dfpn: ルート並行 dfpn 詰み探索を有効にするか．
        root_dfpn_nodes: ルート dfpn のノード予算．
        root_dfpn_depth: ルート dfpn の探索深さ上限 (最大 2047)．
        leaf_mate: MCTS の葉の短手詰み探索 (専用スレッド) を行うか．
        leaf_mate_nodes: leaf-mate 1 回あたりのノード予算．
        leaf_mate_threads: leaf-mate 専用スレッド数．
        cuda: CUDA Execution Provider を使うか．
        tensorrt: TensorRT Execution Provider を使うか．
        trt_engine_cache_dir: TensorRT エンジンキャッシュ保存先．

    Returns:
        ``(JSON 文字列, 人間向けサマリ文字列)`` のタプル．JSON スキーマは
        docs/design/game-analysis/index.md §7．
    """
    fmt = resolve_input_format(input_path, input_format)
    allocator = build_allocator(
        time_ms=time_ms,
        total_time_ms=total_time_ms,
        playouts=playouts,
    )
    option = GameAnalyzer.AnalyzeOption(
        input_path=input_path,
        input_format=fmt,
        allocator=allocator,
        model_path=model_path,
        threads=threads,
        batch_size=batch_size,
        num_candidates=num_candidates,
        root_dfpn=root_dfpn,
        root_dfpn_nodes=root_dfpn_nodes,
        root_dfpn_depth=root_dfpn_depth,
        leaf_mate=leaf_mate,
        leaf_mate_nodes=leaf_mate_nodes,
        leaf_mate_threads=leaf_mate_threads,
        cuda=cuda,
        tensorrt=tensorrt,
        trt_engine_cache_dir=trt_engine_cache_dir,
    )
    result = GameAnalyzer().analyze(option)
    json_str = json.dumps(result, ensure_ascii=False, indent=2)
    return json_str, _format_summary(result)


def _format_summary(result: dict[str, Any]) -> str:
    """解析結果 dict から人間向けサマリ文字列を作る．"""
    inp = result["input"]
    summ = result["summary"]
    names = inp["names"]
    black = (names[0] if len(names) > 0 else None) or "?"
    white = (names[1] if len(names) > 1 else None) or "?"
    win_label = {0: "draw", 1: "black win", 2: "white win"}.get(
        inp["win"], "unknown"
    )
    endgame = f" ({inp['endgame']})" if inp["endgame"] else ""
    model = result["engine"]["model_path"] or "mock"

    def pct(v: float | None) -> str:
        return f"{v * 100:.1f}%" if v is not None else "n/a"

    def num(v: float | None) -> str:
        return f"{v:.4f}" if v is not None else "n/a"

    mr = summ["match_rate"]
    mwl = summ["mean_winrate_loss"]
    lines = [
        f"Game: {black} (black) vs {white} (white)",
        f"Result: {win_label}{endgame}",
        f"Moves: {inp['n_moves']} | model: {model}",
        f"Match rate: black {pct(mr['black'])}, white {pct(mr['white'])}",
        (
            f"Mean winrate loss: black {num(mwl['black'])}, "
            f"white {num(mwl['white'])}"
        ),
    ]
    if summ["worst_moves"]:
        lines.append("Worst moves:")
        for w in summ["worst_moves"]:
            lines.append(
                f"  ply {w['ply']} ({w['side']}): played {w['played']}, "
                f"best {w['best']}, winrate_loss {w['winrate_loss']:.4f}"
            )
    if summ["mates_found"]:
        mates = ", ".join(
            f"ply {m['ply']} ({m['side']})"
            for m in summ["mates_found"]
        )
        lines.append(f"Mates found: {mates}")
    lines.append(
        f"Total: elapsed {summ['total_elapsed_ms'] / 1000:.1f}s, "
        f"playouts {summ['total_playouts']}"
    )
    return "\n".join(lines)
