import logging
from pathlib import Path

from maou.app.search.run import SearchRunner

logger: logging.Logger = logging.getLogger(__name__)


def search(
    *,
    sfen: str,
    moves: str | None = None,
    model_path: Path | None = None,
    threads: int = 1,
    batch_size: int = 8,
    playouts: int | None = None,
    time_ms: int | None = None,
    num_moves: int = 5,
    root_dfpn: bool = False,
    leaf_mate: bool = False,
    leaf_mate_nodes: int = 50,
    cuda: bool = False,
    tensorrt: bool = False,
    trt_engine_cache_dir: Path | None = None,
) -> str:
    """1 局面を MCTS で探索して結果を文字列で返す．

    Args:
        sfen: 基準局面の SFEN 文字列．
        moves: sfen からの指し手列 (USI 形式，空白区切り)．途中局面は
            千日手判定の対局履歴として使われる．
        model_path: ONNX モデルファイルパス．None なら mock 評価器
            (API 検証/開発用)．
        threads: 探索スレッド数．
        batch_size: 評価バッチサイズ．
        playouts: playout 数上限．
        time_ms: 時間上限 (ミリ秒)．playouts と両方未指定なら 1000ms．
        num_moves: 表示する上位候補手数．
        root_dfpn: ルート並行 dfpn 詰み探索を有効にするか．
        leaf_mate: MCTS の各葉で短手詰み探索を行うか．
        leaf_mate_nodes: leaf-mate 1 回あたりのノード予算．
        cuda: CUDA Execution Provider を使うか．
        tensorrt: TensorRT Execution Provider を使うか．
        trt_engine_cache_dir: TensorRT エンジンキャッシュ保存先．

    Returns:
        探索結果のフォーマット済み文字列 (Bestmove / Eval / WinRate / PV /
        Candidates / Stats / 盤面)．
    """
    if playouts is None and time_ms is None:
        # 予算未指定は 1 秒に丸める (Rust 側デフォルトの 2^20 playouts は
        # CPU 実行では長すぎるため)
        time_ms = 1000
    move_list = tuple(moves.split()) if moves else ()
    option = SearchRunner.SearchOption(
        sfen=sfen,
        moves=move_list,
        model_path=model_path,
        threads=threads,
        batch_size=batch_size,
        max_playouts=playouts,
        time_ms=time_ms,
        num_moves=num_moves,
        root_dfpn=root_dfpn,
        leaf_mate=leaf_mate,
        leaf_mate_nodes=leaf_mate_nodes,
        cuda=cuda,
        tensorrt=tensorrt,
        trt_engine_cache_dir=trt_engine_cache_dir,
    )
    runner = SearchRunner()
    result = runner.run(option)
    return f"""

Bestmove: {result["Bestmove"]}
Eval: {result["Eval"]}
WinRate: {result["WinRate"]}
PV: {result["PV"]}
Candidates:
{result["Candidates"]}
Stats: {result["Stats"]}
{result["Board"]}"""
