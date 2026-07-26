"""自己対局の interface 層 (console → app の橋渡し)."""

import logging
from pathlib import Path
from typing import Any

from maou.app.usi.selfplay import SelfplayRunner

logger: logging.Logger = logging.getLogger(__name__)


def selfplay(
    *,
    model_path: Path | None = None,
    games: int = 1,
    parallel: int = 1,
    playouts: int | None = None,
    movetime_ms: int | None = None,
    max_moves: int = 512,
    sfen: str | None = None,
    opening_random_plies: int = 0,
    seed: int = 0,
    verbose: bool = True,
    threads: int = 1,
    batch_size: int = 8,
    node_capacity: int | None = None,
    draw_value_black: int = 500,
    draw_value_white: int = 500,
    resign_value: int = 0,
    resign_consecutive: int = 3,
    opening_script: str | None = None,
    root_dfpn: bool = True,
    root_dfpn_nodes: int = 2_000_000,
    root_dfpn_depth: int = 2047,
    leaf_mate: bool = True,
    leaf_mate_nodes: int = 50,
    leaf_mate_threads: int = 1,
    cuda: bool = False,
    tensorrt: bool = False,
    trt_engine_cache_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """in-process 自己対局を実行して対局結果のリストを返す．

    Args:
        model_path: ONNX モデルファイルパス (None = mock 評価器)．
        games: 対局数．
        parallel: 同時対局数．
        playouts: 1 手の playout 予算 (movetime_ms と排他)．
        movetime_ms: 1 手の思考時間ミリ秒 (playouts と排他)．
        max_moves: 最大手数 (到達で引き分け)．
        sfen: 基準局面 SFEN (None = 平手)．
        opening_random_plies: 序盤ランダム手数 (対局多様化)．
        seed: 乱数シード．
        verbose: 対局ごとの進捗を stderr へ出すか．
        threads: 1 探索あたりのスレッド数．
        batch_size: 評価バッチサイズ．
        node_capacity: ノードプール容量 (エージェント 1 個あたり)．
        draw_value_black: 先手番の引き分け価値 (千分率)．
        draw_value_white: 後手番の引き分け価値 (千分率)．
        resign_value: 投了する root 勝率 (千分率，0 = 投了しない)．
        resign_consecutive: 投了に必要な連続手数．
        opening_script: 強制序盤手順 (両側エージェントに適用)．
        root_dfpn: ルート並行 dfpn 詰み探索を有効にするか．
        root_dfpn_nodes: ルート dfpn のノード予算．
        root_dfpn_depth: ルート dfpn の探索深さ上限．
        leaf_mate: MCTS の葉の短手詰み探索を行うか．
        leaf_mate_nodes: leaf-mate 1 回あたりのノード予算．
        leaf_mate_threads: leaf-mate 専用スレッド数．
        cuda: CUDA Execution Provider を使うか．
        tensorrt: TensorRT Execution Provider を使うか．
        trt_engine_cache_dir: TensorRT エンジンキャッシュ保存先．

    Returns:
        対局 index 順の対局結果 dict のリスト．

    Raises:
        ValueError: playouts と movetime_ms を同時指定した場合．
    """
    if playouts is not None and movetime_ms is not None:
        raise ValueError(
            "playouts と movetime_ms は同時に指定できない"
        )
    option = SelfplayRunner.SelfplayOption(
        model_path=model_path,
        games=games,
        parallel=parallel,
        playouts=playouts,
        movetime_ms=movetime_ms,
        max_moves=max_moves,
        sfen=sfen,
        opening_random_plies=opening_random_plies,
        seed=seed,
        verbose=verbose,
        threads=threads,
        batch_size=batch_size,
        node_capacity=node_capacity,
        draw_value_black=draw_value_black,
        draw_value_white=draw_value_white,
        resign_value=resign_value,
        resign_consecutive=resign_consecutive,
        opening_script=opening_script,
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
    return SelfplayRunner().run(option)
