"""USI エンジン起動の interface 層 (console → app の橋渡し)."""

import logging
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from maou.app.usi.run import UsiRunner

logger: logging.Logger = logging.getLogger(__name__)


def _engine_name() -> str:
    """`id name` に出すエンジン名 (Python パッケージ版数を含める)．

    Returns:
        "maou <version>" 形式のエンジン名．版数が取れなければ "maou"．
    """
    try:
        return f"maou {version('maou')}"
    except PackageNotFoundError:  # 開発ツリー直実行など
        return "maou"


def usi(
    *,
    model_path: Path | None = None,
    threads: int = 1,
    batch_size: int = 8,
    node_capacity: int | None = None,
    network_delay_ms: int = 1000,
    min_think_ms: int = 100,
    root_dfpn: bool = True,
    root_dfpn_nodes: int = 2_000_000,
    root_dfpn_depth: int = 2047,
    leaf_mate: bool = True,
    leaf_mate_nodes: int = 50,
    leaf_mate_threads: int = 1,
    cuda: bool = False,
    tensorrt: bool = False,
    trt_engine_cache_dir: Path | None = None,
) -> None:
    """USI エンジンを標準入出力で起動する (quit/EOF まで戻らない)．

    CLI フラグは初期値で，GUI からの USI `setoption` が上書きする．
    stdout は USI プロトコル専用 (logging は stderr へ出る)．

    Args:
        model_path: ONNX モデルファイルパス．None なら mock 評価器
            (開発検証用)．
        threads: 探索スレッド数．
        batch_size: 評価バッチサイズ．
        node_capacity: ノードプール容量 (None で既定)．
        network_delay_ms: 通信マージン (ミリ秒)．
        min_think_ms: 最低思考時間 (ミリ秒)．
        root_dfpn: ルート並行 dfpn 詰み探索を有効にするか．
        root_dfpn_nodes: ルート dfpn のノード予算．
        root_dfpn_depth: ルート dfpn の探索深さ上限 (最大 2047)．
        leaf_mate: MCTS の葉の短手詰み探索を行うか．
        leaf_mate_nodes: leaf-mate 1 回あたりのノード予算．
        leaf_mate_threads: leaf-mate 専用スレッド数．
        cuda: CUDA Execution Provider を使うか．
        tensorrt: TensorRT Execution Provider を使うか．
        trt_engine_cache_dir: TensorRT エンジンキャッシュ保存先．
    """
    option = UsiRunner.UsiOption(
        engine_name=_engine_name(),
        model_path=model_path,
        threads=threads,
        batch_size=batch_size,
        node_capacity=node_capacity,
        network_delay_ms=network_delay_ms,
        min_think_ms=min_think_ms,
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
    UsiRunner().run(option)
