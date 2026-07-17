"""USI エンジン起動ユースケース (Rust maou_usi の薄いラッパー)."""

import logging
from dataclasses import dataclass
from pathlib import Path

from maou._rust.maou_usi import run_usi as _rust_run_usi

logger: logging.Logger = logging.getLogger(__name__)


class UsiRunner:
    """USI 対局エージェント (Rust maou_usi) を起動するユースケース．

    プロトコルループ (reader スレッド + dispatcher)・対局エージェント・
    時間管理は全て Rust 側 (`maou._rust.maou_usi.run_usi`) が GIL を解放して
    実行する．Python は設定の受け渡しだけを担う薄いラッパー
    (docs/design/usi-engine/index.md §5)．stdout は USI プロトコル専用の
    ため，本クラスは stdout に一切書かない (logging は stderr)．
    """

    @dataclass(kw_only=True, frozen=True)
    class UsiOption:
        """USI エンジンの起動オプション (CLI 初期値．`setoption` が上書き)．

        Attributes:
            engine_name: `id name` に出す名前 (None なら Rust 側既定)．
            model_path: ONNX モデルのパス．None なら mock 評価器
                (開発検証用 — isready 時に `info string` で明示される)．
            threads: 探索スレッド数．
            batch_size: 評価バッチサイズ．
            node_capacity: ノードプール容量 (None で既定 2^20)．
            network_delay_ms: 通信マージン (ミリ秒)．探索予算はこの分
                短くなる．
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

        engine_name: str | None = None
        model_path: Path | None = None
        threads: int = 1
        batch_size: int = 8
        node_capacity: int | None = None
        network_delay_ms: int = 1000
        min_think_ms: int = 100
        root_dfpn: bool = True
        root_dfpn_nodes: int = 2_000_000
        root_dfpn_depth: int = 2047
        leaf_mate: bool = True
        leaf_mate_nodes: int = 50
        leaf_mate_threads: int = 1
        cuda: bool = False
        tensorrt: bool = False
        trt_engine_cache_dir: Path | None = None

    def run(self, option: UsiOption) -> None:
        """USI ループを実行する (GUI からの `quit` / EOF まで戻らない)．

        Args:
            option: 起動オプション．

        Raises:
            RuntimeError: モデルロード失敗などの致命的エラー (Rust 側から
                伝播する)．
        """
        logger.info(
            "Starting USI engine loop (stdout is protocol-only)"
        )
        _rust_run_usi(
            engine_name=option.engine_name,
            model_path=(
                str(option.model_path)
                if option.model_path is not None
                else None
            ),
            threads=option.threads,
            batch_size=option.batch_size,
            node_capacity=option.node_capacity,
            use_cuda=option.cuda,
            use_tensorrt=option.tensorrt,
            trt_engine_cache_dir=(
                str(option.trt_engine_cache_dir)
                if option.trt_engine_cache_dir is not None
                else None
            ),
            network_delay_ms=option.network_delay_ms,
            min_think_ms=option.min_think_ms,
            root_dfpn=option.root_dfpn,
            root_dfpn_nodes=option.root_dfpn_nodes,
            root_dfpn_depth=option.root_dfpn_depth,
            leaf_mate=option.leaf_mate,
            leaf_mate_nodes=option.leaf_mate_nodes,
            leaf_mate_threads=option.leaf_mate_threads,
        )
        logger.info("USI engine loop finished")
