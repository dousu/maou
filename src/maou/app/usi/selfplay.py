"""自己対局ユースケース (Rust maou_usi::selfplay の薄いラッパー)."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from maou._rust.maou_usi import (
    run_selfplay as _rust_run_selfplay,
)

logger: logging.Logger = logging.getLogger(__name__)


class SelfplayRunner:
    """in-process 自己対局 (Rust maou_usi::selfplay) を実行するユースケース．

    1 対局 = エージェント 2 個 (先後独立の探索木) を stdio/プロセスなしで
    直接駆動する．評価器はプロセス内 1 個を全対局で共有し，モデルロード/
    warmup は 1 回だけ行う．終局判定 (宣言/千日手/最大手数/投了) は USI
    対局と同一実装 (docs/design/usi-engine/index.md §9)．
    """

    @dataclass(kw_only=True, frozen=True)
    class SelfplayOption:
        """自己対局のオプション．

        Attributes:
            model_path: ONNX モデルのパス．None なら mock 評価器 (開発用)．
            games: 対局数．
            parallel: 同時対局数 (ワーカースレッド数)．
            playouts: 1 手あたりの playout 予算 (movetime_ms と排他．両方
                None なら 800)．
            movetime_ms: 1 手あたりの思考時間ミリ秒 (playouts と排他)．
            max_moves: 最大手数 (到達で引き分け．既定 512 = 電竜戦)．
            sfen: 基準局面 SFEN (None = 平手初期局面)．
            opening_random_plies: 序盤の driver 直指しランダム手数 (対局
                多様化用，0 = 無効)．
            seed: 乱数シード (対局 index と混合される)．
            verbose: 対局ごとの進捗を stderr へ出すか．
            threads: 1 探索あたりのスレッド数．
            batch_size: 評価バッチサイズ．
            node_capacity: ノードプール容量 (エージェント 1 個あたり．
                1 対局は 2 個持つ)．
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
        """

        model_path: Path | None = None
        games: int = 1
        parallel: int = 1
        playouts: int | None = None
        movetime_ms: int | None = None
        max_moves: int = 512
        sfen: str | None = None
        opening_random_plies: int = 0
        seed: int = 0
        verbose: bool = True
        threads: int = 1
        batch_size: int = 8
        node_capacity: int | None = None
        draw_value_black: int = 500
        draw_value_white: int = 500
        resign_value: int = 0
        resign_consecutive: int = 3
        opening_script: str | None = None
        root_dfpn: bool = True
        root_dfpn_nodes: int = 2_000_000
        root_dfpn_depth: int = 2047
        leaf_mate: bool = True
        leaf_mate_nodes: int = 50
        leaf_mate_threads: int = 1
        cuda: bool = False
        tensorrt: bool = False
        trt_engine_cache_dir: Path | None = None

    def run(
        self, option: SelfplayOption
    ) -> list[dict[str, Any]]:
        """自己対局を実行し対局結果のリストを返す (対局 index 順)．

        Args:
            option: 自己対局オプション．

        Returns:
            対局ごとの dict: ``{game_index, sfen, moves, winner
            ("black"/"white"/None), reason, plies, playouts, elapsed_ms}``．

        Raises:
            RuntimeError: 設定不正・モデルロード失敗・対局中の内部エラー
                (Rust 側から伝播する)．
        """
        logger.info(
            "Starting selfplay: %d game(s), parallel=%d",
            option.games,
            option.parallel,
        )
        records: list[dict[str, Any]] = _rust_run_selfplay(
            model_path=(
                str(option.model_path)
                if option.model_path is not None
                else None
            ),
            games=option.games,
            parallel=option.parallel,
            playouts=option.playouts,
            movetime_ms=option.movetime_ms,
            max_moves=option.max_moves,
            sfen=option.sfen,
            opening_random_plies=option.opening_random_plies,
            seed=option.seed,
            verbose=option.verbose,
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
            draw_value_black=option.draw_value_black,
            draw_value_white=option.draw_value_white,
            resign_value=option.resign_value,
            resign_consecutive=option.resign_consecutive,
            opening_script=option.opening_script,
            root_dfpn=option.root_dfpn,
            root_dfpn_nodes=option.root_dfpn_nodes,
            root_dfpn_depth=option.root_dfpn_depth,
            leaf_mate=option.leaf_mate,
            leaf_mate_nodes=option.leaf_mate_nodes,
            leaf_mate_threads=option.leaf_mate_threads,
        )
        logger.info(
            "Selfplay finished: %d game(s)", len(records)
        )
        return records
