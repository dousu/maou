"""自己対局ユースケース (Rust maou_usi::selfplay の薄いラッパー)."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from maou._rust.maou_usi import (
    run_selfplay as _rust_run_selfplay,
)

logger: logging.Logger = logging.getLogger(__name__)

AB_MODES: tuple[str, ...] = (
    "subtree",
    "maxmoves",
    "budget",
    "horizon",
    "timecurve",
    "spin",
    "proven",
    "batch",
)
"""A/B 対戦で比較できるレバー (Rust maou_usi::ab::AbMode と対応)."""


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
            spin_budget_relief: 空回りを playout 予算から外すか．
            skip_proven_children: 確定済みの子を PUCT 候補から外すか．
            cuda: CUDA Execution Provider を使うか．
            tensorrt: TensorRT Execution Provider を使うか．
            trt_engine_cache_dir: TensorRT エンジンキャッシュ保存先．
            pad_buckets: TensorRT の padding を batch_size 固定でなく 2 冪
                バケットへ切り上げるか (計測用トグル，既定 False)．
            min_think_ms: 最低思考時間ミリ秒 (None = エンジン既定 100)．
                持ち時間モードで 1 手の下限を変えるときだけ指定する．
            ab_mode: A/B 対戦で比較するレバー ([`AB_MODES`]．None = 純粋
                自己対局)．指定するとレバー off 側の相手 B が作られ，
                レバー以外の設定は A と同一になる．
            playouts_b: ``ab_mode="budget"`` の B 側予算 (None = A の 1/8)．
            batch_size_b: ``ab_mode="batch"`` の B 側評価バッチサイズ
                (None = A の 4 倍)．
            horizon_moves: ``ab_mode="horizon"`` の A 側想定残り手数
                (None = エンジン既定)．
            horizon_moves_b: 同 B 側想定残り手数 (None = 25)．
            time_curve_peak_ply: 手数カーブの頂点手数 (None = 55)．
            time_curve_half_width_ply: 頂点から底までの手数 (None = 35)．
            time_curve_peak_permille: 頂点の乗数 permille (None = 1800)．
            time_curve_opening_floor_permille: 序盤側の底の乗数
                permille (None = 700)．
            time_curve_endgame_floor_permille: 終盤側の底の乗数
                permille (None = 1000 = 現行と同配分)．
                ``ab_mode="timecurve"`` では A/B の両者に同じ値が入り，
                差は on/off だけになる．
            record_kifu: 各対局の CSA 棋譜を ``records[i]["csa"]`` に
                含めるか (既定 False)．
            alternate_colors: 対局ごとに先後を入れ替え，ペア (2n, 2n+1) で
                同じ開局系列を共有するか (None = ab_mode 指定時 True)．
            clock_ms: 持ち時間モードの初期持ち時間ミリ秒 (0/None = 無効)．
                playouts/movetime_ms と排他で parallel=1 限定．
            byoyomi_ms: 秒読みミリ秒 (持ち時間モードのみ)．
            inc_ms: 1 手加算ミリ秒 (持ち時間モードのみ)．
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
        spin_budget_relief: bool = False
        skip_proven_children: bool = False
        cuda: bool = False
        tensorrt: bool = False
        trt_engine_cache_dir: Path | None = None
        pad_buckets: bool = False
        min_think_ms: int | None = None
        ab_mode: str | None = None
        playouts_b: int | None = None
        batch_size_b: int | None = None
        horizon_moves: int | None = None
        horizon_moves_b: int | None = None
        time_curve_peak_ply: int | None = None
        time_curve_half_width_ply: int | None = None
        time_curve_peak_permille: int | None = None
        time_curve_opening_floor_permille: int | None = None
        time_curve_endgame_floor_permille: int | None = None
        record_kifu: bool = False
        alternate_colors: bool | None = None
        clock_ms: int | None = None
        byoyomi_ms: int | None = None
        inc_ms: int | None = None

    @dataclass(kw_only=True, frozen=True)
    class SelfplayResult:
        """自己対局の実行結果．

        Attributes:
            records: 対局ごとの dict (対局 index 順)．
            summary: 集計 (Rust maou_usi::ab::summarize)．``ab`` キーは
                A/B 対戦時のみ dict で，それ以外は None．
            wall_ms: 実行全体の壁時計時間ミリ秒 (並列時のスループット算出
                に使う — 対局ごとの elapsed_ms の総和とは一致しない)．
        """

        records: list[dict[str, Any]]
        summary: dict[str, Any]
        wall_ms: int

    def run(self, option: SelfplayOption) -> SelfplayResult:
        """自己対局を実行し対局結果と集計を返す．

        Args:
            option: 自己対局オプション．

        Returns:
            [`SelfplayResult`]．``records`` の 1 件は ``{game_index, sfen,
            moves, winner ("black"/"white"/None), reason, black_player
            ("a"/"b" — A/B 対戦時の色交代記録．純粋自己対局では常に "a"),
            plies, playouts, reused_moves, carried_visits, elapsed_ms,
            remaining_ms}``．

        Raises:
            RuntimeError: 設定不正・モデルロード失敗・対局中の内部エラー
                (Rust 側から伝播する)．
            ValueError: 未知の ``ab_mode``，または持ち時間なしの
                ``ab_mode="horizon"`` (Rust 側から伝播する)．
        """
        logger.info(
            "Starting selfplay: %d game(s), parallel=%d",
            option.games,
            option.parallel,
        )
        started = time.monotonic()
        result: dict[str, Any] = _rust_run_selfplay(
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
            pad_buckets=option.pad_buckets,
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
            spin_budget_relief=option.spin_budget_relief,
            skip_proven_children=option.skip_proven_children,
            min_think_ms=option.min_think_ms,
            ab_mode=option.ab_mode,
            playouts_b=option.playouts_b,
            batch_size_b=option.batch_size_b,
            horizon_moves=option.horizon_moves,
            horizon_moves_b=option.horizon_moves_b,
            time_curve_peak_ply=option.time_curve_peak_ply,
            time_curve_half_width_ply=option.time_curve_half_width_ply,
            time_curve_peak_permille=option.time_curve_peak_permille,
            time_curve_opening_floor_permille=(
                option.time_curve_opening_floor_permille
            ),
            time_curve_endgame_floor_permille=(
                option.time_curve_endgame_floor_permille
            ),
            record_kifu=option.record_kifu,
            alternate_colors=option.alternate_colors,
            clock_ms=option.clock_ms,
            byoyomi_ms=option.byoyomi_ms,
            inc_ms=option.inc_ms,
        )
        wall_ms = int((time.monotonic() - started) * 1000)
        records: list[dict[str, Any]] = result["records"]
        logger.info(
            "Selfplay finished: %d game(s)", len(records)
        )
        return SelfplayRunner.SelfplayResult(
            records=records,
            summary=result["summary"],
            wall_ms=wall_ms,
        )
