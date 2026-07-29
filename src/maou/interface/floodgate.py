"""floodgate 対局の interface 層 (console → app の橋渡し)."""

import logging
from pathlib import Path
from typing import Any

from maou.app.usi.floodgate import (
    DEFAULT_GAME_NAME,
    DEFAULT_HOST,
    DEFAULT_PORT,
    FloodgateRunner,
    build_password,
)

logger: logging.Logger = logging.getLogger(__name__)


def floodgate(
    *,
    login_name: str,
    trip: str,
    game_name: str = DEFAULT_GAME_NAME,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    games: int = 1,
    keep_alive_sec: int = 60,
    connect_timeout_sec: int = 30,
    game_wait_sec: int = 2400,
    reconnect_wait_sec: int = 5,
    verbose: bool = True,
    model_path: Path | None = None,
    threads: int = 1,
    batch_size: int = 8,
    node_capacity: int | None = None,
    network_delay_ms: int = 1000,
    min_think_ms: int = 100,
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
) -> dict[str, Any]:
    """CSA サーバ (floodgate) に接続して対局する．

    指定局数が終わるまで戻らない．floodgate は 1 局ごとにログアウト状態へ
    戻るため，連続対局は対局ごとの再接続として行われる．

    Args:
        login_name: ログイン名 (レーティングの集計単位．事前登録不要)．
        trip: 同名ユーザを区別する秘密文字列．同じ識別子で対局を続けるには
            同じ値を使い続ける．
        game_name: 対局場のゲーム名 (空文字ならパスワード欄に埋め込まない)．
        host: 接続先ホスト．
        port: 接続先ポート．
        games: 対局数 (0 = 無制限)．
        keep_alive_sec: 待機中の keep-alive 間隔 (秒)．
        connect_timeout_sec: 接続とログイン応答のタイムアウト (秒)．
        game_wait_sec: 対局条件を待つ上限 (秒．0 = 無制限)．
        reconnect_wait_sec: 対局間の再接続待ち (秒)．
        verbose: 通信ログを stderr に出すか．
        model_path: ONNX モデルファイルパス．None なら mock 評価器．
        threads: 探索スレッド数．
        batch_size: 評価バッチサイズ．
        node_capacity: ノードプール容量 (None で既定)．
        network_delay_ms: 通信マージン (ミリ秒)．
        min_think_ms: 最低思考時間 (ミリ秒)．
        draw_value_black: 先手番の引き分け価値 (千分率)．
        draw_value_white: 後手番の引き分け価値 (千分率)．
        resign_value: 投了する root 勝率 (千分率，0 = 投了しない)．
        resign_consecutive: 投了に必要な連続手数．
        opening_script: 強制序盤手順 (USI 指し手の空白区切り)．
        root_dfpn: ルート並行 dfpn 詰み探索を行うか．
        root_dfpn_nodes: ルート dfpn のノード予算．
        root_dfpn_depth: ルート dfpn の探索深さ上限．
        leaf_mate: MCTS の葉の短手詰み探索を行うか．
        leaf_mate_nodes: leaf-mate 1 回あたりのノード予算．
        leaf_mate_threads: leaf-mate 専用スレッド数．
        cuda: CUDA Execution Provider を使うか．
        tensorrt: TensorRT Execution Provider を使うか．
        trt_engine_cache_dir: TensorRT エンジンキャッシュ保存先．

    Returns:
        対局結果の辞書 (`{"games": [...]}`)．
    """
    option = FloodgateRunner.FloodgateOption(
        login_name=login_name,
        password=build_password(game_name=game_name, trip=trip),
        host=host,
        port=port,
        games=games,
        keep_alive_sec=keep_alive_sec,
        connect_timeout_sec=connect_timeout_sec,
        game_wait_sec=game_wait_sec,
        reconnect_wait_sec=reconnect_wait_sec,
        verbose=verbose,
        model_path=model_path,
        threads=threads,
        batch_size=batch_size,
        node_capacity=node_capacity,
        network_delay_ms=network_delay_ms,
        min_think_ms=min_think_ms,
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
    return FloodgateRunner().run(option)
