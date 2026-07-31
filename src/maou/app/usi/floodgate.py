"""floodgate (CSA サーバプロトコル) 対局ユースケース (Rust maou_usi の薄いラッパー)."""

import logging
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from maou._rust.maou_usi import run_csa as _rust_run_csa

logger: logging.Logger = logging.getLogger(__name__)

#: floodgate (wdoor) の接続先ホスト．
DEFAULT_HOST = "wdoor.c.u-tokyo.ac.jp"
#: floodgate の接続先ポート．
DEFAULT_PORT = 4081
#: floodgate のレーティング対局場 (持ち時間 300 秒 + 10 秒加算)．
DEFAULT_GAME_NAME = "floodgate-300-10F"
#: 自動生成する trip のバイト数 (16 進 2 倍長の文字列になる)．
TRIP_BYTES = 8


def generate_trip() -> str:
    """trip (パスワード) をランダム生成する．

    floodgate はログイン名を事前登録しない．同名ユーザの区別は
    「ログイン名 + trip の一致」で行われるため，同じ識別子で対局を続ける
    (レーティングを積む) には同じ trip を使い続ける必要がある．

    Returns:
        16 進の乱数文字列．
    """
    return secrets.token_hex(TRIP_BYTES)


def build_password(*, game_name: str, trip: str) -> str:
    """CSA の LOGIN パスワード欄を組み立てる．

    floodgate は CSA モードのパスワード欄にゲーム名を埋め込む規約
    (`floodgate-300-10F,<trip>`)．ゲーム名が空なら trip をそのまま使う
    (ゲーム名を要求しない一般の CSA サーバ向け)．

    Args:
        game_name: 対局場のゲーム名．
        trip: ユーザ固有の秘密文字列．

    Returns:
        LOGIN コマンドのパスワード欄．
    """
    name = game_name.strip()
    return f"{name},{trip}" if name else trip


class FloodgateRunner:
    """CSA サーバに接続して対局するユースケース．

    プロトコル (LOGIN/対局条件/指し手交換/終局)・時間管理・探索は全て Rust 側
    (`maou._rust.maou_usi.run_csa`) が GIL を解放して実行する．Python は設定の
    受け渡しだけを担う薄いラッパー (docs/design/usi-engine/index.md §5 と同じ分担)．

    floodgate は 1 局ごとにログアウト状態へ戻るため，連続対局は対局ごとの
    再接続として実装されている．評価器はプロセス内 1 個を全対局で共有するので，
    モデルロードと warmup は連続対局を通じて 1 回だけ行われる．
    """

    @dataclass(kw_only=True, frozen=True)
    class FloodgateOption:
        """CSA 対局の起動オプション．

        Attributes:
            login_name: ログイン名．floodgate のレーティングはこの名前ごとに
                集計される．事前登録は不要．
            password: LOGIN のパスワード欄 (floodgate では
                `<ゲーム名>,<trip>`)．[`build_password`] で組み立てる．
            host: 接続先ホスト．
            port: 接続先ポート．
            games: 対局数 (0 = 無制限)．
            keep_alive_sec: 待機中に空行を送る間隔 (秒)．CSA 仕様の下限 30 秒を
                下回る値は Rust 側で切り上げられる．
            connect_timeout_sec: 接続とログイン応答のタイムアウト (秒)．
            game_wait_sec: 対局条件を待つ上限 (秒．0 = 無制限)．floodgate は
                毎時 0 分/30 分に対局が組まれる．
            reconnect_wait_sec: 対局間の再接続待ち (秒)．
            verbose: 通信ログを stderr に出すか．
            model_path: ONNX モデルのパス．None なら mock 評価器 (疎通確認用)．
            threads: 探索スレッド数．
            batch_size: 評価バッチサイズ．
            node_capacity: ノードプール容量 (None で既定)．
            network_delay_ms: 通信マージン (ミリ秒)．消費時間はサーバが計測する
                ため，往復遅延と driver の遅れをこの分だけ予算から引く．
            min_think_ms: 最低思考時間 (ミリ秒)．
            time_curve: 手数カーブ (変換期重み付け) を有効にする (既定 True)．
                乗数は裁量枠 (残時間 ÷ horizon) にのみ掛かるので，floodgate の
                10 秒加算は終盤でもそのまま残る．
            time_curve_peak_ply: カーブの頂点手数．
            time_curve_half_width_ply: 頂点から底までの手数．
            time_curve_peak_permille: 頂点の乗数 (permille)．
            time_curve_opening_floor_permille: 序盤側の底の乗数 (permille)．
            time_curve_endgame_floor_permille: 終盤側の底の乗数 (permille)．
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
            defensive_mate: 受け方向の詰み探索 (被詰み検出) を行うか．
            defensive_mate_threads: root 敗着フィルタの並列度 (CPU が空く環境で上げる)．
            cuda: CUDA Execution Provider を使うか．
            tensorrt: TensorRT Execution Provider を使うか．
            trt_engine_cache_dir: TensorRT エンジンキャッシュ保存先．
        """

        login_name: str
        password: str
        host: str = DEFAULT_HOST
        port: int = DEFAULT_PORT
        games: int = 1
        keep_alive_sec: int = 60
        connect_timeout_sec: int = 30
        game_wait_sec: int = 2400
        reconnect_wait_sec: int = 5
        verbose: bool = True
        model_path: Path | None = None
        threads: int = 1
        batch_size: int = 8
        node_capacity: int | None = None
        network_delay_ms: int = 1000
        min_think_ms: int = 100
        time_curve: bool = True
        time_curve_peak_ply: int = 100
        time_curve_half_width_ply: int = 55
        time_curve_peak_permille: int = 2500
        time_curve_opening_floor_permille: int = 300
        time_curve_endgame_floor_permille: int = 1200
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
        defensive_mate: bool = True
        defensive_mate_threads: int = 1
        cuda: bool = False
        tensorrt: bool = False
        trt_engine_cache_dir: Path | None = None

    def run(self, option: FloodgateOption) -> dict[str, Any]:
        """指定局数の対局が終わるまで実行する．

        Args:
            option: 起動オプション．

        Returns:
            `{"games": [{"game_id", "opponent", "my_color", "moves",
            "result", "reason", "elapsed_ms"}, ...]}`．

        Raises:
            RuntimeError: ログイン拒否・モデルロード失敗など復帰できない
                エラー (Rust 側から伝播する)．
        """
        logger.info(
            "Connecting to %s:%s as %s (%s game(s))",
            option.host,
            option.port,
            option.login_name,
            "unlimited" if option.games == 0 else option.games,
        )
        result: dict[str, Any] = _rust_run_csa(
            host=option.host,
            port=option.port,
            login_name=option.login_name,
            password=option.password,
            games=option.games,
            keep_alive_sec=option.keep_alive_sec,
            connect_timeout_sec=option.connect_timeout_sec,
            game_wait_sec=option.game_wait_sec,
            reconnect_wait_sec=option.reconnect_wait_sec,
            verbose=option.verbose,
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
            time_curve=option.time_curve,
            time_curve_peak_ply=option.time_curve_peak_ply,
            time_curve_half_width_ply=option.time_curve_half_width_ply,
            time_curve_peak_permille=option.time_curve_peak_permille,
            time_curve_opening_floor_permille=(
                option.time_curve_opening_floor_permille
            ),
            time_curve_endgame_floor_permille=(
                option.time_curve_endgame_floor_permille
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
            defensive_mate=option.defensive_mate,
            defensive_mate_threads=option.defensive_mate_threads,
        )
        logger.info(
            "Finished %s game(s)", len(result.get("games", []))
        )
        return result
