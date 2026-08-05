import logging

import click

from maou.infra.app_logging import (
    app_logger,
    get_log_level_from_env,
)
from maou.infra.console.lazy import (
    LazyCommandSpec,
    LazyGroup,
    PackageRequirement,
)

_TRAINING_EXTRAS = ("cpu", "cuda", "mpu")

LAZY_COMMANDS: dict[str, LazyCommandSpec] = {
    # CSA/KIF パーサ・HCPE 変換は Rust backend の独自実装 (追加依存なし)
    "hcpe-convert": LazyCommandSpec(
        "maou.infra.console.hcpe_convert", "hcpe_convert"
    ),
    "pre-process": LazyCommandSpec(
        "maou.infra.console.pre_process", "pre_process"
    ),
    "build-game-graph": LazyCommandSpec(
        "maou.infra.console.build_game_graph",
        "build_game_graph",
    ),
    "learn-model": LazyCommandSpec(
        "maou.infra.console.learn_model",
        "learn_model",
        missing_help=(
            "Command 'learn-model' requires training dependencies. "
            "Install with `uv sync --extra cpu` "
            "(or another training extra)."
        ),
        required_packages=(
            PackageRequirement("torch", _TRAINING_EXTRAS),
            PackageRequirement("torchinfo", _TRAINING_EXTRAS),
        ),
    ),
    # utility グループ自体は軽い (fetch-floodgate / split-kifu は torch 不要)．
    # torch を要するサブコマンドは utility_group 側で個別に遅延解決する．
    "utility": LazyCommandSpec(
        "maou.infra.console.utility_group",
        "utility",
    ),
    "search": LazyCommandSpec(
        "maou.infra.console.search_board",
        "search_board",
    ),
    # USI 対局エージェント (将棋所/ShogiGUI/ShogiHome 用; 追加依存なし)
    "usi": LazyCommandSpec(
        "maou.infra.console.usi",
        "usi",
    ),
    # in-process 自己対局 driver (評価器共有 + 並列; 追加依存なし)
    "selfplay": LazyCommandSpec(
        "maou.infra.console.selfplay",
        "selfplay",
    ),
    # CSA サーバ対局 (floodgate でのレーティング計測; 追加依存なし)
    "floodgate": LazyCommandSpec(
        "maou.infra.console.floodgate",
        "floodgate",
    ),
    # 棋譜 1 局の自動解析 (1 手ずつ 1 局面探索; 追加依存なし)
    "analyze-game": LazyCommandSpec(
        "maou.infra.console.analyze_game",
        "analyze_game",
    ),
    # 棋譜解析 GUI (Gradio．コマンド本体は遅延 import で追加依存なし)
    "analyze-gui": LazyCommandSpec(
        "maou.infra.console.analyze_gui",
        "analyze_gui",
        missing_help=(
            "Command 'analyze-gui' requires visualization "
            "dependencies. Install with `uv sync --extra visualize`."
        ),
        required_packages=(
            PackageRequirement("gradio", ("visualize",)),
        ),
    ),
    # 0 手読み評価 (推論は Rust maou_search; 追加依存なし)
    "evaluate": LazyCommandSpec(
        "maou.infra.console.evaluate_board",
        "evaluate_board",
    ),
    "visualize": LazyCommandSpec(
        "maou.infra.console.visualize",
        "visualize",
        missing_help=(
            "Command 'visualize' requires visualization dependencies. "
            "Install with `uv sync --extra visualize`."
        ),
        required_packages=(
            PackageRequirement("gradio", ("visualize",)),
            PackageRequirement("matplotlib", ("visualize",)),
        ),
    ),
}


@click.group(cls=LazyGroup, lazy_commands=LAZY_COMMANDS)
@click.option(
    "--debug-mode",
    "-d",
    is_flag=True,
    help="Enable debug logging.",
)
def main(debug_mode: bool) -> None:
    if debug_mode:
        app_logger.setLevel(logging.DEBUG)
    else:
        # 環境変数MAOU_LOG_LEVELからログレベルを取得
        # デバッグモードが指定されていない場合のみ環境変数を参照
        app_logger.setLevel(get_log_level_from_env())
