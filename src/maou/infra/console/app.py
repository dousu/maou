import logging

import click

from maou.infra.app_logging import (
    app_logger,
    get_log_level_from_env,
)
from maou.infra.console.evaluate_board import evaluate_board
from maou.infra.console.hcpe_convert import hcpe_convert
from maou.infra.console.learn_model import learn_model
from maou.infra.console.pre_process import pre_process
from maou.infra.console.pretrain_cli import pretrain
from maou.infra.console.utility import utility


@click.group()
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


main.add_command(hcpe_convert)
main.add_command(pre_process)
main.add_command(learn_model)
main.add_command(utility)
main.add_command(evaluate_board)
main.add_command(pretrain)
