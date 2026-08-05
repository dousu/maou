"""``maou utility`` グループの定義 (インフラ層)．

このモジュールは**軽い**ことが要件である．`maou utility fetch-floodgate` や
`maou utility split-kifu` は torch を必要としないため，グループを解決するだけで
学習依存を引きずってはならない．

したがって重いサブコマンド (benchmark / stage データ生成) は
[`maou.infra.console.lazy.LazyGroup`] による遅延解決とし，
軽いサブコマンドだけを直接登録する．
"""

import click

from maou.infra.console.fetch_floodgate import fetch_floodgate
from maou.infra.console.lazy import (
    LazyCommandSpec,
    LazyGroup,
    PackageRequirement,
)
from maou.infra.console.screenshot import screenshot
from maou.infra.console.split_kifu import split_kifu

_TRAINING_EXTRAS = ("cpu", "cuda", "mpu")

_TORCH_REQUIRED = (
    PackageRequirement("torch", _TRAINING_EXTRAS),
)

_MISSING_TRAINING_DEPS = (
    "This utility subcommand requires training dependencies. "
    "Install with `uv sync --extra cpu` (or another training extra)."
)

# torch を要するサブコマンドは呼ばれるまで import しない
_LAZY_SUBCOMMANDS: dict[str, LazyCommandSpec] = {
    "benchmark-dataloader": LazyCommandSpec(
        "maou.infra.console.utility",
        "benchmark_dataloader",
        missing_help=_MISSING_TRAINING_DEPS,
        required_packages=_TORCH_REQUIRED,
    ),
    "benchmark-training": LazyCommandSpec(
        "maou.infra.console.utility",
        "benchmark_training",
        missing_help=_MISSING_TRAINING_DEPS,
        required_packages=_TORCH_REQUIRED,
    ),
    "generate-stage1-data": LazyCommandSpec(
        "maou.infra.console.utility",
        "generate_stage1_data",
        missing_help=_MISSING_TRAINING_DEPS,
        required_packages=_TORCH_REQUIRED,
    ),
    "generate-stage2-data": LazyCommandSpec(
        "maou.infra.console.utility",
        "generate_stage2_data",
        missing_help=_MISSING_TRAINING_DEPS,
        required_packages=_TORCH_REQUIRED,
    ),
}


@click.group(cls=LazyGroup, lazy_commands=_LAZY_SUBCOMMANDS)
def utility() -> None:
    """Utility commands for data preparation and ML experiments."""


# 追加依存を持たないサブコマンド
utility.add_command(fetch_floodgate)
utility.add_command(split_kifu)
utility.add_command(screenshot)
