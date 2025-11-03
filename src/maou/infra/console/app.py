import logging
from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import Any, Callable, MutableMapping, Sequence

import click

from maou.infra.app_logging import (
    app_logger,
    get_log_level_from_env,
)


@dataclass(frozen=True)
class LazyCommandSpec:
    module_path: str
    attr_name: str
    missing_help: str | None = None


class LazyGroup(click.Group):
    """Click group that loads subcommands on demand."""

    def __init__(
        self,
        name: str | None = None,
        commands: MutableMapping[str, click.Command]
        | Sequence[click.Command]
        | None = None,
        *,
        lazy_commands: dict[str, LazyCommandSpec] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, commands=commands, **kwargs)
        self._lazy_commands: dict[str, LazyCommandSpec] = (
            lazy_commands or {}
        )
        self._fallback_commands: dict[str, click.Command] = {}

    def list_commands(self, ctx: click.Context) -> list[str]:
        commands = set(super().list_commands(ctx))
        commands.update(self._lazy_commands.keys())
        return sorted(commands)

    def get_command(
        self, ctx: click.Context, cmd_name: str
    ) -> click.Command | None:
        command = super().get_command(ctx, cmd_name)
        if command is not None:
            return command

        lazy_definition = self._lazy_commands.get(cmd_name)
        if lazy_definition is None:
            return None

        module_path = lazy_definition.module_path
        attr_name = lazy_definition.attr_name

        try:
            module: ModuleType = import_module(module_path)
        except ModuleNotFoundError as exc:
            if exc.name is not None and exc.name.startswith(
                module_path
            ):
                raise
            return self._get_fallback_command(
                cmd_name, lazy_definition, exc
            )

        lazy_command = getattr(module, attr_name)
        super().add_command(lazy_command)
        return lazy_command

    def _get_fallback_command(
        self,
        cmd_name: str,
        spec: LazyCommandSpec,
        error: ModuleNotFoundError,
    ) -> click.Command:
        fallback = self._fallback_commands.get(cmd_name)
        if fallback is not None:
            return fallback

        missing_dep = error.name or "optional dependency"
        help_message = spec.missing_help or (
            f"Command '{cmd_name}' requires the optional dependency "
            f"'{missing_dep}'. Install the appropriate extras."
        )

        def _raise_missing_dependency(
            *args: object, **kwargs: object
        ) -> None:
            raise click.ClickException(help_message)

        fallback_command = click.Command(
            name=cmd_name,
            callback=cast_command_callback(
                _raise_missing_dependency
            ),
            help=help_message,
            short_help=help_message,
        )
        self._fallback_commands[cmd_name] = fallback_command
        super().add_command(fallback_command)
        return fallback_command


def cast_command_callback(
    func: Callable[..., None],
) -> Callable[..., None]:
    """Ensure mypy has precise callback types."""
    return func


LAZY_COMMANDS: dict[str, LazyCommandSpec] = {
    "hcpe-convert": LazyCommandSpec(
        "maou.infra.console.hcpe_convert", "hcpe_convert"
    ),
    "pre-process": LazyCommandSpec(
        "maou.infra.console.pre_process", "pre_process"
    ),
    "learn-model": LazyCommandSpec(
        "maou.infra.console.learn_model",
        "learn_model",
        missing_help=(
            "Command 'learn-model' requires training dependencies. "
            "Install with `poetry install -E cpu` (or another training extra)."
        ),
    ),
    "utility": LazyCommandSpec(
        "maou.infra.console.utility",
        "utility",
        missing_help=(
            "Command 'utility' requires training dependencies. "
            "Install with `poetry install -E cpu` (or another training extra)."
        ),
    ),
    "evaluate": LazyCommandSpec(
        "maou.infra.console.evaluate_board",
        "evaluate_board",
        missing_help=(
            "Command 'evaluate' requires inference dependencies. "
            "Install with `poetry install -E onnx-gpu-infer` "
            "or `poetry install -E tensorrt-infer`."
        ),
    ),
    "pretrain": LazyCommandSpec(
        "maou.infra.console.pretrain_cli", "pretrain"
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
