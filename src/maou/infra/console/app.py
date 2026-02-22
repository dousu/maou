import logging
from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from types import ModuleType
from typing import Any, Callable, MutableMapping, Sequence

import click

from maou.infra.app_logging import (
    app_logger,
    get_log_level_from_env,
)


@dataclass(frozen=True)
class PackageRequirement:
    """サブコマンドが必要とするパッケージと対応するextrasグループ．"""

    import_name: str
    extras: tuple[str, ...]


@dataclass(frozen=True)
class LazyCommandSpec:
    """遅延読み込みサブコマンドの定義．"""

    module_path: str
    attr_name: str
    missing_help: str | None = None
    required_packages: tuple[PackageRequirement, ...] = ()


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

        # import 前のパッケージチェック
        if lazy_definition.required_packages:
            missing = self._check_packages(
                lazy_definition.required_packages
            )
            if missing:
                return self._get_dependency_error_command(
                    cmd_name, missing
                )

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

    @staticmethod
    def _check_packages(
        requirements: tuple[PackageRequirement, ...],
    ) -> list[PackageRequirement]:
        """インストールされていないパッケージを返す．

        find_spec はメタデータ確認のみで副作用がないため，
        重いパッケージ(torch等)でも高速にチェックできる．
        """
        return [
            req
            for req in requirements
            if find_spec(req.import_name) is None
        ]

    @staticmethod
    def _build_dependency_message(
        cmd_name: str,
        missing: list[PackageRequirement],
    ) -> str:
        """不足パッケージのエラーメッセージを組み立てる．"""
        lines = [
            f"Command '{cmd_name}' requires the following packages:"
        ]
        for req in missing:
            lines.append(
                f"  - {req.import_name} (not installed)"
            )

        # extras の集合を収集(順序保持)
        extras_options: list[str] = []
        seen: set[str] = set()
        for req in missing:
            for extra in req.extras:
                if extra not in seen:
                    seen.add(extra)
                    extras_options.append(extra)

        lines.append("")
        lines.append("Install with one of:")
        for extra in extras_options:
            lines.append(f"  uv sync --extra {extra}")

        return "\n".join(lines)

    def _get_dependency_error_command(
        self,
        cmd_name: str,
        missing: list[PackageRequirement],
    ) -> click.Command:
        """不足パッケージの詳細情報を持つフォールバックコマンドを返す．

        --help 時にはヘルプテキストとして不足情報を表示し，
        実行時には ClickException でインストール案内を出す．
        """
        fallback = self._fallback_commands.get(cmd_name)
        if fallback is not None:
            return fallback

        dep_message = self._build_dependency_message(
            cmd_name, missing
        )

        help_text = (
            f"[requires additional packages]\n\n{dep_message}"
        )

        pkg_names = ", ".join(r.import_name for r in missing)
        short_help = f"[requires: {pkg_names}]"

        def _raise_missing_dependency(
            *args: object, **kwargs: object
        ) -> None:
            raise click.ClickException(dep_message)

        fallback_command = click.Command(
            name=cmd_name,
            callback=cast_command_callback(
                _raise_missing_dependency
            ),
            help=help_text,
            short_help=short_help,
        )
        self._fallback_commands[cmd_name] = fallback_command
        super().add_command(fallback_command)
        return fallback_command

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


_TRAINING_EXTRAS = ("cpu", "cuda", "mpu", "tpu")

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
            "Install with `uv sync --extra cpu` "
            "(or another training extra)."
        ),
        required_packages=(
            PackageRequirement("torch", _TRAINING_EXTRAS),
            PackageRequirement("torchinfo", _TRAINING_EXTRAS),
        ),
    ),
    "utility": LazyCommandSpec(
        "maou.infra.console.utility",
        "utility",
        missing_help=(
            "Command 'utility' requires training dependencies. "
            "Install with `uv sync --extra cpu` "
            "(or another training extra)."
        ),
        required_packages=(
            PackageRequirement("torch", _TRAINING_EXTRAS),
        ),
    ),
    "evaluate": LazyCommandSpec(
        "maou.infra.console.evaluate_board",
        "evaluate_board",
        missing_help=(
            "Command 'evaluate' requires inference dependencies. "
            "Install with `uv sync --extra onnx-gpu-infer` "
            "or `uv sync --extra tensorrt-infer`."
        ),
        required_packages=(
            PackageRequirement(
                "onnxruntime",
                (
                    "cpu-infer",
                    "onnx-gpu-infer",
                    "tensorrt-infer",
                ),
            ),
        ),
    ),
    "pretrain": LazyCommandSpec(
        "maou.infra.console.pretrain_cli",
        "pretrain",
        required_packages=(
            PackageRequirement("torch", _TRAINING_EXTRAS),
        ),
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
            PackageRequirement("playwright", ("visualize",)),
        ),
    ),
    "screenshot": LazyCommandSpec(
        "maou.infra.console.screenshot",
        "screenshot",
        missing_help=(
            "Command 'screenshot' requires visualization dependencies. "
            "Install with `uv sync --extra visualize`."
        ),
        required_packages=(
            PackageRequirement("playwright", ("visualize",)),
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
