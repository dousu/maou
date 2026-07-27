from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from maou.infra.app_logging import (
    app_logger,
    get_log_level_from_env,
)

if TYPE_CHECKING:
    from maou.infra.file_system.file_data_source import (
        FileDataSource as FileDataSource,
    )
    from maou.infra.file_system.file_system import (
        FileSystem as FileSystem,
    )

    # 以下は実行時には `__getattr__` (PEP 562) が初回アクセスで解決する．
    # **代入を書いてはいけない** — module global が存在すると `__getattr__`
    # が呼ばれず，probe 前の値 (None) がそのまま渡ってしまう
    HAS_BIGQUERY: bool
    HAS_GCS: bool
    HAS_AWS: bool
    BigQueryDataSource: type[Any] | None
    BigQueryFeatureStore: type[Any] | None
    GCS: type[Any] | None
    GCSDataSource: type[Any] | None
    GCSFeatureStore: type[Any] | None
    S3: type[Any] | None
    S3DataSource: type[Any] | None
    S3FeatureStore: type[Any] | None

__all__ = [
    "app_logger",
    "get_log_level_from_env",
    "FileDataSource",
    "FileSystem",
    "HAS_BIGQUERY",
    "HAS_GCS",
    "HAS_AWS",
    "BigQueryDataSource",
    "BigQueryFeatureStore",
    "GCS",
    "GCSDataSource",
    "GCSFeatureStore",
    "S3",
    "S3DataSource",
    "S3FeatureStore",
    "validate_cloud_provider_exclusivity",
    "handle_exception",
    "is_google_colab",
]


def is_google_colab() -> bool:
    """Google Colab環境で実行されているか検出する．

    Returns:
        bool: Google Colab環境の場合True
    """
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


# クラウド連携 (BigQuery / GCS / S3) の在否と実体は初回アクセス時に解決する
# (PEP 562 lazy import — 下の __getattr__)．BigQuery 経路は
# maou.interface.learn 経由で torch を，GCS/S3 は各 SDK を引き込むため，
# ここで eager に probe すると **クラウドを使わないコマンドまで** 数秒の
# import を払う (USI エンジンは GUI 登録で毎回起動するため実害が大きい)．
# 公開名 (HAS_BIGQUERY / BigQueryDataSource / …) と「依存が無ければ
# HAS_* = False・クラス名 = None」の意味論は従来どおり．
_CLOUD_PROBES: dict[
    str, tuple[str, tuple[tuple[str, str], ...]]
] = {
    "BIGQUERY": (
        "HAS_BIGQUERY",
        (
            (
                "maou.infra.bigquery.bq_data_source",
                "BigQueryDataSource",
            ),
            (
                "maou.infra.bigquery.bq_feature_store",
                "BigQueryFeatureStore",
            ),
        ),
    ),
    "GCS": (
        "HAS_GCS",
        (
            ("maou.infra.gcs.gcs", "GCS"),
            ("maou.infra.gcs.gcs_data_source", "GCSDataSource"),
            (
                "maou.infra.gcs.gcs_feature_store",
                "GCSFeatureStore",
            ),
        ),
    ),
    "AWS": (
        "HAS_AWS",
        (
            ("maou.infra.s3.s3", "S3"),
            ("maou.infra.s3.s3_data_source", "S3DataSource"),
            (
                "maou.infra.s3.s3_feature_store",
                "S3FeatureStore",
            ),
        ),
    ),
}

# 公開名 → どの probe に属するか (逆引き)
_CLOUD_OWNER: dict[str, str] = {
    flag: group for group, (flag, _) in _CLOUD_PROBES.items()
} | {
    attr: group
    for group, (_, members) in _CLOUD_PROBES.items()
    for _, attr in members
}


def _resolve_cloud_probe(group: str) -> None:
    """クラウド依存を 1 グループ分 import して結果を globals へ焼き付ける．

    依存が無ければ `HAS_* = False`・各クラス名 = None にする (従来の
    try/except ImportError と同じ意味論)．一度解決したら以後
    `__getattr__` を経由しない．

    Args:
        group: `_CLOUD_PROBES` のキー ("BIGQUERY" / "GCS" / "AWS")．
    """
    from importlib import import_module

    flag, members = _CLOUD_PROBES[group]
    try:
        resolved = {
            attr: cast(
                "type[Any]",
                getattr(import_module(module_path), attr, None),
            )
            for module_path, attr in members
        }
    except ImportError:
        app_logger.debug(
            "%s dependencies not available. Some features will be disabled.",
            group,
        )
        globals()[flag] = False
        for _, attr in members:
            globals()[attr] = None
        return
    globals().update(resolved)
    globals()[flag] = True


def validate_cloud_provider_exclusivity(
    provider_flags: list[bool],
    provider_names: list[str],
    operation: str,
) -> None:
    """Validate that only one cloud provider is selected for a given operation."""
    active_count = sum(provider_flags)
    if active_count > 1:
        error_msg = (
            f"Cannot use multiple cloud providers for {operation} simultaneously. "
            f"Please choose only one: {', '.join(provider_names)}."
        )
        app_logger.error(error_msg)
        raise ValueError(error_msg)


def handle_exception(func: Callable) -> Callable:
    """CLIコマンドの例外ハンドリングデコレータ．

    click.ClickException はClickのエラー表示機構に委ねるため再送出する．
    """
    import functools

    import click

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except click.ClickException:
            raise
        except Exception:
            app_logger.exception(
                "Error occurred", stack_info=True
            )
            raise SystemExit(1)

    return wrapper


# ---------------------------------------------------------------------------
# PEP 562 lazy imports: FileDataSource / FileSystem は torch に間接依存する
# ため，visualize 等の torch 不要コマンドでインポートコストを避ける．
# ---------------------------------------------------------------------------
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "FileDataSource": (
        "maou.infra.file_system.file_data_source",
        "FileDataSource",
    ),
    "FileSystem": (
        "maou.infra.file_system.file_system",
        "FileSystem",
    ),
}


def exit_skipping_teardown(*, tensorrt: bool) -> None:
    """TensorRT EP を使った実行を destructor を経由せず終了する．

    TensorRT EP の teardown は glibc のヒープを壊して abort する
    (``corrupted double-linked list``)．Colab L4 / ort 静的 1.22 + pip
    onnxruntime-gpu 1.22 で **3/3 決定的**に再現し，``--cuda`` のみでは
    0/3 なので **TensorRT EP 固有**である (詳細は
    docs/design/usi-engine/verification.md §8.5)．全出力が終わった
    プロセス終了時に起きるため数値・棋譜は有効だが，終了コードが SIGABRT に
    なるため GUI や対局サーバからはクラッシュと見なされ得る．

    原因は外部ライブラリ側で当面直せないので，出力を flush してから
    destructor を走らせずにプロセスを終える．**TensorRT を使っていない
    ときは何もしない** (通常の終了パスを保ち，将来の teardown バグを
    隠さないため)．

    Args:
        tensorrt: TensorRT EP を有効にして実行したか．
    """
    if not tensorrt:
        return
    sys.stdout.flush()
    sys.stderr.flush()
    logging.shutdown()
    os._exit(0)


def __getattr__(name: str) -> Any:
    group = _CLOUD_OWNER.get(name)
    if group is not None:
        _resolve_cloud_probe(group)
        return globals()[name]
    spec = _LAZY_IMPORTS.get(name)
    if spec is not None:
        module_path, attr_name = spec
        from importlib import import_module

        module = import_module(module_path)
        value = getattr(module, attr_name)
        # キャッシュ: 次回以降は __getattr__ を経由しない
        globals()[name] = value
        return value
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )
