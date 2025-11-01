from typing import Any, Callable

from maou.infra.app_logging import (
    app_logger,
    get_log_level_from_env,
)
from maou.infra.file_system.file_data_source import (
    FileDataSource,
)
from maou.infra.file_system.file_system import FileSystem

# NOTE:
#   Optional cloud provider dependencies (BigQuery, GCS, AWS) might not be
#   installed in every runtime.  To keep the public API of this module stable,
#   we initialise the exported symbols with ``None`` so that "from
#   maou.infra.console.common import S3DataSource" continues to succeed even
#   when the dependency is unavailable.  Callers must consult the corresponding
#   ``HAS_*`` flag before using the objects.
BigQueryDataSource: type[Any] | None = None
BigQueryFeatureStore: type[Any] | None = None
GCS: type[Any] | None = None
GCSDataSource: type[Any] | None = None
GCSFeatureStore: type[Any] | None = None
S3: type[Any] | None = None
S3DataSource: type[Any] | None = None
S3FeatureStore: type[Any] | None = None

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
]

# 必要なライブラリが利用可能かどうかをチェックする変数
HAS_BIGQUERY = False
HAS_GCS = False
HAS_AWS = False

# BigQuery関連のライブラリのインポートを試みる
try:
    from maou.infra.bigquery.bq_data_source import (
        BigQueryDataSource,
    )
    from maou.infra.bigquery.bq_feature_store import (
        BigQueryFeatureStore,
    )

    HAS_BIGQUERY = True
except ImportError:
    app_logger.debug(
        "BigQuery dependencies not available. Some features will be disabled."
    )

# GCS関連のライブラリのインポートを試みる
try:
    from maou.infra.gcs.gcs import GCS
    from maou.infra.gcs.gcs_data_source import GCSDataSource
    from maou.infra.gcs.gcs_feature_store import GCSFeatureStore

    HAS_GCS = True
except ImportError:
    app_logger.debug(
        "GCS dependencies not available. Some features will be disabled."
    )

# AWS S3関連のライブラリのインポートを試みる
try:
    from maou.infra.s3.s3 import S3
    from maou.infra.s3.s3_data_source import S3DataSource
    from maou.infra.s3.s3_feature_store import S3FeatureStore

    HAS_AWS = True
except ImportError:
    app_logger.debug(
        "AWS S3 dependencies not available. Some features will be disabled."
    )


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
    """Decorator to handle exceptions in CLI commands."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception:
            app_logger.exception(
                "Error occurred", stack_info=True
            )

    return wrapper
