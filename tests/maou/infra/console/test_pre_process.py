"""Tests for the ``pre-process`` CLI input-selection diagnostics."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from maou.infra.console.pre_process import (
    _NO_INPUT_SOURCE_MSG,
    describe_missing_input_options,
    pre_process,
)


def _describe(**overrides: object) -> str:
    """Call the helper with every option absent unless overridden."""
    kwargs: dict[str, object] = {
        "input_dataset_id": None,
        "input_table_name": None,
        "input_gcs": None,
        "input_s3": None,
        "input_bucket_name": None,
        "input_prefix": None,
        "input_data_name": None,
        "input_local_cache_dir": None,
    }
    kwargs.update(overrides)
    return describe_missing_input_options(**kwargs)  # type: ignore[arg-type]


class TestDescribeMissingInputOptions:
    """The message must name the option that is actually missing."""

    def test_no_input_at_all_keeps_the_original_message(
        self,
    ) -> None:
        """Characterization: nothing specified is still the generic message.

        This is the one case the original wording was right
        about, so it must survive the change.
        """
        assert _describe() == _NO_INPUT_SOURCE_MSG

    def test_s3_without_cache_dir_names_the_cache_dir(
        self,
    ) -> None:
        """The recorded finding: only the cache dir is missing."""
        msg = _describe(
            input_s3=True,
            input_bucket_name="bucket",
            input_prefix="prefix",
            input_data_name="data",
        )
        assert "--input-s3" in msg
        assert "--input-local-cache-dir" in msg
        assert msg != _NO_INPUT_SOURCE_MSG

    def test_s3_alone_names_every_missing_option(self) -> None:
        """A bare ``--input-s3`` is missing all four companions."""
        msg = _describe(input_s3=True)
        for option in (
            "--input-bucket-name",
            "--input-prefix",
            "--input-data-name",
            "--input-local-cache-dir",
        ):
            assert option in msg

    def test_gcs_is_reported_as_gcs(self) -> None:
        """The message must name the provider the user asked for."""
        msg = _describe(input_gcs=True)
        assert "--input-gcs" in msg
        assert "--input-s3" not in msg

    def test_bigquery_half_specified_names_the_other_half(
        self,
    ) -> None:
        """A dataset id without a table name is a BigQuery input."""
        msg = _describe(input_dataset_id="dataset")
        assert "BigQuery" in msg
        assert "--input-table-name" in msg
        assert "--input-dataset-id" not in msg

    def test_falls_back_when_nothing_is_missing(self) -> None:
        """A fully specified cloud input cannot name a missing option."""
        msg = _describe(
            input_s3=True,
            input_bucket_name="bucket",
            input_prefix="prefix",
            input_data_name="data",
            input_local_cache_dir="cache",
        )
        assert msg == _NO_INPUT_SOURCE_MSG


def _invoke_and_capture_errors(
    monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> tuple[int, str]:
    """Run the CLI and return its exit code plus logged errors.

    ``app_logger`` does not propagate to the root logger, so
    ``caplog`` sees nothing; capture it at the module attribute
    the command actually uses instead.
    """
    logger = MagicMock()
    monkeypatch.setattr(
        "maou.infra.console.pre_process.app_logger", logger
    )
    result = CliRunner().invoke(pre_process, args)
    errors = "\n".join(
        str(call.args[0])
        for call in logger.error.call_args_list
    )
    return result.exit_code, errors


class TestPreProcessCliReportsTheCause:
    """The CLI must surface the helper's message, not the old one."""

    def test_s3_without_cache_dir_reports_the_cache_dir(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pins the wiring: the else branch must use the helper."""
        exit_code, errors = _invoke_and_capture_errors(
            monkeypatch,
            [
                "--input-s3",
                "--input-bucket-name",
                "bucket",
                "--input-prefix",
                "prefix",
                "--input-data-name",
                "data",
                "--output-dir",
                "out",
            ],
        )
        assert exit_code != 0
        assert "--input-local-cache-dir" in errors

    def test_no_input_still_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Characterization: an empty invocation still fails."""
        exit_code, errors = _invoke_and_capture_errors(
            monkeypatch, ["--output-dir", "out"]
        )
        assert exit_code != 0
        assert "Please specify an input source" in errors
