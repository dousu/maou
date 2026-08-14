"""Tests for the optional dependency skip logic in tests/conftest.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

# Load helpers from the root conftest, which is not directly importable
# as a regular module under --import-mode=importlib.
_conftest_path = Path(__file__).parent / "conftest.py"
_spec = importlib.util.spec_from_file_location(
    "root_conftest", _conftest_path
)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_is_optional_dep_error = _mod._is_optional_dep_error
_check_import_attr_error = _mod._check_import_attr_error
_format_uncollected_summary = _mod.format_uncollected_summary
_pytest_make_collect_report = _mod.pytest_make_collect_report


class TestIsOptionalDepError:
    """Tests for _is_optional_dep_error helper function."""

    @pytest.mark.parametrize(
        ("longrepr", "expected"),
        [
            pytest.param(
                "ModuleNotFoundError: No module named 'onnxruntime'",
                "onnxruntime",
                id="onnxruntime",
            ),
            pytest.param(
                "ModuleNotFoundError: No module named 'onnx'",
                "onnx",
                id="onnx",
            ),
            pytest.param(
                "ModuleNotFoundError: No module named 'gradio'",
                "gradio",
                id="gradio",
            ),
            pytest.param(
                "ModuleNotFoundError: No module named 'matplotlib'",
                "matplotlib",
                id="matplotlib",
            ),
        ],
    )
    def test_returns_dep_name_for_known_optional_deps(
        self,
        longrepr: str,
        expected: str,
    ) -> None:
        """Known optional dependency errors should return the dep name."""
        assert _is_optional_dep_error(longrepr) == expected

    def test_returns_none_for_non_import_error(self) -> None:
        """Non-ImportError tracebacks should return None."""
        longrepr = "SyntaxError: invalid syntax"
        assert _is_optional_dep_error(longrepr) is None

    def test_returns_none_for_unknown_module(self) -> None:
        """ImportErrors for non-optional modules should return None."""
        longrepr = (
            "ModuleNotFoundError: No module named 'nonexistent'"
        )
        assert _is_optional_dep_error(longrepr) is None

    def test_returns_none_for_partial_match(self) -> None:
        """Submodule names that start with an optional dep should not match."""
        longrepr = (
            "ModuleNotFoundError: No module named 'onnx_utils'"
        )
        assert _is_optional_dep_error(longrepr) is None

    def test_handles_multiline_traceback(self) -> None:
        """The function should work with full multi-line tracebacks."""
        longrepr = (
            "Traceback (most recent call last):\n"
            "  File 'test_foo.py', line 1, in <module>\n"
            "    import onnx\n"
            "ModuleNotFoundError: No module named 'onnx'\n"
        )
        assert _is_optional_dep_error(longrepr) == "onnx"


class TestTorchIsNotOptional:
    """``torch`` must never be treated as a skippable dependency.

    A collect-phase skip drops a whole module, and torch gates most
    of the suite (`tests/maou/app/learning/` and every other module
    importing it).  While torch sat in ``_OPTIONAL_DEPS`` a run in an
    environment without a GPU extra reported itself green having
    executed almost nothing, which is how eight consecutive audit
    runs came to re-confirm the same finding.  A GPU extra
    (``uv sync --extra cpu`` / ``--extra cuda``) is a prerequisite,
    so its absence must fail loudly.
    """

    def test_torch_is_not_in_the_optional_set(self) -> None:
        """torch must be absent from the skip-conversion set."""
        assert "torch" not in _mod._OPTIONAL_DEPS

    def test_missing_torch_is_not_converted_to_a_skip(
        self,
    ) -> None:
        """A torch ImportError must not be classified as skippable."""
        longrepr = (
            "ModuleNotFoundError: No module named 'torch'"
        )
        assert _is_optional_dep_error(longrepr) is None

    def test_collect_hook_leaves_a_torch_failure_failed(
        self,
    ) -> None:
        """The collect hook must not rewrite a torch failure.

        This is the behaviour the summary accumulator could only
        describe after the fact; here the module is never dropped
        in the first place.
        """
        recorded: dict[str, set[str]] = _mod._UNCOLLECTED_BY_DEP
        before = {
            dep: set(paths) for dep, paths in recorded.items()
        }
        try:
            recorded.clear()

            class _Report:
                outcome = "failed"
                longrepr = "ModuleNotFoundError: No module named 'torch'"

            class _Collector:
                path = Path("tests/a/test_torch_user.py")

            report = _Report()
            gen = _pytest_make_collect_report(_Collector())
            next(gen)
            with pytest.raises(StopIteration):
                gen.send(_FakeOutcome(report))

            assert report.outcome == "failed"
            assert recorded == {}
        finally:
            recorded.clear()
            recorded.update(before)


class TestCheckImportAttrError:
    """Tests for _check_import_attr_error helper function."""

    def test_returns_none_for_unrelated_attribute_error(
        self,
    ) -> None:
        """Non-module AttributeErrors should return None."""
        exc = AttributeError(
            "'str' object has no attribute 'foo'"
        )
        assert _check_import_attr_error(exc) is None

    def test_returns_none_when_module_exists(self) -> None:
        """AttributeErrors for modules that exist should return None."""
        # os.path is a real module that imports fine
        exc = AttributeError(
            "module 'os' has no attribute 'nonexistent'"
        )
        assert _check_import_attr_error(exc) is None

    def test_detects_optional_dep_via_import(self) -> None:
        """Should detect when a module fails to import due to optional dep."""
        exc = AttributeError(
            "module 'some.package' has no attribute 'missing_mod'"
        )
        # Mock importlib.import_module to raise ModuleNotFoundError
        with patch(
            "importlib.import_module",
            side_effect=ModuleNotFoundError(
                "No module named 'gradio'"
            ),
        ):
            assert _check_import_attr_error(exc) == "gradio"

    def test_returns_none_for_non_optional_import_error(
        self,
    ) -> None:
        """Should return None when import fails for a non-optional module."""
        exc = AttributeError(
            "module 'some.package' has no attribute 'missing_mod'"
        )
        with patch(
            "importlib.import_module",
            side_effect=ModuleNotFoundError(
                "No module named 'nonexistent'"
            ),
        ):
            assert _check_import_attr_error(exc) is None

    def test_returns_none_when_import_raises_other_exception(
        self,
    ) -> None:
        """Should return None when import raises a non-ModuleNotFoundError."""
        exc = AttributeError(
            "module 'some.package' has no attribute 'missing_mod'"
        )
        with patch(
            "importlib.import_module",
            side_effect=RuntimeError("something else"),
        ):
            assert _check_import_attr_error(exc) is None


class TestFormatUncollectedSummary:
    """Tests for the uncollected-module terminal summary.

    A collect-phase skip drops a whole module, which pytest's
    tail reports as a single skip.  These tests pin that the
    drop is stated explicitly instead.
    """

    def test_empty_when_nothing_was_dropped(self) -> None:
        """No modules dropped means no summary at all."""
        assert _format_uncollected_summary({}) == []

    def test_names_dep_module_count_and_paths(self) -> None:
        """The summary must name the dep, the count and the paths."""
        lines = _format_uncollected_summary(
            {
                "onnxruntime": {
                    "tests/a/test_one.py",
                    "tests/a/test_two.py",
                }
            }
        )
        text = "\n".join(lines)
        assert "onnxruntime" in text
        assert "2 module(s) skipped entirely" in text
        assert "tests/a/test_one.py" in text
        assert "tests/a/test_two.py" in text

    def test_reports_every_dependency(self) -> None:
        """Each missing dependency gets its own line."""
        lines = _format_uncollected_summary(
            {
                "onnx": {"tests/a/test_one.py"},
                "gradio": {"tests/b/test_two.py"},
            }
        )
        text = "\n".join(lines)
        assert "onnx" in text
        assert "gradio" in text

    def test_collect_hook_records_the_dropped_module(
        self,
    ) -> None:
        """The collect hook must feed the summary accumulator.

        Without this the summary stays empty in exactly the
        situation it exists for.
        """
        recorded: dict[str, set[str]] = _mod._UNCOLLECTED_BY_DEP
        before = {
            dep: set(paths) for dep, paths in recorded.items()
        }
        try:
            recorded.clear()

            class _Report:
                outcome = "failed"
                longrepr = "ModuleNotFoundError: No module named 'onnxruntime'"

            class _Collector:
                path = Path("tests/a/test_dropped.py")

            gen = _pytest_make_collect_report(_Collector())
            next(gen)
            with pytest.raises(StopIteration):
                gen.send(_FakeOutcome(_Report()))

            assert recorded == {
                "onnxruntime": {
                    str(Path("tests/a/test_dropped.py"))
                }
            }
        finally:
            recorded.clear()
            recorded.update(before)


class _FakeOutcome:
    """Stand-in for the pluggy result handed to a hookwrapper."""

    def __init__(self, result: object) -> None:
        self._result = result

    def get_result(self) -> object:
        """Return the wrapped report."""
        return self._result
