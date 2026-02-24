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


class TestIsOptionalDepError:
    """Tests for _is_optional_dep_error helper function."""

    @pytest.mark.parametrize(
        ("longrepr", "expected"),
        [
            pytest.param(
                "ModuleNotFoundError: No module named 'torch'",
                "torch",
                id="torch",
            ),
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
            "ModuleNotFoundError: No module named 'torch_utils'"
        )
        assert _is_optional_dep_error(longrepr) is None

    def test_handles_multiline_traceback(self) -> None:
        """The function should work with full multi-line tracebacks."""
        longrepr = (
            "Traceback (most recent call last):\n"
            "  File 'test_foo.py', line 1, in <module>\n"
            "    import torch\n"
            "ModuleNotFoundError: No module named 'torch'\n"
        )
        assert _is_optional_dep_error(longrepr) == "torch"


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
                "No module named 'torch'"
            ),
        ):
            assert _check_import_attr_error(exc) == "torch"

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
