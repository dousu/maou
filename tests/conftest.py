"""Root-level conftest for handling optional dependency errors.

Converts collection-phase and setup-phase errors caused by known optional
dependencies into skips, allowing the test suite to run in base environments
without torch, onnxruntime, onnx, or gradio installed.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Generator

import pytest

# Optional dependencies that may not be installed in base environments.
# Only errors traceable to these modules are converted to skips;
# all other errors still fail loudly.
_OPTIONAL_DEPS: frozenset[str] = frozenset(
    {"torch", "onnxruntime", "onnx", "gradio", "matplotlib"}
)

# Pattern for AttributeError from pkgutil.resolve_name when a submodule
# fails to import (e.g. "module 'maou.infra.visualization' has no attribute 'gradio_server'").
_ATTR_ERROR_RE: re.Pattern[str] = re.compile(
    r"module '([^']+)' has no attribute '([^']+)'"
)


def _is_optional_dep_error(longrepr_text: str) -> str | None:
    """Return the module name if longrepr indicates a missing optional dependency.

    Args:
        longrepr_text: String representation of the error.

    Returns:
        The optional dependency name if matched, otherwise ``None``.
    """
    if "ModuleNotFoundError" not in longrepr_text:
        return None
    for dep in _OPTIONAL_DEPS:
        if f"No module named '{dep}'" in longrepr_text:
            return dep
    return None


def _check_import_attr_error(
    exc_value: BaseException,
) -> str | None:
    """Check if an AttributeError is caused by a missing optional dependency.

    When ``unittest.mock.patch`` resolves a dotted path like
    ``maou.infra.visualization.gradio_server.Cls``, ``pkgutil.resolve_name``
    catches the ``ImportError`` and raises ``AttributeError`` instead.  This
    function reconstructs the module path and attempts the import to surface
    the original ``ModuleNotFoundError``.

    Args:
        exc_value: The ``AttributeError`` exception instance.

    Returns:
        The optional dependency name if the import fails due to one,
        otherwise ``None``.
    """
    match = _ATTR_ERROR_RE.search(str(exc_value))
    if not match:
        return None

    parent_module, missing_attr = match.groups()
    full_module = f"{parent_module}.{missing_attr}"

    try:
        importlib.import_module(full_module)
    except ModuleNotFoundError as imp_err:
        msg = str(imp_err)
        for dep in _OPTIONAL_DEPS:
            if f"No module named '{dep}'" in msg:
                return dep
    except Exception:  # noqa: BLE001
        pass
    return None


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(
    collector: pytest.Collector,
) -> Generator[None, None, None]:
    """Convert collection errors for optional dependencies into skips."""
    outcome = yield  # pluggy.Result at runtime
    report = outcome.get_result()  # type: ignore[attr-defined]

    if report.outcome != "failed":
        return

    longrepr_text = str(report.longrepr)
    dep = _is_optional_dep_error(longrepr_text)
    if dep is None:
        return

    report.outcome = "skipped"
    report.longrepr = (
        str(collector.path),
        0,
        f"Skipped: optional dependency '{dep}' is not installed",
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item,
    call: pytest.CallInfo[None],
) -> Generator[None, None, None]:
    """Convert test setup errors caused by optional dependencies into skips."""
    outcome = yield  # pluggy.Result at runtime
    report = outcome.get_result()  # type: ignore[attr-defined]

    if report.when != "setup" or not report.failed:
        return

    if call.excinfo is None:
        return

    dep: str | None = None

    # Case 1: Direct ModuleNotFoundError during setup
    if call.excinfo.typename == "ModuleNotFoundError":
        dep = _is_optional_dep_error(str(call.excinfo.value))

    # Case 2: AttributeError from pkgutil.resolve_name (e.g. via @patch)
    elif call.excinfo.typename == "AttributeError":
        dep = _check_import_attr_error(call.excinfo.value)

    if dep is None:
        return

    report.outcome = "skipped"
    report.longrepr = (
        str(item.path),
        item.reportinfo()[1] or 0,
        f"Skipped: optional dependency '{dep}' is not installed",
    )
