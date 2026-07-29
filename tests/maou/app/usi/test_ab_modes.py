"""``--ab-mode`` の選択肢が Rust / Python / CLI の 3 箇所で一致することを pin する．

同じ列挙が 3 箇所にあり (Rust ``AbMode``，``AB_MODES``，click の ``Choice``)，
片方だけに足すと実行時に「未知の ab_mode」で落ちる．
"""

import click
import pytest

from maou._rust.maou_usi import run_selfplay
from maou.app.usi.selfplay import AB_MODES
from maou.infra.console.selfplay import selfplay


def _cli_choices() -> tuple[str, ...]:
    for param in selfplay.params:
        if param.name == "ab_mode":
            assert isinstance(param.type, click.Choice)
            return tuple(param.type.choices)
    raise AssertionError("--ab-mode オプションが見つからない")


def test_cli_choices_match_ab_modes() -> None:
    assert _cli_choices() == AB_MODES


def test_batch_mode_is_available() -> None:
    assert "batch" in AB_MODES
    assert "batch" in _cli_choices()


def test_timecurve_mode_is_available() -> None:
    assert "timecurve" in AB_MODES
    assert "timecurve" in _cli_choices()


def test_rust_knows_every_ab_mode() -> None:
    """3 箇所目 (Rust ``AbMode::parse``) も同じ列挙を持つこと．

    click と ``AB_MODES`` だけを直して Rust を忘れると，実行時に
    「未知の ab_mode」で落ちるまで気付けない．``games=0`` は ab_mode の
    解決より**後**で弾かれるので，対局を走らせずに解決だけを確かめられる．
    """
    for mode in AB_MODES:
        with pytest.raises((ValueError, RuntimeError)) as exc:
            run_selfplay(
                games=0, playouts=4, ab_mode=mode, verbose=False
            )
        assert "未知の ab_mode" not in str(exc.value), mode
