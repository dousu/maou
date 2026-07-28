"""``--ab-mode`` の選択肢が Rust / Python / CLI の 3 箇所で一致することを pin する．

同じ列挙が 3 箇所にあり (Rust ``AbMode``，``AB_MODES``，click の ``Choice``)，
片方だけに足すと実行時に「未知の ab_mode」で落ちる．
"""

import click

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
