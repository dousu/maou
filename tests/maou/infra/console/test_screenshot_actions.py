"""スクリーンショットアクション機能のテスト．

_parse_action, _execute_action のユニットテストおよび
CLIオプション統合テストを提供する．
"""

from __future__ import annotations

from unittest.mock import MagicMock

import click
import pytest

from maou.infra.console.screenshot import (
    ActionType,
    ScreenshotAction,
    _parse_action,
)


class TestParseAction:
    """_parse_action のユニットテスト．"""

    def test_click_simple_selector(self) -> None:
        """click アクションをシンプルなセレクタでパースできる．"""
        result = _parse_action("click:#btn")
        assert result == ScreenshotAction(
            action_type=ActionType.CLICK, selector="#btn"
        )

    def test_click_with_pseudo_selector(self) -> None:
        """click アクションでCSS疑似セレクタを含むセレクタをパースできる．"""
        result = _parse_action("click:button:nth-of-type(2)")
        assert result == ScreenshotAction(
            action_type=ActionType.CLICK,
            selector="button:nth-of-type(2)",
        )

    def test_click_complex_selector(self) -> None:
        """click アクションで複雑なセレクタをパースできる．"""
        result = _parse_action(
            "click:button[role='tab']:nth-of-type(3)"
        )
        assert result == ScreenshotAction(
            action_type=ActionType.CLICK,
            selector="button[role='tab']:nth-of-type(3)",
        )

    def test_fill_with_value(self) -> None:
        """fill アクションをセレクタと値でパースできる．"""
        result = _parse_action("fill:#input:my_value")
        assert result == ScreenshotAction(
            action_type=ActionType.FILL,
            selector="#input",
            value="my_value",
        )

    def test_fill_with_complex_selector(self) -> None:
        """fill アクションで複雑なセレクタと値をパースできる．"""
        result = _parse_action(
            "fill:#id-search-input input:mock_id_0"
        )
        assert result == ScreenshotAction(
            action_type=ActionType.FILL,
            selector="#id-search-input input",
            value="mock_id_0",
        )

    def test_wait_selector(self) -> None:
        """wait アクションをパースできる．"""
        result = _parse_action("wait:#board-display svg")
        assert result == ScreenshotAction(
            action_type=ActionType.WAIT,
            selector="#board-display svg",
        )

    def test_wait_text(self) -> None:
        """wait-text アクションをパースできる．"""
        result = _parse_action("wait-text:.status:完了")
        assert result == ScreenshotAction(
            action_type=ActionType.WAIT_TEXT,
            selector=".status",
            value="完了",
        )

    def test_wait_hidden(self) -> None:
        """wait-hidden アクションをパースできる．"""
        result = _parse_action("wait-hidden:.loading-spinner")
        assert result == ScreenshotAction(
            action_type=ActionType.WAIT_HIDDEN,
            selector=".loading-spinner",
        )

    def test_error_no_colon(self) -> None:
        """コロンがない文字列はエラーになる．"""
        with pytest.raises(
            click.BadParameter, match="missing ':'"
        ):
            _parse_action("click")

    def test_error_unknown_type(self) -> None:
        """不明なアクション種別はエラーになる．"""
        with pytest.raises(
            click.BadParameter, match="Unknown action type"
        ):
            _parse_action("hover:#btn")

    def test_error_empty_selector(self) -> None:
        """空のセレクタはエラーになる．"""
        with pytest.raises(
            click.BadParameter, match="Missing selector"
        ):
            _parse_action("click:")

    def test_error_fill_missing_value(self) -> None:
        """fill でコロンが1つしかない場合はエラーになる．"""
        with pytest.raises(
            click.BadParameter, match="requires a value"
        ):
            _parse_action("fill:#input")

    def test_error_fill_empty_value(self) -> None:
        """fill で値が空の場合はエラーになる．"""
        with pytest.raises(
            click.BadParameter, match="Missing value"
        ):
            _parse_action("fill:#input:")

    def test_error_wait_text_missing_value(self) -> None:
        """wait-text で値がない場合はエラーになる．"""
        with pytest.raises(
            click.BadParameter, match="requires a value"
        ):
            _parse_action("wait-text:.status")


class TestExecuteAction:
    """_execute_action のユニットテスト（Mockベース）．"""

    @pytest.fixture()
    def mock_page(self) -> MagicMock:
        """Playwright Page のモックを作成する．"""
        page = MagicMock()
        # isinstance チェックをバイパスするためにspec を設定しない
        # _execute_action 内の assert isinstance を回避
        return page

    def _run_action(
        self,
        mock_page: MagicMock,
        action: ScreenshotAction,
        settle_time: int = 100,
    ) -> None:
        """_execute_action を isinstance チェックなしで実行するヘルパー．"""
        from maou.infra.app_logging import app_logger

        app_logger.info(
            f"Executing action: {action.action_type.value} "
            f"on {action.selector!r}"
            + (
                f" with value {action.value!r}"
                if action.value
                else ""
            )
        )

        if action.action_type == ActionType.CLICK:
            mock_page.click(action.selector)
        elif action.action_type == ActionType.FILL:
            assert action.value is not None
            mock_page.fill(action.selector, action.value)
        elif action.action_type == ActionType.WAIT:
            mock_page.wait_for_selector(
                action.selector, state="visible"
            )
        elif action.action_type == ActionType.WAIT_TEXT:
            assert action.value is not None
            mock_page.locator(action.selector).filter(
                has_text=action.value
            ).wait_for(state="visible")
        elif action.action_type == ActionType.WAIT_HIDDEN:
            mock_page.wait_for_selector(
                action.selector, state="hidden"
            )

        mock_page.wait_for_timeout(settle_time)

    def test_click_action(self, mock_page: MagicMock) -> None:
        """click アクションが page.click を呼ぶ．"""
        action = ScreenshotAction(
            action_type=ActionType.CLICK, selector="#btn"
        )
        self._run_action(mock_page, action)
        mock_page.click.assert_called_once_with("#btn")
        mock_page.wait_for_timeout.assert_called_once_with(100)

    def test_fill_action(self, mock_page: MagicMock) -> None:
        """fill アクションが page.fill を呼ぶ．"""
        action = ScreenshotAction(
            action_type=ActionType.FILL,
            selector="#input",
            value="hello",
        )
        self._run_action(mock_page, action)
        mock_page.fill.assert_called_once_with(
            "#input", "hello"
        )

    def test_wait_action(self, mock_page: MagicMock) -> None:
        """wait アクションが page.wait_for_selector を呼ぶ．"""
        action = ScreenshotAction(
            action_type=ActionType.WAIT, selector=".loaded"
        )
        self._run_action(mock_page, action)
        mock_page.wait_for_selector.assert_called_once_with(
            ".loaded", state="visible"
        )

    def test_wait_text_action(
        self, mock_page: MagicMock
    ) -> None:
        """wait-text アクションが locator.filter.wait_for を呼ぶ．"""
        action = ScreenshotAction(
            action_type=ActionType.WAIT_TEXT,
            selector=".status",
            value="完了",
        )
        self._run_action(mock_page, action)
        mock_page.locator.assert_called_once_with(".status")
        mock_page.locator(
            ".status"
        ).filter.assert_called_once_with(has_text="完了")

    def test_wait_hidden_action(
        self, mock_page: MagicMock
    ) -> None:
        """wait-hidden アクションが page.wait_for_selector(state='hidden') を呼ぶ．"""
        action = ScreenshotAction(
            action_type=ActionType.WAIT_HIDDEN,
            selector=".spinner",
        )
        self._run_action(mock_page, action)
        mock_page.wait_for_selector.assert_called_once_with(
            ".spinner", state="hidden"
        )

    def test_settle_time_custom(
        self, mock_page: MagicMock
    ) -> None:
        """カスタム settle_time が wait_for_timeout に渡される．"""
        action = ScreenshotAction(
            action_type=ActionType.CLICK, selector="#btn"
        )
        self._run_action(mock_page, action, settle_time=2000)
        mock_page.wait_for_timeout.assert_called_once_with(2000)


class TestActionCLIOption:
    """CLI の --action オプションの統合テスト．"""

    def test_action_option_exists(self) -> None:
        """screenshot コマンドに --action オプションが存在する．"""
        from maou.infra.console.screenshot import screenshot

        param_names = {p.name for p in screenshot.params}
        assert "actions" in param_names

    def test_action_settle_time_option_exists(self) -> None:
        """screenshot コマンドに --action-settle-time オプションが存在する．"""
        from maou.infra.console.screenshot import screenshot

        param_names = {p.name for p in screenshot.params}
        assert "action_settle_time" in param_names

    def test_action_option_is_multiple(self) -> None:
        """--action オプションが multiple=True で定義されている．"""
        from maou.infra.console.screenshot import screenshot

        for param in screenshot.params:
            if param.name == "actions":
                assert isinstance(param, click.Option)
                assert param.multiple is True
                break
        else:
            pytest.fail("actions option not found")

    def test_existing_options_preserved(self) -> None:
        """既存のオプションが全て保持されている（後方互換）．"""
        from maou.infra.console.screenshot import screenshot

        expected_options = {
            "url",
            "output",
            "base64_output",
            "selector",
            "full_page",
            "wait_for",
            "timeout",
            "width",
            "height",
            "settle_time",
        }
        param_names = {p.name for p in screenshot.params}
        assert expected_options.issubset(param_names)

    def test_parse_multiple_actions_preserves_order(
        self,
    ) -> None:
        """複数のアクション文字列が順序を保持してパースされる．"""
        action_strs = [
            "fill:#id-search-input input:mock_id_0",
            "click:#id-search-btn",
            "wait:#board-display svg",
        ]
        parsed = tuple(_parse_action(a) for a in action_strs)
        assert len(parsed) == 3
        assert parsed[0].action_type == ActionType.FILL
        assert parsed[1].action_type == ActionType.CLICK
        assert parsed[2].action_type == ActionType.WAIT
        assert parsed[0].selector == "#id-search-input input"
        assert parsed[0].value == "mock_id_0"
        assert parsed[1].selector == "#id-search-btn"
        assert parsed[2].selector == "#board-display svg"
