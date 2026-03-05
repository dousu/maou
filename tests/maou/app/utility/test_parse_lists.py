"""Tests for _parse_int_list and _parse_float_list in utility CLI."""

from __future__ import annotations

import pytest
from click import BadParameter

from maou.infra.console.utility import (
    _parse_float_list,
    _parse_int_list,
)


class TestParseIntList:
    """_parse_int_list のテスト."""

    def test_none_returns_none(self) -> None:
        assert _parse_int_list(None) is None

    def test_single_value(self) -> None:
        assert _parse_int_list("256") == [256]

    def test_multiple_values(self) -> None:
        assert _parse_int_list("128,256,512") == [128, 256, 512]

    def test_whitespace_stripped(self) -> None:
        assert _parse_int_list(" 128 , 256 , 512 ") == [
            128,
            256,
            512,
        ]

    def test_non_integer_raises_bad_parameter(self) -> None:
        with pytest.raises(
            BadParameter, match="Invalid integer list"
        ):
            _parse_int_list("256,abc,512")

    def test_float_string_raises_bad_parameter(self) -> None:
        with pytest.raises(
            BadParameter, match="Invalid integer list"
        ):
            _parse_int_list("256,1.5,512")

    def test_zero_raises_bad_parameter(self) -> None:
        with pytest.raises(
            BadParameter, match="positive integers"
        ):
            _parse_int_list("0,256")

    def test_negative_raises_bad_parameter(self) -> None:
        with pytest.raises(
            BadParameter, match="positive integers"
        ):
            _parse_int_list("-1,256")

    def test_empty_string_raises_bad_parameter(self) -> None:
        with pytest.raises(BadParameter):
            _parse_int_list("")


class TestParseFloatList:
    """_parse_float_list のテスト."""

    def test_none_returns_none(self) -> None:
        assert _parse_float_list(None) is None

    def test_single_value(self) -> None:
        assert _parse_float_list("0.01") == [0.01]

    def test_multiple_values(self) -> None:
        result = _parse_float_list("0.001,0.01,0.1")
        assert result == [0.001, 0.01, 0.1]

    def test_whitespace_stripped(self) -> None:
        result = _parse_float_list(" 0.001 , 0.01 ")
        assert result == [0.001, 0.01]

    def test_non_numeric_raises_bad_parameter(self) -> None:
        with pytest.raises(
            BadParameter, match="Invalid float list"
        ):
            _parse_float_list("0.01,abc")

    def test_zero_raises_bad_parameter(self) -> None:
        with pytest.raises(BadParameter, match="positive"):
            _parse_float_list("0.0,0.01")

    def test_negative_raises_bad_parameter(self) -> None:
        with pytest.raises(BadParameter, match="positive"):
            _parse_float_list("-0.01,0.01")
