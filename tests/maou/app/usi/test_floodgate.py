"""floodgate 対局ユースケースの単体テスト (認証情報の組み立て)."""

import re

from maou.app.usi.floodgate import (
    DEFAULT_GAME_NAME,
    DEFAULT_HOST,
    DEFAULT_PORT,
    build_password,
    generate_trip,
)


def test_default_target_is_floodgate() -> None:
    """既定の接続先が floodgate のレーティング対局場である．"""
    assert DEFAULT_HOST == "wdoor.c.u-tokyo.ac.jp"
    assert DEFAULT_PORT == 4081
    # 300 秒 + 10 秒加算の対局場 (毎時 0 分/30 分に対局が組まれる)
    assert DEFAULT_GAME_NAME == "floodgate-300-10F"


def test_build_password_embeds_game_name() -> None:
    """floodgate のパスワード欄はゲーム名と trip をカンマで繋ぐ．"""
    assert (
        build_password(
            game_name="floodgate-300-10F", trip="secret"
        )
        == "floodgate-300-10F,secret"
    )


def test_build_password_without_game_name() -> None:
    """ゲーム名が空なら trip だけを送る (一般の CSA サーバ向け)．"""
    assert (
        build_password(game_name="", trip="secret") == "secret"
    )
    assert (
        build_password(game_name="   ", trip="secret")
        == "secret"
    )


def test_generate_trip_is_random_and_usable() -> None:
    """trip は毎回異なり，パスワード欄に使える文字だけからなる．"""
    first = generate_trip()
    second = generate_trip()
    assert first != second
    # 空白を含むとパスワード欄が壊れる (CSA は空白区切り)
    assert re.fullmatch(r"[0-9a-f]+", first)
    assert len(first) >= 16
