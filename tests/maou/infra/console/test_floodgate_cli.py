"""`maou floodgate` の CLI・E2E テスト．

E2E はローカルのモック CSA サーバ相手に 1 局を通す (mock 評価器)．
持ち時間を小さくして数秒で完走させる．
"""

import re
import socket
import threading
from typing import IO

from click.testing import CliRunner, Result

from maou.infra.console.floodgate import floodgate

_GAME_SUMMARY = """BEGIN Game_Summary
Protocol_Version:1.2
Protocol_Mode:Server
Format:Shogi 1.0
Declaration:Jishogi 1.1
Game_ID:{game_id}
Name+:maou_test
Name-:mock_opponent
Your_Turn:+
Rematch_On_Draw:NO
To_Move:+
Max_Moves:512
BEGIN Time
Time_Unit:1sec
Total_Time:10
Byoyomi:1
Least_Time_Per_Move:0
END Time
BEGIN Position
P1-KY-KE-GI-KI-OU-KI-GI-KE-KY
P2 * -HI *  *  *  *  * -KA *
P3-FU-FU-FU-FU-FU-FU-FU-FU-FU
P4 *  *  *  *  *  *  *  *  *
P5 *  *  *  *  *  *  *  *  *
P6 *  *  *  *  *  *  *  *  *
P7+FU+FU+FU+FU+FU+FU+FU+FU+FU
P8 * +KA *  *  *  *  * +HI *
P9+KY+KE+GI+KI+OU+KI+GI+KE+KY
P+
P-
+
END Position
END Game_Summary"""


def _read_line(stream: IO[str]) -> str:
    """内容のある行を 1 つ読む (keep-alive の空行は読み飛ばす)．"""
    while True:
        line = stream.readline()
        if not line:
            return ""
        stripped = line.strip()
        if stripped:
            return stripped


def _serve_one_game(
    listener: socket.socket, game_id: str, captured: list[str]
) -> None:
    """1 接続 = 1 対局を演じ，受け取った LOGIN 行を `captured` に残す．"""
    conn, _ = listener.accept()
    with conn:
        reader = conn.makefile(
            "r", encoding="ascii", newline="\n"
        )
        writer = conn.makefile(
            "w", encoding="ascii", newline="\n"
        )
        captured.append(_read_line(reader))
        writer.write("LOGIN:maou_test OK\n")
        writer.write(
            _GAME_SUMMARY.format(game_id=game_id) + "\n"
        )
        writer.flush()

        assert _read_line(reader).startswith("AGREE")
        writer.write(f"START:{game_id}\n")
        writer.flush()

        first = _read_line(reader)
        assert first.startswith("+"), first
        # 先手の初手では 3d に到達できないので後手のこの応手は常に合法
        writer.write(f"{first},T1\n-3334FU,T1\n")
        writer.flush()

        second = _read_line(reader)
        assert second.startswith("+"), second
        # 後手の投了で終局 (指し手・理由・勝敗の 3 行)
        writer.write(f"{second},T1\n%TORYO,T1\n#RESIGN\n#WIN\n")
        writer.flush()


def _run_against_mock(
    extra_args: list[str],
) -> tuple[Result, list[str]]:
    """モックサーバを立てて CLI を 1 局走らせる．"""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    captured: list[str] = []
    server = threading.Thread(
        target=_serve_one_game,
        args=(listener, "mockgame", captured),
        daemon=True,
    )
    server.start()
    try:
        result = CliRunner().invoke(
            floodgate,
            [
                "--login-name",
                "maou_test",
                "--game-name",
                "mock-300-10F",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--games",
                "1",
                # 8GB DevContainer 用にメモリ確保を小さくする
                "--node-capacity",
                "16384",
                "--no-root-dfpn",
                "--no-leaf-mate",
                "--quiet",
                *extra_args,
            ],
        )
    finally:
        server.join(timeout=120)
        listener.close()
    return result, captured


def test_floodgate_help() -> None:
    """--help が接続せずに使い方を表示する．"""
    result = CliRunner().invoke(floodgate, ["--help"])
    assert result.exit_code == 0
    assert "--login-name" in result.output
    assert "--password" in result.output
    assert "--games" in result.output


def test_floodgate_requires_login_name() -> None:
    """ログイン名は必須 (レーティングの集計単位)．"""
    result = CliRunner().invoke(floodgate, [])
    assert result.exit_code != 0


def test_floodgate_plays_one_game_and_reports_result() -> None:
    """指定した trip でログインし，1 局を終局まで通して結果を出す．"""
    result, captured = _run_against_mock(
        ["--password", "trip123"]
    )
    assert result.exit_code == 0, result.output
    # パスワード欄はゲーム名と trip をカンマで繋いだもの
    assert captured == ["LOGIN maou_test mock-300-10F,trip123"]
    assert "win (#RESIGN)" in result.output
    assert "mock_opponent" in result.output


def test_floodgate_generates_trip_when_password_omitted() -> (
    None
):
    """パスワード未指定なら trip を自動生成してログインに使う．"""
    result, captured = _run_against_mock([])
    assert result.exit_code == 0, result.output
    assert len(captured) == 1
    match = re.fullmatch(
        r"LOGIN maou_test mock-300-10F,([0-9a-f]{16,})",
        captured[0],
    )
    assert match is not None, captured[0]
