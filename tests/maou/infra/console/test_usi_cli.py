"""`maou usi` / `maou-usi` の CLI・E2E テスト．

E2E は subprocess で USI セッション台本を流す (mock 評価器)．探索予算は
秒読み小 + 小さい木で数秒以内に完走する．
"""

import subprocess
import sys
import time

from click.testing import CliRunner

from maou.infra.console.usi import usi

_ENTRY = "from maou.infra.console.usi import main; main()"


def _run_engine(
    script: str, timeout: float = 60.0
) -> list[str]:
    """引数なしエントリポイント相当でセッション台本を流して stdout 行を返す．"""
    proc = subprocess.run(
        [sys.executable, "-c", _ENTRY],
        input=script,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}"
    return proc.stdout.splitlines()


def test_usi_help() -> None:
    """--help がエンジンを起動せずに使い方を表示する．"""
    result = CliRunner().invoke(usi, ["--help"])
    assert result.exit_code == 0
    assert "USI engine" in result.output
    assert "--model-path" in result.output


def test_usi_session_e2e() -> None:
    """usi→isready→position→go の 1 手番セッションが実思考して完走する．

    quit を go と同時に送ると reader が stop フラグを先に立てて 0 playout
    経路になるため，bestmove を待ってから quit を送る (GUI の実挙動と同じ)．
    """
    proc = subprocess.Popen(
        [sys.executable, "-c", _ENTRY],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert proc.stdin is not None and proc.stdout is not None
    lines: list[str] = []
    try:
        proc.stdin.write(
            "usi\n"
            "setoption name RootDfpn value false\n"
            "setoption name LeafMate value false\n"
            "setoption name NodeCapacity value 16384\n"
            "setoption name NetworkDelay value 0\n"
            "isready\n"
            "usinewgame\n"
            "position startpos moves 7g7f\n"
            "go btime 0 wtime 0 byoyomi 300\n"
        )
        proc.stdin.flush()
        # bestmove まで読む (byoyomi 300ms + mock 評価器なので即返る)
        for line in proc.stdout:
            lines.append(line.rstrip("\n"))
            if line.startswith("bestmove "):
                break
        proc.stdin.write("quit\n")
        proc.stdin.flush()
        proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 0
    assert any(
        line.startswith("id name maou") for line in lines
    )
    assert "usiok" in lines
    assert "readyok" in lines
    # mock 評価器の明示 (ModelPath 未指定)
    assert any(
        line.startswith("info string") and "mock" in line
        for line in lines
    )
    bestmove = next(
        line for line in lines if line.startswith("bestmove ")
    )
    move = bestmove.split()[1]
    assert move not in ("resign", "win"), "平手序盤で投了しない"
    # info サマリ: 実思考している (playout > 0) こと，pv は行末尾
    info = next(
        line
        for line in lines
        if line.startswith("info ") and " pv " in line
    )
    assert " score " in info.split(" pv ")[0]
    nodes = int(info.split(" nodes ")[1].split()[0])
    assert nodes > 0, f"実思考していない: {info}"


def test_usi_option_declarations() -> None:
    """usi 応答でオプションが宣言される．"""
    lines = _run_engine("usi\nquit\n")
    decls = [
        line
        for line in lines
        if line.startswith("option name ")
    ]
    names = {line.split()[2] for line in decls}
    assert {
        "ModelPath",
        "Threads",
        "BatchSize",
        "NetworkDelay",
        "USI_Hash",
    } <= names
    # M3 (ponder) まで USI_Ponder は宣言しない (設計 doc §8.4)
    assert "USI_Ponder" not in names


def test_usi_stop_responds_quickly() -> None:
    """go infinite 中の stop に短時間で bestmove が返り quit で終了する．"""
    proc = subprocess.Popen(
        [sys.executable, "-c", _ENTRY],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdin is not None
    try:
        proc.stdin.write(
            "usi\n"
            "setoption name RootDfpn value false\n"
            "setoption name LeafMate value false\n"
            "setoption name NodeCapacity value 16384\n"
            "isready\n"
            "position startpos\n"
            "go infinite\n"
        )
        proc.stdin.flush()
        time.sleep(1.5)  # 無期限探索を回しておく
        stopped_at = time.monotonic()
        proc.stdin.write("stop\nquit\n")
        proc.stdin.flush()
        proc.wait(timeout=15)
        elapsed = time.monotonic() - stopped_at
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 0
    # stop 即応 (mock 評価器なら実測ミリ秒オーダ．CI 余裕を見た緩い上限)
    assert elapsed < 5.0, f"stop から終了まで {elapsed:.1f}s"
    stdout = proc.stdout.read() if proc.stdout else ""
    assert any(
        line.startswith("bestmove ")
        for line in stdout.splitlines()
    )
