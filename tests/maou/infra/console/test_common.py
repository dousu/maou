"""`maou.infra.console.common` のテスト."""

import subprocess
import sys
import textwrap
from pathlib import Path

from maou.infra.console.common import exit_skipping_teardown


def test_exit_skipping_teardown_is_noop_without_tensorrt() -> (
    None
):
    """TensorRT を使っていなければ何もせず戻る (通常の終了パスを保つ)."""
    exit_skipping_teardown(tensorrt=False)


def test_exit_skipping_teardown_flushes_and_exits(
    tmp_path: Path,
) -> None:
    """TensorRT 有効時は出力を flush してから即座にプロセスを終える.

    `os._exit` は destructor・atexit を走らせないため，バッファ済みの出力が
    失われないこと (明示 flush が効いていること) と，呼び出し後のコードが
    実行されないことを別プロセスで確かめる.
    """
    script = tmp_path / "exit_probe.py"
    script.write_text(
        textwrap.dedent(
            """
            from maou.infra.console.common import exit_skipping_teardown

            # flush していないバッファに残した状態で呼ぶ
            print("summary", end="")
            exit_skipping_teardown(tensorrt=True)
            print("NOT REACHED")
            """
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "summary", (
        "flush 済みの出力だけが残り，呼び出し後のコードは実行されない: "
        f"{result.stdout!r}"
    )
