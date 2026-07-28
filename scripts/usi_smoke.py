#!/usr/bin/env python3
"""USI エンジンの headless smoke test (GUI 不要).

`docs/design/usi-engine/verification.md` §7 の手順を，GUI を持たない環境
(CI / 手元の Windows 機) でそのまま流せるようにしたもの．既定ではモデルを
読まず mock 評価器で動くので，wheel が壊れていないことの確認だけなら ONNX
ファイルの転送は要らない．

**一括 pipe は使わない** — `quit` が先に届くと `stop` が立ち，探索が 0
playout で終わる (既知の罠)．必ず応答を待ってから次を送る．

Usage:
    python scripts/usi_smoke.py
    python scripts/usi_smoke.py --model-path model.onnx --keep-alive 500
"""

from __future__ import annotations

import argparse
import io
import queue
import subprocess
import sys
import threading
import time
from typing import Optional

# 先手 5三歩 + 持駒金 / 後手 5一玉 = G*5b の 1 手詰め．dfpn 単独で解けるので
# モデルが無くても `go mate` を検証できる
MATE_IN_1_SFEN = "4k4/9/4P4/9/9/9/9/9/9 b G 1"


class Engine:
    """USI エンジンの子プロセスを行単位で駆動するラッパー.

    stdout は専用スレッドで読み出して Queue に積む (Windows のパイプは
    select() できないため)．stderr も別スレッドで吸い出して溜め込む —
    ドレインしないとエンジンのログでパイプが詰まって停止する．
    """

    def __init__(self, argv: list[str]) -> None:
        self.proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )
        self.stdout_q: queue.Queue[Optional[str]] = (
            queue.Queue()
        )
        self.stderr_lines: list[str] = []
        threading.Thread(
            target=self._pump_stdout, daemon=True
        ).start()
        threading.Thread(
            target=self._pump_stderr, daemon=True
        ).start()

    def _pump_stdout(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.stdout_q.put(line.rstrip("\r\n"))
        self.stdout_q.put(None)  # EOF

    def _pump_stderr(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            self.stderr_lines.append(line.rstrip("\r\n"))

    def send(self, line: str) -> None:
        """1 行送る."""
        assert self.proc.stdin is not None
        print(f"> {line}", flush=True)
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def wait_for(
        self, token: str, timeout: float
    ) -> tuple[str, int]:
        """`token` で始まる行が来るまで読む.

        Returns:
            (一致した行, それまでに観測した空行の数)．空行は `KeepAlive` の
            生存通知なので，未決 2 (既定 on にできるか) の証拠になる．
        """
        deadline = time.monotonic() + timeout
        blanks = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"{token!r} を {timeout}s 待っても来なかった"
                )
            try:
                line = self.stdout_q.get(timeout=remaining)
            except queue.Empty:
                raise TimeoutError(
                    f"{token!r} を {timeout}s 待っても来なかった"
                ) from None
            if line is None:
                raise RuntimeError(
                    "エンジンが終了した (stdout EOF)"
                )
            print(f"< {line!r}", flush=True)
            if line.strip() == "":
                blanks += 1
                continue
            if line.startswith(token):
                return line, blanks

    def dump_stderr(self) -> None:
        """診断用に stderr を吐く (致命エラーは stdout の info string にも出る)."""
        if not self.stderr_lines:
            print("--- engine stderr: (empty) ---", flush=True)
            return
        print("--- engine stderr ---", flush=True)
        for line in self.stderr_lines:
            print(f"  {line}", flush=True)
        print("--- end stderr ---", flush=True)


def run_smoke(
    engine_cmd: str,
    model_path: Optional[str],
    keep_alive: int,
    timeout: float,
) -> None:
    """USI の疎通を一通り確認する. 失敗時は例外を投げる."""
    eng = Engine([engine_cmd])
    try:
        eng.send("usi")
        _, _ = eng.wait_for("usiok", timeout)

        if model_path is not None:
            eng.send(
                f"setoption name ModelPath value {model_path}"
            )
        if keep_alive > 0:
            eng.send(
                f"setoption name KeepAlive value {keep_alive}"
            )
        eng.send("setoption name USI_Ponder value true")

        t0 = time.monotonic()
        eng.send("isready")
        _, blanks = eng.wait_for("readyok", timeout)
        elapsed = time.monotonic() - t0
        print(
            f"[isready] {elapsed:.1f}s / KeepAlive 空行 {blanks} 行",
            flush=True,
        )
        if keep_alive > 0 and blanks == 0:
            print(
                "[warn] 空行が 1 行も出なかった — モデルロードが速すぎて "
                "KeepAlive が発火していない (--keep-alive をもっと小さくするか，"
                "大きいモデルを指定する)",
                flush=True,
            )

        eng.send("position startpos")
        eng.send("go btime 10000 wtime 10000 binc 500 winc 500")
        best, _ = eng.wait_for("bestmove", timeout)
        tokens = best.split()
        if len(tokens) < 2 or tokens[1] in ("resign", "win"):
            raise AssertionError(f"初手が指せていない: {best}")

        # ponder: GUI と同じ手順で自分の手 + 予想した相手の手を進めてから go ponder
        if len(tokens) >= 4:
            eng.send(
                f"position startpos moves {tokens[1]} {tokens[3]}"
            )
            eng.send(
                "go ponder btime 10000 wtime 10000 binc 500 winc 500"
            )
            time.sleep(1.0)
            eng.send("ponderhit")
            eng.wait_for("bestmove", timeout)
        else:
            # mock 評価器では ponder 予想手が付かないことがある
            print(
                f"[warn] bestmove に ponder 予想手が付かない: {best}",
                flush=True,
            )

        # go mate は dfpn 単独なのでモデル無しでも動く
        eng.send(f"position sfen {MATE_IN_1_SFEN}")
        eng.send("go mate 10000")
        mate, _ = eng.wait_for("checkmate", timeout)
        if mate.split()[1:2] in ([], ["nomate"], ["timeout"]):
            raise AssertionError(
                f"1 手詰めを解けていない: {mate}"
            )

        eng.send("quit")
        eng.proc.wait(timeout=30)
        print("smoke ok", flush=True)
    except BaseException:
        eng.dump_stderr()
        eng.proc.kill()
        raise


def main() -> int:
    """コマンドラインから smoke test を走らせる."""
    # Windows の既定コンソール encoding (cp1252 / cp932) では，このスクリプト
    # 自身が出す日本語で UnicodeEncodeError になる (エンジンのプロトコル出力は
    # ASCII 安全なので無関係)．診断メッセージが化けても落ちない方を選ぶ
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(
                encoding="utf-8", errors="replace"
            )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine",
        default="maou-usi",
        help="エンジンの実行ファイル (既定: maou-usi．引数なしエントリポイント)",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="ONNX モデル．省略すると mock 評価器 (指し手の質は無意味)",
    )
    parser.add_argument(
        "--keep-alive",
        type=int,
        default=0,
        help="KeepAlive[ms]．未決 2 の観測用．0 で off (既定)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="1 応答あたりの待ち [s]",
    )
    args = parser.parse_args()

    try:
        run_smoke(
            args.engine,
            args.model_path,
            args.keep_alive,
            args.timeout,
        )
    except Exception as e:  # noqa: BLE001 — CI で原因を 1 行で出したい
        print(
            f"smoke FAILED: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
