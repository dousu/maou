"""``scripts/colab_arm0_job.py`` (Arm 0 の Colab 駆動) のテスト．

VM 依存の部分 (Drive / GPU) は tmp_path で代替し，段階の接続だけを確かめる．
"""

from __future__ import annotations

import shutil
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "scripts")
)

import colab_arm0_job as job_mod


class _UnassignHandler(BaseHTTPRequestHandler):
    """``POST /unassign`` を記録するだけの runtime 管理サービスのスタブ．"""

    calls: ClassVar[list[str]] = []

    def do_POST(self) -> None:
        type(self).calls.append(self.path)
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def unassign_server() -> object:
    _UnassignHandler.calls = []
    srv = HTTPServer(("127.0.0.1", 0), _UnassignHandler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield srv
    srv.shutdown()


class TestUnassignRuntime:
    def test_posts_unassign_after_grace(
        self,
        unassign_server: HTTPServer,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(
            "TBE_RUNTIME_ADDR",
            f"127.0.0.1:{unassign_server.server_port}",
        )
        job_mod._unassign_runtime(0.0, tmp_path / "KEEP_VM")
        assert _UnassignHandler.calls == ["/unassign"]

    def test_hold_file_keeps_vm(
        self,
        unassign_server: HTTPServer,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(
            "TBE_RUNTIME_ADDR",
            f"127.0.0.1:{unassign_server.server_port}",
        )
        hold = tmp_path / "KEEP_VM"
        hold.touch()
        job_mod._unassign_runtime(0.0, hold)
        assert _UnassignHandler.calls == []

    def test_without_runtime_addr_is_noop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("TBE_RUNTIME_ADDR", raising=False)
        job_mod._unassign_runtime(0.0, tmp_path / "KEEP_VM")


@pytest.mark.skipif(
    shutil.which("rsync") is None, reason="rsync がない"
)
class TestPreprocessStageSync:
    """preprocess の出力は learn に入る前に Drive へ退避される．"""

    def _job(self, tmp_path: Path, tag: str) -> job_mod.Job:
        work, drive = tmp_path / "work", tmp_path / "drive"
        (work / "hcpe").mkdir(parents=True)
        (work / "val").mkdir()
        drive.mkdir()
        # maou の代わりに，pre-process の出力ディレクトリへ 1 ファイル書く
        # ダミーコマンドを噛ませる (引数はそのまま無視される)
        fake = (
            "import sys, pathlib; "
            "out = pathlib.Path(sys.argv[sys.argv.index('--output-dir') + 1]); "
            "out.mkdir(parents=True); (out / 'a.feather').write_bytes(b'x' * 10)"
        )
        ns = job_mod._parse_args(
            [
                "--train-hcpe",
                "hcpe",
                "--val-data",
                "val",
                "--work-root",
                str(work),
                "--drive-root",
                str(drive),
                "--tag",
                tag,
                "--maou",
                f"{sys.executable} -c {fake!r}",
            ]
        )
        return job_mod.Job(ns)

    def test_output_is_copied_to_drive(
        self, tmp_path: Path
    ) -> None:
        job = self._job(tmp_path, "t1")
        assert job.preprocess() == 0
        copied = (
            tmp_path
            / "drive"
            / "preprocess"
            / "preprocess_t1"
            / "a.feather"
        )
        assert copied.read_bytes() == b"x" * 10

    def test_no_drive_skips_copy(self, tmp_path: Path) -> None:
        job = self._job(tmp_path, "t2")
        job.drive = None
        assert job.preprocess() == 0
        assert not (tmp_path / "drive" / "preprocess").exists()
