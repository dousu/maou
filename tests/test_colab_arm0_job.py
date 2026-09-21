"""``scripts/colab_arm0_job.py`` (Arm 0 の Colab 駆動) のテスト．

VM 依存の部分 (Drive / GPU) は tmp_path で代替し，段階の接続だけを確かめる．
失敗時の退避 (diag + Drive) と unassign の条件もここで固定する．
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


def _make_job(
    tmp_path: Path,
    tag: str,
    maou_code: str,
    extra: list[str] | None = None,
) -> job_mod.Job:
    """work / drive を tmp に切り，maou の代わりに ``maou_code`` を走らせる Job．"""
    work, drive = tmp_path / "work", tmp_path / "drive"
    (work / "hcpe").mkdir(parents=True)
    (work / "val").mkdir()
    (drive / "val").mkdir(parents=True)
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
            f"{sys.executable} -c {maou_code!r}",
            *(extra or []),
        ]
    )
    return job_mod.Job(ns)


@pytest.mark.skipif(
    shutil.which("rsync") is None, reason="rsync がない"
)
class TestFailureEvacuation:
    """段階が失敗したら診断情報と job dir を Drive へ退避し，OK のときだけ unassign する．"""

    FAIL = "import sys; sys.exit(3)"

    def test_failed_stage_evacuates_to_drive(
        self, tmp_path: Path
    ) -> None:
        job = _make_job(
            tmp_path,
            "f1",
            self.FAIL,
            ["--stages", "preprocess"],
        )
        # soften の出力に相当する大物は退避しない
        job.soft_out.write_bytes(b"f" * 100)
        assert job.main() == 3
        assert job.evacuated is True
        drive_job = (
            tmp_path
            / "drive"
            / "maou_test"
            / "jobs"
            / "arm0_f1"
        )
        assert (
            "STAGE preprocess DONE rc=3"
            in (drive_job / "STATUS").read_text()
        )
        assert (drive_job / "preprocess.log").exists()
        for name in ("dmesg", "free", "df", "nvidia-smi", "ps"):
            assert (drive_job / "diag" / f"{name}.txt").exists()
        snapshot = drive_job / "diag" / "driver_log.txt"
        assert (
            "== preprocess: FAILED rc=3" in snapshot.read_text()
        )
        assert not (drive_job / "arm0_debias.feather").exists()
        assert not (drive_job / "driver.log").exists()

    def test_failed_stage_syncs_models_and_logs(
        self, tmp_path: Path
    ) -> None:
        # learn が途中で落ちても，直前に保存された checkpoint を残す
        code = (
            "import sys, pathlib; "
            "d = pathlib.Path(sys.argv[sys.argv.index('--model-dir') + 1]); "
            "(d / 'model_1.onnx').write_bytes(b'm' * 10); sys.exit(1)"
        )
        job = _make_job(
            tmp_path, "f2", code, ["--stages", "learn"]
        )
        job.pre_out.mkdir(parents=True)
        assert job.main() == 1
        assert job.evacuated is True
        copied = (
            tmp_path
            / "drive"
            / "maou_test"
            / "models"
            / "models_f2"
            / "model_1.onnx"
        )
        assert copied.read_bytes() == b"m" * 10

    def test_no_drive_failure_is_not_evacuated(
        self, tmp_path: Path
    ) -> None:
        job = _make_job(
            tmp_path,
            "f3",
            self.FAIL,
            ["--stages", "preprocess"],
        )
        job.drive = None
        assert job.main() == 3
        assert job.evacuated is False
        # 診断はローカルには残す
        assert (job.diag_dir / "free.txt").exists()

    def test_main_unassigns_after_failed_but_evacuated_job(
        self,
        unassign_server: HTTPServer,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(
            "TBE_RUNTIME_ADDR",
            f"127.0.0.1:{unassign_server.server_port}",
        )
        job = _make_job(
            tmp_path,
            "f4",
            self.FAIL,
            [
                "--stages",
                "preprocess",
                "--unassign-on-done",
                "--unassign-grace-min",
                "0",
            ],
        )
        argv = ["colab_arm0_job.py", *_argv_of(job.ns)]
        monkeypatch.setattr(sys, "argv", argv)
        assert job_mod.main() == 3
        assert _UnassignHandler.calls == ["/unassign"]

    def test_main_keeps_vm_when_evacuation_fails(
        self,
        unassign_server: HTTPServer,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(
            "TBE_RUNTIME_ADDR",
            f"127.0.0.1:{unassign_server.server_port}",
        )
        job = _make_job(
            tmp_path,
            "f5",
            self.FAIL,
            [
                "--stages",
                "preprocess",
                "--unassign-on-done",
                "--unassign-grace-min",
                "0",
                "--no-drive",
            ],
        )
        argv = ["colab_arm0_job.py", *_argv_of(job.ns)]
        monkeypatch.setattr(sys, "argv", argv)
        assert job_mod.main() == 3
        assert _UnassignHandler.calls == []

    def test_evacuate_only_runs_no_stage(
        self, tmp_path: Path
    ) -> None:
        job = _make_job(
            tmp_path, "f6", self.FAIL, ["--evacuate-only"]
        )
        job.status.write_text(
            "STAGE learn DONE rc=1 finished=x\n"
        )
        assert job.main() == 0
        assert job.evacuated is True
        drive_job = (
            tmp_path
            / "drive"
            / "maou_test"
            / "jobs"
            / "arm0_f6"
        )
        assert (
            (drive_job / "STATUS")
            .read_text()
            .startswith("STAGE learn DONE rc=1")
        )
        assert (drive_job / "diag" / "dmesg.txt").exists()


def _argv_of(ns: object) -> list[str]:
    """テスト用に Namespace から main() へ渡す argv を組み立て直す．"""
    a = [
        "--train-hcpe",
        ns.train_hcpe,  # type: ignore[attr-defined]
        "--val-data",
        ns.val_data,  # type: ignore[attr-defined]
        "--work-root",
        str(ns.work_root),  # type: ignore[attr-defined]
        "--drive-root",
        str(ns.drive_root),  # type: ignore[attr-defined]
        "--tag",
        ns.tag,  # type: ignore[attr-defined]
        "--maou",
        ns.maou,  # type: ignore[attr-defined]
        "--stages",
        ns.stages,  # type: ignore[attr-defined]
        "--unassign-grace-min",
        str(ns.unassign_grace_min),  # type: ignore[attr-defined]
    ]
    if ns.unassign_on_done:  # type: ignore[attr-defined]
        a.append("--unassign-on-done")
    if ns.no_drive:  # type: ignore[attr-defined]
        a.append("--no-drive")
    return a


@pytest.mark.skipif(
    shutil.which("rsync") is None, reason="rsync がない"
)
class TestReusePreprocessed:
    """``--train-preprocessed`` は前処理済を Drive から戻し，soften / preprocess を飛ばす．"""

    def test_fetch_restores_and_stages_are_skipped(
        self, tmp_path: Path
    ) -> None:
        work, drive = tmp_path / "work", tmp_path / "drive"
        (drive / "val").mkdir(parents=True)
        pre = drive / "preprocess" / "preprocess_old"
        pre.mkdir(parents=True)
        (pre / "a.feather").write_bytes(b"p" * 7)
        ns = job_mod._parse_args(
            [
                "--train-preprocessed",
                "preprocess/preprocess_old",
                "--val-data",
                "val",
                "--work-root",
                str(work),
                "--drive-root",
                str(drive),
                "--tag",
                "new",
                "--stages",
                "fetch,soften,preprocess",
                "--maou",
                f"{sys.executable} -c 'raise SystemExit(9)'",
            ]
        )
        job = job_mod.Job(ns)
        assert (
            job.pre_out
            == work / "preprocess" / "preprocess_old"
        )
        assert job.main() == 0
        assert (
            job.pre_out / "a.feather"
        ).read_bytes() == b"p" * 7
        status = job.status.read_text()
        assert "STAGE soften DONE rc=0" in status
        assert "STAGE preprocess DONE rc=0" in status

    def test_train_hcpe_required_without_reuse(self) -> None:
        with pytest.raises(SystemExit):
            job_mod._parse_args(["--val-data", "val"])


def test_tree_size_exclude(tmp_path: Path) -> None:
    (tmp_path / "a.feather").write_bytes(b"x" * 5)
    (tmp_path / "b.log").write_bytes(b"y" * 3)
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "driver.log").write_bytes(b"z" * 2)
    assert job_mod._tree_size(tmp_path) == (3, 10)
    assert job_mod._tree_size(
        tmp_path, ("*.feather", "driver.log")
    ) == (1, 3)
