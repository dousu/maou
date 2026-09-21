#!/usr/bin/env python
"""Arm 0 (対照実験) を Colab VM 上で最後まで回す駆動スクリプト．

## 何をするか

`docs/design/training-quality/` の Arm 0 — 「探索値の効果は**軟化**だけで
再現できるか」を判定する対照 — を，1 本の nohup ジョブとして VM 上で
完走させる．段階は 4 つで，各段階の完了を ``STATUS`` に記録するので，
途中で落ちても同じコマンドで再投入すれば済んだ段階を飛ばして続きから走る．

    fetch       Drive から入力 (HCPE / 検証用前処理済) を VM ローカルへコピー
    soften      scripts/soften_result_value.py で (id, searchWinRate) を作る
    preprocess  maou pre-process --search-value-path で置換した学習データを作る
    learn       maou learn-model (ハイパラは ``--`` 以降をそのまま渡す)
    sync        前処理済 / モデル / ログを Drive の所定フォルダへ退避

``preprocess`` の出力は段階完了直後に Drive へ退避し (§7.3 の「1 ステージ
完了時にコピー」)，``learn`` の間は ``--sync-interval-min`` ごとにモデルと
ログを Drive へ退避し続けるので，VM を失っても前処理済データと最後の
checkpoint までは残る．新しい VM で前処理をやり直さずに learn から走らせる
には ``--train-preprocessed preprocess/preprocess_<前の tag>`` を渡す
(``fetch`` が Drive から戻し，``soften`` / ``preprocess`` は飛ばす．
``--tag`` は新しくする — モデル / ログのフォルダは learn-model 1 回ごとに分ける)．

ジョブの記録 (``STATUS``，段階ログ ``*.log``，driver 自身のログ) は
``<work>/maou_test/jobs/arm0_<tag>/`` に置き，終了時 (成否を問わず) に
Drive の同じ相対パスへ退避する (``soften`` の出力 ``*.feather`` は 1 分で
作り直せるので除く)．**失敗 (rc≠0) のとき**はさらに ``diag/`` に VM の状態
(``dmesg`` の末尾 / ``free`` / ``df`` / ``nvidia-smi`` / ``ps``) を採り，
モデルとログの最終退避も行う — 例: 2026-09-21 の learn は memory cgroup の
OOM で DataLoader worker が殺されて落ちたが，その証拠は VM の ``dmesg`` に
しか無く，periodic sync の直前に保存された epoch 1 のモデルも VM に
取り残されていた．

``--unassign-on-done`` を付けると，ジョブが終わったあと (成否を問わず)
``--unassign-grace-min`` だけ待ってから Colab の runtime 管理サービスへ
VM の unassign を要求する (``google.colab.runtime.unassign()`` と同じ
``POST http://$TBE_RUNTIME_ADDR/unassign``．kernel の環境変数を継承する
nohup 子プロセスからでも通る)．猶予の間に ``<work>/maou_test/jobs/arm0_<tag>/KEEP_VM``
を作れば取りやめる．失敗時は上記の退避が Drive で OK と確認できたときだけ
unassign し，退避できなかった (Drive 無し / rsync 不一致) ときは
``UNASSIGN_SKIPPED`` を出して VM を残す．VM が残っているときに退避だけ
やり直すには ``--evacuate-only`` (段階は走らせない)．

## 使い方

VM 上 (``docs/colab-cli-notes.md`` §4 の nohup 方式)::

    nohup python /content/colab_arm0_job.py \\
        --train-hcpe hcpe/hcpe_20260805/train \\
        --val-data preprocess/preprocess_20260901/val \\
        --tag 20260921 \\
        -- --model-architecture vit ... (基準 run と同一のハイパラ) \\
        > /content/job.log 2>&1 &

``--train-hcpe`` / ``--val-data`` / ``--train-preprocessed`` は Drive
(``--drive-root``) と VM ローカル (``--work-root``) で共通の相対パス (§7.4 の
レイアウトを両側で同じに切る)．出力は ``preprocess/preprocess_<tag>`` /
``maou_test/models/models_<tag>`` / ``maou_test/logs/logs_<tag>`` /
``maou_test/jobs/arm0_<tag>`` に置く．

手元で配線を確かめる (Drive 無し / 小データ / 段階を選ぶ)::

    uv run python scripts/colab_arm0_job.py --no-drive \\
        --work-root /tmp/arm0 --train-hcpe hcpe/tiny --val-data pre/tiny_val \\
        --stages soften,preprocess --tag test -- --epoch 1

終わりに番兵 ``JOB_DONE exit=<n>`` を出す (``exec`` の exit code は使えない)．
"""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

STAGES = ("fetch", "soften", "preprocess", "learn", "sync")
DEFAULT_SOFTEN_ARGS = (
    "--min-ply 60 --max-ply 100 --mode empirical --debias"
)
JST = dt.timezone(dt.timedelta(hours=9))
# driver 自身のログの控え (job_dir/driver.log)．退避のとき diag/ へ snapshot
# (別名) を置くので，本体は rsync の件数照合から除く (書き込み中で必ず不一致になる)
DRIVER_LOG = "driver.log"
DRIVER_LOG_SNAPSHOT = "driver_log.txt"
JOB_DIR_SYNC_EXCLUDE = ("*.feather", DRIVER_LOG)
# 失敗時に VM の状態を残すコマンド (diag/<name>.txt)．無いコマンドは
# その旨を書くだけで失敗にしない (手元の --no-drive テストでも通す)
DIAG_COMMANDS: tuple[tuple[str, str], ...] = (
    ("uptime", "uptime"),
    ("dmesg", "dmesg -T 2>&1 | tail -n 300"),
    ("free", "free -m"),
    ("meminfo", "cat /proc/meminfo"),
    (
        "cgroup_memory",
        "cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.peak /sys/fs/cgroup/memory.events 2>&1",
    ),
    ("df", "df -h"),
    ("nvidia-smi", "nvidia-smi"),
    ("ps", "ps aux --sort=-rss | head -n 40"),
    (
        "python",
        f"{shlex.quote(sys.executable)} -m pip list 2>/dev/null | grep -iE '^(maou|torch|onnx|polars|numpy) ' ; {shlex.quote(sys.executable)} -V",
    ),
)
_LOG_COPY: Path | None = None


def _now() -> str:
    return dt.datetime.now(JST).isoformat(timespec="seconds")


def log(msg: str) -> None:
    """時刻つきで 1 行出す (``colab exec`` のタイムアウト延命のため flush)．

    ``Job`` が作られたあとは ``job_dir/driver.log`` にも同じ行を残す
    (nohup の stdout は job_dir の外なので，退避に含めるための控え)．
    """
    line = f"[{_now()}] {msg}"
    print(line, flush=True)
    if _LOG_COPY is not None:
        try:
            with _LOG_COPY.open("a") as f:
                f.write(line + "\n")
        except OSError:
            pass


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """コマンドライン引数を解釈する．``--`` 以降は learn-model へ素通しする．"""
    if "--" in argv:
        i = argv.index("--")
        argv, learn_args = argv[:i], argv[i + 1 :]
    else:
        learn_args = []
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--train-hcpe",
        default="",
        help=(
            "学習側 HCPE ディレクトリ (work-root / drive-root からの相対)．"
            "--train-preprocessed を使わないなら必須"
        ),
    )
    ap.add_argument(
        "--train-preprocessed",
        default="",
        help=(
            "前の run が作った学習側の前処理済ディレクトリ (相対)．"
            "指定すると fetch で Drive から戻し，soften / preprocess を飛ばして"
            " learn からやり直す (新しい VM での再投入用)"
        ),
    )
    ap.add_argument(
        "--val-data",
        required=True,
        help="検証側の前処理済ディレクトリ (相対．探索値なしで前処理したもの)",
    )
    ap.add_argument(
        "--work-root", type=Path, default=Path("/content/shogi")
    )
    ap.add_argument(
        "--drive-root",
        type=Path,
        default=Path("/content/drive/MyDrive/shogi"),
    )
    ap.add_argument(
        "--no-drive",
        action="store_true",
        help="Drive を使わない (fetch / sync を飛ばし，入力は work-root にある前提)",
    )
    ap.add_argument(
        "--tag",
        default=dt.datetime.now(JST).strftime("%Y%m%d"),
        help="出力サブフォルダの日付タグ (既定: JST の今日)",
    )
    ap.add_argument(
        "--stages",
        default=",".join(STAGES),
        help=f"実行する段階 (カンマ区切り．既定: {','.join(STAGES)})",
    )
    ap.add_argument(
        "--soften-args",
        default=DEFAULT_SOFTEN_ARGS,
        help=f"soften_result_value.py へ渡す引数 (既定: {DEFAULT_SOFTEN_ARGS!r})",
    )
    ap.add_argument(
        "--preprocess-args",
        default="",
        help="pre-process へ追加で渡す引数 (基準 run と揃える)",
    )
    ap.add_argument(
        "--sync-interval-min",
        type=float,
        default=30.0,
        help="learn 中に models / logs を Drive へ退避する間隔 (分．0 で無効)",
    )
    ap.add_argument(
        "--maou",
        default="",
        help="maou CLI のパス (既定: このインタプリタで python -m maou)",
    )
    ap.add_argument(
        "--unassign-on-done",
        action="store_true",
        help=(
            "ジョブが終わったら (成否を問わず) VM を unassign する "
            "(Colab の runtime 管理サービスへ POST)．失敗時は診断情報の"
            " Drive 退避が OK のときだけ"
        ),
    )
    ap.add_argument(
        "--unassign-grace-min",
        type=float,
        default=60.0,
        help=(
            "unassign までの猶予 (分)．この間に成果物を colab download できる．"
            "<work>/maou_test/jobs/arm0_<tag>/KEEP_VM があれば取りやめる"
        ),
    )
    ap.add_argument(
        "--evacuate-only",
        action="store_true",
        help=(
            "段階を走らせず，診断情報を集めて job dir / models / logs を"
            " Drive へ退避するだけ (失敗後に VM が残ったときの手動退避用)"
        ),
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="コマンドを表示するだけで実行しない",
    )
    ns = ap.parse_args(argv)
    ns.learn_args = learn_args
    ns.stage_set = [
        s.strip() for s in ns.stages.split(",") if s.strip()
    ]
    for s in ns.stage_set:
        if s not in STAGES:
            ap.error(
                f"unknown stage {s!r} (choose from {', '.join(STAGES)})"
            )
    if not ns.train_hcpe and not ns.train_preprocessed:
        ap.error(
            "--train-hcpe is required unless --train-preprocessed is given"
        )
    return ns


class Job:
    """段階ごとの実行と ``STATUS`` の読み書き．"""

    def __init__(self, ns: argparse.Namespace) -> None:
        global _LOG_COPY
        self.ns = ns
        self.work: Path = ns.work_root
        self.drive: Path | None = (
            None if ns.no_drive else ns.drive_root
        )
        self.train_hcpe = self.work / ns.train_hcpe
        self.val_data = self.work / ns.val_data
        # 前処理済を持ち込む run は soften / preprocess を持たない
        self.reuse_pre: bool = bool(ns.train_preprocessed)
        self.job_rel = (
            Path("maou_test") / "jobs" / f"arm0_{ns.tag}"
        )
        self.job_dir = self.work / self.job_rel
        self.status = self.job_dir / "STATUS"
        self.diag_dir = self.job_dir / "diag"
        self.soft_out = self.job_dir / "arm0_debias.feather"
        self.pre_rel = (
            Path(ns.train_preprocessed)
            if self.reuse_pre
            else Path("preprocess") / f"preprocess_{ns.tag}"
        )
        self.pre_out = self.work / self.pre_rel
        self.model_dir = (
            self.work
            / "maou_test"
            / "models"
            / f"models_{ns.tag}"
        )
        self.log_dir = (
            self.work / "maou_test" / "logs" / f"logs_{ns.tag}"
        )
        self.maou = (
            shlex.split(ns.maou)
            if ns.maou
            else [sys.executable, "-m", "maou"]
        )
        # 失敗時の退避が Drive で確認できたか (main() が unassign の可否に使う)
        self.evacuated: bool = False
        self.job_dir.mkdir(parents=True, exist_ok=True)
        _LOG_COPY = self.job_dir / DRIVER_LOG

    # --- STATUS ---------------------------------------------------------
    def done(self, stage: str) -> bool:
        if not self.status.exists():
            return False
        return any(
            line.startswith(f"STAGE {stage} DONE rc=0")
            for line in self.status.read_text().splitlines()
        )

    def mark(self, stage: str, rc: int) -> None:
        with self.status.open("a") as f:
            f.write(
                f"STAGE {stage} DONE rc={rc} finished={_now()}\n"
            )

    # --- helpers --------------------------------------------------------
    def run(self, cmd: list[str], logname: str) -> int:
        """コマンドを実行し，出力を ``<job_dir>/<logname>.log`` にも残す．"""
        log(f"$ {shlex.join(cmd)}")
        if self.ns.dry_run:
            return 0
        logfile = self.job_dir / f"{logname}.log"
        t0 = time.monotonic()
        with logfile.open("a") as f:
            f.write(f"### {_now()} $ {shlex.join(cmd)}\n")
            f.flush()
            rc = subprocess.call(
                cmd, stdout=f, stderr=subprocess.STDOUT
            )
        log(
            f"  rc={rc} elapsed={time.monotonic() - t0:.0f}s log={logfile}"
        )
        return rc

    def rsync(
        self,
        src: Path,
        dst: Path,
        exclude: tuple[str, ...] = (),
    ) -> int:
        """``src/`` を ``dst/`` へ同期し，両側のサイズと件数を突き合わせる．

        ``exclude`` (rsync の ``--exclude`` と同じ glob) は転送からも
        照合からも外す．
        """
        if not src.exists():
            log(f"  skip rsync: {src} does not exist")
            return 0
        dst.mkdir(parents=True, exist_ok=True)
        opts = [f"--exclude={pat}" for pat in exclude]
        rc = subprocess.call(
            ["rsync", "-a", *opts, f"{src}/", f"{dst}/"]
        )
        s_n, s_b = _tree_size(src, exclude)
        d_n, d_b = _tree_size(dst, exclude)
        ok = "OK" if (s_n, s_b) == (d_n, d_b) else "MISMATCH"
        log(
            f"  rsync {src} -> {dst}: rc={rc} src={s_n} files/{s_b} B "
            f"dst={d_n} files/{d_b} B {ok}"
        )
        return 0 if (rc == 0 and ok == "OK") else 1

    # --- stages ---------------------------------------------------------
    def fetch(self) -> int:
        if self.drive is None:
            log(
                "  --no-drive: inputs are expected under work-root"
            )
            return 0
        rels = [self.ns.val_data]
        if self.reuse_pre:
            rels.insert(0, str(self.pre_rel))
        else:
            rels.insert(0, self.ns.train_hcpe)
        rc = 0
        for rel in rels:
            src, dst = self.drive / rel, self.work / rel
            if not src.exists():
                log(f"  input missing on Drive: {src}")
                return 1
            if self.ns.dry_run:
                log(f"$ rsync -a {src}/ {dst}/")
                continue
            rc |= self.rsync(src, dst)
        return rc

    def soften(self) -> int:
        if self.reuse_pre:
            log(
                "  --train-preprocessed given: nothing to soften"
            )
            return 0
        script = (
            Path(__file__)
            .resolve()
            .with_name("soften_result_value.py")
        )
        if not script.exists():
            log(
                f"  soften_result_value.py not found next to this script: {script}"
            )
            return 1
        cmd = [
            sys.executable,
            str(script),
            str(self.train_hcpe),
            str(self.soft_out),
            *shlex.split(self.ns.soften_args),
        ]
        return self.run(cmd, "soften")

    def preprocess(self) -> int:
        if self.reuse_pre:
            log(
                f"  --train-preprocessed given: reuse {self.pre_out}"
            )
            return (
                0
                if (self.pre_out.exists() or self.ns.dry_run)
                else 1
            )
        cmd = [
            *self.maou,
            "pre-process",
            "--input-path",
            str(self.train_hcpe),
            "--output-dir",
            str(self.pre_out),
            "--search-value-path",
            str(self.soft_out),
            *shlex.split(self.ns.preprocess_args),
        ]
        rc = self.run(cmd, "preprocess")
        if rc != 0 or self.drive is None or self.ns.dry_run:
            return rc
        # 1 時間級の出力なので learn に入る前に退避する (VM を失っても残す)
        log("  sync preprocess output to Drive before learn")
        return self.rsync(
            self.pre_out, self._drive_preprocess_dir()
        )

    def learn(self) -> int:
        extra = list(self.ns.learn_args)
        if "--early-stopping-metric" not in extra:
            extra += ["--early-stopping-metric", "value"]
        for d in (self.model_dir, self.log_dir):
            d.mkdir(parents=True, exist_ok=True)
        cmd = [
            *self.maou,
            "learn-model",
            "--stage3-data-path",
            str(self.pre_out),
            "--stage3-validation-data-path",
            str(self.val_data),
            "--model-dir",
            str(self.model_dir),
            "--log-dir",
            str(self.log_dir),
            *extra,
        ]
        stop = threading.Event()
        syncer = None
        if (
            self.drive is not None
            and self.ns.sync_interval_min > 0
            and not self.ns.dry_run
        ):
            syncer = threading.Thread(
                target=self._periodic_sync,
                args=(stop,),
                daemon=True,
            )
            syncer.start()
        try:
            return self.run(cmd, "learn")
        finally:
            stop.set()
            if syncer is not None:
                syncer.join(timeout=600)

    def _periodic_sync(self, stop: threading.Event) -> None:
        interval = self.ns.sync_interval_min * 60
        while not stop.wait(interval):
            log("  periodic sync of models / logs to Drive")
            self._sync_outputs(include_preprocess=False)

    def _sync_outputs(self, include_preprocess: bool) -> int:
        assert self.drive is not None
        pairs = [
            (
                self.model_dir,
                self.drive
                / "maou_test"
                / "models"
                / f"models_{self.ns.tag}",
            ),
            (
                self.log_dir,
                self.drive
                / "maou_test"
                / "logs"
                / f"logs_{self.ns.tag}",
            ),
        ]
        if include_preprocess:
            pairs.insert(
                0, (self.pre_out, self._drive_preprocess_dir())
            )
        rc = 0
        for src, dst in pairs:
            if self.ns.dry_run:
                log(f"$ rsync -a {src}/ {dst}/")
                continue
            rc |= self.rsync(src, dst)
        return rc

    def _drive_preprocess_dir(self) -> Path:
        assert self.drive is not None
        return self.drive / self.pre_rel

    def _sync_job_dir(self) -> int:
        """job dir (STATUS / 段階ログ / diag) を Drive の同じ相対パスへ退避する．

        driver.log は書き込み中なので snapshot (``diag/driver_log.txt``) を
        置いてから，本体は照合から外して同期する．
        """
        assert self.drive is not None
        self.diag_dir.mkdir(parents=True, exist_ok=True)
        live = self.job_dir / DRIVER_LOG
        if live.exists():
            shutil.copyfile(
                live, self.diag_dir / DRIVER_LOG_SNAPSHOT
            )
        return self.rsync(
            self.job_dir,
            self.drive / self.job_rel,
            exclude=JOB_DIR_SYNC_EXCLUDE,
        )

    def collect_diag(self) -> None:
        """VM の状態を ``diag/`` に書く (OOM なら dmesg にしか証拠が無い)．"""
        self.diag_dir.mkdir(parents=True, exist_ok=True)
        for name, cmd in DIAG_COMMANDS:
            out = self.diag_dir / f"{name}.txt"
            try:
                r = subprocess.run(
                    cmd,
                    shell=True,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                text = f"$ {cmd}\n(rc={r.returncode})\n{r.stdout}{r.stderr}"
            except (OSError, subprocess.SubprocessError) as e:
                text = f"$ {cmd}\n({type(e).__name__}: {e})\n"
            out.write_text(text)
        log(f"  diag written to {self.diag_dir}")

    def evacuate(self, reason: str) -> int:
        """診断情報を集め，モデル / ログ / job dir を Drive へ退避する．

        rc=0 のときだけ ``self.evacuated`` を立てる (unassign の条件)．
        """
        log(f"  evacuate ({reason})")
        if self.ns.dry_run:
            return 0
        self.collect_diag()
        if self.drive is None:
            log(
                "  --no-drive: nothing to evacuate; VM state stays local"
            )
            return 1
        rc = self._sync_outputs(include_preprocess=False)
        rc |= self._sync_job_dir()
        self.evacuated = rc == 0
        log(
            f"  EVACUATED={'OK' if self.evacuated else 'FAILED'}"
        )
        return rc

    def sync(self) -> int:
        if self.drive is None:
            log("  --no-drive: nothing to sync")
            return 0
        rc = self._sync_outputs(
            include_preprocess=not self.reuse_pre
        )
        if self.ns.dry_run:
            return rc
        return rc | self._sync_job_dir()

    # --- driver ---------------------------------------------------------
    def main(self) -> int:
        log(
            f"arm0 job tag={self.ns.tag} work={self.work} drive={self.drive}"
        )
        if self.reuse_pre:
            log(f"  train_preprocessed={self.pre_out}")
        else:
            log(f"  train_hcpe={self.train_hcpe}")
        log(f"  val_data={self.val_data}")
        log(
            f"  outputs: {self.pre_out} / {self.model_dir} / {self.log_dir}"
        )
        log(f"  job dir: {self.job_dir}")
        log(
            f"  learn-model args: {shlex.join(self.ns.learn_args)}"
        )
        if (
            self.drive is not None
            and not self.ns.dry_run
            and not self.drive.exists()
        ):
            log(
                f"  Drive root not found: {self.drive} (mount first)"
            )
            return 2
        if self.ns.evacuate_only:
            return self.evacuate("--evacuate-only")
        for stage in STAGES:
            if stage not in self.ns.stage_set:
                log(f"== {stage}: not selected, skip")
                continue
            if self.done(stage):
                log(f"== {stage}: already DONE, skip")
                continue
            log(f"== {stage}: start")
            rc = getattr(self, stage)()
            if not self.ns.dry_run:
                self.mark(stage, rc)
            if rc != 0:
                log(f"== {stage}: FAILED rc={rc}")
                self.evacuate(f"{stage} failed rc={rc}")
                return rc
            log(f"== {stage}: DONE")
        return 0


def _unassign_runtime(
    grace_min: float, hold_file: Path
) -> None:
    """猶予のあと Colab の runtime 管理サービスへ VM の unassign を要求する．

    ``google.colab.runtime.unassign()`` の中身 (``POST /unassign``) を
    そのまま呼ぶ．frontend への JS 送信は kernel 外では不要なので省く．
    """
    addr = os.environ.get("TBE_RUNTIME_ADDR")
    if not addr:
        log(
            "  TBE_RUNTIME_ADDR is not set; skip unassign (not on Colab?)"
        )
        return
    deadline = time.monotonic() + grace_min * 60
    at = dt.datetime.now(JST) + dt.timedelta(minutes=grace_min)
    log(
        f"  UNASSIGN_AT={at.isoformat(timespec='seconds')} "
        f"(touch {hold_file} to keep the VM)"
    )
    while time.monotonic() < deadline:
        if hold_file.exists():
            log(f"  {hold_file} exists; VM kept")
            return
        time.sleep(30)
    if hold_file.exists():
        log(f"  {hold_file} exists; VM kept")
        return
    req = urllib.request.Request(
        f"http://{addr}/unassign", data=b"", method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            log(f"UNASSIGN_REQUESTED status={resp.status}")
    except urllib.error.HTTPError as e:
        log(f"UNASSIGN_FAILED status={e.code}")
    except OSError as e:
        log(f"UNASSIGN_FAILED {type(e).__name__}: {e}")


def _tree_size(
    root: Path, exclude: tuple[str, ...] = ()
) -> tuple[int, int]:
    """配下の (ファイル数, 合計バイト) を返す (``exclude`` の glob に合う名前は除く)．"""
    files = [
        p
        for p in root.rglob("*")
        if p.is_file()
        and not any(
            fnmatch.fnmatch(p.name, pat) for pat in exclude
        )
    ]
    return len(files), sum(p.stat().st_size for p in files)


def main() -> int:
    ns = _parse_args(sys.argv[1:])
    if shutil.which("rsync") is None and not ns.no_drive:
        log("rsync not found")
        return 2
    job = Job(ns)
    rc = job.main()
    log(f"JOB_DONE exit={rc}")
    if ns.unassign_on_done and not ns.dry_run:
        if rc == 0 or job.evacuated:
            _unassign_runtime(
                ns.unassign_grace_min, job.job_dir / "KEEP_VM"
            )
        else:
            # 退避できていない診断情報は VM にしか無いので残す
            log(
                "UNASSIGN_SKIPPED reason=evacuation not confirmed on Drive;"
                " VM kept for diagnosis"
            )
    return rc


if __name__ == "__main__":
    sys.exit(main())
