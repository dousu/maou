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
checkpoint までは残る．新しい VM で前処理をやり直さずに続きから走らせるには，
Drive の ``preprocess/preprocess_<tag>`` を VM ローカルの同じ相対パスへ
``rsync -a`` で戻し，``<work>/arm0_<tag>/STATUS`` に
``STAGE preprocess DONE rc=0`` の行を書いてから同じコマンドで再投入する
(``soften`` の出力は 1 分で作り直せるので退避しない)．

``--unassign-on-done`` を付けると，全段階が rc=0 で終わったあと
``--unassign-grace-min`` だけ待ってから Colab の runtime 管理サービスへ
VM の unassign を要求する (``google.colab.runtime.unassign()`` と同じ
``POST http://$TBE_RUNTIME_ADDR/unassign``．kernel の環境変数を継承する
nohup 子プロセスからでも通る)．猶予の間に ``<work>/arm0_<tag>/KEEP_VM`` を
作れば取りやめる．失敗 (rc≠0) のときは段階ログを見られるよう VM を残す．

## 使い方

VM 上 (``docs/colab-cli-notes.md`` §4 の nohup 方式)::

    nohup python /content/colab_arm0_job.py \\
        --train-hcpe hcpe/hcpe_20260805/train \\
        --val-data preprocess/preprocess_20260901/val \\
        --tag 20260921 \\
        -- --model-architecture vit ... (基準 run と同一のハイパラ) \\
        > /content/job.log 2>&1 &

``--train-hcpe`` / ``--val-data`` は Drive (``--drive-root``) と VM ローカル
(``--work-root``) で共通の相対パス (§7.4 のレイアウトを両側で同じに切る)．
出力は ``preprocess/preprocess_<tag>`` / ``maou_test/models/models_<tag>`` /
``maou_test/logs/logs_<tag>`` に置く．

手元で配線を確かめる (Drive 無し / 小データ / 段階を選ぶ)::

    uv run python scripts/colab_arm0_job.py --no-drive \\
        --work-root /tmp/arm0 --train-hcpe hcpe/tiny --val-data pre/tiny_val \\
        --stages soften,preprocess --tag test -- --epoch 1

終わりに番兵 ``JOB_DONE exit=<n>`` を出す (``exec`` の exit code は使えない)．
"""

from __future__ import annotations

import argparse
import datetime as dt
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


def _now() -> str:
    return dt.datetime.now(JST).isoformat(timespec="seconds")


def log(msg: str) -> None:
    """時刻つきで 1 行出す (``colab exec`` のタイムアウト延命のため flush)．"""
    print(f"[{_now()}] {msg}", flush=True)


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
        required=True,
        help="学習側 HCPE ディレクトリ (work-root / drive-root からの相対)",
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
            "全段階が rc=0 で終わったら VM を unassign する "
            "(Colab の runtime 管理サービスへ POST)"
        ),
    )
    ap.add_argument(
        "--unassign-grace-min",
        type=float,
        default=60.0,
        help=(
            "unassign までの猶予 (分)．この間に成果物を colab download できる．"
            "<work>/arm0_<tag>/KEEP_VM があれば取りやめる"
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
    return ns


class Job:
    """段階ごとの実行と ``STATUS`` の読み書き．"""

    def __init__(self, ns: argparse.Namespace) -> None:
        self.ns = ns
        self.work: Path = ns.work_root
        self.drive: Path | None = (
            None if ns.no_drive else ns.drive_root
        )
        self.train_hcpe = self.work / ns.train_hcpe
        self.val_data = self.work / ns.val_data
        self.job_dir = self.work / f"arm0_{ns.tag}"
        self.status = self.job_dir / "STATUS"
        self.soft_out = self.job_dir / "arm0_debias.feather"
        self.pre_out = (
            self.work / "preprocess" / f"preprocess_{ns.tag}"
        )
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
        self.job_dir.mkdir(parents=True, exist_ok=True)

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

    def rsync(self, src: Path, dst: Path) -> int:
        """``src/`` を ``dst/`` へ同期し，両側のサイズと件数を突き合わせる．"""
        if not src.exists():
            log(f"  skip rsync: {src} does not exist")
            return 0
        dst.mkdir(parents=True, exist_ok=True)
        rc = subprocess.call(
            ["rsync", "-a", f"{src}/", f"{dst}/"]
        )
        s_n, s_b = _tree_size(src)
        d_n, d_b = _tree_size(dst)
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
        rc = 0
        for rel in (self.ns.train_hcpe, self.ns.val_data):
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
        return (
            self.drive
            / "preprocess"
            / f"preprocess_{self.ns.tag}"
        )

    def sync(self) -> int:
        if self.drive is None:
            log("  --no-drive: nothing to sync")
            return 0
        return self._sync_outputs(include_preprocess=True)

    # --- driver ---------------------------------------------------------
    def main(self) -> int:
        log(
            f"arm0 job tag={self.ns.tag} work={self.work} drive={self.drive}"
        )
        log(f"  train_hcpe={self.train_hcpe}")
        log(f"  val_data={self.val_data}")
        log(
            f"  outputs: {self.pre_out} / {self.model_dir} / {self.log_dir}"
        )
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


def _tree_size(root: Path) -> tuple[int, int]:
    """配下の (ファイル数, 合計バイト) を返す．"""
    files = [p for p in root.rglob("*") if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files)


def main() -> int:
    ns = _parse_args(sys.argv[1:])
    if shutil.which("rsync") is None and not ns.no_drive:
        log("rsync not found")
        return 2
    job = Job(ns)
    rc = job.main()
    print(f"JOB_DONE exit={rc}", flush=True)
    if ns.unassign_on_done and rc == 0 and not ns.dry_run:
        _unassign_runtime(
            ns.unassign_grace_min, job.job_dir / "KEEP_VM"
        )
    return rc


if __name__ == "__main__":
    sys.exit(main())
