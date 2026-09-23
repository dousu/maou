#!/usr/bin/env python3
"""探索値の蓄積 (`maou utility search-values`) を Colab で回す 1 ファイル driver．

## 人間の操作

### セル A — 置く (最初の 1 回だけ)

Drive 書き込みはマウント経由でのみ行う規約 (docs/colab-cli-notes.md §7.1)．

```python
from google.colab import drive; drive.mount("/content/drive")
!mkdir -p /content/drive/MyDrive/shogi/scripts
!curl -sL https://raw.githubusercontent.com/dousu/maou/main/scripts/colab_search_values_job.py \
   -o /content/drive/MyDrive/shogi/scripts/colab_search_values_job.py
```

### セル B — 点検 (毎セッション，投入の前に)

探索は起こさない．teacher の sha256・HCPE・既存シャード・来歴・GPU を見て
PASS / FAIL を出し，**wheel (`maou[tensorrt-infer]`) と GPU provider の
`ldconfig` まで済ませる**ので，セル C の立ち上がりも速くなる．
**ALL PASS でなければ投入しない** (20 時間を捨てないため)．

```python
from google.colab import drive; drive.mount("/content/drive")
%run /content/drive/MyDrive/shogi/scripts/colab_search_values_job.py --check
```

### セル C — 投入 (毎セッション，これで走り出す)

```python
from google.colab import drive; drive.mount("/content/drive")
%run /content/drive/MyDrive/shogi/scripts/colab_search_values_job.py
```

セル C はそのまま keep-alive 兼モニタとして走り続ける．中身を書き換える必要は
無く，**毎セッション B → C を貼るだけ**で続きから貯まる．

## 何が起きるか

1. wheel (`maou[tensorrt-infer]`) と GPU provider を入れ，Drive から
   teacher / HCPE / 既存シャードをローカルへ取る
2. 探索を **nohup の別プロセス**で起こす (このセルを止めてもジョブは死なない)
3. このセル自身が keep-alive 兼モニタになり，5 分ごとに進捗を表示する
   — Colab のアイドル切断はセルが走っている間は起きない
4. ワーカーが `SESSION_HOURS` 経過で探索を畳み，Drive へ最終同期してから
   VM を unassign する

**冪等**なので、切断されたら同じ 3 行をもう一度貼ればよい．ワーカーが生きて
いれば再接続してモニタするだけで，二重起動はしない．

## 落ちないための設計

- 探索は `--resume` なので，どこで切れても既に計算した局面はやり直さない
- `--max-positions` でセッションぶんに区切る．残りからの標本抽出なので
  セッションをまたぐと全体を覆う
- **teacher が前セッションと違うと `search-values` 側が即エラーで止まる**
  (`provenance.json`)．12 回以上の再投入で教師が混ざる事故を防ぐ
"""

from __future__ import annotations

import ast
import datetime as dt
import glob
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# =====================================================================
# CONFIG — ここだけが設定．通常は触らない
# =====================================================================
DRIVE = Path("/content/drive/MyDrive/shogi")
WORK = Path("/content/shogi")

#: teacher (探索値を出すモデル)．DRIVE からの相対パス．
#: **これを変えると既存の蓄積には足せない** (provenance が弾く)．
#: 既存の ply>=120 の蓄積 (`search_values/search_values_20260806`) と同じ
#: teacher なので，前処理の段階で `--search-value-path` を 2 つ渡せば
#: 両方を 1 つの教師信号としてまとめられる (探索のやり直しは要らない)．
#: **teacher を変えるなら OUT_REL も新しい日付にする** (provenance が弾く)．
TEACHER_REL = "maou_test/models/model_20260805_144450_vit-19.8m_16_fp16.onnx"
#: teacher の sha256．**初回実行でも取り違えを検出する**ための事前登録．
#: (2 回目以降は出力側の `provenance.json` が同じことを保証する)
TEACHER_SHA256 = "8949d72912f251dc5537a3a5232ac527f9fbcb6f23aae486269036e0207d448d"
#: 探索対象の HCPE ディレクトリ．DRIVE からの相対パス．
#: **必ず train だけを指す**．`hcpe_20260805/` には `val/` も入っており，
#: 検証側の教師を変えることは許されない (docs 同ページ § 検証データには
#: 適用しない)．val を混ぜると予算を無駄にするうえ，前処理で取り違えると
#: 較正測定の前提が壊れる．
HCPE_REL = "hcpe/hcpe_20260805/train"
#: 探索値の出力先 (シャードのディレクトリ)．DRIVE からの相対パス．
#: 命名は `<種別>_<YYYYMMDD>` (docs/colab-cli-notes.md §7.4)．種別は親フォルダ名で，
#: **帯や条件は名前に詰め込まない** — 由来は worklog に書く (同 §9)．
#: 既存の ply>=120 (`search_values_20260806`) とは別ディレクトリにする．
#: 前処理でどちらを渡すかを後から選べるようにするため．
#:
#: **この値はセッションをまたいで固定する．** 日付を実行時に生成すると
#: セッションごとに別のディレクトリができ，`--resume` が前回のシャードを
#: 見つけられず 13 回ぶんの探索がすべてやり直しになる．
OUT_REL = "search_values/search_values_20260923"
#: ジョブの記録 (STATUS / ログ) の Drive 退避先．docs/colab-cli-notes.md §7.4 の
#: `maou_test/jobs/` に置く．**unassign すると VM 上の記録は消える**ので，
#: 手放す前に必ずここへ逃がす (同 §10)．
JOB_DRIVE_REL = "maou_test/jobs/search_values_20260923"

#: 帯は [MIN_PLY, MAX_PLY)．Arm 1 は ply 60-99．
MIN_PLY = 60
MAX_PLY = 100

#: 1 セッションで探索する局面数．実測 45,800 局面/時 (L4) から
#: 20 時間ぶんに取ってある (24h 壁に対して同期と unassign の余裕を残す)．
POSITIONS_PER_SESSION = 900_000
#: これを過ぎたら探索を畳んで最終同期に入る．
SESSION_HOURS = 20.0
#: 最終同期のあと VM を手放すまでの猶予 (分)．
UNASSIGN_GRACE_MIN = 30.0

#: 探索の設定．決着済みなので変えない (docs/performance.md)．
PLAYOUTS = 800
BATCH_SIZE = 64
THREADS = 1
FLUSH_INTERVAL = 2000
SHARD_ROWS = 5_000_000

#: Drive へシャードを送る間隔 (分)．
SYNC_EVERY_MIN = 30.0

REPO = "dousu/maou"
JST = dt.timezone(dt.timedelta(hours=9))

JOB_DIR = WORK / "jobs" / "search_values"
STATUS = JOB_DIR / "STATUS"
JOB_LOG = JOB_DIR / "job.log"
SEARCH_LOG = JOB_DIR / "search.log"
PIDFILE = JOB_DIR / "worker.pid"
HOLD = JOB_DIR / "KEEP_VM"
DIAG_DIR = JOB_DIR / "diag"

#: 失敗時に採る VM の状態 (docs/colab-cli-notes.md §10 の MUST)．
#: **OOM の証拠は VM の dmesg にしか無く，unassign すると消える．**
DIAG_COMMANDS: tuple[tuple[str, str], ...] = (
    ("uptime", "uptime"),
    ("dmesg", "dmesg -T 2>&1 | tail -n 300"),
    ("free", "free -m"),
    ("meminfo", "cat /proc/meminfo"),
    (
        "cgroup_memory",
        "cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.peak "
        "/sys/fs/cgroup/memory.events 2>&1",
    ),
    ("df", "df -h"),
    ("nvidia-smi", "nvidia-smi"),
    ("ps", "ps aux --sort=-rss | head -n 40"),
    (
        "python",
        f"{shlex.quote(sys.executable)} -m pip list 2>/dev/null "
        "| grep -iE '^(maou|torch|onnx|polars|numpy) ' ; "
        f"{shlex.quote(sys.executable)} -V",
    ),
)


# =====================================================================
# 共通
# =====================================================================
def now() -> str:
    return dt.datetime.now(JST).isoformat(timespec="seconds")


def log(msg: str) -> None:
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    with JOB_LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def set_status(**kw: object) -> None:
    """人間とモニタが読む 1 行 JSON を書く．"""
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    current: dict[str, object] = {}
    if STATUS.exists():
        try:
            current = json.loads(STATUS.read_text())
        except ValueError:
            current = {}
    current.update(kw)
    current["updated"] = now()
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(current, ensure_ascii=False, indent=2)
    )
    os.replace(tmp, STATUS)


def run(cmd: list[str], logfile: Path) -> int:
    """コマンドを実行し，**本当の exit code** を返す (`| tail` は使わない)．"""
    log("$ " + " ".join(cmd))
    with logfile.open("a", encoding="utf-8") as fh:
        proc = subprocess.run(
            cmd,
            stdout=fh,
            stderr=subprocess.STDOUT,
            check=False,
        )
    log(f"  rc={proc.returncode}")
    return proc.returncode


def rsync(src: Path, dst: Path, label: str) -> bool:
    """`src` を `dst` へ同期し，件数とバイト数を突き合わせる．"""
    dst.mkdir(parents=True, exist_ok=True)
    rc = run(
        ["rsync", "-a", "--delete", f"{src}/", f"{dst}/"],
        JOB_LOG,
    )
    s_n, s_b = tree_size(src)
    d_n, d_b = tree_size(dst)
    ok = rc == 0 and (s_n, s_b) == (d_n, d_b)
    log(
        f"  rsync {label}: rc={rc} src={s_n} files/{s_b} B "
        f"dst={d_n} files/{d_b} B {'OK' if ok else 'MISMATCH'}"
    )
    return ok


def tree_size(root: Path) -> tuple[int, int]:
    if not root.exists():
        return (0, 0)
    files = [p for p in root.rglob("*") if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files)


def parse_search_summary() -> dict[str, str] | None:
    """`search.log` の末尾から `search-values` の要約 dict を拾う．

    CLI は `click.echo(result)` で dict の repr を 1 行出す
    (`src/maou/infra/console/search_values.py`)．最後の 1 件を読む．
    SIGTERM で畳んだ回は要約が出ないので None になる．

    Returns:
        `{"searched": ..., "total": ...}` を含む dict．無ければ None．
    """
    if not SEARCH_LOG.exists():
        return None
    for line in reversed(
        SEARCH_LOG.read_text(errors="replace").splitlines()
    ):
        line = line.strip()
        if not line.startswith("{'searched'"):
            continue
        try:
            parsed = ast.literal_eval(line)
        except (ValueError, SyntaxError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def verdict(
    summary: dict[str, str] | None, stopped_for_budget: bool
) -> str:
    """**次のセッションが要るか**を一語で返す．

    `search-values` は `--max-positions` を上限に，残っている局面から
    標本を取る．**取れた数が上限に届かなければ，残りはもう無い**．

    Args:
        summary: `parse_search_summary()` の結果．
        stopped_for_budget: `SESSION_HOURS` で畳んだか．

    Returns:
        `"no"` (蓄積完了) / `"yes"` (続きあり) / `"unknown"`．
    """
    if stopped_for_budget:
        return "yes"
    if summary is None:
        return "unknown"
    try:
        searched = int(summary["searched"])
    except (KeyError, TypeError, ValueError):
        return "unknown"
    return "yes" if searched >= POSITIONS_PER_SESSION else "no"


def collect_diag() -> None:
    """VM の状態を `diag/` に書く (docs/colab-cli-notes.md §10 の MUST)．

    OOM なら証拠は `dmesg` にしかなく，VM を手放すと消える．失敗したときは
    必ずこれを採ってから `sync_job_dir()` で Drive へ逃がす．
    """
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    for name, cmd in DIAG_COMMANDS:
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
        (DIAG_DIR / f"{name}.txt").write_text(text)
    log(f"  diag written to {DIAG_DIR}")


def sync_job_dir() -> bool:
    """STATUS とログを Drive へ逃がす．

    書き込み中のログは rsync の件数照合が必ずずれるので本体は除き，別名の
    snapshot を送る．**VM を手放すと記録は消える**ので，成否を問わず
    unassign の前に必ず通す (docs/colab-cli-notes.md §10)．

    Returns:
        rsync が成功したか．
    """
    dst = DRIVE / JOB_DRIVE_REL
    dst.mkdir(parents=True, exist_ok=True)
    for live, snap in (
        (JOB_LOG, "job_log_snapshot.txt"),
        (SEARCH_LOG, "search_log_snapshot.txt"),
    ):
        if live.exists():
            shutil.copyfile(live, JOB_DIR / snap)
    rc = run(
        [
            "rsync",
            "-a",
            "--exclude",
            JOB_LOG.name,
            "--exclude",
            SEARCH_LOG.name,
            f"{JOB_DIR}/",
            f"{dst}/",
        ],
        JOB_LOG,
    )
    log(f"  rsync job dir -> Drive: rc={rc}")
    return rc == 0


# =====================================================================
# ワーカー (nohup の別プロセスで走る本体)
# =====================================================================
def install_wheel() -> None:
    """Release `latest` から**実行中の Python 版に合う** wheel を入れる．

    手順は `docs/colab-cli-notes.md` §6 と
    `docs/design/position-search/benchmarking.md` § "Colab (GPU)" が正．

    - タグ固定の `releases/tags/latest` を引く (`releases/latest` は
      「最新のリリース」であって `latest` タグとは別物になり得る)
    - **`cp{major}{minor}` で 1 枚に絞る．** Release には cp312 と cp313 が
      並んでいるので，全部を 1 回の `pip install` に渡すと非互換の側で
      コマンドごと落ちる (2026-09-23 に実機で発生)
    - extras は **`tensorrt-infer`**．`search-values` は ONNX GPU 推論で，
      素の `maou` には provider が入らない

    Raises:
        RuntimeError: 対応 wheel が無い / pip が失敗した場合．
    """
    url = f"https://api.github.com/repos/{REPO}/releases/tags/latest"
    with urllib.request.urlopen(url, timeout=60) as resp:
        release = json.load(resp)
    assets = release.get("assets", [])
    pytag = (
        f"cp{sys.version_info.major}{sys.version_info.minor}"
    )
    wheels = [
        a["browser_download_url"]
        for a in assets
        if a["name"].endswith(".whl") and pytag in a["name"]
    ]
    if not wheels:
        raise RuntimeError(
            f"no {pytag} wheel in release {release.get('tag_name')}: "
            f"{[a['name'] for a in assets]}"
        )
    log(f"  wheel: {wheels[0].rsplit('/', 1)[-1]}")
    rc = run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            f"maou[tensorrt-infer] @ {wheels[0]}",
        ],
        JOB_LOG,
    )
    if rc != 0:
        raise RuntimeError(f"pip install failed rc={rc}")
    link_gpu_providers()


def link_gpu_providers() -> None:
    """pip 同梱の provider / TensorRT の `.so` を loader パスへ載せる．

    `benchmarking.md` § "Colab (GPU)" の手順 2．これを踏まないと
    `--tensorrt` / `--cuda` を渡しても EP が解決できない．

    Raises:
        RuntimeError: lib ディレクトリが見つからない，または `ldconfig` の
            あとにも provider / TensorRT が解決できない場合．
    """
    dirs = (
        glob.glob(
            "/usr/local/lib/python3*/dist-packages/onnxruntime/capi"
        )
        + glob.glob(
            "/usr/local/lib/python3*/dist-packages/tensorrt_libs"
        )
        + glob.glob(
            "/usr/local/lib/python3*/dist-packages/nvidia/*/lib"
        )
    )
    if not dirs:
        raise RuntimeError(
            "no onnxruntime/tensorrt lib dirs under dist-packages"
        )
    Path("/etc/ld.so.conf.d/maou.conf").write_text(
        "\n".join(dirs) + "\n"
    )
    run(["ldconfig"], JOB_LOG)
    listed = subprocess.run(
        ["ldconfig", "-p"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    missing = [
        so
        for so in (
            "libonnxruntime_providers_shared",
            "libnvinfer.so.10",
        )
        if so not in listed
    ]
    if missing:
        raise RuntimeError(
            f"unresolved after ldconfig: {missing} "
            f"(searched {len(dirs)} dir(s))"
        )
    log("  GPU providers linked (onnxruntime + TensorRT)")


def unassign(grace_min: float) -> None:
    """猶予のあと VM を手放す (`HOLD` を touch すれば残る)．"""
    addr = os.environ.get("TBE_RUNTIME_ADDR")
    if not addr:
        log("  TBE_RUNTIME_ADDR is not set; skip unassign")
        return
    at = dt.datetime.now(JST) + dt.timedelta(minutes=grace_min)
    log(
        f"  UNASSIGN_AT={at.isoformat(timespec='seconds')} (touch {HOLD} to keep)"
    )
    set_status(
        phase="grace",
        unassign_at=at.isoformat(timespec="seconds"),
    )
    deadline = time.monotonic() + grace_min * 60
    while time.monotonic() < deadline:
        if HOLD.exists():
            log(f"  {HOLD} exists; VM kept")
            return
        time.sleep(30)
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


def worker() -> int:
    """fetch → search → sync → unassign を通しでやる．"""
    drive_out = DRIVE / OUT_REL
    local_out = WORK / OUT_REL
    teacher = WORK / TEACHER_REL
    hcpe = WORK / HCPE_REL

    set_status(phase="fetch", started=now(), pid=os.getpid())
    log(
        f"=== search-values job start (band [{MIN_PLY}, {MAX_PLY}))"
    )

    install_wheel()

    # Drive は読み書きとも一度ローカルへ落としてから触る (docs/colab-cli-notes.md)
    for rel in (TEACHER_REL, HCPE_REL):
        src, dst = DRIVE / rel, WORK / rel
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            run(["rsync", "-a", str(src), str(dst)], JOB_LOG)
            log(f"  fetched {rel} ({dst.stat().st_size} B)")
        elif not rsync(src, dst, rel):
            log(f"FATAL fetch mismatch for {rel}")
            set_status(
                phase="failed", error=f"fetch mismatch: {rel}"
            )
            collect_diag()
            sync_job_dir()
            return 1
        else:
            log(f"  fetched {rel}")

    # 既存シャードは --delete なしで取る (ローカルの方が新しいことはない)
    local_out.mkdir(parents=True, exist_ok=True)
    if drive_out.exists():
        run(
            ["rsync", "-a", f"{drive_out}/", f"{local_out}/"],
            JOB_LOG,
        )
    n, b = tree_size(local_out)
    log(f"  existing shards: {n} files/{b} B")

    digest = hashlib.sha256()
    with teacher.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != TEACHER_SHA256:
        log(
            f"FATAL teacher sha256 mismatch: got {digest.hexdigest()[:16]}... "
            f"expected {TEACHER_SHA256[:16]}...; refusing to search"
        )
        set_status(
            phase="failed", error="teacher sha256 mismatch"
        )
        collect_diag()
        sync_job_dir()
        return 1
    log(
        f"  teacher ok: {teacher.name} sha256={TEACHER_SHA256[:16]}..."
    )

    set_status(phase="search", shards_in=n)
    cmd = [
        sys.executable,
        "-m",
        "maou",
        "utility",
        "search-values",
        "--input-path",
        str(hcpe),
        "--output-path",
        str(local_out),
        "--model-path",
        str(teacher),
        "--min-ply",
        str(MIN_PLY),
        "--max-ply",
        str(MAX_PLY),
        "--max-positions",
        str(POSITIONS_PER_SESSION),
        "--playouts",
        str(PLAYOUTS),
        "--batch-size",
        str(BATCH_SIZE),
        "--threads",
        str(THREADS),
        "--flush-interval",
        str(FLUSH_INTERVAL),
        "--shard-rows",
        str(SHARD_ROWS),
        "--cuda",
        "--tensorrt",
        "--trt-cache-dir",
        str(WORK / "trt_cache"),
        "--resume",
    ]
    log("$ " + " ".join(cmd))
    with SEARCH_LOG.open("a", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            cmd, stdout=fh, stderr=subprocess.STDOUT
        )

    stopped_for_budget = False
    deadline = time.monotonic() + SESSION_HOURS * 3600
    next_sync = time.monotonic() + SYNC_EVERY_MIN * 60
    while proc.poll() is None:
        time.sleep(20)
        if time.monotonic() >= next_sync:
            rsync(local_out, drive_out, "shards (periodic)")
            sync_job_dir()
            n, b = tree_size(local_out)
            set_status(phase="search", shards=n, bytes=b)
            next_sync = time.monotonic() + SYNC_EVERY_MIN * 60
        if time.monotonic() >= deadline:
            # 壁の手前で畳む．flush 済みのシャードは残り resume が拾う
            log("  session budget reached; stopping the search")
            stopped_for_budget = True
            proc.terminate()
            try:
                proc.wait(timeout=600)
            except subprocess.TimeoutExpired:
                proc.kill()
            break
    rc = proc.returncode
    log(f"  search rc={rc}")

    if rc != 0 and not stopped_for_budget:
        # 探索が落ちた理由は VM にしか無い (OOM なら dmesg)．
        # **予算切れは失敗ではない**ので diag は採らない (毎回採ると
        # 本物の失敗の証拠がノイズに埋もれる)
        collect_diag()

    summary = parse_search_summary()
    more = verdict(summary, stopped_for_budget)

    set_status(phase="sync")
    ok = rsync(local_out, drive_out, "shards (final)")
    n, b = tree_size(local_out)
    # STATUS を先に確定させてから逃がす (Drive 側を最終状態にするため)．
    # **次のセッションが要るかは，次回のセル B がここを読んで人間に伝える．**
    set_status(
        phase="done",
        exit=rc,
        shards=n,
        bytes=b,
        evacuated=ok,
        stop_reason="budget"
        if stopped_for_budget
        else "completed",
        searched=(summary or {}).get("searched"),
        rows_total=(summary or {}).get("total"),
        more_sessions_needed=more,
    )
    ok_job = sync_job_dir()
    log(
        f"JOB_DONE exit={rc} shards={n} bytes={b} "
        f"stop_reason={'budget' if stopped_for_budget else 'completed'} "
        f"searched={(summary or {}).get('searched', '?')} "
        f"rows_total={(summary or {}).get('total', '?')} "
        f"MORE_SESSIONS_NEEDED={more} "
        f"EVACUATED={'OK' if ok and ok_job else 'FAILED'}"
    )
    if not (ok and ok_job):
        # 逃がせていない記録は VM にしか無い．手放すと失う
        log(
            "  evacuation incomplete; keeping the VM so nothing is lost"
        )
        return 1
    unassign(UNASSIGN_GRACE_MIN)
    # 予算切れは計画どおりの終わり方なので成功として返す
    return 0 if (rc == 0 or stopped_for_budget) else rc


# =====================================================================
# セル側 (ブラウザの kernel で走る): ワーカーを起こして keep-alive する
# =====================================================================
def check() -> int:
    """投入前の点検．20 時間を捨てる前に前提が揃っているか確かめる．

    探索は起こさない．読み取りと wheel の導入だけで，各項目に PASS / FAIL を出す．
    wheel をここで入れるのは，新しい VM には入っておらず `--max-ply` の有無を
    確かめられないため (投入セルの立ち上がりも速くなる)．

    Returns:
        すべて PASS なら 0，ひとつでも落ちたら 1．
    """
    fails: list[str] = []

    def check_item(
        ok: bool, label: str, detail: str = ""
    ) -> None:
        mark = "PASS" if ok else "FAIL"
        print(
            f"[{mark}] {label}"
            + (f" — {detail}" if detail else "")
        )
        if not ok:
            fails.append(label)

    print(f"=== preflight {now()} ===")
    check_item(DRIVE.exists(), "Drive mounted", str(DRIVE))

    teacher = DRIVE / TEACHER_REL
    if teacher.is_file():
        digest = hashlib.sha256()
        with teacher.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        check_item(
            digest.hexdigest() == TEACHER_SHA256,
            "teacher sha256",
            f"{teacher.name} {digest.hexdigest()[:16]}...",
        )
    else:
        check_item(False, "teacher exists", str(teacher))

    hcpe = DRIVE / HCPE_REL
    shards_in = (
        len(list(hcpe.glob("**/*.feather")))
        if hcpe.is_dir()
        else 0
    )
    check_item(
        shards_in > 0, "HCPE train", f"{shards_in} feather"
    )
    check_item(
        "val" not in HCPE_REL.split("/"),
        "HCPE excludes val",
        HCPE_REL,
    )

    # **前回の判定を先に出す．** セル C を走らせる必要があるかは，
    # 人間がここで判断できなければならない (search.log は VM ごと消える)
    prev = DRIVE / JOB_DRIVE_REL / "STATUS"
    if prev.exists():
        try:
            state = json.loads(prev.read_text())
        except ValueError:
            state = {}
        more = state.get("more_sessions_needed", "unknown")
        print(
            f"[INFO] 前回 ({state.get('updated', '?')}): "
            f"exit={state.get('exit')} "
            f"stop_reason={state.get('stop_reason')} "
            f"searched={state.get('searched')} "
            f"rows_total={state.get('rows_total')}"
        )
        if more == "no":
            print(
                "[DONE] **蓄積は完了しています．セル C は不要です．**"
                " 次は pre-process へ進む"
            )
        elif more == "yes":
            print("[INFO] 続きがあります → セル C を実行する")
        else:
            print(
                "[WARN] 前回の判定が unknown です．search.log の要約が"
                " 残っていない可能性があるので，セル C を実行して"
                " 次回の判定を待つ"
            )
    else:
        print("[INFO] 前回の記録なし (初回セッション)")

    out = DRIVE / OUT_REL
    n, b = tree_size(out)
    print(f"[INFO] output {OUT_REL}: {n} files / {b} B")
    prov = out / "provenance.json"
    if prov.exists():
        runs = json.loads(prov.read_text()).get("runs", [])
        names = {r.get("model_name") for r in runs}
        print(
            f"[INFO] provenance: {len(runs)} run(s), teacher(s) {names}"
        )
        check_item(
            names <= {teacher.name},
            "provenance teacher matches",
            str(names),
        )
    else:
        print("[INFO] provenance: none yet (first session)")

    # 新しい VM には wheel が入っていない．点検で入れておけば投入も速くなる
    try:
        install_wheel()
        check_item(
            True,
            "wheel + GPU providers",
            "tensorrt-infer, ldconfig OK",
        )
    except Exception as exc:
        check_item(
            False,
            "wheel + GPU providers",
            f"{type(exc).__name__}: {exc}",
        )
    rc = subprocess.run(
        [
            sys.executable,
            "-m",
            "maou",
            "utility",
            "search-values",
            "--help",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    check_item(
        "--max-ply" in rc.stdout,
        "wheel has --max-ply",
        "install/upgrade the wheel if this fails",
    )

    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    check_item(gpu.returncode == 0, "GPU", gpu.stdout.strip())

    print(
        f"\n=== {'ALL PASS — 投入してよい' if not fails else 'FAILED: ' + ', '.join(fails)} ==="
    )
    return 1 if fails else 0


def worker_alive() -> bool:
    if not PIDFILE.exists():
        return False
    try:
        pid = int(PIDFILE.read_text().strip())
        os.kill(pid, 0)
    except (
        ValueError,
        ProcessLookupError,
        PermissionError,
        OSError,
    ):
        return False
    return True


def ensure_worker() -> None:
    if worker_alive():
        log(
            f"worker already running (pid {PIDFILE.read_text().strip()}); attaching"
        )
        return
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    # `with` で閉じる — セルは切断のたびに貼り直されるので，開きっぱなしに
    # すると再接続のたびに fd が積もる (子は複製を持つので閉じて問題ない)
    with SEARCH_LOG.open("a") as fh:
        proc = subprocess.Popen(
            [
                sys.executable,
                os.path.abspath(__file__),
                "--worker",
            ],
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    PIDFILE.write_text(str(proc.pid))
    log(f"worker started (pid {proc.pid})")


def monitor() -> None:
    """keep-alive 兼モニタ．止めてよい (ジョブは別プロセス)．"""
    from IPython.display import clear_output

    while True:
        clear_output(wait=True)
        print(f"=== search-values monitor  {now()} ===")
        print(f"worker alive: {worker_alive()}")
        if STATUS.exists():
            print(STATUS.read_text())
        for path, tail in ((JOB_LOG, 12), (SEARCH_LOG, 8)):
            if path.exists():
                lines = path.read_text(
                    errors="replace"
                ).splitlines()
                print(f"\n--- {path.name} (last {tail}) ---")
                print("\n".join(lines[-tail:]))
        if not worker_alive():
            # **失敗も終了である．** `done`/`grace` だけを終了とみなすと，
            # 落ちたワーカーを相手に永久に回り続けて人間が待たされる
            state: dict[str, object] = {}
            if STATUS.exists():
                try:
                    state = json.loads(STATUS.read_text())
                except ValueError:
                    state = {}
            phase = state.get("phase")
            if (
                phase in ("done", "grace", "failed")
                or not state
            ):
                print(
                    f"\nworker is not running (phase={phase or 'unknown'})."
                )
                if phase == "failed":
                    print(f"error: {state.get('error')}")
                    print(
                        f"diag: {DIAG_DIR} "
                        f"(Drive: {JOB_DRIVE_REL})"
                    )
                elif not state:
                    print(
                        "STATUS was never written — the worker died "
                        f"before starting. See {SEARCH_LOG}."
                    )
                print("this cell can be stopped.")
                return
        time.sleep(300)


def main() -> int:
    if "--check" in sys.argv:
        return check()
    if "--worker" in sys.argv:
        try:
            return worker()
        except Exception as exc:  # 何であれ記録して VM は残す
            log(f"FATAL {type(exc).__name__}: {exc}")
            set_status(
                phase="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            try:
                collect_diag()
                sync_job_dir()
            except (
                Exception
            ) as eexc:  # 退避の失敗で原因を隠さない
                log(f"  evacuation failed too: {eexc}")
            return 1
    ensure_worker()
    monitor()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
