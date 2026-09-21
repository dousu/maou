# @title keep-alive: 24 時間セルを実行中のままにして Colab のアイドル切断を防ぐ (5 分ごとに job.log / STATUS の末尾を表示)
"""ブラウザのノートブックに貼って実行する keep-alive セル．

``docs/colab-cli-notes.md`` §10 のとおり，ブラウザで起動したランタイムは
ブラウザ側のセッションが唯一の keep-alive だが，セルを何も実行していないと
数時間でアイドル切断される (2026-09-21 に Arm 0 の learn 中に切断された)．
このセルを実行中のままにしておけばランタイムは busy とみなされる．

``colab exec`` で流すものではない (CLI 側の kernel ではなく，ブラウザが
attach している kernel で走らせる必要がある)．ファイル全体をセルに貼る．
5 分ごとに ``/content/job.log`` と ``/content/shogi/maou_test/jobs/arm0_*/STATUS``
の末尾を表示し直すので，ジョブの進捗もここで見える．停止ボタンで止めてよい
(ジョブ本体は nohup の別プロセスなので影響しない)．
"""

import datetime as dt
import glob
import time

from IPython.display import clear_output

HOURS = 24  # このセルを走らせ続ける時間
INTERVAL_S = 300  # 表示更新の間隔 (秒)
JST = dt.timezone(dt.timedelta(hours=9))

t0 = time.time()
t_end = t0 + HOURS * 3600
end_str = dt.datetime.fromtimestamp(t_end, JST).strftime(
    "%m-%d %H:%M JST"
)
tick = 0
while time.time() < t_end:
    tick += 1
    now = dt.datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S JST")
    clear_output(wait=True)
    print(
        f"[keep-alive] {now}  tick={tick}  "
        f"elapsed={(time.time() - t0) / 3600:.2f}h  ends {end_str}",
        flush=True,
    )
    paths = sorted(
        glob.glob("/content/shogi/maou_test/jobs/arm0_*/STATUS")
    ) + ["/content/job.log"]
    for path in paths:
        try:
            with open(
                path, encoding="utf-8", errors="replace"
            ) as f:
                tail = f.read()[-1200:]
            print(f"\n--- {path} ---\n{tail}", flush=True)
        except OSError as e:
            print(f"\n--- {path}: {e}", flush=True)
    time.sleep(INTERVAL_S)
print("KEEPALIVE_DONE", flush=True)
