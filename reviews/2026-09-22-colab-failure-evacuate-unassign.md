---
status: pending          # pending | approved | applied | rejected
applied_in:
date: 2026-09-22
target: [docs/colab-cli-notes.md, .claude/skills/colab-operator/SKILL.md]
risk: low
reversibility: trivial
---

# Colab 運用規約: 失敗時も診断情報を Drive へ退避してから unassign する / ジョブ記録の置き場

## Trigger

2026-09-21 20:01 JST に再投入した Arm 0 は，learn の epoch 2 (2444/18811) で
22:10:29 JST に落ちた．原因は VM 上の `dmesg` にしか無かった:

```
pt_data_worker invoked oom-killer ... oom_memcg=/jupyter-children
memory: usage 182414336kB, limit 182414336kB
Killed process 477651 (pt_data_worker) anon-rss:35834132kB
```

memory cgroup (`/jupyter-children`，上限 174 GiB) の OOM で，5 本の学習
DataLoader worker が各 ≈34 GiB (合計 ≈169 GiB) + 主プロセス 3.4 GiB で上限に
達していた (各 worker が 1 ファイル ≈ 12 GB の Polars DataFrame + 9 GB の
ColumnarBatch を持ち，前ファイルの ColumnarBatch を解放する前に次を読む．
`moveWinRate` 1496 × f32 で 1 ファイルが LZ4 で約 68 倍に展開される)．

ここで 3 つの取りこぼしが見えた:

1. **失敗後 8 時間以上 VM が空転した**．§10 は「失敗 (rc≠0) のときは段階
   ログを見られるよう VM を残す」としており，driver もそのとおり unassign
   しなかった．ユーザ指示: 「スクリプトが失敗したときもエラーハンドリングで
   unassign するようにしてください．そのときに，デバッグに必要な情報は
   Google Drive に退避しておくようにしてください」(2026-09-22)．
2. OOM の証拠 (`dmesg`) と段階ログ (`learn.log` 3.8 MB) は VM ローカルにしか
   無く，unassign すると消える．退避先が §7.4 の表に無い．
3. epoch 1 で保存されたモデル (22:04) は periodic sync (30 分毎，直前は 21:48)
   の前に learn が落ちたため Drive に無かった．

対応 (同 PR，`scripts/colab_arm0_job.py`): ジョブの記録 (`STATUS` / 段階ログ /
driver ログ) を `maou_test/jobs/arm0_<tag>/` に置き，終了時 (成否を問わず) に
Drive の同じ相対パスへ退避する．失敗時はさらに `diag/` (`dmesg` 末尾 / `free` /
`/proc/meminfo` / cgroup / `df` / `nvidia-smi` / `ps` / 版数) を採り，models /
logs の最終退避も行ってから，退避の rsync 照合が OK のときだけ unassign する
(`--unassign-on-done` が成否を問わなくなる)．退避できなければ `UNASSIGN_SKIPPED`
を出して VM を残す．VM が残っているときは `--evacuate-only` で退避だけやり直せる．
新しい VM で learn からやり直すための `--train-preprocessed` (Drive の前処理済を
`fetch` で戻し，`soften` / `preprocess` を飛ばす) も足した．

## 変更 1 (docs/colab-cli-notes.md §7.4 — ジョブ記録の行を足す)

表に 1 行足す:

```
| ジョブの記録 (`STATUS` / 段階ログ / `diag/`) | `scripts/colab_arm0_job.py` | `maou_test/jobs/` | `maou_test/jobs/` |
```

箇条書きの末尾 (「上表にない一時物は Drive に置かない．…」の前) に足す:

```
- `maou_test/jobs/arm0_<tag>/` は driver の記録 (数 MB): `STATUS`，段階ログ，driver ログの
  snapshot，失敗時の `diag/` (`dmesg` 末尾 / `free` / `df` / `nvidia-smi` / `ps`)．
  再生成できない (失敗の証拠) ので §7.5 の「10 分未満で再生成できるもの」には当たらない．
  `soften` の出力 (`*.feather`，1 分で作り直せる) は退避しない．
```

## 変更 2 (docs/colab-cli-notes.md §7.3 — learn からの再開手段)

「途中で VM を失っても Drive にある中間物から再開できる状態を保つ …」の箇条書きを
次に置き換える:

```
- 途中で VM を失っても Drive にある中間物から再開できる状態を保つ (`search-values --resume` は
  shard ディレクトリを Drive から戻せばそのまま続きから走る．Arm 0 は
  `scripts/colab_arm0_job.py --train-preprocessed preprocess/preprocess_<前の tag>` で
  前処理済を `fetch` が戻し，`soften` / `preprocess` を飛ばして learn から走る．`--tag` は
  新しくする — モデル / ログのフォルダは learn-model 1 回ごとに分ける)．
```

## 変更 3 (docs/colab-cli-notes.md §10 — keep-alive セルの表示パス)

```
  `/content/shogi/arm0_*/STATUS` の末尾を表示し直す (進捗もここで見える)．
```
を
```
  `/content/shogi/maou_test/jobs/arm0_*/STATUS` の末尾を表示し直す (進捗もここで見える)．
```
に置き換える．

## 変更 4 (docs/colab-cli-notes.md §10 — 失敗時も退避してから unassign)

「**完走したら VM 側が自分で unassign する (MUST)**」の箇条書きの末尾
```
  依頼して `colab download` で取る．**失敗 (rc≠0) のときは段階ログを見られるよう VM を
  残す**ので，診断後にユーザがブラウザで削除する．
```
を次に置き換える:

```
  依頼して `colab download` で取る．
- **失敗 (rc≠0) のときも，診断情報を Drive へ退避してから unassign する (MUST)**．
  driver は失敗した段階のあと `diag/` (`dmesg` 末尾 / `free` / `/proc/meminfo` / cgroup /
  `df` / `nvidia-smi` / `ps` / 版数) を採り，段階ログ・driver ログ・保存済みの models / logs
  を Drive (`maou_test/jobs/arm0_<tag>/` と `maou_test/{models,logs}/`) へ退避し，rsync の
  照合が OK のときだけ同じ猶予のあと unassign する．退避できなかった (Drive 無し / 不一致)
  ときは `UNASSIGN_SKIPPED` を出して VM を残すので，直して `--evacuate-only` で退避を
  やり直してからユーザがブラウザで削除する．理由: 2026-09-21 の Arm 0 は learn が 22:10 に
  memory cgroup の OOM で落ちたあと 8 時間以上 VM が空転した．OOM の証拠は VM の `dmesg`
  にしか無く，epoch 1 のモデルも periodic sync の前で VM に取り残されていた．
  原因調査は Drive の `diag/` と段階ログから行い，VM を残す判断は KEEP_VM でユーザが行う．
```

## 変更 5 (.claude/skills/colab-operator/SKILL.md — overrides の同期)

```
>   window for `colab download`; on failure the VM is kept for diagnosis.
```
を
```
>   window for `colab download`. On failure it first copies diagnostics (`dmesg` tail, `free`,
>   `nvidia-smi`, `ps`), stage logs and saved checkpoints to Drive
>   (`maou_test/jobs/arm0_<tag>/`, `maou_test/{models,logs}/`) and unassigns only when that copy
>   verified; otherwise it prints `UNASSIGN_SKIPPED` and keeps the VM (`--evacuate-only` retries).
```
に置き換える．

## 影響

- 規約の変更は「失敗時は VM を残す」→「失敗時は退避してから unassign」の 1 点．
  退避が確認できないときは従来どおり残るので，診断情報を失う経路は増えない．
- Drive のレイアウトに `maou_test/jobs/` が増える (トップレベルは増えない)．
- driver の job dir が `/content/shogi/arm0_<tag>` から
  `/content/shogi/maou_test/jobs/arm0_<tag>` に移る (Drive と同じ相対パス．
  keep-alive セルの glob も追従)．
