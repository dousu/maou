---
status: applied          # pending | approved | applied | rejected
applied_in: 53eeba04
date: 2026-09-21
target: [docs/colab-cli-notes.md, .claude/skills/colab-operator/SKILL.md]
risk: low
reversibility: trivial
---

# Colab 運用規約に keep-alive セル / 完了時の自動 unassign / 段階完了時の退避を足す

## Trigger

Arm 0 (対照実験) を §10 の手順でブラウザ起動の G4 に投入したところ，learn 開始
2 時間後 (2026-09-21 19:07–19:40 JST，VM 起動から約 3.5〜4 時間) にセッションが
切断された．ユーザの観察: **ブラウザのタブを開いているだけでは足りず，ノート
ブックでセルを実行していないと数時間でアイドル切断される**．

同時に 2 つの取りこぼしが見えた:

1. `scripts/colab_arm0_job.py` は preprocess (約 1 時間) の出力を learn の前に
   Drive へ退避していなかった (learn 中の周期退避は models / logs だけ)．
   §7.3 の「1 ステージ完了時にコピー」に反しており，切断で前処理をやり直した．
2. ジョブが完走したあと VM を止める手段が「ユーザがブラウザで削除する」しか
   なく，完走から気付くまで課金が続く．ユーザから `google.colab.runtime.unassign()`
   による自動終了の提案があった．VM 上で中身を確認したところ
   `POST http://$TBE_RUNTIME_ADDR/unassign` (+ frontend への JS 送信) で，
   環境変数は kernel の子プロセスに継承されるので nohup の driver から呼べる．

対応 (同 PR): `scripts/colab_keepalive_cell.py` (貼って実行する keep-alive セル)，
`scripts/colab_arm0_job.py` に preprocess 完了直後の退避と `--unassign-on-done`
(rc=0 のとき猶予 `--unassign-grace-min` (既定 60 分) のあと unassign．
`<work>/arm0_<tag>/KEEP_VM` で取りやめ．失敗時は診断のため VM を残す)．
ユーザ指示: 「keep-alive セルと自動終了と preprocess の退避は colab の使用
ルールとしてドキュメントに残しておいてください」(2026-09-21)．

## 変更 1 (docs/colab-cli-notes.md §7.3 — 段階完了時の退避を具体化)

現行:

```
- Drive へのコピーは「作業のまとまり」単位: 1 ステージ完了時 (hcpe → preprocess → …)，
  epoch 数回ごとの checkpoint，`search-values` の shard がいくつか確定したとき，
  および **`colab stop` の直前**．コピー後に `ls -l` でサイズを突き合わせてから次へ進む．
```

変更後:

```
- Drive へのコピーは「作業のまとまり」単位: 1 ステージ完了時 (hcpe → preprocess → …)，
  epoch 数回ごとの checkpoint，`search-values` の shard がいくつか確定したとき，
  および **`colab stop` の直前**．コピー後に `ls -l` でサイズを突き合わせてから次へ進む．
- **次の段階を始める前に，済んだ段階の出力を退避する (MUST)**．特に 1 時間級の
  `pre-process` 出力は `learn-model` に入る前に Drive へコピーする．learn 中の周期退避
  (models / logs) だけでは，VM を失ったとき前処理からやり直しになる
  (2026-09-21 の Arm 0: learn 中の切断で `preprocess_20260921` を失った)．
  `scripts/colab_arm0_job.py` は preprocess 段階の完了直後に退避する．
```

## 変更 2 (docs/colab-cli-notes.md §10 — keep-alive セルと自動 unassign)

手順ブロックの 1. を次に置き換える:

```
# 1. ユーザ: ブラウザ (colab.research.google.com) で GPU 種別を選んでランタイムを起動し，
#    タブを開いたままにする．必要なら同じノートブックのセルで drive.mount も済ませる (§0(d))．
#    さらに `scripts/colab_keepalive_cell.py` の中身を新しいセルに貼って実行し，
#    ジョブの間ずっと「実行中」にしておく (下記)
```

箇条書きの 1 つ目 (「登録したセッションには keep-alive デーモンが無い (§3)．…」) を
次に置き換える:

```
- 登録したセッションには keep-alive デーモンが無い (§3)．**ブラウザのタブを開いている
  だけでは足りない — ノートブックでセルを実行していないと数時間でアイドル切断される**
  (2026-09-21: Arm 0 の learn 開始 2 時間後，VM 起動から約 3.5〜4 時間で切断)．
  **`scripts/colab_keepalive_cell.py` を新しいセルに貼って実行し，ジョブの間ずっと
  実行中のままにする (MUST)**．24 時間走り，5 分ごとに `/content/job.log` と
  `/content/shogi/arm0_*/STATUS` の末尾を表示し直す (進捗もここで見える)．
  `colab exec` で流すものではない (ブラウザが attach している kernel で走らせる)．
  Colab 側の上限 (最長実行時間) はユーザのプラン次第で，ユーザが把握する．
```

箇条書きの 2 つ目 (「**登録したセッションを `colab stop` しない．**…」) の直後に
次を追加する:

```
- **完走したら VM 側が自分で unassign する (MUST)**．長時間ジョブの driver には
  `google.colab.runtime.unassign()` と同じ `POST http://$TBE_RUNTIME_ADDR/unassign` を
  rc=0 の終わりに入れる (`scripts/colab_arm0_job.py --unassign-on-done`．環境変数は
  kernel の子プロセスに継承されるので nohup からでも通る)．CLI 側から `stop` しない規約は
  そのまま — 止めるのは VM 上のジョブ自身．猶予 (`--unassign-grace-min`，既定 60 分) の間に
  `colab download` で小物 (value-best の `_fp16.onnx` 等) を取る．猶予を過ぎて VM が消えた
  あとは Drive にある (§7.3 で退避済み) ので，ユーザに CPU ランタイム + `drive.mount` を
  依頼して `colab download` で取る．**失敗 (rc≠0) のときは段階ログを見られるよう VM を
  残す**ので，診断後にユーザがブラウザで削除する．
```

## 変更 3 (.claude/skills/colab-operator/SKILL.md — overrides の同期)

```
- Sessions registered with `scripts/colab_adopt_session.py` (browser-started runtimes) have no
  keep-alive daemon and **must not be `colab stop`ped** — the user deletes them in the browser.
```

を次に置き換える:

```
- Sessions registered with `scripts/colab_adopt_session.py` (browser-started runtimes) have no
  keep-alive daemon and **must not be `colab stop`ped**. An open tab is not enough: the user
  runs `scripts/colab_keepalive_cell.py` in a notebook cell for the whole job (idle runtimes
  are cut after a few hours). A long job ends by unassigning its own VM
  (`colab_arm0_job.py --unassign-on-done`, `POST $TBE_RUNTIME_ADDR/unassign`) after a grace
  window for `colab download`; on failure the VM is kept for diagnosis.
- Copy each finished stage's output to Drive before starting the next one (the ~1 h
  `pre-process` output before `learn-model`).
```

## 影響

- Claude Code の運用のみ．`src/` の変更なし (版数据え置き)．
- keep-alive セルの効果と unassign の発火は 2026-09-21 20:01 JST 投入の Arm 0
  (tag `20260921_2001`) で初めて実運用に載せる．結果は worklog に残す．
