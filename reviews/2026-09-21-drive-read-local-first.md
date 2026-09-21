---
status: applied          # pending | approved | applied | rejected
applied_in: 533d72e2
date: 2026-09-21
target: [docs/colab-cli-notes.md, .claude/skills/colab-operator/SKILL.md]
risk: low
reversibility: trivial
---

# Drive からの読み取りも (TensorBoard ログを含め) ローカルへコピーしてから行う

## Trigger

Arm 0 の準備で，基準 run の学習レシピを復元するために Drive 上の TensorBoard
イベントファイル (`MyDrive/shogi/maou_test/logs/…/events.out.tfevents.*`) を
FUSE マウント越しに**直接** `EventAccumulator` で読んだ (2026-09-21)．
その直後の `colab exec` が `RuntimeError: Connection was lost.` で 1 度落ちた
(因果は未確認)．ユーザが「TensorBoard のログもローカルファイルシステムに
コピーしてから読み取るように」と Drive 規約の変更を依頼した．

現行 §7.3 は「入力も同様に，Drive から `/content/shogi/` へコピーしてから使う
(FUSE 直読みは遅く，DataLoader の並列読みで不安定)」と書いており，学習の
**入力データ**を念頭に置いた文面になっている．ログやモデルの**解析目的の
読み取り**が規約の対象に入ることが読み取れない．

## 変更 (docs/colab-cli-notes.md §7.3)

現行:

```
- 入力も同様に，Drive から `/content/shogi/` へコピーしてから使う (FUSE 直読みは遅く，
  DataLoader の並列読みで不安定)．
```

変更後:

```
- **読み取りも同様に，Drive から `/content/shogi/` へコピーしてから使う**．学習の入力
  (HCPE / 前処理済) だけでなく，TensorBoard のイベントファイルやモデルを**解析目的で
  読む場合も含む** (例: `EventAccumulator` で基準 run の LR 曲線を読む)．FUSE 直読みは
  遅く，DataLoader の並列読みで不安定になるうえ，多数の小ファイルを舐める読み取りは
  kernel の接続断を招きうる．`rsync -a /content/drive/MyDrive/shogi/X/ /content/shogi/X/`
  で取り，ローカル側を読む．Drive 上で許されるのは `ls` 相当の一覧取得だけ．
```

併せて `.claude/skills/colab-operator/SKILL.md` の "maou project overrides" の
Drive 行を次に置き換える (SKILL.md は `docs/` ではないが，本メモと同期させる規約
§0(b) に従う):

```
- Drive: write only from inside the VM via the mount, only under `MyDrive/shogi`,
  local-first — for **reads too** (copy to `/content/shogi/` before parsing
  TensorBoard events or models; only `ls` directly on the mount).
```

## 影響

- Claude Code の運用のみ．コード変更なし．
- `scripts/colab_arm0_job.py` の `fetch` 段階は既にこの形 (入力を rsync してから使う)．

## Rollback

該当 2 箇所を元の文に戻す．
