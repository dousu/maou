---
status: applied          # pending | approved | applied | rejected
applied_in: PENDING_SHA
date: 2026-09-23
target: [docs/colab-cli-notes.md]
risk: low
reversibility: trivial
---

# Colab の driver 規約を campaign 固有名から切り離し，新規 driver 用のチェックリストにする

## Trigger

`scripts/colab_search_values_job.py` (探索値蓄積の driver，2 本目) を書いたとき，
§10 が MUST としている **(1) 記録の Drive 退避** と **(2) 失敗時の `diag/` 採取** の
両方を落とした．規約は読んでいたが，本文が `scripts/colab_arm0_job.py` という
**1 本のスクリプトの挙動説明**として書かれているため，「新しい driver を書くときに
何を満たす必要があるか」が読み取れなかった．PR #528 で 2 コミットぶん追加で
修正している (`737c06bd` / `1355f365`)．

同じ固有名が `.claude/skills/colab-operator/SKILL.md` にも漏れていた (user 指摘)．
skill 側は本提案とは別に直接修正した (`docs/` 外のため)．

**Arm 0 は 1 回の実験の名前**であって driver の種類ではない．auto-memory の
process rule「会話依存ラベルを永続成果物に残さない — 機能名で記述する」に反する．

なお **2026-09-21 の Arm 0 の事故そのものへの言及 (272 行，382 行，405 行) は残す**．
これは規則が存在する理由の証拠であり，固有名で書くのが正しい．置き換えるのは
「どの driver でも成り立つべき規範」として書かれている箇所だけ．

## 変更 1 (§7.4 レイアウト表，289 行)

現行:

```
| ジョブの記録 (`STATUS` / 段階ログ / `diag/`) | `scripts/colab_arm0_job.py` | `maou_test/jobs/` | `maou_test/jobs/` |
```

変更後:

```
| ジョブの記録 (`STATUS` / 段階ログ / `diag/`) | 長時間ジョブの driver (`scripts/colab_*_job.py`) | `maou_test/jobs/` | `maou_test/jobs/` |
```

## 変更 2 (§7.4 ジョブ記録の説明，300 行)

現行:

```
- `maou_test/jobs/arm0_<tag>/` は driver の記録 (数 MB): `STATUS`，段階ログ，driver ログの
```

変更後:

```
- `maou_test/jobs/<ジョブ名>_<tag>/` は driver の記録 (数 MB): `STATUS`，段階ログ，driver ログの
```

(実例: `arm0_20260922_0736/`，`search_values_20260923/`)

## 変更 3 (§10 keep-alive の表示パス，385 行)

現行:

```
  `/content/shogi/maou_test/jobs/arm0_*/STATUS` の末尾を表示し直す (進捗もここで見える)．
```

変更後:

```
  `/content/shogi/maou_test/jobs/*/STATUS` の末尾を表示し直す (進捗もここで見える)．
```

`scripts/colab_keepalive_cell.py` の glob は同じ内容へ既に修正済み (本 PR)．
`arm0_*` のままだと探索値ジョブの STATUS が表示されない実害があった．

## 変更 4 (§10 unassign，393 行)

現行:

```
  rc=0 の終わりに入れる (`scripts/colab_arm0_job.py --unassign-on-done`．環境変数は
```

変更後:

```
  rc=0 の終わりに入れる (例: `scripts/colab_arm0_job.py --unassign-on-done`．環境変数は
```

## 変更 5 (§10 失敗時の退避，402 行)

現行:

```
  を Drive (`maou_test/jobs/arm0_<tag>/` と `maou_test/{models,logs}/`) へ退避し，rsync の
```

変更後:

```
  を Drive (`maou_test/jobs/<ジョブ名>_<tag>/` と `maou_test/{models,logs}/`) へ退避し，rsync の
```

## 変更 6 (§10 の末尾に新規小節を追加)

`## 11. Claude Code の判断フロー (要約)` の直前へ:

```markdown
### 10.1 新しい long-running driver を書くときのチェックリスト (MUST)

上の規約は `scripts/colab_arm0_job.py` の説明ではなく，**`scripts/` に置く
すべての長時間ジョブ driver が満たすもの**である．2 本目 (`colab_search_values_job.py`)
は退避と `diag/` 採取を落として書かれ，後から直した．新しい driver を追加するときは
以下を実装してからレビューに出す:

- [ ] **出力はまず VM ローカル**に書き，まとまりごとに Drive へ rsync する (§7.3)
- [ ] Drive の読み書きは**マウント経由**，**`MyDrive/shogi` の中だけ** (§7.1 / §7.2)
- [ ] 出力先は `<種別>_<YYYYMMDD>` (§7.4)．**帯や条件を名前に詰め込まない** (§9)．
      複数セッションに分ける蓄積では**この値を固定する** (実行時生成だと再開できない)
- [ ] `STATUS` と段階ログを `maou_test/jobs/<ジョブ名>_<tag>/` へ逃がす．
      **書き込み中のログは rsync の件数照合が必ずずれる**ので本体は除き，
      別名の snapshot を送る
- [ ] **失敗経路すべて**で `diag/` を採る (`dmesg` 末尾 / `free` / `/proc/meminfo` /
      cgroup / `df` / `nvidia-smi` / `ps` / 版数)．**OOM の証拠は VM の `dmesg` に
      しか無く，unassign すると消える**
- [ ] unassign は **記録と成果物の退避が rsync 照合 OK のときだけ**行う．
      片方でも失敗したら VM を残す (`KEEP_VM` でユーザが明示的に残せること)
- [ ] 成否判定に `cmd | tail` を使わない (exit code が tail のものになる)
- [ ] 投入前の点検手段を持つ (例: `colab_search_values_job.py --check`)．
      数十時間を投じる前に，入力・モデルの同一性・wheel の版・GPU を確かめる
```

## 影響

規範の意味は変わらない．固有名が規範から外れ，2 本目以降の driver を書くときに
何を満たすべきかが本文から読めるようになる．
