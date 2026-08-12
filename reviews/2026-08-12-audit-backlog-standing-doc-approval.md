---
status: applied
applied_in: 9ac52cf
date: 2026-08-12
target: [CLAUDE.md]
risk: medium
reversibility: trivial
---

# `/audit-backlog` の自動帯に対する durable-doc standing approval

## Trigger

ユーザー指摘 (2026-08-12): 「`/audit-backlog` による修正ではユーザが毎回
修正項目を決めているが，多くの部分はユーザ判断が不要に思える」．

これを受けて `.claude/commands/audit-backlog.md` の優先度判定を
**影響度 (T1-T6)** から **判断コスト (P1-P6)** に置き換え，P1-P3 を
「ユーザ判断不要」の自動帯として無停止で修正・PR・マージするようにした．

その P2 が「ドキュメント修正のみ」であり，**CLAUDE.md の現行 MUST 規則と
正面から衝突する**．コマンド側だけを直しても，`docs/` を 1 行直すたびに
承認待ちが発生し，「ユーザ判断不要な部分を先に終わらせる」という要求は
docs に関して成立しない．規則側を直さない限りコマンドは規則違反の指示書
になる．

## Proposed change

`CLAUDE.md` § "Repository-Centric Memory Architecture (MUST)" § "MUST
rules" の第 1 項を差し替える．

**Before**

```markdown
- MUST NOT edit `CLAUDE.md` / `docs/` without an **approved** `reviews/*.md`
  proposal. Draft it `status: pending`; **on explicit user approval the
  model applies the edit itself and commits**, then sets `status: applied`
  + `applied_in: <sha>` (approval is the safeguard against *silent* edits
  — it is not tied to any one command). `/checkpoint-context` step 5 and
  `/audit-and-fix` step 8 both reconcile proposals; either may take the
  approval.
```

**After**

```markdown
- MUST NOT edit `CLAUDE.md` / `docs/` without an **approved** `reviews/*.md`
  proposal. Draft it `status: pending`; **on explicit user approval the
  model applies the edit itself and commits**, then sets `status: applied`
  + `applied_in: <sha>` (approval is the safeguard against *silent* edits
  — it is not tied to any one command). `/checkpoint-context` step 5 and
  `/audit-and-fix` step 8 both reconcile proposals; either may take the
  approval.
- **Standing approval — drift corrections only.** `/audit-backlog` の
  P2 クラス (`.claude/commands/audit-backlog.md` § "priority is decision
  cost") が **drift correction** と判定した durable-doc 修正は，この項が
  与える恒久承認により，その run 内で適用してよい．`reviews/*.md` の
  提案は依然として MUST — 承認の往復だけを省く．
  判定基準は 1 つ: **訂正後の本文が現行コードから一意に決まるか**．
  一意なら適用 (例: doc が `.npy` と書き，writer は `.feather` を書く)．
  一意でないなら — 新しい指針，節の再構成，規則の追加，複数の書き方が
  あり得る記述 — P2 ではないので判断帯に落ち，通常どおり承認を待つ．
  迷った場合は待つ側に倒す．この恒久承認は `/audit-backlog` にのみ
  適用され，`/audit-and-fix` step 8 には及ばない (拡張は別提案とする)．
```

加えて，同 § の `/audit-backlog` の行 (Files 表) に自動帯の存在を書き足す．

**Before**

```markdown
| `.claude/commands/audit-backlog.md` | Writer of `audits/` (deferred / out-of-scope の個別消化). | yes |
```

**After**

```markdown
| `.claude/commands/audit-backlog.md` | Writer of `audits/` (deferred / out-of-scope の個別消化)．判断コスト P1-P6 で分類し，P1-P3 (自動帯) はユーザに聞かずに修正・PR・マージする． | yes |
```

## Motivation

現行規則は「durable doc への *silent* な編集」を防ぐために作られている．
`/audit-backlog` P2 が行うのはその逆で，

- 変更は `reviews/*.md` 提案として残る (silent ではない)，
- 変更は PR として残り，後からユーザーが読める (silent ではない)，
- 変更内容はコードから一意に決まる (選択肢がないので「承認」に意味がない)．

承認が守っているのは *可視性* であって，*一意に決まる訂正の再確認* では
ない．O11 (`docs/commands/pre_process.md` が pre-process の出力を `.npy`
と説明しているが実体は `.feather`) が典型で，ここでユーザーに問える質問は
「コードに合わせますか」しかない．

実害も出ている: O11 は 2026-08-10 の run で見つかりながら「durable doc
なので `reviews/` 提案 + 承認が必要」という理由だけで未修正のまま
out-of-scope backlog に積まれている．承認の往復コストが，訂正そのものの
コストを上回っている状態である．

## Alternatives considered

1. **規則を変えず，P2 を「提案は出すが承認を待つ」にする．**
   コマンド側だけで完結し CLAUDE.md を触らないので最も安全だが，ユーザー
   要求「ドキュメント修正のみ = ユーザ判断不要」を満たさない．doc drift は
   backlog に最も溜まりやすい種類で，そこが自動化から外れると今回の変更の
   効果が大きく削がれる．却下．

2. **durable-doc の承認ゲートを全面撤廃する．**
   一貫はするが，守るべきものまで捨てる．`CLAUDE.md` の規則追加や
   `docs/design/` の設計判断は，コードから一意に決まらない「著述」であり，
   ここに承認が要るのは正しい．P2 の第 2 テスト (一意性) はまさにこの線を
   引くために置かれている．却下．

3. **`/audit-and-fix` step 8 にも同じ恒久承認を広げる．**
   整合性は上がるが，`/audit-and-fix` は path 全体を新規に走査する run で，
   drift の判定が「その場で読んだ理解」に依存する．`/audit-backlog` は
   **既に別 run が記録し，step 2 で HEAD に対して再検証した** finding しか
   扱わないため，一意性判定の土台が違う．今回は範囲を絞り，必要になったら
   別提案とする．

## What this enables

- doc drift が「一意な訂正」である限り，発見と同じ run で消える．backlog に
  残るのは**判断が要る doc 変更だけ**になり，`coverage.md` の残件が
  「本当にユーザーを待っているもの」を意味するようになる．
- ユーザーへの質問が，判断のあるものだけに絞られる．

## What this constrains

- P2 判定を誤ると durable doc が未確認のままマージされる．そのため
  コマンド側で fail-safe を「上に倒す」(迷ったら判断帯) と明記し，判定の
  根拠を `audits/` の record § Classification に残すことを必須にしている．
  誤りは PR 履歴と record の両方から追える．
- 「一意に決まる」の判定は毎回書き残す義務が生じる (record に「どのテストで
  P2 と判定したか」を書く)．黙って P2 にすることはできない．

## Rollback plan

`CLAUDE.md` の当該 2 箇所を revert し，`.claude/commands/audit-backlog.md`
の 4d を「常に承認を待つ」1 分岐に戻す (P2 クラス自体は残してよい —
自動帯から外れるだけ)．コードは一切影響を受けない．既に適用済みの doc
訂正はコードと一致しているので，revert しても訂正を戻す必要はない．
