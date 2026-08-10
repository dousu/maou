---
status: pending
applied_in:
date: 2026-08-10
target:
  - audits/README.md
risk: low
reversibility: trivial (2箇所の短い追記)
---

# record の散文節が worklist と誤読される罠を README で塞ぐ

## Trigger

2026-08-10，`/audit-and-fix src/maou/infra/file_system high` の実行後に
ユーザーから「未修正分は `/audit-backlog` で回収される認識だが漏れは
ないか」と確認があり，照合したところ **約9件が backlog 行を持たずに
落ちていた**．

`.claude/commands/` 側の再発防止 (step 9x / 5d の reconciliation) は
`d6e3cd6` で適用済み．本提案はそれが依存する **durable doc 側** 1点．

## Motivation

落ちた9件のうち1件は，統合ミスではなく **README の record shape が
招いた**．

`2026-08-10-cross-module-consistency-and-pipeline-doc.md` (applied,
`8c13fa9`) で record shape に追加した節がこれである:

```markdown
## Cross-module sweep
<step 2.5 で導出した sweep key と，各 key の結果．finding だけでなく
**clean だった key も書く** — 「調べて一貫していた」は次の隣接 path 監査が
同じ Explore sweep を再実行しないための結果である．意図的な分岐は理由と
ともにここに記録する．>
```

「**finding だけでなく** clean だった key も書く」と読めるため，
finding をここに書けば記録されたことになる，と自然に解釈できる．
実際そう解釈した結果，「`.feather` で終わったまま中途書き込みの
ファイルはどのフィルタも通過する」という **open な finding** が
この節にしか存在せず，`/audit-backlog` から永久に不可視だった
(後から `D15` として行を追加)．

README は既に § "Records are accounts, not worklists" で
「どのコマンドも record を worklist として読まない」と述べているが，
その記述は **Deferred 節についての説明**として置かれており，
新設した `## Cross-module sweep` 節や自由記述の余談には及んでいない．
罠は「record 全体が worklist ではない」という一般則を，
節ごとの説明が上書きしてしまう点にある．

## Proposed change

### (A) record shape の `## Cross-module sweep` に1文追記

before:
```markdown
## Cross-module sweep
<step 2.5 で導出した sweep key と，各 key の結果．finding だけでなく
**clean だった key も書く** — 「調べて一貫していた」は次の隣接 path 監査が
同じ Explore sweep を再実行しないための結果である．意図的な分岐は理由と
ともにここに記録する．>
```
after:
```markdown
## Cross-module sweep
<step 2.5 で導出した sweep key と，各 key の結果．finding だけでなく
**clean だった key も書く** — 「調べて一貫していた」は次の隣接 path 監査が
同じ Explore sweep を再実行しないための結果である．意図的な分岐は理由と
ともにここに記録する．

**この節は worklist ではない．** ここに書ける「結果」は clean な key と
意図的な分岐 — つまり **これ以上やることがないもの** だけである．
open な finding をここに書いても `coverage.md` に行がなければ
`/audit-backlog` からは見えない．finding は必ず backlog 行を持たせ，
この節にはその要約と参照だけを書くこと．>
```

### (B) § "Records are accounts, not worklists" に1段落追記

`## Records are accounts, not worklists` 節の末尾 (Correction の
コードブロックの直前) に挿入:

```markdown
**これは特定の節ではなく record 全体の性質である．** Deferred 節だけ
でなく，`## Cross-module sweep`，`## Out of scope`，および本文中の
どんな余談についても同じく当てはまる．どのコマンドも open work を
`coverage.md` の2表からのみ収集するので，**行を持たない finding は
どこに書かれていても存在しないのと同じ**である．節を新設するときは
「ここに書けば記録されたことになる」と読まれないか必ず確認すること —
2026-08-10 に `## Cross-module sweep` でこれが実際に起きた．
```

## Alternatives considered

- **README を触らず，コマンド側の 9x / 5d だけで足りるとする.**
  9x は「散文節は worklist ではない」と明示しているので，
  実行時の防止としては足りる．ただし README は record shape の
  **正本**であり，将来 record に節を追加する人が読むのは README の方．
  罠を作った側 (shape 定義) を直さないと同じ形の節がまた生える．
- **`## Cross-module sweep` 節そのものを廃止する.**
  罠は消えるが，clean な key の記録は実測で価値があった
  (隣接 path 監査が同じ `Explore` sweep を再実行せずに済む)．
  節ではなく説明文の不足が原因なので，廃止は過剰．
- **finding を record に書くこと自体を禁じる.**
  backlog 行は凝縮された索引であり，record は durable な理由付け．
  両方書くのが現行設計 (README § "Deferred findings") で，
  そこは正しい．問題は「行なしで record にだけ書く」ケース．

## What this enables

- record に新しい節を足す人が，その節を worklist と誤読させない
  書き方を選べる (罠の再生産を止める)．
- 「行を持たない finding は存在しないのと同じ」という不変条件が，
  Deferred 節の説明ではなく **record 全体の性質**として一箇所に立つ．

## What this constrains

- `## Cross-module sweep` に finding を書くときは，backlog 行を先に
  作ってから参照する順序になる．手間が1つ増えるが，これは
  reconciliation (9x) が要求する順序と同じなので実質的な追加負担は
  ない．

## Rollback plan

(A) と (B) の追記を削除するだけ．相互依存なし．コードに影響なし．
