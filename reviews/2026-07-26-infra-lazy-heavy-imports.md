---
title: CLAUDE.md に「軽量コマンドの起動経路は重い依存を module-level に import しない」規約を追加
date: 2026-07-26
status: applied
applied_in: 39b1580
target:
  - CLAUDE.md
risk: low
reversibility: easy
---

# 提案: 重い依存の module-level import 禁止を CLAUDE.md の規約にする

## Trigger

worklog/2026-07-26-214115.md — USI M4 の計測中に，`maou usi` の起動が
**6.2 秒**かかっていることが判明した．内訳は
`maou.infra.console.common` が module-level で BigQuery を試験 import し，
`maou.infra.bigquery.bq_data_source` → `maou.interface.learn` 経由で
**torch 2.09s + google-cloud-bigquery 1.93s** を読み込んでいたため
(`python -X importtime` で実測)．USI エンジンは GUI に登録され**対局の
たびに起動**されるので，起動時間はそのまま体感になる (spawn→usiok
5.9〜15.7 秒を実測)．遅延化後は **0.11 秒**．

同種の回帰は過去にもあり，compass には
「interface.learn の module-level torch import を infra 継承禁止
(base install 破壊)」という invariant が既にある．つまり **2 回目**である．
compass は per-machine な scratchpad なので，恒久規約は CLAUDE.md
(committed) 側に持たせたい．

## 提案する変更 (CLAUDE.md)

`## Critical Rules (MUST)` の `### Architecture` 節に 2 行追加する:

```
- MUST NOT import heavy optional dependencies (torch, cloud SDKs) at
  module level in `src/maou/infra/console/` or any path a light command
  reaches — use PEP 562 `__getattr__` lazy resolution
- MUST NOT leave a module-level assignment for a name resolved lazily by
  `__getattr__` (the global shadows the hook and leaks the pre-probe value)
```

## リスクと理由

- **risk: low** — 既存コードは既にこの形 (`console/common.py` の
  `_LAZY_IMPORTS` / `_CLOUD_PROBES`) に揃っており，
  `tests/maou/infra/console/test_console_import_weight.py` が機械的に
  検査している．規約の明文化のみ．
- **reversibility: easy** — 2 行の削除で戻る．

## 代替案と却下理由

- **テストだけで担保する (規約化しない)**: テストは `usi`/`selfplay` の
  2 経路しか見ていないため，新しい軽量コマンドを足したときに検査対象へ
  加え忘れる．規約があれば追加時に気付ける．
- **compass の invariant だけで担保する**: compass は gitignore された
  per-machine ファイルで，かつ ~9KB の cap があり evict 対象になり得る．
  2 回発生した回帰の恒久的な置き場所としては弱い．

## ロールバック

CLAUDE.md の該当 2 行を削除する．コード側は無変更で動作する．
