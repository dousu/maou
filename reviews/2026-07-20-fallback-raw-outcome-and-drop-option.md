---
title: bestMoveWinRateフォールバック戦略の追加オプション化 (raw-outcome / drop-below-threshold)
date: 2026-07-20
status: applied
applied_in: 248d17d
target:
  - docs/commands/pre_process.md
  - docs/adr-005-move-win-rates.md
risk: low
reversibility: easy
---

# 提案: `--best-move-win-rate-fallback` / `--drop-below-threshold` オプション追加のドキュメント反映

## 背景

evaluateコマンドで初期局面の評価値が-700(勝率25%)にずれる問題を調査した結果，
根本原因は `--position-count-threshold` によるフォールバックが実データのほぼ
全局面で発動しており，`bestMoveWinRate`(value-target-mode
`best-move-win-rate` が学習で使う値)が軒並み固定値0.5になっていたこと．
ADR-005 (`docs/adr-005-move-win-rates.md`) はフォールバック値として
「均等配分(1/N) / 固定値0.5」を意図的に採用しているが，これは「サンプルが
少ない局面のノイズを抑制する」ことを主目的とした設計であり，「フォールバック
がほぼ全局面で発生し学習信号そのものが消える」ケースは想定されていなかった．

## 変更内容 (実装済み，`src/` 側は本レビュー対象外)

`maou pre-process` に2つの新規オプションを追加した(コード変更は完了，
バージョンは `maou 0.50.0 → 0.51.0`):

1. **`--best-move-win-rate-fallback {uniform,raw-outcome}`** (デフォルト:
   `uniform`，既存動作を維持)
   - `raw-outcome` を指定すると，`count < position_count_threshold` の
     局面の `bestMoveWinRate` を，平滑化なしの実勝敗
     (`win_values / label_values` の最大値)で記録する．
     サンプル数が少ないほど 0.0/1.0 に近い極端な値になり得るが，
     「情報が失われて0.5に潰れる」よりはマシという判断．
   - `moveWinRate` 配列(policy側，`--policy-target-mode win-rate` が使う)
     は本オプションの影響を受けず，常に均等配分(1/N)のまま
     (ユーザー承認: value側のみへの適用に限定するスコープ)．
   - 実装: `IntermediateDataStore._compute_move_win_rates`
     (`src/maou/domain/data/intermediate_store.py`)
2. **`--drop-below-threshold`** (デフォルト: `False`)
   - 出現回数(`count`)が `position_count_threshold` 未満の局面を，
     フォールバック値を記録する代わりに出力から完全に除外する．
   - `--best-move-win-rate-fallback` より優先される(除外された局面には
     フォールバック計算自体が行われない)．
   - 実装: `IntermediateDataStore._finalize_chunk`
     (`src/maou/domain/data/intermediate_store.py`)
   - 既存の `fallback_positions` レポート(CLI出力JSON)は，フォールバック
     適用数と除外数の合計(=閾値未満で影響を受けた局面数)を報告するよう
     意味を拡張した．

いずれもオプトイン(デフォルト値は既存動作を保持)であり，ADR-005が
採用した「均等配分(1/N) / 固定値0.5」というデフォルトの決定そのものは
覆していない．

## ドキュメント変更内容(本レビューの承認対象)

### `docs/commands/pre_process.md`

「Intermediate caching & workers」テーブルに以下の2行を追加:

```
| `--best-move-win-rate-fallback {uniform,raw-outcome}` | `uniform` | 閾値未満局面での `bestMoveWinRate` 算出方法．`uniform`(デフォルト)は固定値0.5．`raw-outcome` は平滑化なしの実勝敗(`win_values/label_values` の最大)をそのまま使い，少数サンプルでは0.0/1.0等の極端値になり得る．`moveWinRate` 配列自体は本オプションに関わらず常に均等配分(1/N)のまま．`--drop-below-threshold` 指定時は無視される．【F:src/maou/infra/console/pre_process.py†L269-L279】 |
| `--drop-below-threshold` | `False`(フラグ) | 出現回数(`count`)が `--position-count-threshold` 未満の局面を出力から完全に除外する．フォールバック値そのものを記録したくない場合に使う．`--best-move-win-rate-fallback` より優先される．【F:src/maou/infra/console/pre_process.py†L280-L288】 |
```

### `docs/adr-005-move-win-rates.md`

「フォールバック値の選択理由」節の直後に，2026-07-20時点の運用上の追記として
以下を追加(ADR本体の決定事項は変更しない，事後の運用知見の追記):

```
#### 追記 (2026-07-20): フォールバック多発時のオプトイン代替策

実データ(floodgate棋譜)では局面の重複が想定より少なく，
`--position-count-threshold` を2〜3に設定してもほぼ全局面がフォールバック
してしまい，`bestMoveWinRate` の学習信号が実質消失する事例が確認された
(evaluateコマンドでの初期局面評価値異常として顕在化)．

上記「均等配分(1/N) / 固定値0.5」というデフォルトの決定自体は維持しつつ，
必要な場合にのみ以下のオプトイン代替策を選択できるようにした:

- `--best-move-win-rate-fallback raw-outcome`: `bestMoveWinRate` を
  平滑化なしの実勝敗で記録する(0.0/1.0等の極端値を許容し，情報の消失を防ぐ)．
  `moveWinRate` 配列(policy側)は対象外．
- `--drop-below-threshold`: 閾値未満の局面を出力から除外する．

両オプションともデフォルトはFalse相当(既存動作維持)であり，
新たな実験・データ生成のたびに明示的な選択が必要．
```

## リスクと理由

- **risk: low** — 両オプションともデフォルトは既存動作(ADR-005決定)を
  完全に保持するopt-inフラグであり，既存の再現性ある実験結果や
  デフォルト挙動には一切影響しない．
- **reversibility: easy** — ドキュメントの追記のみで，コード側の
  ロジック自体も独立した分岐として実装済み(既存パスへの変更は
  「bestMoveWinRateの値」のみで，`moveWinRate` 配列や `resultValue` の
  計算式は無変更)．

## ロールバック

ドキュメント追記を取り消すのみ．コード側もCLIオプションを未指定にすれば
既存デフォルト動作(`uniform`, `drop_below_threshold=False`)に完全復帰する．
