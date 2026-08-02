---
title: 勝率フォールバックを勝率値にする修正のドキュメント反映 (1/N → 0.5 / 実勝敗)
date: 2026-08-03
status: pending
target:
  - docs/commands/pre_process.md
  - docs/adr-005-move-win-rates.md
risk: low
reversibility: easy
---

# 提案: `moveWinRate` フォールバックの意味変更をドキュメントに反映する

## 背景

`maou pre-process` は出現回数が `--position-count-threshold` 未満の局面に対し，
`moveWinRate` 配列へ `1/N` を書いていた (`N` = その局面で観測された指し手の
種類数)．ADR-005 §2 はこれを「合法手への均等配分」と説明していたが，実際の
`indices` は合法手ではなく**棋譜中で実際に指された手**である
(`hcpe_transform.py` の `unique_moves`)．

したがって**出現 1 回の局面では `N = 1` となり `moveWinRate = 1.0`** になる．

`moveWinRate` は `learn-model --value-target-mode best-move-win-rate` (既定) で
行ごとの最大値を取って value 教師になるため，**出現 1 回の局面はすべて
value target = 1.0 (勝敗によらず「勝ち」)** として学習されていた．
2026-07-20 の調査どおり実データではほぼ全局面が閾値未満になるため，影響は
学習データの全域に及ぶ．

`1/N` は分布としての正規化値であり，勝率としては意味を持たない．
フォールバック値が勝率の軸に載っていなかったことが原因である．

## 修正 (コードは適用済み: `4b4c62e`, maou 0.74.0 → 0.74.1)

フォールバック値を `--best-move-win-rate-fallback` に従わせた:

| モード | `moveWinRate` (修正前) | `moveWinRate` (修正後) | `bestMoveWinRate` |
|---|---|---|---|
| `uniform` (既定) | `1/N` | **`0.5`** (中立値) | `0.5` (変更なし) |
| `raw-outcome` | `1/N` | **実勝敗** (出現 1 回なら 0.0/1.0，引き分け 0.5) | 実勝敗 (変更なし) |

`bestMoveWinRate` は常に配列の最大値とし，配列とスカラーの意味を一致させた．
**スカラー側の値は両モードとも従来と同一**で，変わるのは配列のみ．
閾値以上の局面 (Beta 平滑化経路) は無変更．

実測 (既定 `--value-target-mode best-move-win-rate`，出現 1 回の局面):

```
                    修正前    uniform   raw-outcome
count=1 負け        1.0   →   0.5       0.0
count=1 勝ち        1.0   →   0.5       1.0
```

## ドキュメント変更内容

### 1. `docs/commands/pre_process.md` (現在 L49)

`--best-move-win-rate-fallback` の行から
「`moveWinRate` 配列自体は本オプションに関わらず常に均等配分(1/N)のまま」
を削除し，配列・スカラー双方に適用される旨と，`raw-outcome` では出現 1 回の
局面で 0.0/1.0 (引き分け 0.5) になる旨に差し替える．
`--position-count-threshold` の行の「均等配分」表現も同様に修正する．

### 2. `docs/adr-005-move-win-rates.md`

- §2 見出し「フォールバック戦略: 合法手への均等配分(方式B)」と本文 (L70-81) を
  改訂．`1/N` を採用した判断が「`indices` = 合法手」という誤った前提に
  立っていたこと，および `moveWinRate` が value 教師の元であるため
  フォールバック値は勝率の軸に載る必要があることを記す．
- 「フォールバック値の選択理由」表 (L85-90) を更新．`均等配分 (1/N)` を却下
  (理由: 勝率でない値が value 教師に流れる)，`固定値 0.5` を採用に変更する．
- 追記 (2026-07-20) の「`moveWinRate` 配列(policy側)は対象外」(L104) を
  現状に合わせて訂正する．

## 副作用 (要判断，本 review の対象外)

`--policy-target-mode win-rate` (既定) は `moveWinRate` を正規化して policy
教師にする．`raw-outcome` では**負けた対局由来の出現 1 回局面は全要素 0 に
なり，正規化後も行和 0**となる (`_safe_normalize` は合計 0 の行をそのまま
返す)．KLDivLoss はその行に対し損失を生まないため，**該当局面の policy 学習
信号が消える**．

- `uniform` では従来どおり行和 1 が保たれる (全要素 0.5 → 正規化で均等)．
- 回避策は `--policy-target-mode move-label` の併用．
- policy 側フォールバックの設計変更は本 review では扱わない．

## リスク

- **低**: 既定モード `uniform` でも value 教師が 1.0 → 0.5 に変わるため，
  **既存の前処理データを再生成しない限り挙動は変わらない**が，再生成後は
  学習結果が変わる．棋力への影響は未測定 (compass の「レバーは A/B で確認して
  から既定化」に該当するかは要判断 — バグ修正であってレバー追加ではない，
  という整理を提案する)．
- 逆行は容易 (フォールバック値を戻すだけ)．
