---
title: 最終手選択のタイブレークに policy 事前確率を加える
date: 2026-07-29
status: pending
target:
  - docs/design/position-search/index.md
risk: low
reversibility: easy
---

# 提案: §6 最終手選択の基準に prior タイブレークを追記する

## Trigger

user 指示「playouts=0 のとき prior 最大手にフォールバックするを導入して
ください」．

実機で踏んだ事象 (Colab L4 / TensorRT cold cache):

```
Bestmove: 1g1f
Candidates:
1g1f (visits=0, prior=0.0610)   ← これが選ばれた
1i1h (visits=0, prior=0.0014)
2g2f (visits=0, prior=0.4135)   ← policy 最大
...
```

`select_best_root_index` (`search.rs:349`) は `prior` を**一切参照していな
かった**．全子が `visits = 0` / `q = 0.0` のとき比較式が常に false になり，
`best_i` が 0 のまま = 合法手生成順の先頭が返る．`RootChildStat.prior`
(`search.rs:336`) はフィールドとして存在するが選択ロジックからは死んでいた．

## 変更内容 (実装済み — maou_search v0.29.0)

タイブレークの最後段，生成順の**手前**に prior を挟む:

```
負け確定でない > visits 最大 > Q 最大 > prior 最大 > 合法手生成順の先頭
```

`skip_proven_children` 有効時の確定値上書き (`best_root_index`,
`search.rs:412-432`) と root 確定時の子選択は不変．

## 根拠

- **prior は探索が無くても NN から得られている唯一の情報**．訪問数と Q で
  差が付かないときにこれを無視して生成順に落ちる理由がない．
- **主要エンジンの順序と一致する** — 設計 doc §6 が既に記録しているとおり
  lc0 は `terminal rank → N → Q → P`，dlshogi は
  `proven → move_count → nnrate`．どちらも policy を最終タイブレークに使う．
  つまり本変更は §6 の調査結果に実装を**寄せる**もので，新しい設計判断を
  持ち込んではいない．
- **健全性規則より下位に置いた**．負け確定の除外は prior より優先する
  (policy 最大でも負け確定なら選ばない)．

## 影響範囲

- **playout がある通常の探索では挙動が変わらない**．visits と Q が完全同値に
  なるのは (a) 全子未訪問 (b) 探索が極端に浅い の 2 ケースのみ．
- §5.1 の warmup 前払いで「TensorRT のエンジンビルドが予算を食い切る」主要因は
  解消済み．本変更が効くのは**予算そのものが 1 バッチ未満**のとき (遅い機械の
  短い秒読み等) で，残っていた最後の穴を塞ぐ．
- USI / CSA / 自己対局 / 棋譜解析はいずれも `best_move` をそのまま使うため
  自動的に恩恵を受ける (`agent.rs:978` は投了判定のみで手の差し替えをしない)．

## 提案する記述 (§6 の項目 3 を差し替え + 理由の段落を追加)

> 3. それ以外 → **robust child**: 負け確定 (ルート視点 proven=0) の手を除外して
>    訪問回数最大 → 同数なら Q 最大 → 同率なら **policy 事前確率最大** → なお
>    同率なら合法手生成順で先頭 (`select_best_root_index`)．全手が負け確定なら
>    除外なしで同基準 (どれも同値)．
>
> **prior を最終タイブレークに使う理由** (maou_search v0.29.0)．予算が 1 バッチ
> にも満たないと playout が 1 回も回らず，全子が `visits = 0` / `q = 0.0` に
> なる．prior を見ないと生成順の先頭という**評価に基づかない手**が返る — 実測で
> prior 0.061 の手が prior 0.414 の手より優先された (Colab L4，TensorRT の
> エンジンビルドが予算を食い切ったケース．固定費の前払いは §5.1 で解消したが，
> 遅い機械の短い秒読みでは予算自体が 1 バッチ未満になり得る)．prior は探索が
> 無くても NN から得られている唯一の情報なので，訪問数と Q で差が付かない
> ときはこれに従う．主要エンジンの順序とも一致する — lc0 は
> `terminal rank → N → Q → P`，dlshogi は `proven → move_count → nnrate`．

## 検証

- 新規テスト 3 件 (`search.rs`):
  - `test_select_best_root_index_falls_back_to_prior_when_unvisited` —
    実測局面の再現 (生成順先頭が prior 0.061，本命が prior 0.414)
  - `test_prior_is_only_the_last_tiebreak` — visits / Q が prior に優先する
  - `test_prior_does_not_override_losing_exclusion` — 負け確定除外が最優先
- **canonical 29te / 39te を RAN して照合済み** (TRIPWIRE):
  29te **396,516** ノード / 39te **17,593,615** ノード — compass 記録値と一致．
