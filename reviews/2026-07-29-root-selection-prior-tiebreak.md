---
title: playout 0 のときの最終手選択と勝率報告を評価に基づくものにする
date: 2026-07-29
status: pending
target:
  - docs/design/position-search/index.md
risk: low
reversibility: easy
---

# 提案: §6 に prior タイブレークと playout 0 時の勝率を追記する

playout が 1 回も回らなかったときの報告が 2 箇所とも壊れていた．
**(A) 指し手**が評価に基づかない (生成順の先頭)，**(B) 勝率**が
「データなし」を「敗勢確定」として報告する．どちらも同じ根に由来するので
1 つの提案にまとめる．

---

# (A) §6 最終手選択の基準に prior タイブレークを追記する

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

---

# (B) playout 0 のときの勝率に root の NN value を返す

## Trigger

user 指摘「playout=0のときにEvalが見た目おかしな値になっていないですか？」

(A) の修正後も `Eval: -16578.61` / `WinRate: 0.0000` が出ていた．

## 原因

`collect_result` (`search.rs:1793`) が `winrate = root_proven.unwrap_or(best.q)`
としており，best が未訪問なら `best.q = 0.0`．**未訪問の Q が 0.0 なのは
「敗勢確定」ではなく「データなし」**だが，`winrate_to_eval(0.0)` は
クランプ飽和値 **-16578** を返す．「勝率 0 を確信した」ように見える．

root 局面は探索前に必ず NN 評価されている (`search.rs:1524` の
`evaluate_batch`) が，**その value は捨てられ priors だけが使われていた**．

Python 側は既にこの罠を認識していて，候補手表示では
「未訪問の winrate 0 は『データなし』であり敗勢ではないため数値を表示しない」
(`app/search/run.py:141`) と分岐している．同じ扱いがトップレベルの
`Eval` / `WinRate` に無かった．

## 表示だけの問題ではない

`agent.rs:991` の投了判定が `outcome.winrate < threshold` を見ている．
`resign_value` を有効にした対局で playout 0 の手番が来ると，
**勝率 0.0 として投了カウントが進む**．既定は `resign_value = 0` (投了しない)
なので現状は発現しないが，有効化した瞬間に踏む．

## 変更内容 (実装済み — maou_search v0.29.0)

- root 評価の value を `Shared::root_value: Option<f64>` に保持する
  (warm start = subtree 再利用では root を再評価しないため `None`)．
- `collect_result` と `Shared::build_snapshot` (探索中の `info` 用) の双方で，
  `best.visits == 0` のとき `best.q` でなく `root_value` を返す．
- 合法手なしの局面 (`search.rs:1488` の `winrate: 0.0`) は**本物の敗勢**
  なので変更しない．

## 提案する記述 (§6 の末尾に追記)

> **playout が 1 回も回らなかったときの勝率** (maou_search v0.29.0)．
> 予算が 1 バッチに満たないと best_move が未訪問のままになり Q が存在しない．
> 未訪問の Q は 0.0 だが，これは「敗勢確定」ではなく「データなし」であり，
> そのまま報告すると評価値が飽和値 (-16578) になって「勝率 0 を確信した」
> ように見え，USI の投了判定 (`resign_value`) まで誤らせる．root 局面は探索前に
> 必ず NN 評価されているので，**その value (= 探索抜きの評価，`maou evaluate`
> と同じ値) を返す**．探索できなかったときは 0 手読みの評価へ縮退する，
> という整理．合法手なしの局面は本物の敗勢なので 0.0 のままとする．

## 検証

- 新規テスト `test_zero_playout_reports_root_value_not_saturated_loss` —
  初回推論が予算を超える評価器で playout 0 を作り，`winrate > 0.0` かつ
  飽和端でないことを固定．
- 実測 (ViT19.8M fp16 / CPU): `maou search --time-ms 1` (playouts=0) が
  `Eval 279.93 / WinRate 0.6146` を返し，**`maou evaluate` と完全一致**．
  `--time-ms 3000` は `Eval 140.15 / playouts=60` で非回帰．
- canonical 29te/39te を再 RAN して照合済み (上と同値)．
