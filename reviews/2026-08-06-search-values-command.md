---
title: 探索由来の value 教師を作る maou utility search-values を追加する
date: 2026-08-06
status: applied
applied_in: e997e03
target:
  - docs/commands/utility_search_values.md
  - docs/commands/pre_process.md
risk: low
reversibility: easy
---

# 提案: `maou utility search-values` と `pre-process --search-value-path` のドキュメントを追加する

## 背景

`docs/design/training-quality/index.md` §3 Step 3 の **B 案** (floodgate 局面を
直接探索して `resultValue` だけ差し替える) を実装した (user 判断, 2026-08-06)．

狙いは情報量ではなく**機序を壊す**ことである．現在の value 教師は
1 対局の約 110 局面すべてで同じ 0/1 なので「どの対局か」を思い出せば
学習データには当たるが，未知の対局では 1 ビットも稼げない．
記憶の痕跡が手数とともに増える実測 (Brier 比 ply 0-19 で 1.08，
ply 120+ で 2.31 — §5.3) はこの機序と整合する．
探索値は同一対局の中でも局面ごとに異なるので近道が効かなくなる．

B を選んだ理由 (A = 自己対局に対して):

- **分布シフトが無い** — 局面も policy 教師も floodgate のまま
- **反証が速い** — 1 回の学習で「value の過学習開始が epoch 11 より
  後ろへ動くか」を見れば済む
- **maou は現状 floodgate 集団より弱い** — policy 教師を自己対局に
  置き換えるのは自己蒸留のバイアスを負う

## 実装 (src/) — Rust 変更なし

`maou._rust.maou_search.SearchEngine` が既に Python へ出ており
(`winrate` / GPU フラグ / モデル 1 回ロード)，`PyBoard.set_hcp()` →
`.sfen()` で HCPE から SFEN も作れるため，**Rust 側の変更は不要**だった．
§3.2 の 3 段貫通は `maou_usi` 経由の自己対局のための作業であり，
B では迂回できる．

| 追加 | 内容 |
|---|---|
| `maou utility search-values` | HCPE の局面を探索し `(id, searchWinRate, playouts, stop)` を feather へ書く．`id` は Zobrist hash で前処理出力の `id` と同じキー |
| `maou pre-process --search-value-path` | 該当局面の `resultValue` を探索値へ差し替える．無い局面は対局結果のまま残るので**部分適用できる** |

設計上の要点:

- **選定はラベルと独立** — `--min-ply` (既定 60) と hash 重複のみ．
  「モデルが外している局面を選ぶ」は学習分布をモデルの誤りへ偏らせ，
  較正測定の前提を壊すので採らない
- **同一 hash は 1 回だけ探索** — 前処理は hash で集約するので 2 回目以降は無駄
- **`--flush-interval` (既定 500) と `--resume`** — 数十万局面を数日かけて
  回すので，最後にしか書かないと中断で全損する．定期的に書き出し，
  再実行で続きから再開する
- **走査は hash だけ集める** — SFEN まで先に作ると数百万局面で GB 単位の
  メモリを食うため，SFEN は探索の直前に作る
- **進捗は tqdm** — 走査と探索の 2 段階に進捗バーを出す
  (ETA / 局面毎秒 / 直近のフラッシュ位置 / 探索勝率の移動平均)
- `SearchEngine.search()` は `node_capacity` を受け取らない
  (コンストラクタにも無い) ため，該当オプションは提供しない

## ドキュメント変更

### 1. `docs/commands/utility_search_values.md` (新規)

Overview (なぜ要るか / 選定がラベルと独立であること / 部分適用 /
千日手履歴が失われる注意) + Usage + CLI options 表 + 出力スキーマ + コスト表．

### 2. `docs/commands/pre_process.md`

`--search-value-path` の行を追加．

## 検証

- **単体**: 選定 (`select_positions`) と部分適用 (`apply_search_values`) を
  14 ケースで固定．特に「探索できた局面だけ差し替わり残りは対局結果のまま」を
  明示的に固定した (部分実行が使い物にならなくなる回帰を防ぐ)
- **実データ**: floodgate 878 局の HCPE で mock 評価器と実モデル (ONNX) の
  両方を通し，`--resume` の往復 (25 + 12 = 37 行，id 重複なし) と
  途中フラッシュを確認
- **round-trip**: `pre-process --search-value-path` で
  **対象 40 行だけが探索値へ変わり，対象外は完全に一致**することを実データで確認
  (スキーマ・行数も不変)
- **GPU での検証手順**は `docs/commands/utility_search_values.md` の Usage 節

## リスク

- **低**: 既存の挙動は変わらない．`--search-value-path` 未指定なら
  前処理は従来どおり
- **中 / 運用**: 探索値は maou 自身の推定なので，モデルが弱い局面では
  弱い教師になる (自己蒸留)．対局結果という外部の真値を捨てる面がある．
  ply < 60 は対局結果のまま残るので序盤の外部信号は保たれる
- **未検証**: 効果 (held-out ECE / value の過学習開始エポック) は
  GPU で探索値を作って再学習するまで分からない
- 版数は `maou 0.79.0` → `0.80.0` (CLI コマンド追加 = feat 相当)
