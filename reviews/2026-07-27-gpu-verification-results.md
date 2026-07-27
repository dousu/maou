---
title: GPU 検証の結果を docs へ反映 (未決 1/5 の決着 + verification.md の実測値・判定基準更新)
date: 2026-07-27
status: applied
applied_in: fda1ac4
target:
  - docs/design/usi-engine/index.md
  - docs/design/usi-engine/verification.md
risk: low
reversibility: easy
---

# 提案: GPU (Colab L4) 検証の結果を durable docs へ反映

## Trigger

worklog/2026-07-27-140516.md — Colab L4 で ①-④ を実測し，未決 1 (TimeStrategy
horizon) と未決 5 (バッチ aggregator) が決着した．同時に手順書
(`verification.md`) の前提が実測で覆った箇所がある (推奨スレッド数・regime
ゲート・`playouts` を指標に使えないこと)．**手順書が古い前提のまま残ると，
次に同じ検証をする人が同じ罠を踏む**ため反映する．

## ドキュメント変更内容 (本レビューの承認対象)

### `docs/design/usi-engine/index.md` §12 未決事項の表

- **未決 1 (TimeStrategy の定数)**: 「GPU 実測待ち」→ **決着**．
  「horizon 40 を据え置き．Colab L4 / ViT 19.8M で 40 vs 20 = +89 Elo
  (paired t=+1.75)，60 vs 40 = 40 側が +61 Elo (paired t=-2.10)．終局時の
  残り持ち時間は horizon 20/40/60 で 1.6s / 5.5-6.5s / 10.1s と単調に増え，
  配分が実際に効いていることを確認済み．40 vs 50 級の細かい調整は n≈400 局
  を要し未検証」
- **未決 5 (バッチ aggregator)**: 「GPU 実測待ち」→ **決着 (現状は見送り)**．
  「GPU でも並列自己対局は parallel 1/2/4/8 = 4.7k/5.6k/5.9k/5.9k playouts/秒
  と 1.26 倍で頭打ち．単発の長い探索は 10.9k 出るため律速は GPU 飽和ではなく
  バッチ充填で，対局をまたぐ aggregator には約 2 倍の伸びしろがある．
  実装は次 campaign の課題として起票し，現行構成では採用しない」
- 未決 2 (keep-alive 既定値) は GUI 実機待ちのまま変更しない．

### `docs/design/usi-engine/verification.md`

1. **§1-2 の推奨設定を実測値へ**: `--threads 2` → **`--threads 1`**
   (threads 2 はどの並列度でも約 4 割遅い．1 手 800 playouts 級の短い探索では
   探索内スレッド並列がオーバーヘッドになる)．実効 NPS の実測値を追記
   (単発 30s = 10,909 / 500ms = 8,101 playouts/秒)．
2. **§4.1 regime ゲートの改訂**: 「`総 playouts ÷ 総手数`」を判定から外し，
   **`timeout` の件数**と**終局時の残り持ち時間の A/B 差**に置き換える．
   理由: 引き分け終端・千日手・証明済み局面が近いと，探索は新しい葉を開かず
   終端 backprop だけを回すため `playouts` が同一 wall clock で 27k → 260k
   まで膨張し，探索量の指標にならない (実測)．
   **判定前に throughput が NN 評価の物理上限を超えていないか突き合わせる**
   注意書きを追加．
3. **§4 に `--max-moves 512` を必須として明記** (256 以下だと終盤が上記の
   膨張に汚染される)．`--clock-ms 30000 --inc-ms 500` を実測済みの推奨値に．
4. **§5 の決定規則に実測結果を追記** (1.26 倍で頭打ち → 採用検討側)．
5. **§8 GUI 実機検証は未実施のまま**変更しない．
6. 末尾に **既知の課題**節を新設: dfpn 偽証明アラート (`STRICT VERIFY None`)
   の頻度，終端再訪による探索の空回り，parallel=4 終了時のヒープ破壊 —
   いずれも worklog を参照先として 1 行ずつ．

## リスクと理由

- **risk: low** — ドキュメントのみ．コードの挙動変更は既にマージ済みの
  PR #403-#406 が持ち，本提案はその結果と実測値の記述反映．
- **reversibility: easy** — 2 ファイルの該当節の巻き戻しのみ．

## ロールバック

index.md §12 の 2 行と verification.md の該当節を元に戻す．
