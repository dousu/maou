---
title: search-values のコスト表を実測へ訂正し，探索予算の knob を公開する
date: 2026-08-07
status: applied
applied_in: TBD
target:
  - docs/commands/utility_search_values.md
risk: low
reversibility: easy
---

# 提案: コスト表を実測値へ差し替え，未公開の探索予算 knob を CLI へ出す

## 背景

`docs/commands/utility_search_values.md` の Cost 節は
**公称 10,257 playouts/s を 800 で割った 0.078 秒/局面**を根拠に
「1M = 22 時間」と書いた．これは compass § TRIPWIRE
「**公称パラメータを信じない．実装を直接叩いて発火量を予測**」に反しており，
実測と食い違う．

user 実測:

| GPU | 実測 | 1 局面あたり |
|---|---|---|
| **G4** | 1M = 22 時間 | 0.079 秒 |
| **L4** | 300k = 18 時間 | **0.216 秒** |

**G4 が L4 の 2.7 倍速い**という結果になっており，
`compass § North-star` に「L4 batch 64 = 10,257 p/s」と記録されている値は
search-values の L4 実測と整合しない (どちらかのラベルか条件が誤っている)．

user 制約: **G4 は高価なので学習に温存し，search-values は安い GPU で回したい**．
したがって「なぜ遅いのか」を切り分ける価値がある．

## 実装の穴 (コード確認済み)

`SearchEngine` / `search()` が受け取るのに **CLI が出していない** knob がある．
そのため現状は原因の切り分けすらできない．

| knob | 受け取り先 | 既定 | CLI |
|---|---|---|---|
| `pad_buckets` | `SearchEngine.__init__` | False (= `batch_size` へ固定 padding) | **未公開** |
| `root_dfpn_nodes` | `search()` | **2,000,000** | 未公開 (on/off のみ) |
| `root_dfpn_depth` | `search()` | 2047 | 未公開 |
| `leaf_mate_nodes` / `leaf_mate_threads` | `search()` | 50 / 1 | 未公開 (on/off のみ) |
| `defensive_mate` / `defensive_mate_threads` | `search()` | Rust 既定 (root 敗着フィルタ) | **未公開** |

## 想定される律速 (未検証．A/B で切り分ける)

1. **CPU 側の詰み探索**．`--min-ply 60` は**戦術的に濃い局面を狙って選んでいる**
   ので dfpn の最悪ケースにあたる．`root_dfpn` の既定予算は 2,000,000 ノードで，
   さらに `defensive_mate` (root 敗着フィルタ) も走る．
   **これが支配的なら GPU の種類はほとんど効かず，vCPU 数の差が G4/L4 の差を
   説明する**．user の目的 (安い GPU で回す) にとって最良のシナリオ．
2. **batch が埋まらず padding が無駄になる**．TensorRT は既定で全評価バッチを
   `batch_size` (64) へ padding する．**この用途は 1 局面 1 探索で毎回 root から
   立ち上げる**ので序盤の playout は葉が少ない．compass の実測
   `cost(n) ≈ 0.15 + 0.084·n` ms より，1 件を 64 へ padding すると 5.5 ms／
   実 1 件なら 0.23 ms で **24 倍の無駄**．`pad_buckets=True` (2 冪バケット) が
   効く可能性があるが**未公開なので試せない**．
3. 木の再利用が効かない — B 案の構造的コスト．自己対局 (A 案) は手をまたいで
   subtree を再利用でき評価キューも埋まり続けるので，1 レコードあたりでは
   A の方が安い可能性がある (ただし A の 0.11 秒/レコードも同じ公称値からの
   計算なので**同様に信用できない**)．

## 変更内容

### 1. `docs/commands/utility_search_values.md`

- Cost 節を**実測値の表**へ差し替え (G4 / L4 の両方，1 局面あたりの秒数を明記)．
  公称 playouts/s からの割り算は**根拠にしない**と明記
- 「律速の切り分け」節を追加: 手元の出力から `stop` 分布と `playouts` 分布を
  見れば詰み探索が支配的かどうかが分かること，200 局面 A/B の手順
- 追加する knob (`--pad-buckets` / `--root-dfpn-nodes` / `--leaf-mate-nodes` /
  `--defensive-mate`) を CLI options 表へ追加

### 2. src/ (併せて実施)

上記 knob を CLI → interface → `SearchValueOption` → `SearchEngine`/`search()`
へ通す．**既定値は現行の挙動を保つ** (すべて Rust 既定へ委譲)．

## 検討したが採用しなかった案

- **詰み探索を既定で切る**: 詰みが証明された局面の探索値は**教師として最良**
  (0/1 の真値になる) なので，速度のためだけに既定を変えるのは
  compass § VETOES「レバーは A/B で確認してから既定化」に反する．
  knob を出して測れるようにするだけに留める
- **`playouts` を既定で下げる**: 教師の質が変わる．質とコストの曲線が未測定
- **複数局面を 1 つの評価キューへ混ぜる (`search_many`)**: batch 埋まらず問題の
  本質的な解だが **Rust の新機能**であり，まず律速を切り分けてから判断する

## リスク

- **低**: knob 追加は既定動作を変えない
- **未検証**: 上の律速仮説はどちらも未測定．A/B の結果次第で提案が変わる
- 版数は `maou 0.81.0` → `0.82.0` (CLI オプション追加 = feat 相当)
