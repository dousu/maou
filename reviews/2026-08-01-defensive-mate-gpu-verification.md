---
title: 受け方向詰み探索の GPU 検証手順を verification.md に追加する
date: 2026-08-01
status: pending
target:
  - docs/design/usi-engine/verification.md
risk: low
reversibility: trivial
---

# 提案: §4.7 に受け方向詰み探索 (`--ab-mode defmate`) の検証手順を追加する

## Trigger

user 指示「Colab の L4 で GPU 環境確認を実施するので，検証方法を提示して
ほしい」．PR #424 (`8412f48`) でマージした受け方向詰み探索の GPU 検証．

CPU 側の全数走査は済んでいる (gate 該当 7.6% / 30 局中 7 局に避けられた敗着)．
GPU で確かめるのは **(a) 機構が実機で発火するか / (b) CPU 競合で探索速度を
奪わないか / (c) 棋力に効くか** の 3 点．

## 前提 — 計器

発火量を数えられなければ「効果なし」と結論してはいけない (計測規律)．
`SearchStats` に 2 つのカウンタを追加した:

- `defensive_mates` — 受け方向が「手番側が詰まされる」と証明した回数
  (root + 王手中の葉)．**0 なら機構が発火していない**
- `filtered_root_moves` — 敗着として除外した root 候補手の数．
  **0 なら着手選択に影響していない**

`maou search` の出力行と PyO3 の `PySearchResult` に出る
(`maou analyze-game` も同じ `SearchEngine` 経由なので棋譜の再解析で拾える)．

---

# 検証手順 (本レビューの追加対象)

環境は §1 のとおり．以下は共通の GPU 引数とする:

```
GPU="--threads 1 --batch-size 64 --tensorrt --cuda --trt-cache-dir trt_cache/"
MODEL="--model-path model_20260725_044443_vit-19.8m_32_fp16.onnx"
```

## Stage A — 機構が実機で発火するか (3 局面 / 数分)

自己対局で実際に踏んだ局面を pin してある．**期待値まで一致すること**を見る．

### A-1 敗着局面 (まだ受かる / 合法手 4)

```
maou search --sfen 'l8/3k1s3/2nppp3/1lG4p1/p1p1+R4/P1P1NR3/1G1PPg3/1S2K1+b2/LN7 b BSgsnl8p 117' \
  --time-ms 11000 $MODEL $GPU
```

| 期待 | 値 |
|---|---|
| bestmove | **5h6i** (唯一の受かる手) |
| 候補一覧 | 残り 3 手すべてに `proven=loss` (4f4g / 5h5i / 5h6h) |
| `filtered_root_moves` | **1 以上** |
| `defensive_mates` | **1 以上** |

**`filtered_root_moves` が 3 未満でも異常ではない** — 木の proven 伝播
(leaf-mate 経由) が先に確定させた手はフィルタが上書きしないため，内訳は
実行ごとに動く．**見るべきは「除外が 3 手」と「bestmove = 5h6i」**．

対照 (`--no-defensive-mate`): `filtered_root_moves=0`．対局では 4f4g を選び
11 手後に詰まされた．**bestmove が変わることがこの機構の存在理由**なので，
両者で同じ手が出た場合は発火していないか gate に掛かっていないかを疑う．

DevContainer / mock 評価器での実測 (参考値．GPU では内訳が動く):

```
Bestmove: 5h6i
5h6i (visits=990,    winrate=0.4846, proven なし)
5h6h (visits=541509, winrate=0.4987, proven=loss)
4f4g (visits=482872, winrate=0.4988, proven=loss)   ← 対局で選んだ敗着
5h5i (visits=1744,   winrate=0.1197, proven=loss)
Stats: ... leaf_mates=6439 defensive_mates=19 filtered_root_moves=2 ...
```

**visits では 5h6h / 4f4g が 50 万回で圧倒的なのに 990 回の 5h6i が選ばれる**
— MCTS の訪問分布では敗着が勝っており，確定値の除外だけが手を変えている．

### A-2 被詰み局面 (もう詰んでいる / 合法手 3)

```
maou search --sfen 'l8/3k1s3/2nppp3/1lG4p1/p1p1+R4/P1P1N4/1G1PPR3/1S2K1+b2/LN1s5 b BGSgnl8p 119' \
  --time-ms 11000 $MODEL $GPU
```

| 期待 | 値 |
|---|---|
| `defensive_mates` | **1 以上** |
| `WinRate` | **0.0000** (対照は 0.49 前後の互角) |
| bestmove | 5h6i (最長抵抗 = mate-40) |
| `filtered_root_moves` | **0** — 既に詰んでいるので候補選別はしない (設計どおり) |

**ここが「相手の手番だけ詰みが見えて自分の手番では互角を示す」の直接の反証**．
DevContainer / mock 実測: `WinRate: 0.0000` / `defensive_mates=50` /
`filtered_root_moves=0`．

### A-3 静かな局面 (偽陽性ゼロと無害性)

```
maou search --sfen 'lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1' \
  --time-ms 11000 $MODEL $GPU
```

期待: `defensive_mates=0` かつ `filtered_root_moves=0`．
**偽の被詰みは投了・自滅につながるので，ここが 0 でなければ即調査**．
DevContainer / mock 実測: `WinRate: 0.5000` / 両カウンタ 0．

## Stage B — CPU 競合で探索速度を奪わないか

Stage A-3 と同じ静かな局面で `--defensive-mate` / `--no-defensive-mate` の
`nps` を比べる．GPU 律速なら差は小さいはず．

先に `nproc` で vCPU 数を確認する — 常駐スレッドは
MCTS 1 + 攻め dfpn 1 + 受け dfpn 1 + leaf-mate 1 = **4 本**になる．
`--defensive-mate-threads` を上げる場合はさらに増える．

| 判定 | 基準 |
|---|---|
| 許容 | nps 低下 5% 未満 |
| 要調査 | 5% 以上 → `--defensive-mate-threads 1` のまま A/B へ (並列度を上げない) |

**gate 該当が全局面の 7.6% (終盤 21.6%) なので，静かな局面での低下は
そのまま「常時払う税」になる**．ここは終盤の局面ではなく静かな局面で測る．

## Stage C — 棋力 A/B (`--ab-mode defmate`)

```
maou selfplay --ab-mode defmate --clock-ms 300000 --inc-ms 10000 \
  --games 40 --parallel 1 $MODEL $GPU \
  --output ab_defmate.jsonl --kifu-dir kifu_defmate/
```

regime ゲート (先に通すこと):

- **持ち時間モード必須** — 固定 playout 予算では敗着フィルタが使える壁時計が
  変わってしまう (フィルタは MCTS 終了の停止フラグで打ち切られる)
- **`--parallel 1` 必須** — フィルタが CPU を使うので同時対局は互いを歪める
- **GPU 必須** — CPU 22 p/s では手の分布が別物になる

**期待効果の見積り**: CPU 全数走査では 30 局中 7 局 (23%) に「避けられた
強制詰みの敗着」があった．1 局あたり平均 0.23 手の改善であり，
**n=40 が検出できる ~150 Elo 級に届くかは未知**．有意にならなくても
機構が効いていないことにはならない — だから Stage D を併せて見る．

## Stage D — 棋譜による直接確認 (Elo より感度が高い)

`--kifu-dir` の CSA を CPU 側と同じ走査に掛け，**A 側の敗着回数が B 側より
減っているか**を見る．これが最も直接的な効果指標:

1. 各局面で合法手 ≤ 16 を gate
2. 各合法手を指した後の局面を攻め方向 dfpn (`find_shortest=false`) で判定
3. 「指した手が敗着 かつ 安全な代替手が存在した」件数を A/B で比較

判定: **A 側の件数が B 側より少なければ機構は効いている**．Elo が有意で
なくても，この差は直接観測できる．

## 記録先

結果は §9 のとおり worklog + compass へ．A/B が有意でなかった場合も
**「発火したが Elo 差は検出限界未満」と「発火しなかった」を必ず区別して
書くこと** — 前者は追試の対象，後者は実装のバグである．
