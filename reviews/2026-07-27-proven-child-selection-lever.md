---
status: pending
applied_in:
date: 2026-07-27
target:
  - docs/commands/selfplay.md
  - docs/design/usi-engine/verification.md
risk: low
reversibility: trivial
---

# 提案: 確定済み子の選択除外レバー (`--skip-proven` / `--ab-mode proven`) を docs へ反映する

## Trigger

空回りの予算開放 (`--spin-relief`) は実 playout を +1.34% しか増やさず棄却した
(verification.md §4.5)．そこで**会計ではなく選択**を変えるレバーを実装した:
確定済み (proven) の子を PUCT の候補から外し，全子が確定した親をその場で
確定化して畳む (MCTS-Solver 相当)．降下自体が消えるので**固定予算でも持ち時間
モードでも効く**．

CPU 実測 (mock / 千日手終端が支配的な局面 / 400 playouts/手，**1 手あたり**):

| | 実 playout/手 | 空回り/手 | throughput |
|---|---|---|---|
| off | 312 | 91 (予算の 22.6%) | 1,039 playouts/秒 |
| on | **376 (+20.5%)** | **27 (-70%)** | **1,244 playouts/秒 (+20%)** |

前レバーと違い**空回りが実探索に転換している**．A/B に値する．

## ドキュメント変更内容 (本レビューの承認対象)

### (a) `docs/commands/selfplay.md` — CLI オプション表に 1 行 + ab-mode 追記

| Flag | Required | Description |
| --- | --- | --- |
| `--skip-proven/--no-skip-proven` | default off | Exclude proven children (mate / repetition / resolved subtrees) from PUCT selection so descents open new leaves instead of backpropagating a known value (MCTS-Solver). Works under both fixed budgets and the real clock. |

`--ab-mode` へ `proven` を追加 (A = on / B = off)．

### (b) `docs/design/usi-engine/verification.md` — §4.6 として新設

> ## 4.6 確定済み子の選択除外 (`--ab-mode proven`) — GPU A/B 待ち
>
> §4.5 で棄却した予算開放の代わりに，**選択**側で空回りを消すレバー．確定済み
> (詰み・千日手・確定伝播済み) の子を PUCT の候補から外し，全子が確定した親を
> その場で確定化する．会計と違い**降下そのものが消える**ので，時計が拘束条件の
> 持ち時間モードでも効く．
>
> **CPU 発火確認 (mock / 千日手終端が支配的な局面 / 400 playouts/手)**: 1 手
> あたり実 playout 312 → **376 (+20.5%)**，空回り 91 → **27 (-70%)**，
> throughput 1,039 → **1,244 playouts/秒**．§4.5 のレバーと違い空回りが実探索へ
> 転換している．
>
> **注意 — 効く局面と効かない局面がある**: 空回りの源が
> **(a) 確定済み終端 (詰み・千日手)** なら効く．
> **(b) 深さ上限・最大手数の超過**なら効かない — この打ち切りは
> `mark_terminal` しない (reroot で深さが変わると stale になるため) ので確定化
> できず，候補から外せない．実測でも平手 40 手・`--max-moves 40` の局面 (空回り
> の大半が (b)) では実 playout が **1 件も変わらなかった**．GPU 実測の 98% 空回り
> は終盤の詰み・千日手由来 (`--max-moves 512` に対し終局 105 手) なので (a) に
> 当たると考えられるが，**未確認**．
>
> ### 手順
>
> 1. **発火量 (A/B より先)**: レバー on/off の**素の 2 run** を比較する
>    (`--ab-mode` はサマリが A/B 両者を合算するので発火量が読めない)．
>
>    ```python
>    for f in ("", "--skip-proven"):
>        !maou selfplay --games 4 --playouts 800 --max-moves 512 \
>            --opening-random-plies 8 --seed 1 {f} \
>            --model-path /content/model_fp16.onnx --tensorrt --cuda \
>            --threads 1 --batch-size 256 --trt-cache-dir /content/trt_cache
>    ```
>
>    **1 手あたりに直して比べる** (`playouts ÷ plies`)．局数・手数が変わるため
>    総量は比較にならない．実 playout/手 が有意に増えていなければ，その分布では
>    空回りが (b) 由来なので A/B を回さない．
> 2. **棋力 A/B** (発火が確認できた場合のみ):
>
>    ```python
>    !maou selfplay --games 40 --ab-mode proven --playouts 800 \
>        --resign-value 0 --max-moves 512 --opening-random-plies 8 --seed 1 \
>        --model-path /content/model_fp16.onnx --tensorrt --cuda \
>        --threads 1 --batch-size 256 --trt-cache-dir /content/trt_cache \
>        --output /content/ab_proven.jsonl
>    ```
>
>    判定は §4.3 と同じ (paired の平均と t 値を第一に，n=40 の検出限界 ~150 Elo を
>    踏まえる)．§4.5 の換算 (1 doubling ≈ 60 Elo) で発火量から期待値を出し，
>    符号と桁が整合するかを見る．
> 3. 有意に強ければ既定 on 化 + USI option 化を別 PR で起票する．
>
> ### 着手選択への副作用 (確認済み)
>
> 確定済みの子は訪問が伸びないため robust child では選ばれなくなる．そのため
> 有効時は `best_root_index` が**確定値で上書き判定**する (確定値 > robust child
> の推定 q なら確定側)．「確実な引き分け」を「不確実な劣勢」と取り違えないため
> で，逆に推定が引き分けより良ければ上書きしない
> (`test_skip_proven_children_prefers_sure_draw_over_worse_guess` /
> `..._keeps_better_unproven_move` で pin)．千日手模様の指し手選択が変わるので，
> **棋力 A/B の前に千日手判定の回帰 (`reasons` の `repetition` 件数) も見ること**．

## 代替案と棄却理由

- **深さ上限超過も `mark_terminal` する**: 保留 (別 PR)．`max_ply` 由来は reroot で
  深さが変わり stale になるため不可．絶対手数基準の `max_moves_to_draw` は
  reroot 不変なので確定化できる可能性があるが，subtree 再利用との相互作用の
  検証が要る．本 PR の範囲外．
- **既定 on で出す**: 棄却．着手選択 (特に千日手模様) が変わる．A/B が先．
- **`best_root_index` を変えない**: 棄却．確定済みの子の訪問が伸びなくなる以上，
  robust child だけでは確実な引き分けを取りこぼす．レバー on のときだけ
  上書きするので既定経路は bit-identical．

## リスクと理由

- **risk: low** — 既定 off の計測用トグル．既定経路は不変
  (`skip_proven_children = false` で選択も着手選択も従来と同一)．
- **reversibility: trivial** — フラグと `AbMode::Proven` を削るだけ．

## ロールバック

`--skip-proven` / `--ab-mode proven` / `SearchOptions::skip_proven_children` と
`best_root_index` の上書き分岐を削除し，docs の該当記述を戻す．
