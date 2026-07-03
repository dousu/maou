# 合駒最適化

合駒 (特に連続中合い) は詰将棋ソルバーの主要なボトルネックである．飛び駒 (飛・角・香) による
遠距離王手に対し，玉と飛び駒の間のマスへ駒を打つ防御 (中合い) のうち，飛び駒がその合駒を
取り進んで再び王手となりさらに合駒できる再帰構造を持つものがある．n マスのチェーンで各マス
k 種の合駒が可能なら最悪 O(k^n) の分岐が生じる．

### 8.1 中合い遅延展開 (DelayedMoveList)

**出典:** KomoringHeights v0.5.0 (合駒の遅延展開)

合駒を即座に全展開すると，AND ノードの δ (= Σ 子 φ) が合駒の数だけ過大評価され，探索が
合駒方向へ過剰に誘導される．**合駒を遅延させ，必要になるまで活性化しない**ことでこれを抑える．

**実装:** `DelayedMoveList` (`movegen/delayed_move_list.rs`)．parity 版 (成/不成ペアの
chain 化) を含め常時有効．

- **遅延対象**:
  - AND ノード (守備): 全ての駒打ち (合駒) を遅延可能とする．
  - OR ノード (攻め): 成/不成ペアを遅延可能とする (parity DML)．
- **chain (双方向リスト)**: `prev[i]` / `next[i]` で同マス・対称な手を連結する．
  - 同一着手先 (同じ中合いマス) の手を chain 化する．
  - 王手でない通常の合駒打ち同士を chain 化する (**中合い対称性**)．
- **semantics** (`has_unresolved_prev`): 子 `i` を展開する際，`prev` chain に未解決の hand が
  あれば `i` を skip し次反復へ繰り延べる．`update_best_child` で繰り延べた子を復活させる．
- **parity DML の健全性**: 成/不成ペアの chain 化は深い局面での depth-limit 偽反証による
  TT 汚染 (false NoMate) を抑える補助になる．その根治は反証の scope 化
  ([loop-ghi §7.2](loop-ghi.md)) が担い，DML は guidance の頑健性改善として併用する．

```mermaid
flowchart TD
    A[AND ノード: 応手生成] --> B{手の分類}
    B -->|玉移動・駒移動| C[即展開]
    B -->|合駒 drop = 中合い| D[DelayedMoveList へ chain 化]
    C --> E{非合駒で反証?}
    E -->|yes| F[AND 反証 — 合駒展開を回避]
    E -->|no| G[chain head から 1 手ずつ活性化]
    G --> H{prev 未解決?}
    H -->|yes| I[skip し次反復へ繰延]
    H -->|no| J[展開・探索]
    I --> G
    J --> G
```

非合駒の応手で反証できれば合駒の展開自体を省け，逐次活性化で不要な分岐を抑える．

### 8.2 無駄合いの除外 (最短手数のため既定 on)

無駄合い (取られて終わる無意味な中合い) は詰将棋規約上**手数に数えない**ため，最短手数を正しく
求めるには手生成段階で除外する必要がある．`compute_futile_and_chain_squares` が合駒マスを
futile / chain / normal に分類し，`generate_interpositions` が **futile マスへの駒打ちを skip**
する (駒移動による合駒は盤上駒の relocation = 無駄合い対象外ゆえ除外しない)．

**無駄合いの正確な定義 (誤りやすい)**: 中合いが無駄合いになる条件は「守備の支えがない」だけでは
**不十分**で，**玉がどこへ逃げても同じ手数以下で詰む (中合いしても詰み手数が変わらない)** ことが
必須である．単に支えなしで除外すると，**中合いで攻め方の駒をずらす (取らせて利きを変える) 正当な
受けテクニック**を誤って消す．`king_can_escape_after_slider_capture` が「飛び駒が合駒を取り進んだ
後に玉が逃げられるか」を検査してこれを防ぐ (逃げられれば駒ずらしが効く = 無駄合いでない)．

futile skip は常時有効である．この filter を欠くと無駄合いが max-resistance に算入され
最短手数を過大評価する (例: tsume_4 は無駄合い込み 13 手 → 規約上の 11 手)．

**正解の oracle はユーザ**: 無駄合いの判定は微妙で，最短手数/正解 PV の最終確認先は常にユーザで
ある (KH MinLength は無駄合いを数えるため最短 oracle にならない; [move-ordering-and-pv §9-b.3]
(move-ordering-and-pv.md))．

同様に，同一マスへの異種合駒間で取り進み後の局面の証明/反証を転用する cross-deduction も
評価されたが，deep AND fan-out の主因を削らないため採用していない (現行コードに機構は無く，
採否計測の経緯は git 履歴に残る)．

### 8.3 持ち駒越境・dominance との連携

合駒探索で生じる「同一盤面・異なる持ち駒」の局面群は，TT の持ち駒優越と forward-chain 代替
([transposition-table.md §6.1-6.2](transposition-table.md)) で広く再利用される．また持ち駒の
多様性が指数爆発する局面は visit-history dominance ([loop-ghi §7.3](loop-ghi.md)) で枝刈り
される．これらが DML と併せて中合いチェーンの探索量を抑える．

旧版の Futile/Chain 3 分類・チェーンドロップ 3 カテゴリ制限・cross_deduce neighbor_scan・
reverse disproof sharing・refutable disproof は，二エンジン期に Dual TT 上で開発された機構で
あり統一 mid では DML + 持ち駒越境 + dominance に整理・代替された (記録は git 履歴)．

### 8.4 無駄合い-free len budget — find_shortest 探索側

§8.2 の post-pass は**証明済み PV の報告手数**を無駄合い抜きへ補正するが，それだけでは
`find_shortest` の **len 予算 (反復深化の tight-len 再探索) 自体**が無駄合い込みの raw ply を
数えてしまう．受け方が無駄合い (取り返される透過中合い) で raw 手数を膨らませると，攻め方が
真の最短 ≤L 手詰を持っていても raw 手数が L を超えて `build_expansion` の予算切れ cutoff
(`len < 1 手` → disprove) に触れ，**短い詰みを偽 disproof** してしまう (この不整合を残すと
39te で root=39 手詰を持ちながら len=37 以下を偽反証し root を 43 手と過大評価する)．

**対策 (無駄合い len credit)**: 透過中合いへの合駒 drop を len 予算から credit し，len 予算を
「無駄合い抜き手数」に一致させる．

- `transparent_interposition_squares` (`movegen/mod.rs`): AND ノードで飛び駒王手中のとき，
  `compute_futile_and_chain_squares` の **chain マス** (取り返される透過中合いの代表マス) を返す
  (OR ノード / 非王手 / 非飛び駒王手では空)．
- `child_len` (`search/mod.rs`): 子局面へ渡す len 予算を計算する．AND ノードで **chain マスへの
  合駒 drop** 子は `len.add(1)` (通常は `len.sub(1)`)．直後の攻め方の取り返し (`len.sub(1)`) と
  相殺し，**合駒+取り返しの 2 手 pair の len コストを 0** にする (= 無駄合いは len を消費しない)．
  `step_best_child` と `build_expansion` の seed・look-ahead 双方で同一値を使う．
- `current_result` (`search/expansion.rs`): AND-proven の mate_len 集計 (max-resistance) で，
  chain drop 子は `r.len().sub(2)` (末尾 `+1` 後に `r.len()-1` = 取り返し後局面の手数)．
  → AND ノードの詰み手数が無駄合いで膨らまず `find_shortest` が真の最短へ収束する．build 時の
  chain マスは `LocalExpansion.chain_sqs` に保持する．
- **`len < DEPTH_MAX` のときのみ発火**する (tight-len 再探索でのみ len を credit)．first-mate 探索
  (`find_shortest=false`, len=DEPTH_MAX) では発火せず，canonical first-mate anchor
  (29te 9,288 / 39te 4,272,957 node) は不変である．

**探索側 (len 予算) と post-search 側 (post-pass) の役割分担**: 「無駄合い除外を探索側の
`proven_len` を discount して行う」案は，`find_shortest` の len 予算 (actual ply) と衝突し無駄合い
込み actual line を budget-cut して偽 disproof を生むため成立しない．本方式はこれと別物で，
`proven_len` を直接改変せず **len 予算 (入力側) の credit と mate_len 集計 (出力側) の除外を両方向
consistent** に行うため単位衝突を起こさない．

**perf 特性**: 無駄合いを正しく展開するため，`find_shortest` では最短確定のための反証 pass
(len=d−2 の不存在証明) が探索の支配項になる (39te で総ノードの ~90%)．無駄合いを**刈る**
(探索しない) 案は，futile filter が chain 代表合駒を**本物のチェーン受けの起点として残している**
ため刈ると偽詰みを生む (unsound) → 不可．透過中合いの取り返しへ collapse する高速化案は計測の
結果 perf 中立で不採用となった．
