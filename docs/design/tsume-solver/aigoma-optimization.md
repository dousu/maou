# 合駒最適化

チェーン合駒(連続合い駒)は詰将棋ソルバーの主要なボトルネックである．
飛び駒(飛車・角・香)による遠距離王手に対して，玉と飛び駒の間のマスに
駒を打つ(合駒)防御手のうち，飛び駒がその合駒を取り進むことで再び王手となり，
さらに合駒が可能になる再帰的構造を指す．
n マスのチェーンに対して各マスで k 種の合駒が可能な場合，最悪 O(k^n) の分岐が発生する．

### 8.1 Futile/Chain 合駒分類

合駒マス(between squares)を以下の3カテゴリに分類する．

```
  Rook check along rank (e.g. R on 8g checks King on 1g):

  R        between squares (7g..2g)            K
  8g   7g   6g   5g   4g   3g   2g           1g
  [R]--[C]--[C]--[C]--[C]--[N]--[F]--[K]
        ^                   ^    ^
        |                   |    futile: no defender support,
        |                   |            closer to K than breakpoint
        |                   normal (breakpoint):
        |                   defender has support here
        chain: no defender support,
               farther from K than breakpoint,
               R captures -> re-check -> more drops

  [C] = chain     max 3 drops (fwd/diag/knight)
  [N] = normal    all 7 piece types
  [F] = futile    skipped entirely
```

**実装:** `compute_futile_and_chain_squares` (pns.rs)

#### 通常マス (Normal)

守備側の利きが存在する，または玉に隣接し飛び駒が取り進んだ後に逃げ道がある場合．
全駒種(歩→香→桂→銀→金→角→飛)の合駒を生成する．

#### 無駄合いマス (Futile)

以下のすべてを満たすマス:
- 守備側(玉以外)の利きがない
- 玉に隣接していないか，隣接していても取り進み後に逃げ道がない
- ブレークポイント(通常マス)より玉側にある

無駄合いマスへの駒打ちは完全にスキップされる．

#### チェーンマス (Chain)

Futile の条件を満たすが，ブレークポイントより飛び駒側にあるマス．
飛び駒が取り進んだ後に再び王手となり，さらなる合駒が可能な再帰構造を生む．
チェーンマスへの合駒は3カテゴリの代表駒に限定される(§8.2)．

**補助関数:** `king_can_escape_after_slider_capture` (pns.rs)
飛び駒が合駒マスに取り進んだ状態をシミュレートし，玉の逃げ道を判定する．

### 8.2 チェーンドロップ3カテゴリ制限

チェーンマスへの駒打ちを以下の3カテゴリから各1手に制限する．

**実装:** `generate_chain_drops` (pns.rs)

| カテゴリ | 駒種 | 代表の選択 |
|---------|------|----------|
| 前方利き系 | 歩→香→銀→金→飛 | 最弱の合法駒1つ |
| 斜め利き系 | 角 | 角のみ |
| 跳躍系 | 桂 | 桂のみ |

**根拠:** 前方利き系では弱い駒で詰みが証明できれば強い駒でも証明できる
(攻め方が合駒を取った後，手に入る駒が強いほど詰ませやすい)．
角と桂は利きの方向が異なるため独立カテゴリとなる．

**効果:** 合駒マスあたりの駒打ち数を最大7手から最大3手に削減．

### 8.3 合駒遅延展開 (KomoringHeights v0.5.0)

**出典:** KomoringHeights v0.5.0

AND ノードの合駒(駒打ち)を即座に展開せず `deferred_children` に分離する．

```mermaid
flowchart TD
    A[AND node: generate defenses] --> B{classify moves}
    B -->|king moves, supported| C[children: expand immediately]
    B -->|drops = interpose| D[deferred_children: delay]

    C --> E{all children proved?}
    E -->|no: some child disproved| F[AND node disproved - done]
    E -->|yes| G[activate 1 deferred drop]

    G --> H{prefilter: TT proof?}
    H -->|yes| I[skip - already proved]
    H -->|no| J[add to children, search]
    I --> K{more deferred?}
    J --> K
    K -->|yes| G
    K -->|no| L[AND node proved]
```

**実装:** `mid()` 内の子ノード初期化 (`solver.rs`)，PNS の AND ノード展開 (`pns.rs`)

1. AND ノードの子を分類:
   - 駒移動(玉逃げ・紐付き合駒) → `children`(即座に展開)
   - 駒打ち(合駒) → `deferred_children`(遅延)
2. 非合駒応手を先に探索し，TT に証明を蓄積
3. `children` が空になったら `deferred_children` から1手ずつ活性化:
   弱い駒から順に活性化し，証明済み TT エントリを強い駒の探索で援用

**効果:** 非合駒応手で反証できれば合駒の展開自体を回避．
逐次活性化により不要な分岐を抑制．

### 8.4 TT ベース合駒プレフィルタ

合駒を `deferred_children` に追加する前に TT で証明済みか確認する．

**実装:** `try_prefilter_block` (`solver.rs`)

1. 合駒を盤上で実行
2. 攻め方の合法手から合駒マスへの捕獲かつ王手になる手を探索
3. 捕獲後の局面を TT で参照
4. pn=0(証明済み)なら合駒の OR ノードも証明 → 展開不要

**IDS との相乗効果:** 浅い IDS 反復でチェーン末端の証明が TT に蓄積され，
深い反復では浅いレベルの合駒がプレフィルタで即座にスキップされる．
これによりチェーン合駒がボトムアップに折り畳まれる．

### 8.5 同一マス証明転用

同一マスへの異なる駒種の合駒間で TT エントリを相互利用する．

**実装:** `cross_deduce_children` (`solver.rs`)

同一マス S への合駒 P1, P2, ..., Pn は，攻め方が捕獲した後の盤面(position_key)が
全て同一になる(異なるのは攻め方の持ち駒のみ)．

1. 合駒 i が証明済みになった後，同一マスの未解決合駒 j を列挙
2. 合駒 j の捕獲後の攻め方持ち駒を計算:
   `hand_j = base_hand - solved_piece + piece_j`
3. TT で捕獲後局面を参照: `look_up(pc_pk, &hand_j, pc_remaining)`
4. pn=0 なら合駒 j も証明 → `deferred_children` から除去

### 8.6 合駒 DN バイアス

AND ノードの合駒(駒打ち)の初期 dn にバイアスを加算し，探索優先度を下げる．

**実装:** 定数 `INTERPOSE_DN_BIAS = 8` (mod.rs)

非合駒応手の初期 dn(=1)より十分大きく設定し，
df-pn の自然な閾値制御で king move → drop の順序を実現する．
遅延展開(§8.3)が主要な制御手段であり，DN バイアスは補助的な役割．

### 8.7 チェーンマス内→外順序

チェーンマスの合駒を玉に近い側(内側)から飛び駒に近い側(外側)の順にソートする．

**実装:** `mid()` 内のチェーンドロップ順序付け (`solver.rs`)

```rust
deferred_children.sort_by_key(|(m, _, _, _)| {
    let to = m.to_sq();
    let dr = (to.row() - king_sq.row()).abs();
    let dc = (to.col() - king_sq.col()).abs();
    dr.max(dc)  // チェビシェフ距離: 内側(小)優先
});
```

- S1(内側)の証明で蓄積した TT エントリが S2(外側)の探索で再利用可能
- 短いサブチェーンから順に証明が蓄積され，長いサブチェーンの TT 再利用効率が向上

### 8.8 チェーン深さ DN スケーリング

チェーン合駒のみで構成される AND ノードで，DN バイアスをチェーン内位置に応じてスケーリングする．

**実装:** `mid()` 内の DN バイアス計算 (`solver.rs`)

```rust
let bias = if let Some(ksq) = chain_king_sq {
    let dist = chebyshev_distance(to, ksq);
    INTERPOSE_DN_BIAS * dist
} else {
    INTERPOSE_DN_BIAS
};
```

- 内側マス(d=1): `INTERPOSE_DN_BIAS × 1` — 最小バイアス，優先的に探索
- 外側マス(d=5): `INTERPOSE_DN_BIAS × 5` — 大きなバイアス，後回し
- `chain_king_sq` はチェーン判定時に保持した玉位置(チェーン AND ノードのみ)

§8.7(ソート順)と組み合わせることで相乗効果がある．

---

