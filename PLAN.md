# AND ノード合駒の事前証明 (Interposition Pre-Solve via df-pn)

## 背景

39手詰(6九合駒)で ply 2〜8 の OR サブ問題が 10M ノードでも UNKNOWN になる．
原因は ANDノードの合駒分岐(金・銀・飛・角)が各サブツリーで独立に証明を要求し，
指数的に膨張すること．

ply 10 以降は 349 ノード以下で瞬殺 → 合駒サブ問題は本質的に「簡単」だが，
df-pn の MID ループでの閾値管理・スラッシングにより効率的に解けていない．

## 設計思想: Simplest-First via Recursive df-pn

Lambda df-pn の「簡単な方から解く」原則を AND ノード子初期化で実現:

**各合駒の子局面(OR ノード = 攻め方番)に対して，df-pn の `mid()` を
ノード予算付きで再帰呼び出しし，証明/反証できるものを MID ループ前に確定させる．**

### なぜ df-pn (mid) を使うか

| 手法 | 深さ制限 | 閾値管理 | TT 活用 | 効率 |
|------|---------|---------|---------|------|
| static_mate | budget 依存(浅い) | なし | TT hit のみ | 低い |
| **mid() 再帰** | depth パラメータ | あり(1+ε) | 完全 | 高い |

- `static_mate` は単純再帰で閾値管理がないため，合駒の多い局面で指数的に膨張
- `mid()` は df-pn の全機能(TT, 閾値制御, 持ち駒優越, 遅延合駒)を使えるため，
  同じノード数でも遥かに深い問題を解ける
- 既存の TT を共有するため，1つ目の合駒で蓄積された証明が2つ目以降に活きる

## 実装プラン

### Step 1: `interpose_pre_solve` メソッドの追加 (dfpn.rs)

```rust
/// AND ノードの合駒子ノードに対する事前証明/反証試行．
///
/// 合駒を do_move した状態(OR ノード = 攻め方番)で，
/// `mid()` をノード予算付きで呼び出す．
///
/// 共有 TT を使用するため，1つ目の合駒の証明で蓄積されたエントリが
/// 2つ目以降の合駒で TT ヒットし，事実上ゼロコストで証明される．
///
/// # 引数
/// - `board`: 合駒後の盤面(攻め方番)
/// - `node_budget`: この子に割り当てるノード予算
/// - `ply`: 現在の探索深さ
/// - `remaining`: 残り手数
///
/// # 戻り値
/// TT lookup の結果 `(pn, dn)`:
/// - `pn == 0` → 証明成功
/// - `dn == 0` → 反証成功
/// - その他 → 予算内で未確定
fn interpose_pre_solve(
    &mut self,
    board: &mut Board,
    node_budget: u64,
    ply: u32,
    child_pk: u64,
    child_hand: &[u8; HAND_KINDS],
    remaining: u16,
) -> (u32, u32)
```

**ロジック:**

```rust
fn interpose_pre_solve(&mut self, board: &mut Board, ...) -> (u32, u32) {
    // 1. まず TT を確認(既に証明/反証済みなら即返却)
    let (pn, dn) = self.look_up_pn_dn(child_pk, child_hand, remaining);
    if pn == 0 || dn == 0 {
        return (pn, dn);
    }

    // 2. ノード予算を設定して mid() を呼び出す
    //    - 現在の nodes_searched を記録し，budget 分だけ探索を許可
    let saved_max_nodes = self.max_nodes;
    self.max_nodes = self.nodes_searched.saturating_add(node_budget);

    // 3. mid() を OR ノードとして呼び出す(攻め方番)
    //    pn_threshold = INF-1, dn_threshold = INF-1 で上限なし
    //    (ノード予算で制御するため)
    self.mid(board, INF - 1, INF - 1, ply + 1, true);

    // 4. max_nodes を復元
    self.max_nodes = saved_max_nodes;

    // 5. TT から結果を取得して返す
    self.look_up_pn_dn(child_pk, child_hand, remaining)
}
```

**重要な設計判断:**
- `self.max_nodes` を一時的に制限することで，既存の mid() の
  ノード制限チェック(L1003)がそのまま budget 制限として機能する
- mid() 終了後に max_nodes を復元するため，親の探索には影響しない
- TT は共有 → 子の証明結果が即座に親に見える

### Step 2: AND ノード子初期化フェーズでの統合

`mid` 関数の子初期化フェーズ(L1389-1405)を拡張:

```rust
if !or_node && m.is_drop() {
    // Pre-Solve: 合駒子局面を df-pn で解く試み
    if self.interpose_pre_solve_nodes > 0 && remaining >= 3 {
        let captured = board.do_move(*m);
        let child_remaining = remaining.saturating_sub(1);

        let (cpn, cdn) = self.interpose_pre_solve(
            board,
            self.interpose_pre_solve_nodes,
            ply,
            child_pk,
            &child_hand,
            child_remaining,
        );
        board.undo_move(*m, captured);

        if cpn == 0 {
            // 証明済み: and_proof を更新し children に追加しない
            let child_proof = self.table.get_proof_hand(child_pk, &child_hand);
            let adj = adjust_hand_for_move(*m, &child_proof);
            for k in 0..HAND_KINDS {
                and_proof[k] = and_proof[k].max(adj[k]);
            }
            #[cfg(feature = "profile")]
            { self.profile_stats.pre_solve_proved += 1; }
            continue;  // この子は完了
        }
        if cdn == 0 {
            // 不詰み確定: AND ノード全体が反証
            board.undo_move は上で済み
            self.store(pos_key, att_hand, INF, 0, remaining);
            #[cfg(feature = "profile")]
            { self.profile_stats.pre_solve_disproved += 1; }
            self.path.remove(&full_hash);
            return;
        }

        // 未確定: TT に更新された pn/dn が入っているため，
        // 通常の deferred/children 分類に進む
        // (pn/dn が大きければ MID ループで自然に後回しにされる)
        #[cfg(feature = "profile")]
        { self.profile_stats.pre_solve_exhausted += 1; }
    }

    // 従来通り deferred/children に分類
    push_move(&mut deferred_children, (*m, child_full_hash, child_pk, child_hand));
}
```

### Step 3: path (ループ検出) の整合性

`interpose_pre_solve` 内の `mid()` は `self.path` を共有する．

- 現在の AND ノードの `full_hash` は既に path に追加されている(L1423)
  → **OK**: 子の mid() が同じ局面に戻るとループ検出で正しく切断される
- Pre-Solve 終了後，path は mid() 内で自動清掃される
  → **OK**: mid() は自分が追加した path エントリを return 前に除去する(L1839)

ただし注意: Pre-Solve は子初期化フェーズ(L1389)で呼ばれるが，
この時点で path には現在のノードの full_hash が**まだ追加されていない**(L1423は後)．

→ **修正必要**: Pre-Solve を呼ぶ前に path.insert(full_hash) するか，
   Pre-Solve 呼び出しを L1423 の後に移動する必要がある．

**解決策:** 子初期化フェーズを2段階に分割:
1. 通常の子初期化(non-drop: そのまま children へ)
2. path.insert(full_hash) (L1423，現在の位置)
3. 合駒 drop の Pre-Solve + 分類

### Step 4: DfPnSolver にパラメータ追加

```rust
pub struct DfPnSolver {
    // ...既存フィールド...
    /// 合駒事前証明のノード予算(/子)．0 で無効．
    interpose_pre_solve_nodes: u64,  // デフォルト 4096
}
```

**デフォルト値の根拠:**
- 39手詰の ply 10 以降は df-pn で 349 ノード以下で証明完了
- 4096 ノードあれば ply 4〜8 のサブ問題も一部証明可能
- TT ヒットはノードカウントに含まれないため，
  2つ目以降の合駒は TT ヒットにより事実上ゼロコスト
- budget が大きすぎると Pre-Solve 自体がボトルネックになるため，
  ベンチマークで最適値を決定

### Step 5: profile 統計の追加

```rust
#[cfg(feature = "profile")]
pub struct ProfileStats {
    // ...既存...
    pub pre_solve_count: u64,      // Pre-Solve 実行回数
    pub pre_solve_ns: u64,         // Pre-Solve 合計時間
    pub pre_solve_proved: u64,     // 証明成功数
    pub pre_solve_disproved: u64,  // 反証成功数
    pub pre_solve_exhausted: u64,  // 未確定数
    pub pre_solve_nodes: u64,      // Pre-Solve 消費ノード合計
}
```

### Step 6: テスト

1. **既存テスト全パス**: `cargo test -p maou_shogi` (non-ignored)
2. **39手詰テスト**: `test_tsume_39te_aigoma` (ignored) が解けるか確認
3. **回帰テスト**: 短手数テストで性能劣化がないか
4. **Pre-Solve 動作テスト**: 39手詰 PV 上の各合駒局面で proved/exhausted 分類を検証

### Step 7: ベンチマーク

`bench_39te_budgets` を拡張:
```rust
let pre_solve_nodes = [0u64, 1024, 2048, 4096, 8192, 16384];
// budget=0 は Pre-Solve 無効(従来動作)
```

## TT 共有による連鎖証明のメカニズム

39手詰の6九合駒(金・銀・飛・角)を例にとると:

```
1. 飛合の Pre-Solve 開始 (budget: 4096 nodes)
   → mid() が 6九飛合後の OR ツリーを探索
   → 証明成功 (例: 300 nodes)
   → TT に「6九飛合取り後の局面」が証明済みとして記録

2. 角合の Pre-Solve 開始 (budget: 4096 nodes)
   → mid() が 6九角合後の OR ツリーを探索
   → 攻め方が角を取る王手 → 取り後局面は飛合取り後と盤面同一
   → 持ち駒優越: 攻め方は角を取得(飛より弱いが他の持ち駒で補完)
   → TT ヒット → 証明成功 (例: 5 nodes)

3. 金合/銀合も同様に TT ヒットで即証明

結果: 4つの合駒分岐が合計 ~350 nodes で全証明
      → この AND ノードの pn が 0 になり，親 OR ノードの探索から除外
```

## NPS への影響と許容範囲

- Pre-Solve は子初期化フェーズの1回のみ実行(MID ループ毎ではない)
- 合駒4種 × budget 4096 = 最大 ~16K ノード/AND ノード
- しかし TT 共有により実際の消費は初回のみ(2つ目以降はほぼ無料)
- 証明済み合駒は children から完全除外 → MID ループのコストが激減

**NPSの低下は容認．simplest-first 原則に従い，
NPS よりも「証明できる子を先に証明して MID ループの負荷を軽減する」
探索効率を重視する．**

## 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `rust/maou_shogi/src/dfpn.rs` | interpose_pre_solve メソッド，AND 子初期化の統合，profile 統計 |
| `rust/maou_shogi/Cargo.toml` | バージョンバンプ (minor: 新機能) |

## 段階的アプローチ

1. **Phase A**: `interpose_pre_solve` メソッド実装 + path 整合性修正
2. **Phase B**: AND ノード子初期化への統合
3. **Phase C**: テスト(既存全パス + 39手詰)
4. **Phase D**: ベンチマーク + budget パラメータ調整
