# ループ・GHI 対策

### 7.1 経路依存フラグ付き GHI 対策 (Kishimoto & Müller 2004/2005)

**出典:** Kishimoto & Müller, "A solution to the GHI problem for depth-first proof-number search" (IS 175.4, 2005)

GHI (Graph History Interaction) は，同一局面が異なる探索経路で
異なる結果を持つ問題．千日手のような繰り返し検出は経路に依存するため，
ある経路で得た反証が別の経路では無効になりうる．

KomoringHeights は dual TT (base/twin) で経路依存/非依存の不詰を区別する．

**実装:** solver.rs (`path: FxHashSet`，ループ検出，GHI 伝播)

maou_shogi では dual TT の代わりに経路依存フラグ方式を採用:

1. **ループ検出**: `path: FxHashSet<u64>` で現在の探索パス上の全ノードハッシュを保持．
   子ノードが path 上に存在すれば循環と判定し，即座に `(INF, 0)` を返す
2. **経路依存反証**: ループ検出に由来する反証を `path_dependent = true` で TT に保存
3. **IDS 間清掃**: `clear_working()` で WorkingTT 全クリア(経路依存反証を含む)し，
   `clear_proven_disproofs()` で ProvenTT の confirmed disproof を除去．
   異なる深さの探索で自動的に再評価
4. **Remaining 免除**: 経路依存エントリは remaining チェックをバイパス
   (`e.remaining >= remaining || e.path_dependent`)

**出典との差異:**
- 論文の dual TT 方式ほど完全ではないが，経路依存の反証が TT を
  永続的に汚染する問題を軽減する実用的な妥協案

### 7.2 NM Remaining 伝播

深さ制限に由来する不詰(NM: Non-Mate)の深さ情報を正確に伝播する．

**実装:** mod.rs (`propagate_nm_remaining`)

```
nm_remaining = min(child_remaining + 1, current_remaining)
```

- 子の NM が `REMAINING_INFINITE` なら親も `REMAINING_INFINITE`
- 有限 remaining の NM は深い IDS 反復で再評価される
- `REMAINING_INFINITE = u16::MAX`: 深さ非依存の真の証明/反証
