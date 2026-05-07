# DFPN Transposition Table Eviction 調査結果

## 調査概要
verbose ログで「root entry (pos_key=0x58caa5eab2d4fe89, remaining=31, hand=[0,0,0,0,0,0,0]) が 8M→9M ノード付近で TT から消失」する現象を解明した。

## アーキテクチャ概要

### 二重 TT 構造 (Dual TT, Plan D v0.24.0+)
- **ProvenTT**: proof (pn=0) + confirmed disproof (dn=0, !path_dependent, remaining=INFINITE)
  - 永続エントリのみ（depth 非依存）
  - size ~2.4M at issue point
- **WorkingTT**: intermediate (pn>0, dn>0) + depth-limited/path-dependent disproof
  - 一時エントリ（depth 依存）
  - size ~127K after GC

### クラスタ構成
- ProvenTT: PROVEN_CLUSTER_SIZE=4 エントリ/クラスタ
- WorkingTT: WORKING_CLUSTER_SIZE=8 エントリ/クラスタ
- LeafDisproofTT: LEAF_CLUSTER_SIZE=4 エントリ/クラスタ（remaining≤2 の反証用）

## Eviction ポリシー

### 1. store_impl() での登録判定

#### 前提チェック: ProvenTT での支配判定
```rust
// 既存 proof/disproof に支配されているなら登録不要 → dominated_skip++
if is_proof() && hand_gte_forward_chain(&hand, &existing.hand) {
    return; // 自 hand が支配される → 挿入不要
}
if !is_proof() && !path_dependent && hand_gte_forward_chain(&existing.hand, &hand) {
    return; // 既存 disproof が自を支配 → 挿入不要
}
```

### 2. ProvenTT への insertion (store_proven)

#### 積極的削減ポリシー
- **同一 hand の proof 削除**: 新しい方が priority（mate_distance で判定）
- **被支配 proof 削除**: hand_gte で支配される既存 proof を除去
- **WorkingTT intermediate の積極除去**: proof found 時に同 pos_key の intermediate を全除去

#### replace_weakest_proven のポリシー
```rust
// amount ベースで置換対象を選定
// - proof with distance: 128..191 (長いほど高priority)
// - proof without distance: 64
// - confirmed disproof: 32
// foreign > same-key で優先度を決定
// 新エントリの amount >= 既存最弱 amount の場合のみ置換
```

### 3. WorkingTT への insertion

#### store_working_intermediate
- **同一 hand/pos_key の update**: 既存エントリの pn/dn/remaining を更新 → amount++
- **新規追加**: 空きスロットを優先
- **Overflow時**: `replace_weakest_in()` で外 key の amount 最小エントリを置換

#### store_working_disproof
- **中間エントリ削除**: 新しい disproof が登録されると，同 pos_key の intermediate は全除去
- **被支配 disproof 削除**: disproof の hand_gte で被支配する既存 disproof を除去
- **Overflow処理フロー**:
  1. `replace_weakest_for_disproof_in()` で foreign depth-limited disproof を優先
  2. 次に foreign intermediate (lowest amount)
  3. NM-to-NM 置換（同 pos_key の depth-limited disproof を上書き）
  4. LeafDisproofTT へ（remaining≤2）
  5. `replace_weakest_in()` フォールバック

### 4. GC トリガーと処理

#### Overflow-triggered GC (2M ノード毎)
```rust
let overflow = self.table.drain_working_overflow();
if overflow > 10_000 && self.nodes_searched >= self.next_overflow_gc {
    self.mark_path_entries_for_gc_protection(); // amount=255 で保護
    let removed = self.table.gc_working_overflow();
    self.next_overflow_gc = self.nodes_searched + 1_000_000;
}
```

**mark_path_entries_for_gc_protection の処理:**
```rust
fn mark_path_entries_for_gc_protection(&mut self) {
    for i in 0..self.path_len {
        self.table.protect_working_entry(
            self.path_pos_key[i],
            &self.path_hand[i],
        );
    }
    // root 自体も保護（path に含まれない場合がある）
    if self.path_len > 0 {
        self.table.protect_working_entry(
            self.diag_root_pk,
            &self.diag_root_hand,
        );
    }
}
```

#### gc_working_sampling の処理フロー
```
Phase 1: サンプリングで amount 分布を収集（stride で 約8192 エントリ）
Phase 2: nth_element で除去閾値を決定（下位 50% を除去対象）
Phase 3: Obsolete intermediate 除去（ProvenTT に proof/disproof ある局面）
Phase 4: 閾値以下の disproof を除去
Phase 5: CutAmount（max_amount > 32 の場合，全エントリ amount を 1/2）
```

## Root Entry 消失メカニズム

### なぜ root entry が evict されるのか

1. **Root entry が intermediate として WorkingTT に登録される**
   - root は pn/dn がもに > 0（未解決）
   - WorkingTT に intermediate として格納
   - 初期 amount = 1 (低い!)

2. **Protection 欠落のウィンドウ**
   - `mark_path_entries_for_gc_protection()` は GC **直前** のみ呼ばれる
   - GC トリガー判定 `overflow > 10_000 && nodes >= next_overflow_gc` を満たさない場合は **保護されない**
   - GC 実行まで amount = 1 のままで，サンプリング下位 50% に該当しやすい

3. **GC_REMOVAL_RATIO = 0.5 による aggressive removal**
   - サンプリング 8192 エントリの下位 50% = 4096 レベル以下を除去
   - root (amount=1) はほぼ確実に対象

4. **Obsolete intermediate 除去のバグリスク**
   - Phase 3 で ProvenTT に proof/disproof がある場合 intermediate を除去
   - root の別 hand variant が ProvenTT に registered されている場合，root が誤除去される可能性
   - hand_gte 判定の境界で誤判定の可能性

### TT サイズ急減: 2.4M → 127K

**2.4M (ProvenTT) + working intermediate が gc() で全削除される理由:**

```rust
pub(super) fn gc(&mut self, target_size: usize) {
    if self.len() <= target_size { return; }
    
    // Phase 1: remaining <= 8 の intermediate を除去
    for fe in self.working.iter_mut() {
        if fe.entry.dn == 0 { continue; } // disproof skip
        if fe.entry.remaining() <= 8 {
            fe.pos_key = 0;
        }
    }
    
    if self.len() <= target_size { return; }
    
    // Phase 2: WorkingTT 全クリア（ProvenTT は保持）
    self.retain_proofs(); // = clear_working()
}
```

- 8M ノード時点で TT が target_size（例：2M エントリ）を超過
- Phase 1 で remaining≤8 の intermediate を除去 → 効果不十分
- Phase 2 で WorkingTT を完全クリア → **2.4M → 127K（ProvenTT のみ）**

## 診断カウンタと意味

### verbose feature での tracking
- `diag_intermediate_new`: 新規 intermediate 挿入数
- `diag_intermediate_update`: 既存 intermediate の更新数
- `diag_dominated_skip`: support domination で skip した数
- `diag_remaining_dist[33]`: remaining 値別の insertion distribution
  - [0]: remaining=0
  - [1]: remaining=1..4
  - [2]: remaining=5..31
  - [3]: remaining=INFINITE

### remaining=31 の意味
- depth 制限 (depth - ply = 31) での登録
- 浅い depth limit で登録された中間エントリ
- IDS の depth が進むと obsolete になりやすい

## 根本原因

1. **Root entry は intermediate として amount=1 で登録される**
   - ProvenTT ではなく WorkingTT に格納（IDS depth 依存）
   - 優先度低い

2. **Protection window が限定される**
   - `mark_path_entries_for_gc_protection()` は GC トリガー時のみ
   - overflow > 10_000 かつ 1M クールダウン中でないことが条件
   - 小規模 overflow では GC すら実行されない

3. **GC_REMOVAL_RATIO = 0.5 が aggressive**
   - amount=1 のエントリは確実に除去対象
   - root は探索パスの頻出ノード = 再利用率高い → 高 priority に昇格する機会が失われやすい

4. **GC 直前の phase 2 で WorkingTT 全削除**
   - remaining 条件でない intermediate も全除去される可能性
   - root の remaining=31 は target_size 超過時に除去対象になりやすい

## 推奨修正案

1. **Root entry のプリエンプティブ保護**
   - store_working_intermediate で root を登録する時点で amount=255 に初期化
   - または root に special marker flag を設置

2. **Path protection を早期化**
   - GC トリガー判定を loosen する（overflow > 5_000, cooldown = 500K）
   - または IDS transition 時に事前保護

3. **GC_REMOVAL_RATIO の adaptive 化**
   - root depth が深い場合は ratio を 0.3 に下げる
   - または obsolete intermediate 除去をより aggressive に

4. **Intermediate amount の初期値見直し**
   - amount=1 ではなく amount=4 で登録
   - または depth-based scaling: amount = min(depth / 4, 8)