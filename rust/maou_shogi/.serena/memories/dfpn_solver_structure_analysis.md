# DFPN Solver Implementation Analysis

## Key Files
- **Main solver**: `/workspaces/maou/rust/maou_shogi/src/dfpn/solver.rs`
- **Transposition Table**: `/workspaces/maou/rust/maou_shogi/src/dfpn/tt.rs`
- **Tests**: `/workspaces/maou/rust/maou_shogi/src/dfpn/tests.rs`
- **Profile stats**: `/workspaces/maou/rust/maou_shogi/src/dfpn/profile.rs`

## 39-Move Tsume Test Details
- **Test function**: `test_tsume_39te_aigoma` (line 1607 in tests.rs)
- **SFEN**: `"9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1"`
- **Expected moves**: 39-move checkmate (fully tested with expected sequence)
- **Settings**:
  - depth=41, max_nodes=10_000_000, draw_ply=32767, timeout=60s
  - find_shortest=false

## Node Counting

### Primary Counter: `nodes_searched`
- **Type**: `pub(super) nodes_searched: u64` (line 185 in solver.rs)
- **Incremented**: In the `mid()` function at line 2043-2048:
  ```rust
  self.nodes_searched += 1;
  if (ply as usize) < 64 {
      self.ply_nodes[ply as usize] += 1;
  }
  ```
- **Reset**: At beginning of `solve()` function (line 1385)

### Supporting Node Tracking
- **ply_nodes**: `[u64; 64]` - per-ply node count array (line 186)
- **max_ply**: Maximum ply reached during search
- **ply_iters**: `[u64; 64]` - per-ply MID loop iteration count (line 188)
- **ply_stag_penalties**: `[u64; 64]` - per-ply stagnation penalty count (line 189)

### Current Instrumentation Gaps
- **No duplicate visit counter**: `nodes_searched` counts ALL node visits, but doesn't distinguish between:
  - First visit to a node position
  - Re-visits to the same position (due to different proofs/dispproofs at same ply but different IDS depths)
  - Visits due to TT eviction followed by re-exploration

## Search Loop Architecture (MID function)

### Location
- **Function**: `pub(super) fn mid(...)` (line 2023 in solver.rs)
- **Signature**: Takes `(pn_threshold, dn_threshold, ply, or_node)`

### Node Expansion Flow
1. **Entry checks** (2032-2105):
   - Node limit check (`nodes_searched >= max_nodes`)
   - Timeout check (every 1024 nodes)
   - Increment `nodes_searched` counter
   - Periodic GC (every 100K nodes)

2. **TT Lookup** (2107-2150):
   - Calculate `pos_key` (position key without hand info)
   - Call `look_up_pn_dn(pos_key, &att_hand, remaining)`
   - If `tt_pn==0` or `tt_dn==0`: return (terminal condition)
   - If `tt_pn >= pn_threshold` or `tt_dn >= dn_threshold`: return (threshold cutoff)

3. **Move Generation** (2229-2295):
   - Generate check moves (OR nodes) or defense moves (AND nodes)
   - Dynamic move ordering: TT best move + killer moves
   - Create `children` array with `(Move, full_hash, pos_key, attacker_hand)` tuples

4. **Child Node Initialization** (2320-2550):
   - For each move:
     - Execute move: `board.do_move(*m)`
     - Calculate `child_pk = position_key(board)` 
     - Look up child: `look_up_pn_dn(child_pk, &child_hand, child_remaining)`
     - **Inline evaluation**: PN_UNIT heuristics (mate-in-1, mate-in-3 detection)
     - **Store to TT**: `store()`, `store_with_best_move()`, `store_with_best_move_and_distance()`
     - Undo move: `board.undo_move(*m, captured)`

5. **Main MID Loop** (following child init):
   - Loop iterations continue, evaluating pn/dn changes
   - Recursively call `mid()` for selected nodes
   - Update parent pn/dn based on children results

## TT Lookup Functions

### Primary Lookup: `look_up_pn_dn` (solver.rs:1067)
```rust
pub(super) fn look_up_pn_dn(
    &self,
    pos_key: u64,
    hand: &[u8; HAND_KINDS],
    remaining: u16,
) -> (u32, u32, u32)  // (pn, dn, source)
```
- Delegates to `look_up_pn_dn_impl()` with `neighbor_scan=true`
- Uses `skip_refutable_disproof` flag (PNS vs MID mode)
- Returns `(PN_UNIT, PN_UNIT, 0)` for unexplored nodes

### Implementation variants in TT:
- `look_up_working()`: WorkingTT intermediate entries
- `look_up_proven()`: ProvenTT proof/disproof entries  
- `look_up_proven_skip_refutable()`: ProvenTT with refutable disproof filtering (PNS mode)

### Special cases handled:
- **remaining=0** (depth limit reached): 
  - If proof exists: return `(0, INF, 0)` 
  - Otherwise: return `(INF, 0, 0)`
- **Refutable disproof**: Can be skipped in PNS, visible in MID

## TT Store Functions

### Main Store (solver.rs:1169)
```rust
pub(super) fn store(
    &mut self,
    pos_key: u64,
    hand: [u8; HAND_KINDS],
    pn: u32,
    dn: u32,
    remaining: u16,
    source: u32,
) {
    if pn == 0 && self.ancestor_has_proof() {
        return;  // Skip ProvenTT if ancestor has proof
    }
    self.table.store(pos_key, hand, pn, dn, remaining, source);
}
```

### Variants
- `store_with_best_move()`: Also stores best move info
- `store_with_best_move_and_distance()`: Stores mate distance for PV extraction
- `store_tagged_proof()`: (dead code - replaced by refutable disproof mechanism)
- `store_refutable_disproof()`: Stores proofs marked as refutable (v0.24.75+)

### TT Storage Logic (tt.rs):
- **ProvenTT**: Stores proof (pn=0), confirmed disproof (dn=0), refutable disproof
- **WorkingTT**: Stores intermediate entries, depth-limited disproof
- **LeafDisproofTT**: Remaining≤2 specialized buffer (32MB)

## Transposition Table Structure

### Dual TT Architecture (v0.24.0+)
1. **ProvenTT** (permanent entries):
   - Proof: pn=0 
   - Confirmed disproof: dn=0, remaining=INFINITE, non-path-dependent
   - Refutable disproof: (v0.24.75+) marked with flag bit 7

2. **WorkingTT** (GC-eligible):
   - Intermediate entries: pn,dn > 0
   - Depth-limited disproof: remaining < INFINITE
   - Path-dependent disproof

3. **LeafDisproofTT**:
   - Specialized for remaining≤2 entries
   - 512K clusters × 4 entries × 16B = 32MB

### Hash function
- `pos_key`: 64-bit Zobrist hash (board without hand)
- `hand_hash`: Computed from attacker's hand pieces
- Combined with neighbor scanning for robust lookup

## Profiling & Statistics

### ProfileStats Structure (profile.rs)
Collected when `profile` feature enabled. Tracks:
- **Operation timings** (nanoseconds):
  - `position_key_ns`, `loop_detect_ns`, `tt_lookup_ns`, `tt_store_ns`
  - `movegen_check_ns`, `movegen_defense_ns`
  - `do_move_ns`, `undo_move_ns`
  - `child_init_ns`, `main_loop_collect_ns`, `depth_limit_terminal_ns`
  - `nm_promotion_refutable_ns`, `capture_tt_lookahead_ns`
  - `cross_deduce_ns`, `prefilter_ns`
  
- **Call counts**: Corresponding `_count` fields for each operation

- **TT overflow stats**:
  - `tt_overflow_count`, `tt_proven_overflow_count`, `tt_working_overflow_count`
  - `tt_overflow_no_victim_count`
  - `tt_max_entries_per_position`

- **Phase timings**:
  - `mid_total_ns`, `pns_total_ns`

### Diagnostic Fields in DfPnSolver
- `prefilter_hits`: TT prefilter fire count
- `nodes_searched`: Total nodes visited (already existing)
- `diag_*` fields (tt_diag feature): Detailed breakdown per ply
  - `diag_ply_visits[ply]`: Visit count per ply
  - `diag_ply_proofs[ply]`: Proof store count per ply
  - Terminal/threshold exit counts per ply
  - Refutable check statistics
  - Cross-deduce hit counts

## Areas for Duplicate Visit Counting

### Challenge
- Current `nodes_searched` counts **total visits** including re-visits
- Need to track **duplicate visits**: visits to same position at different times during same solve

### Potential Implementation Points
1. **Position revisit detection**:
   - Hash-based memo of visited (pos_key, att_hand) at specific ply depths
   - Compare current visit count vs. prior count from TT

2. **TT entry reuse tracking**:
   - Track entries evicted from WorkingTT that are later re-explored
   - Count how many times same position yields same TT hit

3. **IDS depth revisits**:
   - When IDS increases depth, count positions revisited at deeper search
   - Distinguish from fresh exploration

4. **Move generation revisits**:
   - In move loop, count how many times same child position is revisited
   - (Currently each child visited once per parent call, but across IDS iterations)

### Recommended Hook Points for Instrumentation
1. **In MID function entry** (line 2043):
   - Before `nodes_searched += 1`, check if (pos_key, att_hand) was seen before
   - Track with HashSet or HashMap

2. **In TT lookup** (line 2152):
   - Increment counter when lookup hits existing entry vs. fresh search

3. **In child loop** (line 2321):
   - Track child position revisits within same parent call
   - Distinguish new children from revisited ones
