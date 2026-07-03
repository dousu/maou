# 転置表管理

統一 mid は **単一の len-aware 転置表 (TT)** を持つ．旧版の Dual TT (ProvenTT + WorkingTT)・
FrontierTT・LeafDisproofTT・refutable disproof エントリは廃止された (記録は git 履歴)．
本節はエントリ構造・持ち駒優越・cross-hand 参照・GC を扱う．

### 6.1 持ち駒優越 (hand dominance, Nagai 2002)

**出典:** Nagai 2002 (持ち駒の包含関係による TT 再利用)

盤面が同一で持ち駒が異なる局面間の包含関係を使う:

```
  証明 (pn=0) の再利用:            反証 (dn=0) の再利用:
  TT: hand={P,G} -> pn=0          TT: hand={P,P,G} -> dn=0
  query: hand={P,P,G}             query: hand={P}
  {P,P,G} >= {P,G} ? YES          {P} <= {P,P,G} ? YES
  -> 証明を再利用                  -> 反証を再利用
  (持ち駒が多いほど詰ませやすい)  (持ち駒が少ないほど詰ませにくい)
```

- TT キー: `position_key(board)` = 盤面ハッシュ (**持ち駒を含まない**)．
- 同一クラスタ内に同一 `pos_key` の複数 hand エントリを保持し，lookup 時に支配関係を判定する．
- **実装:** `hand_gte` (`proof_hand.rs`), `look_up*` (`tt/mod.rs`)．

### 6.2 forward-chain 持ち駒代替 (maou 独自)

持ち駒優越の拡張として，前方利き系の駒種間で代替関係を認める (`hand_gte_forward_chain`):

- 歩の不足 → 香で代替可能 / 香の不足 → 飛で代替可能 / 歩の不足 → 飛で代替可能 (カスケード)．
- 桂・銀・金・角は独立 (利きの方向が異なるため代替不可)．

チェーン合駒で攻め方が合駒を取った後の持ち駒構成が違っても，前方利き系では「弱い駒で詰めば
強い駒でも詰む」ため代替を認め，証明/反証の再利用率を高める．

### 6.3 len-aware エントリ構造

**実装:** `tt/entry.rs`．1 エントリ = **64 byte (1 キャッシュライン)**:

| フィールド | 型 | 役割 |
|-----------|----|----|
| `board_key` | u64 | 盤面キー (持ち駒除外; cross-hand のため) |
| `pn`, `dn` | u64 ×2 | 証明数・反証数 |
| `proven_len`, `disproven_len` | u16 ×2 | **詰み/不詰の手数** (len-aware) |
| `min_depth_rep` | u16 | min_depth (15 bit) + 千日手フラグ (REP_BIT) |
| `hand`, `parent_hand` | Hand=[u8;7] ×2 | 当該局面・親局面の攻め方持ち駒 |
| `sum_mask` | BitSet64 | δ 集約で sum する子の集合 ([proof-numbers §4.1](proof-disproof-numbers.md)) |
| `amount` | u32 | 探索量 (GC 優先度) |
| `parent_board_key` | u64 | DAG 親参照 (二重計数除去, [proof-numbers §4.2](proof-disproof-numbers.md)) |

**mate length の保持 (proven_len / disproven_len)** は KomoringHeights で実装されている手法で，
最短手順保証 (OR=最短・AND=最長を選ぶ PV 復元) と len-aware cross-hand 集約の健全性に用いる．
`SearchResult` も同じく len-aware である ([search §2.2](search-architecture.md))．

### 6.4 cross-hand 親参照 (look_up_parent)

子の集約に必要な親エントリを，**厳密一致 hand があればそれを，無ければ上位/下位 hand から
境界を合成**して返す (KomoringHeights の親参照に基づく手法)．

**実装:** `look_up_parent` / `apply_delta_hand` (`tt/mod.rs`)．

- クラスタを走査し `hand_gte_forward_chain` で支配するエントリを探す．
- 厳密一致が無い場合，`apply_delta_hand(parent_hand, entry_hand, child_hand)` で上位 (証明)・
  下位 (反証) 境界を合成し，`(parent_board_key, parent_hand, pn_bound, dn_bound)` を返す．
- 証明駒・反証駒 (極小証明駒 / 極大反証駒) の計算と組み合わせ，持ち駒越境の集約を行う
  ([proof-numbers §4.3](proof-disproof-numbers.md))．証明駒・反証駒の活用は KomoringHeights で
  整理された手法に基づく．

### 6.5 ガベージコレクション (amount ベース)

TT が埋まると `look_up` がクラスタ走査で O(容量) に退化するため，低 amount エントリを間引く．
サンプリング GC は KomoringHeights を参考にした方式である．

**実装:** `collect_garbage` / `maybe_collect_garbage` (`tt/mod.rs`)．

- **発火条件**: hashfull ≥ 0.5．`emplace` から 4096 build ごとに `maybe_collect_garbage` を点検．
- **方式**: 約 20,000 エントリを stride サンプリングして amount 分布を取り，下位 50% の amount を
  閾値に削除・compact する (`amount` = 探索量 = 有用度の代理)．
- 探索中の局面 (path 上) は amount が高く保たれるため evict されにくい．
- 証明済/反証済 (final) エントリは GC 対象外 (STRICT verify の PV 復元が辿る証明木を保護)．

### 6.7 千日手テーブルの generation GC (長手数対応)

**出典:** 経路ハッシュ用の固定サイズテーブルと generation GC は KomoringHeights で実装されて
いる方式に基づく．

千日手 (循環) 由来の**経路依存**結果は，主 TT (局面キー) でなく **経路ハッシュ (`path_key`)
をキーとする専用テーブル** `RepetitionTable` (`tt/mod.rs`) に記録する
([loop-ghi.md §7.1](loop-ghi.md))．主 TT と同じく **固定サイズの循環配列 (open-addressing +
線形走査, multiply-shift index)** で持ち，メモリを bound する:

- 各エントリに **generation** (16 bit) を付す．挿入が `テーブルサイズ / 20` 回進むごとに
  generation を 1 増やし，一定 generation ごとに `collect_garbage` で**直近数 generation を残して
  古いエントリを消し compaction** する．これで千日手テーブルが問題サイズに比例して**青天井に
  膨張するのを防ぐ** (旧実装は無制限 `FxHashMap` で，ミクロコスモス級 1525 手の探索で OOM した)．
- **健全性**: GC でエントリが退去しても，`look_up` は rep 表を引けなければ `make_unknown` へ
  フォールスルーして**再探索**するだけ (千日手の正しさは `path_depths` の on-path 検出が独立に
  担保する; [loop-ghi §7.1](loop-ghi.md))．compaction が不完全で一部エントリが到達不能になっても
  同様に再探索されるだけで，誤った不詰は生じない．
- **サイズ**: 既定は主 TT と同数 (1 エントリ 24 byte)．`REPSIZE` env で調整可．通常の (短手数)
  詰将棋では GC が発火せず挙動不変で，長手数探索でのみ効く．

### 6.8 旧アーキとの差異

| 旧 (二エンジン期) | 現行 (統一 mid) |
|---|---|
| Dual TT (ProvenTT hand_hash 混合 + WorkingTT pos_key) | **単一 TT** (pos_key クラスタ + hand 別エントリ) |
| FrontierTT / LeafDisproofTT | 廃止 |
| refutable disproof エントリ (flags bit 7) | 廃止 (horizon 反証は scope 化, [loop-ghi §7.2](loop-ghi.md)) |
| Zobrist hand_hash 混合インデクシング | 盤面キーのみ + cross-hand 支配判定 |
| Pareto frontier / `|pn-dn|` 置換 | amount ベース GC |
| エントリ 24〜40 byte 圧縮の試行 | 64 byte (1 キャッシュライン) 固定 + len-aware |
