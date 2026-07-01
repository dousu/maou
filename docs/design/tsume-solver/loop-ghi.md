# ループ・GHI 対策

詰将棋の探索木は転置 (DAG) と千日手 (循環) を含むため，同一局面が異なる経路で異なる結果を
持ちうる (GHI 問題)．本節は循環検出・経路依存反証の健全な扱い・dominance 枝刈りを扱う．

### 7.1 GHI と千日手 (Kishimoto & Müller 2004/2005)

**出典:** Kishimoto & Müller, "A solution to the GHI problem for depth-first proof-number search"
(Information Sciences 175.4, 2005)

GHI (Graph History Interaction): 千日手のような繰り返し判定は**経路に依存する**ため，ある経路で
得た反証が別の経路では無効になりうる．これを無視して反証を局面キーで TT に書くと，別経路で
過剰適用して偽の不詰 (false NoMate) を生む．

**実装:** 探索パス上の局面検出と `repetition` 結果の伝播 (`search/expansion.rs`,
`search_result.rs`)．

- 探索パス上の局面深さは `path_depths` で保持し，子が path 上の先祖を指す場合に循環と判定する．
- 循環を含む結果は `SearchResult::make_repetition(.., repetition_start)` で表現し，
  `repetition_start < depth` のとき (= 自分より浅い循環) のみ親へ伝播する (GHI soundness)．
- 千日手由来の結果は局面キーの通常エントリとして確定させず，循環の起点情報を保つ．
- 循環子の検出 (`does_have_old_child`) は TCA の inc_flag も駆動する
  ([threshold-control.md §3.2](threshold-control.md))．

### 7.2 経路依存反証の scope 化 (horizon disproof)

**実装:** `Params::scope_disproof` (既定 **true**), `Params::repetition` (既定 false)．

深さ上限 (horizon) 到達による仮反証 (`look_up_pn_dn` で `remaining == 0` → `(INF, 0)`) が
伝播して集約反証 (curr.dn==0) を生んだ場合，それを**絶対的な確定反証として TT に書くと
soundness バグ**になる (より深く探索すれば詰む局面を「不詰」と固定してしまう)．

- `scope_disproof=true`: 集約反証を **remaining scope 付き**で格納する．`e.remaining() >= remaining`
  を満たす lookup でのみ反証として有効になり，より深い ply (= remaining 大) の transposition
  では再探索される．これで horizon 反証による false NoMate を健全に根治する．
- `scope_disproof=false`: horizon 反証を確定化し TT を汚染する (soundness バグ; 既定で使わない)．
- `repetition=true` (opt-in): 反証が循環依存である場合のみ scope 限定し，それ以外は確定
  (`REMAINING_INFINITE`) で格納する精緻化．soundness-critical のため既定 false (baseline 不変)．

旧版の depth-limited disproof 格納閾値や refutable disproof エントリは，この scope 化に統合・
代替された (記録は [legacy/](legacy/README.md))．

### 7.3 visit-history dominance (maou 独自)

**実装:** `Params::use_visit_history_dominance` (既定 **true**), `dom_path` (`path_stack.rs`)．

子展開時に `(child_pos_key, child_hand)` が現在の探索パス上の先祖に**持ち駒支配されている**
場合 (同一 `board_key` で `hand_gte_forward_chain(ancestor_hand, child_hand)` 成立) を，
循環と同様の**経路依存不詰**として枝刈りする．

- **健全性の根拠:** 攻め方が同じ盤面を「より少ない (または等価以下の) 持ち駒」で再訪している
  なら，過去のより有利な状態でも詰められなかった事実から，現在も詰められないと健全に推論できる．
- chain 合駒で持ち駒の多様性が指数爆発する局面の枝刈りに効く
  ([aigoma-optimization.md](aigoma-optimization.md))．
- 経路依存反証として扱われるため永続 TT エントリを汚染しない (§7.2 と整合)．発火回数は
  診断カウンタ `dom_fires` で計測する．

> **note:** `Params::path_dominance` (既定 off) は mid path の別形式の dominance を試す opt-in で
> あり，現状ノード増のため既定では除外する (visit-history dominance とは別機構)．

### 7.4 len-aware による不詰深さの扱い

旧版は深さ制限由来の不詰 (NM) の深さを `nm_remaining` で個別伝播していたが，統一 mid は
TT エントリの `proven_len` / `disproven_len` (len-aware, [TT §6.3](transposition-table.md)) と
§7.2 の scope 化で扱う．`REMAINING_INFINITE` (深さ非依存の確定結果) と scope 付き (深さ依存の
仮結果) を区別し，深い反復で自動的に再評価する．

### 7.5 転置による偽証明 (proof-tree 循環) と STRICT verify の権威化

§7.1 の `path_depths` は **現在の探索スタック上**の循環しか検出しない．一方，**転置 (TT) の
proven エントリを別分岐で再利用**すると，**探索スタックに現れない proof-tree の循環**が
形成され得る (proven 局面 X が分岐 B1 で確定 → 別分岐 B2 で X を TT 再利用 → B2 の経路が X の
証明内へ戻る循環)．この循環は「玉が逃げられる非詰み」を proven と記録する **偽証明 (false
proof / false mate)** を生む．探索順序 (閾値ゆるめ等) に依存して露出するが，バグ自体は順序に
依存せず潜在する (localize 例: AND 局面 `9/9/4+N1pS1/9/4+R4/5+B3/6R1P/3S2k2/9 w` で玉が `3h2i`
へ合法脱出できるのに proven と記録)．

- **TT look_up の穴** (`tt/mod.rs::look_up`): proven (`pn==0`) 分岐は即 `make_final(true)` を
  返し，unknown 分岐が行う `is_possible_repetition` + rep-table 再チェックを**欠く**．ただし
  cross-branch の clean-proof 再利用では `is_possible_repetition` が立たないため，proven 分岐に
  同じ再チェックを足しても捕捉できない．探索側での完全な GHI-safe 化は **proof-path 循環追跡**を
  要する (Kishimoto & Müller の source-node 方式等) research-level 課題として残る．
- **採用した健全化 — STRICT verify の権威化 (maou_shogi 3.4.2)**: `verify_proof`
  ([move-ordering-and-pv.md](move-ordering-and-pv.md)) は全合法防御を実 replay し `path.contains`
  で proof-path 循環も検出する **GHI-correct な厳密判定**である．`solve_impl` はこれを**最終
  権威**とし，**STRICT VERIFY が `Some(d)` を返したときのみ `Checkmate` を返す**．`None` (偽証明
  もしくは verify budget 不完全) は偽の詰みを返さず `Unknown` を返す (**soundness > completeness**:
  真の詰みの取りこぼしより偽詰みの回避を優先)．default 探索では verify が常に `Some` ゆえ挙動は
  不変 (探索が偽 proven を出した場合のみ `Unknown` へ落ちる)．
