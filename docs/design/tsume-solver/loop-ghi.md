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

### 7.5 verify 内 GHI (経路依存 None の memo 汚染) と STRICT verify の権威化

**再診断 (3.4.3) による訂正**: 本節の旧版 (3.4.2 時点) は「転置 (TT) の proven エントリの
cross-branch 再利用が proof-tree 循環を作り偽証明 (false mate) を生む」と診断していたが，
これは**誤診**だった．localize 例とされた AND 局面 `9/9/4+N1pS1/9/4+R4/5+B3/6R1P/3S2k2/9 w`
は，全 4 防御 (3h2i/3h2h/3h4h/3h4i) の子局面を個別に solve して **全 hand で真の詰み**と
STRICT 確認された (「3h2i で脱出可能」は誤認)．**探索は偽 proven を作っておらず，TT
lookup の proven 無条件返却も KH user-engine と同構造で健全**である (千日手由来の結論は
`set_repetition` が rep flag + path_key 別テーブルへ隔離し，proven/disproven Final に昇格
しない — §7.1 の設計により「TT の Final = 経路非依存な真の結論」が不変条件として成立する)．

実バグは **verify 側**にあった (**verify 内 GHI**; maou_shogi 3.4.3 で根治):

- **機構**: `verify_proof` は path 上の祖先と一致した子を千日手 (受け方脱出) として `None` で
  拒否する．この None は**その経路でのみ有効**な経路依存結果だが，旧実装はこれを経路非依存の
  `memo` に書き込んでいた．証明候補が祖先へ戻る**循環近傍** (例: 王 3h↔2i・飛 3g↔2g の 4-ply
  循環) を先に訪れた文脈で書かれた None が残り，以後**本物の証明線**の verify が memo=None を
  引いて root まで連鎖 → **偽 Unknown** (偽証明ではなく検証の偽陰性)．探索順序 (閾値等) で
  「どの文脈が先に verify するか」が変わるため order-dependent に露出する．
- **fix (dep 伝播; §7.1 の `repetition_start < depth` ゲートと同型)**: 各再帰が「結果が依存した
  最浅の path index」を `dep_out` で親へ返す．None は依存が自 subtree 内で閉じた場合のみ memo
  し，祖先依存の None (と budget 枯渇 None) は memo せず別文脈で再検証させる．**Some は常に
  memo 可** — Some の導出木には循環拒否が入り込めない (AND は任意の None で None 化し，OR の
  None 候補は Some を支えない) ため構成的に経路非依存である．
- **STRICT verify の権威化 (3.4.2) は多層防御として維持**: `solve_impl` は `verify_proof`
  ([move-ordering-and-pv.md §9-b.1](move-ordering-and-pv.md)) を**最終権威**とし，**STRICT
  VERIFY が `Some(d)` を返したときのみ `Checkmate` を返す**．`None` (万一の偽証明 / verify
  budget 不完全) は偽の詰みを返さず `Unknown` を返す (**soundness > completeness**)．これに
  より閾値チューニングは健全性に影響しない (効率のみ; [threshold §3.1](threshold-control.md))．
  verify の実行コストは 2-tier 化で削減した ([move-ordering-and-pv.md §9-b.1](move-ordering-and-pv.md))．
