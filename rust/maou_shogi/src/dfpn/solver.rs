//! DfPnSolver 構造体と探索コアロジック．

#[cfg(feature = "visit_diag")]
use rustc_hash::FxHashMap;
use std::time::{Duration, Instant};

use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::moves::Move;
use crate::types::Color;

use super::entry::PNS_MAX_ARENA_NODES;
use super::{CheckCache, DISPROOF_THRESHOLD_ADAPTIVE};

/// path 配列の容量．depth の最大値(41) + マージン．
pub(super) const PATH_CAPACITY: usize = 48;

/// 詰将棋の探索結果．
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TsumeResult {
    /// 詰みが見つかった場合．手順を含む．
    Checkmate {
        moves: Vec<Move>,
        nodes_searched: u64,
    },
    /// 詰みは証明済みだが PV (手順)の復元に失敗した場合．
    ///
    /// TT エントリ上限 (`MAX_TT_ENTRIES_PER_POSITION`) 等により，
    /// 詰み証明後に手順を復元できないケースで返される．
    CheckmateNoPv { nodes_searched: u64 },
    /// 不詰の場合．
    NoCheckmate { nodes_searched: u64 },
    /// 探索制限に達した場合(nodes上限 or depth上限)．
    Unknown { nodes_searched: u64 },
}

/// Df-Pn ソルバー．
pub struct DfPnSolver {
    /// 最大ノード数．
    pub(super) max_nodes: u64,
    /// 実行時間制限．
    pub(super) timeout: Duration,
    /// 探索ノード数．
    pub(super) nodes_searched: u64,
    /// 探索開始時刻．
    pub(super) start_time: Instant,
    /// タイムアウトしたかどうか．
    pub(super) timed_out: bool,
    /// 攻め方の手番色(solve 時に設定)．
    pub(super) attacker: Color,
    /// 最短手数探索を行うかどうか(デフォルト: true)．
    ///
    /// true の場合，`solve()` は `complete_or_proofs()` を呼び出して
    /// 全 OR ノードの未証明子を追加証明し，最短手順を保証する．
    /// false の場合，最初に見つかった詰み手順をそのまま返す．
    pub(super) find_shortest: bool,
    /// find_shortest の per-iter refinement cap を override する (0 = formula
    /// `max(iter0*2, 200_000)`)．find_shortest=true の cost は探索 core でなくこの ~200K
    /// refinement floor が支配する．cap を下げれば cost が core に漸近するが，shorter mate が
    /// cap 内で見つかる問題では非最短を返しうる (find_shortest 品質との tradeoff)．
    pub(super) param_refine_iter_cap: u64,
    /// PV 復元フェーズで未証明子1つあたりに割り当てるノード予算(デフォルト: 1024)．
    ///
    /// 長手数の詰将棋で [`TsumeResult::CheckmateNoPv`] が返る場合，
    /// この値を増やすことで PV 復元の成功率が向上する．
    pub(super) pv_nodes_per_child: u64,
    /// TT GC 閾値: TT のエントリ数がこの値を超えると GC を実行する．
    ///
    /// 0 にすると GC を無効化する．
    /// デフォルトは 0(無効)．超長手数問題で OOM を防ぐ場合に設定する．
    /// 推奨値: 探索ノード数の 1/5〜1/2 程度(例: 100M ノードなら 20M〜50M)．
    pub(super) tt_gc_threshold: usize,
    /// 直前の `generate_defense_moves` で計算されたチェーンマスのビットボード．
    ///
    /// `mid()` 内で合駒がチェーンマスへのドロップかどうかを判定するために使用．
    /// 各 `mid()` 呼び出しで更新され，飛び駒の王手がない場合は空．
    pub(super) chain_bb_cache: Bitboard,
    /// 王手生成キャッシュ(E2 最適化)．
    pub(super) check_cache: CheckCache,
    /// サマリキャッシュの disproof ヒット回数 (tt_diag 診断用)．現在は未使用．
    #[cfg(feature = "tt_diag")]
    #[allow(dead_code)]
    pub(super) diag_pc_summary_disproof_hits: u64,
    /// PNS アリーナの最大ノード数．
    ///
    /// デフォルトは `PNS_MAX_ARENA_NODES` (5M)．大きくすると spin 率が下がる
    /// 代わりにメモリ消費が増える (1 ノード ≈ 80〜120 bytes)．
    /// 例: 10M で約 800〜1,200 MB．
    pub(super) param_pns_arena_max: usize,
    /// 深さ制限反証 (depth-limited disproof) を WorkingTT に格納する
    /// 最小 `remaining` 閾値．
    ///
    /// `remaining < param_disproof_remaining_threshold` の depth-limited
    /// disproof は格納をスキップする (path_dependent と confirmed disproof は
    /// 対象外)．
    ///
    /// - `DISPROOF_THRESHOLD_ADAPTIVE` (sentinel = `u16::MAX`) の場合，
    ///   solve() 時に `outer_solve_depth` に基づいて自動決定する．
    /// - 正の u16 値: 固定閾値として使用 (テスト・チューニング用)．
    /// - 0: スキップなし．
    ///
    /// デフォルトは `DISPROOF_THRESHOLD_ADAPTIVE` (depth-adaptive)．
    /// 深い問題での WorkingTT churn (eviction) の削減を狙いつつ，
    /// shallow 問題での退行を回避する．
    pub(super) param_disproof_remaining_threshold: u16,

    /// **visit_history dominance check**．
    ///
    /// true で，子展開時に `(child_pos_key, child_hand)` が現在の探索パス
    /// `path_pos_key[i] / path_hand[i]` の祖先で支配されている場合
    /// (`hand_gte_forward_chain(ancestor_hand, child_hand)` が同一 board_key で
    /// 成立) を loop と同じく経路依存不詰として扱う．
    ///
    /// **Why:** 攻め方が同じ局面に「より少ない持ち駒」で再訪している場合，
    /// 過去のより有利な状態でも詰めなかった事実から現在も詰まないと sound に
    /// 推論できる．chain aigoma で hand 多様性が指数爆発する局面の枝刈りに効く．
    ///
    /// **soundness:** 経路依存 (path_dependent) 反証として扱われるため
    /// 永続 TT エントリは汚染されない．`is_loop_child` と同じパスで処理されるため
    /// 既存の GHI 機構と整合する．
    ///
    /// デフォルトは false (既存挙動と同等)．効果測定のため opt-in で有効化する．
    pub(super) param_use_visit_history_dominance: bool,

    /// **OR disproof 交集合演算**．
    ///
    /// true で，OR ノード全子反証時 (`current_dn == 0`) の disproof_hand を
    /// 全子の `disproof_hand` の要素ごと min (交集合) で計算して store する．
    /// 既存実装は `att_hand` をそのまま渡しており，子の精密 disproof_hand
    /// 情報を活用していなかった．
    ///
    /// **Why:** 全子 (攻め手) で反証された場合，「これ以下の持ち駒では全子で詰まない」
    /// 最も強い disproof hand は子 disproof_hand の要素 min．これにより TT
    /// cross-branch ヒット率向上が期待される．
    ///
    /// **soundness:** 子の `get_disproof_hand` 自体は sound (TT lookup で
    /// `hand_gte_forward_chain` を確認済み)．要素 min はその制約を強める
    /// 方向の操作のため不正は導入しない．
    ///
    /// デフォルトは false．
    pub(super) param_use_handset_combination: bool,

    /// `DelayedMoveList`．
    ///
    /// true で，AND ノード multi-child loop で同マス合駒の chain (prev/next 双方向リスト) を
    /// 構築し，「prev が未解決なら next の子を skip」する semantics を適用する．
    /// 全合駒展開による pn 過大評価を抑止する．
    ///
    /// 関連: `delayed_move_list.rs`．
    pub(super) param_use_delayed_move_list: bool,

    /// path-aware DAG 補正．
    ///
    /// true で，AND multi-child loop で transposition DAG (child の TT-stored
    /// 親が現 path 上の先祖と一致する) を検出し，sum 集約から除外し
    /// max のみで集約する．
    ///
    /// 実装は runtime `parent_map` を使用．`or_insert` で
    /// child の最初の親のみ記録．以後の同 child 再訪問では parent_map
    /// から最初の親を取得し，それが path 上にあれば DAG と判定．
    pub(super) param_use_dag_correction: bool,



    /// 1+ε 閾値 epsilon (PN_UNIT スケール考慮で PN_UNIT が候補)．
    pub(super) param_threshold_epsilon: u32,
    /// first-visit 初期 pn から edge_cost を外し，純 `init_pn_dn_*` (df-pn+ 難易度推定)
    /// のみを使う．pn/dn を難易度推定に限定し move 選好は brief evaluation tie-break に
    /// 分離する．edge_cost を pn に折込むと pn が二重信号となり delta-sum 算術を歪める
    /// ため，それを切り離す．move_brief_eval tie-break は decouple 時も維持される (同じ ordering)．
    pub(super) param_decouple_edge_cost: bool,
    /// 1 手詰 detection を適用する最大 depth．
    /// `self.depth <= この値` のとき AND parent で child OR の 1 手詰を即 proven 化する．
    pub(super) param_obvious_final_max_depth: u32,
    /// find_shortest budget 探索で blocked-proven (stored_md > md_budget) の中間レンジ
    /// lookup を `(PN_UNIT, INF)` にする (proven entry = 絶対的に詰みなので dn=INF は sound)．
    /// false = `(PN_UNIT, PN_UNIT)` リセット (subtree 全再探索)．dual-range の find_shortest
    /// 収束改善用．
    pub(super) param_middle_range: bool,
    /// coherent scale mode．ON で seed を unit=2 スケールに揃え
    /// (pure 難易度推定 seed / (PN_UNIT/2))，local expansion の完全 comparer (δ tie-break)
    /// を有効化する．epsilon=1 / tca_clamp / deferred_penalty(/8) / is_sum_delta と併用して
    /// selectivity を coherent whole として効かせる．
    pub(super) param_scale: bool,
    /// local expansion の完全 comparer (δ tie-break + amount) を
    /// unit-16 でも有効化する (scale とは独立)．
    pub(super) param_full_comparer: bool,
    /// full_comparer の δ tie-break を **AND ノードのみ**に限定する．
    /// OR の δ は王手選択を変え非最短 (Mate-31) を誘発するが deep breadth 主因は AND fan-out なので
    /// AND のみ δ を効かせれば最短 29 維持 + defender selectivity を得られると期待した．
    ///
    /// 実測では Mate-29 は保つが AND-δ は breadth を**増やす**．full_comparer の効率は OR-δ 由来で
    /// Mate-31 と不可分であり，δ は「証明が楽な (が長い) 詰み」へ誘導する．最短 29 には使えないため
    /// default OFF．
    pub(super) param_comparer_and_only: bool,
    /// OR 子展開で合駒 capture cross-deduction を有効化する．
    /// first-visit OR 子が取り王手で既証明局面に到達するなら，hand dominance で
    /// 展開せず証明確定する (`try_capture_tt_proof`)．合駒共有により深い ply の
    /// 合駒 fan-out を削る目的．
    pub(super) param_capture_dedup: bool,
    /// 子の seed pn を king-escape 難易度で enrich する．
    /// 詰める側 (`attacker.opponent()`) の玉の safe_escapes を pn に加算し，玉の機動性を
    /// 奪う王手を seed 段階で優先させる狙いだった．
    ///
    /// 実測では悪化する: ordering が改善するなら deep commitment (eps 大) が breadth を
    /// 減らすはずが逆に爆発した．rich_seed は front child を正しくしておらず，むしろ edge_cost
    /// の差別化に noise を加えて悪化させる (seed quality は探索量の lever でない)．
    /// 真因は詰め手の move-ordering 品質で，安価な heuristic では埋まらない．default OFF．
    pub(super) param_rich_seed: bool,
    /// AND ノードで一般 合駒 cross-deduction を有効化する．
    /// drop 合駒 defense が refute (child pn=0) された直後に，同一マスの兄弟 drop 合駒を
    /// `cross_deduce_children` (capture 後局面の hand-dominance proof 転用) で展開せず証明する狙い．
    ///
    /// 実測では小幅に悪化する: Mate-29 維持 (sound) だが deep AND breadth は不変．
    /// deep AND fan-out は同一マス drop 兄弟の proof 共有では削れない (defender breadth は
    /// king 逃げ/capture/別マス合駒が主)．default OFF．
    pub(super) param_cross_dedup: bool,
    /// AND ノード move 生成に PNS の 無駄合い filter を適用する．
    /// 通常は `movegen::generate_legal_moves` で**全**合法手 (無駄合い含む) を生成するが，
    /// `generate_defense_moves_inner` (pns.rs) は futile 合駒を除外し chain マスは歩のみ生成する
    /// 検証済の 無駄合い filter．
    ///
    /// 実測ではほぼ無変化 (Mate-29)．deep AND breadth は futile 無駄合い でなく genuine な防御
    /// (玉移動/駒取り) + 攻め方の drop 王手の多様性由来で，無駄合い filter では削れない．
    pub(super) param_muda_filter: bool,
    /// TCA extension formula を `max(thpn, pn+1)` 形式にするか．
    pub(super) param_tca_clamp: bool,
    /// root level IDS (1.7× threshold growth) 有効．
    pub(super) param_root_ids_enable: bool,
    /// parity DML (非駒打ち成/不成 deferral)．**default true**．
    /// true で `build_delayed_chain` が成/不成ペアも chain 化する版になり，
    /// 歩/角/飛/香(rank2/8) の成・不成ペアを OR/AND 両方で chain 化して breadth を削る．
    /// false にすると深い局面で depth-limit 偽反証の TT 汚染による false NoMate を返しうる．
    /// その根治は [`DfPnSolver::param_scope_disproof`] が担い，dml は guidance robustness 改善として
    /// 併用する．
    pub(super) param_dml: bool,
    /// 集約 disproof (curr.dn==0) を絶対 (REMAINING_INFINITE = ProvenTT confirmed) ではなく
    /// **remaining scope** で store する (**default true**)．depth-limit 偽反証
    /// (`look_up_pn_dn_impl` remaining==0 → (INF,0)) が伝播して生じた集約 disproof が WorkingTT に
    /// scope 付きで入り，より深い ply (= remaining 大) の transposition lookup では再探索される
    /// (`e.remaining()>=remaining`)．これで horizon disproof による false NoMate を根治する．
    /// false にすると horizon disproof を confirmed 化し TT を汚染する soundness バグになる．
    pub(super) param_scope_disproof: bool,
    /// `is_sum_delta_node` 判定を適用するか．
    /// true で OR ノードの香成/不成 near-duplicate child を **max 集約** に切替え，δ値の
    /// 二重計上 (= breadth 発散) を抑える．この分岐を入れないと香車王手の多い
    /// 局面で delta が過大になり distinct-position breadth が発散する．
    /// default false (baseline 不変)．
    pub(super) param_is_sum_delta_node: bool,
    /// 極小証明駒 (minimal proof hand) を proof store に適用するか．
    /// true で proven 局面を「実際の攻め方持ち駒」ではなく [`super::proof_hand`] が計算する
    /// **極小証明駒** (子の要素 max + other-evasions 補正) で store する．
    /// hand-dominance (`hand_gte_forward_chain`) の集約が強まり proof tree を圧縮する狙い．
    /// soundness-critical のため default false (baseline byte-identical)．
    pub(super) param_minimal_proof_hand: bool,
    /// 極大反証駒 (maximal disproof hand) を disproof store に適用するか．
    /// true で不詰局面を「実際の攻め方持ち駒」ではなく [`super::proof_hand`] が計算する
    /// **極大反証駒** (子の要素 min + other-checks 補正) で store する．
    /// disproof の hand-dominance が広がり失敗ラインを刈る狙い．
    /// soundness-critical (誤ると false-NoMate) のため default false (baseline byte-identical)．
    /// `param_minimal_proof_hand` と独立に A/B 可能．
    pub(super) param_minimal_disproof_hand: bool,
    /// repetition (千日手) taint + targeted disproof scoping を使うか．
    /// true で，disproof (dn==0) が cycle に依存する (subtree に `repetition_start < 自 ply` の
    /// 千日手がある) 場合のみ scope 限定で store し，それ以外は confirmed (`REMAINING_INFINITE`)
    /// で store する．`param_scope_disproof` の全 disproof を一律 scope 限定する再探索を，
    /// path-dependent なものだけに精密化して置換える狙い．
    /// soundness-critical (誤ると false-NoMate) のため default false (baseline byte-identical)．
    pub(super) param_repetition: bool,
    /// deferred penalty 除数 (0=無効, 8=有効)．
    pub(super) param_deferred_penalty_denom: u32,
    /// deferred penalty `.max(1)` floor．
    pub(super) param_deferred_penalty_floor: bool,
    /// TCA gate を is_shallow ベースにするか (true=is_shallow, false=!is_first_visit)．
    pub(super) param_tca_use_shallow_gate: bool,
    /// 診断: deferred penalty 発火フレーム数．
    pub(super) diag_deferred_frames: u64,
    /// mid 探索パス上の board.hash → ply (千日手検出 + 参照祖先 ply 特定用)．
    /// 連続パス保持と同型の flat スタック ([`super::path_stack::PathStack`])．
    /// `FxHashMap` の散在 DRAM アクセスを排して memory-bound を解消する (探索不変)．
    pub(super) path_depths: super::path_stack::PathStack,
    /// mid 探索ノード数 (= emplace 呼び出し回数)．
    pub(super) nodes: u64,
    /// mid double-count elimination 用の明示的 expansion stack．
    /// 各 search_impl frame が自 LocalExpansion を push/truncate し，祖先を辿れるようにする．
    pub(super) expansion_stack: Vec<super::local_expansion::LocalExpansion>,
    /// mid build_expansion の per-node Vec 再利用 pool (node pop で返却・再取得)．
    /// ヒープ alloc/free を削減する (探索不変; [`super::mid::BufPool`])．
    pub(super) expansion_buf_pool: super::mid::BufPool,
    /// mid double-count elimination 発火数 (診断用)．
    pub(super) dag_fires: u64,
    /// mid 探索 path 上の hand-dominance 反復検出．
    /// position_key → 現探索 path 上の祖先 `(attacker_hand, depth)` のスタック．
    /// 子局面が同一 board_key かつ攻め方持駒が祖先以下 (= 劣位) なら反復として刈る．
    /// `board.hash` だけの exact 千日手 (`path_depths`) を持駒 superset 方向へ一般化する．
    /// flat スタック実装 ([`super::path_stack::DomPathStack`])．
    pub(super) dom_path: super::path_stack::DomPathStack,
    /// mid dominance pruning 発火数 (診断用)．
    pub(super) dom_fires: u64,
    /// mid path dominance を有効化するか (`V4_DOM` env で opt-in; default off)．
    /// 現状の実装は node 増となるため default では除外する．
    pub(super) param_path_dominance: bool,
    /// 診断: deferred penalty 合計値．
    pub(super) diag_deferred_penalty_sum: u64,
    /// 診断: has_old_child=true (is_shallow gate)．
    pub(super) diag_tca_shallow_fire: u64,
    /// 診断: has_old_child=false だが旧ロジックでは true になるケース．
    pub(super) diag_tca_shallow_would_fire: u64,


    /// TCA inc_flag を有効化する．
    ///
    /// true で，mid() が「loop child を検出した時に inc_flag を increment し，
    /// inc_flag > 0 の間は閾値拡張 (eff_pn_th/eff_dn_th) を maintain する」
    /// 形に変更．既定では loop_child_count > 0 を **同一 mid() フレーム内で
    /// 検出された場合のみ** 適用するが，このフラグは inc_flag を recursion で伝播する．
    ///
    /// 効果: 深い transposition chain で，先祖フレームで検出された
    /// loop が子孫フレームの閾値にも反映され，wrong branch から早期復帰する．
    pub(super) param_use_tca: bool,



    /// 診断: root (ply 0) trace．
    /// true で，mid() の OR/AND multi-child loop が ply==0 で iterate するたびに
    /// 主要 metrics を eprintln! でダンプする (feature flag 不要)．
    ///
    /// 出力例 (interval=10000 nodes):
    /// ```text
    /// [root_trace ply0 iter=42 nodes=100000] children=23 best_idx=3 best_move=S*7i
    ///   current_pn=156 current_dn=89 pn_th=200 dn_th=200
    ///   child[0] move=S*7i pn=124 dn=45 (root_visits=520)
    ///   ...
    /// ```
    pub(super) param_root_trace: bool,

    /// 診断: trace 対象の ply (default 0 = root)．
    pub(super) param_trace_ply: u32,

    /// 診断: 全 path 上の child の pn/dn を 1 dump 当たり最大何件出すか．
    pub(super) param_trace_full_children: bool,

    /// 診断: TT lookup hit rate を計測する．`param_tt_diag` で有効化．
    /// `look_up_pn_dn` 呼び出しごとに以下を更新:
    /// - `diag_tt_lookups`: 総呼び出し回数
    /// - `diag_tt_misses`: 初期値 (PN_UNIT, PN_UNIT, 0) が返った回数
    /// - `diag_tt_proven`: (0, INF, _) が返った回数
    /// - `diag_tt_disproven`: (INF, 0, _) が返った回数
    /// - `diag_tt_working`: その他 (working entry hit)．
    pub(super) param_tt_lookup_diag: bool,
    // Cell で `&self` の look_up_pn_dn からも更新可能にする
    pub(super) diag_tt_lookups: std::cell::Cell<u64>,
    pub(super) diag_tt_misses: std::cell::Cell<u64>,
    pub(super) diag_tt_proven: std::cell::Cell<u64>,
    pub(super) diag_tt_disproven: std::cell::Cell<u64>,
    pub(super) diag_tt_working: std::cell::Cell<u64>,

    /// 診断: per-depth proven 蓄積カウンタ．
    /// `store_proof_with_tag` 呼び出し時に `path_len` (= 現在の ply) を
    /// インデックスとして increment．solve() 完了後に proof がどの深さで
    /// 累積したかのヒストグラムが取れる．
    pub(super) diag_proven_per_ply: [u64; 64],

    /// 診断: AND ノード multi-child loop 完了時の coverage 統計．
    /// `(proven_count, total_children)` のヒストグラム．
    /// proven_count ratio が低いまま loop が exit → AND coverage が不足．
    pub(super) diag_and_visit_count: u64,
    pub(super) diag_and_proven_sum: u64,  // proven_count の合計
    pub(super) diag_and_total_sum: u64,   // total_children の合計
    pub(super) diag_and_zero_proven: u64, // proven_count == 0 で exit した回数
    pub(super) diag_and_full_proven: u64, // proven_count == total で exit (本来は ProveAND して exit)

    /// 診断: OR scan coverage 統計 (AND と対称)．
    pub(super) diag_or_visit_count: u64,
    pub(super) diag_or_proven_count_visits: u64, // proven child があった visits の数
    pub(super) diag_or_total_sum: u64,

    /// 診断: per-position revisit count．`(pos_key, or_node) → visit_count`．
    /// max 1M entries に制限してメモリ暴走防止．
    pub(super) diag_pos_visits: rustc_hash::FxHashMap<u64, u32>,
    pub(super) diag_pos_visits_capped: bool, // 1M 超えたら collection 停止

    /// AND node exhaustive defender prove．
    /// default false．true で AND multi-child loop の best_idx 選択を
    /// "未訪問 defender 優先 round-robin" に切り替える．具体的には:
    /// - 全 children のうち cpn != 0 (まだ proven でない) defender の中で
    ///   `self.exhaustive_and_rr_counter % unproven_count` 番目を選択
    /// - これにより同じ defender に固定せず全 defender を巡回 prove
    pub(super) param_use_exhaustive_and: bool,

    /// per-AND-position の proven defender bitmap．
    /// `pos_key → u64 bitmap` で「どの children index がすでに proven 化されたか」
    /// を persistent 追跡．`param_use_and_proven_bitmap=true` で有効化．
    /// AND scan で bitmap 上の bit が立つ defender は **selection 候補から除外**．
    /// cpn==0 を毎回 lookup で判定する代わりに memoize し，
    /// remaining mismatch や TT churn による誤判定からも保護する．
    pub(super) param_use_and_proven_bitmap: bool,

    /// root child_pn_th の絶対 floor．
    /// 0 (default) で無効．> 0 で root (ply=0) の OR child_pn_th を最低
    /// この値まで引き上げる．これにより 1 つの child に深く commit するための
    /// pn 予算を保証する．推奨値の出発点: 100_000 (= 6250 * PN_UNIT)．
    pub(super) param_root_child_pn_floor: u32,

    /// OR ノード best_idx 選択の mate path commitment．
    /// default false で従来挙動 (argmin pn)．true で:
    /// argmin pn の tie 時 (`pn == best_pn`) に max(dn) で tie-break する．
    /// = defender の抵抗が強い attack を優先 → 探索 commit 強化．
    pub(super) param_or_dn_tiebreak: bool,


    /// 診断: periodic GC (overflow-based working TT GC) を無効化する．
    /// default false (= GC fire OK)．true で `nodes_searched % 100_000 == 0` の
    /// GC トリガを skip する．catastrophic forgetting の検証用．
    pub(super) param_disable_periodic_gc: bool,

    /// 診断: IDS の浅い depth 反復を skip して full depth から開始する．
    /// default false (= 通常 IDS depth=2,4,6,...)．true で `ids_depth=saved_depth` 直行．
    /// IDS による TT 再評価 (remaining 違いで初期値復活) の影響を排除する検証用．
    pub(super) param_skip_ids_shallow: bool,

    /// root_trace の dump 間隔 (nodes)．`param_root_trace=true` のときに使用．
    pub(super) root_trace_interval: u64,

    /// per-move 差別化．
    ///
    /// true で，OR child の `edge_cost_or` を `edge_cost_or_with_support` に
    /// 切り替え．`compute_checkers_at` で to_sq の attack/defense support を算出し，
    /// 受け駒 ≥ 2 のマスを後回し，攻め支援 > 受け支援のマスを優先する．
    pub(super) param_use_per_move_support: bool,

    // === M-1 refutable check fast path 改善フラグ ===
    /// **F1**: `all_checks_refutable_recursive_inner` で false 確定 check
    /// で早期 return せず，全 check を評価して store する．
    /// partial coverage の積み上げで fast path 発火率向上を狙う．
    /// trade-off: recursive cost +50%〜100% の見込み．
    pub(super) param_refut_full_eval: bool,
    /// **F2**: fast path で部分 match した場合，残り missing check のみを
    /// recursive で評価し，全 check 完成度を効率的に達成する．
    pub(super) param_refut_partial_recursion: bool,
    /// **F3**: OR レベル refutable 成功も `refutable_check_succeeded` cache
    /// に格納する．false NoMate の根源になりうるため，
    /// `(pos_key, outer_solve_depth)` でタグ付けし IDS depth ごとに分離．
    pub(super) param_refut_or_success_cache: bool,
    /// **F4**: fast path lookup を ProvenTT のみから WorkingTT も含む
    /// `look_up` に拡張．depth-limited disproof (rem>=floor) も match
    /// として count し，coverage を向上．
    pub(super) param_refut_extended_lookup: bool,
    /// TT 診断: 監視対象の ply(0 = 無効)．
    ///
    /// 指定 ply で MID ループの再帰前後に TT サイズを出力し，
    /// エントリ爆増の原因を特定する．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_ply: u32,
    /// TT 診断: 監視対象の手(USI 形式，例: "P*7g")．
    ///
    /// 空文字列の場合は ply のみでフィルタする．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_move_usi: String,
    /// TT 診断: MID ループの反復回数上限(0 = 無制限)．
    ///
    /// 爆増が起きる手を特定した後，少数回の反復に絞って詳細を確認する．
    #[cfg(feature = "tt_diag")]
    pub(super) diag_max_iterations: u32,
    /// TT 診断: 境界層 filter が発火した MID 数 (現在は未使用)．
    #[cfg(feature = "tt_diag")]
    #[allow(dead_code)]
    pub(super) diag_alpha_x_filter_applied: u64,
    /// IDS-17 無効化フラグ．
    /// true にすると saved_depth 20-26 での depth=16→17 挿入をスキップする．
    /// デフォルト false = IDS-17 有効 (depth=17 を明示的に経由)．
    pub(super) param_no_ids17: bool,
    /// solve() 呼び出し間で ProvenTT を保持するフラグ．
    /// true のとき，solve() 冒頭でWorkingTT のみクリアし ProvenTT を引き継ぐ．
    /// 逐次 backward 解析で前のソルブの ProvenTT を再利用する際に使用する．
    pub(super) preserve_proven_tt: bool,
    /// refutable check の再帰深さ (デフォルト 5)．
    pub(super) param_refutable_depth: u32,
    /// refutable check の呼び出し回数上限 (デフォルト 10,000)．
    pub(super) param_refutable_call_limit: u32,
    /// PNS 初期フェーズ専用の refutable call limit (デフォルト 500)．
    /// MID では param_refutable_call_limit (=10,000) を使用する．
    pub(super) param_pns_refutable_call_limit: u32,
    /// 境界層 PNS 責任転嫁の残り呼出予算 (現在は未使用)．
    #[allow(dead_code)]
    pub(super) a6_boundary_pns_calls_remaining: u32,
    /// ノード制限到達時点の WorkingTT pn/dn 分布スナップショット (分析用)．
    ///
    /// max_nodes に達した最初の mid() 呼び出しで `collect_working_pn_dn_dist()` を
    /// 実行して保存する．None の場合はノード制限に達していないことを示す．
    pub(super) pn_dn_snapshot: Option<([u64; 32], [u64; 32], Vec<u64>)>,
    /// IDS 各 depth 反復終了時点の WorkingTT pn/dn 分布スナップショット列 (分析用)．
    ///
    /// `mid_fallback` 内で各 IDS depth のMID 完了後，TT 遷移 (retain_working_intermediates
    /// / clear_working) の直前に収集する．
    /// `(ids_depth, nodes_searched, elapsed_secs, pn_hist, dn_hist, joint_hist)`
    pub(super) pn_dn_per_depth: Vec<(u32, u64, f64, [u64; 32], [u64; 32], Vec<u64>)>,
    /// 各局面 (board.hash) の MID 訪問回数 (visit_diag feature 時のみ存在)．
    #[cfg(feature = "visit_diag")]
    pub(super) visit_counts: FxHashMap<u64, u32>,
    /// 各局面が初めて訪問された ply (visit_diag feature 時のみ存在)．
    #[cfg(feature = "visit_diag")]
    pub(super) visit_first_ply: FxHashMap<u64, u8>,
}

impl DfPnSolver {
    /// 新しいソルバーを生成する(タイムアウト 300 秒)．
    pub fn new(depth: u32, max_nodes: u64, draw_ply: u32) -> Self {
        Self::with_timeout(depth, max_nodes, draw_ply, 300)
    }

    /// タイムアウト指定付きでソルバーを生成する．
    ///
    /// # Panics
    ///
    /// `depth >= PATH_CAPACITY`(48)の場合パニックする．
    pub fn with_timeout(depth: u32, max_nodes: u64, _draw_ply: u32, timeout_secs: u64) -> Self {
        assert!(
            (depth as usize) < PATH_CAPACITY,
            "depth {} exceeds path capacity {}",
            depth,
            PATH_CAPACITY,
        );
        DfPnSolver {
            max_nodes,
            timeout: Duration::from_secs(timeout_secs),
            find_shortest: true,
            param_refine_iter_cap: 0,
            pv_nodes_per_child: 1024,
            chain_bb_cache: Bitboard::EMPTY,
            check_cache: CheckCache::new(),
            #[cfg(feature = "tt_diag")]
            diag_pc_summary_disproof_hits: 0,
            param_refut_full_eval: false,
            param_refut_partial_recursion: false,
            // F3 (or_success_cache): default ON．
            // 深い問題で nodes / time を大幅削減し，shallow 問題では no-op (safe)．
            // full_hash keying で false positive を防止済み．
            param_refut_or_success_cache: true,
            param_refut_extended_lookup: false,
            tt_gc_threshold: 0,
            param_pns_arena_max: PNS_MAX_ARENA_NODES,
            // default: ADAPTIVE．refutable depth floor により false-NoMate を
            // 起こさず adaptive を default にできる．
            // depth ≤ 19: 0, depth 20-22: 1, depth ≥ 23: 3．
            param_disproof_remaining_threshold: DISPROOF_THRESHOLD_ADAPTIVE,
            param_use_visit_history_dominance: true,
            // default ON．常時 OR disproof_hand を要素 min で集約する．
            // OFF にしたい場合は `set_use_handset_combination(false)` を明示呼出．
            param_use_handset_combination: true,
            // DelayedMoveList．AND ノードで同マス合駒 chain の prev が未解決なら
            // next を skip する．default ON．
            param_use_delayed_move_list: true,
            // path-aware DAG 補正．opt-in (default false)．
            param_use_dag_correction: false,
            param_threshold_epsilon: 2,
            param_decouple_edge_cost: false,
            param_obvious_final_max_depth: 25,
            param_middle_range: false,
            param_scale: false,
            param_full_comparer: false,
            param_capture_dedup: false,
            param_rich_seed: false,
            param_cross_dedup: false,
            param_comparer_and_only: false,
            param_muda_filter: false,
            param_tca_clamp: false,
            param_root_ids_enable: false,
            param_dml: true,
            param_scope_disproof: true,
            param_is_sum_delta_node: false,
            param_minimal_proof_hand: false,
            param_minimal_disproof_hand: false,
            param_repetition: false,
            param_deferred_penalty_denom: 0,
            param_deferred_penalty_floor: false,
            param_tca_use_shallow_gate: false,
            diag_deferred_frames: 0,
            path_depths: super::path_stack::PathStack::new(),
            nodes: 0,
            expansion_stack: Vec::new(),
            expansion_buf_pool: super::mid::BufPool::default(),
            dag_fires: 0,
            dom_path: super::path_stack::DomPathStack::new(),
            dom_fires: 0,
            param_path_dominance: false,
            diag_deferred_penalty_sum: 0,
            diag_tca_shallow_fire: 0,
            diag_tca_shallow_would_fire: 0,
            param_use_tca: false,
            param_root_trace: false,
            param_trace_ply: 0,
            param_trace_full_children: false,
            param_tt_lookup_diag: false,
            diag_tt_lookups: std::cell::Cell::new(0),
            diag_tt_misses: std::cell::Cell::new(0),
            diag_tt_proven: std::cell::Cell::new(0),
            diag_tt_disproven: std::cell::Cell::new(0),
            diag_tt_working: std::cell::Cell::new(0),
            diag_proven_per_ply: [0u64; 64],
            param_root_child_pn_floor: 0,
            diag_and_visit_count: 0,
            diag_and_proven_sum: 0,
            diag_and_total_sum: 0,
            diag_and_zero_proven: 0,
            diag_and_full_proven: 0,
            diag_or_visit_count: 0,
            diag_or_proven_count_visits: 0,
            diag_or_total_sum: 0,
            diag_pos_visits: rustc_hash::FxHashMap::default(),
            diag_pos_visits_capped: false,
            param_use_exhaustive_and: false,
            param_use_and_proven_bitmap: false,
            param_or_dn_tiebreak: false,
            root_trace_interval: 10_000,
            param_disable_periodic_gc: false,
            param_skip_ids_shallow: false,
            // per-move attack/defense support 差別化．opt-in (default false)．
            param_use_per_move_support: false,
            nodes_searched: 0,
            start_time: Instant::now(),
            timed_out: false,
            attacker: Color::Black,
            #[cfg(feature = "tt_diag")]
            diag_ply: 0,
            #[cfg(feature = "tt_diag")]
            diag_move_usi: String::new(),
            #[cfg(feature = "tt_diag")]
            diag_max_iterations: 0,
            #[cfg(feature = "tt_diag")]
            diag_alpha_x_filter_applied: 0,
            param_no_ids17: false,
            param_refutable_depth: Self::DEFAULT_REFUTABLE_DEPTH,
            param_refutable_call_limit: Self::DEFAULT_REFUTABLE_CALL_LIMIT,
            param_pns_refutable_call_limit: 500,
            a6_boundary_pns_calls_remaining: 0,
            pn_dn_snapshot: None,
            pn_dn_per_depth: Vec::new(),
            preserve_proven_tt: false,
            #[cfg(feature = "visit_diag")]
            visit_counts: FxHashMap::default(),
            #[cfg(feature = "visit_diag")]
            visit_first_ply: FxHashMap::default(),
        }
    }

    /// solve() 呼び出し間で ProvenTT を引き継ぐフラグを設定する．
    /// true にすると solve() 冒頭で WorkingTT のみクリアし ProvenTT を保持する．
    pub fn set_preserve_proven_tt(&mut self, preserve: bool) -> &mut Self {
        self.preserve_proven_tt = preserve;
        self
    }

    /// 重複訪問レポートを返す (visit_diag feature 時のみ利用可)．
    ///
    /// 各局面の訪問回数分布と上位 `top_n` 局面をまとめた文字列を返す．
    #[cfg(feature = "visit_diag")]
    pub fn visit_summary(&self, top_n: usize) -> String {
        use std::fmt::Write as _;
        let total_visits: u64 = self.visit_counts.values().map(|&c| c as u64).sum();
        let unique_positions = self.visit_counts.len();
        let revisits = total_visits.saturating_sub(unique_positions as u64);
        let revisit_rate = if total_visits > 0 {
            100.0 * revisits as f64 / total_visits as f64
        } else {
            0.0
        };

        let mut sorted: Vec<(u64, u32, u8)> = self
            .visit_counts
            .iter()
            .map(|(&h, &cnt)| {
                let ply = self.visit_first_ply.get(&h).copied().unwrap_or(255);
                (h, cnt, ply)
            })
            .collect();
        sorted.sort_unstable_by(|a, b| b.1.cmp(&a.1));

        // ply 別集計: (total_visits, unique_positions) per ply
        let mut ply_total = [0u64; 64];
        let mut ply_unique = [0u64; 64];
        for &(_, cnt, ply) in &sorted {
            if (ply as usize) < 64 {
                ply_total[ply as usize] += cnt as u64;
                ply_unique[ply as usize] += 1;
            }
        }

        let mut out = String::new();
        writeln!(out, "=== 重複訪問レポート ===").unwrap();
        writeln!(out, "総訪問数    : {}", total_visits).unwrap();
        writeln!(out, "ユニーク局面: {}", unique_positions).unwrap();
        writeln!(out, "重複訪問数  : {} ({:.1}%)", revisits, revisit_rate).unwrap();
        writeln!(out).unwrap();

        writeln!(out, "--- ply 別集計 (revisit 率が高い順) ---").unwrap();
        let mut ply_rows: Vec<(usize, u64, u64)> = ply_total
            .iter()
            .enumerate()
            .filter(|(_, &t)| t > 0)
            .map(|(p, &t)| {
                let u = ply_unique[p];
                (p, t, u)
            })
            .collect();
        ply_rows.sort_unstable_by(|a, b| {
            let ra = a.1.saturating_sub(a.2) * b.2;
            let rb = b.1.saturating_sub(b.2) * a.2;
            rb.cmp(&ra)
        });
        for (ply, total, unique) in &ply_rows {
            let rev = total.saturating_sub(*unique);
            let rate = if *total > 0 {
                100.0 * rev as f64 / *total as f64
            } else {
                0.0
            };
            writeln!(
                out,
                "  ply {:2}: total={:8}  unique={:7}  revisits={:7}  rate={:.1}%",
                ply, total, unique, rev, rate
            )
            .unwrap();
        }
        writeln!(out).unwrap();

        writeln!(out, "--- 上位 {} 局面 (訪問回数順) ---", top_n).unwrap();
        for (rank, &(hash, cnt, ply)) in sorted.iter().take(top_n).enumerate() {
            writeln!(
                out,
                "  {:3}. hash={:#018x}  count={:6}  first_ply={}",
                rank + 1,
                hash,
                cnt,
                ply
            )
            .unwrap();
        }

        // 訪問回数の分布ヒストグラム
        writeln!(out).unwrap();
        writeln!(out, "--- 訪問回数分布 ---").unwrap();
        let thresholds = [1u32, 2, 5, 10, 20, 50, 100, 500, 1000];
        let mut prev = 0u32;
        for &th in &thresholds {
            let count = sorted
                .iter()
                .filter(|&&(_, c, _)| c >= prev + 1 && c <= th)
                .count();
            writeln!(out, "  {}-{} 回: {} 局面", prev + 1, th, count).unwrap();
            prev = th;
        }
        let count_over = sorted.iter().filter(|&&(_, c, _)| c > prev).count();
        writeln!(out, "  {}+ 回: {} 局面", prev + 1, count_over).unwrap();

        out
    }

    /// PNS アリーナの最大ノード数を設定する．
    ///
    /// デフォルトは `PNS_MAX_ARENA_NODES` (5M)．大きくすると arena spin 率が
    /// 下がる代わりにメモリ消費が増える．`min_value` (1024) 未満は丸められる．
    pub fn set_pns_arena_max(&mut self, max_nodes: usize) {
        self.param_pns_arena_max = max_nodes.max(1024);
    }

    /// visit_history dominance check を有効化する．
    ///
    /// true で子展開時に経路上の祖先 hand が現 hand を支配する局面を
    /// 経路依存不詰として枝刈りする．chain aigoma で hand 多様性が爆発する
    /// 局面の枝刈り効果を期待．デフォルトは false．
    ///
    /// 詳細: [`DfPnSolver::param_use_visit_history_dominance`]．
    pub fn set_use_visit_history_dominance(&mut self, on: bool) {
        self.param_use_visit_history_dominance = on;
    }

    /// OR disproof 交集合演算を有効化する．
    ///
    /// true で，OR ノード全子反証時の disproof_hand を子 disproof_hand の
    /// 要素ごと min (交集合) で計算する．既存実装は `att_hand` 単一伝播のみ．
    /// デフォルトは false．
    ///
    /// 詳細: [`DfPnSolver::param_use_handset_combination`]．
    pub fn set_use_handset_combination(&mut self, on: bool) {
        self.param_use_handset_combination = on;
    }

    /// `DelayedMoveList` を有効化/無効化する．
    ///
    /// 有効時，AND ノード multi-child loop で同マス合駒 chain を構築し，
    /// 「prev が未解決なら next を skip」する semantics を適用．
    /// 全合駒展開による pn 過大評価を抑止し，合駒可能局面のノード数削減を狙う．
    ///
    /// デフォルト OFF．関連: [`DfPnSolver::param_use_delayed_move_list`]．
    pub fn set_use_delayed_move_list(&mut self, on: bool) {
        self.param_use_delayed_move_list = on;
    }

    /// TCA inc_flag 機構の有効化．
    /// default false．有効化すると mid() loop で inc_flag を propagate し，
    /// 深い transposition chain での threshold extension を継続させる．
    pub fn set_use_tca(&mut self, on: bool) -> &mut Self {
        self.param_use_tca = on;
        self
    }

    /// 診断: periodic GC を無効化する．
    /// default false．catastrophic forgetting の検証用．
    /// 大規模 working TT で OOM の可能性があるので long-running 検証時注意．
    pub fn set_disable_periodic_gc(&mut self, on: bool) -> &mut Self {
        self.param_disable_periodic_gc = on;
        self
    }

    /// 診断: IDS の浅い depth 反復 (depth=2,4,6,...) を skip し，
    /// 最初から `saved_depth` full depth で MID を実行する．
    /// default false．`remaining` 違いによる TT 再評価コストを排除する検証用．
    pub fn set_skip_ids_shallow(&mut self, on: bool) -> &mut Self {
        self.param_skip_ids_shallow = on;
        self
    }

    /// 診断: per-ply mid() trace の有効化．
    /// default false．有効化すると mid() の OR/AND multi-child loop で
    /// `ply == param_trace_ply` のたびに root_trace_interval ノード経過ごとに
    /// per-iteration 状態を eprintln! でダンプする．
    pub fn set_root_trace(&mut self, on: bool, interval_nodes: u64) -> &mut Self {
        self.param_root_trace = on;
        if interval_nodes > 0 {
            self.root_trace_interval = interval_nodes;
        }
        self
    }

    /// 診断: trace 対象の ply を設定．`set_root_trace` と組合せて使用．
    /// default 0 (= root)．例: 1 で root child 入った直後 (defender's first move) を trace．
    pub fn set_trace_ply(&mut self, ply: u32) -> &mut Self {
        self.param_trace_ply = ply;
        self
    }

    /// 診断: trace 時に top-10 child のみでなく全 children を出力する．
    pub fn set_trace_full_children(&mut self, on: bool) -> &mut Self {
        self.param_trace_full_children = on;
        self
    }

    /// 診断: TT lookup hit rate を計測する．
    pub fn set_tt_lookup_diag(&mut self, on: bool) -> &mut Self {
        self.param_tt_lookup_diag = on;
        self
    }

    /// 診断: solve() 完了後に TT lookup 統計を取得．
    /// `(total, miss, proven, disproven, working)` のタプル．
    pub fn get_tt_lookup_stats(&self) -> (u64, u64, u64, u64, u64) {
        (
            self.diag_tt_lookups.get(),
            self.diag_tt_misses.get(),
            self.diag_tt_proven.get(),
            self.diag_tt_disproven.get(),
            self.diag_tt_working.get(),
        )
    }

    /// 診断: per-depth proven 蓄積ヒストグラム取得．
    /// 配列 `[u64; 64]` で index = ply, 値 = 当該 ply で proven 確定回数．
    pub fn get_proven_per_ply(&self) -> [u64; 64] {
        self.diag_proven_per_ply
    }

    /// deferred penalty 除数を設定 (0=無効, 8=有効)．
    pub fn set_deferred_penalty_denom(&mut self, denom: u32) -> &mut Self {
        self.param_deferred_penalty_denom = denom;
        self
    }

    /// deferred penalty `.max(1)` floor を設定．
    pub fn set_deferred_penalty_floor(&mut self, floor: bool) -> &mut Self {
        self.param_deferred_penalty_floor = floor;
        self
    }

    /// 1+ε threshold epsilon を設定．
    pub fn set_threshold_epsilon(&mut self, eps: u32) -> &mut Self {
        self.param_threshold_epsilon = eps.max(1);
        self
    }

    /// parity DML (非駒打ち成/不成 deferral) を設定．
    /// 詳細: [`DfPnSolver::param_dml`]．
    pub fn set_dml(&mut self, on: bool) -> &mut Self {
        self.param_dml = on;
        self
    }

    /// 集約 disproof を remaining scope で store する．
    /// 詳細: [`DfPnSolver::param_scope_disproof`]．
    pub fn set_scope_disproof(&mut self, on: bool) -> &mut Self {
        self.param_scope_disproof = on;
        self
    }

    /// 1 手詰 detection を適用する最大 depth．
    /// 詳細: [`DfPnSolver::param_obvious_final_max_depth`]．
    pub fn set_obvious_final_max_depth(&mut self, d: u32) -> &mut Self {
        self.param_obvious_final_max_depth = d;
        self
    }

    /// find_shortest 中間レンジ lookup を (PN_UNIT, INF) にする．
    /// 詳細: [`DfPnSolver::param_middle_range`]．
    pub fn set_middle_range(&mut self, on: bool) -> &mut Self {
        self.param_middle_range = on;
        self
    }

    /// first-visit 初期 pn の edge_cost 折込を外し，純 df-pn+ にする．
    /// 詳細: [`DfPnSolver::param_decouple_edge_cost`]．
    pub fn set_decouple_edge_cost(&mut self, on: bool) -> &mut Self {
        self.param_decouple_edge_cost = on;
        self
    }

    /// TCA extension formula を `max(thpn, pn+1)` clamp 式にするか．
    pub fn set_tca_clamp(&mut self, on: bool) -> &mut Self {
        self.param_tca_clamp = on;
        self
    }

    /// coherent scale mode を有効化 (seed unit=2 + 完全 comparer)．
    /// 詳細: [`DfPnSolver::param_scale`]．
    pub fn set_scale(&mut self, on: bool) -> &mut Self {
        self.param_scale = on;
        self
    }

    /// 完全 comparer (δ tie-break) を unit-16 でも有効化．
    pub fn set_full_comparer(&mut self, on: bool) -> &mut Self {
        self.param_full_comparer = on;
        self
    }

    /// OR 子展開で合駒 capture cross-deduction を有効化．
    pub fn set_capture_dedup(&mut self, on: bool) -> &mut Self {
        self.param_capture_dedup = on;
        self
    }

    /// 子の seed pn を king-escape 難易度で enrich する (default OFF)．
    pub fn set_rich_seed(&mut self, on: bool) -> &mut Self {
        self.param_rich_seed = on;
        self
    }

    /// AND ノードで一般 合駒 cross-deduction を有効化 (default OFF)．
    pub fn set_cross_dedup(&mut self, on: bool) -> &mut Self {
        self.param_cross_dedup = on;
        self
    }

    /// full_comparer の δ tie-break を AND ノードのみに限定 (default OFF)．
    pub fn set_comparer_and_only(&mut self, on: bool) -> &mut Self {
        self.param_comparer_and_only = on;
        self
    }

    /// AND ノードに PNS の 無駄合い filter を適用 (default OFF)．
    pub fn set_muda_filter(&mut self, on: bool) -> &mut Self {
        self.param_muda_filter = on;
        self
    }

    /// root level IDS を有効化．
    pub fn set_root_ids_enable(&mut self, on: bool) -> &mut Self {
        self.param_root_ids_enable = on;
        self
    }

    /// `is_sum_delta_node` 判定 (OR 香成/不成 max 集約) を有効化．
    /// 詳細: [`DfPnSolver::param_is_sum_delta_node`]．
    pub fn set_is_sum_delta_node(&mut self, on: bool) -> &mut Self {
        self.param_is_sum_delta_node = on;
        self
    }

    /// 極小証明駒 (minimal proof hand) を有効化．
    /// 詳細: [`DfPnSolver::param_minimal_proof_hand`]．
    pub fn set_minimal_proof_hand(&mut self, on: bool) -> &mut Self {
        self.param_minimal_proof_hand = on;
        self
    }

    /// 極大反証駒 (maximal disproof hand) を有効化．
    /// 詳細: [`DfPnSolver::param_minimal_disproof_hand`]．
    pub fn set_minimal_disproof_hand(&mut self, on: bool) -> &mut Self {
        self.param_minimal_disproof_hand = on;
        self
    }

    /// repetition taint + targeted disproof scoping を有効化．
    /// 詳細: [`DfPnSolver::param_repetition`]．
    pub fn set_repetition(&mut self, on: bool) -> &mut Self {
        self.param_repetition = on;
        self
    }

    /// TCA gate モードを設定 (true=is_shallow, false=!is_first_visit)．
    pub fn set_tca_shallow_gate(&mut self, on: bool) -> &mut Self {
        self.param_tca_use_shallow_gate = on;
        self
    }

    /// 診断: deferred penalty + TCA gate 統計を取得．
    /// `(deferred_frames, deferred_penalty_sum, tca_shallow_fire, tca_would_fire_old)`
    pub fn get_phase21_diag(&self) -> (u64, u64, u64, u64) {
        (
            self.diag_deferred_frames,
            self.diag_deferred_penalty_sum,
            self.diag_tca_shallow_fire,
            self.diag_tca_shallow_would_fire,
        )
    }

    /// root child_pn_th の絶対 floor を設定．
    /// 0 で無効．推奨値: 100_000 (= 6250 * PN_UNIT)．
    /// ply=0 の OR child の pn 予算をこの値以上に引き上げ，深い commit を可能にする．
    pub fn set_root_child_pn_floor(&mut self, floor: u32) -> &mut Self {
        self.param_root_child_pn_floor = floor;
        self
    }

    /// 診断: AND scan coverage 統計取得．
    /// `(visit_count, proven_sum, total_sum, zero_proven_count, full_proven_count)`
    /// - 平均 coverage = proven_sum / total_sum
    /// - zero_proven_count: proven=0 で exit した AND scan 回数 (proof 進捗なし)
    /// - full_proven_count: 全 children proven で exit (AND proven)
    pub fn get_and_coverage_stats(&self) -> (u64, u64, u64, u64, u64) {
        (
            self.diag_and_visit_count,
            self.diag_and_proven_sum,
            self.diag_and_total_sum,
            self.diag_and_zero_proven,
            self.diag_and_full_proven,
        )
    }

    /// 診断: OR scan coverage 統計取得．
    /// `(visit_count, proven_count_visits, total_sum)`
    pub fn get_or_coverage_stats(&self) -> (u64, u64, u64) {
        (
            self.diag_or_visit_count,
            self.diag_or_proven_count_visits,
            self.diag_or_total_sum,
        )
    }

    /// 診断: per-position visit count のヒストグラム取得．
    /// 戻り値: `(unique_positions, total_visits, capped, top_10_counts)`
    /// `top_10_counts[i]` = 上位 10 件の最多 visit 数 (降順)．
    pub fn get_pos_visit_stats(&self) -> (usize, u64, bool, Vec<u32>) {
        let unique = self.diag_pos_visits.len();
        let total: u64 = self.diag_pos_visits.values().map(|&v| v as u64).sum();
        let capped = self.diag_pos_visits_capped;
        let mut counts: Vec<u32> = self.diag_pos_visits.values().copied().collect();
        counts.sort_unstable_by(|a, b| b.cmp(a));
        counts.truncate(10);
        (unique, total, capped, counts)
    }

    /// 診断: visit count buckets でヒストグラム．
    /// `bucket[i]` = visit_count が `2^i` 以上 `2^(i+1)` 未満の position 数
    pub fn get_pos_visit_histogram(&self) -> [u64; 24] {
        let mut h = [0u64; 24];
        for &count in self.diag_pos_visits.values() {
            let bucket = if count == 0 {
                0
            } else {
                (32 - count.leading_zeros() - 1).min(23) as usize
            };
            h[bucket] += 1;
        }
        h
    }

    /// OR ノード best_idx 選択の dn tie-break を有効化する．
    /// default false．true で `pn == best_pn` 同点時に `max(dn)` で tie-break．
    /// = defender の抵抗が強い attack を優先 → 探索 commit 強化．
    pub fn set_or_dn_tiebreak(&mut self, on: bool) -> &mut Self {
        self.param_or_dn_tiebreak = on;
        self
    }

    /// AND node exhaustive defender prove を有効化．
    /// default false．true で AND multi-child loop の best_idx 選択を
    /// "全 unproven defender を順次 prove するため round-robin" に変更．
    pub fn set_use_exhaustive_and(&mut self, on: bool) -> &mut Self {
        self.param_use_exhaustive_and = on;
        self
    }

    /// AND proven defender bitmap を有効化．
    /// default false．true で per-AND-position に proven defender index bitmap
    /// を持ち，proven 化済 defender を selection から確実に除外する．
    pub fn set_use_and_proven_bitmap(&mut self, on: bool) -> &mut Self {
        self.param_use_and_proven_bitmap = on;
        self
    }

    pub fn set_use_dag_correction(&mut self, on: bool) {
        self.param_use_dag_correction = on;
    }

    /// per-move attack/defense support 差別化を ON/OFF．
    /// 有効時 `edge_cost_or_with_support` を呼ぶ．
    pub fn set_use_per_move_support(&mut self, on: bool) {
        self.param_use_per_move_support = on;
    }

    /// depth-limited disproof の WorkingTT 格納閾値を明示的に設定する．
    ///
    /// `remaining < threshold` の depth-limited disproof はスキップされる．
    /// `DISPROOF_THRESHOLD_ADAPTIVE` を渡すと adaptive モードに戻る．
    /// 有効値の目安: 0 (無効) / 2〜3 (深い問題向け) / 6 以上 (過剰)．
    ///
    /// **注意**: threshold > 0 は no-mate 証明 (NoCheckmate) の予算要求を
    /// 著しく増加させる場合がある．no-mate が期待される局面では default=0 を
    /// 使うか予算を増やすこと．
    pub fn set_disproof_remaining_threshold(&mut self, threshold: u16) {
        self.param_disproof_remaining_threshold = threshold;
    }

    /// depth-adaptive disproof threshold を opt-in で有効化する．
    ///
    /// solve() 時に `outer_solve_depth` に基づいて閾値を自動決定する:
    /// - depth ≤ 21: 0 (スキップなし)
    /// - depth = 22:   2
    /// - depth ≥ 23 (ply 18 等): 3
    ///
    /// **注意**: adaptive は depth ≥ 23 の深い問題向けに threshold=3 を適用
    /// するため，no-mate 証明が期待される深い局面では退行しうる
    /// (`test_no_checkmate_counter_check` 等．default OFF とした理由)．
    /// 深い詰将棋だけを解く用途で明示的に有効化する．
    pub fn enable_adaptive_disproof_remaining_threshold(&mut self) {
        self.param_disproof_remaining_threshold = DISPROOF_THRESHOLD_ADAPTIVE;
        // solve() 入口で outer_solve_depth を見て TT に反映される．
    }

    /// デフォルトパラメータでソルバーを生成する．
    pub fn default_solver() -> Self {
        Self::new(31, 1_048_576, 32767)
    }

    /// 最短手数探索の有無を設定する．
    ///
    /// `false` にすると最初に見つかった詰み手順をそのまま返す(高速化)．
    /// find_shortest の per-iter refinement cap override (0 = formula)．
    pub fn set_refine_iter_cap(&mut self, cap: u64) -> &mut Self {
        self.param_refine_iter_cap = cap;
        self
    }

    pub fn set_find_shortest(&mut self, v: bool) -> &mut Self {
        self.find_shortest = v;
        self
    }

    /// PV 復元フェーズの1子あたりノード予算を設定する．
    ///
    /// デフォルトは 1024．長手数(17手以上)の詰将棋で
    /// `CheckmateNoPv` が返る場合に増やすと効果的．
    pub fn set_pv_nodes_per_child(&mut self, v: u64) -> &mut Self {
        self.pv_nodes_per_child = v;
        self
    }

    /// refutable check のパラメータを設定する．
    pub fn set_refutable_params(&mut self, depth: u32, call_limit: u32) -> &mut Self {
        self.param_refutable_depth = depth;
        self.param_refutable_call_limit = call_limit;
        self
    }

    /// PNS 初期フェーズ専用の refutable call limit を設定する．
    /// デフォルト 500．MID の param_refutable_call_limit (=10,000) とは独立．
    pub fn set_pns_refutable_call_limit(&mut self, limit: u32) -> &mut Self {
        self.param_pns_refutable_call_limit = limit;
        self
    }

    /// IDS-17 無効化検証用: saved_depth 20-26 での depth=16→17 挿入をスキップするか設定する．
    /// true にすると depth=16 の次は saved_depth へ直接ジャンプする．
    /// デフォルト false = IDS-17 有効．
    pub fn set_no_ids17(&mut self, enable: bool) -> &mut Self {
        self.param_no_ids17 = enable;
        self
    }

    /// **M-1 F1**: refutable check で全 check を必ず評価する．
    /// false 確定 check で早期 return せず，全 AND 子の disproof を store．
    pub fn set_refut_full_eval(&mut self, enable: bool) -> &mut Self {
        self.param_refut_full_eval = enable;
        self
    }

    /// **M-1 F2**: refutable check で fast path 部分 match を活用する．
    /// missing check のみを recursive で評価する．
    pub fn set_refut_partial_recursion(&mut self, enable: bool) -> &mut Self {
        self.param_refut_partial_recursion = enable;
        self
    }

    /// **M-1 F3**: OR レベル refutable 成功局面をキャッシュする．
    /// solve() 内のみで有効．clear() で IDS 透過汚染を防ぐ．
    pub fn set_refut_or_success_cache(&mut self, enable: bool) -> &mut Self {
        self.param_refut_or_success_cache = enable;
        self
    }

    /// **M-1 F4**: refutable fast path lookup を WorkingTT 含めに拡張する．
    /// depth-limited disproof (rem >= floor) も match として count．
    pub fn set_refut_extended_lookup(&mut self, enable: bool) -> &mut Self {
        self.param_refut_extended_lookup = enable;
        self
    }

    /// TT GC 閾値を設定する．
    ///
    /// TT のエントリ数がこの値を超えると GC を実行する．
    /// 0 にすると GC を無効化する．デフォルトは 0(GC 無効)．
    pub fn set_tt_gc_threshold(&mut self, v: usize) -> &mut Self {
        self.tt_gc_threshold = v;
        self
    }

    /// WorkingTT の pn/dn 分布を返す (分析用)．
    ///
    /// ノード制限到達時点のスナップショットがあればそれを返す．
    /// スナップショットがない場合は現在の WorkingTT を収集する．
    /// 返り値: (pn_hist, dn_hist, joint_hist)
    /// - pn_hist: pn 値の log2 ヒストグラム (32 バケット)
    /// - dn_hist: dn 値の log2 ヒストグラム (32 バケット)
    /// - joint_hist: (pn バケット, dn バケット) の 2D ヒストグラム (32×32 = 1024 要素)
    pub fn collect_pn_dn_dist(&self) -> ([u64; 32], [u64; 32], Vec<u64>) {
        if let Some((pn, dn, joint)) = &self.pn_dn_snapshot {
            (*pn, *dn, joint.clone())
        } else {
            // スナップショット未設定時は空の分布を返す．
            ([0u64; 32], [0u64; 32], vec![0u64; 1024])
        }
    }

    /// IDS 各 depth 反復終了時点の WorkingTT pn/dn 分布スナップショット列を返す (分析用)．
    ///
    /// `mid_fallback` 内で各 IDS depth のMID 完了後，TT 遷移前に収集する．
    /// 返り値: `(ids_depth, nodes_searched, elapsed_secs, pn_hist, dn_hist, joint_hist)` のスライス
    pub fn collect_pn_dn_dist_per_depth(
        &self,
    ) -> &[(u32, u64, f64, [u64; 32], [u64; 32], Vec<u64>)] {
        &self.pn_dn_per_depth
    }

    /// TT 診断の監視対象を設定する．
    ///
    /// 指定 ply で指定手(USI 形式)が選択された MID ループ反復ごとに，
    /// TT サイズの変化を stderr に出力する．
    ///
    /// # 引数
    ///
    /// - `ply`: 監視対象の ply(0 で無効化)
    /// - `move_usi`: 監視対象の手(空文字列で ply のみフィルタ)
    /// - `max_iterations`: MID ループの反復回数上限(0 で無制限)
    #[cfg(feature = "tt_diag")]
    pub fn set_tt_diag(&mut self, ply: u32, move_usi: &str, max_iterations: u32) -> &mut Self {
        self.diag_ply = ply;
        self.diag_move_usi = move_usi.to_string();
        self.diag_max_iterations = max_iterations;
        self
    }

    /// タイムアウトしたかどうかを返す．
    #[inline]
    pub(super) fn is_timed_out(&self) -> bool {
        self.timed_out || self.start_time.elapsed() >= self.timeout
    }

    /// 詰将棋を解く(Best-First PNS + MID フォールバック)．
    ///
    /// `board` は攻め方の手番から開始する局面．
    /// 片玉局面(攻め方に玉がない)を想定するが，両玉でも動作する．
    ///
    /// 第 1 段: Best-First PNS で探索木をメモリ上に構築し，
    ///          グローバルに最適なノード選択を行う．
    /// 第 2 段: PNS がアリーナ上限に達した場合，残りの予算で
    ///          IDS-dfpn (MID) にフォールバックする．
    pub fn solve(&mut self, board: &mut Board) -> TsumeResult {
        // production は mid (SearchImpl) を使用する．
        // mid は max_nodes / timeout を honor し TsumeResult::{Checkmate, NoCheckmate, Unknown}
        // を返す．未解決 (budget/timeout) は Unknown (false NoMate を出さない)．
        self.solve_impl(board)
    }

    /// デフォルトの refutable check 呼び出し回数上限．
    const DEFAULT_REFUTABLE_CALL_LIMIT: u32 = 10_000;
    /// デフォルトの refutable check 再帰深さ (0 = 適応的，self.depth に基づく)．
    const DEFAULT_REFUTABLE_DEPTH: u32 = 0;

    /// 王手リストを `out` へ直接追記する zero-copy 経路．
    ///
    /// `generate_check_moves_cached` の ArrayVec<_, 593> 値返し (2.4KB 級
    /// stack copy) を避け，cache hit 時は cache 内 slice から，miss 時は
    /// 生成結果から直接 extend する．生成内容・順序は従来と同一．
    pub(super) fn check_moves_into(&self, board: &mut Board, out: &mut Vec<Move>) {
        let hash = board.hash;
        if let Some(cached) = self.check_cache.get_slice(hash) {
            mate_cand_bump_cache(false, true);
            out.extend_from_slice(cached);
            return;
        }
        mate_cand_bump_cache(false, false);
        let moves = self.generate_check_moves(board);
        self.check_cache.insert(hash, &moves);
        out.extend_from_slice(moves.as_slice());
    }

    /// per-child 1 手詰 lookahead 専用の zero-copy 経路．
    ///
    /// `generate_check_moves_cached` は cache hit でも ArrayVec<_, 593> の値返し
    /// (全コピー × 2) になるため，hit 時は cache 内 slice を直接
    /// `mate_move_in_1ply` へ渡す．生成内容・順序は従来と同一 (semantics 不変)．
    /// 返り値: (1 手詰の手, 王手手段の有無)．
    pub(super) fn mate1ply_with_cached_checks(&self, board: &mut Board) -> (Option<Move>, bool) {
        let hash = board.hash;
        let turn = board.turn;
        if let Some(cached) = self.check_cache.get_slice(hash) {
            // mate_move_in_1ply は board のみ触る (check_cache へ再挿入しない) ため
            // 借用中の slice は有効なまま．
            mate_cand_bump_cache(true, true);
            let mm = board.mate_move_in_1ply(cached, turn);
            if mate_cand_enabled() {
                record_mate_cand(board, cached, mm, turn);
            }
            return (mm, !cached.is_empty());
        }
        mate_cand_bump_cache(true, false);
        let moves = self.generate_check_moves(board);
        self.check_cache.insert(hash, &moves);
        let mm = board.mate_move_in_1ply(moves.as_slice(), turn);
        if mate_cand_enabled() {
            record_mate_cand(board, moves.as_slice(), mm, turn);
        }
        (mm, !moves.is_empty())
    }

    /// look-ahead 専用の 1 手詰判定 (玉から Chebyshev 距離 ≤2 の候補のみ検査)．
    ///
    /// full check list は従来通り生成し check_cache に格納する (expansion が再利用 = cache 内容不変)
    /// が，詰み判定の scan は距離 ≤2 候補に限定する (`mate_move_in_1ply_maxdist(_, _, 2)`)．距離 ≤2
    /// は full scan と同一の Some/None・同一の詰み手を返す (= node 不変)，かつ遠方候補の検証・
    /// do_move fallback を省ける．look-ahead は詰み手 (Option) のみ使う
    /// (`does_have_mate_possibility` が不詰判定を担うため has_checks 不要)．
    pub(super) fn mate1ply_cached_near2(&self, board: &mut Board) -> Option<Move> {
        let hash = board.hash;
        let turn = board.turn;
        if let Some(cached) = self.check_cache.get_slice(hash) {
            return board.mate_move_in_1ply_maxdist(cached, turn, 2);
        }
        let moves = self.generate_check_moves(board);
        self.check_cache.insert(hash, &moves);
        board.mate_move_in_1ply_maxdist(moves.as_slice(), turn, 2)
    }

    /// 敵玉隣接 geometry から候補を構成する look-ahead 1 手詰判定．
    ///
    /// [`DfPnSolver::generate_mate_candidates`] が敵玉隣接 geometry から駒種順で
    /// 1 手詰候補を構成し，検証済 [`Board::mate_move_in_1ply_maxdist`] でスキャンする．
    /// 従来 `mate1ply_cached_near2` との違いは候補列挙のみ (full movegen を回避)．
    /// `check_cache` は更新しない (node が展開される場合は expansion 側が必要時に full 生成する)．
    ///
    /// 逆王手局面 (攻め方自玉が王手 = king-geometry 列挙の前提外) は従来 scan へ
    /// fallback する．`MATE1PLY_VERIFY` 時は健全性 (偽 1 手詰の無さ) と full scan との一致を照合．
    pub(super) fn mate1ply(&self, board: &mut Board) -> Option<Move> {
        let turn = board.turn;
        if board.king_square(turn).is_some() && board.is_in_check(turn) {
            // 逆王手: king-geometry 列挙の前提外 → 従来の near2 scan で判定する．
            return self.mate1ply_cached_near2(board);
        }
        let candidates = self.generate_mate_candidates(board);
        let kh = board.mate_move_in_1ply_maxdist(candidates.as_slice(), turn, u8::MAX);
        if mate1ply_verify() {
            self.verify_mate1ply(board, kh);
        }
        kh
    }

    /// `MATE1PLY_VERIFY` 用: 候補列挙の健全性と従来 full scan との一致を照合し統計へ加算する．
    /// (a) kh が返す手は必ず真の 1 手詰 (do_move 後に詰み) でなければならない (偽 1 手詰 = bug)．
    /// (b) full scan (`mate1ply_with_cached_checks`, 全王手) との Some/None・手の一致を記録する
    ///     (手の相違・kh miss は構成差ゆえ許容; FALSE_MATE=0 のみ必須)．
    fn verify_mate1ply(&self, board: &mut Board, kh: Option<Move>) {
        let turn = board.turn;
        let defender = turn.opponent();
        let full = self.mate1ply_with_cached_checks(board).0;
        let mut false_pos = false;
        if let Some(m) = kh {
            let cap = board.do_move(m);
            let real = board.is_in_check(defender) && !crate::movegen::has_any_legal_move(board);
            board.undo_move(m, cap);
            false_pos = !real;
        }
        record_mate1ply(kh, full, false_pos);
        if false_pos {
            eprintln!(
                "[mate1ply] FALSE MATE kh={:?} sfen={}",
                kh.map(|m| m.to_usi()),
                board.sfen()
            );
        }
        // mate_miss (full=Some, kh=None) を分類 dump する (最初の 40 件)．near2==full ゆえ
        // miss は dist≤2 の near-king 詰み = (a) 列挙ギャップ か (b) 候補列挙が落とす
        // pattern (玉移動開き王手 等) か を判別する手掛かり．
        if kh.is_none() {
            if let Some(fm) = full {
                let n = KH_MISS_DUMP.with(|c| {
                    let v = c.get();
                    c.set(v + 1);
                    v
                });
                if n < 40 {
                    let kingd = board.king_square(defender).map(|k| {
                        let to = fm.to_sq();
                        (to.col() as i32 - k.col() as i32)
                            .abs()
                            .max((to.row() as i32 - k.row() as i32).abs())
                    });
                    let from_is_king =
                        (!fm.is_drop()) && board.king_square(turn) == Some(fm.from_sq());
                    eprintln!(
                        "[mate1ply] MISS full={} drop={} promo={} king_move={} to_king_dist={:?} sfen={}",
                        fm.to_usi(),
                        fm.is_drop(),
                        fm.is_promotion(),
                        from_is_king,
                        kingd,
                        board.sfen(),
                    );
                }
            }
        }
    }
}

/// look-ahead 1 手詰判定で king-geometry 候補列挙を使うか．[`DfPnSolver::mate1ply`] を参照．
pub(super) fn mate1ply_enabled() -> bool {
    // 常時 ON．look-ahead 1 手詰判定を king-geometry 候補列挙にする．
    true
}

/// `MATE1PLY_VERIFY`: 候補列挙の健全性 + full scan との一致を毎 look-ahead 照合する
/// (default OFF; 重い)．
pub(super) fn mate1ply_verify() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("MATE1PLY_VERIFY").is_ok())
}

thread_local! {
    /// [calls, mate_found, diffmove, mate_miss, false_pos]．`MATE1PLY_VERIFY` 時のみ加算．
    static MATE1PLY_STATS: std::cell::Cell<[u64; 5]> = const { std::cell::Cell::new([0; 5]) };
    /// mate_miss dump の出力件数 (最初の数件のみ詳細を出す)．
    static KH_MISS_DUMP: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// kh verify 1 件分を加算する．
fn record_mate1ply(kh: Option<Move>, full: Option<Move>, false_pos: bool) {
    MATE1PLY_STATS.with(|s| {
        let mut v = s.get();
        v[0] += 1;
        if kh.is_some() {
            v[1] += 1;
        }
        match (kh, full) {
            (Some(a), Some(b)) if a != b => v[2] += 1,
            (None, Some(_)) => v[3] += 1,
            _ => {}
        }
        if false_pos {
            v[4] += 1;
        }
        s.set(v);
    });
}

/// kh verify 統計を reset する (solve 開始時)．
pub(super) fn reset_mate1ply_stats() {
    MATE1PLY_STATS.with(|s| s.set([0; 5]));
}

/// kh verify 統計を report する (solve 終了時; gate off なら no-op)．
pub(super) fn report_mate1ply_stats() {
    if !mate1ply_verify() {
        return;
    }
    MATE1PLY_STATS.with(|s| {
        let v = s.get();
        if v[0] == 0 {
            return;
        }
        eprintln!(
            "[mate1ply] calls={} mate_found={} | vs full: diffmove={} mate_miss(full=Some,kh=None)={} | FALSE_MATE(bug)={}",
            v[0], v[1], v[2], v[3], v[4],
        );
    });
}

// ===== MATE1PLY_CAND 診断 =====================================================
// look-ahead の 1 手詰検出について「候補手 (王手) の分布」と「実際に詰みになった手の幾何」を
// 計測する (env `MATE1PLY_CAND` で gate; production path は不変)．目的: king-centric
// constructive mate1ply が **full scan と同じ Some/None を返せる (= node 不変)** か，候補をどこまで
// 絞れるか (= 高速化余地) を経験的に解析する．非詰みは to_sq の玉チェビシェフ距離で分類する
// (非遠方駒の王手は必ず玉隣接，遠方/開き王手のみ遠い)．

#[derive(Default)]
struct MateCandStats {
    calls: u64,
    n_mate: u64,
    sum_checks: u64,
    max_checks: u64,
    sum_checks_king_adj: u64, // to_sq が玉チェビシェフ距離 ≤1 の王手数 (= 非遠方候補)
    sum_checks_slider_far: u64, // 遠方 (距離>1) の飛び駒王手数
    mate_drop: u64,
    mate_move: u64,
    mate_capture: u64,
    mate_slider: u64,
    mate_far: u64, // 詰み手の to_sq が玉から距離>1
    mate_far_slider: u64,
    mate_far_nonslider: u64, // 距離>1 かつ非飛び駒 = 開き王手の疑い (要注目)
    mate_dist: [u64; 9],
    // 差分解析: full scan (ground truth) vs 距離フィルタ subset の verdict 比較 (full が Some の時のみ)．
    // subset ⊆ full なので「subset Some だが full None」は原理上起きない (= 起きたら bug)．
    // MISS = full Some だが subset None (= subset が見落とす遠方詰み)．DIFFMOVE = 両方 Some だが手が
    // 異なる (= どちらも検証済の合法 1 手詰だが選択が違う; proof-hand 変化源)．BUG = subset Some/full None．
    near1_miss: u64, // dist≤1 subset が full の詰みを見落とす
    near1_diffmove: u64,
    near1_bug: u64,
    near2_miss: u64, // dist≤2 subset (= 玉隣接+桂) が見落とす
    near2_diffmove: u64,
    near2_bug: u64,
    // check_cache prepay 値の計測 (constructive で look-ahead generate を除いた際の generate 削減量推定)．
    la_hit: u64,   // look-ahead の cache hit
    la_miss: u64,  // look-ahead の cache miss (= generate 発生)
    cmi_hit: u64,  // expansion (check_moves_into) の cache hit (= 事前生成の再利用)
    cmi_miss: u64, // expansion の cache miss (= generate 発生)
}

/// check_cache hit/miss を計測する (MATE1PLY_CAND 時のみ; la=look-ahead か expansion か)．
pub(super) fn mate_cand_bump_cache(la: bool, hit: bool) {
    if !mate_cand_enabled() {
        return;
    }
    MATE_CAND_STATS.with(|s| {
        let mut s = s.borrow_mut();
        match (la, hit) {
            (true, true) => s.la_hit += 1,
            (true, false) => s.la_miss += 1,
            (false, true) => s.cmi_hit += 1,
            (false, false) => s.cmi_miss += 1,
        }
    });
}

thread_local! {
    static MATE_CAND_STATS: std::cell::RefCell<MateCandStats> =
        std::cell::RefCell::new(MateCandStats::default());
}

fn mate_cand_enabled() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("MATE1PLY_CAND").is_ok())
}

/// MATE1PLY_CAND 統計を reset する (solve 開始時)．
pub(super) fn reset_mate_cand_stats() {
    if mate_cand_enabled() {
        MATE_CAND_STATS.with(|s| *s.borrow_mut() = MateCandStats::default());
    }
}

/// MATE1PLY_CAND 統計を report する (solve 終了時; gate off なら no-op)．
pub(super) fn report_mate_cand_stats() {
    if !mate_cand_enabled() {
        return;
    }
    MATE_CAND_STATS.with(|s| {
        let s = s.borrow();
        if s.calls == 0 {
            return;
        }
        let f = s.calls as f64;
        eprintln!(
            "[mate_cand] calls={} mate={} ({:.2}%) | avg_checks={:.2} max_checks={} | per-call avg: king_adj(≤1)={:.2} slider_far(>1)={:.3}",
            s.calls,
            s.n_mate,
            100.0 * s.n_mate as f64 / f,
            s.sum_checks as f64 / f,
            s.max_checks,
            s.sum_checks_king_adj as f64 / f,
            s.sum_checks_slider_far as f64 / f,
        );
        eprintln!(
            "[mate_cand] 候補総数={} うち玉隣接(≤1)={} ({:.1}%) 遠方飛び駒(>1)={} ({:.2}%)",
            s.sum_checks,
            s.sum_checks_king_adj,
            100.0 * s.sum_checks_king_adj as f64 / s.sum_checks.max(1) as f64,
            s.sum_checks_slider_far,
            100.0 * s.sum_checks_slider_far as f64 / s.sum_checks.max(1) as f64,
        );
        eprintln!(
            "[mate_cand] mate move: drop={} move={} capture={} slider={} | far(dist>1)={} far_slider={} far_nonslider={}",
            s.mate_drop, s.mate_move, s.mate_capture, s.mate_slider,
            s.mate_far, s.mate_far_slider, s.mate_far_nonslider,
        );
        eprintln!("[mate_cand] mate_dist(チェビシェフ 0..8)={:?}", s.mate_dist);
        eprintln!(
            "[mate_cand] DIFF vs full(ground truth): near1(≤1) miss={} diffmove={} bug={} | near2(≤2) miss={} diffmove={} bug={}",
            s.near1_miss, s.near1_diffmove, s.near1_bug,
            s.near2_miss, s.near2_diffmove, s.near2_bug,
        );
        eprintln!(
            "[mate_cand]   → near1 が node を変える look-ahead = {} ({:.3}% of {} mates), near2 = {} ({:.3}%)",
            s.near1_miss + s.near1_diffmove,
            100.0 * (s.near1_miss + s.near1_diffmove) as f64 / s.n_mate.max(1) as f64,
            s.n_mate,
            s.near2_miss + s.near2_diffmove,
            100.0 * (s.near2_miss + s.near2_diffmove) as f64 / s.n_mate.max(1) as f64,
        );
        let gen_now = s.la_miss + s.cmi_miss;
        // constructive (look-ahead generate 除去) 後の generate ≈ expansion の miss のみ．
        // look-ahead が prepay した分 (la_miss で生成 → 後で cmi_hit 再利用) のうち expansion
        // されない分が純減．下限推定: la が prepay した cmi_hit は最大 la_miss．
        eprintln!(
            "[mate_cand] cache: LA(hit={} miss={}) EXP(hit={} miss={}) | generate計={} (LA駆動={} EXP駆動={})",
            s.la_hit, s.la_miss, s.cmi_hit, s.cmi_miss, gen_now, s.la_miss, s.cmi_miss,
        );
    });
}

/// 1 look-ahead の候補 (王手) 分布と詰み手の幾何を統計へ加算する (MATE1PLY_CAND 時のみ呼ぶ)．
/// full scan の結果 `mm` を ground truth とし，距離フィルタ subset の verdict と比較する
/// (board は mate_move_in_1ply 呼び出しで一時的に do_move/undo するが復元される)．
fn record_mate_cand(board: &mut Board, checks: &[Move], mm: Option<Move>, turn: Color) {
    use crate::types::{PieceType, Square};
    let king = match board.king_square(turn.opponent()) {
        Some(k) => k,
        None => return,
    };
    let kc = king.col() as i32;
    let kr = king.row() as i32;
    let cheby = |sq: Square| -> i32 {
        (sq.col() as i32 - kc)
            .abs()
            .max((sq.row() as i32 - kr).abs())
    };
    let checker_pt = |m: Move| -> PieceType {
        if m.is_drop() {
            m.drop_piece_type().unwrap()
        } else {
            let raw = PieceType::from_u8(m.moving_piece_type_raw()).unwrap();
            if m.is_promotion() {
                raw.promoted().unwrap()
            } else {
                raw
            }
        }
    };
    let is_slider = |pt: PieceType| {
        matches!(
            pt,
            PieceType::Lance
                | PieceType::Bishop
                | PieceType::Rook
                | PieceType::Horse
                | PieceType::Dragon
        )
    };
    MATE_CAND_STATS.with(|s| {
        let mut s = s.borrow_mut();
        s.calls += 1;
        s.sum_checks += checks.len() as u64;
        if checks.len() as u64 > s.max_checks {
            s.max_checks = checks.len() as u64;
        }
        for &c in checks {
            let d = cheby(c.to_sq());
            if d <= 1 {
                s.sum_checks_king_adj += 1;
            }
            if d > 1 && is_slider(checker_pt(c)) {
                s.sum_checks_slider_far += 1;
            }
        }
        if let Some(m) = mm {
            s.n_mate += 1;
            let d = cheby(m.to_sq());
            s.mate_dist[d.clamp(0, 8) as usize] += 1;
            if m.is_drop() {
                s.mate_drop += 1;
            } else {
                s.mate_move += 1;
                if m.captured_piece_raw() != 0 {
                    s.mate_capture += 1;
                }
            }
            let sl = is_slider(checker_pt(m));
            if sl {
                s.mate_slider += 1;
            }
            if d > 1 {
                s.mate_far += 1;
                if sl {
                    s.mate_far_slider += 1;
                } else {
                    s.mate_far_nonslider += 1;
                }
            }
        }
    });

    // === 差分解析: full scan (ground truth) vs 距離フィルタ subset ===
    // full が詰みを返した look-ahead でのみ実施 (no-mate は subset も必ず None = 一致, 計測不要)．
    let full_mm = match mm {
        Some(m) => m,
        None => return,
    };
    // 距離フィルタした subset で同じ verified predicate (mate_move_in_1ply) を実行する．
    // subset ⊆ full ゆえ subset が Some なら full も Some (full_mm)．subset Some/full None は
    // 起き得ない (起きたら mate_move_in_1ply の順序依存 bug)．board は復元される．
    let near1: Vec<Move> = checks
        .iter()
        .copied()
        .filter(|c| cheby(c.to_sq()) <= 1)
        .collect();
    let near2: Vec<Move> = checks
        .iter()
        .copied()
        .filter(|c| cheby(c.to_sq()) <= 2)
        .collect();
    let mm1 = board.mate_move_in_1ply(&near1, turn);
    let mm2 = board.mate_move_in_1ply(&near2, turn);
    MATE_CAND_STATS.with(|s| {
        let mut s = s.borrow_mut();
        match mm1 {
            None => s.near1_miss += 1,
            Some(m) if m != full_mm => s.near1_diffmove += 1,
            Some(_) => {}
        }
        match mm2 {
            None => s.near2_miss += 1,
            Some(m) if m != full_mm => s.near2_diffmove += 1,
            Some(_) => {}
        }
    });
    // bug 検出: full が None なのに subset が Some は原理上不可能 (subset⊆full)．
    // full_mm が Some の本経路では起き得ないが，対称性のため near*_bug は別途 (no-mate 経路) で
    // 計測する余地を残す (現状は未使用 = 常に 0)．
}
