//! 探索木 — 固定容量ノードプールと lock-free 統計．
//!
//! # 統計の意味論
//!
//! ノード統計 (visits / wins) は「**親の手番側から見た**」意味論で保持する:
//! `Q(edge) = wins / visits` がそのまま親手番側の勝率になり，
//! virtual loss は「visits だけ前置インクリメントする」ことで実現できる
//! (wins が付かない in-flight 訪問は Q を押し下げ，他スレッド/同一バッチ内の
//! 探索を別の枝へ分散させる)．
//!
//! # 展開の同期
//!
//! ノードは `UNEXPANDED → EXPANDING → (EXPANDED | TERMINAL_LOSS)` と一方向に
//! 遷移する．`EXPANDING` への CAS に成功したスレッドだけが評価・展開の所有権を
//! 持ち，edges を設定してから `EXPANDED` を Release store する．他スレッドは
//! `EXPANDING` を見たら衝突 (collision) として手を引く．
//!
//! 千日手による終端化のみ `UNEXPANDED → TERMINAL_*` を直接遷移する —
//! 木に合流が無く root への経路はノード毎に一意なため判定は決定的で，
//! 複数スレッドが同時に到達しても同じ状態を store する冪等な操作になる
//! ([`crate::repetition`])．

use std::sync::atomic::{AtomicU32, AtomicU64, AtomicU8, Ordering};
use std::sync::OnceLock;

use maou_shogi::moves::Move;

/// 子ノード未生成を表す番兵値 (pool index として不使用)．
pub const NULL_NODE: u32 = u32::MAX;

/// wins 固定小数点表現のスケール (2^16)．
///
/// 値域 [0,1] の勝率を u64 の `fetch_add` だけで lock-free に加算するための量子化．
/// 精度は 2^-16 ≈ 1.5e-5．visits が 2^48 (1M NPS で約 9 年) に達するまで
/// 総和は u64 に収まる．
const WIN_FP_SCALE: f64 = 65536.0;

/// ノードの展開状態．
pub mod node_state {
    /// 未展開 (まだ評価されていない葉)．
    pub const UNEXPANDED: u8 = 0;
    /// 展開中 (あるスレッドが評価の所有権を取得済み)．
    pub const EXPANDING: u8 = 1;
    /// 展開済 (edges 参照可)．
    pub const EXPANDED: u8 = 2;
    /// 手番側の負け終端 (合法手なし = 詰まされている，または連続王手の
    /// 千日手で手番側が王手をかけ続けた側)．
    pub const TERMINAL_LOSS: u8 = 3;
    /// 引き分け終端 (千日手)．
    pub const TERMINAL_DRAW: u8 = 4;
    /// 手番側の勝ち終端 (相手が連続王手の千日手で負けた)．
    pub const TERMINAL_WIN: u8 = 5;
}

/// ノードの確定状態 (AND-OR 伝播による勝敗/引き分けの確定)．
///
/// [`node_state`] の終端が「葉そのものの確定」を表すのに対し，こちらは
/// 展開済みの内部ノードに AND-OR 伝播で付く ([`crate::search`])．
/// 値はそのノードの手番側視点．詰み探索 (dfpn) 統合時の結果注入口にもなる．
pub mod proven {
    /// 未確定．
    pub const NONE: u8 = 0;
    /// 手番側の負け確定．
    pub const LOSS: u8 = 1;
    /// 引き分け確定 (千日手)．
    pub const DRAW: u8 = 2;
    /// 手番側の勝ち確定．
    pub const WIN: u8 = 3;
}

/// 子ノードへの辺．
pub struct Edge {
    /// この辺に対応する指し手．
    pub mv: Move,
    /// policy 事前確率 (親局面の合法手内で正規化済み)．
    pub prior: f32,
    /// 子ノードの pool index ([`NULL_NODE`] = 未生成)．
    pub child: AtomicU32,
}

impl Edge {
    /// 未生成の子を指す辺を作る．
    pub fn new(mv: Move, prior: f32) -> Edge {
        Edge {
            mv,
            prior,
            child: AtomicU32::new(NULL_NODE),
        }
    }
}

/// 探索木ノード．
///
/// visits は選択時に前置インクリメントされ (virtual loss を兼ねる)，
/// wins は評価完了後のバックプロパゲーションで加算される．
pub struct Node {
    /// 訪問回数 (評価待ち in-flight 分を含む)．
    /// u32 だと 1M NPS × 約 71 分でルートが飽和するため u64 で持つ．
    visits: AtomicU64,
    /// 勝ち数和 (親手番視点，[`WIN_FP_SCALE`] 固定小数点)．
    wins_fp: AtomicU64,
    /// 展開状態 ([`node_state`])．
    state: AtomicU8,
    /// 確定状態 ([`proven`] — AND-OR 伝播で付く，手番側視点)．
    proven: AtomicU8,
    /// 子辺の配列 (EXPANDING の所有スレッドが一度だけ設定する)．
    edges: OnceLock<Box<[Edge]>>,
}

impl Node {
    fn new() -> Node {
        Node {
            visits: AtomicU64::new(0),
            wins_fp: AtomicU64::new(0),
            state: AtomicU8::new(node_state::UNEXPANDED),
            proven: AtomicU8::new(proven::NONE),
            edges: OnceLock::new(),
        }
    }

    /// 訪問回数を返す．
    #[inline]
    pub fn visits(&self) -> u64 {
        self.visits.load(Ordering::Relaxed)
    }

    /// 訪問回数を前置インクリメントする (virtual loss)．
    #[inline]
    pub fn add_visit(&self) {
        self.visits.fetch_add(1, Ordering::Relaxed);
    }

    /// 前置インクリメントを取り消す (衝突時のロールバック)．
    #[inline]
    pub fn revert_visit(&self) {
        self.visits.fetch_sub(1, Ordering::Relaxed);
    }

    /// 勝ち数和に w (親手番視点の勝率 [0,1]) を加算する．
    #[inline]
    pub fn add_win(&self, w: f64) {
        debug_assert!((0.0..=1.0).contains(&w));
        self.wins_fp
            .fetch_add((w * WIN_FP_SCALE) as u64, Ordering::Relaxed);
    }

    /// 勝ち数和 (親手番視点) を返す．
    #[inline]
    pub fn wins(&self) -> f64 {
        self.wins_fp.load(Ordering::Relaxed) as f64 / WIN_FP_SCALE
    }

    /// 展開状態を返す (Acquire — EXPANDED を見たら edges 参照可)．
    #[inline]
    pub fn state(&self) -> u8 {
        self.state.load(Ordering::Acquire)
    }

    /// UNEXPANDED → EXPANDING の CAS を試みる．成功したスレッドが評価の所有権を持つ．
    #[inline]
    pub fn try_begin_expansion(&self) -> bool {
        self.state
            .compare_exchange(
                node_state::UNEXPANDED,
                node_state::EXPANDING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    /// edges を設定して EXPANDED に遷移する (EXPANDING の所有スレッド専用)．
    pub fn finish_expansion(&self, edges: Box<[Edge]>) {
        assert!(
            self.edges.set(edges).is_ok(),
            "edges は EXPANDING の所有スレッドのみが一度だけ設定する"
        );
        self.state.store(node_state::EXPANDED, Ordering::Release);
    }

    /// 終端状態 ([`node_state::TERMINAL_LOSS`] / [`node_state::TERMINAL_DRAW`] /
    /// [`node_state::TERMINAL_WIN`]) としてマークする．
    ///
    /// 合法手なし終端 (詰み) は EXPANDING の所有スレッドが呼ぶ．千日手終端は
    /// UNEXPANDED から直接呼ばれる (経路毎に判定が決定的なため冪等 —
    /// [`crate::repetition`])．
    pub fn mark_terminal(&self, state: u8) {
        debug_assert!(
            matches!(
                state,
                node_state::TERMINAL_LOSS | node_state::TERMINAL_DRAW | node_state::TERMINAL_WIN
            ),
            "終端状態のみ"
        );
        self.state.store(state, Ordering::Release);
    }

    /// 確定値 (このノードの手番側から見た勝率) を返す．
    ///
    /// 終端状態 (葉の確定) と [`proven`] (AND-OR 伝播による内部ノードの確定)
    /// のどちらで確定していても `Some` になる．未確定は `None`．
    #[inline]
    pub fn proven_value(&self) -> Option<f64> {
        match self.state() {
            node_state::TERMINAL_LOSS => return Some(0.0),
            node_state::TERMINAL_DRAW => return Some(0.5),
            node_state::TERMINAL_WIN => return Some(1.0),
            _ => {}
        }
        match self.proven.load(Ordering::Acquire) {
            proven::NONE => None,
            proven::LOSS => Some(0.0),
            proven::DRAW => Some(0.5),
            proven::WIN => Some(1.0),
            other => unreachable!("未知の確定状態: {other}"),
        }
    }

    /// 確定状態を書き込む (NONE からの CAS)．新規確定なら true を返す．
    ///
    /// 木に合流が無く root への経路がノード毎に一意なため，各ノードの確定値は
    /// ゲーム理論的に一意 — 複数スレッドが同時に確定させても同じ値になる
    /// (CAS に負けた場合は既存値との一致を debug_assert で検証する)．
    pub fn try_mark_proven(&self, p: u8) -> bool {
        debug_assert!(
            matches!(p, proven::LOSS | proven::DRAW | proven::WIN),
            "確定状態のみ"
        );
        match self
            .proven
            .compare_exchange(proven::NONE, p, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => true,
            Err(existing) => {
                debug_assert_eq!(existing, p, "確定値はノード毎に一意のはず");
                false
            }
        }
    }

    /// 子辺の配列を返す．state が EXPANDED になってから呼ぶこと．
    #[inline]
    pub fn edges(&self) -> &[Edge] {
        self.edges
            .get()
            .expect("EXPANDED になる前に edges は参照されない")
    }
}

/// 固定容量ノードプール．
///
/// 全ノードを一括で事前確保し，index の単調増加でロックフリーに割り当てる．
/// 容量到達後の [`NodePool::alloc`] は `None` を返す．quiescent 状態で
/// [`NodePool::compact`] を呼ぶと低訪問サブツリーを刈り取って空きを作れる．
pub struct NodePool {
    nodes: Box<[Node]>,
    next: AtomicU32,
}

/// GC ([`NodePool::compact`]) 1 回の結果統計．
#[derive(Clone, Copy, Debug)]
pub struct CompactStats {
    /// 残存ノード数 (compact 直後の使用数)．
    pub kept: u32,
    /// 解放されたノード数．
    pub freed: u32,
    /// 採用された訪問数閾値 (これ未満の visits のノードを解放した)．
    pub visit_threshold: u32,
}

/// [`NodePool::compact`] の閾値探索用ヒストグラムの上限．
/// これ以上の visits は最上位バケツに畳まれる (閾値はこの値で頭打ち)．
const COMPACT_HIST_CAP: u32 = 4096;

impl NodePool {
    /// capacity 個のノードを事前確保する．
    ///
    /// # Panics
    ///
    /// capacity が 0 または [`NULL_NODE`] 以上のとき．
    pub fn new(capacity: u32) -> NodePool {
        assert!(
            capacity > 0 && capacity < NULL_NODE,
            "capacity は 1..NULL_NODE の範囲であること"
        );
        let nodes: Box<[Node]> = (0..capacity).map(|_| Node::new()).collect();
        NodePool {
            nodes,
            next: AtomicU32::new(0),
        }
    }

    /// 新しいノードを割り当て，その index を返す．容量到達時は `None`．
    #[inline]
    pub fn alloc(&self) -> Option<u32> {
        let idx = self.next.fetch_add(1, Ordering::Relaxed);
        if (idx as usize) < self.nodes.len() {
            Some(idx)
        } else {
            None
        }
    }

    /// index からノードを参照する．
    #[inline]
    pub fn get(&self, idx: u32) -> &Node {
        &self.nodes[idx as usize]
    }

    /// 割り当て済みノード数を返す．
    pub fn used(&self) -> u32 {
        self.next
            .load(Ordering::Relaxed)
            .min(self.nodes.len() as u32)
    }

    /// 低訪問サブツリーを刈り取り，生存ノードをプール前方に詰め直す
    /// (stop-the-world mark-compact GC)．
    ///
    /// visits は「子 ≤ 親」の単調性を持つ (訪問は経路単位で加算・巻き戻しされる)
    /// ため，「visits ≥ T」のノード集合は必ずルートから連結する．そこで残存数が
    /// `keep_target` 以下になる最小の閾値 T をヒストグラムで選び，T 未満の
    /// ノードを一括解放する．解放された子への辺は [`NULL_NODE`] に戻り，
    /// 再訪問時に再展開 (再評価) される．CAS 競合でリークしたノードや
    /// ロールバックで visits = 0 に戻ったノードもここで回収される．
    /// index 0 (ルート) は閾値によらず必ず残す．
    ///
    /// `keep_target` を容量より十分小さく取るほど 1 回の解放量が増え，次の枯渇
    /// までの間隔が延びる．計算量は O(使用ノード数 + 生存辺数)，追加メモリは
    /// index 再配置表の 4 バイト × 使用ノード数．
    ///
    /// # 前提
    ///
    /// 全探索スレッドが停止した quiescent 状態でのみ呼べる
    /// (`&mut self` を要求するためコンパイル時に強制される)．
    pub fn compact(&mut self, keep_target: u32) -> CompactStats {
        let used = self.used() as usize;
        let keep_target = keep_target.max(1);

        // 残存数が keep_target 以下になる最小の閾値を visits ヒストグラムから選ぶ．
        // 全ノードの visits が COMPACT_HIST_CAP 以上になる病的なケースでは閾値が
        // 頭打ちし keep_target を超えて残すが，それには 2^12 × 容量回の playout が
        // 必要で実用上到達しない
        let mut hist = vec![0u32; COMPACT_HIST_CAP as usize + 1];
        for node in &self.nodes[..used] {
            hist[node.visits().min(u64::from(COMPACT_HIST_CAP)) as usize] += 1;
        }
        let mut threshold = COMPACT_HIST_CAP;
        let mut kept_count = hist[COMPACT_HIST_CAP as usize];
        for t in (1..COMPACT_HIST_CAP).rev() {
            if kept_count + hist[t as usize] > keep_target {
                break;
            }
            kept_count += hist[t as usize];
            threshold = t;
        }

        // 生存ノードを前方へ詰める (旧 index → 新 index の再配置表を作りながら)．
        // 生存ノードの新 index は旧 index 以下になるため昇順の swap で安全に移動できる
        let mut map = vec![NULL_NODE; used];
        let mut write = 0usize;
        // ループ内で self.nodes を swap で可変借用するため iterator 化できない
        #[allow(clippy::needless_range_loop)]
        for read in 0..used {
            if read == 0 || self.nodes[read].visits() >= u64::from(threshold) {
                debug_assert_ne!(
                    self.nodes[read].state(),
                    node_state::EXPANDING,
                    "quiescent 状態に EXPANDING ノードは存在しない"
                );
                map[read] = write as u32;
                if write != read {
                    self.nodes.swap(write, read);
                }
                write += 1;
            }
        }

        // 生存ノードの辺を新 index に張り替える．刈られた子への辺は NULL_NODE に
        // 戻る (再配置表の初期値)．排他参照下なので Relaxed で十分 (探索再開側との
        // happens-before はスレッド spawn が与える)
        for node in &self.nodes[..write] {
            if let Some(edges) = node.edges.get() {
                for e in edges.iter() {
                    let c = e.child.load(Ordering::Relaxed);
                    if c != NULL_NODE {
                        e.child.store(map[c as usize], Ordering::Relaxed);
                    }
                }
            }
        }

        // 解放スロットを初期状態に戻す (edges の heap 領域もここで解放される)
        for node in &mut self.nodes[write..used] {
            *node = Node::new();
        }
        self.next.store(write as u32, Ordering::Relaxed);

        CompactStats {
            kept: write as u32,
            freed: (used - write) as u32,
            visit_threshold: threshold,
        }
    }

    /// プール容量を返す．
    pub fn capacity(&self) -> u32 {
        self.nodes.len() as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_win_fixed_point_precision() {
        let pool = NodePool::new(1);
        let node = pool.get(0);
        node.add_win(0.5);
        node.add_win(0.5);
        node.add_win(0.25);
        assert!((node.wins() - 1.25).abs() < 1e-3);
    }

    #[test]
    fn test_pool_alloc_exhaustion() {
        let pool = NodePool::new(2);
        assert_eq!(pool.alloc(), Some(0));
        assert_eq!(pool.alloc(), Some(1));
        assert_eq!(pool.alloc(), None);
        assert_eq!(pool.used(), 2);
        assert_eq!(pool.capacity(), 2);
    }

    #[test]
    fn test_expansion_state_transition() {
        let pool = NodePool::new(1);
        let node = pool.get(0);
        assert_eq!(node.state(), node_state::UNEXPANDED);
        assert!(node.try_begin_expansion());
        assert_eq!(node.state(), node_state::EXPANDING);
        // 二重の所有権取得は失敗する
        assert!(!node.try_begin_expansion());
        node.finish_expansion(Box::new([]));
        assert_eq!(node.state(), node_state::EXPANDED);
        assert!(node.edges().is_empty());
    }

    #[test]
    fn test_visit_revert() {
        let pool = NodePool::new(1);
        let node = pool.get(0);
        node.add_visit();
        node.add_visit();
        node.revert_visit();
        assert_eq!(node.visits(), 1);
    }

    #[test]
    fn test_compact_prunes_low_visit_subtrees() {
        use maou_shogi::board::Board;
        use maou_shogi::movegen::generate_legal_moves;

        let mut board = Board::empty();
        board
            .set_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
            .expect("正当な SFEN");
        let mvs = generate_legal_moves(&mut board);

        // root(v10) ── a(v6) ── c(v4)，root ── b(v1)，leak(v0，どこからも未参照)
        let mut pool = NodePool::new(8);
        let root = pool.alloc().expect("空きがある");
        let a = pool.alloc().expect("空きがある");
        let b = pool.alloc().expect("空きがある");
        let c = pool.alloc().expect("空きがある");
        let _leak = pool.alloc().expect("空きがある");

        let e_a = Edge::new(mvs[0], 0.6);
        e_a.child.store(a, Ordering::Relaxed);
        let e_b = Edge::new(mvs[1], 0.4);
        e_b.child.store(b, Ordering::Relaxed);
        pool.get(root).finish_expansion(Box::new([e_a, e_b]));
        let e_c = Edge::new(mvs[0], 1.0);
        e_c.child.store(c, Ordering::Relaxed);
        pool.get(a).finish_expansion(Box::new([e_c]));

        for _ in 0..10 {
            pool.get(root).add_visit();
        }
        for _ in 0..6 {
            pool.get(a).add_visit();
        }
        pool.get(b).add_visit();
        for _ in 0..4 {
            pool.get(c).add_visit();
        }
        pool.get(c).add_win(0.75);

        let st = pool.compact(3);
        // 残存 <= 3 となる最小閾値は 2: {root(10), a(6), c(4)} が残り
        // b(1) と leak(0) が解放される
        assert_eq!(st.visit_threshold, 2);
        assert_eq!(st.kept, 3);
        assert_eq!(st.freed, 2);
        assert_eq!(pool.used(), 3);

        // root は index 0 のまま統計が保存され，刈られた b への辺は未生成に戻る
        let root_node = pool.get(0);
        assert_eq!(root_node.visits(), 10);
        let edges = root_node.edges();
        assert_eq!(edges[0].child.load(Ordering::Relaxed), 1);
        assert_eq!(edges[1].child.load(Ordering::Relaxed), NULL_NODE);

        // a → c の辺は新 index に張り替わり，c の統計も保存される
        let a_node = pool.get(1);
        assert_eq!(a_node.visits(), 6);
        let c_idx = a_node.edges()[0].child.load(Ordering::Relaxed);
        assert_eq!(c_idx, 2);
        let c_node = pool.get(2);
        assert_eq!(c_node.visits(), 4);
        assert!((c_node.wins() - 0.75).abs() < 1e-3);

        // 解放スロットは初期状態で再割り当てできる
        let fresh = pool.alloc().expect("解放によって空きができている");
        assert_eq!(fresh, 3);
        assert_eq!(pool.get(fresh).visits(), 0);
        assert_eq!(pool.get(fresh).state(), node_state::UNEXPANDED);
    }

    #[test]
    fn test_compact_always_keeps_root() {
        let mut pool = NodePool::new(4);
        assert_eq!(pool.alloc(), Some(0));
        // visits 0 のままでも index 0 (ルート) は解放されない
        let st = pool.compact(1);
        assert_eq!(st.kept, 1);
        assert_eq!(st.freed, 0);
        assert_eq!(pool.used(), 1);
        assert_eq!(pool.get(0).state(), node_state::UNEXPANDED);
    }
}
