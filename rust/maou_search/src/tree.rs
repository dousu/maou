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

use std::sync::atomic::{AtomicU32, AtomicU64, AtomicU8, Ordering};
use std::sync::OnceLock;

use maou_shogi::moves::Move;

/// 子ノード未生成を表す番兵値 (pool index として不使用)．
pub const NULL_NODE: u32 = u32::MAX;

/// wins 固定小数点表現のスケール (2^16)．
///
/// 値域 [0,1] の勝率を u64 の `fetch_add` だけで lock-free に加算するための量子化．
/// 精度は 2^-16 ≈ 1.5e-5．visits が u32 上限 (2^32) に達しても総和は 2^48 で
/// u64 に収まる．
const WIN_FP_SCALE: f64 = 65536.0;

/// ノードの展開状態．
pub mod node_state {
    /// 未展開 (まだ評価されていない葉)．
    pub const UNEXPANDED: u8 = 0;
    /// 展開中 (あるスレッドが評価の所有権を取得済み)．
    pub const EXPANDING: u8 = 1;
    /// 展開済 (edges 参照可)．
    pub const EXPANDED: u8 = 2;
    /// 手番側に合法手がない終端 (詰まされている = 手番側の負け)．
    pub const TERMINAL_LOSS: u8 = 3;
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
    visits: AtomicU32,
    /// 勝ち数和 (親手番視点，[`WIN_FP_SCALE`] 固定小数点)．
    wins_fp: AtomicU64,
    /// 展開状態 ([`node_state`])．
    state: AtomicU8,
    /// 子辺の配列 (EXPANDING の所有スレッドが一度だけ設定する)．
    edges: OnceLock<Box<[Edge]>>,
}

impl Node {
    fn new() -> Node {
        Node {
            visits: AtomicU32::new(0),
            wins_fp: AtomicU64::new(0),
            state: AtomicU8::new(node_state::UNEXPANDED),
            edges: OnceLock::new(),
        }
    }

    /// 訪問回数を返す．
    #[inline]
    pub fn visits(&self) -> u32 {
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

    /// 合法手なし終端 (手番側の負け) としてマークする (EXPANDING の所有スレッド専用)．
    pub fn mark_terminal_loss(&self) {
        self.state
            .store(node_state::TERMINAL_LOSS, Ordering::Release);
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
/// 容量到達後の [`NodePool::alloc`] は `None` を返す (解放・再利用は将来拡張)．
pub struct NodePool {
    nodes: Box<[Node]>,
    next: AtomicU32,
}

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
}
