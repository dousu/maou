//! Tier 3 (twinkling-hatching-duckling, v0.65.0): 合駒遅延展開．
//!
//! KomoringHeights `delayed_move_list.hpp` (v1.1.0) の Rust 移植．
//!
//! ## 設計
//!
//! 同じ地点への合駒や成/不成ペアなど，すぐに展開する必要のない指し手を
//! 双方向リスト化し，「prev が未解決なら next は展開しない」という
//! 順序制約を強制する．これにより同等の兄弟手を並列に探索することによる
//! pn 過大評価を抑止する．
//!
//! ## 例
//!
//! SQ_5c への合駒であれば，
//! `△5二歩 -> △5二香 -> △5二桂 -> ... -> △5二金`
//! の双方向リストを構成する (1 つの局面に複数のリストが存在し得る)．
//!
//! | i | move | prev | next |
//! |---|------|------|------|
//! | 0 | △5二歩 | null | 1 |
//! | 1 | △5二香 |  0   | 2 |
//! | … |  …   |  …   | … |
//! | 6 | △5二金 |  5   | null |
//!
//! ## 使い方
//!
//! ```ignore
//! let dml = DelayedMoveList::build(&moves, /*or_node=*/false);
//! for (i, m) in moves.iter().enumerate() {
//!     // prev チェイン上に未解決手があればスキップ
//!     if dml.has_unresolved_prev(i, |j| !child_is_final[j]) {
//!         continue;
//!     }
//!     // ...explore moves[i]...
//! }
//! ```

use crate::moves::Move;
use crate::types::PieceType;

/// 1 ノードあたりの最大同マス chain 数 (KH と同じ)．
const MAX_DELAY_HEADS: usize = 10;

/// 指し手の遅延展開を管理する per-node 構造体．
///
/// 配列を pos_idx で添字化．`prev[i]` および `next[i]` はそれぞれ
/// 「直前/直後に展開すべき手の index + 1」を表し，`0` は無し．
/// (`i+1` 形式で記憶することで `Option<usize>` の分岐コストを削減．)
pub(super) struct DelayedMoveList {
    prev: Vec<u32>,
    next: Vec<u32>,
}

impl DelayedMoveList {
    /// 与えられた `moves` から double-linked chain を構築する．
    ///
    /// `or_node = true` なら攻方視点 (OR ノード)，`false` なら受方視点 (AND ノード)．
    /// AND ノードでは全 drop が遅延対象，OR ノードでは drop は対象外
    /// (`IsDelayable` の意味論)．成/不成ペア (同 from, 同 to の駒移動) は
    /// 両ノードで対象とする．
    pub(super) fn build(moves: &[Move], or_node: bool) -> Self {
        Self::build_with_types(moves, or_node, &[], false)
    }

    /// `raw_pts[i]` = `moves[i]` の移動元駒種 raw ID (1=歩..14=龍; 駒打ちは 0)．`us_black` は
    /// 手番側が先手か (enemy_field/香の段判定用)．`raw_pts` が空なら従来の粗い fallback (dummy 歩)．
    /// KH `IsDelayable` の忠実版 (盤上移動は歩/角/飛, 及び香の敵陣 2/8 段昇り のみ遅延対象)．
    pub(super) fn build_with_types(
        moves: &[Move],
        or_node: bool,
        raw_pts: &[u8],
        us_black: bool,
    ) -> Self {
        Self::build_inner(moves, or_node, raw_pts, us_black, &[])
    }

    /// `build_with_types` + KH `IsSame` の中合い対称性 (delayed_move_list.hpp:152-156)．
    /// `interp_chain[i]` = `moves[i]` が「攻方支援なし & 逆王手でない drop」なら true
    /// (呼出側で Board から算出)．この条件を満たす 2 つの drop は **別 to_sq でも同一 chain**
    /// とみなし後回しにする (KH 忠実版; 「逆王手でない中合いはだいたい無意味」)．
    pub(super) fn build_with_types_interp(
        moves: &[Move],
        or_node: bool,
        raw_pts: &[u8],
        us_black: bool,
        interp_chain: &[bool],
    ) -> Self {
        Self::build_inner(moves, or_node, raw_pts, us_black, interp_chain)
    }

    fn build_inner(
        moves: &[Move],
        or_node: bool,
        raw_pts: &[u8],
        us_black: bool,
        interp_chain: &[bool],
    ) -> Self {
        let n = moves.len();
        let mut prev = vec![0u32; n];
        let mut next = vec![0u32; n];

        // head_moves: (move, idx) の最大 MAX_DELAY_HEADS 件．各 chain の末尾を追跡．
        let mut heads: [(Move, u32); MAX_DELAY_HEADS] = [(Move(0), 0); MAX_DELAY_HEADS];
        let mut heads_len = 0usize;

        for (i_raw, &m) in moves.iter().enumerate() {
            let raw_pt = raw_pts.get(i_raw).copied();
            if !is_delayable_typed(m, or_node, raw_pt, us_black) {
                continue;
            }

            // 既存の chain と「同等」な手を探す
            let mut linked = false;
            for j in 0..heads_len {
                let (head_move, head_idx) = heads[j];
                // KH IsSame: 同 to_sq (同マス合駒/成不成ペア) または「支援なし & 非逆王手」の
                // 別マス中合いペア (interp_chain 両 true)．
                let same = is_same(head_move, m)
                    || (interp_chain
                        .get(head_idx as usize)
                        .copied()
                        .unwrap_or(false)
                        && interp_chain.get(i_raw).copied().unwrap_or(false));
                if same {
                    // 既存 chain の末尾に追加: head_idx → i_raw
                    next[head_idx as usize] = i_raw as u32 + 1;
                    prev[i_raw] = head_idx + 1;
                    heads[j] = (m, i_raw as u32);
                    linked = true;
                    break;
                }
            }

            if !linked && heads_len < MAX_DELAY_HEADS {
                heads[heads_len] = (m, i_raw as u32);
                heads_len += 1;
            }
        }

        Self { prev, next }
    }

    /// `i_raw` の直前に展開すべき手の index ．無ければ `None`．
    #[inline(always)]
    pub(super) fn prev(&self, i_raw: usize) -> Option<usize> {
        let p = self.prev[i_raw];
        if p == 0 {
            None
        } else {
            Some((p - 1) as usize)
        }
    }

    /// `i_raw` の直後に展開すべき手の index．無ければ `None`．
    #[inline(always)]
    #[allow(dead_code)] // 将来の Next() walking 用に保持
    pub(super) fn next(&self, i_raw: usize) -> Option<usize> {
        let n = self.next[i_raw];
        if n == 0 {
            None
        } else {
            Some((n - 1) as usize)
        }
    }

    /// `i_raw` の prev chain に未解決の手があるか．
    /// `is_resolved(j)` が `true` なら j は解決済み (final)．
    pub(super) fn has_unresolved_prev<F: Fn(usize) -> bool>(
        &self,
        i_raw: usize,
        is_resolved: F,
    ) -> bool {
        let mut cur = self.prev(i_raw);
        while let Some(j) = cur {
            if !is_resolved(j) {
                return true;
            }
            cur = self.prev(j);
        }
        false
    }
}

/// `move` が遅延展開対象か．
///
/// - `or_node=false` (AND ノード=受方): 全 drop (合駒) が対象
/// - `or_node=true`  (OR ノード=攻方): drop は対象外，成/不成ペアのみ対象
///
/// 成/不成ペアの判定は `is_same` で from/to が同じ非 drop 手を検出することで実現．
/// `is_delayable` 自身は粗いフィルタとしてこのような手の候補を pre-screen する．
/// KH `IsDelayable` 忠実版 (移動元駒種 `raw_pt` を使う)．`raw_pt=None`/0 (駒打ち以外で
/// 不明) のときは従来の粗い fallback (`is_delayable`) に委ねる．
#[inline]
fn is_delayable_typed(m: Move, or_node: bool, raw_pt: Option<u8>, us_black: bool) -> bool {
    if m.is_drop() {
        // 駒打ち (合駒) は AND ノードでのみ遅延対象．
        return !or_node;
    }
    let pt = match raw_pt {
        Some(p) if p != 0 => p,
        // 駒種不明 → 従来の粗い fallback．
        _ => return is_delayable(m, or_node),
    };
    let to = m.to_sq();
    let from = m.from_sq();
    // KH: enemy_field(us).test(from) || enemy_field(us).test(to)．
    let in_enemy = if us_black {
        from.row() <= 2 || to.row() <= 2
    } else {
        from.row() >= 6 || to.row() >= 6
    };
    if !in_enemy {
        return false;
    }
    match pt {
        1 | 5 | 6 => true, // 歩 / 角 / 飛 (不成遅延)
        2 => {
            // 香: 敵陣最奥手前 (先手=2段=row1, 後手=8段=row7) への移動のみ遅延．
            if us_black {
                to.row() == 1
            } else {
                to.row() == 7
            }
        }
        _ => false,
    }
}

#[inline]
fn is_delayable(m: Move, or_node: bool) -> bool {
    if m.is_drop() {
        // AND ノードのみ drop を遅延対象とする．
        !or_node
    } else {
        // 成/不成ペアは move 単独からは判定できないので IsSame に委ねる．
        // この段階では一律 true を返してペア候補に含め，IsSame で
        // 「同 from + 同 to の他の手」が存在しなければ chain に入らない (head のみ)．
        // ただし from/to が enemy field 関連かつ歩/角/飛/香 (KH と同じ駒種制約)
        // の場合のみ true とすることで無関係な手のオーバーヘッドを削減．
        let to = m.to_sq();
        let from = m.from_sq();
        let pt = pt_of_move(m);
        if let Some(pt) = pt {
            let in_enemy = is_in_enemy_field(from, true)
                || is_in_enemy_field(to, true)
                || is_in_enemy_field(from, false)
                || is_in_enemy_field(to, false);
            if !in_enemy {
                return false;
            }
            matches!(
                pt,
                PieceType::Pawn | PieceType::Bishop | PieceType::Rook | PieceType::Lance
            )
        } else {
            false
        }
    }
}

/// `m1` と `m2` が「同等」な手か (どちらを遅延すべきか判定)．
///
/// - 両 drop: 同 to_sq なら同等 (合駒選択の対称性)．
/// - 両非 drop: 同 from + 同 to なら同等 (成/不成ペア)．
/// - drop と非 drop の混合: 同等でない．
///
/// **注意**: KH では更に「support_cnt==0 で no-check のドロップは別 to でも同等」
/// という条件がある (delayed_move_list.hpp:152-156)．これは 攻方支援なし & 王手に
/// ならない中合の対称性検出で，[`DelayedMoveList::build_with_types_interp`] に
/// `interp_chain` flag (呼出側で Board から算出) として実装済 (V4_KHMOVES gate)．
/// 本 `is_same` 自体は Board を持たないため同マス/成不成のみ判定する．
#[inline]
fn is_same(m1: Move, m2: Move) -> bool {
    if m1.is_drop() && m2.is_drop() {
        m1.to_sq() == m2.to_sq()
    } else if !m1.is_drop() && !m2.is_drop() {
        m1.from_sq() == m2.from_sq() && m1.to_sq() == m2.to_sq()
    } else {
        false
    }
}

/// `pt_of_move`: 駒打ちなら drop_piece_type，それ以外は None．
/// 非 drop の駒種は Board がないと判定不能のため，is_delayable の
/// 非 drop 経路では「from/to が enemy field 周辺の歩/角/飛/香」相当の
/// 粗いフィルタを適用するに留める．本格的な駒種判定は Phase 2 で改善．
#[inline]
fn pt_of_move(m: Move) -> Option<PieceType> {
    if m.is_drop() {
        m.drop_piece_type()
    } else {
        // 非 drop は Board なしでは駒種が分からない．
        // is_delayable 側で「常に候補」として扱い，is_same が同 from+同 to で
        // ペアを特定する．粗いがオーバーヘッドは限定的．
        Some(PieceType::Pawn) // dummy: 非 drop は全て promoted-able と仮定
    }
}

#[inline]
fn is_in_enemy_field(sq: crate::types::Square, black_pov: bool) -> bool {
    // 黒視点: row 0-2 (rank 1-3) が敵陣．
    // 白視点: row 6-8 (rank 7-9) が敵陣．
    let row = sq.row();
    if black_pov {
        row <= 2
    } else {
        row >= 6
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moves::Move;
    use crate::types::{PieceType, Square};

    /// 同マスへの 3 個の drop が chain になることを確認．
    #[test]
    fn test_delayed_move_list_drop_chain_at_and_node() {
        let sq = Square::from_raw_u8(40); // 5e 相当
        let m_p = Move::new_drop(sq, PieceType::Pawn);
        let m_l = Move::new_drop(sq, PieceType::Lance);
        let m_n = Move::new_drop(sq, PieceType::Knight);
        let moves = vec![m_p, m_l, m_n];

        let dml = DelayedMoveList::build(&moves, /*or_node=*/ false);
        // index 0 が head (prev=None), 1,2 は prev あり
        assert_eq!(dml.prev(0), None);
        assert_eq!(dml.prev(1), Some(0));
        assert_eq!(dml.prev(2), Some(1));
        // next の chain
        assert_eq!(dml.next(0), Some(1));
        assert_eq!(dml.next(1), Some(2));
        assert_eq!(dml.next(2), None);
    }

    /// 異なるマスへの drop は別 chain になることを確認．
    #[test]
    fn test_delayed_move_list_different_squares() {
        let sq_a = Square::from_raw_u8(40);
        let sq_b = Square::from_raw_u8(41);
        let m_pa = Move::new_drop(sq_a, PieceType::Pawn);
        let m_pb = Move::new_drop(sq_b, PieceType::Pawn);
        let m_la = Move::new_drop(sq_a, PieceType::Lance);
        let moves = vec![m_pa, m_pb, m_la];

        let dml = DelayedMoveList::build(&moves, /*or_node=*/ false);
        // index 0 と 1 はそれぞれ head (異なる sq)
        assert_eq!(dml.prev(0), None);
        assert_eq!(dml.prev(1), None);
        // index 2 (m_la) は index 0 (m_pa) と同じマスなので chain
        assert_eq!(dml.prev(2), Some(0));
    }

    /// OR ノードでは drop が遅延対象外であることを確認．
    #[test]
    fn test_delayed_move_list_or_node_drop_not_delayable() {
        let sq = Square::from_raw_u8(40);
        let m_p = Move::new_drop(sq, PieceType::Pawn);
        let m_l = Move::new_drop(sq, PieceType::Lance);
        let moves = vec![m_p, m_l];

        let dml = DelayedMoveList::build(&moves, /*or_node=*/ true);
        // OR ノードの drop は遅延対象外なので chain なし
        assert_eq!(dml.prev(0), None);
        assert_eq!(dml.prev(1), None);
    }

    /// has_unresolved_prev の動作確認．
    #[test]
    fn test_has_unresolved_prev() {
        let sq = Square::from_raw_u8(40);
        let moves = vec![
            Move::new_drop(sq, PieceType::Pawn),
            Move::new_drop(sq, PieceType::Lance),
            Move::new_drop(sq, PieceType::Knight),
        ];
        let dml = DelayedMoveList::build(&moves, /*or_node=*/ false);

        // index 0: prev なし → 常に false
        assert!(!dml.has_unresolved_prev(0, |_| true));
        assert!(!dml.has_unresolved_prev(0, |_| false));

        // index 1: prev=0．0 が unresolved (false) なら true．
        assert!(dml.has_unresolved_prev(1, |j| j != 0));
        // 0 が resolved (true) なら false．
        assert!(!dml.has_unresolved_prev(1, |_| true));

        // index 2: prev=1 → prev=0．chain 上に 1 つでも unresolved があれば true．
        assert!(dml.has_unresolved_prev(2, |j| j == 1)); // 0 が unresolved
        assert!(dml.has_unresolved_prev(2, |j| j == 0)); // 1 が unresolved
        assert!(!dml.has_unresolved_prev(2, |_| true)); // 全て resolved
    }
}
