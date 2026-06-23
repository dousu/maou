//! 探索結果の値型．
//!
//! LocalExpansion / TT / 探索ループがやり取りする中核データ．探索結果を pn/dn,
//! len (MateLen), amount, proof-hand などとともに 1 つの値型に保持する．
//!
//! ## 型
//! - `PnDn = u64`: 証明数・反証数．
//! - `SearchAmount = u32`，`Depth = i32`．
//! - `Hand` は持ち駒表現 (`[u8; HAND_KINDS]`) を再利用する．

use super::mate_len::MateLen;
use crate::types::HAND_KINDS;

/// 証明数・反証数．
pub(super) type PnDn = u64;
/// 探索量 (GC 優先度・tie-break 用)．
pub(super) type SearchAmount = u32;
/// 探索深さ (千日手 rep_start に使用)．
pub(super) type Depth = i32;
/// 攻め方持ち駒 (`[u8; HAND_KINDS]`)．
pub(super) type Hand = [u8; HAND_KINDS];

/// pn/dn の上限値．`u64::MAX / 2 - 1`．和をとっても溢れない余地を残す．
pub(super) const K_INFINITE_PN_DN: PnDn = u64::MAX / 2 - 1;

/// 和が `K_INFINITE_PN_DN` を超えないよう clamp する．
#[inline]
pub(super) fn clamp_pn_dn(v: PnDn) -> PnDn {
    v.min(K_INFINITE_PN_DN)
}

/// 64bit のビット集合 (δ を和で計上する子の集合 = sum_mask)．
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(super) struct BitSet64(pub u64);

impl BitSet64 {
    /// 全 bit を立てる (既定で全子を sum 集計)．
    #[inline]
    pub(super) const fn full() -> Self {
        BitSet64(u64::MAX)
    }
    /// `i` bit が立っているか．
    #[inline]
    pub(super) const fn test(self, i: usize) -> bool {
        (self.0 >> i) & 1 == 1
    }
    /// `i` bit を落とす．
    #[inline]
    pub(super) fn reset(&mut self, i: usize) {
        self.0 &= !(1u64 << i);
    }
}

/// 探索結果値型．
///
/// Unknown と Final を union で共有せず両データを保持し，pn/dn を discriminant とする
/// (`is_final() = pn==0 || dn==0`)．
#[derive(Clone, Copy, Debug)]
pub(super) struct SearchResult {
    pn: PnDn,
    dn: PnDn,
    len: MateLen,
    amount: SearchAmount,
    // --- UnknownData (!IsFinal のとき有効) ---
    sum_mask: BitSet64,
    is_first_visit: bool,
    // --- FinalData (IsFinal のとき有効) ---
    repetition_start: Depth,
    hand: Hand,
}

impl SearchResult {
    /// Unknown (初回)．sum_mask=Full, is_first_visit=true．
    pub(super) fn make_first_visit(pn: PnDn, dn: PnDn, len: MateLen, amount: SearchAmount) -> Self {
        Self::new_unknown_inner(pn, dn, len, amount, BitSet64::full(), true)
    }
    /// Unknown (2 回目以降)．
    pub(super) fn make_unknown(
        pn: PnDn,
        dn: PnDn,
        len: MateLen,
        amount: SearchAmount,
        sum_mask: BitSet64,
    ) -> Self {
        Self::new_unknown_inner(pn, dn, len, amount, sum_mask, false)
    }
    /// Final (詰み proven=true / 不詰 proven=false)．
    pub(super) fn make_final(proven: bool, hand: Hand, len: MateLen, amount: SearchAmount) -> Self {
        let (pn, dn) = if proven {
            (0, K_INFINITE_PN_DN)
        } else {
            (K_INFINITE_PN_DN, 0)
        };
        Self::new_final_inner(
            pn,
            dn,
            len,
            amount,
            super::mate_len::KDEPTH_MAX as Depth,
            hand,
        )
    }
    /// 千日手．pn=INF, dn=0, rep_start を保持．
    pub(super) fn make_repetition(
        hand: Hand,
        len: MateLen,
        amount: SearchAmount,
        rep_start: Depth,
    ) -> Self {
        Self::new_final_inner(K_INFINITE_PN_DN, 0, len, amount, rep_start, hand)
    }

    #[inline]
    fn new_unknown_inner(
        pn: PnDn,
        dn: PnDn,
        len: MateLen,
        amount: SearchAmount,
        sum_mask: BitSet64,
        is_first_visit: bool,
    ) -> Self {
        Self {
            pn,
            dn,
            len,
            amount,
            sum_mask,
            is_first_visit,
            repetition_start: 0,
            hand: [0; HAND_KINDS],
        }
    }
    #[inline]
    fn new_final_inner(
        pn: PnDn,
        dn: PnDn,
        len: MateLen,
        amount: SearchAmount,
        rep_start: Depth,
        hand: Hand,
    ) -> Self {
        Self {
            pn,
            dn,
            len,
            amount,
            sum_mask: BitSet64::full(),
            is_first_visit: false,
            repetition_start: rep_start,
            hand,
        }
    }

    #[inline]
    pub(super) fn pn(self) -> PnDn {
        self.pn
    }
    #[inline]
    pub(super) fn dn(self) -> PnDn {
        self.dn
    }
    /// φ値 (OR=pn, AND=dn)．
    #[inline]
    pub(super) fn phi(self, or_node: bool) -> PnDn {
        if or_node {
            self.pn
        } else {
            self.dn
        }
    }
    /// δ値 (OR=dn, AND=pn)．
    #[inline]
    pub(super) fn delta(self, or_node: bool) -> PnDn {
        if or_node {
            self.dn
        } else {
            self.pn
        }
    }
    /// 結論が出ているか (pn==0 または dn==0)．
    #[inline]
    pub(super) fn is_final(self) -> bool {
        self.pn == 0 || self.dn == 0
    }
    #[inline]
    pub(super) fn len(self) -> MateLen {
        self.len
    }
    #[inline]
    pub(super) fn amount(self) -> SearchAmount {
        self.amount
    }
    /// UnknownData: sum_mask (`!is_final()` のとき有効)．
    #[inline]
    pub(super) fn sum_mask(self) -> BitSet64 {
        self.sum_mask
    }
    /// UnknownData: is_first_visit (`!is_final()` のとき有効)．
    #[inline]
    pub(super) fn is_first_visit(self) -> bool {
        self.is_first_visit
    }
    /// FinalData: hand (`is_final()` のとき有効)．
    #[inline]
    pub(super) fn hand(self) -> Hand {
        self.hand
    }
    /// FinalData: repetition_start (`is_final()` のとき有効)．
    #[inline]
    pub(super) fn repetition_start(self) -> Depth {
        self.repetition_start
    }
    /// 千日手結果か (dn==0 かつ rep_start が有効値)．
    /// rep_start != KDEPTH_MAX を千日手とみなす (make_final は KDEPTH_MAX を入れる)．
    #[inline]
    pub(super) fn is_repetition(self) -> bool {
        self.dn == 0 && self.repetition_start != super::mate_len::KDEPTH_MAX as Depth
    }
}

/// 閾値拡張 (TCA)．
/// **非累積** max(th, val+1)．`is_final()` なら更新しない．
#[inline]
pub(super) fn extend_search_threshold(result: SearchResult, thpn: &mut PnDn, thdn: &mut PnDn) {
    if !result.is_final() {
        if result.pn() < K_INFINITE_PN_DN {
            *thpn = (*thpn).max(result.pn() + 1);
        }
        if result.dn() < K_INFINITE_PN_DN {
            *thdn = (*thdn).max(result.dn() + 1);
        }
    }
}

/// 探索結果比較の (狭義) 半順序．
/// φ → δ → (proven の Len) → (disproven の rep_start) → amount．最終 tie は
/// move eval で破るが，それは LocalExpansion 側で行う．
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Ordering3 {
    Less,
    Greater,
    Equivalent,
}

/// 探索結果の比較関数．
pub(super) fn compare_results(or_node: bool, lhs: SearchResult, rhs: SearchResult) -> Ordering3 {
    // 1. φ値 昇順
    if lhs.phi(or_node) < rhs.phi(or_node) {
        return Ordering3::Less;
    } else if lhs.phi(or_node) > rhs.phi(or_node) {
        return Ordering3::Greater;
    }
    // 2. δ値 昇順
    if lhs.delta(or_node) < rhs.delta(or_node) {
        return Ordering3::Less;
    } else if lhs.delta(or_node) > rhs.delta(or_node) {
        return Ordering3::Greater;
    }
    // 3. proven (pn==0): OR は短い詰み / AND は長い詰みを優先
    if lhs.pn() == 0 {
        if lhs.len() < rhs.len() {
            return if or_node {
                Ordering3::Less
            } else {
                Ordering3::Greater
            };
        } else if lhs.len() > rhs.len() {
            return if or_node {
                Ordering3::Greater
            } else {
                Ordering3::Less
            };
        }
    }
    // 4. disproven (dn==0): rep_start 順 (OR=小さい順, AND=大きい順)
    if lhs.dn() == 0 {
        let l = lhs.repetition_start();
        let r = rhs.repetition_start();
        if l != r {
            // !or_node ^ (l < r) なら Less
            if (!or_node) ^ (l < r) {
                return Ordering3::Less;
            } else {
                return Ordering3::Greater;
            }
        }
    }
    // 5. amount 昇順
    if lhs.amount() < rhs.amount() {
        return Ordering3::Less;
    } else if lhs.amount() > rhs.amount() {
        return Ordering3::Greater;
    }
    Ordering3::Equivalent
}

#[cfg(test)]
mod tests {
    use super::super::mate_len::MateLen;
    use super::*;

    fn h0() -> Hand {
        [0; HAND_KINDS]
    }

    #[test]
    fn final_proven_disproven() {
        let p = SearchResult::make_final(true, h0(), MateLen::from_len(5), 1);
        assert!(p.is_final());
        assert_eq!(p.pn(), 0);
        assert_eq!(p.dn(), K_INFINITE_PN_DN);

        let d = SearchResult::make_final(false, h0(), MateLen::from_len(5), 1);
        assert!(d.is_final());
        assert_eq!(d.dn(), 0);
        assert!(!d.is_repetition()); // rep_start = KDEPTH_MAX => not repetition
    }

    #[test]
    fn repetition_is_distinct_from_disproof() {
        let r = SearchResult::make_repetition(h0(), MateLen::from_len(0), 1, 3);
        assert!(r.is_final());
        assert_eq!(r.dn(), 0);
        assert!(r.is_repetition());
        assert_eq!(r.repetition_start(), 3);
    }

    #[test]
    fn unknown_carries_sum_mask_and_first_visit() {
        let u = SearchResult::make_first_visit(2, 4, MateLen::from_len(10), 1);
        assert!(!u.is_final());
        assert!(u.is_first_visit());
        assert_eq!(u.sum_mask(), BitSet64::full());

        let mut m = BitSet64::full();
        m.reset(3);
        let u2 = SearchResult::make_unknown(2, 4, MateLen::from_len(10), 1, m);
        assert!(!u2.is_first_visit());
        assert!(!u2.sum_mask().test(3));
        assert!(u2.sum_mask().test(2));
    }

    #[test]
    fn extend_threshold_non_cumulative() {
        let u = SearchResult::make_unknown(5, 7, MateLen::from_len(10), 1, BitSet64::full());
        let (mut thpn, mut thdn) = (1u64, 1u64);
        extend_search_threshold(u, &mut thpn, &mut thdn);
        assert_eq!(thpn, 6); // max(1, 5+1)
        assert_eq!(thdn, 8); // max(1, 7+1)
                             // final は更新しない
        let f = SearchResult::make_final(true, h0(), MateLen::from_len(5), 1);
        let (mut a, mut b) = (3u64, 3u64);
        extend_search_threshold(f, &mut a, &mut b);
        assert_eq!((a, b), (3, 3));
    }

    #[test]
    fn comparer_phi_then_delta() {
        let len = MateLen::from_len(10);
        let a = SearchResult::make_unknown(2, 9, len, 1, BitSet64::full()); // OR: phi=2
        let b = SearchResult::make_unknown(3, 1, len, 1, BitSet64::full()); // OR: phi=3
        assert_eq!(compare_results(true, a, b), Ordering3::Less); // a better (smaller pn)
                                                                  // 同 phi なら delta (OR=dn) 昇順
        let c = SearchResult::make_unknown(2, 5, len, 1, BitSet64::full());
        let d = SearchResult::make_unknown(2, 9, len, 1, BitSet64::full());
        assert_eq!(compare_results(true, c, d), Ordering3::Less);
    }

    #[test]
    fn comparer_proven_len_or_vs_and() {
        let short = SearchResult::make_final(true, h0(), MateLen::from_len(3), 1);
        let long = SearchResult::make_final(true, h0(), MateLen::from_len(7), 1);
        // OR: 短い詰みが Less (良い)
        assert_eq!(compare_results(true, short, long), Ordering3::Less);
        // AND: 長い詰みが Less (良い = 抵抗が長い)
        assert_eq!(compare_results(false, short, long), Ordering3::Greater);
    }
}
