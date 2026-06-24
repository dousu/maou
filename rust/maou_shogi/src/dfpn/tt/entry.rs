//! 置換表エントリ．**置換表の肝**であり，len 非依存 (1 局面 1 結果) との決定的な差はここにある:
//! - 各局面に **`proven_len` / `disproven_len`** を保持し，探索 `len` で proven/disproven/unknown
//!   を判定する (len-aware)．
//! - **優等/劣等局面** (持ち駒の包含) から pn/dn を合成する (cross-hand)．
//!
//! single-thread 前提のため atomic / lock は不要で，`min_depth` 等は plain field とする
//! (single-thread では値の意味論は同一)．

use crate::dfpn::mate_len::{MateLen, DEPTH_MAX_PLUS1_MATE_LEN, KDEPTH_MAX, MINUS1_MATE_LEN};
use crate::dfpn::search_result::{BitSet64, Depth, Hand, PnDn, SearchAmount, K_INFINITE_PN_DN};
use crate::types::HAND_KINDS;

/// final 化時に amount へ足す優先度ボーナス．
const K_FINAL_AMOUNT_BONUS: SearchAmount = 1000;

/// 無効持ち駒 (空 slot 判定に使う sentinel)．hand は各駒 ≤ 18 なので 0xFF を sentinel に使える．
pub(crate) const NULL_HAND: Hand = [0xFF; HAND_KINDS];

/// `min_depth_rep` の千日手フラグ bit．
/// `min_depth` (∈ [0, KDEPTH_MAX=4000]) は 15bit に収まるため，最上位 bit を流用する
/// (Entry を 64B = 1 cache line に収めるための pack; 探索意味論は不変)．
const REP_BIT: u16 = 0x8000;
/// `min_depth_rep` の下位 15bit (min_depth 本体)．
const MIN_DEPTH_MASK: u16 = 0x7FFF;

/// 持ち駒 `a` が `b` と等しいか優等か (全駒種で `a[i] >= b[i]`)．componentwise 比較．
#[inline]
pub(super) fn hand_is_equal_or_superior(a: Hand, b: Hand) -> bool {
    let mut i = 0;
    while i < HAND_KINDS {
        if a[i] < b[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// 飽和加算 (amount のオーバーフロー回避)．
#[inline]
fn saturated_add(a: SearchAmount, b: SearchAmount) -> SearchAmount {
    a.saturating_add(b)
}

/// 置換表エントリ．
///
/// **64 バイト = 1 cache line** に収めるため `#[repr(C, align(64))]` でフィールドを整列する
/// (cache-line を跨ぐと TT look_up が memory-bound になる)．
/// `proven_len`/`disproven_len` は 16bit (KDEPTH_MAX=4000 < 2^16)，`min_depth` (∈[0,4000]) は
/// 15bit + 千日手フラグ 1bit を `min_depth_rep` に pack する．値域は全て保たれ探索は不変．
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub(super) struct Entry {
    board_key: u64,        // 盤面ハッシュ値
    parent_board_key: u64, // 親局面 (DAG 二重カウント補正の親参照用)
    pn: PnDn,
    dn: PnDn,
    sum_mask: BitSet64,   // δ を和で計上する子の集合
    amount: SearchAmount, // 探索量
    proven_len: u16,      // 詰み手数 +1 (len_plus_1; これ以上の len なら詰み)
    disproven_len: u16,   // 不詰手数 +1 (len_plus_1; これ以下の len なら不詰)
    min_depth_rep: u16,   // bit15=千日手フラグ / bits0..14=格納最浅深さ (TCA old-child 判定)
    hand: Hand,           // 現局面の持ち駒 (NULL_HAND = 空 slot)
    parent_hand: Hand,
}

impl Entry {
    /// 空 slot (`hand == NULL_HAND`)．
    pub(super) fn null() -> Self {
        Self {
            board_key: 0,
            parent_board_key: 0,
            pn: 1,
            dn: 1,
            sum_mask: BitSet64::full(),
            amount: 1,
            proven_len: DEPTH_MAX_PLUS1_MATE_LEN.raw() as u16,
            disproven_len: MINUS1_MATE_LEN.raw() as u16,
            // KDEPTH_MAX(4000) < 0x7FFF．rep bit = 0 (千日手フラグなし)．
            min_depth_rep: KDEPTH_MAX as u16,
            hand: NULL_HAND,
            parent_hand: NULL_HAND,
        }
    }

    /// 新規局面でエントリを初期化する．
    pub(super) fn init(&mut self, board_key: u64, hand: Hand) {
        self.hand = hand;
        self.amount = 1;
        self.board_key = board_key;
        self.proven_len = DEPTH_MAX_PLUS1_MATE_LEN.raw() as u16; // まだ詰みは見つかっていない
        self.disproven_len = MINUS1_MATE_LEN.raw() as u16; // まだ不詰は見つかっていない
        self.pn = 1;
        self.dn = 1;
        // min_depth = KDEPTH_MAX, rep bit = 0 を同時に設定．
        self.min_depth_rep = KDEPTH_MAX as u16;
        self.parent_board_key = 0;
        self.parent_hand = NULL_HAND;
        self.sum_mask = BitSet64::full();
    }

    /// 空 slot か．cluster scan の終端判定に使う．
    #[inline]
    pub(super) fn is_null(&self) -> bool {
        self.hand == NULL_HAND
    }
    /// 盤面ハッシュ一致．
    #[inline]
    pub(super) fn is_for(&self, board_key: u64) -> bool {
        self.board_key == board_key
    }
    /// 盤面 + 持ち駒一致．
    #[inline]
    pub(super) fn is_for_hand(&self, board_key: u64, hand: Hand) -> bool {
        self.board_key == board_key && self.hand == hand
    }
    #[inline]
    pub(super) fn amount(&self) -> SearchAmount {
        self.amount
    }
    #[inline]
    pub(super) fn pn(&self) -> PnDn {
        self.pn
    }
    #[inline]
    pub(super) fn dn(&self) -> PnDn {
        self.dn
    }
    #[inline]
    pub(super) fn get_hand(&self) -> Hand {
        self.hand
    }
    #[inline]
    pub(super) fn sum_mask(&self) -> BitSet64 {
        self.sum_mask
    }
    #[inline]
    pub(super) fn proven_len(&self) -> MateLen {
        MateLen::from_raw_u16(self.proven_len)
    }
    #[inline]
    pub(super) fn disproven_len(&self) -> MateLen {
        MateLen::from_raw_u16(self.disproven_len)
    }
    #[inline]
    pub(super) fn is_possible_repetition(&self) -> bool {
        (self.min_depth_rep & REP_BIT) != 0
    }
    /// `min_depth_rep` から min_depth 本体 (下位 15bit) を i16 で取り出す．
    #[inline]
    fn raw_min_depth(&self) -> i16 {
        (self.min_depth_rep & MIN_DEPTH_MASK) as i16
    }
    /// min_depth 本体を更新する (千日手フラグ bit は保つ)．`d` は [0, 4000]．
    #[inline]
    fn set_raw_min_depth(&mut self, d: i16) {
        self.min_depth_rep = (self.min_depth_rep & REP_BIT) | (d as u16 & MIN_DEPTH_MASK);
    }
    /// 親候補 (DAG 親参照用)．
    #[inline]
    pub(super) fn parent_board_key(&self) -> u64 {
        self.parent_board_key
    }
    #[inline]
    pub(super) fn get_parent_hand(&self) -> Hand {
        self.parent_hand
    }
    /// 千日手フラグを立てる．
    #[inline]
    pub(super) fn set_possible_repetition(&mut self) {
        self.min_depth_rep |= REP_BIT;
    }

    /// 空 slot 化．GC で低 amount entry を間引く際に使う．
    #[inline]
    pub(super) fn set_null(&mut self) {
        self.hand = NULL_HAND;
    }

    /// amount を半減．GC で max_amount が飽和に近いとき全体縮小に使う．
    #[inline]
    pub(super) fn cut_amount(&mut self) {
        self.amount = (self.amount / 2).max(1);
    }

    /// 盤面キー (GC compaction の再配置用)．
    #[inline]
    pub(super) fn board_key(&self) -> u64 {
        self.board_key
    }

    /// 未解決 (unknown) 局面の pn/dn・amount・親情報を更新する．
    pub(super) fn update_unknown(
        &mut self,
        depth: Depth,
        pn: PnDn,
        dn: PnDn,
        amount: SearchAmount,
        sum_mask: BitSet64,
        parent_board_key: u64,
        parent_hand: Hand,
    ) {
        let depth16 = depth as i16;
        if depth16 < self.raw_min_depth() {
            self.set_raw_min_depth(depth16);
        }
        self.pn = pn;
        self.dn = dn;
        self.sum_mask = sum_mask;
        self.amount = self.amount.max(amount);
        self.parent_board_key = parent_board_key;
        self.parent_hand = parent_hand;
    }

    /// proven_len を短い方へ更新する．
    pub(super) fn update_proven(&mut self, len: MateLen, amount: SearchAmount) {
        if len.raw() < self.proven_len as u32 {
            self.proven_len = len.raw() as u16;
        }
        self.amount = self.amount.max(saturated_add(amount, K_FINAL_AMOUNT_BONUS));
    }

    /// disproven_len を長い方へ更新する．
    pub(super) fn update_disproven(&mut self, len: MateLen, amount: SearchAmount) {
        if len.raw() > self.disproven_len as u32 {
            self.disproven_len = len.raw() as u16;
        }
        self.amount = self.amount.max(saturated_add(amount, K_FINAL_AMOUNT_BONUS));
    }

    /// **置換表の肝**．現局面 (hand, depth, len) に対し
    /// エントリ (= board_key 一致前提) から pn/dn を取得・合成する．
    /// 戻り値: pn/dn を更新したか，または現局面に一致するエントリなら `true`．
    pub(super) fn look_up(
        &mut self,
        hand: Hand,
        depth: Depth,
        len: MateLen,
        pn: &mut PnDn,
        dn: &mut PnDn,
        use_old_child: &mut bool,
    ) -> bool {
        let depth16 = depth as i16;
        let entry_hand = self.hand;
        // 1. 一致局面
        if entry_hand == hand {
            return self.look_up_exact(depth16, len, pn, dn, use_old_child);
        }
        // 2. 現局面が劣等局面 (entry が優等)
        if hand_is_equal_or_superior(entry_hand, hand) {
            return self.look_up_inferior(depth16, len, pn, dn, use_old_child);
        }
        // 3. 現局面が優等局面
        if hand_is_equal_or_superior(hand, entry_hand) {
            return self.look_up_superior(depth16, len, pn, dn, use_old_child);
        }
        false
    }

    /// 一致局面 (持ち駒完全一致) の pn/dn 取得．proven/disproven なら確定値，
    /// unknown なら格納 pn/dn を max へ lift する．
    fn look_up_exact(
        &mut self,
        depth16: i16,
        len: MateLen,
        pn: &mut PnDn,
        dn: &mut PnDn,
        use_old_child: &mut bool,
    ) -> bool {
        if len.raw() >= self.proven_len as u32 {
            *pn = 0;
            *dn = K_INFINITE_PN_DN;
        } else if len.raw() <= self.disproven_len as u32 {
            *pn = K_INFINITE_PN_DN;
            *dn = 0;
        } else {
            let min_depth = self.raw_min_depth();
            if depth16 < min_depth {
                self.set_raw_min_depth(depth16); // min_depth を浅い方へ更新
            }
            if *pn < self.pn || *dn < self.dn {
                *pn = (*pn).max(self.pn);
                *dn = (*dn).max(self.dn);
                if min_depth < depth16 {
                    *use_old_child = true;
                }
            }
        }
        true
    }

    /// 現局面が優等 (持ち駒が多い) のときの pn/dn 取得．
    fn look_up_superior(
        &self,
        depth16: i16,
        len: MateLen,
        pn: &mut PnDn,
        dn: &mut PnDn,
        use_old_child: &mut bool,
    ) -> bool {
        if len.raw() >= self.proven_len as u32 {
            // 優等局面は高々 proven_len 手詰み
            *pn = 0;
            *dn = K_INFINITE_PN_DN;
            return true;
        }
        let min_depth = self.raw_min_depth();
        if min_depth <= depth16 && *dn < self.dn {
            *dn = self.dn;
            if min_depth < depth16 {
                *use_old_child = true;
            }
            return true;
        }
        false
    }

    /// 現局面が劣等 (持ち駒が少ない) のときの pn/dn 取得．
    fn look_up_inferior(
        &self,
        depth16: i16,
        len: MateLen,
        pn: &mut PnDn,
        dn: &mut PnDn,
        use_old_child: &mut bool,
    ) -> bool {
        if len.raw() <= self.disproven_len as u32 {
            // 劣等局面は少なくとも disproven_len 手不詰
            *pn = K_INFINITE_PN_DN;
            *dn = 0;
            return true;
        }
        let min_depth = self.raw_min_depth();
        if min_depth <= depth16 && *pn < self.pn {
            *pn = self.pn;
            if min_depth < depth16 {
                *use_old_child = true;
            }
            return true;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::HAND_KINDS;

    fn hand(pawns: u8) -> Hand {
        let mut h = [0u8; HAND_KINDS];
        h[0] = pawns;
        h
    }

    fn fresh(board_key: u64, h: Hand) -> Entry {
        let mut e = Entry::null();
        e.init(board_key, h);
        e
    }

    #[test]
    fn null_slot_detection() {
        let e = Entry::null();
        assert!(e.is_null());
        let e2 = fresh(0x1234, hand(1));
        assert!(!e2.is_null());
        assert!(e2.is_for(0x1234));
        assert!(e2.is_for_hand(0x1234, hand(1)));
        assert!(!e2.is_for_hand(0x1234, hand(2)));
    }

    #[test]
    fn exact_len_aware_proven_disproven() {
        // proven_len=10 にしたエントリ: len>=10 で詰み，len<=disproven で不詰，間は pn/dn．
        let mut e = fresh(0x1, hand(1));
        e.update_proven(MateLen::from_len(10), 5); // proven at len 10
        e.update_disproven(MateLen::from_len(2), 5); // disproven at len 2

        let (mut pn, mut dn, mut old) = (1u64, 1u64, false);
        // len=11 >= proven_len(10) => proven
        assert!(e.look_up(
            hand(1),
            0,
            MateLen::from_len(11),
            &mut pn,
            &mut dn,
            &mut old
        ));
        assert_eq!((pn, dn), (0, K_INFINITE_PN_DN));

        let (mut pn, mut dn) = (1u64, 1u64);
        // len=1 <= disproven_len(2) => disproven
        e.look_up(hand(1), 0, MateLen::from_len(1), &mut pn, &mut dn, &mut old);
        assert_eq!((pn, dn), (K_INFINITE_PN_DN, 0));
    }

    #[test]
    fn exact_unknown_lifts_pn_dn() {
        let mut e = fresh(0x1, hand(1));
        e.update_proven(MateLen::from_len(50), 1);
        e.update_disproven(MateLen::from_len(2), 1);
        e.update_unknown(3, 7, 9, 1, BitSet64::full(), 0, NULL_HAND);
        // len=10 は (disproven=2, proven=50) の間 => pn/dn を max へ lift．
        let (mut pn, mut dn, mut old) = (1u64, 1u64, false);
        // depth(0) < min_depth(3) => min_depth を 0 へ下げる．old_child は min_depth<depth が条件で立たない．
        e.look_up(
            hand(1),
            0,
            MateLen::from_len(10),
            &mut pn,
            &mut dn,
            &mut old,
        );
        assert_eq!((pn, dn), (7, 9));
        assert!(!old);
    }

    #[test]
    fn superior_position_proven_via_fewer_hand_entry() {
        // エントリは hand=1 で len=10 詰み．現局面 hand=3 (優等) は len>=10 で詰み．
        let mut e = fresh(0x1, hand(1));
        e.update_proven(MateLen::from_len(10), 1);
        let (mut pn, mut dn, mut old) = (1u64, 1u64, false);
        let ret = e.look_up(
            hand(3),
            0,
            MateLen::from_len(12),
            &mut pn,
            &mut dn,
            &mut old,
        );
        assert!(ret);
        assert_eq!((pn, dn), (0, K_INFINITE_PN_DN)); // 優等 => 詰み
    }

    #[test]
    fn inferior_position_disproven_via_more_hand_entry() {
        // エントリは hand=3 で len=2 不詰．現局面 hand=1 (劣等) は len<=2 で不詰．
        let mut e = fresh(0x1, hand(3));
        e.update_disproven(MateLen::from_len(2), 1);
        let (mut pn, mut dn, mut old) = (1u64, 1u64, false);
        let ret = e.look_up(hand(1), 0, MateLen::from_len(1), &mut pn, &mut dn, &mut old);
        assert!(ret);
        assert_eq!((pn, dn), (K_INFINITE_PN_DN, 0)); // 劣等 => 不詰
    }

    #[test]
    fn unrelated_hand_no_update() {
        // 銀1 のエントリ vs 歩1 の現局面: 互いに優劣なし => false．
        let mut s = [0u8; HAND_KINDS];
        s[5] = 1; // some non-pawn piece
        let mut e = fresh(0x1, s);
        let (mut pn, mut dn, mut old) = (1u64, 1u64, false);
        let ret = e.look_up(
            hand(1),
            0,
            MateLen::from_len(10),
            &mut pn,
            &mut dn,
            &mut old,
        );
        assert!(!ret);
        assert_eq!((pn, dn), (1, 1)); // unchanged
    }
}
