//! KH `tt::detail::Entry` の忠実移植 (`ttentry.hpp`)．
//!
//! mid_v4 (KH verbatim 再現) の置換表エントリ．**置換表の肝**であり，maou 旧 `v3_tt`
//! (len 非依存・1 局面 1 結果) との決定的な差はここにある:
//! - 各局面に **`proven_len_` / `disproven_len_`** を保持し，探索 `len` で proven/disproven/unknown
//!   を判定する (len-aware; ttentry.hpp:488-511)．
//! - **優等/劣等局面** (持ち駒の包含) から pn/dn を合成する (cross-hand; LookUpSuperior/Inferior)．
//!
//! 並列化は対象外 (KH single-thread; main thread にはノイズなし) のため，KH の
//! `std::atomic<Hand>` / `std::atomic<min_depth>` / `shared_lock` は plain field とする
//! (single-thread では値の意味論は同一)．

use super::mate_len::{MateLen, DEPTH_MAX_PLUS1_MATE_LEN, KDEPTH_MAX, MINUS1_MATE_LEN};
use super::search_result::{BitSet64, Depth, Hand, PnDn, SearchAmount, K_INFINITE_PN_DN};
use crate::types::HAND_KINDS;

/// KH `kFinalAmountBonus` (ttentry.hpp:19)．final 化時に amount へ足す優先度ボーナス．
const K_FINAL_AMOUNT_BONUS: SearchAmount = 1000;

/// 無効持ち駒 (KH `kNullHand` 相当)．空 slot 判定に使う．maou hand は各駒 ≤ 18 なので
/// 0xFF を sentinel に使える．
pub(super) const NULL_HAND: Hand = [0xFF; HAND_KINDS];

/// 千日手の可能性 (KH `RepetitionState`, ttentry.hpp:575)．
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RepetitionState {
    None,
    PossibleRepetition,
}

/// 持ち駒 `a` が `b` と等しいか優等か (全駒種で `a[i] >= b[i]`)．
/// KH `hand_is_equal_or_superior(a, b)` の componentwise 版．
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

/// `SaturatedAdd` (amount のオーバーフロー回避)．
#[inline]
fn saturated_add(a: SearchAmount, b: SearchAmount) -> SearchAmount {
    a.saturating_add(b)
}

/// KH `tt::detail::Entry` (ttentry.hpp:155-)．
#[derive(Clone, Copy)]
pub(super) struct Entry {
    hand: Hand,             // 現局面の持ち駒 (NULL_HAND = 空 slot)
    amount: SearchAmount,   // 探索量
    board_key: u64,         // 盤面ハッシュ値 (KH `Key`)
    proven_len: MateLen,    // 詰み手数 (これ以上の len なら詰み)
    disproven_len: MateLen, // 不詰手数 (これ以下の len なら不詰)
    pn: PnDn,
    dn: PnDn,
    min_depth: i16,        // この局面を格納した最も浅い深さ (TCA old-child 判定)
    parent_board_key: u64, // 親局面 (DAG 二重カウント補正の LookUpParent 用)
    parent_hand: Hand,
    repetition_state: RepetitionState,
    sum_mask: BitSet64, // δ を和で計上する子の集合
}

impl Entry {
    /// 空 slot (KH default 構築; `hand_ == kNullHand`)．
    pub(super) fn null() -> Self {
        Self {
            hand: NULL_HAND,
            amount: 1,
            board_key: 0,
            proven_len: DEPTH_MAX_PLUS1_MATE_LEN,
            disproven_len: MINUS1_MATE_LEN,
            pn: 1,
            dn: 1,
            min_depth: KDEPTH_MAX as i16,
            parent_board_key: 0,
            parent_hand: NULL_HAND,
            repetition_state: RepetitionState::None,
            sum_mask: BitSet64::full(),
        }
    }

    /// KH `Init(board_key, hand)` (ttentry.hpp:205)．新規局面でエントリを初期化する．
    pub(super) fn init(&mut self, board_key: u64, hand: Hand) {
        self.hand = hand;
        self.amount = 1;
        self.board_key = board_key;
        self.proven_len = DEPTH_MAX_PLUS1_MATE_LEN; // まだ詰みは見つかっていない
        self.disproven_len = MINUS1_MATE_LEN; // まだ不詰は見つかっていない
        self.pn = 1;
        self.dn = 1;
        self.min_depth = KDEPTH_MAX as i16;
        self.parent_board_key = 0;
        self.parent_hand = NULL_HAND;
        self.repetition_state = RepetitionState::None;
        self.sum_mask = BitSet64::full();
    }

    /// 空 slot か (KH `IsNull`)．cluster scan の終端判定に使う．
    #[inline]
    pub(super) fn is_null(&self) -> bool {
        self.hand == NULL_HAND
    }
    /// 盤面ハッシュ一致 (KH `IsFor(board_key)`, ttentry.hpp:247)．
    #[inline]
    pub(super) fn is_for(&self, board_key: u64) -> bool {
        self.board_key == board_key
    }
    /// 盤面 + 持ち駒一致 (KH `IsFor(board_key, hand)`, ttentry.hpp:258)．
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
        self.proven_len
    }
    #[inline]
    pub(super) fn disproven_len(&self) -> MateLen {
        self.disproven_len
    }
    #[inline]
    pub(super) fn is_possible_repetition(&self) -> bool {
        self.repetition_state == RepetitionState::PossibleRepetition
    }
    /// 親候補 (DAG LookUpParent 用)．
    #[inline]
    pub(super) fn parent_board_key(&self) -> u64 {
        self.parent_board_key
    }
    #[inline]
    pub(super) fn get_parent_hand(&self) -> Hand {
        self.parent_hand
    }
    /// 格納された最も浅い深さ (テスト/診断用)．
    #[inline]
    pub(super) fn min_depth(&self) -> i16 {
        self.min_depth
    }
    /// 千日手フラグを立てる (KH `SetPossibleRepetition`)．
    #[inline]
    pub(super) fn set_possible_repetition(&mut self) {
        self.repetition_state = RepetitionState::PossibleRepetition;
    }

    /// 空 slot 化 (KH `SetNull`, ttentry.hpp:237)．GC で低 amount entry を間引く際に使う．
    #[inline]
    pub(super) fn set_null(&mut self) {
        self.hand = NULL_HAND;
    }

    /// amount を半減 (KH `CutAmount`, ttentry.hpp:277)．GC で max_amount が飽和に近いとき全体縮小．
    #[inline]
    pub(super) fn cut_amount(&mut self) {
        self.amount = (self.amount / 2).max(1);
    }

    /// 盤面キー (GC compaction の再配置 `PointerOf` 用)．
    #[inline]
    pub(super) fn board_key(&self) -> u64 {
        self.board_key
    }

    /// KH `UpdateUnknown` (ttentry.hpp:306)．
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
        if depth16 < self.min_depth {
            self.min_depth = depth16;
        }
        self.pn = pn;
        self.dn = dn;
        self.sum_mask = sum_mask;
        self.amount = self.amount.max(amount);
        self.parent_board_key = parent_board_key;
        self.parent_hand = parent_hand;
    }

    /// KH `UpdateProven(len, amount)` (ttentry.hpp:330)．proven_len を短い方へ更新．
    pub(super) fn update_proven(&mut self, len: MateLen, amount: SearchAmount) {
        if len < self.proven_len {
            self.proven_len = len;
        }
        self.amount = self.amount.max(saturated_add(amount, K_FINAL_AMOUNT_BONUS));
    }

    /// KH `UpdateDisproven(len, amount)` (ttentry.hpp:343)．disproven_len を長い方へ更新．
    pub(super) fn update_disproven(&mut self, len: MateLen, amount: SearchAmount) {
        if len > self.disproven_len {
            self.disproven_len = len;
        }
        self.amount = self.amount.max(saturated_add(amount, K_FINAL_AMOUNT_BONUS));
    }

    /// **置換表の肝** (KH `LookUp`, ttentry.hpp:378)．現局面 (hand, depth, len) に対し
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

    /// KH `LookUpExact` (ttentry.hpp:488)．`min_depth_out` は const-self を保つため
    /// 呼び出し側で min_depth の更新を反映する (KH は atomic store; ここでは out param)．
    fn look_up_exact(
        &mut self,
        depth16: i16,
        len: MateLen,
        pn: &mut PnDn,
        dn: &mut PnDn,
        use_old_child: &mut bool,
    ) -> bool {
        if len >= self.proven_len {
            *pn = 0;
            *dn = K_INFINITE_PN_DN;
        } else if len <= self.disproven_len {
            *pn = K_INFINITE_PN_DN;
            *dn = 0;
        } else {
            let min_depth = self.min_depth;
            if depth16 < min_depth {
                self.min_depth = depth16; // KH: min_depth_.store(depth16)
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

    /// KH `LookUpSuperior` (ttentry.hpp:522)．現局面が優等 (持ち駒が多い)．
    fn look_up_superior(
        &self,
        depth16: i16,
        len: MateLen,
        pn: &mut PnDn,
        dn: &mut PnDn,
        use_old_child: &mut bool,
    ) -> bool {
        if len >= self.proven_len {
            // 優等局面は高々 proven_len_ 手詰み
            *pn = 0;
            *dn = K_INFINITE_PN_DN;
            return true;
        }
        let min_depth = self.min_depth;
        if min_depth <= depth16 && *dn < self.dn {
            *dn = self.dn;
            if min_depth < depth16 {
                *use_old_child = true;
            }
            return true;
        }
        false
    }

    /// KH `LookUpInferior` (ttentry.hpp:551)．現局面が劣等 (持ち駒が少ない)．
    fn look_up_inferior(
        &self,
        depth16: i16,
        len: MateLen,
        pn: &mut PnDn,
        dn: &mut PnDn,
        use_old_child: &mut bool,
    ) -> bool {
        if len <= self.disproven_len {
            // 劣等局面は少なくとも disproven_len_ 手不詰
            *pn = K_INFINITE_PN_DN;
            *dn = 0;
            return true;
        }
        let min_depth = self.min_depth;
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
        // depth(0) < min_depth(3) => min_depth を 0 へ下げる (KH atomic store)．old_child は min_depth<depth が条件で立たない．
        e.look_up(
            hand(1),
            0,
            MateLen::from_len(10),
            &mut pn,
            &mut dn,
            &mut old,
        );
        assert_eq!((pn, dn), (7, 9));
        assert_eq!(e.min_depth(), 0); // depth16(0) < 旧 min_depth(3) => 0 へ更新
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
