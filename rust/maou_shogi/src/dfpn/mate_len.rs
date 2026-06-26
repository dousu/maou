//! 詰み手数 (`MateLen`) を表す型．
//!
//! 探索の全ノードに `MateLen len` を thread し，子ノードへは `len - 1` を伝播する．
//!
//! ## 設計
//! - 内部は **詰み手数 + 1** (`len_plus_1`) を保持する．「-1 手」という範囲外初期値を
//!   表現したいため．
//! - `len()` は `max(len_plus_1, 1) - 1` で，-1 手を 0 手へ切り上げる．
//! - 比較は `len_plus_1` の大小で行う (手数の大小と一致)．
//! - 減算は無限 (`KDEPTH_MAX` 以上) からは無限のまま (saturating)．

/// 探索の最大深さ = MateLen の最大値．
pub(super) const KDEPTH_MAX: u32 = 4000;

/// 詰み／不詰手数を表す．内部表現は `len + 1` (範囲外の -1 手を表現可能にするため)．
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct MateLen {
    /// 詰み／不詰手数 + 1．
    len_plus_1: u32,
}

impl MateLen {
    /// `len` 手詰み(不詰)で構築する．
    #[inline]
    pub(super) const fn from_len(len: u32) -> Self {
        Self {
            len_plus_1: len + 1,
        }
    }

    /// `len_plus_1` から直接構築する．
    #[inline]
    const fn from_len_plus_1(len_plus_1: u32) -> Self {
        Self { len_plus_1 }
    }

    /// 詰み手数を返す = `max(len_plus_1, 1) - 1`．
    /// `-1` 手 (`len_plus_1 == 0`) は 0 手へ切り上げる．
    #[inline]
    pub(super) const fn len(self) -> u32 {
        let lp1 = if self.len_plus_1 == 0 {
            1
        } else {
            self.len_plus_1
        };
        lp1 - 1
    }

    /// 内部表現 `len_plus_1` を返す (TT/16bit 変換・比較の生値用)．
    #[inline]
    pub(super) const fn raw(self) -> u32 {
        self.len_plus_1
    }

    /// 16bit の生値 `len_plus_1` から構築する (TT entry の u16 格納からの復元用)．
    ///
    /// `len_plus_1` は `[0, KDEPTH_MAX+2] = [0, 4002]` に収まり 16bit で格納できる．
    #[inline]
    pub(super) const fn from_raw_u16(len_plus_1: u16) -> Self {
        Self::from_len_plus_1(len_plus_1 as u32)
    }

    /// `self + rhs` 手．
    #[inline]
    pub(super) const fn add(self, rhs: u32) -> Self {
        Self::from_len_plus_1(self.len_plus_1.wrapping_add(rhs))
    }

    /// `self - rhs` 手．
    /// **無限 (`KDEPTH_MAX` 以上) からは何を引いても無限**のまま (high saturate)．
    /// **下限は `len_plus_1 == 0` (= `MINUS1`) で saturate** し，それ以下へ wrap させない．
    /// 旧実装は `wrapping_sub` で 0 を下回ると u32::MAX (≒無限) へ飛び，len 予算を使い切った
    /// 深いノードが「無限予算」扱いになって find_shortest の余詰で偽 proof (len 予算超過の証明) を
    /// 生んでいた．予算切れは `MINUS1` に留め，TT look_up で auto-disprove させる．
    #[inline]
    pub(super) const fn sub(self, rhs: u32) -> Self {
        // 下限 0 で clamp (0 を下回る wrap を防止)．
        let new_len_plus_1 = self.len_plus_1.saturating_sub(rhs);
        let depth_max_plus_1 = KDEPTH_MAX + 1;
        // 引く前が無限で，引いた後に無限を下回るなら無限へ戻す (high saturate)．
        if self.len_plus_1 >= depth_max_plus_1 && new_len_plus_1 < depth_max_plus_1 {
            return Self::from_len_plus_1(depth_max_plus_1);
        }
        Self::from_len_plus_1(new_len_plus_1)
    }
}

/// 手数の大小比較 = `len_plus_1` の大小．
impl PartialOrd for MateLen {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MateLen {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.len_plus_1.cmp(&other.len_plus_1)
    }
}

impl std::fmt::Debug for MateLen {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // len_plus_1>0 なら len_plus_1-1, さもなくば -1 を表示する．
        if self.len_plus_1 > 0 {
            write!(f, "{}", self.len_plus_1 - 1)
        } else {
            write!(f, "-1")
        }
    }
}

/// 0 手詰み／0 手不詰．
pub(super) const ZERO_MATE_LEN: MateLen = MateLen::from_len(0);
/// 最大手数 = 無限．None モードの探索開始 len．
pub(super) const DEPTH_MAX_MATE_LEN: MateLen = MateLen::from_len(KDEPTH_MAX);
/// `ZERO_MATE_LEN - 1` (範囲外初期値)．
pub(super) const MINUS1_MATE_LEN: MateLen = ZERO_MATE_LEN.sub(1);
/// `DEPTH_MAX_MATE_LEN + 1` (範囲外初期値)．
pub(super) const DEPTH_MAX_PLUS1_MATE_LEN: MateLen = DEPTH_MAX_MATE_LEN.add(1);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matelen_len_roundtrip() {
        assert_eq!(MateLen::from_len(0).len(), 0);
        assert_eq!(MateLen::from_len(33).len(), 33);
        assert_eq!(MateLen::from_len(KDEPTH_MAX).len(), KDEPTH_MAX);
        // -1 手 (MINUS1_MATE_LEN) は len() で 0 へ切り上げ (max(len_plus_1,1)-1)．
        assert_eq!(MINUS1_MATE_LEN.len(), 0);
        assert_eq!(MINUS1_MATE_LEN.raw(), 0); // len_plus_1 == 0
    }

    #[test]
    fn matelen_ordering() {
        // MINUS1 < ZERO < ... < DEPTH_MAX < DEPTH_MAX_PLUS1．
        assert!(MINUS1_MATE_LEN < ZERO_MATE_LEN);
        assert!(ZERO_MATE_LEN < MateLen::from_len(1));
        assert!(MateLen::from_len(33) < DEPTH_MAX_MATE_LEN);
        assert!(DEPTH_MAX_MATE_LEN < DEPTH_MAX_PLUS1_MATE_LEN);
    }

    #[test]
    fn matelen_sub_saturates_at_infinity() {
        // 無限 (DEPTH_MAX_MATE_LEN) から引いても無限のまま．
        assert_eq!(DEPTH_MAX_MATE_LEN.sub(1), DEPTH_MAX_MATE_LEN);
        assert_eq!(DEPTH_MAX_MATE_LEN.sub(100), DEPTH_MAX_MATE_LEN);
        // 有限値は普通に減る．
        assert_eq!(MateLen::from_len(33).sub(1).len(), 32);
    }

    #[test]
    fn matelen_add() {
        assert_eq!(MateLen::from_len(10).add(1).len(), 11);
        assert_eq!(ZERO_MATE_LEN.add(1).len(), 1);
    }
}
