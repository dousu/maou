//! KH `MateLen` の忠実移植 (`mate_len.hpp`)．
//!
//! mid_v4 (KH verbatim 再現) の基盤型．KH は探索の全ノードに `MateLen len` を thread し，
//! `Emplace(tt, n, len-1, …)` で子へ伝播する．maou の旧 LE path (mid_v3) はこれを一切
//! thread していなかった (構造的非忠実)．本型はそれを KH と byte 単位で一致させるための土台．
//!
//! ## 設計 (KH `detail::MateLenImpl<T>` と同一)
//! - 内部は **詰み手数 + 1** (`len_plus_1`) を保持する．「-1 手」という範囲外初期値を
//!   表現したいため (KH コメント mate_len.hpp:18)．
//! - `Len()` は `max(len_plus_1, 1) - 1` で，-1 手を 0 手へ切り上げる．
//! - 比較は `len_plus_1` の大小で行う (手数の大小と一致)．
//! - 減算は無限 (`kDepthMax` 以上) からは無限のまま (saturating; mate_len.hpp:83-91)．
//!
//! KH 定数: `kDepthMax = 4000` (typedefs.hpp:173)．

/// KH `kDepthMax` (typedefs.hpp:173)．探索の最大深さ = MateLen の最大値．
pub(super) const KDEPTH_MAX: u32 = 4000;

/// KH `MateLen` (= `detail::MateLenImpl<std::uint32_t>`, mate_len.hpp:124)．
///
/// 詰み／不詰手数を表す．内部表現は `len + 1` (範囲外の -1 手を表現可能にするため)．
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct MateLen {
    /// 詰み／不詰手数 + 1 (KH `len_plus_1_`)．
    len_plus_1: u32,
}

impl MateLen {
    /// `len` 手詰み(不詰)で構築する (KH `MateLenImpl(T len)`, mate_len.hpp:40)．
    #[inline]
    pub(super) const fn from_len(len: u32) -> Self {
        Self { len_plus_1: len + 1 }
    }

    /// `len_plus_1` から直接構築する (KH `DirectConstructTag` 経路, mate_len.hpp:109)．
    #[inline]
    const fn from_len_plus_1(len_plus_1: u32) -> Self {
        Self { len_plus_1 }
    }

    /// 詰み手数を返す = `max(len_plus_1, 1) - 1` (KH `Len()`, mate_len.hpp:62)．
    /// `-1` 手 (`len_plus_1 == 0`) は 0 手へ切り上げる．
    #[inline]
    pub(super) const fn len(self) -> u32 {
        let lp1 = if self.len_plus_1 == 0 { 1 } else { self.len_plus_1 };
        lp1 - 1
    }

    /// 内部表現 `len_plus_1` を返す (TT/16bit 変換・比較の生値用)．
    #[inline]
    pub(super) const fn raw(self) -> u32 {
        self.len_plus_1
    }

    /// `self + rhs` 手 (KH `operator+`, mate_len.hpp:75)．
    #[inline]
    pub(super) const fn add(self, rhs: u32) -> Self {
        Self::from_len_plus_1(self.len_plus_1.wrapping_add(rhs))
    }

    /// `self - rhs` 手 (KH `operator-`, mate_len.hpp:83-91)．
    /// **無限 (`kDepthMax` 以上) からは何を引いても無限**のまま (saturating)．
    #[inline]
    pub(super) const fn sub(self, rhs: u32) -> Self {
        let new_len_plus_1 = self.len_plus_1.wrapping_sub(rhs);
        let depth_max_plus_1 = KDEPTH_MAX + 1;
        // KH: 引く前が無限で，引いた後に無限を下回るなら無限へ戻す．
        if self.len_plus_1 >= depth_max_plus_1 && new_len_plus_1 < depth_max_plus_1 {
            return Self::from_len_plus_1(depth_max_plus_1);
        }
        Self::from_len_plus_1(new_len_plus_1)
    }
}

/// 手数の大小比較 = `len_plus_1` の大小 (KH `operator<`, mate_len.hpp:70)．
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
        // KH `operator<<`: len_plus_1>0 なら len_plus_1-1, さもなくば -1 (mate_len.hpp:94)．
        if self.len_plus_1 > 0 {
            write!(f, "{}", self.len_plus_1 - 1)
        } else {
            write!(f, "-1")
        }
    }
}

/// KH `kZeroMateLen` (0 手詰み／0 手不詰, mate_len.hpp:137)．
pub(super) const ZERO_MATE_LEN: MateLen = MateLen::from_len(0);
/// KH `kDepthMaxMateLen` (最大手数 = 無限, mate_len.hpp:141)．None モードの探索開始 len．
pub(super) const DEPTH_MAX_MATE_LEN: MateLen = MateLen::from_len(KDEPTH_MAX);
/// KH `kMinus1MateLen` (= `kZeroMateLen - 1`, 範囲外初期値, mate_len.hpp:155)．
pub(super) const MINUS1_MATE_LEN: MateLen = ZERO_MATE_LEN.sub(1);
/// KH `kDepthMaxPlus1MateLen` (= `kDepthMaxMateLen + 1`, 範囲外初期値, mate_len.hpp:162)．
pub(super) const DEPTH_MAX_PLUS1_MATE_LEN: MateLen = DEPTH_MAX_MATE_LEN.add(1);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matelen_len_roundtrip() {
        assert_eq!(MateLen::from_len(0).len(), 0);
        assert_eq!(MateLen::from_len(33).len(), 33);
        assert_eq!(MateLen::from_len(KDEPTH_MAX).len(), KDEPTH_MAX);
        // -1 手 (kMinus1MateLen) は Len() で 0 へ切り上げ (KH max(len_plus_1,1)-1)．
        assert_eq!(MINUS1_MATE_LEN.len(), 0);
        assert_eq!(MINUS1_MATE_LEN.raw(), 0); // len_plus_1 == 0
    }

    #[test]
    fn matelen_ordering() {
        // kMinus1 < kZero < ... < kDepthMax < kDepthMaxPlus1 (KH 範囲)．
        assert!(MINUS1_MATE_LEN < ZERO_MATE_LEN);
        assert!(ZERO_MATE_LEN < MateLen::from_len(1));
        assert!(MateLen::from_len(33) < DEPTH_MAX_MATE_LEN);
        assert!(DEPTH_MAX_MATE_LEN < DEPTH_MAX_PLUS1_MATE_LEN);
    }

    #[test]
    fn matelen_sub_saturates_at_infinity() {
        // 無限 (kDepthMaxMateLen) から引いても無限のまま (KH mate_len.hpp:87-89)．
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
