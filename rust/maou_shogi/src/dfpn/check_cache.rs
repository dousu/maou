//! 王手生成キャッシュ (E2 最適化)．
//!
//! 局面ハッシュを key とする direct-mapped 王手リストキャッシュ．
//! MID ループで同一局面が繰り返し出現するためヒット率が高い．

use arrayvec::ArrayVec;

use crate::moves::Move;

use super::MAX_MOVES;

/// 王手生成キャッシュのサイズ(2^13 = 8192 エントリ，direct-mapped)．
const CHECK_CACHE_SIZE: usize = 8192;

/// 1エントリあたりのキャッシュ容量(典型的な王手数は 3-15)．
/// この容量を超える王手数の局面は cache 対象外となる．
const CHECK_CACHE_CAPACITY: usize = 32;

/// 王手生成キャッシュの1エントリ．
struct CheckCacheEntry {
    hash: u64,
    moves: ArrayVec<Move, CHECK_CACHE_CAPACITY>,
}

impl Default for CheckCacheEntry {
    fn default() -> Self {
        Self {
            hash: 0,
            moves: ArrayVec::new(),
        }
    }
}

/// 局面ハッシュをキーとする王手リストのキャッシュ．
///
/// `generate_check_moves` の結果を direct-mapped テーブルに保存し，
/// 同一局面への再計算を回避する．MID ループで同一局面が繰り返し
/// 出現するため，キャッシュヒット率が高い．
///
/// 内部可変性(UnsafeCell)を使用して `&self` でアクセス可能にする．
/// これにより `generate_check_moves_cached` を `&self` で呼び出せ，
/// mid() のスタックフレーム最適化を阻害しない．
pub(super) struct CheckCache {
    table: std::cell::UnsafeCell<Vec<CheckCacheEntry>>,
}

impl CheckCache {
    pub(super) fn new() -> Self {
        let mut table = Vec::with_capacity(CHECK_CACHE_SIZE);
        for _ in 0..CHECK_CACHE_SIZE {
            table.push(CheckCacheEntry::default());
        }
        Self {
            table: std::cell::UnsafeCell::new(table),
        }
    }

    /// キャッシュ内の王手リストをコピーせず slice で借用する (zero-copy 経路)．
    ///
    /// 返り値の slice は次の `insert` まで有効．呼び出し側は借用中に本 cache へ
    /// 再挿入しうる処理 (`generate_check_moves_cached` 等) を呼ばないこと．
    #[inline(always)]
    pub(super) fn get_slice(&self, hash: u64) -> Option<&[Move]> {
        let table = unsafe { &*self.table.get() };
        let idx = (hash as usize) & (CHECK_CACHE_SIZE - 1);
        let entry = &table[idx];
        if entry.hash == hash {
            Some(entry.moves.as_slice())
        } else {
            None
        }
    }

    /// 王手リストをキャッシュに格納する．
    #[inline(always)]
    pub(super) fn insert(&self, hash: u64, moves: &ArrayVec<Move, MAX_MOVES>) {
        if moves.len() <= CHECK_CACHE_CAPACITY {
            let table = unsafe { &mut *self.table.get() };
            let idx = (hash as usize) & (CHECK_CACHE_SIZE - 1);
            let entry = &mut table[idx];
            entry.hash = hash;
            entry.moves.clear();
            for &m in moves.iter() {
                entry.moves.push(m);
            }
        }
    }
}
