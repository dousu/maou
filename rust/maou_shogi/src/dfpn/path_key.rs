//! 探索経路を表す経路ハッシュ (path key)．
//!
//! 探索経路を表す Zobrist ハッシュ．各手の寄与を**深さ (ply) ごとに変える**ことで，手順が前後して
//! 同じ局面に至る経路同士でハッシュが衝突しないようにする．mid の
//! repetition 検出 ([`super::tt`]) のキーに使う．
//!
//! `path_key_after` は XOR 差分なので逆写像も同じ関数 (before/after が同一)．

use std::sync::LazyLock;

use crate::moves::Move;

use super::solver::PATH_CAPACITY;

/// 盤面マス数．
const SQ_NB: usize = 81;
/// 深さ index の上限 (= path stack 容量)．
const MAX_PLY: usize = PATH_CAPACITY;
/// 持ち駒駒種数．
const HAND_KINDS: usize = crate::types::HAND_KINDS;

/// 経路ハッシュの事前計算テーブル (深さ index 付き Zobrist)．
/// 外側次元は `Vec` (ヒープ) で持つ — `MAX_PLY` が大きい (長手数対応) と
/// `[[u64; MAX_PLY]; SQ_NB]` の一括構築が test thread の小スタックを溢れさせるため．
struct PathKeyTables {
    move_to: Vec<[u64; MAX_PLY]>,
    move_from: Vec<[u64; MAX_PLY]>,
    promote: [u64; MAX_PLY],
    dropped_pr: Vec<[u64; MAX_PLY]>,
}

/// splitmix64 PRNG (固定 seed 334334)．
#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

static TABLES: LazyLock<PathKeyTables> = LazyLock::new(|| {
    let mut s: u64 = 334_334;
    let mut t = PathKeyTables {
        move_to: vec![[0; MAX_PLY]; SQ_NB],
        move_from: vec![[0; MAX_PLY]; SQ_NB],
        promote: [0; MAX_PLY],
        dropped_pr: vec![[0; MAX_PLY]; HAND_KINDS],
    };
    for sq in 0..SQ_NB {
        for d in 0..MAX_PLY {
            t.move_from[sq][d] = splitmix64(&mut s);
            t.move_to[sq][d] = splitmix64(&mut s);
        }
    }
    for d in 0..MAX_PLY {
        t.promote[d] = splitmix64(&mut s);
    }
    for pr in 0..HAND_KINDS {
        for d in 0..MAX_PLY {
            t.dropped_pr[pr][d] = splitmix64(&mut s);
        }
    }
    t
});

/// 現在の `path_key` と手 `m` (深さ `ply` で指す) から 1 手後の path key を返す．
///
/// XOR 差分なので
/// `path_key_after(path_key_after(k, m, ply), m, ply) == k` (逆写像も同関数)．
#[inline]
pub(super) fn path_key_after(path_key: u64, m: Move, ply: usize) -> u64 {
    let t = &*TABLES;
    let p = ply.min(MAX_PLY - 1);
    let mut k = path_key ^ t.move_to[m.to_sq().index()][p];
    if m.is_drop() {
        if let Some(pt) = m.drop_piece_type() {
            if let Some(hi) = pt.hand_index() {
                k ^= t.dropped_pr[hi][p];
            }
        }
    } else {
        k ^= t.move_from[m.from_sq().index()][p];
        if m.is_promotion() {
            k ^= t.promote[p];
        }
    }
    k
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::movegen;

    fn first_move(sfen: &str) -> (Board, Move) {
        let mut b = Board::empty();
        b.set_sfen(sfen).unwrap();
        let m = movegen::generate_legal_moves(&mut b)[0];
        (b, m)
    }

    #[test]
    fn xor_roundtrip_is_identity() {
        // before == after なので 2 回適用で元に戻る．
        let (_b, m) = first_move("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1");
        let k0 = 0x1234_5678_9ABC_DEF0u64;
        let k1 = path_key_after(k0, m, 3);
        assert_ne!(k1, k0, "a move must change the path key");
        assert_eq!(
            path_key_after(k1, m, 3),
            k0,
            "XOR diff must be self-inverse"
        );
    }

    #[test]
    fn depth_changes_contribution() {
        // 同じ手でも深さが違えば寄与が違う (順序入替えの衝突回避)．
        let (_b, m) = first_move("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1");
        assert_ne!(path_key_after(0, m, 2), path_key_after(0, m, 5));
    }

    #[test]
    fn deterministic() {
        let (_b, m) = first_move("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1");
        assert_eq!(path_key_after(42, m, 4), path_key_after(42, m, 4));
    }
}
