//! 探索ヒューリスティック: df-pn+ / DFPN-E の初期 pn/dn 推定と move ordering．
//!
//! # DFPN-E (Kishimoto et al., NeurIPS 2019) エッジコスト型ヒューリスティック
//!
//! 標準 df-pn+ はノード(局面)の特徴で初期 pn/dn を設定するが，
//! DFPN-E は**エッジ(手)**の特徴に基づくコストを加算する．
//! 展開済みノードではエッジコストをゼロに戻すため，
//! 実質的には初期 pn への加算として機能する．
//!
//! 詰将棋での手の質:
//! - OR ノードの王手: 成+取 > 取/成 > 近い静か手 > 遠い静か手
//! - AND ノードの応手: 合駒(攻め方有利) < 駒取り < 玉の逃げ

use crate::board::Board;
use crate::moves::Move;
use crate::types::{Color, Square};

use super::PN_UNIT;

/// `to` への利き数に玉の隣接分を補正する．
///
/// `compute_checkers_at` は玉を除外して利き数を数えるため，玉が `to` に
/// 隣接していれば +1 して玉込みの利き数にする (initial pn/dn 推定用)．
#[cfg_attr(feature = "effect_table", allow(dead_code))]
#[inline]
pub(super) fn king_supports(board: &Board, to: Square, color: crate::types::Color) -> u32 {
    match board.king_square(color) {
        Some(k) => {
            let dc = (k.col() as i32 - to.col() as i32).abs();
            let dr = (k.row() as i32 - to.row() as i32).abs();
            u32::from(dc <= 1 && dr <= 1 && (dc != 0 || dr != 0))
        }
        None => 0,
    }
}

/// OR ノード(攻め方の王手)の per-move (pn, dn) tuple を返す．
///
/// 加算は全て PN_UNIT 単位:
/// - 受け駒 ≥ 2: pn += PN_UNIT (後回し)
/// - 攻め支援 + drop_bonus > 受け支援: dn += PN_UNIT (探索優先)
/// - 金/銀取り: dn += PN_UNIT
/// - その他の駒取り: pn += PN_UNIT
/// - 静か手: pn += PN_UNIT
pub(super) fn init_pn_dn_or(
    board: &crate::board::Board,
    m: Move,
    attacker: crate::types::Color,
) -> (u32, u32) {
    let mut pn = PN_UNIT;
    let mut dn = PN_UNIT;

    let to = m.to_sq();
    // attack/defense support = `to` への玉込み利き数．
    // effect テーブル有効時は `effect_count` (玉込みの全利き数) が
    // `compute_checkers_at(玉除外).count() + king_supports(玉補正)` と完全一致するため，
    // per-child の compute_checkers_at × 2 を参照に置換する (探索不変)．
    #[cfg(feature = "effect_table")]
    let (attack_support, defense_support) = (
        u32::from(board.effect_count(attacker, to)),
        u32::from(board.effect_count(attacker.opponent(), to)),
    );
    #[cfg(not(feature = "effect_table"))]
    let (attack_support, defense_support) = {
        let att_bb = board.compute_checkers_at(to, attacker);
        let def_bb = board.compute_checkers_at(to, attacker.opponent());
        // compute_checkers_at は玉を除外するので，玉が `to` に隣接していれば
        // support に +1 して玉込みの利き数にする．
        (
            att_bb.count() + king_supports(board, to, attacker),
            def_bb.count() + king_supports(board, to, attacker.opponent()),
        )
    };
    let drop_bonus: u32 = if m.is_drop() { 1 } else { 0 };

    if defense_support >= 2 {
        pn += PN_UNIT;
    }

    if attack_support + drop_bonus > defense_support {
        dn += PN_UNIT;
    } else {
        let captured = m.captured_piece_raw();
        if captured > 0 {
            let cap_pt = crate::types::PieceType::from_u8(captured);
            if matches!(
                cap_pt,
                Some(crate::types::PieceType::Gold) | Some(crate::types::PieceType::Silver)
            ) {
                dn += PN_UNIT;
            } else {
                pn += PN_UNIT;
            }
        } else {
            pn += PN_UNIT;
        }
    }

    (pn, dn)
}

/// AND ノード(防御側の応手)の per-move (pn, dn) tuple を返す (U = PN_UNIT)．
///
/// - 駒取り応手: (2U, U)
/// - 玉移動: (U, U)
/// - 攻め支援 < 受け支援 + drop_bonus (good escape): (2U, U)
/// - その他 (bad escape): (U, 2U)
pub(super) fn init_pn_dn_and(
    board: &crate::board::Board,
    m: Move,
    attacker: crate::types::Color,
) -> (u32, u32) {
    let defender = attacker.opponent();

    if m.captured_piece_raw() > 0 {
        return (2 * PN_UNIT, PN_UNIT);
    }

    let king_sq = board.king_square(defender);
    if !m.is_drop() {
        if let Some(ksq) = king_sq {
            if m.from_sq() == ksq {
                return (PN_UNIT, PN_UNIT);
            }
        }
    }

    let to = m.to_sq();
    // attack/defense support = `to` への玉込み利き数 (effect 有効時は参照に置換; 探索不変)．
    #[cfg(feature = "effect_table")]
    let (attack_support, defense_support) = (
        u32::from(board.effect_count(attacker, to)),
        u32::from(board.effect_count(defender, to)),
    );
    #[cfg(not(feature = "effect_table"))]
    let (attack_support, defense_support) = {
        let att_bb = board.compute_checkers_at(to, attacker);
        let def_bb = board.compute_checkers_at(to, defender);
        // compute_checkers_at は玉を除外するので，玉が `to` 隣接なら support に +1 (玉込みの利き数)．
        (
            att_bb.count() + king_supports(board, to, attacker),
            def_bb.count() + king_supports(board, to, defender),
        )
    };
    let drop_bonus: u32 = if m.is_drop() { 1 } else { 0 };

    if attack_support < defense_support + drop_bonus {
        // good escape
        (2 * PN_UNIT, PN_UNIT)
    } else {
        // bad escape
        (PN_UNIT, 2 * PN_UNIT)
    }
}

/// move ordering key を返す (値が小さいほど「良い手」)．
///
/// 基準:
/// - 成れるのに成らない歩/角/飛: +1000
/// - 移動後の駒価値が高いほど優先 (−pt_value)
/// - 玉に近いほど優先 (+10 × distance)
pub(super) fn move_brief_eval(m: Move, king_sq: Square, board: &Board) -> i32 {
    let to = m.to_sq();
    let mut value: i32 = 0;

    // 移動後の駒種 raw ID (1=Pawn .. 14=Dragon)
    let raw_pt: u8 = if m.is_drop() {
        m.drop_piece_type().map(|pt| pt as u8).unwrap_or(0)
    } else {
        // board.piece_at returns raw piece byte; strip color (& 0x0F)
        board.piece_at(m.from_sq()) & 0x0F
    };

    // 成れるのに成らない歩/角/飛: +1000 penalty
    if !m.is_drop() && !m.is_promotion() && matches!(raw_pt, 1 | 5 | 6) {
        let from = m.from_sq();
        let us = board.turn;
        let in_enemy = |sq: Square| -> bool {
            let r = sq.row();
            match us {
                Color::Black => r <= 2,
                Color::White => r >= 6,
            }
        };
        if in_enemy(from) || in_enemy(to) {
            value += 1000;
        }
    }

    let after_raw = if m.is_promotion() { raw_pt + 8 } else { raw_pt };
    let pt_value: i32 = match after_raw {
        1 => 10,      // Pawn
        2 => 20,      // Lance
        3 => 20,      // Knight
        4 => 30,      // Silver
        5 => 50,      // Bishop
        6 => 50,      // Rook
        7 => 50,      // Gold
        8 => 80,      // King (AND 玉捕獲手の tie-break に必要)
        9..=12 => 50, // ProPawn..ProSilver
        13 => 80,     // Horse
        14 => 80,     // Dragon
        _ => 0,
    };
    value -= pt_value;

    let dc = (to.col() as i32 - king_sq.col() as i32).abs();
    let dr = (to.row() as i32 - king_sq.row() as i32).abs();
    value += 10 * dc.max(dr);

    value
}

// === Move ordering keys (探索の手生成順を再現するソートキー) ===

/// 王手生成の生成順を再現する sort key．square index は (file-1)*9+(rank-1) なので raw square で
/// 昇順比較できる．順序: ① 盤上移動 (from-square 昇順 → to-square 昇順 → 不成→成)，② 駒打ち
/// (PAWN,LANCE,KNIGHT,SILVER,GOLD,BISHOP,ROOK 順 → to-square 昇順)．
/// 注: 開き王手/直接王手の細分順は未反映 (直接王手のみの局面では from-square 順で一致)．
fn drop_piece_order(pt: crate::types::PieceType) -> u64 {
    use crate::types::PieceType;
    // 駒打ちの drops[] 順: PAWN 先頭 + KNIGHT,LANCE,SILVER,GOLD,BISHOP,ROOK (桂が香より先)．
    match pt {
        PieceType::Pawn => 0,
        PieceType::Knight => 1,
        PieceType::Lance => 2,
        PieceType::Silver => 3,
        PieceType::Gold => 4,
        PieceType::Bishop => 5,
        PieceType::Rook => 6,
        _ => 7,
    }
}

pub(super) fn check_order_key(
    m: crate::moves::Move,
    discoverers: crate::bitboard::Bitboard,
    def_king: Option<crate::types::Square>,
) -> u64 {
    use crate::types::PieceType;
    if let Some(pt) = m.drop_piece_type() {
        // group 2: 駒打ち王手．駒打ち王手の生成順
        // (PAWN,LANCE,KNIGHT,SILVER,GOLD,BISHOP,ROOK) → 各 check_square 昇順．
        // ⚠ evasion drops[] (KNIGHT,LANCE,..) とは Knight/Lance が逆順なので別 mapping．
        let pmo = match pt {
            PieceType::Pawn => 0u64,
            PieceType::Lance => 1,
            PieceType::Knight => 2,
            PieceType::Silver => 3,
            PieceType::Gold => 4,
            PieceType::Bishop => 5,
            PieceType::Rook => 6,
            _ => 7,
        };
        (2u64 << 42) | (pmo << 7) | (m.to_sq().raw_u8() as u64)
    } else {
        // 盤上移動王手．王手生成順:
        //   ① 開き王手候補 from (= discoverers = blockers_for_king(them) & pieces(us)) を square 昇順で，
        //      各 from は pin-line 外への移動 (純開き王手) → pin-line 上への移動 (直接王手) の順に生成．
        //   ② 非候補 from (直接王手のみ) を square 昇順で生成．
        let from = m.from_sq();
        let to = m.to_sq();
        let pf = if m.is_promotion() { 0u64 } else { 1u64 };
        let disc = discoverers.contains(from);
        let group = if disc { 0u64 } else { 1u64 };
        // 開き王手候補 from 内: pin-line 外 (sub 0) → pin-line 上 (sub 1)．
        let sub = if disc {
            match def_king {
                Some(k) if crate::attack::line_through(k, from).contains(to) => 1u64,
                _ => 0u64,
            }
        } else {
            0u64
        };
        (group << 42)
            | ((from.raw_u8() as u64) << 22)
            | (sub << 21)
            | (pf << 20)
            | ((to.raw_u8() as u64) << 11)
    }
}

/// 盤上移動の駒種を駒種別生成順へ写像する (Evasion 非玉手の group 番号)．
/// PAWN→0, LANCE→1, KNIGHT→2, SILVER→3, BISHOP→4, ROOK→5, GOLD/成駒(金移動)→6, HORSE→7, DRAGON→8．
fn piece_gen_order(raw_pt: u8) -> u64 {
    match raw_pt {
        1 => 0,      // Pawn
        2 => 1,      // Lance
        3 => 2,      // Knight
        4 => 3,      // Silver
        5 => 4,      // Bishop
        6 => 5,      // Rook
        7 => 6,      // Gold
        9..=12 => 6, // と金/成香/成桂/成銀 (金の動き)
        13 => 7,     // Horse
        14 => 8,     // Dragon
        _ => 9,
    }
}

/// 王手回避手の生成順を再現する key．
/// ① 玉移動 (to-square 昇順) → ② 非玉の盤上移動 (駒種 group 順 → from-square 昇順 → 不成/成) →
/// ③ 駒打ち (PAWN..ROOK 順 → to-square 昇順)．`board` は親 (受け方手番) 局面．
pub(super) fn evasion_order_key(board: &crate::board::Board, m: crate::moves::Move) -> u64 {
    if let Some(pt) = m.drop_piece_type() {
        // group 2 (駒打ちは最後)．evasion drop 順は **歩を全マス先に並べ，続いて各 to_sq ごとに
        // drops[] 順 (香→桂→銀→金→角→飛)** で生成する (例: P*8c P*8d P*8e P*8f / L*8c G*8c R*8c
        // / L*8d ...)．歩優先 (bit39) → to_sq 主 (bits 11-) → 駒種 従 (低 bit) とすることで同マス
        // drop が連続し，DML の同マス chain (to1==to2) が成立する (中合い G*8c が deferred される)．
        let pawn_first = if pt == crate::types::PieceType::Pawn {
            0u64
        } else {
            1u64
        };
        (2u64 << 40)
            | (pawn_first << 39)
            | ((m.to_sq().raw_u8() as u64) << 11)
            | drop_piece_order(pt)
    } else {
        let from = m.from_sq().raw_u8() as u64;
        let to = m.to_sq().raw_u8() as u64;
        let pf = if m.is_promotion() { 0u64 } else { 1u64 };
        let own_king = board.king_square(board.turn);
        if own_king == Some(m.from_sq()) {
            // group 0: 玉移動 (to-square 昇順)．
            to << 11
        } else {
            // group 1: 非玉の盤上移動 (駒種 group → from 昇順 → 成/不成 → to 昇順)．
            let raw_pt = board.piece_at(m.from_sq()) & 0x0F;
            let go = piece_gen_order(raw_pt);
            (1u64 << 40) | (go << 30) | (from << 20) | (pf << 19) | (to << 11)
        }
    }
}
