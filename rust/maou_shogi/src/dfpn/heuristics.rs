//! 探索ヒューリスティック: df-pn+ / DFPN-E の初期 pn/dn 推定と move ordering．

use crate::board::Board;
use crate::moves::Move;
use crate::types::{Color, Square};

use super::PN_UNIT;

/// DFPN-E (Kishimoto et al., NeurIPS 2019) エッジコスト型ヒューリスティック．
///
/// 標準 df-pn+ はノード(局面)の特徴で初期 pn/dn を設定するが，
/// DFPN-E は**エッジ(手)**の特徴に基づくコストを加算する．
/// 展開済みノードではエッジコストをゼロに戻すため，
/// 実質的には初期 pn への加算として機能する．
///
/// 詰将棋での手の質:
/// - OR ノードの王手: 成+取 > 取/成 > 近い静か手 > 遠い静か手
/// - AND ノードの応手: 合駒(攻め方有利) < 駒取り < 玉の逃げ

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
    if !m.is_drop() && !m.is_promotion() {
        if matches!(raw_pt, 1 | 5 | 6) {
            let from = m.from_sq();
            let us = board.turn;
            let in_enemy = |sq: Square| -> bool {
                let r = sq.row() as u8;
                match us {
                    Color::Black => r <= 2,
                    Color::White => r >= 6,
                }
            };
            if in_enemy(from) || in_enemy(to) {
                value += 1000;
            }
        }
    }

    let after_raw = if m.is_promotion() { raw_pt + 8 } else { raw_pt };
    let pt_value: i32 = match after_raw {
        1 => 10,                // Pawn
        2 => 20,                // Lance
        3 => 20,                // Knight
        4 => 30,                // Silver
        5 => 50,                // Bishop
        6 => 50,                // Rook
        7 => 50,                // Gold
        8 => 80, // King (AND 玉捕獲手の tie-break に必要)
        9 | 10 | 11 | 12 => 50, // ProPawn..ProSilver
        13 => 80, // Horse
        14 => 80, // Dragon
        _ => 0,
    };
    value -= pt_value;

    let dc = (to.col() as i32 - king_sq.col() as i32).abs();
    let dr = (to.row() as i32 - king_sq.row() as i32).abs();
    value += 10 * dc.max(dr);

    value
}
