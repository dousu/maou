//! KH (Komori df-pn) 流の極小証明駒 (minimal proof hand) 計算．
//!
//! 参照: `.tmp_diag/kh/hands.hpp` (`HandSet` / `AddIfHandGivesOtherEvasions` /
//! `RemoveIfHandGivesOtherChecks` / `BeforeHand`)．
//!
//! # なぜ必要か
//!
//! `mid_v2` は proven 局面を**実際の攻め方持ち駒**のまま TT に store する．TT lookup は
//! hand-dominance (`hand_gte_forward_chain`) で集約するが，store された持ち駒が極小でない
//! ため集約が弱く proof tree が膨れる (29te で KH 比 9×)．KH は AND ノードで全子の証明駒の
//! **要素 max**を取り，離れ王手の合駒要件を `AddIfHandGivesOtherEvasions` で補正することで
//! **極小証明駒**を構成する．本 module はその計算を maou の `[u8; HAND_KINDS]` 表現で忠実に
//! 再現する．
//!
//! # soundness の要諦
//!
//! 「攻め方持ち駒が減る = (駒数保存により) 防御方持ち駒が増える」ため，離れ王手では空証明駒
//! `[0; 7]` は unsound (防御が合駒駒を得て逃れうる)．`add_if_hand_gives_other_evasions` が
//! 「攻め方がその駒種を独占している」事実を証明駒に記録することで根治する．接触王手 (玉と
//! 王手駒の間が空) では合駒不能なので空証明駒で sound．
//!
//! # 駒種順序
//!
//! 本 module は終始 `PieceType::hand_index()` 順 (歩0 香1 桂2 銀3 金4 角5 飛6) を使う．
//! `drop_move_index()` (角4 飛5 金6) とは異なるため**絶対に混ぜない**．

use crate::attack;
use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::types::{Color, PieceType, HAND_KINDS};

/// 要素ごとの max (KH `HandSet::Update`, ProofHand 用)．
#[inline]
pub(super) fn hand_max(a: &[u8; HAND_KINDS], b: &[u8; HAND_KINDS]) -> [u8; HAND_KINDS] {
    let mut r = *a;
    for k in 0..HAND_KINDS {
        r[k] = r[k].max(b[k]);
    }
    r
}

/// 要素ごとの min (KH `HandSet::Update`, DisproofHand 用 — Stage 5)．
#[inline]
pub(super) fn hand_min(a: &[u8; HAND_KINDS], b: &[u8; HAND_KINDS]) -> [u8; HAND_KINDS] {
    let mut r = *a;
    for k in 0..HAND_KINDS {
        r[k] = r[k].min(b[k]);
    }
    r
}

/// 証明駒を攻め方の実際の持ち駒で clip する防御 guard (= legacy `mid` solver.rs:5274)．
///
/// 攻め方持ち駒を超える証明駒は決して再利用できず無意味なため，要素 min で抑える．
/// `add_if_hand_gives_other_evasions` は `board.hand[them]` (= 攻め方持ち駒) のみを加えるので
/// 本来 clip は no-op になるはずだが，多段集約での過大値・将来の改変に対する保険として適用する．
#[inline]
pub(super) fn hand_clip(ph: &[u8; HAND_KINDS], att: &[u8; HAND_KINDS]) -> [u8; HAND_KINDS] {
    let mut r = *ph;
    for k in 0..HAND_KINDS {
        r[k] = r[k].min(att[k]);
    }
    r
}

/// 手番側 (`us`) が合駒で歩を打てる合法マスの集合．
///
/// movegen の駒打ち生成 (movegen.rs:176-192) と同一規則: 空マス かつ 二歩でない筋 かつ
/// 行き所のない段でない．
#[inline]
fn legal_pawn_drop_mask(board: &Board, us: Color) -> Bitboard {
    let empty = !board.all_occupied();
    let our_pawns = board.piece_bb[us.index()][PieceType::Pawn as usize];
    let forbidden = match us {
        Color::Black => Bitboard::rank_mask(0),
        Color::White => Bitboard::rank_mask(8),
    };
    empty & !our_pawns.occupied_files() & !forbidden
}

/// KH `AddIfHandGivesOtherEvasions` (hands.hpp:193)．証明駒の soundness keystone．
///
/// AND ノード (防御側手番) で詰みが確定したとき，子局面の証明駒を集約した `ph` に対し，
/// **単一かつ離れ王手**なら「攻め方が独占する合駒駒」を証明駒へ記録する．
///
/// - `us` = `board.turn` = 防御側 (詰む側)．`them` = 攻め方．
/// - 王手駒が 1 枚でない (両王手・無王手) → 何もしない (KH `pop_count() != 1`)．
/// - 玉と王手駒の間が空 (接触王手) → 合駒不能 → 何もしない．
/// - 離れ王手: 防御が持っていない各駒種 `pr` について `ph[pr] = 攻め方の pr 枚数`．
///   (攻め方が pr を独占 ⇒ 防御は pr で合駒できない，という情報を証明駒に付与する．)
///   歩は合駒マスが歩打ち可能 (二歩でない等) のときのみ適用する．
pub(super) fn add_if_hand_gives_other_evasions(
    board: &Board,
    ph: [u8; HAND_KINDS],
) -> [u8; HAND_KINDS] {
    let us = board.turn;
    let them = us.opponent();

    let checkers = board.checkers_of(us);
    if checkers.count() != 1 {
        return ph;
    }
    let (king_sq, checker_sq) = match (board.king_square(us), checkers.lsb()) {
        (Some(k), Some(c)) => (k, c),
        _ => return ph,
    };
    let between = attack::between_bb(king_sq, checker_sq);
    if between.is_empty() {
        // 接触王手: 合駒できないので空証明駒のままで sound．
        return ph;
    }

    let mut ph = ph;
    let pawn_mask = legal_pawn_drop_mask(board, us);
    for (hi, &pt) in PieceType::HAND_PIECES.iter().enumerate() {
        if pt == PieceType::Pawn && (between & pawn_mask).is_empty() {
            // 合駒マスに歩を打てない (二歩 / 行き所なし) → 歩独占を記録しない．
            continue;
        }
        if board.hand[us.index()][hi] == 0 {
            // 防御が pr を持っていない → 攻めが独占 → 攻めの枚数を証明駒に記録．
            ph[hi] = board.hand[them.index()][hi];
        }
    }
    ph
}

/// 終端 (防御側 0 手 = 即詰) AND ノードの証明駒．
///
/// KH `HandSet{ProofHandTag}.Get(matedPos)` の子なし版 = `add_if(matedPos, 空)`．
/// 接触王手なら `[0; 7]`，離れ王手なら攻め方が独占する合駒駒を記録する．
#[inline]
pub(super) fn proof_hand_terminal_and(board: &Board) -> [u8; HAND_KINDS] {
    add_if_hand_gives_other_evasions(board, [0; HAND_KINDS])
}

/// KH `HandSet{ProofHandTag}` 相当の証明駒集約器 (AND ノード用)．
///
/// `update` で各子の証明駒を要素 max し，`get` で `AddIfHandGivesOtherEvasions` 補正を施す．
pub(super) struct ProofHandSet {
    val: [u8; HAND_KINDS],
}

impl ProofHandSet {
    #[inline]
    pub(super) fn new() -> Self {
        Self {
            val: [0; HAND_KINDS],
        }
    }

    /// 子局面の証明駒を要素 max で取り込む (KH `HandSet::Update` ProofHand)．
    #[inline]
    pub(super) fn update(&mut self, child: &[u8; HAND_KINDS]) {
        self.val = hand_max(&self.val, child);
    }

    /// 現局面 (AND, 防御側手番) の証明駒を取得する (KH `HandSet::Get` ProofHand)．
    #[inline]
    pub(super) fn get(&self, board: &Board) -> [u8; HAND_KINDS] {
        add_if_hand_gives_other_evasions(board, self.val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;

    fn board(sfen: &str) -> Board {
        let mut b = Board::empty();
        b.set_sfen(sfen).unwrap_or_else(|e| panic!("bad sfen {sfen}: {e:?}"));
        b
    }

    /// テスト前提の検証: 手番側が単一王手を受けているか，離れ王手かを確認する．
    fn assert_single_check(b: &Board, distant: bool) {
        let us = b.turn;
        let c = b.checkers_of(us);
        assert_eq!(c.count(), 1, "setup: expected exactly one checker");
        let between = attack::between_bb(b.king_square(us).unwrap(), c.lsb().unwrap());
        assert_eq!(
            between.is_not_empty(),
            distant,
            "setup: distant-check expectation mismatch"
        );
    }

    #[test]
    fn hand_max_min_clip_basic() {
        let a = [2, 0, 1, 0, 3, 0, 0];
        let b = [1, 5, 0, 0, 4, 0, 1];
        assert_eq!(hand_max(&a, &b), [2, 5, 1, 0, 4, 0, 1]);
        assert_eq!(hand_min(&a, &b), [1, 0, 0, 0, 3, 0, 0]);
        // clip: 証明駒 a を持ち駒 cap で抑える
        let cap = [1, 1, 1, 1, 1, 0, 0];
        assert_eq!(hand_clip(&a, &cap), [1, 0, 1, 0, 1, 0, 0]);
    }

    #[test]
    fn contact_check_terminal_is_empty() {
        // 白玉 5a に黒金 5b が接触王手．間が無いので証明駒は空．
        let b = board("4k4/4G4/9/9/9/9/9/9/4K4 w - 1");
        assert_single_check(&b, false);
        assert_eq!(proof_hand_terminal_and(&b), [0; HAND_KINDS]);
    }

    #[test]
    fn distant_check_monopolized_gold_recorded() {
        // 白玉 5a に黒香 5i が離れ王手．黒は金1枚を持ち駒に持つ．
        // 防御(白)は何も持っていないので，攻めが独占する金が証明駒に入る (金=index4)．
        // 香は盤上(持ち駒0)なので記録されない．
        let b = board("4k4/9/9/9/9/9/9/9/K3L4 w G 1");
        assert_single_check(&b, true);
        assert_eq!(proof_hand_terminal_and(&b), [0, 0, 0, 0, 1, 0, 0]);
    }

    #[test]
    fn distant_check_defender_has_gold_not_recorded() {
        // 同上だが防御(白)も金を1枚持つ → 防御が合駒できるので独占記録しない → 空．
        let b = board("4k4/9/9/9/9/9/9/9/K3L4 w Gg 1");
        assert_single_check(&b, true);
        assert_eq!(proof_hand_terminal_and(&b), [0; HAND_KINDS]);
    }

    #[test]
    fn distant_check_monopolized_pawn_recorded_when_legal() {
        // 離れ王手 (香) ＆ 黒が歩2枚持ち．間マス(5b..5h)は二歩でなく歩打ち可能なので
        // 歩独占が記録される (歩=index0=2)．
        let b = board("4k4/9/9/9/9/9/9/9/K3L4 w 2P 1");
        assert_single_check(&b, true);
        assert_eq!(proof_hand_terminal_and(&b), [2, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn distant_check_pawn_skipped_when_nifu_blocks_all() {
        // 白玉 1a に黒角 4d が離れ王手 (対角 1a-2b-3c-4d，間マス=2b,3c)．
        // 白は 2h,3h に歩 → 筋2,3 が二歩 → 間マス 2b,3c 双方で歩打ち不可 → 歩独占は記録しない．
        // 黒は歩1枚を持つが歩は skip されるので証明駒は空．
        let b = board("8k/9/9/5B3/9/9/9/6pp1/K8 w P 1");
        assert_single_check(&b, true);
        assert_eq!(proof_hand_terminal_and(&b), [0; HAND_KINDS]);
    }

    #[test]
    fn no_check_returns_input_unchanged() {
        // 無王手局面では add_if は恒等．
        let b = board("4k4/9/9/9/9/9/9/9/4K4 w - 1");
        assert_eq!(b.checkers_of(b.turn).count(), 0, "setup: no check");
        let input = [1, 2, 3, 0, 0, 0, 0];
        assert_eq!(add_if_hand_gives_other_evasions(&b, input), input);
    }

    #[test]
    fn proof_hand_set_update_then_get() {
        // 接触王手局面で 2 子の証明駒を要素 max 集約．get は接触なので add_if no-op．
        let b = board("4k4/4G4/9/9/9/9/9/9/4K4 w - 1");
        let mut set = ProofHandSet::new();
        set.update(&[2, 0, 0, 0, 0, 0, 0]);
        set.update(&[1, 0, 0, 2, 0, 0, 0]);
        assert_eq!(set.get(&b), [2, 0, 0, 2, 0, 0, 0]);
    }
}
