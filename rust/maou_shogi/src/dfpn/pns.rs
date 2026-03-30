//! 手生成，PNS (Proof Number Search)，PV 復元，公開 API．

use arrayvec::ArrayVec;
use rustc_hash::FxHashSet;
#[cfg(feature = "profile")]
use std::time::Instant;

use crate::attack;
use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::movegen;
use crate::moves::Move;
use crate::types::{Color, Piece, PieceType, Square, HAND_KINDS};

use super::entry::{PnsNode, PNS_MAX_ARENA_NODES};
use super::solver::{DfPnSolver, TsumeResult};
use super::{
    adjust_hand_for_move, edge_cost_and, edge_cost_or,
    position_key, propagate_nm_remaining, push_move, sacrifice_check_boost,
    INF, MAX_MOVES, PN_UNIT, REMAINING_INFINITE,
};

impl DfPnSolver {

    /// 玉方の王手回避手を生成する(合い効かずを除外)．
    ///
    /// 全合法手生成の代わりに回避手のみを直接生成する:
    /// 1. 玉の移動(攻め方に利かれていないマスへ)
    /// 2. 王手駒の捕獲(ピンされていない駒による)
    /// 3. 合い駒(飛び駒の王手の場合，間のマスへ移動または打つ)
    ///
    /// 合い効かず(futile interposition)もフィルタする．
    pub(super) fn generate_defense_moves(
        &mut self,
        board: &mut Board,
    ) -> ArrayVec<Move, MAX_MOVES> {
        self.generate_defense_moves_inner(board, false)
    }

    /// 王手回避手の内部実装．
    ///
    /// `early_exit == true` の場合，最初の合法手が見つかった時点で即座にリターンする．
    /// `has_any_defense` と `generate_defense_moves` の両方がこの関数を共有し，
    /// ロジックの重複を排除する．
    pub(super) fn generate_defense_moves_inner(
        &mut self,
        board: &mut Board,
        early_exit: bool,
    ) -> ArrayVec<Move, MAX_MOVES> {
        self.chain_bb_cache = Bitboard::EMPTY;
        let defender = board.turn;
        let attacker = defender.opponent();

        let king_sq = match board.king_square(defender) {
            Some(sq) => sq,
            None => {
                let legal = movegen::generate_legal_moves(board);
                let mut out = ArrayVec::new();
                for m in legal {
                    push_move(&mut out, m);
                    if early_exit {
                        return out;
                    }
                }
                return out;
            }
        };

        // 王手している駒を特定
        let checkers = board.compute_checkers_at(king_sq, attacker);
        if checkers.is_empty() {
            // 王手されていない(通常ありえないが安全策)
            let legal = movegen::generate_legal_moves(board);
            let mut out = ArrayVec::new();
            for m in legal {
                push_move(&mut out, m);
                if early_exit {
                    return out;
                }
            }
            return out;
        }

        let all_occ = board.all_occupied();
        let our_occ = board.occupied[defender.index()];
        let mut moves = ArrayVec::<Move, MAX_MOVES>::new();

        // --- 1. 玉の移動 ---
        let king_attacks = attack::step_attacks(defender, PieceType::King, king_sq);
        let king_targets = king_attacks & !our_occ;
        for to in king_targets {
            let captured_piece = board.squares[to.index()];
            let captured_raw = captured_piece.0;
            let m = Move::new_move(king_sq, to, false, captured_raw, PieceType::King as u8);
            // 移動先が安全か(攻め方に利かれていないか)チェック
            let captured = board.do_move(m);
            let safe = !board.is_in_check(defender);
            board.undo_move(m, captured);
            if safe {
                push_move(&mut moves, m);
                if early_exit {
                    return moves;
                }
            }
        }

        // 両王手の場合，玉移動のみ可能
        if checkers.count() > 1 {
            return moves;
        }

        // 単一の王手駒
        let checker_sq = checkers.lsb().unwrap();

        // --- 2. 王手駒の捕獲(玉以外の駒で) ---
        self.generate_capture_checker(
            board, &mut moves, checker_sq, king_sq, defender, all_occ, our_occ,
        );
        if early_exit && !moves.is_empty() {
            return moves;
        }

        // --- 3. 合い駒(飛び駒の王手の場合) ---
        let sliding_checker = self.find_sliding_checker(board, king_sq, attacker);
        if sliding_checker.is_some() {
            let between = attack::between_bb(checker_sq, king_sq);
            if between.is_not_empty() {
                // 合い効かず・チェーンマスを計算
                let (futile, chain) = self.compute_futile_and_chain_squares(
                    board, &between, king_sq, checker_sq, defender, attacker,
                );
                self.chain_bb_cache = chain;
                // 間のマスへの合い駒
                self.generate_interpositions(
                    board, &mut moves, &between, &futile, &chain, king_sq, defender, all_occ, our_occ,
                );
            }
        }

        moves
    }

    /// 王手駒を玉以外の駒で捕獲する手を生成する．
    pub(super) fn generate_capture_checker(
        &self,
        board: &mut Board,
        moves: &mut ArrayVec<Move, MAX_MOVES>,
        checker_sq: Square,
        king_sq: Square,
        defender: Color,
        all_occ: Bitboard,
        _our_occ: Bitboard,
    ) {
        // 逆引き利き計算: checker_sq を攻撃できる自駒のビットボードを直接求める．
        // is_attacked_by と同じ逆射パターンで，全自駒イテレーション(~16駒)を
        // 駒種ごとのビットボード AND(~10回)に置き換え，該当駒(2-4駒)のみ処理する．
        let attacker = defender.opponent();
        let d = defender.index();
        let mut can_capture = Bitboard::EMPTY;

        // 歩: checker_sq から「相手視点の歩の利き方向」に自歩があるか
        can_capture |= attack::step_attacks(attacker, PieceType::Pawn, checker_sq)
            & board.piece_bb[d][PieceType::Pawn as usize];
        // 桂
        can_capture |= attack::step_attacks(attacker, PieceType::Knight, checker_sq)
            & board.piece_bb[d][PieceType::Knight as usize];
        // 銀
        can_capture |= attack::step_attacks(attacker, PieceType::Silver, checker_sq)
            & board.piece_bb[d][PieceType::Silver as usize];
        // 金 + 成駒
        let gold_reach = attack::step_attacks(attacker, PieceType::Gold, checker_sq);
        can_capture |= gold_reach
            & (board.piece_bb[d][PieceType::Gold as usize]
                | board.piece_bb[d][PieceType::ProPawn as usize]
                | board.piece_bb[d][PieceType::ProLance as usize]
                | board.piece_bb[d][PieceType::ProKnight as usize]
                | board.piece_bb[d][PieceType::ProSilver as usize]);
        // 王・馬・龍(ステップ部分): 玉は後で除外
        let king_reach = attack::step_attacks(attacker, PieceType::King, checker_sq);
        can_capture |= king_reach
            & (board.piece_bb[d][PieceType::Horse as usize]
                | board.piece_bb[d][PieceType::Dragon as usize]);
        // 香
        can_capture |= attack::lance_attacks(attacker, checker_sq, all_occ)
            & board.piece_bb[d][PieceType::Lance as usize];
        // 角・馬
        can_capture |= attack::bishop_attacks(checker_sq, all_occ)
            & (board.piece_bb[d][PieceType::Bishop as usize]
                | board.piece_bb[d][PieceType::Horse as usize]);
        // 飛・龍
        can_capture |= attack::rook_attacks(checker_sq, all_occ)
            & (board.piece_bb[d][PieceType::Rook as usize]
                | board.piece_bb[d][PieceType::Dragon as usize]);

        // 玉を除外(玉による取りは呼び出し元で処理済み)
        can_capture.clear(king_sq);

        let captured_raw = board.squares[checker_sq.index()].0;

        while can_capture.is_not_empty() {
            let from = can_capture.pop_lsb();
            let piece = board.squares[from.index()];
            let pt = piece.piece_type().unwrap();
            let in_promo_zone =
                checker_sq.is_promotion_zone(defender) || from.is_promotion_zone(defender);

            if pt.can_promote() && in_promo_zone {
                let m = Move::new_move(from, checker_sq, true, captured_raw, pt as u8);
                if self.is_evasion_legal(board, m, defender) {
                    push_move(moves, m);
                }
                if !movegen::must_promote(defender, pt, checker_sq) {
                    let m = Move::new_move(from, checker_sq, false, captured_raw, pt as u8);
                    if self.is_evasion_legal(board, m, defender) {
                        push_move(moves, m);
                    }
                }
            } else if !movegen::must_promote(defender, pt, checker_sq) {
                let m = Move::new_move(from, checker_sq, false, captured_raw, pt as u8);
                if self.is_evasion_legal(board, m, defender) {
                    push_move(moves, m);
                }
            }
        }
    }

    /// 合い効かずマスとチェーンマス(駒打ち用)を計算する．
    ///
    /// 戻り値 `(futile, chain)`:
    /// - `futile`: 完全に無駄なマス(駒打ちスキップ)
    /// - `chain`: 中合いチェーン防御マス(代表駒のみドロップ生成)
    ///
    /// 各マスの判定基準:
    /// - 守備側(玉を除く)がそのマスに利いていれば，ひもがついているため無駄合いではない
    /// - 玉の隣接マスで，攻め方がチェッカー以外から利かせていなければ，
    ///   玉が取り返せるため無駄合いではない
    /// - それ以外は個別には無駄合い(チェッカーに取られて再び同筋の王手になる)
    ///
    /// ただし中合いチェーン防御を考慮する:
    /// マス X が上記基準で無駄合いでも，X と玉の間に非無駄合いマスが存在すれば，
    /// X への駒打ちは「捨て合い→チェッカー前進→…→非無駄合いマスでブロック」
    /// のチェーン防御の起点となり得る．このようなマスは `chain` に分類し，
    /// 最弱の駒(歩)のみをドロップ候補とする．
    ///
    /// 歩を代表とする根拠: 中合いで打った駒はチェッカーに取られるため，
    /// 攻め方に渡す駒が弱いほど守備側に有利(包含関係)．
    /// 歩の中合いで不詰みなら他の駒種でも不詰み，歩で詰みなら他も詰み．
    pub(super) fn compute_futile_and_chain_squares(
        &self,
        board: &Board,
        between: &Bitboard,
        king_sq: Square,
        checker_sq: Square,
        defender: Color,
        attacker: Color,
    ) -> (Bitboard, Bitboard) {
        let king_step = attack::step_attacks(
            defender.opponent(),
            PieceType::King,
            king_sq,
        );
        let mut futile = Bitboard::EMPTY;

        for sq in *between {
            // 守備側(玉除く)がひもをつけている → 取り返せるので無駄合いではない
            if board.is_attacked_by_excluding(sq, defender, true, None) {
                continue;
            }
            if king_step.contains(sq) {
                // 攻め方の他駒が利いていない → 玉が取り返せる → 無駄合いではない
                if !board.is_attacked_by_excluding(sq, attacker, false, Some(checker_sq)) {
                    continue;
                }
                // 飛び駒が取り進んだ後に玉の逃げ道があれば無駄合いではない
                if self.king_can_escape_after_slider_capture(
                    board, sq, checker_sq, king_sq, &king_step, defender, attacker,
                ) {
                    continue;
                }
            }
            futile.set(sq);
        }

        // 中合いチェーン伝搬: 玉側からチェッカー方向へ走査し，
        // 非無駄合いマスより遠い(チェッカー側の)無駄合いマスを chain に移す．
        let mut chain = Bitboard::EMPTY;
        if futile.is_not_empty() && futile != *between {
            let dc = (king_sq.col() as i32 - checker_sq.col() as i32).signum();
            let dr = (king_sq.row() as i32 - checker_sq.row() as i32).signum();
            let step_c = -dc;
            let step_r = -dr;
            let mut c = king_sq.col() as i32 + step_c;
            let mut r = king_sq.row() as i32 + step_r;
            let mut has_non_futile = false;
            while c >= 0 && c < 9 && r >= 0 && r < 9 {
                let sq = Square::new(c as u8, r as u8);
                if !between.contains(sq) {
                    break;
                }
                if !futile.contains(sq) {
                    has_non_futile = true;
                } else if has_non_futile {
                    futile.clear(sq);
                    chain.set(sq);
                }
                c += step_c;
                r += step_r;
            }
        }

        (futile, chain)
    }

    /// 飛び駒が玉隣接マスへ取り進んだ後に玉の逃げ道があるか判定する．
    ///
    /// 飛び駒が `capture_sq` へ移動した場合を想定し，
    /// 玉の全ステップ先が安全かどうかを検査する．
    /// 1つでも安全なマスがあれば `true`(= 無駄合いではない)を返す．
    ///
    /// NOTE: 玉が `king_sq` から離れることで新たに発生する素抜き攻撃
    /// (discovered attack) は考慮していない．このため判定は保守的
    /// (futile 判定を甘くする方向)に寄る．
    pub(super) fn king_can_escape_after_slider_capture(
        &self,
        board: &Board,
        capture_sq: Square,
        checker_sq: Square,
        _king_sq: Square,
        king_step: &Bitboard,
        defender: Color,
        attacker: Color,
    ) -> bool {
        let our_occ = board.occupied[defender.index()];

        // 飛び駒移動後の占有: checker_sq が空き，capture_sq に飛び駒が入る
        let mut occ = board.all_occupied();
        occ.clear(checker_sq);
        occ.set(capture_sq);

        // 飛び駒の capture_sq からの利きを計算
        let checker_piece = board.squares[checker_sq.index()];
        let checker_pt = match checker_piece.piece_type() {
            Some(pt) => pt,
            None => return false,
        };
        let slider_attacks = match checker_pt {
            PieceType::Rook => attack::rook_attacks(capture_sq, occ),
            PieceType::Dragon => {
                attack::rook_attacks(capture_sq, occ)
                    | attack::step_attacks(attacker, PieceType::King, capture_sq)
            }
            PieceType::Bishop => attack::bishop_attacks(capture_sq, occ),
            PieceType::Horse => {
                attack::bishop_attacks(capture_sq, occ)
                    | attack::step_attacks(attacker, PieceType::King, capture_sq)
            }
            PieceType::Lance => attack::lance_attacks(attacker, capture_sq, occ),
            _ => return false,
        };

        // 玉の逃げ先候補: 自駒のないマス(capture_sq の飛び駒は敵駒だが防御済み)
        let escape_candidates = *king_step & !our_occ;

        for esc in escape_candidates {
            // capture_sq は飛び駒が守られているため玉で取れない(既に確認済み)
            if esc == capture_sq {
                continue;
            }
            // 飛び駒の新位置から利かれているか
            if slider_attacks.contains(esc) {
                continue;
            }
            // 他の攻め駒(飛び駒の旧位置を除外)から利かれているか
            if board.is_attacked_by_excluding(esc, attacker, false, Some(checker_sq)) {
                continue;
            }
            // 安全な逃げ先がある → 無駄合いではない
            return true;
        }

        // 逃げ道なし → 詰み → 無駄合い
        false
    }

    /// 間のマスへの合い駒手を生成する(移動・打ち)．
    ///
    /// `futile`: 完全に無駄なマス(駒打ちスキップ)．
    /// `chain`: チェーン防御マス(代表駒のみドロップ生成)．
    ///
    /// chain マスでは包含関係を利用し，3カテゴリの代表駒のみ生成する:
    /// - 前方利き系: 歩(代表) ⊇ {歩,香,銀,金,飛}
    /// - 斜め利き系: 角(代表) — 前方に利かないため歩とは異なる包含
    /// - 跳躍系: 桂(代表) — 打てない段があり独立
    ///
    /// 中合いで打った駒はチェッカーに取られるため，攻め方に渡す駒が弱いほど
    /// 守備側に有利(包含関係)．各カテゴリ内では最弱の駒が代表となる．
    pub(super) fn generate_interpositions(
        &self,
        board: &mut Board,
        moves: &mut ArrayVec<Move, MAX_MOVES>,
        between: &Bitboard,
        futile: &Bitboard,
        chain: &Bitboard,
        king_sq: Square,
        defender: Color,
        all_occ: Bitboard,
        _our_occ: Bitboard,
    ) {
        let king_step = attack::step_attacks(
            defender.opponent(),
            PieceType::King,
            king_sq,
        );
        let attacker = defender.opponent();
        // futile | chain = 駒移動の無駄合いフィルタ対象
        let futile_or_chain = *futile | *chain;

        let d = defender.index();

        for to in *between {
            // --- 駒移動による合い駒 ---
            // 逆引き利き計算: to マスに到達できる自駒のビットボードを直接求める．
            let mut can_interpose = Bitboard::EMPTY;
            can_interpose |= attack::step_attacks(attacker, PieceType::Pawn, to)
                & board.piece_bb[d][PieceType::Pawn as usize];
            can_interpose |= attack::step_attacks(attacker, PieceType::Knight, to)
                & board.piece_bb[d][PieceType::Knight as usize];
            can_interpose |= attack::step_attacks(attacker, PieceType::Silver, to)
                & board.piece_bb[d][PieceType::Silver as usize];
            let gold_reach = attack::step_attacks(attacker, PieceType::Gold, to);
            can_interpose |= gold_reach
                & (board.piece_bb[d][PieceType::Gold as usize]
                    | board.piece_bb[d][PieceType::ProPawn as usize]
                    | board.piece_bb[d][PieceType::ProLance as usize]
                    | board.piece_bb[d][PieceType::ProKnight as usize]
                    | board.piece_bb[d][PieceType::ProSilver as usize]);
            let king_reach = attack::step_attacks(attacker, PieceType::King, to);
            can_interpose |= king_reach
                & (board.piece_bb[d][PieceType::Horse as usize]
                    | board.piece_bb[d][PieceType::Dragon as usize]);
            can_interpose |= attack::lance_attacks(attacker, to, all_occ)
                & board.piece_bb[d][PieceType::Lance as usize];
            can_interpose |= attack::bishop_attacks(to, all_occ)
                & (board.piece_bb[d][PieceType::Bishop as usize]
                    | board.piece_bb[d][PieceType::Horse as usize]);
            can_interpose |= attack::rook_attacks(to, all_occ)
                & (board.piece_bb[d][PieceType::Rook as usize]
                    | board.piece_bb[d][PieceType::Dragon as usize]);
            // 玉は合駒に使えない
            can_interpose.clear(king_sq);

            while can_interpose.is_not_empty() {
                let from = can_interpose.pop_lsb();
                let piece = board.squares[from.index()];
                let pt = piece.piece_type().unwrap();

                // 駒移動による合い駒の無駄合いフィルタ:
                // futile/chain マスへの移動でも，以下の場合は無駄合いではない:
                // (a) 移動後の駒にひもがついている(from を除いた守備側の利き)
                // (b) 移動元が玉の隣接マスで，空いた後に攻め方から利かれず
                //     玉の逃げ道が新たに生まれる
                if futile_or_chain.contains(to) {
                    let has_support = board.is_attacked_by_excluding(
                        to, defender, true, Some(from),
                    );
                    let opens_escape = king_step.contains(from)
                        && !board.is_attacked_by_excluding(from, attacker, false, None);
                    if !has_support && !opens_escape {
                        continue;
                    }
                }

                let captured_raw = board.squares[to.index()].0;
                let in_promo_zone =
                    to.is_promotion_zone(defender) || from.is_promotion_zone(defender);

                if pt.can_promote() && in_promo_zone {
                    let m = Move::new_move(from, to, true, captured_raw, pt as u8);
                    if self.is_evasion_legal(board, m, defender) {
                        push_move(moves, m);
                    }
                    if !movegen::must_promote(defender, pt, to) {
                        let m = Move::new_move(from, to, false, captured_raw, pt as u8);
                        if self.is_evasion_legal(board, m, defender) {
                            push_move(moves, m);
                        }
                    }
                } else if !movegen::must_promote(defender, pt, to) {
                    let m = Move::new_move(from, to, false, captured_raw, pt as u8);
                    if self.is_evasion_legal(board, m, defender) {
                        push_move(moves, m);
                    }
                }
            }

            // --- 駒打ちによる合い駒 ---
            if futile.contains(to) {
                // 完全無駄合い: スキップ
                continue;
            }
            let is_chain = chain.contains(to);
            if is_chain {
                // chain マス: 3カテゴリの代表駒のみ生成
                self.generate_chain_drops(board, moves, to, defender);
            } else {
                // 通常マス: 全駒種を弱い駒から生成(歩→香→桂→銀→金→角→飛)．
                // 弱い合駒を先に証明すると，攻め方が合駒を取った後の
                // 証明パスが TT に蓄積され，強い合駒探索時に同じ詰み筋を
                // TT ヒットで援用できる(攻め方の持ち駒が増えるため)．
                for (hand_idx, &pt) in PieceType::HAND_PIECES.iter().enumerate() {
                    if board.hand[defender.index()][hand_idx] == 0 {
                        continue;
                    }
                    if movegen::must_promote(defender, pt, to) {
                        continue;
                    }
                    if pt == PieceType::Pawn {
                        let our_pawns =
                            board.piece_bb[defender.index()][PieceType::Pawn as usize];
                        let col = to.col();
                        if (our_pawns & Bitboard::file_mask(col)).is_not_empty() {
                            continue;
                        }
                    }
                    let m = Move::new_drop(to, pt);
                    if pt == PieceType::Pawn && movegen::is_pawn_drop_mate(board, m) {
                        continue;
                    }
                    push_move(moves, m);
                }
            }
        }
    }

    /// チェーンマスへの代表駒ドロップを生成する．
    ///
    /// 3カテゴリの代表駒を試す:
    /// 1. 前方利き系: 歩→香→銀→金→飛(最弱の合法駒1つ)
    /// 2. 斜め利き系: 角
    /// 3. 跳躍系: 桂
    pub(super) fn generate_chain_drops(
        &self,
        board: &mut Board,
        moves: &mut ArrayVec<Move, MAX_MOVES>,
        to: Square,
        defender: Color,
    ) {
        let di = defender.index();

        // カテゴリ1: 前方利き系(歩,香,銀,金,飛) — 最弱の合法駒1つ
        const FORWARD_PIECES: [(usize, PieceType); 5] = [
            (0, PieceType::Pawn),   // hand_idx=0
            (1, PieceType::Lance),  // hand_idx=1
            (3, PieceType::Silver), // hand_idx=3
            (4, PieceType::Gold),   // hand_idx=4
            (6, PieceType::Rook),   // hand_idx=6
        ];
        for &(hand_idx, pt) in &FORWARD_PIECES {
            if board.hand[di][hand_idx] == 0 {
                continue;
            }
            if movegen::must_promote(defender, pt, to) {
                continue;
            }
            if pt == PieceType::Pawn {
                let our_pawns = board.piece_bb[di][PieceType::Pawn as usize];
                let col = to.col();
                if (our_pawns & Bitboard::file_mask(col)).is_not_empty() {
                    continue; // 二歩
                }
                let m = Move::new_drop(to, pt);
                if movegen::is_pawn_drop_mate(board, m) {
                    continue; // 打ち歩詰め
                }
                push_move(moves, m);
            } else {
                push_move(moves, Move::new_drop(to, pt));
            }
            break; // カテゴリ内最弱で代表
        }

        // カテゴリ2: 角
        let bishop_idx = 5; // HAND_PIECES[5] = Bishop
        if board.hand[di][bishop_idx] > 0
            && !movegen::must_promote(defender, PieceType::Bishop, to)
        {
            push_move(moves, Move::new_drop(to, PieceType::Bishop));
        }

        // カテゴリ3: 桂
        let knight_idx = 2; // HAND_PIECES[2] = Knight
        if board.hand[di][knight_idx] > 0
            && !movegen::must_promote(defender, PieceType::Knight, to)
        {
            push_move(moves, Move::new_drop(to, PieceType::Knight));
        }
    }

    /// 回避手の合法性チェック(ピンの確認)．
    #[inline]
    pub(super) fn is_evasion_legal(&self, board: &mut Board, m: Move, defender: Color) -> bool {
        let captured = board.do_move(m);
        let in_check = board.is_in_check(defender);
        board.undo_move(m, captured);
        !in_check
    }

    /// 飛び駒で王手している駒のマスを返す(単一の場合のみ)．
    ///
    /// 飛び駒が複数(両王手)の場合は None を返す(合い駒不可のため)．
    /// 飛び駒の王手がない場合も None を返す．
    pub(super) fn find_sliding_checker(
        &self,
        board: &Board,
        king_sq: Square,
        attacker: Color,
    ) -> Option<Square> {
        let occ = board.all_occupied();
        let att = attacker.index();

        let mut checkers = attack::rook_attacks(king_sq, occ)
            & (board.piece_bb[att][PieceType::Rook as usize]
                | board.piece_bb[att][PieceType::Dragon as usize]);
        checkers = checkers
            | (attack::bishop_attacks(king_sq, occ)
                & (board.piece_bb[att][PieceType::Bishop as usize]
                    | board.piece_bb[att][PieceType::Horse as usize]));
        // 香は防御側(玉方)の前方に利く:
        // lance_attacks(defender, king_sq, occ) で玉の前方レイを取得
        let defender = attacker.opponent();
        checkers = checkers
            | (attack::lance_attacks(defender, king_sq, occ)
                & board.piece_bb[att][PieceType::Lance as usize]);

        // 単一の飛び駒のみ対象
        if checkers.count() == 1 {
            checkers.lsb()
        } else {
            None
        }
    }

    /// 攻め方の王手になる手を生成する．
    ///
    /// 最適化: 玉方の玉に王手がかかる手のみを直接生成する．
    /// 全合法手を生成してからフィルタする方式と比べ，生成候補を大幅に削減する．
    pub(super) fn generate_check_moves(
        &self,
        board: &mut Board,
    ) -> ArrayVec<Move, MAX_MOVES> {
        let us = board.turn;
        let them = us.opponent();
        let has_own_king = board.king_square(us).is_some();

        let king_sq = match board.king_square(them) {
            Some(sq) => sq,
            None => return ArrayVec::new(),
        };

        let our_occ = board.occupied[us.index()];
        let all_occ = board.all_occupied();
        let empty = !all_occ;

        // 各駒種について「このマスに置くと玉に王手がかかる」ターゲットを事前計算
        // step_attacks(them, pt, king_sq) は「玉から見た逆利き」= 王手元になれるマス

        let mut moves = ArrayVec::<Move, MAX_MOVES>::new();

        // --- 1. 駒打ち: ターゲットマスのみに打つ ---
        for (hand_idx, &pt) in PieceType::HAND_PIECES.iter().enumerate() {
            if board.hand[us.index()][hand_idx] == 0 {
                continue;
            }

            // この駒種で王手になるマスを計算
            let check_targets = match pt {
                PieceType::Lance => attack::lance_attacks(them, king_sq, all_occ),
                PieceType::Bishop => attack::bishop_attacks(king_sq, all_occ),
                PieceType::Rook => attack::rook_attacks(king_sq, all_occ),
                _ => attack::step_attacks(them, pt, king_sq),
            };
            let mut drop_targets = check_targets & empty;

            // 二歩チェック: 歩が存在する筋を一括マスクして除外
            if pt == PieceType::Pawn {
                let our_pawns = board.piece_bb[us.index()][PieceType::Pawn as usize];
                drop_targets &= !our_pawns.occupied_files();
            }

            // 行き所のない駒の制限
            match pt {
                PieceType::Pawn | PieceType::Lance => {
                    let forbidden = match us {
                        Color::Black => Bitboard::rank_mask(0),
                        Color::White => Bitboard::rank_mask(8),
                    };
                    drop_targets &= !forbidden;
                }
                PieceType::Knight => {
                    let forbidden = match us {
                        Color::Black => Bitboard::rank_mask(0) | Bitboard::rank_mask(1),
                        Color::White => Bitboard::rank_mask(7) | Bitboard::rank_mask(8),
                    };
                    drop_targets &= !forbidden;
                }
                _ => {}
            }

            for to in drop_targets {
                let m = Move::new_drop(to, pt);
                // 打ち歩詰めチェック
                if pt == PieceType::Pawn && movegen::is_pawn_drop_mate(board, m) {
                    continue;
                }
                // 駒打ちは自玉への王手放置にならない(片玉でも両玉でも)
                push_move(&mut moves, m);
            }
        }

        // --- 2. 盤上の駒の移動 ---
        // 直接王手: 移動先から玉に利きがある手
        // 開き王手: 駒が移動することで背後のスライド駒から玉に利きが通る手

        // 開き王手の候補を事前計算:
        // 玉からのレイ上にいる自駒で，その間に他の駒がない場合，
        // そこから移動すると開き王手になりうる
        let discoverers = board.compute_discoverers(us, king_sq);

        let mut our_bb = our_occ;
        while our_bb.is_not_empty() {
            let from = our_bb.pop_lsb();
            let piece = board.squares[from.index()];
            let pt = piece.piece_type().unwrap();
            let attacks = attack::piece_attacks(us, pt, from, all_occ);
            let targets = attacks & !our_occ;

            let is_discoverer = discoverers.contains(from);

            for to in targets {
                let captured_piece = board.squares[to.index()];
                let captured_raw = captured_piece.0;
                let in_promo_zone = to.is_promotion_zone(us) || from.is_promotion_zone(us);

                // 開き王手の判定: to が from→king_sq のライン上にある場合，
                // 移動後も飛び駒の利きを遮断するため開き王手にならない
                let gives_discovered = is_discoverer
                    && !attack::line_through(from, king_sq).contains(to);

                // 成り先の駒種での王手チェック
                if pt.can_promote() && in_promo_zone {
                    let promoted_pt = pt.promoted().unwrap();
                    let gives_direct = self.attacks_square(us, promoted_pt, to, all_occ, king_sq);
                    if gives_direct || gives_discovered {
                        let m = Move::new_move(from, to, true, captured_raw, pt as u8);
                        if self.is_legal_quick(board, m, has_own_king) {
                            push_move(&mut moves, m);
                        }
                    }

                    // 不成
                    if !movegen::must_promote(us, pt, to) {
                        let gives_direct =
                            self.attacks_square(us, pt, to, all_occ, king_sq);
                        if gives_direct || gives_discovered {
                            let m = Move::new_move(from, to, false, captured_raw, pt as u8);
                            if self.is_legal_quick(board, m, has_own_king) {
                                push_move(&mut moves, m);
                            }
                        }
                    }
                } else if !movegen::must_promote(us, pt, to) {
                    let gives_direct = self.attacks_square(us, pt, to, all_occ, king_sq);
                    if gives_direct || gives_discovered {
                        let m = Move::new_move(from, to, false, captured_raw, pt as u8);
                        if self.is_legal_quick(board, m, has_own_king) {
                            push_move(&mut moves, m);
                        }
                    }
                }
            }
        }

        // 手順序: 成+取 > 成 > 取 > その他，同カテゴリ内は玉との距離でタイブレーク
        // 近接王手を優先し，詰みに至る手を早期発見する．
        // 距離はチェビシェフ距離(0-8)を使用し，カテゴリ(0-3) * 16 + 距離 で
        // 単一キーにエンコードする．
        let king_col = king_sq.col() as i8;
        let king_row = king_sq.row() as i8;
        moves.sort_unstable_by_key(|m| {
            let promo = m.is_promotion();
            let capture = m.captured_piece_raw() > 0;
            let category: u8 = match (promo, capture) {
                (true, true) => 0,
                (true, false) => 1,
                (false, true) => 2,
                (false, false) => 3,
            };
            let to = m.to_sq();
            let dc = (to.col() as i8 - king_col).unsigned_abs();
            let dr = (to.row() as i8 - king_row).unsigned_abs();
            let dist = dc.max(dr); // チェビシェフ距離(0-8)
            (category as u16) * 16 + dist as u16
        });

        moves
    }

    /// 指定マスに置いた駒が玉のマスに利いているか判定する．
    #[inline]
    pub(super) fn attacks_square(
        &self,
        color: Color,
        pt: PieceType,
        from: Square,
        occ: Bitboard,
        target: Square,
    ) -> bool {
        attack::piece_attacks(color, pt, from, occ).contains(target)
    }

    /// 開き王手の元になりうる自駒を計算する．
    /// 合法性の簡易チェック(自玉の王手放置のみ)．
    ///
    /// 片玉の場合(自玉なし)は常に合法．
    #[inline]
    pub(super) fn is_legal_quick(&self, board: &mut Board, m: Move, has_own_king: bool) -> bool {
        if !has_own_king {
            return true;
        }
        let us = board.turn;
        let captured = board.do_move(m);
        let in_check = board.is_in_check(us);
        board.undo_move(m, captured);
        !in_check
    }

    /// PV パス上の OR ノードで未証明の子ノードを追加証明する．
    ///
    /// Df-Pn は OR ノードで1つの子ノードが証明されると他を未探索のまま残す．
    /// PV 抽出で正確な最短詰み手数を計算するため，PV 上の OR ノードで
    /// 未証明の王手を追加証明する．反復的に PV を更新し収束させる．
    ///
    /// # 制限事項
    ///
    /// - 反復回数は2回固定．長手数(29手詰め等)で PV 上の未証明子が多い場合，
    ///   2回の反復では PV が収束せず `CheckmateNoPv` になることがある．
    ///   この場合 `pv_nodes_per_child` を増やしても改善されない．
    /// - `extract_pv_recursive` の深度制限(`depth * 2`)により PV 復元が
    ///   打ち切られるケースでも `CheckmateNoPv` になる(後述)．
    pub(super) fn complete_or_proofs(&mut self, board: &mut Board) {
        let saved_max = self.max_nodes;
        // 証明完了フェーズのノード予算:
        //   主探索ノード数と pv_nodes_per_child*8 の小さい方を追加予算とする．
        //   ただし短手数の詰将棋 (少ノードで解けた場合) でも PV 復元に
        //   十分なノードを確保するため，最低 pv_nodes_per_child ノードを保証する．
        let mid_nodes = self.nodes_searched;
        let budget_cap = self.pv_nodes_per_child.saturating_mul(8);

        // Phase 1: PV を抽出 → PV 上の OR ノードを完成 → 再抽出
        // find_shortest のときは最大 4 回反復:
        //   追加証明で PV が変化 → 新 PV 上の未証明 OR 子がさらにある →
        //   もう 1 回証明 → PV が安定，という収束サイクルを回す．
        // changed == false で早期終了するため，収束済みの場合は追加コストなし．
        self.max_nodes =
            self.nodes_searched.saturating_add(
                mid_nodes.min(budget_cap).max(self.pv_nodes_per_child),
            );

        // Phase 1: PV を抽出 → PV 上の OR ノードを完成 → 再抽出
        // 2回固定: 1回目で新たに証明された子が PV を短縮する可能性があるため
        // 2回目を実行する．changed == false で早期終了するため，
        // 収束済みの場合は追加コストなし．
        for _ in 0..2 {
            if self.is_timed_out()
                || self.nodes_searched >= self.max_nodes
            {
                break;
            }
            let pv = self.extract_pv_limited(board, 100_000);
            if pv.is_empty() {
                break;
            }
            let changed =
                self.complete_pv_or_nodes(board, &pv);
            if !changed {
                break;
            }
        }

        self.max_nodes = saved_max;
    }

    /// PV 上の各 OR ノードで未証明子ノードの追加証明を試みる．
    ///
    /// クローンした盤面で PV を辿り，各 OR ノードで未証明の王手を追加証明する．
    /// 返り値: 新たに証明された子ノードがあれば true．
    pub(super) fn complete_pv_or_nodes(&mut self, board: &mut Board, pv: &[Move]) -> bool {
        let mut board_clone = board.clone();
        let mut any_changed = false;

        for (i, pv_move) in pv.iter().enumerate() {
            let or_node = i % 2 == 0;
            let ply = i as u32;

            if or_node {
                let moves = self.generate_check_moves(&mut board_clone);
                for m in &moves {
                    if self.nodes_searched >= self.max_nodes {
                        break;
                    }
                    let captured = board_clone.do_move(*m);
                    let (cpn, cdn) = self.look_up_board(&board_clone);

                    if cpn != 0 && cdn != 0 {
                        let saved = self.max_nodes;
                        // 1子あたり pv_nodes_per_child ノード上限．
                        // PV 沿いの各未証明子に対する追加証明の予算．
                        self.max_nodes = self
                            .nodes_searched
                            .saturating_add(self.pv_nodes_per_child)
                            .min(saved);
                        self.mid(&mut board_clone, INF - 1, INF - 1, ply + 1, false);
                        self.max_nodes = saved;

                        if self.look_up_board(&board_clone).0 == 0 {
                            any_changed = true;
                        }
                    }

                    board_clone.undo_move(*m, captured);
                }
            }

            // PV に沿って盤面を進める(前進専用: undo_move は不要)
            let _captured = board_clone.do_move(*pv_move);
        }

        any_changed
    }

    /// 訪問数制限付き PV 復元．
    pub(super) fn extract_pv_limited(&mut self, board: &mut Board, max_visits: u64) -> Vec<Move> {
        let mut board_clone = board.clone();
        let mut visits = 0u64;
        self.extract_pv_recursive_inner(
            &mut board_clone,
            true,
            &mut FxHashSet::default(),
            0,
            false,
            &mut visits,
            max_visits,
        )
    }

    /// PV 復元の再帰実装．
    ///
    /// 各ノードで全候補手のサブPVを生成し，攻め方は最短，玉方は最長を選ぶ．
    /// ループ検出にはフルハッシュ，TT 参照には位置キー＋持ち駒を使用する．
    ///
    /// # 深度制限
    ///
    /// 再帰深度は `self.depth * 2` で打ち切る．OR/AND が交互に呼ばれるため，
    /// depth=31 なら最大62手の PV を復元できる．理論上，AND ノードで
    /// 複数の応手候補を評価する際に ply が詰み手数の2倍を超えうるが，
    /// TT に証明済みエントリがあれば即座に返るため，実用上は十分な深さである．
    ///
    /// ただし `complete_pv_or_nodes` で新たに証明された子が PV を変化させた場合，
    /// AND ノードの応手評価で ply が急増し，深度制限に達して空リストが返ることがある．
    /// この場合 `pv_nodes_per_child` を増やしても改善されず，`depth` の増加が必要．
    pub(super) fn extract_pv_recursive(
        &mut self,
        board: &mut Board,
        or_node: bool,
        visited: &mut FxHashSet<u64>,
        ply: u32,
    ) -> Vec<Move> {
        let mut visits = 0u64;
        self.extract_pv_recursive_inner(board, or_node, visited, ply, false, &mut visits, u64::MAX)
    }

    /// `extract_pv_recursive` の内部実装．
    ///
    /// `diag` が true の場合，各plyでの候補手・sub_pv長・選択理由を
    /// 標準エラーに出力する．
    /// `visits` は TT 参照回数を計測し，`max_visits` を超えると空リストを返す．
    pub(super) fn extract_pv_recursive_inner(
        &mut self,
        board: &mut Board,
        or_node: bool,
        visited: &mut FxHashSet<u64>,
        ply: u32,
        diag: bool,
        visits: &mut u64,
        max_visits: u64,
    ) -> Vec<Move> {
        // 訪問数制限チェック
        *visits += 1;
        if *visits > max_visits {
            return Vec::new();
        }

        // スタックオーバーフロー防止: 探索手数の2倍を再帰深度の上限とする
        if ply >= self.depth.saturating_mul(2) {
            if diag {
                verbose_eprintln!("[PV diag] ply={} depth_limit reached (max={})", ply, self.depth.saturating_mul(2));
            }
            return Vec::new();
        }
        let full_hash = board.hash;

        // ループ検出(フルハッシュ)
        if visited.contains(&full_hash) {
            if diag {
                verbose_eprintln!("[PV diag] ply={} loop detected hash={:#x}", ply, full_hash);
            }
            return Vec::new();
        }

        let (node_pn, _node_dn) = self.look_up_board(board);

        if or_node {
            if node_pn != 0 {
                if diag {
                    verbose_eprintln!("[PV diag] ply={} OR node unproven pn={}", ply, node_pn);
                }
                return Vec::new();
            }

            let moves = self.generate_check_moves(board);
            if moves.is_empty() {
                if diag {
                    verbose_eprintln!("[PV diag] ply={} OR node no check moves", ply);
                }
                return Vec::new();
            }

            if diag {
                verbose_eprintln!("[PV diag] ply={} OR node, {} check moves", ply, moves.len());
            }

            let mut best_pv: Option<Vec<Move>> = None;

            for m in &moves {
                if *visits > max_visits {
                    break;
                }
                let captured = board.do_move(*m);
                let (child_pn, _) = self.look_up_board(board);

                if child_pn == 0 {
                    visited.insert(full_hash);
                    let sub_pv =
                        self.extract_pv_recursive_inner(
                            board, false, visited, ply + 1, diag,
                            visits, max_visits,
                        );
                    visited.remove(&full_hash);

                    let total_len = 1 + sub_pv.len();
                    // OR ノードの PV は奇数長でなければならない
                    // (攻め方の手で始まり攻め方の手で終わる)
                    if total_len % 2 == 0 && !sub_pv.is_empty() {
                        if diag {
                            verbose_eprintln!(
                                "[PV diag] ply={} OR skip {} (even len={}, sub={})",
                                ply, m.to_usi(), total_len, sub_pv.len()
                            );
                        }
                        board.undo_move(*m, captured);
                        continue;
                    }
                    let is_better = match &best_pv {
                        None => true,
                        Some(prev) => total_len < prev.len(),
                    };

                    if diag {
                        verbose_eprintln!(
                            "[PV diag] ply={} OR candidate {} len={} better={}{}",
                            ply, m.to_usi(), total_len, is_better,
                            if let Some(prev) = &best_pv {
                                format!(" (prev_best={})", prev.len())
                            } else {
                                String::new()
                            }
                        );
                    }

                    if is_better {
                        let mut pv = vec![*m];
                        pv.extend(sub_pv);
                        best_pv = Some(pv);
                    }
                } else if diag {
                    verbose_eprintln!(
                        "[PV diag] ply={} OR child {} unproven pn={}",
                        ply, m.to_usi(), child_pn
                    );
                }

                board.undo_move(*m, captured);
            }

            #[cfg(feature = "verbose")]
            if diag {
                if let Some(ref pv) = best_pv {
                    eprintln!(
                        "[PV diag] ply={} OR CHOSEN {} (pv_len={})",
                        ply, pv[0].to_usi(), pv.len()
                    );
                } else {
                    eprintln!("[PV diag] ply={} OR no valid PV found", ply);
                }
            }

            best_pv.unwrap_or_default()
        } else {
            // AND ノード: 呼び出し側の OR ノードで child_pn == 0 を
            // 確認した後のみ再帰するため，この局面は証明済みのはず．
            let moves = self.generate_defense_moves(board);
            if moves.is_empty() {
                if diag {
                    verbose_eprintln!("[PV diag] ply={} AND node no defense (checkmate)", ply);
                }
                return Vec::new();
            }

            if diag {
                verbose_eprintln!("[PV diag] ply={} AND node, {} defense moves", ply, moves.len());
            }

            let mut best_pv: Option<Vec<Move>> = None;
            let mut best_is_capture = false;
            let mut best_is_drop = false;

            for m in &moves {
                if *visits > max_visits {
                    break;
                }
                let captured = board.do_move(*m);
                let (child_pn, _) = self.look_up_board(board);

                if child_pn == 0 {
                    visited.insert(full_hash);
                    let sub_pv =
                        self.extract_pv_recursive_inner(
                            board, true, visited, ply + 1, diag,
                            visits, max_visits,
                        );
                    visited.remove(&full_hash);

                    let total_len = 1 + sub_pv.len();
                    // AND ノードの PV は偶数長でなければならない
                    // (玉方の手で始まり攻め方の手で終わる)
                    if total_len % 2 == 1 {
                        if diag {
                            verbose_eprintln!(
                                "[PV diag] ply={} AND skip {} (odd len={}, sub={})",
                                ply, m.to_usi(), total_len, sub_pv.len()
                            );
                        }
                        board.undo_move(*m, captured);
                        continue;
                    }
                    let is_capture = m.captured_piece_raw() > 0;
                    let is_drop = m.is_drop();
                    let is_better = match &best_pv {
                        None => true,
                        Some(prev) => {
                            if total_len > prev.len() {
                                true
                            } else if total_len == prev.len()
                                && is_capture
                                && !best_is_capture
                            {
                                true
                            } else if total_len == prev.len()
                                && is_drop
                                && !best_is_drop
                                && is_capture == best_is_capture
                            {
                                // 同率 & 駒取り状況も同じ場合，
                                // 合駒(打ち駒)を優先する．
                                // 打ち駒のサブツリーは探索で重点的に証明
                                // されやすく，sub_pv が正確な傾向がある．
                                true
                            } else {
                                false
                            }
                        }
                    };

                    if diag {
                        verbose_eprintln!(
                            "[PV diag] ply={} AND candidate {} len={} capture={} drop={} better={}{}",
                            ply, m.to_usi(), total_len, is_capture, is_drop, is_better,
                            if let Some(prev) = &best_pv {
                                format!(" (prev_best={} prev_cap={} prev_drop={})", prev.len(), best_is_capture, best_is_drop)
                            } else {
                                String::new()
                            }
                        );
                    }

                    if is_better {
                        let mut pv = vec![*m];
                        pv.extend(sub_pv);
                        best_pv = Some(pv);
                        best_is_capture = is_capture;
                        best_is_drop = is_drop;
                    }
                } else if diag {
                    verbose_eprintln!(
                        "[PV diag] ply={} AND child {} unproven pn={}",
                        ply, m.to_usi(), child_pn
                    );
                }

                board.undo_move(*m, captured);
            }

            #[cfg(feature = "verbose")]
            if diag {
                if let Some(ref pv) = best_pv {
                    eprintln!(
                        "[PV diag] ply={} AND CHOSEN {} (pv_len={})",
                        ply, pv[0].to_usi(), pv.len()
                    );
                } else {
                    eprintln!("[PV diag] ply={} AND no valid PV found", ply);
                }
            }

            best_pv.unwrap_or_default()
        }
    }

    /// 探索ノード数を返す．
    pub fn nodes_searched(&self) -> u64 {
        self.nodes_searched
    }

    /// IDS-dfpn (MID) フォールバック．
    ///
    /// PNS がアリーナ上限に達した場合に呼び出される．
    /// PNS で蓄積された TT エントリ(証明・中間値)を引き継ぎ，
    /// 残りのノード予算で IDS-dfpn を実行する．
    pub(super) fn mid_fallback(&mut self, board: &mut Board) {
        let pk = position_key(board);
        let att_hand = board.hand[self.attacker.index()];
        self.diag_root_pk = pk;
        self.diag_root_hand = att_hand;
        let saved_depth = self.depth;
        let mut ids_depth: u32 = 2;
        let total_max_nodes = self.max_nodes;
        // PNS で蓄積された中間エントリ(pn>0, dn>0)を除去し，
        // 証明(pn=0)のみ保持する．
        //
        // 中間エントリを保持すると以下の問題が発生する:
        // 1. HashMap サイズ増大により CPU キャッシュ効率が低下し NPS が半減する
        //    (338K entries → ~126 NPS vs 12K entries → ~194 NPS)．
        // 2. MID の child init で cpn>1/cdn>1 として扱われ，
        //    底辺の簡単な詰みを再発見する機会が失われる．
        //
        // 証明エントリ(pn=0)は prefilter/cross_deduce に直接活用される．
        self.table.retain_proofs_only();

        #[cfg(feature = "tt_diag")]
        eprintln!("[mid_fallback] after retain_proofs_only: TT_pos={} nodes_so_far={} total_budget={}",
            self.table.len(), self.nodes_searched, total_max_nodes);

        // 停滞検出用: 前回の IDS 反復終了時の root pn/dn を保持する．
        // IDS 反復後に root_pn/dn が変化しなかった場合，MID が
        // dn 閾値カスケード縮退により進捗不能と判断する．
        let mut prev_root_pn: u32 = 0;
        let mut prev_root_dn: u32 = 0;

        // 合駒チェーン最適化の IDS 反復間デルタ追跡
        #[cfg(feature = "tt_diag")]
        let mut prev_prefilter_hits = self.prefilter_hits;
        #[cfg(feature = "tt_diag")]
        let mut prev_cross_deduce = self.diag_cross_deduce_hits;
        #[cfg(feature = "tt_diag")]
        let mut prev_deferred_act = self.diag_mid_deferred_activations;
        #[cfg(feature = "tt_diag")]
        let mut prev_prefilter_skip = self.diag_prefilter_skip_remaining;
        #[cfg(feature = "tt_diag")]
        let mut prev_prefilter_miss = self.diag_prefilter_miss;

        loop {
            if ids_depth > saved_depth {
                ids_depth = saved_depth;
            }
            self.depth = ids_depth;
            self.path.clear();
            let remaining = ids_depth as u16;
            let (root_pn, _, _) = self.look_up_pn_dn(pk, &att_hand, remaining);
            verbose_eprintln!("[ids] depth={}/{} root_pn={} nodes={} time={:.1}s",
                ids_depth, saved_depth, root_pn, self.nodes_searched,
                self.start_time.elapsed().as_secs_f64());
            if root_pn == 0 {
                verbose_eprintln!("[ids] root proved, break");
                break;
            }
            let _budget = if ids_depth < saved_depth {
                // 残り IDS ステップ数に応じた予算配分．
                let remaining_budget =
                    total_max_nodes.saturating_sub(self.nodes_searched);
                let remaining_steps =
                    ((saved_depth.saturating_sub(ids_depth)) / 2)
                        .max(1) as u64 + 1;
                let b = (remaining_budget / (remaining_steps + 1))
                    .max(1024);
                self.max_nodes = self.nodes_searched.saturating_add(b);
                b
            } else {
                // フルデプス: 残り予算の全てを割り当て．
                self.max_nodes = total_max_nodes;
                total_max_nodes.saturating_sub(self.nodes_searched)
            };

            #[cfg(feature = "tt_diag")]
            let pre_nodes = self.nodes_searched;
            #[cfg(feature = "tt_diag")]
            let pre_max_ply = self.max_ply;
            // IDS 反復ごとに max_ply をリセットし，各反復の到達深さを追跡する
            self.max_ply = 0;

            if ids_depth == saved_depth {
                // フルデプス: MID 先行(動的予算) + Frontier Variant フォールバック．
                //
                // 方針B(§10.2): MID を固定 1/2 予算ではなくチャンク分割し，
                // TT エントリ成長を監視する．MID が新しい TT エントリを
                // 生成しなくなった時点で停滞と判定し，残り予算を動的に
                // Frontier に回す．方針A で MID の閾値余裕が拡大したため，
                // MID が進捗できる範囲をまず効率的に処理し，
                // 閾値飢餓で停滞した時点で速やかに Frontier に切り替える．
                //
                // TT 清掃なしでシームレスに遷移: MID が蓄積した
                // 証明・反証・中間エントリを Frontier がそのまま活用する．
                let remaining_budget =
                    total_max_nodes.saturating_sub(self.nodes_searched);
                // MID の最大予算は全体の 1/2(従来と同じ上限)
                let mid_max_budget = remaining_budget / 2;
                // チャンクサイズ: 1M (TT 停滞の早期検出のため固定)
                let chunk_size: u64 = 1_000_000;
                let mid_deadline = self.nodes_searched.saturating_add(mid_max_budget);
                let mut prev_tt_len = self.table.len();

                while self.nodes_searched < mid_deadline && !self.timed_out {
                    let chunk_end = (self.nodes_searched + chunk_size).min(mid_deadline);
                    self.max_nodes = chunk_end;
                    let (root_pn, root_dn, _) = self.look_up_pn_dn(pk, &att_hand, remaining);
                    if root_pn == 0 || root_dn == 0 {
                        break; // 証明/反証完了
                    }
                    self.mid(board, INF - 1, INF - 1, 0, true);

                    let (r_pn, r_dn, _) = self.look_up_pn_dn(pk, &att_hand, remaining);
                    if r_pn == 0 || r_dn == 0 {
                        break; // 証明/反証完了
                    }

                    // TT 成長チェック: 新規エントリが生成されていなければ停滞
                    let curr_tt_len = self.table.len();
                    if curr_tt_len <= prev_tt_len {
                        verbose_eprintln!("[ids] MID stagnation: TT {} → {}, shifting to Frontier",
                            prev_tt_len, curr_tt_len);
                        break;
                    }
                    prev_tt_len = curr_tt_len;
                }

                // MID で未解決 → Frontier Variant にフォールバック
                let (r_pn, r_dn, _) = self.look_up_pn_dn(pk, &att_hand, remaining);
                if r_pn != 0 && r_dn != 0
                    && self.nodes_searched < total_max_nodes
                    && !self.timed_out
                {
                    self.max_nodes = total_max_nodes;
                    self.frontier_variant(board, total_max_nodes);
                }
            } else {
                // 浅い反復: 通常の MID で TT をウォームアップ
                let (root_pn, root_dn, _) = self.look_up_pn_dn(pk, &att_hand, remaining);
                if root_pn != 0 && root_dn != 0
                    && self.nodes_searched < self.max_nodes
                    && !self.timed_out
                {
                    verbose_eprintln!("[ids] calling MID: depth={} root_pn={} root_dn={} nodes={}",
                        ids_depth, root_pn, root_dn, self.nodes_searched);
                    #[cfg(feature = "profile")]
                    let _mid_wall_start = Instant::now();
                    self.mid(board, INF - 1, INF - 1, 0, true);
                    verbose_eprintln!("[ids] MID returned: depth={} nodes={} time={:.1}s",
                        ids_depth, self.nodes_searched, self.start_time.elapsed().as_secs_f64());
                    #[cfg(feature = "profile")]
                    {
                        self.profile_stats.mid_total_ns += _mid_wall_start.elapsed().as_nanos() as u64;
                        self.profile_stats.mid_total_count += 1;
                    }
                }
            }

            #[cfg(feature = "tt_diag")]
            {
                let used = self.nodes_searched - pre_nodes;
                let (pn, dn, _) = self.look_up_pn_dn(pk, &att_hand, remaining);
                let d_prefilter = self.prefilter_hits - prev_prefilter_hits;
                let d_cross = self.diag_cross_deduce_hits - prev_cross_deduce;
                let d_deferred = self.diag_mid_deferred_activations - prev_deferred_act;
                let d_pf_skip = self.diag_prefilter_skip_remaining - prev_prefilter_skip;
                let d_pf_miss = self.diag_prefilter_miss - prev_prefilter_miss;
                let ids_elapsed = self.start_time.elapsed().as_secs_f64();
                verbose_eprintln!("[ids_diag] depth={}/{} budget={} used={} TT_pos={} root_pn={} root_dn={} max_ply={} \
                    prefilter_hit={} prefilter_skip_rem={} prefilter_miss={} cross={} act={} \
                    cap_tt={}/{} thr_exits={} term_exits={} in_path={} lb_prov={} lb_thr={} lb_nodes={} \
                    init_and_dis={} single_ch={} node_lim={} time={:.2}s",
                    ids_depth, saved_depth, _budget, used,
                    self.table.len(), pn, dn,
                    self.max_ply,
                    d_prefilter, d_pf_skip, d_pf_miss,
                    d_cross, d_deferred,
                    self.diag_capture_tt_hits, self.diag_capture_tt_calls,
                    self.diag_threshold_exits,
                    self.diag_terminal_exits,
                    self.diag_in_path_exits,
                    self.diag_loop_break_proved,
                    self.diag_loop_break_threshold,
                    self.diag_loop_break_nodes,
                    self.diag_init_and_disproof_exits,
                    self.diag_single_child_exits,
                    self.diag_node_limit_exits,
                    ids_elapsed);
                // ply ヒストグラム出力(非ゼロ ply のみ)
                let mut ply_str = String::new();
                for p in 0..64 {
                    if self.diag_ply_visits[p] > 0 {
                        if !ply_str.is_empty() { ply_str.push_str(", "); }
                        ply_str.push_str(&format!("{}:{}", p, self.diag_ply_visits[p]));
                    }
                }
                if !ply_str.is_empty() {
                    verbose_eprintln!("[ids_diag] ply_visits: {}", ply_str);
                }
                // single-child counter (repurposed diag_ply_proofs)
                let mut sc_str = String::new();
                for p in 0..64 {
                    if self.diag_ply_proofs[p] > 0 {
                        if !sc_str.is_empty() { sc_str.push_str(", "); }
                        sc_str.push_str(&format!("{}:{}", p, self.diag_ply_proofs[p]));
                    }
                }
                if !sc_str.is_empty() {
                    verbose_eprintln!("[ids_diag] single_child: {}", sc_str);
                }
                // リセット
                self.diag_ply_visits = [0u64; 64];
                self.diag_ply_proofs = [0u64; 64];
                self.diag_capture_tt_calls = 0;
                self.diag_capture_tt_hits = 0;
                self.diag_threshold_exits = 0;
                self.diag_terminal_exits = 0;
                self.diag_loop_break_proved = 0;
                self.diag_loop_break_threshold = 0;
                self.diag_loop_break_nodes = 0;
                self.diag_in_path_exits = 0;
                self.diag_init_and_disproof_exits = 0;
                self.diag_single_child_exits = 0;
                self.diag_node_limit_exits = 0;
                prev_prefilter_hits = self.prefilter_hits;
                prev_cross_deduce = self.diag_cross_deduce_hits;
                prev_deferred_act = self.diag_mid_deferred_activations;
                prev_prefilter_skip = self.diag_prefilter_skip_remaining;
                prev_prefilter_miss = self.diag_prefilter_miss;
            }
            let (root_pn2, root_dn2, _) = self.look_up_pn_dn(pk, &att_hand, remaining);
            verbose_eprintln!("[ids] after MID: depth={} root_pn={} root_dn={} nodes={} time={:.1}s",
                ids_depth, root_pn2, root_dn2, self.nodes_searched,
                self.start_time.elapsed().as_secs_f64());
            if root_pn2 == 0 {
                verbose_eprintln!("[ids] proved at depth={}, break", ids_depth);
                break;
            }
            // IDS NM 判定: 構造的判定のみ信頼する．
            //
            // 1. NM remaining が REMAINING_INFINITE なら真の不詰として打ち切る．
            //    MID が深さ制限なしに全パスを網羅して NM を確定した場合にのみ
            //    REMAINING_INFINITE が伝搬される．
            // 2. 全王手が再帰的に反証可能なら REMAINING_INFINITE に昇格して打ち切る．
            //
            // depth 制限由来の仮 NM (remaining < REMAINING_INFINITE) は昇格しない．
            // 深い詰みが存在する局面でも浅い IDS 深さでは NM になるのが当然であり，
            // これを真の不詰と判定すると偽陽性が発生する(例: 39手詰め)．
            if root_dn2 == 0 {
                let root_nm_rem = self.table.get_disproof_remaining(pk, &att_hand);
                verbose_eprintln!("[ids] NM: depth={} nm_rem={} REMAINING_INFINITE={}",
                    ids_depth, root_nm_rem, REMAINING_INFINITE);
                if root_nm_rem == REMAINING_INFINITE {
                    verbose_eprintln!("[ids] NM INFINITE, break");
                    break;
                }
                let checks = self.generate_check_moves(board);
                let refutable = if checks.is_empty() {
                    true
                } else {
                    self.depth_limit_all_checks_refutable(board, &checks)
                };
                verbose_eprintln!("[ids] refutable={} checks={}", refutable, checks.len());
                if checks.is_empty() || refutable {
                    verbose_eprintln!("[ids] NM promoted to INFINITE, break");
                    // att_hand で保存(TT ヒット率最大化)
                    self.table.store(
                        pk, att_hand, INF, 0, REMAINING_INFINITE, pk,
                    );
                    break;
                }
                // depth 制限由来の仮 NM → IDS を続行してより深い探索で再検証．
            }
            if self.nodes_searched >= total_max_nodes || self.timed_out {
                break;
            }
            if ids_depth >= saved_depth {
                break;
            }

            // 停滞検出: 予算を使い切ったのに root pn/dn が変化していない場合，
            // dn 閾値カスケード縮退により MID が深部に到達できていない．
            let stagnated = root_pn2 > 0
                && root_dn2 > 0
                && root_pn2 == prev_root_pn
                && root_dn2 == prev_root_dn;

            // 時間ベースの反復カットオフ:
            // 浅い IDS 反復が残り時間の 1/4 以上消費した場合，
            // この深さでの NPS が低く後続の深さに時間を回せないため
            // 最大深さへ即座にジャンプする．
            let total_elapsed = self.start_time.elapsed().as_secs_f64();
            let remaining_time = self.timeout.as_secs_f64() - total_elapsed;
            let time_exceeded = ids_depth < saved_depth
                && remaining_time < self.timeout.as_secs_f64() * 0.25;

            // IDS 反復間でのクリーンアップ:
            // 1. 経路依存の反証を除去(ループ由来，異なる深さでは無効)
            // 2. 浅い反復のスラッシング防止エントリと浅い反証を除去
            self.table.remove_path_dependent_disproofs();
            self.table.remove_stale_for_ids();

            if stagnated || time_exceeded {
                ids_depth = saved_depth;
            } else {
                // 深さ進行:
                // saved_depth > 31 の場合: 段階的 IDS (2→4→8→16→32→41)
                //   深い問題では段階的に TT を構築して探索効率を上げる
                // saved_depth <= 31 の場合: 直接ジャンプ (2→4→31)
                //   浅い問題では中間深さの探索が無駄になるため
                let next = ids_depth.saturating_mul(2).max(ids_depth + 2);
                if saved_depth <= 31 && next > 4 && next < saved_depth {
                    ids_depth = saved_depth;
                } else {
                    ids_depth = next.min(saved_depth);
                }
            }

            prev_root_pn = root_pn2;
            prev_root_dn = root_dn2;
        }
        self.depth = saved_depth;
        self.max_nodes = total_max_nodes;
    }

    /// Frontier Variant: PNS→局所 MID サイクル．
    ///
    /// IDS のフルデプス反復で使用される．浅い IDS 反復で蓄積された
    /// TT 証明・確定反証を活用しつつ，PNS のグローバル最適選択で
    /// MID の閾値飢餓(§10.4)を回避する．
    ///
    /// ### PNS→MID サイクルの相乗効果
    ///
    /// - PNS の TT 書き込みが MID の child init で TT ヒット率を向上させる．
    /// - MID の証明蓄積が次の PNS サイクルでのフロンティア選択精度を向上させる．
    /// - 各サイクルで MID 後に `retain_proofs()` を呼び，
    ///   中間エントリの蓄積による PNS フロンティア選択の汚染を防止する．
    ///   証明(pn=0)と確定反証(dn=0, 非経路依存)は保持される．
    fn frontier_variant(&mut self, board: &mut Board, total_max_nodes: u64) {
        let pk = position_key(board);
        let att_hand = board.hand[self.attacker.index()];

        let mut frontier_iters = 0u32;
        const MAX_FRONTIER_ITERS: u32 = 50;

        while frontier_iters < MAX_FRONTIER_ITERS
            && self.nodes_searched < total_max_nodes
            && !self.timed_out
        {
            frontier_iters += 1;
            let remaining_budget = total_max_nodes.saturating_sub(self.nodes_searched);
            if remaining_budget < 10_000 {
                break;
            }

            // PNS フェーズ: TT を更新しフロンティアを特定
            let pns_budget = (remaining_budget / 20).max(10_000).min(50_000);
            self.max_nodes = self.nodes_searched.saturating_add(pns_budget);
            let _pv = self.pns_main(board);

            let (r_pn, r_dn, _) = self.look_up_pn_dn(pk, &att_hand, self.depth as u16);
            if r_pn == 0 || r_dn == 0 {
                break; // PNS で証明または反証完了
            }

            // MID フェーズ: PNS で更新された TT を活用して局所探索
            let remaining_budget2 = total_max_nodes.saturating_sub(self.nodes_searched);
            let mid_budget = (remaining_budget2 / 4).max(50_000).min(remaining_budget2);
            self.max_nodes = self.nodes_searched.saturating_add(mid_budget);
            self.path.clear();

            let (r_pn2, r_dn2, _) = self.look_up_pn_dn(pk, &att_hand, self.depth as u16);
            if r_pn2 == 0 || r_dn2 == 0 {
                break;
            }
            self.mid(board, INF - 1, INF - 1, 0, true);

            let (r_pn3, r_dn3, _) = self.look_up_pn_dn(pk, &att_hand, self.depth as u16);
            if r_pn3 == 0 || r_dn3 == 0 {
                break;
            }

            // サイクル間 TT 清掃: MID が蓄積した中間エントリを除去し，
            // 次の PNS サイクルが新鮮な状態でフロンティアを選択できるようにする．
            // 証明(pn=0)と確定反証(dn=0, 非経路依存)は保持する．
            self.table.retain_proofs();
        }
        self.max_nodes = total_max_nodes;
    }

    // ================================================================
    // Best-First Proof Number Search (PNS)
    // ================================================================

    /// Best-First PNS メインループ．
    ///
    /// 明示的な探索木(アリーナ)上で most-proving node を選択・展開し，
    /// pn/dn をルートまでバックアップする．df-pn の閾値制御を必要とせず，
    /// グローバルに最適なノード選択を行う．
    ///
    /// アリーナが `PNS_MAX_ARENA_NODES` に達した場合は探索を打ち切り，
    /// 呼び出し元で MID ベースの探索にフォールバックする．
    pub(super) fn pns_main(&mut self, board: &mut Board) -> Option<Vec<Move>> {
        let mut arena: Vec<PnsNode> = Vec::with_capacity(
            PNS_MAX_ARENA_NODES.min(1024 * 1024),
        );

        // ルートノード生成
        let pk = position_key(board);
        let fh = board.hash;
        let hand = board.hand[self.attacker.index()];
        arena.push(PnsNode {
            pos_key: pk,
            full_hash: fh,
            hand,
            pn: PN_UNIT,
            dn: PN_UNIT,
            parent: u32::MAX,
            move_from_parent: Move(0),
            or_node: true,
            expanded: false,
            children: Vec::new(),
            remaining: self.depth as u16,
            deferred_drops: Vec::new(),
        });

        // 再利用バッファ(ループ内のアロケーション回避)
        let max_path = self.depth as usize + 2;
        let mut path: Vec<u32> = Vec::with_capacity(max_path);
        let mut captures: Vec<Piece> = Vec::with_capacity(max_path);
        let mut ancestors: FxHashSet<u64> =
            FxHashSet::with_capacity_and_hasher(max_path, Default::default());

        // PNS メインループ
        let mut pns_iters: u64 = 0;
        // PNS 収束検出: root_pn が一定反復数改善しなければ打ち切る．
        // PNS はアリーナ飽和後に選択ウォークだけを繰り返し
        // 予算を消費するため，早期打ち切りで MID に予算を回す．
        let mut best_root_pn: u32 = u32::MAX;
        let mut iters_since_improvement: u64 = 0;
        const PNS_STAGNATION_LIMIT: u64 = 500_000;
        loop {
            pns_iters += 1;
            // 終了条件: ルート証明/反証
            if arena[0].pn == 0 || arena[0].dn == 0 {
                break;
            }
            // 終了条件: ノード制限・タイムアウト
            if self.nodes_searched >= self.max_nodes || self.timed_out {
                break;
            }
            // 終了条件: アリーナ満杯
            if arena.len() >= PNS_MAX_ARENA_NODES {
                break;
            }
            // 終了条件: PNS 収束停滞
            if arena[0].pn < best_root_pn {
                best_root_pn = arena[0].pn;
                iters_since_improvement = 0;
            } else {
                iters_since_improvement += 1;
                if iters_since_improvement >= PNS_STAGNATION_LIMIT {
                    #[cfg(feature = "tt_diag")]
                    eprintln!("[pns_diag] stagnation: root_pn={} no improvement for {} iters, stopping",
                        arena[0].pn, PNS_STAGNATION_LIMIT);
                    break;
                }
            }
            // 定期タイムアウトチェック
            if pns_iters & 0xFF == 0 && self.is_timed_out() {
                self.timed_out = true;
                break;
            }

            // Most-proving node 選択 + 盤面復元
            path.clear();
            captures.clear();
            ancestors.clear();
            path.push(0);
            ancestors.insert(arena[0].full_hash);
            let mut current = 0u32;

            let mut skip_expand = false;

            while arena[current as usize].expanded {
                let ci = current as usize;

                // AND ノード: 全子証明済み + deferred_drops → 逐次活性化
                if !arena[ci].or_node && !arena[ci].deferred_drops.is_empty() {
                    let all_proven = arena[ci].children.iter()
                        .all(|&c| arena[c as usize].pn == 0);
                    if all_proven {
                        // 証明済み子の直接 OR ノードのみ TT に flush．
                        // 中間ノード(子ありの AND/OR)は actual hand で格納するため
                        // 既存の中間エントリを不正に evict し MID 収束を妨げる恐れがある．
                        // 直接 OR 子ノードのみ flush することで，try_prefilter_block が
                        // 「合駒後の OR ノードが証明済みか」を判定できるようにする．
                        for &child_idx in &arena[ci].children {
                            let child = &arena[child_idx as usize];
                            if child.pn == 0 && child.or_node {
                                self.store(
                                    child.pos_key, child.hand, 0, INF,
                                    REMAINING_INFINITE, child.pos_key,
                                );
                            }
                        }

                        let and_remaining = arena[ci].remaining;
                        let mut and_proof = [0u8; HAND_KINDS];

                        let mut activated_unproven = false;
                        while !arena[ci].deferred_drops.is_empty() {
                            let next_drop = arena[ci].deferred_drops.remove(0);
                            #[cfg(feature = "tt_diag")]
                            { self.diag_pns_deferred_activations += 1; }
                            #[cfg(feature = "tt_diag")]
                            eprintln!(
                                "[pns_seq] AND node idx={}: activate drop {} (deferred_remaining={}), tt_entries={}",
                                ci, next_drop.to_usi(), arena[ci].deferred_drops.len(),
                                self.table.total_entries(),
                            );

                            // TT ベースプレフィルタ: 合駒の捕獲後局面が
                            // 既に TT で証明済みなら展開不要．
                            // PNS flush で先行子の証明が TT に反映された後に
                            // チェックすることで，同一マスの兄弟合駒を
                            // TT ヒットで即座に証明できる．
                            if and_remaining >= 3 {
                                let cap_pre = board.do_move(next_drop);
                                let child_hand_pre = board.hand[self.attacker.index()];
                                board.undo_move(next_drop, cap_pre);
                                if self.try_prefilter_block(
                                    board, next_drop, &child_hand_pre,
                                    and_remaining, &mut and_proof,
                                ) {
                                    self.prefilter_hits += 1;
                                    #[cfg(feature = "tt_diag")]
                                    {
                                        self.diag_pns_deferred_already_proven += 1;
                                        verbose_eprintln!(
                                            "[pns_seq]   prefilter hit for {} → proven",
                                            next_drop.to_usi(),
                                        );
                                    }
                                    // アリーナにも証明済みとして追加
                                    let cap_pf = board.do_move(next_drop);
                                    let child_pk_pf = position_key(board);
                                    let child_fh_pf = board.hash;
                                    let child_hand_pf =
                                        board.hand[self.attacker.index()];
                                    let child_remaining_pf =
                                        and_remaining.saturating_sub(1);
                                    board.undo_move(next_drop, cap_pf);
                                    let pf_idx = arena.len() as u32;
                                    arena.push(PnsNode {
                                        pos_key: child_pk_pf,
                                        full_hash: child_fh_pf,
                                        hand: child_hand_pf,
                                        pn: 0,
                                        dn: INF,
                                        parent: current,
                                        move_from_parent: next_drop,
                                        or_node: true,
                                        expanded: true,
                                        children: Vec::new(),
                                        remaining: child_remaining_pf,
                                        deferred_drops: Vec::new(),
                                    });
                                    arena[ci].children.push(pf_idx);
                                    continue;
                                }
                            }

                            let cap = board.do_move(next_drop);
                            let child_fh = board.hash;
                            let child_pk = position_key(board);
                            let child_hand = board.hand[self.attacker.index()];
                            let child_remaining =
                                arena[ci].remaining.saturating_sub(1);

                            let is_loop = ancestors.contains(&child_fh);
                            let (cpn, cdn) = if is_loop {
                                (INF, 0u32)
                            } else {
                                let (p, d, _) = self.look_up_pn_dn(
                                    child_pk, &child_hand, child_remaining,
                                );
                                (p, d)
                            };

                            board.undo_move(next_drop, cap);

                            // 反証(防御成功) → AND ノード反証
                            if cdn == 0 {
                                arena[ci].pn = INF;
                                arena[ci].dn = 0;
                                skip_expand = true;
                                activated_unproven = true;
                                break;
                            }

                            // 子ノード生成
                            let child_idx = arena.len() as u32;
                            arena.push(PnsNode {
                                pos_key: child_pk,
                                full_hash: child_fh,
                                hand: child_hand,
                                pn: cpn,
                                dn: cdn,
                                parent: current,
                                move_from_parent: next_drop,
                                or_node: true,
                                expanded: cpn == 0 || cdn == 0,
                                children: Vec::new(),
                                remaining: child_remaining,
                                deferred_drops: Vec::new(),
                            });
                            arena[ci].children.push(child_idx);

                            // 証明済み → 次の deferred drop へ
                            if cpn == 0 {
                                #[cfg(feature = "tt_diag")]
                                { self.diag_pns_deferred_already_proven += 1; }
                                continue;
                            }

                            // 未証明 → この子を MPN として選択
                            let cap = board.do_move(next_drop);
                            captures.push(cap);
                            path.push(child_idx);
                            ancestors.insert(child_fh);
                            current = child_idx;
                            activated_unproven = true;
                            break;
                        }

                        if activated_unproven {
                            break; // MPN 選択ループ終了
                        }

                        // 全 deferred が証明済み → AND ノード証明
                        arena[ci].pn = 0;
                        arena[ci].dn = INF;
                        skip_expand = true;
                        break;
                    }
                }

                // 通常の子ノード選択
                if arena[ci].children.is_empty() {
                    // 子ノードなし(展開済みだが全消去等) → リーフとして再展開
                    break;
                }
                let best_child = if arena[ci].or_node {
                    *arena[ci].children.iter()
                        .min_by_key(|&&c| (arena[c as usize].pn, arena[c as usize].dn))
                        .expect("children non-empty (guarded above)")
                } else {
                    *arena[ci].children.iter()
                        .min_by_key(|&&c| (arena[c as usize].dn, arena[c as usize].pn))
                        .expect("children non-empty (guarded above)")
                };
                let child_move = arena[best_child as usize].move_from_parent;
                let captured = board.do_move(child_move);
                captures.push(captured);
                path.push(best_child);
                ancestors.insert(arena[best_child as usize].full_hash);
                current = best_child;
            }

            // リーフ展開(逐次活性化で解決済みの場合はスキップ)
            if !skip_expand {
                let ply = (path.len() - 1) as u32;
                self.pns_expand(board, &mut arena, current, ply, &ancestors);
            }

            // 盤面をルートに戻す
            for i in (1..path.len()).rev() {
                let child_move = arena[path[i] as usize].move_from_parent;
                board.undo_move(child_move, captures[i - 1]);
            }

            // バックアップ: 展開ノードからルートまで pn/dn を更新
            Self::pns_backup(&mut arena, current);
        }

        // 診断: PNS 終了時の状態
        #[cfg(feature = "tt_diag")]
        {
            let pns_nodes_used = self.nodes_searched;
            let root_pn = arena[0].pn;
            let root_dn = arena[0].dn;
            let pns_elapsed = self.start_time.elapsed().as_secs_f64();
            verbose_eprintln!("[pns_diag] arena={}/{} iters={} nodes_used={} root_pn={} root_dn={} TT_pos={} time={:.2}s",
                arena.len(), PNS_MAX_ARENA_NODES, pns_iters, pns_nodes_used, root_pn, root_dn,
                self.table.len(), pns_elapsed);
        }

        // 証明/反証結果を TT に格納(PV 抽出用)
        self.pns_store_to_tt(&arena);

        // デバッグ: 証明ツリーの整合性チェック
        #[cfg(debug_assertions)]
        if arena[0].pn == 0 {
            Self::validate_pns_proof(&arena, 0);
        }

        // ルートが証明済みならアリーナから直接 PV を抽出
        if arena[0].pn == 0 {
            let mut visited: FxHashSet<u64> = FxHashSet::default();
            let pv = self.pns_extract_pv(board, &arena, 0, &mut visited);
            if !pv.is_empty() && pv.len() % 2 == 1 {
                return Some(pv);
            }
        }
        None
    }

    /// PNS ノード展開: リーフノードの子を生成し初期 pn/dn を設定する．
    ///
    /// 子ノードの初期化では既存 TT エントリおよびヒューリスティック
    /// (DFPN-E エッジコスト，Deep df-pn 深さバイアス，静的詰め判定)を利用する．
    pub(super) fn pns_expand(
        &mut self,
        board: &mut Board,
        arena: &mut Vec<PnsNode>,
        node_idx: u32,
        ply: u32,
        ancestors: &FxHashSet<u64>,
    ) {
        self.nodes_searched += 1;
        if ply > self.max_ply {
            self.max_ply = ply;
        }

        let or_node = arena[node_idx as usize].or_node;
        let remaining = arena[node_idx as usize].remaining;
        let pos_key = arena[node_idx as usize].pos_key;
        let att_hand = arena[node_idx as usize].hand;

        // 終端: 深さ制限 / 手数制限
        if remaining == 0 || ply >= self.depth || board.ply() as u32 >= self.draw_ply {
            arena[node_idx as usize].pn = INF;
            arena[node_idx as usize].dn = 0;
            arena[node_idx as usize].expanded = true;
            if or_node {
                // OR ノードの深さ制限: 王手が0手なら真の不詰(REMAINING_INFINITE)．
                // 王手がある場合: 2手延長で全王手が即座に反証可能かを確認．
                let checks = self.generate_check_moves(board);
                if checks.is_empty() {
                    self.store(pos_key, att_hand, INF, 0,
                        REMAINING_INFINITE, pos_key);
                } else if self.depth_limit_all_checks_refutable(board, &checks) {
                    self.store(pos_key, att_hand, INF, 0,
                        REMAINING_INFINITE, pos_key);
                } else {
                    self.store(pos_key, att_hand, INF, 0, 0, pos_key);
                }
            } else {
                self.store(pos_key, att_hand, INF, 0, 0, pos_key);
            }
            return;
        }

        // 合法手生成
        let moves = if or_node {
            self.generate_check_moves(board)
        } else {
            self.generate_defense_moves(board)
        };

        if moves.is_empty() {
            if or_node {
                // 王手手段なし → 不詰
                arena[node_idx as usize].pn = INF;
                arena[node_idx as usize].dn = 0;
                self.store(pos_key, att_hand, INF, 0, REMAINING_INFINITE, pos_key);
            } else {
                // 応手なし → 詰み
                arena[node_idx as usize].pn = 0;
                arena[node_idx as usize].dn = INF;
                self.store(pos_key, [0; HAND_KINDS], 0, INF, REMAINING_INFINITE, pos_key);
            }
            arena[node_idx as usize].expanded = true;
            return;
        }

        // DFPN-E: 守備側玉の位置(OR ノードのエッジコスト計算用)
        let defender_king_sq = if or_node {
            board.king_square(board.turn.opponent())
        } else {
            None
        };

        let child_remaining = remaining.saturating_sub(1);
        let child_or_node = !or_node;
        let mut or_nm_min_remaining: u16 = REMAINING_INFINITE;
        // AND ノードの合駒逐次活性化: 最初の unproven drop のみ子ノード生成
        let mut first_unproven_drop_added = or_node; // OR ノードでは無効

        for m in &moves {
            let captured = board.do_move(*m);
            let child_fh = board.hash;
            let child_pk = position_key(board);
            let child_hand = board.hand[self.attacker.index()];

            // ループ検出: 祖先と同一局面なら即座に不詰/無限ループ扱い
            let is_loop = ancestors.contains(&child_fh);

            let (mut cpn, mut cdn) = if is_loop {
                (INF, 0u32)
            } else {
                let (p, d, _) = self.look_up_pn_dn(child_pk, &child_hand, child_remaining);
                (p, d)
            };

            // TT に初期値(1,1)しかない場合: ヒューリスティック初期化
            if cpn == PN_UNIT && cdn == PN_UNIT && !is_loop {
                if child_or_node {
                    // 子は OR ノード(攻め方手番): 王手数ベース
                    let checks = self.generate_check_moves(board);
                    if checks.is_empty() {
                        cpn = INF;
                        cdn = 0;
                        self.store(child_pk, child_hand, INF, 0, REMAINING_INFINITE, child_pk);
                    } else if self.has_mate_in_1_with(board, &checks) {
                        cpn = 0;
                        cdn = INF;
                        self.store(child_pk, child_hand, 0, INF, REMAINING_INFINITE, child_pk);
                    } else {
                        let nc = checks.len() as u32;
                        cpn = self.heuristic_or_pn(board, nc)
                            .saturating_add(edge_cost_and(*m))
                            .saturating_add(sacrifice_check_boost(board, &checks));
                        cdn = PN_UNIT;
                        self.store(child_pk, child_hand, cpn, cdn, child_remaining, child_pk);
                    }
                } else {
                    // 子は AND ノード(玉方手番): 応手数ベース
                    let defenses = self.generate_defense_moves(board);
                    if defenses.is_empty() {
                        cpn = 0;
                        cdn = INF;
                        self.store(child_pk, [0; HAND_KINDS], 0, INF, REMAINING_INFINITE, child_pk);
                    } else {
                        let n = defenses.len() as u32;
                        cpn = self.heuristic_and_pn(board, n);
                        if let Some(ksq) = defender_king_sq {
                            cpn = cpn.saturating_add(edge_cost_or(*m, ksq));
                        }
                        cdn = PN_UNIT;
                        self.store(child_pk, child_hand, cpn, cdn, child_remaining, child_pk);
                    }
                }
            }

            board.undo_move(*m, captured);

            // OR ノードで子が即座に証明 → 親を証明して終了
            if or_node && cpn == 0 {
                let child_ph = self.table.get_proof_hand(child_pk, &child_hand);
                let mut proof = adjust_hand_for_move(*m, &child_ph);
                for k in 0..HAND_KINDS {
                    proof[k] = proof[k].min(att_hand[k]);
                }
                arena[node_idx as usize].pn = 0;
                arena[node_idx as usize].dn = INF;
                arena[node_idx as usize].expanded = true;
                self.store(pos_key, proof, 0, INF, REMAINING_INFINITE, pos_key);
                return;
            }
            // AND ノードで子が即座に反証 → 親を反証して終了
            // att_hand で保存(TT ヒット率最大化)
            if !or_node && cdn == 0 {
                arena[node_idx as usize].pn = INF;
                arena[node_idx as usize].dn = 0;
                arena[node_idx as usize].expanded = true;
                let child_rem = self.table.get_effective_disproof_info(
                    child_pk, &child_hand, child_remaining,
                ).map(|(r, _)| r).unwrap_or(0);
                let prop_rem = propagate_nm_remaining(child_rem, remaining);
                self.store(pos_key, att_hand, INF, 0, prop_rem, pos_key);
                return;
            }
            // OR ノードで子が反証済み → 子を追加せずスキップ
            if or_node && cdn == 0 {
                let child_rem = self.table.get_effective_disproof_info(
                    child_pk, &child_hand, child_remaining,
                ).map(|(r, _)| r).unwrap_or(0);
                or_nm_min_remaining = or_nm_min_remaining.min(child_rem);
                continue;
            }

            // AND ノードの合駒逐次活性化:
            // 未証明の drop を1つだけ子ノードとして展開し，残りは deferred_drops に格納．
            // 弱い駒から順に証明 → TT 蓄積 → 強い駒の探索で援用する．
            // 既に証明済み(cpn=0)の drop はそのまま子ノードとして追加する．
            if !or_node && m.is_drop() && cpn > 0 {
                if first_unproven_drop_added {
                    #[cfg(feature = "tt_diag")]
                    eprintln!(
                        "[pns_seq] pns_expand: defer drop {} at AND node idx={}",
                        m.to_usi(), node_idx,
                    );
                    arena[node_idx as usize].deferred_drops.push(*m);
                    continue;
                }
                first_unproven_drop_added = true;
            }

            let child_idx = arena.len() as u32;
            arena.push(PnsNode {
                pos_key: child_pk,
                full_hash: child_fh,
                hand: child_hand,
                pn: cpn,
                dn: cdn,
                parent: node_idx,
                move_from_parent: *m,
                or_node: child_or_node,
                expanded: cpn == 0 || cdn == 0,
                children: Vec::new(),
                remaining: child_remaining,
                deferred_drops: Vec::new(),
            });
            arena[node_idx as usize].children.push(child_idx);
        }

        // OR ノードで全子が反証済み(children 空)
        if or_node && arena[node_idx as usize].children.is_empty() {
            arena[node_idx as usize].pn = INF;
            arena[node_idx as usize].dn = 0;
            arena[node_idx as usize].expanded = true;
            let mut prop_rem = propagate_nm_remaining(or_nm_min_remaining, remaining);
            // 2手延長による REMAINING_INFINITE 昇格
            if prop_rem != REMAINING_INFINITE {
                let checks = self.generate_check_moves(board);
                if checks.is_empty()
                    || self.depth_limit_all_checks_refutable(board, &checks)
                {
                    prop_rem = REMAINING_INFINITE;
                }
            }
            self.store(pos_key, att_hand, INF, 0, prop_rem, pos_key);
            return;
        }

        arena[node_idx as usize].expanded = true;
    }

    /// PNS バックアップ: 展開ノードからルートまで pn/dn を再計算する．
    ///
    /// OR ノード: pn = min(child_pn), dn = sum(child_dn)
    /// AND ノード: WPN = max(child_pn) + (unproven_count - 1), dn = min(child_dn)
    /// pn/dn が変化しなくなった時点で伝播を打ち切る．
    pub(super) fn pns_backup(arena: &mut [PnsNode], start_idx: u32) {
        let mut current = start_idx;
        loop {
            let ni = current as usize;

            if !arena[ni].expanded || arena[ni].children.is_empty() {
                // 終端ノード(子なし): pn/dn は展開時に設定済み
                if arena[ni].parent == u32::MAX {
                    return;
                }
                current = arena[ni].parent;
                continue;
            }

            // AND ノードが展開時の早期脱出で反証済み(dn=0)の場合，
            // 部分的な子しかアリーナにないため再計算をスキップする．
            // (再計算すると部分的な子だけから pn=0 と誤判定される)
            if !arena[ni].or_node && arena[ni].dn == 0 {
                if arena[ni].parent == u32::MAX {
                    return;
                }
                current = arena[ni].parent;
                continue;
            }

            let old_pn = arena[ni].pn;
            let old_dn = arena[ni].dn;

            let (new_pn, new_dn) = if arena[ni].or_node {
                // OR ノード: pn = min(child_pn), dn = sum(child_dn)
                let mut min_pn = INF;
                let mut sum_dn: u64 = 0;
                let num_children = arena[ni].children.len();
                for i in 0..num_children {
                    let ci = arena[ni].children[i] as usize;
                    if arena[ci].pn < min_pn {
                        min_pn = arena[ci].pn;
                    }
                    sum_dn = sum_dn.saturating_add(arena[ci].dn as u64);
                }
                (min_pn, sum_dn.min(INF as u64) as u32)
            } else {
                // AND ノード: WPN, dn = min(child_dn)
                let mut max_pn: u32 = 0;
                let mut min_dn = INF;
                let mut unproven: u32 = 0;
                let mut disproved = false;
                let num_children = arena[ni].children.len();
                for i in 0..num_children {
                    let ci = arena[ni].children[i] as usize;
                    if arena[ci].dn == 0 {
                        disproved = true;
                        break;
                    }
                    if arena[ci].pn == 0 {
                        // VPN: 証明済み子を pn 合計から除外
                        continue;
                    }
                    if arena[ci].pn > max_pn {
                        max_pn = arena[ci].pn;
                    }
                    if arena[ci].dn < min_dn {
                        min_dn = arena[ci].dn;
                    }
                    unproven += 1;
                }
                if disproved {
                    (INF, 0u32)
                } else if unproven == 0 && arena[ni].deferred_drops.is_empty() {
                    // 全子証明済み + deferred なし → AND ノード証明
                    (0u32, INF)
                } else if unproven == 0 {
                    // 全子証明済みだが deferred_drops 残り → 未完了
                    // MPN 選択時に次の合駒を活性化するため pn=1, dn=1 で保持
                    (PN_UNIT, PN_UNIT)
                } else {
                    let pn = (max_pn as u64)
                        .saturating_add((unproven as u64 - 1) * PN_UNIT as u64)
                        .min(INF as u64) as u32;
                    (pn, min_dn)
                }
            };

            arena[ni].pn = new_pn;
            arena[ni].dn = new_dn;

            // pn/dn が変化しなければ伝播打ち切り
            if new_pn == old_pn && new_dn == old_dn {
                return;
            }
            if arena[ni].parent == u32::MAX {
                return;
            }
            current = arena[ni].parent;
        }
    }

    /// PNS 証明ツリーの整合性を検証する(デバッグ用)．
    ///
    /// OR ノード(pn=0): 少なくとも1つの子が pn=0
    /// AND ノード(pn=0): 全ての子が pn=0(展開済みの場合)
    #[cfg(debug_assertions)]
    pub(super) fn validate_pns_proof(arena: &[PnsNode], idx: u32) {
        let node = &arena[idx as usize];
        if node.pn != 0 {
            return;
        }
        if !node.expanded || node.children.is_empty() {
            // リーフ: TT/ヒューリスティックから取得した pn=0
            verbose_eprintln!("  PNS leaf proven: idx={}, or={}, pk={:#x}, move={}",
                idx, node.or_node, node.pos_key,
                if idx == 0 { "root".to_string() } else { node.move_from_parent.to_usi() });
            return;
        }
        if node.or_node {
            // OR: 少なくとも1つの子が pn=0
            let has_proven = node.children.iter().any(|&c| arena[c as usize].pn == 0);
            assert!(has_proven,
                "PNS BUG: OR node {} (pk={:#x}) is proven but no child has pn=0. children: {:?}",
                idx, node.pos_key,
                node.children.iter().map(|&c| {
                    let ch = &arena[c as usize];
                    format!("{}(pn={},dn={},or={},exp={})",
                        ch.move_from_parent.to_usi(), ch.pn, ch.dn, ch.or_node, ch.expanded)
                }).collect::<Vec<_>>());
            // 証明された子を表示
            #[cfg(feature = "verbose")]
            {
                let proven_child = node.children.iter()
                    .find(|&&c| arena[c as usize].pn == 0).unwrap();
                eprintln!("  PNS OR proven: idx={}, pk={:#x}, best_child={} ({})",
                    idx, node.pos_key, proven_child,
                    arena[*proven_child as usize].move_from_parent.to_usi());
            }
            // 再帰: 証明された子のみ
            for &c in &node.children {
                if arena[c as usize].pn == 0 {
                    Self::validate_pns_proof(arena, c);
                }
            }
        } else {
            verbose_eprintln!("  PNS AND proven: idx={}, pk={:#x}, move={}, {} children",
                idx, node.pos_key, node.move_from_parent.to_usi(), node.children.len());
            // AND: 全子が pn=0
            for &c in &node.children {
                assert!(arena[c as usize].pn == 0,
                    "PNS BUG: AND node {} (pk={:#x}) is proven but child {} ({}) has pn={}",
                    idx, node.pos_key, c,
                    arena[c as usize].move_from_parent.to_usi(),
                    arena[c as usize].pn);
                Self::validate_pns_proof(arena, c);
            }
        }
    }

    /// PNS アリーナから直接 PV を抽出する．
    ///
    /// TT ベースの `extract_pv` と異なり，PNS が構築した明示的な探索木を
    /// 辿るため，PNS が証明した経路を正確に復元できる．
    /// 展開されていないリーフ(TT から pn=0 を取得した子)では
    /// TT ベースの `extract_pv_recursive` にフォールバックする．
    ///
    /// - OR ノード: 証明済み子のうち最短 PV を選択
    /// - AND ノード: 証明済み子のうち最長 PV を選択(最長抵抗)
    pub(super) fn pns_extract_pv(
        &mut self,
        board: &mut Board,
        arena: &[PnsNode],
        node_idx: u32,
        visited: &mut FxHashSet<u64>,
    ) -> Vec<Move> {
        let node = &arena[node_idx as usize];

        // 未証明ノード → PV なし
        if node.pn != 0 {
            return Vec::new();
        }

        // ループ検出
        if visited.contains(&node.full_hash) {
            return Vec::new();
        }

        // 未展開リーフまたは子なし(終端) → TT フォールバック
        if !node.expanded || node.children.is_empty() {
            return self.extract_pv_recursive(board, node.or_node, visited, 0);
        }

        if node.or_node {
            // OR ノード: 証明済み子から最短 PV を選択
            let mut best_pv: Option<Vec<Move>> = None;

            for &ci in &node.children {
                let child = &arena[ci as usize];
                if child.pn != 0 {
                    continue;
                }

                let captured = board.do_move(child.move_from_parent);
                visited.insert(node.full_hash);
                let sub_pv = self.pns_extract_pv(board, arena, ci, visited);
                visited.remove(&node.full_hash);
                board.undo_move(child.move_from_parent, captured);

                // sub_pv が空でないか，AND 終端(応手なし=詰み)なら有効
                let total_len = 1 + sub_pv.len();
                // 奇数長(攻め方の手で終わる)のみ有効な PV
                if total_len % 2 == 0 && !sub_pv.is_empty() {
                    continue;
                }
                let is_better = match &best_pv {
                    None => true,
                    Some(prev) => total_len < prev.len(),
                };
                if is_better {
                    let mut pv = vec![child.move_from_parent];
                    pv.extend(sub_pv);
                    best_pv = Some(pv);
                }
            }

            best_pv.unwrap_or_default()
        } else {
            // AND ノード: 全子が証明済み，最長 PV を選択(最長抵抗)
            let mut best_pv: Option<Vec<Move>> = None;
            let mut best_is_capture = false;

            for &ci in &node.children {
                let child = &arena[ci as usize];
                if child.pn != 0 {
                    continue;
                }

                let captured = board.do_move(child.move_from_parent);
                visited.insert(node.full_hash);
                let sub_pv = self.pns_extract_pv(board, arena, ci, visited);
                visited.remove(&node.full_hash);
                board.undo_move(child.move_from_parent, captured);

                let total_len = 1 + sub_pv.len();
                // AND ノードの PV は偶数長でなければならない
                if total_len % 2 == 1 {
                    continue;
                }
                let is_capture = child.move_from_parent.captured_piece_raw() > 0;
                let is_better = match &best_pv {
                    None => true,
                    Some(prev) => {
                        if total_len > prev.len() {
                            true
                        } else if total_len == prev.len()
                            && is_capture
                            && !best_is_capture
                        {
                            true
                        } else {
                            false
                        }
                    }
                };
                if is_better {
                    let mut pv = vec![child.move_from_parent];
                    pv.extend(sub_pv);
                    best_pv = Some(pv);
                    best_is_capture = is_capture;
                }
            }

            best_pv.unwrap_or_default()
        }
    }

    pub(super) fn pns_store_to_tt(&mut self, arena: &[PnsNode]) {
        for node in arena {
            if node.pn == 0 && node.expanded && !node.children.is_empty() {
                // 証明済み中間ノード
                if node.or_node {
                    // OR 証明: 証明子の手を TT Best Move に記録
                    let best_child = node.children.iter()
                        .find(|&&c| arena[c as usize].pn == 0);
                    if let Some(&ci) = best_child {
                        let best_move16 = arena[ci as usize].move_from_parent.to_move16();
                        self.store_with_best_move(
                            node.pos_key, node.hand, 0, INF,
                            REMAINING_INFINITE, node.pos_key, best_move16,
                        );
                    }
                } else {
                    // AND 証明: 全子が証明済み
                    self.store(
                        node.pos_key, node.hand, 0, INF,
                        REMAINING_INFINITE, node.pos_key,
                    );
                }
            } else if node.dn == 0 && node.expanded {
                // PNS の反証(NM)は TT にバックプロパゲーションしない．
                // 理由: PNS はアリーナサイズに制限された最良優先探索であり，
                // 探索木を完全には展開しない．そのため PNS の NM は
                // 「アリーナ内で反証された」という意味でしかなく，
                // MID(DFS) の NM のように「深さ R 以内で完全に反証された」
                // とは保証できない．PNS NM を remaining 付きで TT に格納すると，
                // 後続の mid_fallback が NM エントリをヒットして探索をスキップし，
                // 偽 NoCheckmate を引き起こす．
                // 展開フェーズ(expand_pns_node)で各ノードの NM は既に
                // TT に個別に記録済みなので，backprop での追加格納は不要．
            }
        }
    }
}


/// デフォルトタイムアウト(300秒)で詰将棋を解く便利関数．
///
/// [`solve_tsume_with_timeout`] のラッパーで，タイムアウト・PV ノード予算・
/// TT GC 閾値にはデフォルト値を使用する．
///
/// # 引数
///
/// - `sfen`: 局面の SFEN 文字列．
/// - `depth`: 最大探索手数(`None` でデフォルト 31)．
/// - `nodes`: 最大ノード数(`None` でデフォルト 1,048,576)．
/// - `draw_ply`: 引き分け手数(`None` でデフォルト 32767)．
///
/// # 戻り値
///
/// [`TsumeResult`] を返す．SFEN パースエラー時は `Err` を返す．
pub fn solve_tsume(
    sfen: &str,
    depth: Option<u32>,
    nodes: Option<u64>,
    draw_ply: Option<u32>,
) -> Result<TsumeResult, crate::board::SfenError> {
    solve_tsume_with_timeout(sfen, depth, nodes, draw_ply, None, None, None, None)
}

/// タイムアウト指定付きで詰将棋を解く便利関数．
///
/// # 戻り値
///
/// 詰みが証明された場合でも，PV 復元フェーズ(`complete_pv_or_nodes`)の
/// ノード予算が不足すると [`TsumeResult::CheckmateNoPv`] が返ることがある．
/// 特に長手数(17手以上)の詰将棋では，PV 沿いの各未証明子に対する
/// 追加証明の1子あたり予算(デフォルト 1024 ノード)が不足しやすい．
/// `pv_nodes_per_child` を増やすことで改善できる．
///
/// # 引数
///
/// - `find_shortest`: 最短手数探索を行うか(None でデフォルト true)．
///   false にすると `complete_or_proofs()` による追加探索をスキップし，
///   最初に見つかった詰み手順をそのまま返す．ノード数は削減されるが，
///   返される手順が最短とは限らない．
/// - `pv_nodes_per_child`: PV 復元時の1子あたりノード予算(None でデフォルト 1024)．
///   長手数の詰将棋で `CheckmateNoPv` が返る場合に増やすと効果的．
pub fn solve_tsume_with_timeout(
    sfen: &str,
    depth: Option<u32>,
    nodes: Option<u64>,
    draw_ply: Option<u32>,
    timeout_secs: Option<u64>,
    find_shortest: Option<bool>,
    pv_nodes_per_child: Option<u64>,
    tt_gc_threshold: Option<usize>,
) -> Result<TsumeResult, crate::board::SfenError> {
    let mut board = Board::empty();
    board.set_sfen(sfen)?;

    let mut solver = DfPnSolver::with_timeout(
        depth.unwrap_or(31),
        nodes.unwrap_or(1_048_576),
        draw_ply.unwrap_or(32767),
        timeout_secs.unwrap_or(300),
    );
    solver.set_find_shortest(find_shortest.unwrap_or(true));
    if let Some(budget) = pv_nodes_per_child {
        solver.set_pv_nodes_per_child(budget);
    }
    if let Some(gc) = tt_gc_threshold {
        solver.set_tt_gc_threshold(gc);
    }

    Ok(solver.solve(&mut board))
}

