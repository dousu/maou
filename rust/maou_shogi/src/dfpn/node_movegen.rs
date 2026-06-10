//! dfpn OR/AND ノードの手生成・合法手ヘルパ群．
//!
//! mid_v3（現行エンジン）と legacy PNS 探索（pns.rs, 廃止予定）の両方が依存する
//! **恒久的** な movegen ヘルパ．pns.rs から切り出した（v2.0.x, mid/pns 廃止方針）．
//! ここには探索ロジックを置かない（手生成・合法性判定のみ）．


use arrayvec::ArrayVec;
#[cfg(feature = "profile")]
use std::time::Instant;

use crate::attack;
use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::movegen;
use crate::moves::Move;
use crate::types::{Color, PieceType, Square};

use super::solver::DfPnSolver;
use super::{
    push_move, MAX_MOVES,
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
            // Movegen 最適化: do_move/undo_move を省略し，is_attacked_by_excluding で
            // 玉の元マスを占有から除外して直接利き判定する．
            let safe = !board.is_attacked_by_excluding(to, attacker, false, Some(king_sq));
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
                // 通常マス: 全駒種を生成する．
                // KH/yaneuraou parity (V3_KHPAR): 生成順は 歩→桂→香→銀→金→角→飛
                // (yaneuraou movegen.cpp GenerateDropMoves: PAWN 先行 + drops[] が
                // KNIGHT,LANCE,SILVER,GOLD,BISHOP,ROOK 順)．同 to_sq drop は DML で
                // chain 化され先頭が chain head になるため，この順が KH の chain head
                // 選択 (歩可能なら歩，二歩なら桂) を再現する (39te 実測: N*6b)．
                // default は旧来の弱い駒から順 (歩→香→桂→…)．
                // 生成集合は不変で順序のみ = 合法性に影響なし．
                const DROP_ORDER_KH: [(usize, PieceType); 7] = [
                    (0, PieceType::Pawn),
                    (2, PieceType::Knight),
                    (1, PieceType::Lance),
                    (3, PieceType::Silver),
                    (4, PieceType::Gold),
                    (5, PieceType::Bishop),
                    (6, PieceType::Rook),
                ];
                const DROP_ORDER_LEGACY: [(usize, PieceType); 7] = [
                    (0, PieceType::Pawn),
                    (1, PieceType::Lance),
                    (2, PieceType::Knight),
                    (3, PieceType::Silver),
                    (4, PieceType::Gold),
                    (5, PieceType::Bishop),
                    (6, PieceType::Rook),
                ];
                let drop_order: &[(usize, PieceType); 7] = if super::kh_parity_order() {
                    &DROP_ORDER_KH
                } else {
                    &DROP_ORDER_LEGACY
                };
                for &(hand_idx, pt) in drop_order.iter() {
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
        // 3 カテゴリの代表を一旦集めてから push する (DML chain head = 最初に push された手なので，
        // KH parity 実験 V3_KH_INT 時に並べ替えできるように)．
        let mut cat1: Option<Move> = None;
        let mut cat1_is_pawn = false;
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
                cat1 = Some(m);
                cat1_is_pawn = true;
            } else {
                cat1 = Some(Move::new_drop(to, pt));
            }
            break; // カテゴリ内最弱で代表
        }

        // カテゴリ2: 角
        let bishop_idx = 5; // HAND_PIECES[5] = Bishop
        let cat2: Option<Move> = if board.hand[di][bishop_idx] > 0
            && !movegen::must_promote(defender, PieceType::Bishop, to)
        {
            Some(Move::new_drop(to, PieceType::Bishop))
        } else {
            None
        };

        // カテゴリ3: 桂
        let knight_idx = 2; // HAND_PIECES[2] = Knight
        let cat3: Option<Move> = if board.hand[di][knight_idx] > 0
            && !movegen::must_promote(defender, PieceType::Knight, to)
        {
            Some(Move::new_drop(to, PieceType::Knight))
        } else {
            None
        };

        // KH parity: 歩が打てない (二歩等) chain マスでは，DML chain head を桂にする．
        // KH は同状況で N*6c を代表に選び (maou 既定は cat1 の香 L*6c)，これに揃えると 39te の
        // deep frontier が 4-6% 縮む (p9 -5.2% / p11 -6.0% / p13 -4.2%; 29te は 18,539 で不変)．
        // 健全性: DML chain は全合い駒を保持し head 順のみ変える (defer された手は head 確定後に
        // activate される) ため集合は不変＝合法性に影響なし．[[guidance pivot: 中合い代表 lever]]
        if !cat1_is_pawn && cat3.is_some() {
            // 桂を先頭に: knight, cat1(香等), bishop
            push_move(moves, cat3.unwrap());
            if let Some(m) = cat1 {
                push_move(moves, m);
            }
            if let Some(m) = cat2 {
                push_move(moves, m);
            }
        } else {
            // 既定順: cat1, bishop, knight
            if let Some(m) = cat1 {
                push_move(moves, m);
            }
            if let Some(m) = cat2 {
                push_move(moves, m);
            }
            if let Some(m) = cat3 {
                push_move(moves, m);
            }
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
        // 逆王手 (counter-check): 攻め方自身が王手されている場合，駒打ちでも自玉の
        // 王手を解消しないものは非合法 (打った駒が王手を遮らない限り王手放置になる)．
        // 自玉が王手されていなければ駒打ちは決して自玉を王手に晒さない (駒を動かさない
        // ため開き王手も起きない) ので従来通り検証を省く．
        let own_in_check = has_own_king && board.is_in_check(us);

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

        // --- 1. 盤上の駒の移動 (KH/yaneuraou parity: 盤上移動を駒打ちより先に生成) ---
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

        // --- 2. 駒打ち: ターゲットマスのみに打つ ---
        // yaneuraou generate_checks の駒打ち順 = P,L,N,S,G,B,R (movegen.cpp:853-866)．
        let mut drops = ArrayVec::<Move, MAX_MOVES>::new();
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
                // 通常は駒打ちで自玉への王手放置は起きないが，逆王手中は打った駒が
                // 王手を遮らなければ非合法 (`own_in_check` のときだけ検証する)．
                if own_in_check && !self.is_legal_quick(board, m, has_own_king) {
                    continue;
                }
                push_move(&mut drops, m);
            }
        }

        if super::kh_parity_order() {
            // KH/yaneuraou parity (V3_KHPAR): 生成順をそのまま返す
            // (盤上移動 from マス昇順 → 駒打ち P,L,N,S,G,B,R)．独自 sort なし —
            // LE 構築側の comparer が pn/dn → δ → move_brief_eval で並べ，
            // 完全同点の安定順がこの生成順 = KH MovePicker (yaneuraou
            // generate_checks) の tie order と一致する．
            for m in drops {
                push_move(&mut moves, m);
            }
            moves
        } else {
            // 旧挙動 (default): 駒打ち → 盤上移動 の生成順に，
            // 成+取 > 成 > 取 > その他のカテゴリ + 玉とのチェビシェフ距離の
            // 独自 sort を適用する．近接王手を優先し，詰みに至る手を早期発見する．
            // カテゴリ(0-3) * 16 + 距離 で単一キーにエンコードする．
            let mut out = ArrayVec::<Move, MAX_MOVES>::new();
            for m in drops {
                push_move(&mut out, m);
            }
            for m in moves {
                push_move(&mut out, m);
            }
            let king_col = king_sq.col() as i8;
            let king_row = king_sq.row() as i8;
            out.sort_unstable_by_key(|m| {
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
            out
        }
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
}
