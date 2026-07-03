//! dfpn OR/AND ノードの手生成・合法手ヘルパ群．
//!
//! 探索ロジックから独立した movegen ヘルパ．
//! ここには探索ロジックを置かない (手生成・合法性判定のみ)．

pub(crate) mod check_cache;
pub(crate) mod delayed_move_list;
pub(crate) mod mate1ply;

use arrayvec::ArrayVec;

use crate::attack;
use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::movegen;
use crate::moves::Move;
use crate::types::{Color, PieceType, Square};

use super::solver::DfPnSolver;
use super::{push_move, MAX_MOVES};

thread_local! {
    /// is_legal_quick が実際に do_move した回数 (do_moves breakdown 用; solve 毎に reset)．
    static LEGAL_QUICK_DM: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}
/// is_legal_quick の do_move 回数を取得 (report 用)．
pub(super) fn legal_quick_dm() -> u64 {
    LEGAL_QUICK_DM.with(|c| c.get())
}
/// is_legal_quick の do_move カウンタを reset (solve 開始時)．
pub(super) fn reset_legal_quick_dm() {
    LEGAL_QUICK_DM.with(|c| c.set(0));
}

impl DfPnSolver {
    /// 玉方の王手回避手を生成する(合い効かずを除外)．
    ///
    /// 全合法手生成の代わりに回避手のみを直接生成する:
    /// 1. 玉の移動(攻め方に利かれていないマスへ)
    /// 2. 王手駒の捕獲(ピンされていない駒による)
    /// 3. 合い駒(飛び駒の王手の場合，間のマスへ移動または打つ)
    ///
    /// 合い効かず(futile interposition)もフィルタする．
    ///
    /// `early_exit == true` の場合，最初の合法手が見つかった時点で即座にリターンする．
    /// `has_any_defense` もこの関数を共有し，ロジックの重複を排除する．
    pub(super) fn generate_defense_moves_inner(
        &mut self,
        board: &mut Board,
        early_exit: bool,
    ) -> ArrayVec<Move, MAX_MOVES> {
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

        // 受け方の pin (合法性判定用)．単一王手の回避手 (捕獲 to=checker / 合駒
        // to∈between) は王手自体を必ず解消するので，違法になり得るのは from 退去の
        // 開き王手 = 「from が pin されていて to が pin ray 外」の場合だけ．
        // per-move の do/undo + is_in_check (is_evasion_legal) と完全同値:
        //  - pin ray は king-from を通る唯一の直線で，第二 pinner は幾何的に存在しない
        //    (同一線上の 2 つ目の slider は blocker 2 枚で pinner にならない)．
        //  - 王手駒が pin ray 上にある場合は blocker 2 枚 (from + 王手駒) で from は
        //    pin されない (王手駒も between & all_occ の blocker に数えられる)．
        let pinned = board.compute_pinned(defender, king_sq);

        // --- 2. 王手駒の捕獲(玉以外の駒で) ---
        self.generate_capture_checker(
            board, &mut moves, checker_sq, king_sq, defender, all_occ, our_occ, pinned,
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
                // 間のマスへの合い駒
                self.generate_interpositions(
                    board, &mut moves, &between, &futile, &chain, king_sq, defender, all_occ,
                    our_occ, pinned,
                );
            }
        }

        moves
    }

    /// 王手駒を玉以外の駒で捕獲する手を生成する．
    #[allow(clippy::too_many_arguments)]
    pub(super) fn generate_capture_checker(
        &self,
        board: &mut Board,
        moves: &mut ArrayVec<Move, MAX_MOVES>,
        checker_sq: Square,
        king_sq: Square,
        defender: Color,
        all_occ: Bitboard,
        _our_occ: Bitboard,
        pinned: Bitboard,
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
            // 捕獲は王手を必ず解消するので，合法性は pin 述語だけで決まる
            // (is_evasion_legal の do/undo + is_in_check と同値; 呼び出し元コメント参照)．
            if pinned.contains(from) && !attack::line_through(king_sq, from).contains(checker_sq) {
                continue;
            }
            let piece = board.squares[from.index()];
            let pt = piece.piece_type().unwrap();
            let in_promo_zone =
                checker_sq.is_promotion_zone(defender) || from.is_promotion_zone(defender);

            if pt.can_promote() && in_promo_zone {
                let m = Move::new_move(from, checker_sq, true, captured_raw, pt as u8);
                push_move(moves, m);
                if !movegen::must_promote(defender, pt, checker_sq) {
                    let m = Move::new_move(from, checker_sq, false, captured_raw, pt as u8);
                    push_move(moves, m);
                }
            } else if !movegen::must_promote(defender, pt, checker_sq) {
                let m = Move::new_move(from, checker_sq, false, captured_raw, pt as u8);
                push_move(moves, m);
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
        let king_step = attack::step_attacks(defender.opponent(), PieceType::King, king_sq);
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

    /// find_shortest の len 予算を無駄合い-free 化する: AND node (受け方手番・飛び駒王手中) で
    /// **透過的に取り返される中合いマス (chain)** を返す．これらへの合駒 drop は無駄合いゆえ len を
    /// 減じない (child len = `len.add(1)` → 攻め方の取り返し `len.sub(1)` と相殺し pair cost 0)．
    /// 非 AND / 非王手 / 非飛び駒王手 / 間マス無しでは空 bitboard (= 全 drop が通常 decrement)．
    /// `futile` マスは generate_interpositions で drop 自体が skip されるため対象外 (chain のみ返す)．
    pub(super) fn transparent_interposition_squares(&self, board: &Board) -> Bitboard {
        let attacker = self.attacker;
        let defender = attacker.opponent();
        if board.turn != defender {
            return Bitboard::EMPTY; // OR node (攻め方手番) では credit しない
        }
        let king_sq = match board.king_square(defender) {
            Some(k) => k,
            None => return Bitboard::EMPTY,
        };
        let checker_sq = match self.find_sliding_checker(board, king_sq, attacker) {
            Some(c) => c,
            None => return Bitboard::EMPTY,
        };
        let between = attack::between_bb(checker_sq, king_sq);
        if !between.is_not_empty() {
            return Bitboard::EMPTY;
        }
        let (_futile, chain) = self.compute_futile_and_chain_squares(
            board, &between, king_sq, checker_sq, defender, attacker,
        );
        chain
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
    #[allow(clippy::too_many_arguments)]
    pub(super) fn generate_interpositions(
        &self,
        board: &mut Board,
        moves: &mut ArrayVec<Move, MAX_MOVES>,
        between: &Bitboard,
        futile: &Bitboard,
        _chain: &Bitboard,
        king_sq: Square,
        defender: Color,
        all_occ: Bitboard,
        _our_occ: Bitboard,
        pinned: Bitboard,
    ) {
        let attacker = defender.opponent();
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
                // 合駒 (to∈between) は王手を必ず遮断するので，合法性は pin 述語だけで
                // 決まる (is_evasion_legal の do/undo + is_in_check と同値;
                // generate_defense_moves_inner のコメント参照)．
                if pinned.contains(from) && !attack::line_through(king_sq, from).contains(to) {
                    continue;
                }
                let piece = board.squares[from.index()];
                let pt = piece.piece_type().unwrap();

                // 駒移動による合い駒は無駄合いフィルタを適用せず全て生成する
                // (無駄合いの除外は駒打ちの futile/chain 分類でのみ行う)．
                let captured_raw = board.squares[to.index()].0;
                let in_promo_zone =
                    to.is_promotion_zone(defender) || from.is_promotion_zone(defender);

                if pt.can_promote() && in_promo_zone {
                    let m = Move::new_move(from, to, true, captured_raw, pt as u8);
                    push_move(moves, m);
                    if !movegen::must_promote(defender, pt, to) {
                        let m = Move::new_move(from, to, false, captured_raw, pt as u8);
                        push_move(moves, m);
                    }
                } else if !movegen::must_promote(defender, pt, to) {
                    let m = Move::new_move(from, to, false, captured_raw, pt as u8);
                    push_move(moves, m);
                }
            }

            // --- 駒打ちによる合い駒 ---
            // 無駄合いマス (futile: 守備の支えなし & 飛び駒取り進み後に玉の逃げ道なし &
            // breakpoint より玉側) への駒打ちは **無駄合い** ゆえスキップする (詰将棋規約上
            // 無駄合いは手数に数えない; 攻め方は只取りして詰みが変わらない)．駒移動による合駒は
            // 盤上駒の relocation = 無駄合い対象外なので上で生成済 (futile でも消さない)．
            if !futile.contains(to) {
                // 間のマスへ全駒種の合駒 drop を生成する．
                // 生成順は 歩→香→桂→銀→金→角→飛 (弱い駒から順)．
                // 同 to_sq の複数 drop は後段で連結され先頭が代表になるため，
                // この順序は代表駒の選択 (歩可能なら歩，二歩なら次の駒) を決める．
                // 生成集合は不変で順序のみ = 合法性に影響なし．
                const DROP_ORDER_LEGACY: [(usize, PieceType); 7] = [
                    (0, PieceType::Pawn),
                    (1, PieceType::Lance),
                    (2, PieceType::Knight),
                    (3, PieceType::Silver),
                    (4, PieceType::Gold),
                    (5, PieceType::Bishop),
                    (6, PieceType::Rook),
                ];
                let drop_order: &[(usize, PieceType); 7] = &DROP_ORDER_LEGACY;
                for &(hand_idx, pt) in drop_order.iter() {
                    if board.hand[defender.index()][hand_idx] == 0 {
                        continue;
                    }
                    if movegen::must_promote(defender, pt, to) {
                        continue;
                    }
                    if pt == PieceType::Pawn {
                        let our_pawns = board.piece_bb[defender.index()][PieceType::Pawn as usize];
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
    pub(super) fn generate_check_moves(&self, board: &mut Board) -> ArrayVec<Move, MAX_MOVES> {
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

        // 自玉露出の fast path 用: 王手中でなく，動かす駒が pin されておらず
        // 玉自身でもない盤上移動は自玉を王手に晒し得ない (取りでも取られた駒のマスを
        // 自駒が塞ぐため開き露出は起きない) → per-move の do/undo 検証を省略できる．
        let own_king_sq = board.king_square(us);
        let pinned_own = match own_king_sq {
            Some(k) => board.compute_pinned(us, k),
            None => Bitboard::EMPTY,
        };

        // 各駒種について「このマスに置くと玉に王手がかかる」ターゲットを事前計算
        // step_attacks(them, pt, king_sq) は「玉から見た逆利き」= 王手元になれるマス

        let mut moves = ArrayVec::<Move, MAX_MOVES>::new();

        // --- 1. 盤上の駒の移動 (盤上移動を駒打ちより先に生成) ---
        // 直接王手: 移動先から玉に利きがある手
        // 開き王手: 駒が移動することで背後のスライド駒から玉に利きが通る手

        // 開き王手の候補を事前計算:
        // 玉からのレイ上にいる自駒で，その間に他の駒がない場合，
        // そこから移動すると開き王手になりうる
        let discoverers = board.compute_discoverers(us, king_sq);

        // 逆利き (王手元マス) bitboard の駒種別 lazy cache．
        // rev_check(pt) = {to | piece_attacks(us, pt, to, all_occ).contains(king_sq)}
        // stepper は白黒の利きが点対称なので them 側 step，slider は対称性で king 起点
        // (駒打ち生成と同一の逆利き idiom)．per-target で利きテーブルを参照する
        // 素朴実装よりマス毎の利き計算を削減する．
        let mut rev_cache: [Option<Bitboard>; 15] = [None; 15];
        let mut rev_check = |pt: PieceType| -> Bitboard {
            let i = pt as usize;
            if let Some(bb) = rev_cache[i] {
                return bb;
            }
            let bb = match pt {
                PieceType::Lance => attack::lance_attacks(them, king_sq, all_occ),
                PieceType::Bishop => attack::bishop_attacks(king_sq, all_occ),
                PieceType::Rook => attack::rook_attacks(king_sq, all_occ),
                PieceType::Horse => attack::horse_attacks(them, king_sq, all_occ),
                PieceType::Dragon => attack::dragon_attacks(them, king_sq, all_occ),
                _ => attack::step_attacks(them, pt, king_sq),
            };
            rev_cache[i] = Some(bb);
            bb
        };

        let mut our_bb = our_occ;
        while our_bb.is_not_empty() {
            let from = our_bb.pop_lsb();
            let piece = board.squares[from.index()];
            let pt = piece.piece_type().unwrap();
            let attacks = attack::piece_attacks(us, pt, from, all_occ);
            let targets = attacks & !our_occ;

            let is_discoverer = discoverers.contains(from);

            // 直接王手になり得る to の mask (成り可能駒は成駒分も union)．
            // 非 discoverer は mask 外の to が王手にならないため事前に枝刈りする．
            // bitboard 昇順の subset 走査なので生成順・生成列は従来と完全一致．
            let scan = if is_discoverer {
                targets
            } else {
                let mut mask = rev_check(pt);
                if pt.can_promote() {
                    mask |= rev_check(pt.promoted().unwrap());
                }
                targets & mask
            };

            // この駒の移動が自玉を露出させ得ないなら per-move の do/undo 検証
            // (is_legal_quick) を省略する (結果は同値; 上記コメント参照)．
            let fast_legal = !has_own_king
                || (!own_in_check && Some(from) != own_king_sq && !pinned_own.contains(from));

            for to in scan {
                let captured_piece = board.squares[to.index()];
                let captured_raw = captured_piece.0;
                let in_promo_zone = to.is_promotion_zone(us) || from.is_promotion_zone(us);

                // 開き王手の判定: to が from→king_sq のライン上にある場合，
                // 移動後も飛び駒の利きを遮断するため開き王手にならない
                let gives_discovered =
                    is_discoverer && !attack::line_through(from, king_sq).contains(to);

                // 成り先の駒種での王手チェック (逆利き mask の O(1) contains;
                // attacks_square と同値)
                if pt.can_promote() && in_promo_zone {
                    let promoted_pt = pt.promoted().unwrap();
                    let gives_direct = rev_check(promoted_pt).contains(to);
                    if gives_direct || gives_discovered {
                        let m = Move::new_move(from, to, true, captured_raw, pt as u8);
                        if fast_legal || self.is_legal_quick(board, m, has_own_king) {
                            push_move(&mut moves, m);
                        }
                    }

                    // 不成
                    if !movegen::must_promote(us, pt, to) {
                        let gives_direct = rev_check(pt).contains(to);
                        if gives_direct || gives_discovered {
                            let m = Move::new_move(from, to, false, captured_raw, pt as u8);
                            if fast_legal || self.is_legal_quick(board, m, has_own_king) {
                                push_move(&mut moves, m);
                            }
                        }
                    }
                } else if !movegen::must_promote(us, pt, to) {
                    let gives_direct = rev_check(pt).contains(to);
                    if gives_direct || gives_discovered {
                        let m = Move::new_move(from, to, false, captured_raw, pt as u8);
                        if fast_legal || self.is_legal_quick(board, m, has_own_king) {
                            push_move(&mut moves, m);
                        }
                    }
                }
            }
        }

        // --- 2. 駒打ち: ターゲットマスのみに打つ ---
        // 駒打ち順 = P,L,N,S,G,B,R．
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

        // 駒打ち → 盤上移動 の生成順に，
        // 成+取 > 成 > 取 > その他のカテゴリ + 玉とのチェビシェフ距離の
        // sort を適用する．近接王手を優先し，詰みに至る手を早期発見する．
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

    /// 1 手詰判定の **候補列挙部**．敵玉隣接 geometry から 1 手詰候補 (王手) を
    /// 駒種順 (打ち = 飛/香/角/金/銀/桂, 移動 = 龍/飛/馬/角/香 → 金/銀/桂/歩) で
    /// 直接構成する (full movegen 不要)．
    ///
    /// 列挙候補は [`Board::mate_move_in_1ply_maxdist`] (per-candidate mate 判定 + 開き
    /// 王手の do_move fallback) でスキャンする ⇒ 返り値 = この順序での最初の 1 手詰．
    /// 本列挙は遠方駒打ち詰み・両王手詰みを生成しない (隣接 geometry のみを対象とする)．
    ///
    /// **前提**: 攻め方が非王手 (`!checkers`)．逆王手局面 (攻め方自玉が王手) は呼び出し側
    /// ([`super::solver::DfPnSolver::mate1ply`]) が full scan へ fallback する．
    /// `board` は読むだけ (do_move しない)．
    pub(super) fn generate_mate_candidates(&self, board: &Board) -> ArrayVec<Move, MAX_MOVES> {
        let mut out = ArrayVec::<Move, MAX_MOVES>::new();
        self.for_each_mate_candidate(board, |m| {
            super::push_move(&mut out, m);
            std::ops::ControlFlow::Continue(())
        });
        out
    }

    /// 敵玉隣接 geometry から 1 手詰候補手を駒種順で構成し，各候補を `emit` へ渡す (zero-collect)．
    ///
    /// `emit` が [`std::ops::ControlFlow::Break(m)`] を返した時点で `Some(m)` を返して列挙を
    /// 打ち切る (look-ahead fused 経路の「最初の詰みで短絡」用)．最後まで列挙すれば `None`．
    /// 候補の構成内容・**順序**は従来の [`Self::generate_mate_candidates`] と完全に同一
    /// (= 返す詰み手が一致 → node 不変)．`generate_mate_candidates` はこの上の薄い収集ラッパ．
    pub(super) fn for_each_mate_candidate<F>(&self, board: &Board, mut emit: F) -> Option<Move>
    where
        F: FnMut(Move) -> std::ops::ControlFlow<Move>,
    {
        use std::ops::ControlFlow;
        let us = board.turn;
        let them = us.opponent();
        let king = board.king_square(them)?;
        let usi = us.index();
        let all_occ = board.all_occupied();
        let empty = !all_occ;
        let our_occ = board.occupied[usi];
        let bb_move = !our_occ;

        // 攻め方自玉の pin (盤上移動が自玉の王手放置にならないよう除外する; 駒打ちは非王手
        // 前提ゆえ自玉を晒さず常に合法)．
        let our_king = board.king_square(us);
        let our_pinned = match our_king {
            Some(k) => board.compute_pinned(us, k),
            None => Bitboard::EMPTY,
        };

        // 敵玉の逆利き = 「その駒を置く / 動かすと敵玉に王手になる」マス集合．
        let k_neighbors = attack::step_attacks(us, PieceType::King, king);
        let gold_chk = attack::step_attacks(them, PieceType::Gold, king);
        let silver_chk = attack::step_attacks(them, PieceType::Silver, king);
        let knight_chk = attack::step_attacks(them, PieceType::Knight, king);
        let pawn_chk = attack::step_attacks(them, PieceType::Pawn, king);

        let hand = board.hand[usi];
        let in_hand = |pt: PieceType| hand[pt.hand_index().unwrap()] > 0;
        let can_promo =
            |from: Square, to: Square| to.is_promotion_zone(us) || from.is_promotion_zone(us);

        // 盤上移動候補を emit (自玉 pin 違反 = 自玉王手放置は除外; 駒種・取り駒は board から読む)．
        // `emit` は &mut で受け取り (closure を捕獲しない) ので直接 emit と排他しない．
        let push_bm =
            |from: Square, to: Square, promote: bool, emit: &mut F| -> ControlFlow<Move> {
                if let Some(ok) = our_king {
                    if our_pinned.contains(from) && !attack::line_through(from, ok).contains(to) {
                        return ControlFlow::Continue(());
                    }
                }
                let captured_raw = board.squares[to.index()].0;
                let raw_pt = board.squares[from.index()].piece_type().unwrap() as u8;
                emit(Move::new_move(from, to, promote, captured_raw, raw_pt))
            };

        // 行き所のない駒の禁止段 (駒打ち)．
        let last_rank = match us {
            Color::Black => Bitboard::rank_mask(0),
            Color::White => Bitboard::rank_mask(8),
        };
        let knight_forbidden = match us {
            Color::Black => Bitboard::rank_mask(0) | Bitboard::rank_mask(1),
            Color::White => Bitboard::rank_mask(7) | Bitboard::rank_mask(8),
        };

        // ===== 1. 駒打ち (順: 飛 → 香 → 角 → 金 → 銀 → 桂; 歩打ち = 打ち歩詰め反則ゆえ除外) =====
        // 飛打ち: 敵玉の上下左右 (rook step ∩ 8 近傍) の空マス．
        if in_hand(PieceType::Rook) {
            for to in attack::rook_attacks(king, Bitboard::EMPTY) & k_neighbors & empty {
                if let ControlFlow::Break(m) = emit(Move::new_drop(to, PieceType::Rook)) {
                    return Some(m);
                }
            }
        }
        // 香打ち: 敵玉の直下 (歩の逆利き) の空マス (最終段除く)．
        if in_hand(PieceType::Lance) {
            for to in pawn_chk & empty & !last_rank {
                if let ControlFlow::Break(m) = emit(Move::new_drop(to, PieceType::Lance)) {
                    return Some(m);
                }
            }
        }
        // 角打ち: 敵玉の斜め 4 近傍の空マス．
        if in_hand(PieceType::Bishop) {
            for to in attack::bishop_attacks(king, Bitboard::EMPTY) & k_neighbors & empty {
                if let ControlFlow::Break(m) = emit(Move::new_drop(to, PieceType::Bishop)) {
                    return Some(m);
                }
            }
        }
        // 金打ち: 敵玉の金の逆利きの空マス．
        if in_hand(PieceType::Gold) {
            for to in gold_chk & empty {
                if let ControlFlow::Break(m) = emit(Move::new_drop(to, PieceType::Gold)) {
                    return Some(m);
                }
            }
        }
        // 銀打ち: 敵玉の銀の逆利きの空マス．
        if in_hand(PieceType::Silver) {
            for to in silver_chk & empty {
                if let ControlFlow::Break(m) = emit(Move::new_drop(to, PieceType::Silver)) {
                    return Some(m);
                }
            }
        }
        // 桂打ち: 敵玉の桂の逆利きの空マス (最終 2 段除く)．
        if in_hand(PieceType::Knight) {
            for to in knight_chk & empty & !knight_forbidden {
                if let ControlFlow::Break(m) = emit(Move::new_drop(to, PieceType::Knight)) {
                    return Some(m);
                }
            }
        }

        // ===== 2. 移動による王手 (順: 龍 → 飛 → 馬 → 角 → 香 → 金 → 銀 → 桂 → 歩) =====
        // 龍: 敵玉 8 近傍への移動 (成り無し)．
        for from in board.piece_bb[usi][PieceType::Dragon as usize] {
            for to in attack::dragon_attacks(us, from, all_occ) & bb_move & k_neighbors {
                if let ControlFlow::Break(m) = push_bm(from, to, false, &mut emit) {
                    return Some(m);
                }
            }
        }
        // 飛: 敵玉 8 近傍への移動 (敵陣絡みは成り)．
        for from in board.piece_bb[usi][PieceType::Rook as usize] {
            for to in attack::rook_attacks(from, all_occ) & bb_move & k_neighbors {
                if let ControlFlow::Break(m) = push_bm(from, to, can_promo(from, to), &mut emit) {
                    return Some(m);
                }
            }
        }
        // 馬: 敵玉 8 近傍への移動 (成り無し)．
        for from in board.piece_bb[usi][PieceType::Horse as usize] {
            for to in attack::horse_attacks(us, from, all_occ) & bb_move & k_neighbors {
                if let ControlFlow::Break(m) = push_bm(from, to, false, &mut emit) {
                    return Some(m);
                }
            }
        }
        // 角: 敵玉 8 近傍への移動 (敵陣絡みは成り)．
        for from in board.piece_bb[usi][PieceType::Bishop as usize] {
            for to in attack::bishop_attacks(from, all_occ) & bb_move & k_neighbors {
                if let ControlFlow::Break(m) = push_bm(from, to, can_promo(from, to), &mut emit) {
                    return Some(m);
                }
            }
        }
        // 香: 玉の金の逆利き範囲への移動 (成り = 金を優先, 次に不成 = 串刺し)．
        for from in board.piece_bb[usi][PieceType::Lance as usize] {
            for to in attack::lance_attacks(us, from, all_occ) & bb_move & gold_chk {
                if can_promo(from, to) {
                    if let ControlFlow::Break(m) = push_bm(from, to, true, &mut emit) {
                        return Some(m);
                    }
                }
                if !movegen::must_promote(us, PieceType::Lance, to) {
                    if let ControlFlow::Break(m) = push_bm(from, to, false, &mut emit) {
                        return Some(m);
                    }
                }
            }
        }
        // 金 (成金含む): 玉の金の逆利き範囲への移動 (成り無し)．
        let golds = board.piece_bb[usi][PieceType::Gold as usize]
            | board.piece_bb[usi][PieceType::ProPawn as usize]
            | board.piece_bb[usi][PieceType::ProLance as usize]
            | board.piece_bb[usi][PieceType::ProKnight as usize]
            | board.piece_bb[usi][PieceType::ProSilver as usize];
        for from in golds {
            for to in attack::step_attacks(us, PieceType::Gold, from) & bb_move & gold_chk {
                if let ControlFlow::Break(m) = push_bm(from, to, false, &mut emit) {
                    return Some(m);
                }
            }
        }
        // 銀: 玉 8 近傍への移動 (各 to で不成を優先, 次に成 = 金)．
        for from in board.piece_bb[usi][PieceType::Silver as usize] {
            for to in attack::step_attacks(us, PieceType::Silver, from) & bb_move & k_neighbors {
                if silver_chk.contains(to) {
                    if let ControlFlow::Break(m) = push_bm(from, to, false, &mut emit) {
                        return Some(m);
                    }
                }
                if gold_chk.contains(to) && can_promo(from, to) {
                    if let ControlFlow::Break(m) = push_bm(from, to, true, &mut emit) {
                        return Some(m);
                    }
                }
            }
        }
        // 桂: 桂の利き先への移動 (各 to で不成を優先, 次に成 = 金)．
        for from in board.piece_bb[usi][PieceType::Knight as usize] {
            for to in attack::step_attacks(us, PieceType::Knight, from) & bb_move {
                if knight_chk.contains(to) {
                    if let ControlFlow::Break(m) = push_bm(from, to, false, &mut emit) {
                        return Some(m);
                    }
                }
                if gold_chk.contains(to) && can_promo(from, to) {
                    if let ControlFlow::Break(m) = push_bm(from, to, true, &mut emit) {
                        return Some(m);
                    }
                }
            }
        }
        // 歩 不成 (玉直下への前進, 敵陣でない)．
        for from in board.piece_bb[usi][PieceType::Pawn as usize] {
            for to in attack::step_attacks(us, PieceType::Pawn, from) & bb_move & pawn_chk {
                if !to.is_promotion_zone(us) {
                    if let ControlFlow::Break(m) = push_bm(from, to, false, &mut emit) {
                        return Some(m);
                    }
                }
            }
        }
        // 歩 成り (と金で王手)．
        for from in board.piece_bb[usi][PieceType::Pawn as usize] {
            for to in attack::step_attacks(us, PieceType::Pawn, from) & bb_move & gold_chk {
                if to.is_promotion_zone(us) {
                    if let ControlFlow::Break(m) = push_bm(from, to, true, &mut emit) {
                        return Some(m);
                    }
                }
            }
        }

        None
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
        LEGAL_QUICK_DM.with(|c| c.set(c.get() + 1));
        let captured = board.do_move(m);
        let in_check = board.is_in_check(us);
        board.undo_move(m, captured);
        !in_check
    }
}
