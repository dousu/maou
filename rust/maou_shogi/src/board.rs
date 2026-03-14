use crate::attack;
use crate::bitboard::Bitboard;
use crate::moves::Move;
use crate::sfen::{self, HIRATE_SFEN};
pub use crate::sfen::SfenError;
use crate::types::{Color, HAND_KINDS, Piece, PIECE_BB_SIZE, PieceType, Square};
use crate::zobrist::ZOBRIST;

/// 将棋盤の状態．
///
/// Mailbox (駒配列) + Bitboard のハイブリッド表現．
///
/// フィールドは `pub(crate)` であり，外部クレートからは
/// アクセサメソッドを通じて参照する．直接のフィールド書き換えは
/// Mailbox・Bitboard・Zobrist hash の整合性を崩すため，
/// 変更には `put_piece`/`remove_piece`/`do_move`/`undo_move` を使用すること．
#[derive(Clone)]
pub struct Board {
    /// 81マスの駒配列(cshogi互換ID: 0-30)．
    pub(crate) squares: [Piece; 81],
    /// 駒種×色ごとのビットボード．piece_bb[color][piece_type as usize]
    ///
    /// インデックス0は未使用(PieceTypeは1から始まるため)．
    /// `piece_bb[color][pt as usize]` で直接アクセスできるよう，
    /// サイズを PIECE_TYPES_NUM + 1 = 15 としている．
    pub(crate) piece_bb: [[Bitboard; PIECE_BB_SIZE]; 2],
    /// 色ごとの全駒占有ビットボード．
    pub(crate) occupied: [Bitboard; 2],
    /// 持ち駒 [色][駒種7] = 個数．
    /// 順序: 歩(0), 香(1), 桂(2), 銀(3), 金(4), 角(5), 飛(6)
    pub(crate) hand: [[u8; HAND_KINDS]; 2],
    /// 手番．
    pub(crate) turn: Color,
    /// 手数．
    pub(crate) ply: u16,
    /// Zobrist hash(インクリメンタル更新)．
    pub(crate) hash: u64,
}

impl Board {
    /// 平手初期局面で生成する．
    pub fn new() -> Board {
        let mut board = Board::empty();
        board.set_sfen(HIRATE_SFEN).expect("invalid hirate SFEN");
        board
    }

    /// 空盤で生成する．
    pub fn empty() -> Board {
        Board {
            squares: [Piece::EMPTY; 81],
            piece_bb: [[Bitboard::EMPTY; PIECE_BB_SIZE]; 2],
            occupied: [Bitboard::EMPTY; 2],
            hand: [[0u8; HAND_KINDS]; 2],
            turn: Color::Black,
            ply: 1,
            hash: 0,
        }
    }

    /// SFEN文字列から局面を設定する．
    pub fn set_sfen(&mut self, sfen: &str) -> Result<(), SfenError> {
        let pos = sfen::parse_sfen(sfen)?;

        self.squares = pos.squares;
        self.turn = pos.turn;
        self.hand = pos.hand;
        self.ply = pos.ply;

        // ビットボードを再構築
        self.piece_bb = [[Bitboard::EMPTY; PIECE_BB_SIZE]; 2];
        self.occupied = [Bitboard::EMPTY; 2];

        for sq_idx in 0..81u8 {
            let piece = self.squares[sq_idx as usize];
            if !piece.is_empty() {
                let color = piece.color().unwrap();
                let pt = piece.piece_type().unwrap();
                self.piece_bb[color.index()][pt as usize].set(Square(sq_idx));
                self.occupied[color.index()].set(Square(sq_idx));
            }
        }

        // Zobrist hashを計算
        self.hash = self.compute_hash();

        Ok(())
    }

    /// SFEN文字列を返す．
    pub fn sfen(&self) -> String {
        sfen::to_sfen(&self.squares, self.turn, &self.hand, self.ply)
    }

    /// 全駒の占有ビットボードを返す．
    #[inline]
    pub fn all_occupied(&self) -> Bitboard {
        self.occupied[0] | self.occupied[1]
    }

    /// 指定した色の玉のマスを返す．
    pub fn king_square(&self, color: Color) -> Option<Square> {
        let bb = self.piece_bb[color.index()][PieceType::King as usize];
        if bb.is_empty() {
            None
        } else {
            let mut bb_copy = bb;
            Some(bb_copy.pop_lsb())
        }
    }

    /// 指定した色が王手されているか(相手の利きが自玉にかかっているか)．
    ///
    /// Inverse bitboard approach: 対象マスから各駒種の利きを逆算し，
    /// 相手の対応する駒種のbitboardと交差判定する．
    pub fn is_in_check(&self, color: Color) -> bool {
        if let Some(king_sq) = self.king_square(color) {
            self.is_attacked_by(king_sq, color.opponent())
        } else {
            false
        }
    }

    /// 指定したマスに対して指定した色からの利きがあるか．
    ///
    /// Inverse bitboard approach: 対象マスから各駒種の利きパターンを逆方向に展開し，
    /// 攻撃側の対応する駒種のbitboardとの交差で判定する．
    /// 全81マスをスキャンする代わりに，駒種ごとのbitboard演算で高速化．
    pub fn is_attacked_by(&self, sq: Square, attacker_color: Color) -> bool {
        let occ = self.all_occupied();
        let opp = attacker_color.index();

        // 歩: 対象マスから「相手の歩の利き方向の逆」を見る
        // = 対象マスから attacker_color の歩の利きを計算し，相手の歩bitboardと交差判定
        // 注: 歩の利きは非対称(色依存)なので，attacker_colorの逆方向を使う
        let defender = attacker_color.opponent();
        if (attack::step_attacks(defender, PieceType::Pawn, sq)
            & self.piece_bb[opp][PieceType::Pawn as usize])
            .is_not_empty()
        {
            return true;
        }

        // 桂: 同様にinverse
        if (attack::step_attacks(defender, PieceType::Knight, sq)
            & self.piece_bb[opp][PieceType::Knight as usize])
            .is_not_empty()
        {
            return true;
        }

        // 銀
        if (attack::step_attacks(defender, PieceType::Silver, sq)
            & self.piece_bb[opp][PieceType::Silver as usize])
            .is_not_empty()
        {
            return true;
        }

        // 金 + 成駒(と,成香,成桂,成銀): 金と同じ動き
        let gold_movers = self.piece_bb[opp][PieceType::Gold as usize]
            | self.piece_bb[opp][PieceType::ProPawn as usize]
            | self.piece_bb[opp][PieceType::ProLance as usize]
            | self.piece_bb[opp][PieceType::ProKnight as usize]
            | self.piece_bb[opp][PieceType::ProSilver as usize];
        if (attack::step_attacks(defender, PieceType::Gold, sq) & gold_movers).is_not_empty() {
            return true;
        }

        // 王
        if (attack::step_attacks(defender, PieceType::King, sq)
            & self.piece_bb[opp][PieceType::King as usize])
            .is_not_empty()
        {
            return true;
        }

        // 香: 対象マスからdefenderの香の利き方向にattackerの香がいるか
        if (attack::lance_attacks(defender, sq, occ)
            & self.piece_bb[opp][PieceType::Lance as usize])
            .is_not_empty()
        {
            return true;
        }

        // 角・馬: 対象マスから斜め方向に角or馬がいるか
        let bishop_attacks = attack::bishop_attacks(sq, occ);
        if (bishop_attacks
            & (self.piece_bb[opp][PieceType::Bishop as usize]
                | self.piece_bb[opp][PieceType::Horse as usize]))
            .is_not_empty()
        {
            return true;
        }

        // 飛・龍: 対象マスから縦横方向に飛or龍がいるか
        let rook_attacks = attack::rook_attacks(sq, occ);
        if (rook_attacks
            & (self.piece_bb[opp][PieceType::Rook as usize]
                | self.piece_bb[opp][PieceType::Dragon as usize]))
            .is_not_empty()
        {
            return true;
        }

        // 馬のステップ部分(前後左右1マス): 王と同じ判定ではカバーされない
        // 馬は斜め走り+前後左右1マスなので，斜めは上のbishop_attacksでカバー済み
        // 前後左右1マスは馬固有のステップ
        if (attack::step_attacks(defender, PieceType::Horse, sq)
            & self.piece_bb[opp][PieceType::Horse as usize])
            .is_not_empty()
        {
            return true;
        }

        // 龍のステップ部分(斜め1マス): 同様に龍固有のステップ
        if (attack::step_attacks(defender, PieceType::Dragon, sq)
            & self.piece_bb[opp][PieceType::Dragon as usize])
            .is_not_empty()
        {
            return true;
        }

        false
    }

    /// 駒を置く(ビットボードも更新)．
    pub fn put_piece(&mut self, sq: Square, piece: Piece) {
        debug_assert!(self.squares[sq.index()].is_empty());
        let color = piece.color().unwrap();
        let pt = piece.piece_type().unwrap();
        self.squares[sq.index()] = piece;
        self.piece_bb[color.index()][pt as usize].set(sq);
        self.occupied[color.index()].set(sq);
        self.hash ^= ZOBRIST.board_hash(color, pt, sq);
    }

    /// 駒を除去する(ビットボードも更新)．
    pub fn remove_piece(&mut self, sq: Square) -> Piece {
        let piece = self.squares[sq.index()];
        debug_assert!(!piece.is_empty());
        let color = piece.color().unwrap();
        let pt = piece.piece_type().unwrap();
        self.squares[sq.index()] = Piece::EMPTY;
        self.piece_bb[color.index()][pt as usize].clear(sq);
        self.occupied[color.index()].clear(sq);
        self.hash ^= ZOBRIST.board_hash(color, pt, sq);
        piece
    }

    /// 手を実行する(取った駒を返す)．
    pub fn do_move(&mut self, m: Move) -> Piece {
        let captured;

        if m.is_drop() {
            // 駒打ち
            let pt = m.drop_piece_type().unwrap();
            let to = m.to_sq();
            let piece = Piece::new(self.turn, pt);

            // 持ち駒から減らす
            let hi = pt.hand_index().unwrap();
            debug_assert!(self.hand[self.turn.index()][hi] > 0);
            self.hash ^= ZOBRIST.hand_hash(self.turn, hi, self.hand[self.turn.index()][hi] as usize);
            self.hand[self.turn.index()][hi] -= 1;
            self.hash ^= ZOBRIST.hand_hash(self.turn, hi, self.hand[self.turn.index()][hi] as usize);

            self.put_piece(to, piece);
            captured = Piece::EMPTY;
        } else {
            // 盤上の駒を動かす
            let from = m.from_sq();
            let to = m.to_sq();

            // 移動元の駒を除去
            let moving_piece = self.remove_piece(from);
            let pt = moving_piece.piece_type().unwrap();

            // 移動先に駒があれば取る
            if !self.squares[to.index()].is_empty() {
                let cap = self.remove_piece(to);
                let cap_pt = cap.piece_type().unwrap();
                let cap_hand_pt = cap_pt.captured_to_hand();
                // 王は持ち駒にできない(合法手では王の捕獲は発生しないが，
                // 疑似合法手の検証中に発生しうる)
                if let Some(hi) = cap_hand_pt.hand_index() {
                    self.hash ^= ZOBRIST.hand_hash(self.turn, hi, self.hand[self.turn.index()][hi] as usize);
                    self.hand[self.turn.index()][hi] += 1;
                    self.hash ^= ZOBRIST.hand_hash(self.turn, hi, self.hand[self.turn.index()][hi] as usize);
                }
                captured = cap;
            } else {
                captured = Piece::EMPTY;
            }

            // 成りの場合は駒種を変更
            let new_pt = if m.is_promotion() {
                pt.promoted().expect("cannot promote this piece")
            } else {
                pt
            };

            self.put_piece(to, Piece::new(self.turn, new_pt));
        }

        // 手番を交代
        self.hash ^= ZOBRIST.turn_hash();
        self.turn = self.turn.opponent();
        self.ply += 1;

        captured
    }

    /// 手を取り消す．
    pub fn undo_move(&mut self, m: Move, captured: Piece) {
        // 手番を戻す
        self.turn = self.turn.opponent();
        self.hash ^= ZOBRIST.turn_hash();
        self.ply -= 1;

        if m.is_drop() {
            let to = m.to_sq();
            let pt = m.drop_piece_type().unwrap();

            // 盤上から駒を除去
            self.remove_piece(to);

            // 持ち駒に戻す
            let hi = pt.hand_index().unwrap();
            self.hash ^= ZOBRIST.hand_hash(self.turn, hi, self.hand[self.turn.index()][hi] as usize);
            self.hand[self.turn.index()][hi] += 1;
            self.hash ^= ZOBRIST.hand_hash(self.turn, hi, self.hand[self.turn.index()][hi] as usize);
        } else {
            let from = m.from_sq();
            let to = m.to_sq();

            // 移動先の駒を除去
            let moved_piece = self.remove_piece(to);
            let moved_pt = moved_piece.piece_type().unwrap();

            // 成りの場合は元に戻す
            let original_pt = if m.is_promotion() {
                moved_pt.unpromoted().unwrap()
            } else {
                moved_pt
            };

            // 移動元に駒を戻す
            self.put_piece(from, Piece::new(self.turn, original_pt));

            // 取った駒を復元
            if !captured.is_empty() {
                let cap_pt = captured.piece_type().unwrap();
                self.put_piece(to, captured);

                // 持ち駒から取った駒を除去(王は持ち駒にならない)
                let cap_hand_pt = cap_pt.captured_to_hand();
                if let Some(hi) = cap_hand_pt.hand_index() {
                    self.hash ^= ZOBRIST.hand_hash(self.turn, hi, self.hand[self.turn.index()][hi] as usize);
                    self.hand[self.turn.index()][hi] -= 1;
                    self.hash ^= ZOBRIST.hand_hash(self.turn, hi, self.hand[self.turn.index()][hi] as usize);
                }
            }
        }
    }

    /// 盤面全体からZobrist hashを計算する(初期化用)．
    pub fn compute_hash(&self) -> u64 {
        let mut hash = 0u64;

        // 盤上の駒
        for sq_idx in 0..81u8 {
            let piece = self.squares[sq_idx as usize];
            if !piece.is_empty() {
                let color = piece.color().unwrap();
                let pt = piece.piece_type().unwrap();
                hash ^= ZOBRIST.board_hash(color, pt, Square(sq_idx));
            }
        }

        // 持ち駒
        for color_idx in 0..2 {
            let color = Color::from_u8(color_idx as u8).unwrap();
            for kind in 0..7 {
                let count = self.hand[color_idx][kind];
                if count > 0 {
                    hash ^= ZOBRIST.hand_hash(color, kind, count as usize);
                }
            }
        }

        // 手番
        if self.turn == Color::White {
            hash ^= ZOBRIST.turn_hash();
        }

        hash
    }

    /// 盤面の駒配列(81要素)をcshogi互換IDで返す．
    pub fn pieces(&self) -> [u8; 81] {
        let mut result = [0u8; 81];
        for i in 0..81 {
            result[i] = self.squares[i].0;
        }
        result
    }

    /// 持ち駒を返す．
    /// 返り値: ([先手の持ち駒; 7], [後手の持ち駒; 7])
    /// 順序: 歩, 香, 桂, 銀, 金, 角, 飛
    pub fn pieces_in_hand(&self) -> ([u8; HAND_KINDS], [u8; HAND_KINDS]) {
        (self.hand[0], self.hand[1])
    }

    /// 指定マスの駒をcshogi IDで返す．
    #[inline]
    pub fn piece_at(&self, sq: Square) -> u8 {
        self.squares[sq.index()].0
    }

    /// 手番を返す．
    #[inline]
    pub fn turn(&self) -> Color {
        self.turn
    }

    /// 手数を返す．
    #[inline]
    pub fn ply(&self) -> u16 {
        self.ply
    }

    /// Zobrist hashを返す．
    #[inline]
    pub fn hash(&self) -> u64 {
        self.hash
    }

    /// 盤面が有効かどうか検証する．
    ///
    /// 片玉局面(詰将棋)もサポートする．
    /// 各色の玉は0枚または1枚のみ有効．
    pub fn is_ok(&self) -> bool {
        // 各色の王は0枚か1枚
        let bk = self.piece_bb[0][PieceType::King as usize];
        let wk = self.piece_bb[1][PieceType::King as usize];
        if bk.count() > 1 || wk.count() > 1 {
            return false;
        }
        // 少なくとも1枚の玉が存在すること
        if bk.count() == 0 && wk.count() == 0 {
            return false;
        }

        // ビットボードとmailboxの整合性
        for sq_idx in 0..81u8 {
            let piece = self.squares[sq_idx as usize];
            let sq = Square(sq_idx);
            if piece.is_empty() {
                if self.occupied[0].contains(sq) || self.occupied[1].contains(sq) {
                    return false;
                }
            } else {
                let color = piece.color().unwrap();
                let pt = piece.piece_type().unwrap();
                if !self.piece_bb[color.index()][pt as usize].contains(sq) {
                    return false;
                }
                if !self.occupied[color.index()].contains(sq) {
                    return false;
                }
            }
        }

        true
    }
}

impl Default for Board {
    fn default() -> Self {
        Board::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_board() {
        let board = Board::new();
        assert_eq!(board.turn, Color::Black);
        assert!(board.is_ok());

        // 平手の歩の位置確認
        let pieces = board.pieces();
        // 先手の歩は7段目(row=6): col 0-8 → squares[col*9+6]
        for col in 0..9 {
            assert_eq!(pieces[col * 9 + 6], 1); // BPAWN = 1
        }
    }

    #[test]
    fn test_sfen_roundtrip() {
        let board = Board::new();
        let sfen = board.sfen();
        let mut board2 = Board::empty();
        board2.set_sfen(&sfen).unwrap();
        assert_eq!(board.pieces(), board2.pieces());
        assert_eq!(board.turn, board2.turn);
        assert_eq!(board.hand, board2.hand);
    }

    #[test]
    fn test_king_square() {
        let board = Board::new();
        // 先手の王は5九(col=4, row=8) → square = 4*9+8 = 44
        assert_eq!(board.king_square(Color::Black), Some(Square(44)));
        // 後手の王は5一(col=4, row=0) → square = 4*9+0 = 36
        assert_eq!(board.king_square(Color::White), Some(Square(36)));
    }

    #[test]
    fn test_do_undo_move() {
        let mut board = Board::new();
        let original_sfen = board.sfen();
        let original_hash = board.hash;

        // 7六歩: from=Square(6,6)=60, to=Square(6,5)=59
        // (USI: 7g7f → col=7-1=6, row=g=6 → f=5)
        let m = Move::new_move(
            Square::new(6, 6),
            Square::new(6, 5),
            false,
            0,
            PieceType::Pawn as u8,
        );
        let captured = board.do_move(m);
        assert_eq!(captured, Piece::EMPTY);
        assert_eq!(board.turn, Color::White);

        // 取り消し
        board.undo_move(m, captured);
        assert_eq!(board.sfen(), original_sfen);
        assert_eq!(board.hash, original_hash);
    }

    #[test]
    fn test_is_in_check() {
        // 王手がかかっている局面を設定
        let mut board = Board::empty();
        board
            .set_sfen("4k4/9/9/9/9/9/9/9/4K3R b - 1")
            .unwrap();
        // 先手の飛車(1九=col8,row8)は後手の玉(5一=col4,row0)に利いていない(別の行)
        // → 別の局面にする
        board
            .set_sfen("4k4/9/9/9/9/9/9/9/4K4 b R 1")
            .unwrap();
        assert!(!board.is_in_check(Color::White));
    }

    #[test]
    fn test_single_king_tsume() {
        // 片玉局面(詰将棋): 攻め方に玉がない
        let mut board = Board::empty();
        board
            .set_sfen("4k4/9/4G4/9/9/9/9/9/9 b G 1")
            .unwrap();
        assert!(board.is_ok());
        assert_eq!(board.king_square(Color::Black), None);
        assert!(board.king_square(Color::White).is_some());

        // 攻め方に玉がなくても合法手を生成できる
        assert!(!board.is_in_check(Color::Black)); // 玉がない → 王手されていない
    }

    #[test]
    fn test_single_king_is_ok() {
        // 片玉は有効
        let mut board = Board::empty();
        board.set_sfen("4k4/9/9/9/9/9/9/9/9 b G 1").unwrap();
        assert!(board.is_ok());

        // 玉なしは無効
        let mut board = Board::empty();
        board.set_sfen("9/9/9/9/9/9/9/9/9 b G 1").unwrap();
        assert!(!board.is_ok());
    }
}
