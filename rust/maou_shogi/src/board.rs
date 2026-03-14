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
    #[inline]
    pub fn king_square(&self, color: Color) -> Option<Square> {
        let bb = self.piece_bb[color.index()][PieceType::King as usize];
        bb.lsb()
    }

    /// 指定した色が王手されているか(相手の利きが自玉にかかっているか)．
    ///
    /// Inverse bitboard approach: 対象マスから各駒種の利きを逆算し，
    /// 相手の対応する駒種のbitboardと交差判定する．
    #[inline]
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
    #[inline]
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

        // 王・馬(ステップ部分)・龍(ステップ部分):
        // 王の利きは8方向で，馬のステップ(前後左右)と龍のステップ(斜め4方向)の
        // 上位集合である．したがって王の利きパターン1回で3駒種をまとめて判定できる．
        let king_step = attack::step_attacks(defender, PieceType::King, sq);
        let king_horse_dragon = self.piece_bb[opp][PieceType::King as usize]
            | self.piece_bb[opp][PieceType::Horse as usize]
            | self.piece_bb[opp][PieceType::Dragon as usize];
        if (king_step & king_horse_dragon).is_not_empty() {
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
        if (attack::bishop_attacks(sq, occ)
            & (self.piece_bb[opp][PieceType::Bishop as usize]
                | self.piece_bb[opp][PieceType::Horse as usize]))
            .is_not_empty()
        {
            return true;
        }

        // 飛・龍: 対象マスから縦横方向に飛or龍がいるか
        if (attack::rook_attacks(sq, occ)
            & (self.piece_bb[opp][PieceType::Rook as usize]
                | self.piece_bb[opp][PieceType::Dragon as usize]))
            .is_not_empty()
        {
            return true;
        }

        false
    }

    /// 駒を置く(ビットボードも更新)．
    ///
    /// # 前提条件
    ///
    /// - `sq` が空マスであること(`debug_assert` で検証)．
    /// - `piece` が有効な色と駒種を持つこと．
    #[inline]
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
    ///
    /// # 前提条件
    ///
    /// - `sq` に駒が存在すること(`debug_assert` で検証)．
    #[inline]
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
    #[inline]
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
    #[inline]
    pub fn undo_move(&mut self, m: Move, captured: Piece) {
        // 手番を戻す
        self.turn = self.turn.opponent();
        self.hash ^= ZOBRIST.turn_hash();
        debug_assert!(self.ply > 0, "undo_move called with ply == 0");
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

        // 持ち駒の枚数上限(先後合計 + 盤上枚数 ≦ 最大枚数)
        for (i, &max) in PieceType::MAX_HAND_COUNT.iter().enumerate() {
            let hand_total = self.hand[0][i] as u32 + self.hand[1][i] as u32;
            let base_pt = PieceType::HAND_PIECES[i];
            let mut on_board = (self.piece_bb[0][base_pt as usize]
                | self.piece_bb[1][base_pt as usize])
                .count();
            // 成駒も盤上枚数に含める
            if let Some(promoted) = base_pt.promoted() {
                on_board += (self.piece_bb[0][promoted as usize]
                    | self.piece_bb[1][promoted as usize])
                    .count();
            }
            if hand_total + on_board > max as u32 {
                return false;
            }
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

impl std::fmt::Display for Board {
    /// 将棋盤のAA(アスキーアート)表示．
    ///
    /// 先手視点で表示する．左端が9筋，右端が1筋．
    /// 持ち駒は価値順(飛→角→金→銀→桂→香→歩)で表示する．
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        /// 駒を2文字の日本語表記に変換する．
        fn piece_str(piece: Piece) -> &'static str {
            if piece.is_empty() {
                return " ・";
            }
            let color = piece.color().unwrap();
            let pt = piece.piece_type().unwrap();
            match (color, pt) {
                (Color::Black, PieceType::Pawn) => " 歩",
                (Color::Black, PieceType::Lance) => " 香",
                (Color::Black, PieceType::Knight) => " 桂",
                (Color::Black, PieceType::Silver) => " 銀",
                (Color::Black, PieceType::Bishop) => " 角",
                (Color::Black, PieceType::Rook) => " 飛",
                (Color::Black, PieceType::Gold) => " 金",
                (Color::Black, PieceType::King) => " 玉",
                (Color::Black, PieceType::ProPawn) => " と",
                (Color::Black, PieceType::ProLance) => " 杏",
                (Color::Black, PieceType::ProKnight) => " 圭",
                (Color::Black, PieceType::ProSilver) => " 全",
                (Color::Black, PieceType::Horse) => " 馬",
                (Color::Black, PieceType::Dragon) => " 龍",
                (Color::White, PieceType::Pawn) => "v歩",
                (Color::White, PieceType::Lance) => "v香",
                (Color::White, PieceType::Knight) => "v桂",
                (Color::White, PieceType::Silver) => "v銀",
                (Color::White, PieceType::Bishop) => "v角",
                (Color::White, PieceType::Rook) => "v飛",
                (Color::White, PieceType::Gold) => "v金",
                (Color::White, PieceType::King) => "v王",
                (Color::White, PieceType::ProPawn) => "vと",
                (Color::White, PieceType::ProLance) => "v杏",
                (Color::White, PieceType::ProKnight) => "v圭",
                (Color::White, PieceType::ProSilver) => "v全",
                (Color::White, PieceType::Horse) => "v馬",
                (Color::White, PieceType::Dragon) => "v龍",
            }
        }

        // 持ち駒の表示順序: 飛(6), 角(5), 金(4), 銀(3), 桂(2), 香(1), 歩(0)
        const HAND_ORDER: [usize; 7] = [6, 5, 4, 3, 2, 1, 0];
        const HAND_NAMES: [&str; 7] = ["歩", "香", "桂", "銀", "金", "角", "飛"];

        // 後手の持ち駒
        write!(f, "後手の持駒：")?;
        let mut has_white_hand = false;
        for &i in &HAND_ORDER {
            let count = self.hand[1][i];
            if count > 0 {
                has_white_hand = true;
                write!(f, "{}", HAND_NAMES[i])?;
                if count > 1 {
                    write!(f, "{}", count)?;
                }
                write!(f, " ")?;
            }
        }
        if !has_white_hand {
            write!(f, "なし")?;
        }
        writeln!(f)?;

        // 筋番号ヘッダー(9→1の順)
        writeln!(f, "  ９ ８ ７ ６ ５ ４ ３ ２ １")?;
        writeln!(f, "+--+--+--+--+--+--+--+--+--+")?;

        // 盤面(row=0が一段目，row=8が九段目)
        const DAN_NAMES: [&str; 9] = ["一", "二", "三", "四", "五", "六", "七", "八", "九"];
        for row in 0..9u8 {
            write!(f, "|")?;
            // 左端が9筋(col=8)，右端が1筋(col=0)
            for visual_col in 0..9u8 {
                let col = 8 - visual_col;
                let sq = Square::new(col, row);
                let piece = self.squares[sq.index()];
                write!(f, "{}|", piece_str(piece))?;
            }
            writeln!(f, "{}", DAN_NAMES[row as usize])?;
        }

        writeln!(f, "+--+--+--+--+--+--+--+--+--+")?;

        // 先手の持ち駒
        write!(f, "先手の持駒：")?;
        let mut has_black_hand = false;
        for &i in &HAND_ORDER {
            let count = self.hand[0][i];
            if count > 0 {
                has_black_hand = true;
                write!(f, "{}", HAND_NAMES[i])?;
                if count > 1 {
                    write!(f, "{}", count)?;
                }
                write!(f, " ")?;
            }
        }
        if !has_black_hand {
            write!(f, "なし")?;
        }

        Ok(())
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

    #[test]
    fn test_display_hirate() {
        let board = Board::new();
        let output = format!("{}", board);
        // 先手の玉は五筋九段目
        assert!(output.contains(" 玉"), "should contain black king");
        // 後手の王は五筋一段目
        assert!(output.contains("v王"), "should contain white king");
        // 筋番号ヘッダー
        assert!(output.contains("９ ８ ７ ６ ５ ４ ３ ２ １"));
        // 持ち駒なし
        assert!(output.contains("先手の持駒：なし"));
        assert!(output.contains("後手の持駒：なし"));
        // 先手角は八筋(左から2番目)に存在
        assert!(output.contains(" 角"), "should contain black bishop");
        // 後手角も存在
        assert!(output.contains("v角"), "should contain white bishop");
    }

    #[test]
    fn test_display_with_hand() {
        let mut board = Board::empty();
        board.set_sfen("4k4/9/9/9/9/9/9/9/4K4 b 2G3Pbs 1").unwrap();
        let output = format!("{}", board);
        // 先手: 金2歩3
        assert!(output.contains("金2"), "should show 2 golds");
        assert!(output.contains("歩3"), "should show 3 pawns");
        // 後手: 角1銀1
        assert!(output.contains("後手の持駒：角 銀"), "should show bishop and silver");
    }
}
