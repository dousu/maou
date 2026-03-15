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
    /// 全駒占有ビットボード(キャッシュ)．
    ///
    /// `occupied[0] | occupied[1]` を事前計算し保持する．
    /// `all_occupied()` の呼び出しコストを排除する．
    pub(crate) all_occ: Bitboard,
    /// 持ち駒 [色][駒種7] = 個数．
    /// 順序: 歩(0), 香(1), 桂(2), 銀(3), 金(4), 角(5), 飛(6)
    pub(crate) hand: [[u8; HAND_KINDS]; 2],
    /// 手番．
    pub(crate) turn: Color,
    /// 手数．
    pub(crate) ply: u16,
    /// Zobrist hash(インクリメンタル更新)．盤面+持ち駒+手番を含む完全ハッシュ．
    pub(crate) hash: u64,
    /// 盤面のみの Zobrist hash(持ち駒を除外)．
    ///
    /// `hash` から持ち駒成分を除いたもの．position_key() を O(1) にするために
    /// インクリメンタルに維持する．盤上の駒配置と手番のみを反映する．
    pub(crate) board_hash: u64,
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
            all_occ: Bitboard::EMPTY,
            hand: [[0u8; HAND_KINDS]; 2],
            turn: Color::Black,
            ply: 1,
            hash: 0,
            board_hash: 0,
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
        self.all_occ = Bitboard::EMPTY;

        for sq_idx in 0..81u8 {
            let piece = self.squares[sq_idx as usize];
            if !piece.is_empty() {
                let color = piece.color().unwrap();
                let pt = piece.piece_type().unwrap();
                self.piece_bb[color.index()][pt as usize].set(Square(sq_idx));
                self.occupied[color.index()].set(Square(sq_idx));
                self.all_occ.set(Square(sq_idx));
            }
        }

        // Zobrist hashを計算
        self.hash = self.compute_hash();
        self.board_hash = self.compute_board_hash();

        Ok(())
    }

    /// SFEN文字列を返す．
    pub fn sfen(&self) -> String {
        sfen::to_sfen(&self.squares, self.turn, &self.hand, self.ply)
    }

    /// 全駒の占有ビットボードを返す．
    #[inline]
    pub fn all_occupied(&self) -> Bitboard {
        self.all_occ
    }

    /// 指定した色の玉のマスを返す．
    #[inline]
    pub fn king_square(&self, color: Color) -> Option<Square> {
        let bb = self.piece_bb[color.index()][PieceType::King as usize];
        bb.lsb()
    }

    /// 指定した色が王手されているか(相手の利きが自玉にかかっているか)．
    ///
    /// 玉が存在しない場合(片玉の詰将棋など)は `false` を返す．
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

    /// 指定した玉に王手している相手駒のビットボードを返す．
    ///
    /// `color` は守備側(玉の色)．王手している全ての駒のマスが立つ．
    #[inline]
    pub fn checkers_of(&self, color: Color) -> Bitboard {
        if let Some(king_sq) = self.king_square(color) {
            self.compute_checkers_at(king_sq, color.opponent())
        } else {
            Bitboard::EMPTY
        }
    }

    /// 指定マスに対して指定した色の駒で王手しているもののビットボードを返す．
    ///
    /// Inverse bitboard approach: 対象マスから各駒種の利きパターンを逆方向に展開し，
    /// 攻撃側の対応する駒種のbitboardとの交差で判定する．
    pub fn compute_checkers_at(&self, king_sq: Square, attacker: Color) -> Bitboard {
        let occ = self.all_occupied();
        let att = attacker.index();
        let defender = attacker.opponent();

        let mut checkers = Bitboard::EMPTY;

        // 歩
        checkers |= attack::step_attacks(defender, PieceType::Pawn, king_sq)
            & self.piece_bb[att][PieceType::Pawn as usize];
        // 桂
        checkers |= attack::step_attacks(defender, PieceType::Knight, king_sq)
            & self.piece_bb[att][PieceType::Knight as usize];
        // 銀
        checkers |= attack::step_attacks(defender, PieceType::Silver, king_sq)
            & self.piece_bb[att][PieceType::Silver as usize];
        // 金 + 成駒
        let gold_movers = self.piece_bb[att][PieceType::Gold as usize]
            | self.piece_bb[att][PieceType::ProPawn as usize]
            | self.piece_bb[att][PieceType::ProLance as usize]
            | self.piece_bb[att][PieceType::ProKnight as usize]
            | self.piece_bb[att][PieceType::ProSilver as usize];
        checkers |= attack::step_attacks(defender, PieceType::Gold, king_sq) & gold_movers;
        // 馬・龍(ステップ部分)
        let king_step = attack::step_attacks(defender, PieceType::King, king_sq);
        checkers |= king_step
            & (self.piece_bb[att][PieceType::Horse as usize]
                | self.piece_bb[att][PieceType::Dragon as usize]);
        // 香
        checkers |= attack::lance_attacks(defender, king_sq, occ)
            & self.piece_bb[att][PieceType::Lance as usize];
        // 角・馬
        checkers |= attack::bishop_attacks(king_sq, occ)
            & (self.piece_bb[att][PieceType::Bishop as usize]
                | self.piece_bb[att][PieceType::Horse as usize]);
        // 飛・龍
        checkers |= attack::rook_attacks(king_sq, occ)
            & (self.piece_bb[att][PieceType::Rook as usize]
                | self.piece_bb[att][PieceType::Dragon as usize]);

        checkers
    }

    /// 自玉にピンされている自駒のビットボードを返す．
    ///
    /// 自玉と相手の飛び駒の間に自駒が1枚だけある場合，その駒はピンされている．
    pub fn compute_pinned(&self, us: Color, king_sq: Square) -> Bitboard {
        let them = us.opponent();
        let all_occ = self.all_occupied();
        let our_occ = self.occupied[us.index()];
        let mut pinned = Bitboard::EMPTY;

        // 飛・龍によるピン(縦横)
        let rook_like = self.piece_bb[them.index()][PieceType::Rook as usize]
            | self.piece_bb[them.index()][PieceType::Dragon as usize];
        let rook_xray = attack::rook_attacks(king_sq, Bitboard::EMPTY);
        let rook_pinners = rook_xray & rook_like;
        for pinner_sq in rook_pinners {
            let between = attack::between_bb(king_sq, pinner_sq);
            let blockers = between & all_occ;
            if blockers.count() == 1 {
                pinned |= blockers & our_occ;
            }
        }

        // 角・馬によるピン(斜め)
        let bishop_like = self.piece_bb[them.index()][PieceType::Bishop as usize]
            | self.piece_bb[them.index()][PieceType::Horse as usize];
        let bishop_xray = attack::bishop_attacks(king_sq, Bitboard::EMPTY);
        let bishop_pinners = bishop_xray & bishop_like;
        for pinner_sq in bishop_pinners {
            let between = attack::between_bb(king_sq, pinner_sq);
            let blockers = between & all_occ;
            if blockers.count() == 1 {
                pinned |= blockers & our_occ;
            }
        }

        // 香によるピン
        let enemy_lance = self.piece_bb[them.index()][PieceType::Lance as usize];
        let lance_xray = attack::lance_attacks(us, king_sq, Bitboard::EMPTY);
        let lance_pinners = lance_xray & enemy_lance;
        for pinner_sq in lance_pinners {
            let between = attack::between_bb(king_sq, pinner_sq);
            let blockers = between & all_occ;
            if blockers.count() == 1 {
                pinned |= blockers & our_occ;
            }
        }

        pinned
    }

    /// 玉の危険マスを計算する(相手の利きがあるマス)．
    ///
    /// 玉を盤面から除去した状態で相手の利きを計算する(X-ray対策)．
    pub fn compute_king_danger(&self, us: Color, king_sq: Square) -> Bitboard {
        let them = us.opponent();

        // 玉を除去した占有ビットボード(X-ray: 飛び駒が玉を貫通する)
        let mut occ_no_king = self.all_occupied();
        occ_no_king.clear(king_sq);

        let opp = them.index();
        let mut danger = Bitboard::EMPTY;

        // 歩
        for sq in self.piece_bb[opp][PieceType::Pawn as usize] {
            danger |= attack::step_attacks(them, PieceType::Pawn, sq);
        }
        // 桂
        for sq in self.piece_bb[opp][PieceType::Knight as usize] {
            danger |= attack::step_attacks(them, PieceType::Knight, sq);
        }
        // 銀
        for sq in self.piece_bb[opp][PieceType::Silver as usize] {
            danger |= attack::step_attacks(them, PieceType::Silver, sq);
        }
        // 金 + 成駒
        let gold_movers = self.piece_bb[opp][PieceType::Gold as usize]
            | self.piece_bb[opp][PieceType::ProPawn as usize]
            | self.piece_bb[opp][PieceType::ProLance as usize]
            | self.piece_bb[opp][PieceType::ProKnight as usize]
            | self.piece_bb[opp][PieceType::ProSilver as usize];
        for sq in gold_movers {
            danger |= attack::step_attacks(them, PieceType::Gold, sq);
        }
        // 王
        for sq in self.piece_bb[opp][PieceType::King as usize] {
            danger |= attack::step_attacks(them, PieceType::King, sq);
        }
        // 香
        for sq in self.piece_bb[opp][PieceType::Lance as usize] {
            danger |= attack::lance_attacks(them, sq, occ_no_king);
        }
        // 角
        for sq in self.piece_bb[opp][PieceType::Bishop as usize] {
            danger |= attack::bishop_attacks(sq, occ_no_king);
        }
        // 馬
        for sq in self.piece_bb[opp][PieceType::Horse as usize] {
            danger |= attack::horse_attacks(them, sq, occ_no_king);
        }
        // 飛
        for sq in self.piece_bb[opp][PieceType::Rook as usize] {
            danger |= attack::rook_attacks(sq, occ_no_king);
        }
        // 龍
        for sq in self.piece_bb[opp][PieceType::Dragon as usize] {
            danger |= attack::dragon_attacks(them, sq, occ_no_king);
        }

        danger
    }

    /// 指定マスに対する利き判定(除外マスク付き)．
    ///
    /// `exclude_king`: 玉の利きを除外するか．
    /// `excluded_sq`: 指定マスの駒を除外して利き計算する(X-ray対策)．
    pub fn is_attacked_by_excluding(
        &self,
        sq: Square,
        attacker_color: Color,
        exclude_king: bool,
        excluded_sq: Option<Square>,
    ) -> bool {
        let mut occ = self.all_occupied();
        let att = attacker_color.index();
        let defender = attacker_color.opponent();

        let mask = match excluded_sq {
            Some(esq) => {
                let mut m = Bitboard::EMPTY;
                m.set(esq);
                occ = occ & !m;
                !m
            }
            None => !Bitboard::EMPTY,
        };

        // 歩
        if (attack::step_attacks(defender, PieceType::Pawn, sq)
            & self.piece_bb[att][PieceType::Pawn as usize]
            & mask)
            .is_not_empty()
        {
            return true;
        }
        // 桂
        if (attack::step_attacks(defender, PieceType::Knight, sq)
            & self.piece_bb[att][PieceType::Knight as usize]
            & mask)
            .is_not_empty()
        {
            return true;
        }
        // 銀
        if (attack::step_attacks(defender, PieceType::Silver, sq)
            & self.piece_bb[att][PieceType::Silver as usize]
            & mask)
            .is_not_empty()
        {
            return true;
        }
        // 金 + 成駒
        let gold_movers = (self.piece_bb[att][PieceType::Gold as usize]
            | self.piece_bb[att][PieceType::ProPawn as usize]
            | self.piece_bb[att][PieceType::ProLance as usize]
            | self.piece_bb[att][PieceType::ProKnight as usize]
            | self.piece_bb[att][PieceType::ProSilver as usize])
            & mask;
        if (attack::step_attacks(defender, PieceType::Gold, sq) & gold_movers).is_not_empty() {
            return true;
        }
        // 玉・馬・龍(ステップ部分)
        let king_step = attack::step_attacks(defender, PieceType::King, sq);
        let mut step_pieces = self.piece_bb[att][PieceType::Horse as usize]
            | self.piece_bb[att][PieceType::Dragon as usize];
        if !exclude_king {
            step_pieces |= self.piece_bb[att][PieceType::King as usize];
        }
        if (king_step & step_pieces & mask).is_not_empty() {
            return true;
        }
        // 香
        if (attack::lance_attacks(defender, sq, occ)
            & self.piece_bb[att][PieceType::Lance as usize]
            & mask)
            .is_not_empty()
        {
            return true;
        }
        // 角・馬(スライド部分)
        if (attack::bishop_attacks(sq, occ)
            & (self.piece_bb[att][PieceType::Bishop as usize]
                | self.piece_bb[att][PieceType::Horse as usize])
            & mask)
            .is_not_empty()
        {
            return true;
        }
        // 飛・龍(スライド部分)
        if (attack::rook_attacks(sq, occ)
            & (self.piece_bb[att][PieceType::Rook as usize]
                | self.piece_bb[att][PieceType::Dragon as usize])
            & mask)
            .is_not_empty()
        {
            return true;
        }
        false
    }

    /// 飛び駒と玉の間に自駒が1枚だけある場合のブロッカーを返す．
    ///
    /// `compute_pinned` と `compute_discoverers` で共通するパターン．
    /// `slider_bb`: 検査対象の飛び駒のビットボード．
    /// `xray_attacks`: 遮蔽なしでの飛び利き(玉マスから)．
    /// `blocker_occ`: ブロッカーとして認める駒の占有(自駒 or 相手駒)．
    fn find_single_blockers(
        &self,
        king_sq: Square,
        slider_bb: Bitboard,
        xray_attacks: Bitboard,
        blocker_occ: Bitboard,
    ) -> Bitboard {
        let all_occ = self.all_occupied();
        let candidates = xray_attacks & slider_bb;
        let mut result = Bitboard::EMPTY;
        for slider_sq in candidates {
            let between = attack::between_bb(king_sq, slider_sq);
            let blockers = between & all_occ;
            if blockers.count() == 1 {
                result |= blockers & blocker_occ;
            }
        }
        result
    }

    /// 指定マスに飛び駒の開き王手候補になる自駒のビットボードを返す．
    ///
    /// 自駒の飛び駒と相手玉の間に自駒が1枚だけある場合，
    /// その駒が移動すると開き王手になる．
    pub fn compute_discoverers(&self, us: Color, opp_king_sq: Square) -> Bitboard {
        let our_occ = self.occupied[us.index()];
        let mut discoverers = Bitboard::EMPTY;

        // 飛車・龍
        let rook_like = self.piece_bb[us.index()][PieceType::Rook as usize]
            | self.piece_bb[us.index()][PieceType::Dragon as usize];
        discoverers |= self.find_single_blockers(
            opp_king_sq,
            rook_like,
            attack::rook_attacks(opp_king_sq, Bitboard::EMPTY),
            our_occ,
        );

        // 角・馬
        let bishop_like = self.piece_bb[us.index()][PieceType::Bishop as usize]
            | self.piece_bb[us.index()][PieceType::Horse as usize];
        discoverers |= self.find_single_blockers(
            opp_king_sq,
            bishop_like,
            attack::bishop_attacks(opp_king_sq, Bitboard::EMPTY),
            our_occ,
        );

        // 香
        let lance = self.piece_bb[us.index()][PieceType::Lance as usize];
        // 相手玉から見て自分の香が攻撃してくる方向
        let them = us.opponent();
        discoverers |= self.find_single_blockers(
            opp_king_sq,
            lance,
            attack::lance_attacks(them, opp_king_sq, Bitboard::EMPTY),
            our_occ,
        );

        discoverers
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

    // ============================================================
    // ビットボードベースの1手詰め判定 (mateMoveIn1Ply 相当)
    // ============================================================

    /// ビットボード演算のみで1手詰めを判定する(do_move/undo_move 不要)．
    ///
    /// cshogi の `mateMoveIn1Ply()` に相当する高速版．
    /// 与えられた王手手リストから，詰みになる手があれば返す．
    ///
    /// # アルゴリズム
    ///
    /// 各王手手について以下を確認:
    /// 1. 玉に逃げ場がないか(ビットボード利き計算)
    /// 2. 王手駒を取り返せないか(ピン判定含む)
    /// 3. 合い駒が効かないか(飛び駒の王手のみ)
    pub fn mate_move_in_1ply(
        &self,
        checks: &[Move],
        attacker: Color,
    ) -> Option<Move> {
        let defender = attacker.opponent();
        let king_sq = self.king_square(defender)?;
        let all_occ = self.all_occupied();
        let att_idx = attacker.index();
        let def_idx = defender.index();
        let att_occ = self.occupied[att_idx];
        let def_occ = self.occupied[def_idx];
        let king_bb = Bitboard::from_square(king_sq);

        // 玉の移動先候補(ステップ利き)
        let king_step = attack::step_attacks(defender, PieceType::King, king_sq);

        // 攻め方と玉の間にいる守備駒(ピン候補) - 王手駒の捕獲可否判定に使用
        let pinned = self.compute_pinned(defender, king_sq);

        for &m in checks {
            if self.is_checkmate_after_bb(
                m, attacker, defender, king_sq, king_bb, king_step,
                all_occ, att_occ, def_occ, att_idx, def_idx, &pinned,
            ) {
                return Some(m);
            }
        }
        None
    }

    /// 指定した王手手を指した後に詰みになるかビットボードで判定する．
    ///
    /// 盤面を変更せず，ビットボード演算のみで判定する．
    #[allow(clippy::too_many_arguments)]
    fn is_checkmate_after_bb(
        &self,
        m: Move,
        attacker: Color,
        defender: Color,
        king_sq: Square,
        king_bb: Bitboard,
        king_step: Bitboard,
        all_occ: Bitboard,
        _att_occ: Bitboard,
        def_occ: Bitboard,
        att_idx: usize,
        def_idx: usize,
        pinned: &Bitboard,
    ) -> bool {
        let to_sq = m.to_sq();
        let to_bb = Bitboard::from_square(to_sq);

        // 王手後の駒種と移動元を取得
        let (from_opt, checker_pt) = if m.is_drop() {
            (None, m.drop_piece_type().unwrap())
        } else {
            let from = m.from_sq();
            let raw_pt = unsafe {
                // moving_piece_type_raw は有効な PieceType 値を保証
                std::mem::transmute::<u8, PieceType>(m.moving_piece_type_raw())
            };
            let pt = if m.is_promotion() {
                raw_pt.promoted().unwrap()
            } else {
                raw_pt
            };
            (Some(from), pt)
        };

        // === 王手後の占有ビットボードを計算 ===
        let occ_after = match from_opt {
            Some(from) => (all_occ & !Bitboard::from_square(from)) | to_bb,
            None => all_occ | to_bb,
        };
        // 守備側の占有(駒取りの場合は to_sq が除去される)
        let def_occ_after = def_occ & !to_bb;

        // === 1. 玉の逃げ場チェック ===
        // 玉を除去した占有(X-ray: 飛び駒が玉を貫通)
        let occ_no_king = occ_after & !king_bb;

        // 玉の移動先候補: 味方駒がいないマス
        let king_targets = king_step & !def_occ_after;

        // 王手駒の利きを計算(玉を除いた占有で)
        let checker_attacks = attack::piece_attacks(attacker, checker_pt, to_sq, occ_no_king);

        // 各逃げ先が安全かチェック
        for esc_sq in king_targets {
            if esc_sq == to_sq {
                // 玉が王手駒を取る場合: 王手駒を除去した状態でチェック
                let occ_esc = occ_no_king & !to_bb;
                if !self.is_sq_attacked_after_move(
                    esc_sq, attacker, occ_esc, att_idx, def_idx,
                    from_opt, to_sq, true,
                ) {
                    return false; // 玉が王手駒を取って逃げられる
                }
            } else if checker_attacks.contains(esc_sq) {
                // 王手駒の利きに入っている → 逃げられない(他の駒も確認不要)
                continue;
            } else {
                // 他の攻め方駒の利きがあるか確認
                if !self.is_sq_attacked_after_move(
                    esc_sq, attacker, occ_no_king, att_idx, def_idx,
                    from_opt, to_sq, false,
                ) {
                    return false; // 安全な逃げ場がある
                }
            }
        }

        // === 2. 王手駒の捕獲チェック(玉以外) ===
        // ピンされていない守備駒で王手駒を取れるか
        if self.can_capture_checker_bb(
            to_sq, king_sq, defender, occ_after, def_occ, def_idx, pinned,
        ) {
            return false;
        }

        // === 3. 合い駒チェック(飛び駒の王手のみ) ===
        if Self::is_sliding_piece(checker_pt) {
            let between = attack::between_bb(to_sq, king_sq);
            if between.is_not_empty()
                && self.can_interpose_bb(
                    &between, king_sq, defender, occ_after, def_occ,
                    def_idx, att_idx, pinned, from_opt,
                )
            {
                return false;
            }
        }

        true // 詰み!
    }

    /// 飛び駒かどうか判定する．
    #[inline]
    fn is_sliding_piece(pt: PieceType) -> bool {
        matches!(
            pt,
            PieceType::Lance
                | PieceType::Bishop
                | PieceType::Rook
                | PieceType::Horse
                | PieceType::Dragon
        )
    }

    /// 王手後の盤面で，指定マスが攻め方に利かれているか判定する．
    ///
    /// `occ` は調整済みの占有ビットボード(玉除去済み等)．
    /// `from_opt` は移動した駒の元位置(その位置の駒は除外)．
    /// `exclude_to` が true の場合，to_sq の駒も除外する(玉が to_sq に移動して
    /// 王手駒を取る場合)．
    #[allow(clippy::too_many_arguments)]
    fn is_sq_attacked_after_move(
        &self,
        sq: Square,
        attacker: Color,
        occ: Bitboard,
        att_idx: usize,
        _def_idx: usize,
        from_opt: Option<Square>,
        to_sq: Square,
        exclude_to: bool,
    ) -> bool {
        let defender = attacker.opponent();

        // from_sq にいた駒は移動済みなので除外するマスク
        let exclude_mask = match from_opt {
            Some(from) => !Bitboard::from_square(from),
            None => Bitboard::ALL,
        };
        // to_sq の駒も除外する場合(玉が王手駒を取る)
        let exclude_mask = if exclude_to {
            exclude_mask & !Bitboard::from_square(to_sq)
        } else {
            exclude_mask
        };

        // to_sq にいる移動後の駒は exclude_to でなければ攻撃に参加する
        // → piece_bb にはまだ反映されていないので，
        //   exclude_to でない場合は to_sq の駒の利きを別途チェック
        // ただし piece_bb ベースの判定では from_opt の駒も残っているため，
        // exclude_mask で除外する

        // 歩
        if (attack::step_attacks(defender, PieceType::Pawn, sq)
            & self.piece_bb[att_idx][PieceType::Pawn as usize]
            & exclude_mask)
            .is_not_empty()
        {
            return true;
        }
        // 桂
        if (attack::step_attacks(defender, PieceType::Knight, sq)
            & self.piece_bb[att_idx][PieceType::Knight as usize]
            & exclude_mask)
            .is_not_empty()
        {
            return true;
        }
        // 銀
        if (attack::step_attacks(defender, PieceType::Silver, sq)
            & self.piece_bb[att_idx][PieceType::Silver as usize]
            & exclude_mask)
            .is_not_empty()
        {
            return true;
        }
        // 金 + 成駒
        let gold_movers = (self.piece_bb[att_idx][PieceType::Gold as usize]
            | self.piece_bb[att_idx][PieceType::ProPawn as usize]
            | self.piece_bb[att_idx][PieceType::ProLance as usize]
            | self.piece_bb[att_idx][PieceType::ProKnight as usize]
            | self.piece_bb[att_idx][PieceType::ProSilver as usize])
            & exclude_mask;
        if (attack::step_attacks(defender, PieceType::Gold, sq) & gold_movers).is_not_empty() {
            return true;
        }
        // 王・馬・龍(ステップ部分)
        let king_step = attack::step_attacks(defender, PieceType::King, sq);
        let step_pieces = (self.piece_bb[att_idx][PieceType::King as usize]
            | self.piece_bb[att_idx][PieceType::Horse as usize]
            | self.piece_bb[att_idx][PieceType::Dragon as usize])
            & exclude_mask;
        if (king_step & step_pieces).is_not_empty() {
            return true;
        }
        // 香
        if (attack::lance_attacks(defender, sq, occ)
            & self.piece_bb[att_idx][PieceType::Lance as usize]
            & exclude_mask)
            .is_not_empty()
        {
            return true;
        }
        // 角・馬(スライド部分)
        if (attack::bishop_attacks(sq, occ)
            & (self.piece_bb[att_idx][PieceType::Bishop as usize]
                | self.piece_bb[att_idx][PieceType::Horse as usize])
            & exclude_mask)
            .is_not_empty()
        {
            return true;
        }
        // 飛・龍(スライド部分)
        if (attack::rook_attacks(sq, occ)
            & (self.piece_bb[att_idx][PieceType::Rook as usize]
                | self.piece_bb[att_idx][PieceType::Dragon as usize])
            & exclude_mask)
            .is_not_empty()
        {
            return true;
        }

        false
    }

    /// 王手駒を玉以外の守備駒で取れるか(ビットボード判定)．
    ///
    /// ピンされた駒は取り返しに使えない(取ると玉が素抜かれる)．
    #[allow(clippy::too_many_arguments)]
    fn can_capture_checker_bb(
        &self,
        checker_sq: Square,
        _king_sq: Square,
        defender: Color,
        occ: Bitboard,
        _def_occ: Bitboard,
        def_idx: usize,
        pinned: &Bitboard,
    ) -> bool {
        let attacker = defender.opponent();

        // 歩
        if (attack::step_attacks(attacker, PieceType::Pawn, checker_sq)
            & self.piece_bb[def_idx][PieceType::Pawn as usize]
            & !*pinned)
            .is_not_empty()
        {
            return true;
        }
        // 桂
        if (attack::step_attacks(attacker, PieceType::Knight, checker_sq)
            & self.piece_bb[def_idx][PieceType::Knight as usize]
            & !*pinned)
            .is_not_empty()
        {
            return true;
        }
        // 銀
        if (attack::step_attacks(attacker, PieceType::Silver, checker_sq)
            & self.piece_bb[def_idx][PieceType::Silver as usize]
            & !*pinned)
            .is_not_empty()
        {
            return true;
        }
        // 金 + 成駒
        let gold_movers = (self.piece_bb[def_idx][PieceType::Gold as usize]
            | self.piece_bb[def_idx][PieceType::ProPawn as usize]
            | self.piece_bb[def_idx][PieceType::ProLance as usize]
            | self.piece_bb[def_idx][PieceType::ProKnight as usize]
            | self.piece_bb[def_idx][PieceType::ProSilver as usize])
            & !*pinned;
        if (attack::step_attacks(attacker, PieceType::Gold, checker_sq) & gold_movers)
            .is_not_empty()
        {
            return true;
        }
        // 馬・龍(ステップ部分) - 玉は除外(玉での取り返しは逃げ場チェックで処理済み)
        let step_pieces = (self.piece_bb[def_idx][PieceType::Horse as usize]
            | self.piece_bb[def_idx][PieceType::Dragon as usize])
            & !*pinned;
        let king_step = attack::step_attacks(attacker, PieceType::King, checker_sq);
        if (king_step & step_pieces).is_not_empty() {
            return true;
        }
        // 香
        if (attack::lance_attacks(attacker, checker_sq, occ)
            & self.piece_bb[def_idx][PieceType::Lance as usize]
            & !*pinned)
            .is_not_empty()
        {
            return true;
        }
        // 角・馬(スライド)
        if (attack::bishop_attacks(checker_sq, occ)
            & (self.piece_bb[def_idx][PieceType::Bishop as usize]
                | self.piece_bb[def_idx][PieceType::Horse as usize])
            & !*pinned)
            .is_not_empty()
        {
            return true;
        }
        // 飛・龍(スライド)
        if (attack::rook_attacks(checker_sq, occ)
            & (self.piece_bb[def_idx][PieceType::Rook as usize]
                | self.piece_bb[def_idx][PieceType::Dragon as usize])
            & !*pinned)
            .is_not_empty()
        {
            return true;
        }

        false
    }

    /// 飛び駒の王手に対して合い駒できるか(ビットボード判定)．
    ///
    /// between のマスに守備駒を移動または打てるか確認する．
    /// ピンされた駒は合い駒に使えない．
    #[allow(clippy::too_many_arguments)]
    fn can_interpose_bb(
        &self,
        between: &Bitboard,
        king_sq: Square,
        defender: Color,
        occ: Bitboard,
        _def_occ: Bitboard,
        def_idx: usize,
        _att_idx: usize,
        pinned: &Bitboard,
        _checker_from: Option<Square>,
    ) -> bool {
        let attacker = defender.opponent();

        for to in *between {
            // --- 駒打ちによる合い駒 ---
            if !occ.contains(to) {
                // 空きマスなら持ち駒を打てる可能性
                for (hand_idx, &pt) in PieceType::HAND_PIECES.iter().enumerate() {
                    if self.hand[defender.index()][hand_idx] == 0 {
                        continue;
                    }
                    // 行き所のない駒チェック
                    match (defender, pt) {
                        (Color::Black, PieceType::Pawn | PieceType::Lance) if to.row() == 0 => {
                            continue
                        }
                        (Color::White, PieceType::Pawn | PieceType::Lance) if to.row() == 8 => {
                            continue
                        }
                        (Color::Black, PieceType::Knight) if to.row() <= 1 => continue,
                        (Color::White, PieceType::Knight) if to.row() >= 7 => continue,
                        _ => {}
                    }
                    // 二歩チェック
                    if pt == PieceType::Pawn {
                        let our_pawns =
                            self.piece_bb[defender.index()][PieceType::Pawn as usize];
                        let file = Bitboard::file_mask(to.col());
                        if (our_pawns & file).is_not_empty() {
                            continue;
                        }
                    }
                    return true; // 合い駒が打てる
                }
            }

            // --- 駒移動による合い駒 ---
            // 各守備駒が to に移動できるか(ピンされていない駒のみ)
            // 歩
            if (attack::step_attacks(attacker, PieceType::Pawn, to)
                & self.piece_bb[def_idx][PieceType::Pawn as usize]
                & !*pinned)
                .is_not_empty()
            {
                return true;
            }
            // 桂
            if (attack::step_attacks(attacker, PieceType::Knight, to)
                & self.piece_bb[def_idx][PieceType::Knight as usize]
                & !*pinned)
                .is_not_empty()
            {
                return true;
            }
            // 銀
            if (attack::step_attacks(attacker, PieceType::Silver, to)
                & self.piece_bb[def_idx][PieceType::Silver as usize]
                & !*pinned)
                .is_not_empty()
            {
                return true;
            }
            // 金 + 成駒
            let gold_movers = (self.piece_bb[def_idx][PieceType::Gold as usize]
                | self.piece_bb[def_idx][PieceType::ProPawn as usize]
                | self.piece_bb[def_idx][PieceType::ProLance as usize]
                | self.piece_bb[def_idx][PieceType::ProKnight as usize]
                | self.piece_bb[def_idx][PieceType::ProSilver as usize])
                & !*pinned;
            if (attack::step_attacks(attacker, PieceType::Gold, to) & gold_movers).is_not_empty() {
                return true;
            }
            // 馬・龍(ステップ部分) - 玉は除外
            let step_pieces = (self.piece_bb[def_idx][PieceType::Horse as usize]
                | self.piece_bb[def_idx][PieceType::Dragon as usize])
                & !*pinned
                & !Bitboard::from_square(king_sq);
            let king_step_to = attack::step_attacks(attacker, PieceType::King, to);
            if (king_step_to & step_pieces).is_not_empty() {
                return true;
            }
            // 香
            if (attack::lance_attacks(attacker, to, occ)
                & self.piece_bb[def_idx][PieceType::Lance as usize]
                & !*pinned)
                .is_not_empty()
            {
                return true;
            }
            // 角・馬(スライド)
            if (attack::bishop_attacks(to, occ)
                & (self.piece_bb[def_idx][PieceType::Bishop as usize]
                    | self.piece_bb[def_idx][PieceType::Horse as usize])
                & !*pinned
                & !Bitboard::from_square(king_sq))
                .is_not_empty()
            {
                return true;
            }
            // 飛・龍(スライド)
            if (attack::rook_attacks(to, occ)
                & (self.piece_bb[def_idx][PieceType::Rook as usize]
                    | self.piece_bb[def_idx][PieceType::Dragon as usize])
                & !*pinned
                & !Bitboard::from_square(king_sq))
                .is_not_empty()
            {
                return true;
            }
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
        debug_assert!(!piece.is_empty());
        // Safety: piece が空でないことは debug_assert で検証済み
        let ci = unsafe { piece.color_index_unchecked() };
        let pt = unsafe { piece.piece_type_raw_unchecked() };
        self.squares[sq.index()] = piece;
        self.piece_bb[ci][pt as usize].set(sq);
        self.occupied[ci].set(sq);
        self.all_occ.set(sq);
        let z = ZOBRIST.board_hash_raw(ci, pt as usize, sq);
        self.hash ^= z;
        self.board_hash ^= z;
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
        // Safety: piece が空でないことは debug_assert で検証済み
        let ci = unsafe { piece.color_index_unchecked() };
        let pt = unsafe { piece.piece_type_raw_unchecked() };
        self.squares[sq.index()] = Piece::EMPTY;
        self.piece_bb[ci][pt as usize].clear(sq);
        self.occupied[ci].clear(sq);
        self.all_occ.clear(sq);
        let z = ZOBRIST.board_hash_raw(ci, pt as usize, sq);
        self.hash ^= z;
        self.board_hash ^= z;
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
            if self.hand[self.turn.index()][hi] > 0 {
                self.hash ^= ZOBRIST.hand_hash(self.turn, hi, self.hand[self.turn.index()][hi] as usize);
            }

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
                    if self.hand[self.turn.index()][hi] > 0 {
                        self.hash ^= ZOBRIST.hand_hash(self.turn, hi, self.hand[self.turn.index()][hi] as usize);
                    }
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
        let turn_z = ZOBRIST.turn_hash();
        self.hash ^= turn_z;
        self.board_hash ^= turn_z;
        self.turn = self.turn.opponent();
        self.ply += 1;

        captured
    }

    /// 手を取り消す．
    ///
    /// # 前提条件
    ///
    /// `do_move` で実行した手に対してのみ呼ぶこと．
    /// `do_move` を経ずに呼んだ場合はパニックする．
    #[inline]
    pub fn undo_move(&mut self, m: Move, captured: Piece) {
        // 手番を戻す
        self.turn = self.turn.opponent();
        let turn_z = ZOBRIST.turn_hash();
        self.hash ^= turn_z;
        self.board_hash ^= turn_z;
        debug_assert!(self.ply > 1, "undo_move called without prior do_move");
        self.ply -= 1;

        if m.is_drop() {
            let to = m.to_sq();
            let pt = m.drop_piece_type().unwrap();

            // 盤上から駒を除去
            self.remove_piece(to);

            // 持ち駒に戻す
            let hi = pt.hand_index().unwrap();
            if self.hand[self.turn.index()][hi] > 0 {
                self.hash ^= ZOBRIST.hand_hash(self.turn, hi, self.hand[self.turn.index()][hi] as usize);
            }
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
                    if self.hand[self.turn.index()][hi] > 0 {
                        self.hash ^= ZOBRIST.hand_hash(self.turn, hi, self.hand[self.turn.index()][hi] as usize);
                    }
                }
            }
        }
    }

    /// 盤面全体からZobrist hashを計算する(初期化用)．
    pub fn compute_hash(&self) -> u64 {
        let mut hash = self.compute_board_hash();

        // 持ち駒
        for color in [Color::Black, Color::White] {
            for kind in 0..7 {
                let count = self.hand[color.index()][kind];
                if count > 0 {
                    hash ^= ZOBRIST.hand_hash(color, kind, count as usize);
                }
            }
        }

        hash
    }

    /// 盤面のみのZobrist hashを計算する(持ち駒を除外，初期化用)．
    pub fn compute_board_hash(&self) -> u64 {
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

    /// 手番を設定し，Zobrist hashを再計算する(外部クレート向け)．
    pub fn set_turn(&mut self, color: Color) {
        self.turn = color;
        self.hash = self.compute_hash();
        self.board_hash = self.compute_board_hash();
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

        // Zobristハッシュの整合性
        if self.hash != self.compute_hash() {
            return false;
        }
        if self.board_hash != self.compute_board_hash() {
            return false;
        }
        if self.all_occ != (self.occupied[0] | self.occupied[1]) {
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

    /// 16-bit move から完全な 32-bit Move を生成する．
    ///
    /// 盤面の状態から captured piece と moving piece type を補完する．
    /// 不正な move16(移動元に駒がない等)の場合は `None` を返す．
    pub fn move_from_move16(&self, move16: u16) -> Option<Move> {
        self.complete_move(Move::from_move16(move16))
    }

    /// USI 文字列から完全な 32-bit Move を生成する．
    ///
    /// 盤面の状態から captured piece と moving piece type を補完する．
    pub fn move_from_usi(&self, usi: &str) -> Option<Move> {
        self.complete_move(Move::from_usi(usi)?)
    }

    /// 16-bit Move に盤面情報(移動駒種・取得駒)を補完して 32-bit Move を返す．
    fn complete_move(&self, m16: Move) -> Option<Move> {
        if m16.is_drop() {
            Some(m16)
        } else {
            let from = m16.from_sq();
            let to = m16.to_sq();
            let moving_piece = self.squares[from.index()];
            if moving_piece.is_empty() {
                return None;
            }
            let moving_pt = moving_piece.piece_type()? as u8;
            let captured = self.squares[to.index()].0;
            Some(Move::new_move(from, to, m16.is_promotion(), captured, moving_pt))
        }
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
