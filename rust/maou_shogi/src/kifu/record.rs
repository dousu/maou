//! 棋譜レコード — CSA/KIF パーサ共通の出力表現．
//!
//! `moves` は cshogi 互換の 32-bit エンコーディングで保持する:
//! - bits 0-6: to (0-80)
//! - bits 7-13: from (0-80，通常手) / 81 + drop_move_index (駒打ち)
//! - bit 14: promotion flag
//! - bits 16-19: 移動前の駒種 (`PieceType as u8`，駒打ちは 0)
//! - bits 20-23: 捕獲した駒種 (`PieceType as u8`，捕獲なしは 0)
//!
//! これは [`crate::moves::Move`] の内部表現 (上位ビット配置が異なる) では
//! **ない**ことに注意．既存パイプライン (HCPE 変換) が cshogi パーサの
//! 出力 int をそのまま扱ってきたため，パーサ出力はこの配置を維持する．

use crate::types::{PieceType, Square};

/// 対局結果: 引き分け (cshogi DRAW 互換)．
pub const WIN_DRAW: u8 = 0;
/// 対局結果: 先手勝ち (cshogi BLACK_WIN 互換)．
pub const WIN_BLACK: u8 = 1;
/// 対局結果: 後手勝ち (cshogi WHITE_WIN 互換)．
pub const WIN_WHITE: u8 = 2;

/// パースした 1 棋譜．
#[derive(Debug, Clone, Default)]
pub struct GameRecord {
    /// バージョン行 (CSA "V2.2" 等．KIF は空文字列)．
    pub version: String,
    /// メタ情報 (CSA `$KEY:value` / KIF ヘッダ行) — 出現順．
    pub var_info: Vec<(String, String)>,
    /// 対局者名 [先手/下手, 後手/上手]．
    /// CSA は未指定でも `Some("")` (cshogi 互換)，KIF は未指定なら `None`．
    pub names: [Option<String>; 2],
    /// レーティング [先手, 後手]．
    /// CSA `'black_rate:` / `'white_rate:` コメント行由来 (KIF は常に 0.0)．
    pub ratings: [f32; 2],
    /// 初期局面の SFEN．
    pub sfen: String,
    /// 指し手 (cshogi 互換 32-bit エンコーディング，モジュール doc 参照)．
    pub moves: Vec<u32>,
    /// 消費時間 (秒)．
    /// cshogi 互換の仕様により `moves` と長さが一致しないことがある
    /// (終局後の `T` 行や KIF 投了行の時間が追加される)．
    pub times: Vec<i32>,
    /// 評価値 (CSA `'** <score>` コメント由来)．`moves` と同長 (既定 0)．
    pub scores: Vec<i32>,
    /// 指し手コメント．`moves` と同長 (コメントの無い手は空文字列)．
    pub comments: Vec<String>,
    /// 指し手より前のコメント (CSA `'` 行 / KIF `*` 行のゲームコメント)．
    pub header_comment: String,
    /// 終局状態 ("%TORYO" 等)．未終局なら `None`．
    pub endgame: Option<String>,
    /// 勝敗 ([`WIN_DRAW`] / [`WIN_BLACK`] / [`WIN_WHITE`])．
    /// KIF で「まで」行が無い場合は `None` (CSA は常に `Some`)．
    pub win: Option<u8>,
}

/// 棋譜パースエラー．
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("line {line_no}: {message}")]
pub struct KifuParseError {
    /// 1-origin の行番号 (入力文字列内)．
    pub line_no: usize,
    /// エラー内容．
    pub message: String,
}

impl KifuParseError {
    pub(crate) fn new(line_no: usize, message: impl Into<String>) -> Self {
        KifuParseError {
            line_no,
            message: message.into(),
        }
    }
}

/// 盤上の駒を動かす手を cshogi 互換 32-bit にエンコードする．
pub(crate) fn encode_move(
    from: Square,
    to: Square,
    promote: bool,
    moving_pt: PieceType,
    captured_pt: Option<PieceType>,
) -> u32 {
    let mut v = (to.raw_u8() as u32) | ((from.raw_u8() as u32) << 7);
    if promote {
        v |= 1 << 14;
    }
    v |= (moving_pt as u32) << 16;
    if let Some(cap) = captured_pt {
        v |= (cap as u32) << 20;
    }
    v
}

/// 駒打ちを cshogi 互換 32-bit にエンコードする．
///
/// 打てない駒種 (玉・成駒) は `None`．
pub(crate) fn encode_drop(to: Square, pt: PieceType) -> Option<u32> {
    let idx = pt.drop_move_index()?;
    Some((to.raw_u8() as u32) | ((81 + idx) << 7))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_move_cshogi_layout() {
        // +7776FU (cshogi 実測 0x00011e3b)．
        // CSA "7776" = 7七 → 7六．Square::new(col=file-1, row=rank-1):
        // 7七 = new(6,6) = 60，7六 = new(6,5) = 59．
        let v = encode_move(
            Square::new(6, 6),
            Square::new(6, 5),
            false,
            PieceType::Pawn,
            None,
        );
        assert_eq!(v, 0x00011e3b);
    }

    #[test]
    fn test_encode_capture_promote() {
        // 8h2b+ (実測 0x0055630a): from=70? 8八 = new(7,7) = 70，2二 = new(1,1) = 10
        let v = encode_move(
            Square::new(7, 7),
            Square::new(1, 1),
            true,
            PieceType::Bishop,
            Some(PieceType::Bishop),
        );
        assert_eq!(v, 0x0055630a);
    }

    #[test]
    fn test_encode_drop_cshogi_layout() {
        // B*9e (実測 0x00002acc): 9五 = new(8,4) = 76，角 drop_move_index=4
        let v = encode_drop(Square::new(8, 4), PieceType::Bishop).unwrap();
        assert_eq!(v, 0x00002acc);
        assert!(encode_drop(Square::new(0, 0), PieceType::King).is_none());
        assert!(encode_drop(Square::new(0, 0), PieceType::Horse).is_none());
    }
}
