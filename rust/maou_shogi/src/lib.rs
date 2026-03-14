pub(crate) mod attack;
pub(crate) mod bitboard;
pub mod board;
pub mod dfpn;
pub mod feature;
pub mod hcp;
pub mod movegen;
pub mod moves;
// piece は内部実装．cshogi互換の変換関数のみ提供し，
// 駒の生成・判定は types::Piece の公開APIを通じて行う．
pub(crate) mod piece;
pub mod position;
// sfen は pub(crate) だが，SfenError は board.rs 経由で
// `pub use crate::sfen::SfenError` により外部に公開している．
// SFEN パーサの内部実装を隠蔽しつつ，エラー型のみ公開する意図的な設計．
pub(crate) mod sfen;
pub mod types;
pub(crate) mod zobrist;
