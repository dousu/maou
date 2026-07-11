//! 棋譜フォーマット (CSA / KIF) パーサ．
//!
//! cshogi 依存を除去するための完全独自実装．出力の `moves` は既存
//! パイプライン互換のため cshogi 互換 32-bit エンコーディングを用いる
//! (詳細は [`record`] のモジュール doc)．
//!
//! parity 検証: `tests/kifu_parity.rs` が cshogi (oracle) の出力から
//! 生成した golden fixtures (`tests/fixtures/kifu/`) と照合する．

pub mod csa;
pub mod kif;
pub mod record;

pub use csa::{parse_csa_multi, parse_csa_str};
pub use kif::parse_kif_str;
pub use record::{GameRecord, KifuParseError, WIN_BLACK, WIN_DRAW, WIN_WHITE};
