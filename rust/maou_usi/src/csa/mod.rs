//! CSA サーバプロトコル transport (floodgate / 電竜戦本戦の接続方式)．
//!
//! 設計 docs/design/usi-engine/index.md §4 が想定していた「将来の CSA transport を
//! agent 無変更で追加する」の実装．[`crate::stdio`] と同じ層に位置し，
//! [`crate::agent::Agent`] を stdio なしで駆動する．
//!
//! - [`protocol`]: CSA 行 ⇔ 型付きメッセージの parse/serialize (pure．IO なし)
//! - [`client`]: TCP セッション (ログイン → 対局条件 → 指し手交換 → 終局) と
//!   連続対局のための再接続ループ
//!
//! 持ち時間の責務分界は USI 経路と同じ: transport がサーバ通知
//! (`Total_Time` / `Byoyomi` / `Increment`) から残り時間を追跡して
//! [`crate::protocol::ClockParams`] を組み立て，1 手の予算配分は
//! [`crate::time`] (TimeStrategy) が行う．

pub mod client;
pub mod protocol;

pub use client::{run_csa, CsaConfig, CsaGameResult, CsaSessionSummary};
pub use protocol::{GameSummary, ServerMessage, TimeRule};
