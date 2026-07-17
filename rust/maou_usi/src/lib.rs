//! maou_usi — USI (Universal Shogi Interface) 対局エージェント．
//!
//! 設計: docs/design/usi-engine/index.md．レイヤー構成:
//!
//! - [`protocol`]: USI 行 ⇔ 型付きコマンドの parse/serialize (pure．IO なし)
//! - [`agent`]: 対局エージェント = 状態機械 + 戦略 (transport 非依存)
//! - [`backend`]: [`agent::SearchBackend`] の実装 (maou_search)
//! - [`time`][]: 時間管理 (持ち時間 → 1 手予算の変換レイヤー)
//! - [`stdio`]: 標準入出力 transport (reader スレッド + dispatcher)
//!
//! プロトコル層とエージェントを分離しているのは，自己対局 driver (M4) が
//! agent を stdio なしで直接駆動するためと，将来の CSA transport を agent
//! 無変更で追加するため．

pub mod agent;
pub mod backend;
pub mod protocol;
pub mod stdio;
pub mod time;

pub use agent::{Agent, EngineConfig, SearchBackend, SearchBudget, SearchOutcome};
pub use backend::MaouSearchBackend;
pub use stdio::run_stdio;
pub use time::{TimeBudget, TimeStrategyConfig};
