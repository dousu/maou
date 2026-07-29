//! maou_usi — USI (Universal Shogi Interface) 対局エージェント．
//!
//! 設計: docs/design/usi-engine/index.md．レイヤー構成:
//!
//! - [`protocol`]: USI 行 ⇔ 型付きコマンドの parse/serialize (pure．IO なし)
//! - [`agent`]: 対局エージェント = 状態機械 + 戦略 (transport 非依存)
//! - [`backend`]: [`agent::SearchBackend`] の実装 (maou_search)
//! - [`time`][]: 時間管理 (持ち時間 → 1 手予算の変換レイヤー)
//! - [`stdio`]: 標準入出力 transport (reader スレッド + dispatcher)
//! - [`ab`]: 自己対局 A/B 計測 (レバー割り当て + 結果集計．pure)
//!
//! プロトコル層とエージェントを分離しているのは，自己対局 driver (M4) が
//! agent を stdio なしで直接駆動するためと，将来の CSA transport を agent
//! 無変更で追加するため．

pub mod ab;
pub mod agent;
pub mod backend;
pub mod csa;
pub mod protocol;
pub mod selfplay;
pub mod stdio;
pub mod time;

pub use ab::{build_ab, summarize, AbMode, AbOptions, AbSetup, AbSummary, RunSummary};
pub use agent::{Agent, EngineConfig, SearchBackend, SearchBudget, SearchOutcome};
pub use backend::MaouSearchBackend;
pub use selfplay::{run_selfplay, GameEndReason, GameOutcome, SelfplayConfig};
pub use stdio::run_stdio;
pub use time::{TimeBudget, TimeCurve, TimeStrategyConfig};
