//! maou_search — 1局面探索エンジン (MCTS)．
//!
//! 与えられた局面に対して PUCT ベースのモンテカルロ木探索を行い，
//! 最有力手と評価値 (手番側勝率) を返す．
//!
//! # 設計方針
//!
//! - **pure Rust**: PyO3 に依存しない．Python への露出は maou_rust 側の責務．
//! - **評価器の抽象化**: NN 推論は [`Evaluator`] trait の背後に置く．
//!   本 crate は [`MockEvaluator`] (決定論的擬似乱数) のみを提供し，
//!   ONNX 等の実推論は別実装として後から差し込む．
//! - **バッチ収集 first-class**: GPU バッチ推論が最大のボトルネックになる想定のため，
//!   各探索スレッドは virtual loss (前置 visits) を使って葉をバッチ単位で収集し，
//!   [`Evaluator::evaluate_batch`] にまとめて渡す．バッチ充填率と探索効率の
//!   トレードオフは [`SearchStats`] (collisions / avg batch) で観測できる．
//! - **固定容量ノードプール**: メモリ使用量はノードプール容量で上限を張る．
//!   容量到達時は stop-the-world GC で低訪問サブツリーを刈り取って継続する
//!   (無効化可．[`SearchOptions`] の `gc_enabled` / `gc_keep_ratio`)．
//!
//! # 現時点の制限
//!
//! - 千日手・連続王手の千日手は未考慮 (最大深さ到達で引き分け 0.5 と扱う)．
//! - 勝敗確定ノードの AND-OR 伝播，詰み探索 (dfpn) 統合は未実装．

pub mod evaluator;
pub mod search;
pub mod tree;

pub use evaluator::{EvalItem, EvalResult, Evaluator, MockEvaluator};
pub use search::{
    RootChildStat, SearchLimits, SearchOptions, SearchResult, SearchStats, Searcher, StopCause,
};
