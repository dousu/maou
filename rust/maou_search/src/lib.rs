//! maou_search — 1局面探索エンジン (MCTS)．
//!
//! 与えられた局面に対して PUCT ベースのモンテカルロ木探索を行い，
//! 最有力手と評価値 (手番側勝率) を返す．
//!
//! # 設計方針
//!
//! - **pure Rust**: PyO3 に依存しない．Python への露出は maou_rust 側の責務．
//! - **評価器の抽象化**: NN 推論は [`Evaluator`] trait の背後に置く．
//!   デフォルトビルドは [`MockEvaluator`] (決定論的擬似乱数) のみで pure Rust を
//!   維持し，ONNX Runtime による実推論 (`onnx::OnnxEvaluator`) は `onnx`
//!   feature (CUDA は `onnx-cuda`) で opt-in する (wheel 可搬性の維持)．
//! - **バッチ収集 first-class**: GPU バッチ推論が最大のボトルネックになる想定のため，
//!   各探索スレッドは virtual loss (前置 visits) を使って葉をバッチ単位で収集し，
//!   [`Evaluator::evaluate_batch`] にまとめて渡す．バッチ充填率と探索効率の
//!   トレードオフは [`SearchStats`] (collisions / avg batch) で観測できる．
//! - **固定容量ノードプール**: メモリ使用量はノードプール容量で上限を張る．
//!   容量到達時は stop-the-world GC で低訪問サブツリーを刈り取って継続する
//!   (無効化可．[`SearchOptions`] の `gc_enabled` / `gc_keep_ratio`)．
//!
//! - **千日手検出**: 対局履歴 + 探索経路のハッシュ後方走査で同一局面の
//!   再出現を検出し，連続王手の千日手は王手をかけ続けた側の負けとして
//!   終端評価する ([`repetition`]．語彙・判定は maou_shogi の
//!   `Position::is_perpetual_check_move` / dfpn の on-path 検出に揃えている)．
//! - **AND-OR 勝敗確定伝播**: 詰み/千日手で確定した葉の値を祖先へ連鎖的に
//!   昇格させ，確定ノードは再評価せず固定値で短絡する．root が確定したら
//!   探索を早期停止する．詰み探索 (dfpn) 統合時の結果注入口を兼ねる．
//! - **ルート並行詰み探索**: [`SearchOptions`] の `root_dfpn` で dfpn
//!   (maou_shogi) を専用スレッドで並行実行し，詰みが証明されたら即停止して
//!   詰み手順を best_move / PV として返す (既定で有効)．
//! - **葉の非同期詰み探索**: `leaf_mate` で MCTS が展開した葉の短手詰みを
//!   専用 mate スレッドが df-pn で証明し，AND-OR 伝播へ注入する
//!   (既定で有効．探索スレッドは依頼を積むだけで NPS に影響しない)．
//!
//! # 現時点の制限
//!
//! - 優越局面 (盤面同一で持駒優越/劣位) による千日手の一般化は未実装．

pub mod evaluator;
pub mod feature;
pub mod label;
#[cfg(feature = "onnx")]
pub mod onnx;
pub mod position;
pub mod preprocess;
pub mod repetition;
pub mod search;
pub mod tree;

pub use evaluator::{EvalItem, EvalResult, Evaluator, MockEvaluator};
#[cfg(feature = "onnx")]
pub use onnx::OnnxEvaluator;
pub use position::{build_board_and_history, PositionSetupError};
pub use repetition::{HistoryEntry, RepetitionOutcome};
pub use search::{
    ReusableTree, RootChildStat, RootSnapshot, SearchLimits, SearchOptions, SearchResult,
    SearchStats, Searcher, StopCause,
};
