//! Df-Pn のデータ構造定義．

/// PNS アリーナの**デフォルト**最大ノード数(メモリ上限)．
///
/// 1ノード ≈ 80〜120 bytes(children Vec 含む)．
/// 5M ノードで約 400〜600 MB を使用する．
///
/// `DfPnSolver::param_pns_arena_max` で実行時に変更可能．
/// この定数はデフォルト値および初期 `Vec::with_capacity` の上限として使用する．
pub(super) const PNS_MAX_ARENA_NODES: usize = 5_000_000;
