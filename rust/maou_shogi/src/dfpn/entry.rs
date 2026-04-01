//! Df-Pn のデータ構造定義．

use std::collections::VecDeque;

use crate::moves::Move;
use crate::types::HAND_KINDS;


/// 転置表のエントリ(証明駒/反証駒対応)．
///
/// - hand: 登録時の攻め方の持ち駒(証明駒/反証駒として使用)
/// - pn, dn: 証明数・反証数
/// - remaining: 登録時の残り探索深さ(`depth - ply`)．
///   `REMAINING_INFINITE` は深さ制限なし(真の証明/反証)を示す．
///   深さ制限による不詰(`dn=0`)は `remaining` が有限値となり，
///   より深い探索で再評価可能になる．
/// フィールド配置は `source`(u64) を先頭にし 8-byte アライメントの
/// パディングを最小化する．
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub(super) struct DfPnEntry {
    /// SNDA (Kishimoto 2010) のソースノードハッシュ．
    pub(super) source: u64,
    pub(super) pn: u32,
    pub(super) dn: u32,
    pub(super) hand: [u8; HAND_KINDS],
    /// GHI (Graph History Interaction) 対策フラグ．
    pub(super) path_dependent: bool,
    pub(super) remaining: u16,
    /// TT Best Move: この局面で最も有望だった手の Move16 エンコーディング．
    pub(super) best_move: u16,
    /// 探索投資量メトリック(KomoringHeights の amount\_ に相当)．
    /// GC / 置換時に大きい amount のエントリを優先保持する．
    pub(super) amount: u16,
}

/// Best-First Proof Number Search (PNS) のノード．
///
/// PNS では探索木を明示的にメモリ上に保持し，
/// 常に最も有望なリーフ(most-proving node)を展開する．
/// df-pn の閾値ベースの深さ優先探索と異なり，
/// グローバルに最適なノード選択を行うため thrashing が発生しない．
///
/// 参考: Allis (1994), Seo, Iida & Uiterwijk (2001, PN*)
pub(super) struct PnsNode {
    /// 盤面ハッシュ(持ち駒除外)．TT キー．
    pub(super) pos_key: u64,
    /// 完全ハッシュ(持ち駒込み)．ループ検出用．
    pub(super) full_hash: u64,
    /// 攻め方の持ち駒．
    pub(super) hand: [u8; HAND_KINDS],
    /// 証明数．
    pub(super) pn: u32,
    /// 反証数．
    pub(super) dn: u32,
    /// 親ノードのインデックス(`u32::MAX` = ルート)．
    pub(super) parent: u32,
    /// 親から到達する手．
    pub(super) move_from_parent: Move,
    /// OR ノード(攻め方手番)か AND ノード(玉方手番)か．
    pub(super) or_node: bool,
    /// 展開済みフラグ．
    pub(super) expanded: bool,
    /// 子ノードのインデックス(アリーナ内)．
    pub(super) children: Vec<u32>,
    /// 最善子ノードのキャッシュ(backup 時に更新)．
    /// OR ノード: min(pn, dn) の子，AND ノード: min(dn, pn) の子．
    pub(super) cached_best: u32,
    /// 残り探索深さ．
    pub(super) remaining: u16,
    /// AND ノード用: 逐次活性化待ちの合駒(drop)手．
    /// 弱い駒から順に1つずつ子ノードとして展開する．
    pub(super) deferred_drops: VecDeque<Move>,
}

/// PNS アリーナの最大ノード数(メモリ上限)．
///
/// 1ノード ≈ 80〜120 bytes(children Vec 含む)．
/// 5M ノードで約 400〜600 MB を使用する．
pub(super) const PNS_MAX_ARENA_NODES: usize = 5_000_000;

