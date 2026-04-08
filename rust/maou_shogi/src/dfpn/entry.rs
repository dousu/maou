//! Df-Pn のデータ構造定義．

use std::collections::VecDeque;

use crate::moves::Move;
use crate::types::HAND_KINDS;

/// remaining_flags のビット 15: path_dependent フラグ．
const PATH_DEPENDENT_BIT: u16 = 0x8000;
/// remaining_flags のビット 0-14: remaining 値のマスク．
const REMAINING_MASK: u16 = 0x7FFF;

/// 転置表のエントリ(証明駒/反証駒対応)．
///
/// - hand: 登録時の攻め方の持ち駒(証明駒/反証駒として使用)
/// - pn, dn: 証明数・反証数
/// - remaining_flags: ビット 0-14 = 残り探索深さ(`depth - ply`)，
///   ビット 15 = path_dependent フラグ(GHI 対策)．
///   `REMAINING_INFINITE`(0x7FFF) は深さ制限なし(真の証明/反証)を示す．
/// - mate_distance: 詰み手数 (proven entry のみ有効)．
///   pn=0 のとき，この局面から詰みまでの手数 (= longest resistance 下で
///   の残り plies) を保存する．PV 抽出時に AND ノードで再帰なしに
///   最長抵抗の child を選択するために使う．
///   非 proven entry では未使用 (0 のまま)．
///
/// v0.24.0: エントリ圧縮(source u64→u32, amount u16→u8,
/// path_dependent を remaining_flags に pack)で 32→24 bytes に削減．
///
/// v0.24.23: mate_distance(u16) を追加(24→28 bytes)．TTFlatEntry は
/// 32→40 bytes に拡大するが，PV 抽出で再帰なしの longest resistance 判定を
/// 可能にして visit budget を不要にする．
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub(super) struct DfPnEntry {
    /// SNDA (Kishimoto 2010) のソースノードハッシュ(上位 32 bit 切り捨て)．
    ///
    /// SNDA 衝突確率は 2^{-32} であり，TT キャッシュの用途では十分．
    pub(super) source: u32,
    pub(super) pn: u32,
    pub(super) dn: u32,
    pub(super) hand: [u8; HAND_KINDS],
    /// 探索投資量メトリック(KomoringHeights の amount\_ に相当)．
    /// GC / 置換時に大きい amount のエントリを優先保持する．
    /// u8 に圧縮(0-255)．PROOF_BONUS=100, DISPROOF_BONUS=50 が収まる．
    pub(super) amount: u8,
    /// ビット 0-14: remaining depth, ビット 15: path_dependent flag.
    remaining_flags: u16,
    /// TT Best Move: この局面で最も有望だった手の Move16 エンコーディング．
    pub(super) best_move: u16,
    /// 詰み手数 (proven entry でのみ有効)．pn=0 のとき，この局面から
    /// 詰みまでの残り plies を保存する．非 proven entry では 0．
    pub(super) mate_distance: u16,
}

// コンパイル時にサイズを検証(28 bytes == DfPnEntry with mate_distance added)
const _: () = assert!(
    std::mem::size_of::<DfPnEntry>() == 28,
    "DfPnEntry must be 28 bytes with mate_distance field"
);

impl DfPnEntry {
    /// ゼロ初期化されたエントリ(EMPTY 定数用)．
    pub(super) const ZERO: Self = DfPnEntry {
        source: 0,
        pn: 0,
        dn: 0,
        hand: [0; HAND_KINDS],
        amount: 0,
        remaining_flags: 0,
        best_move: 0,
        mate_distance: 0,
    };

    /// remaining 値を取得する(ビット 0-14)．
    #[inline(always)]
    pub(super) fn remaining(&self) -> u16 {
        self.remaining_flags & REMAINING_MASK
    }

    /// path_dependent フラグを取得する(ビット 15)．
    #[inline(always)]
    pub(super) fn path_dependent(&self) -> bool {
        self.remaining_flags & PATH_DEPENDENT_BIT != 0
    }

    /// remaining と path_dependent を同時にセットした remaining_flags を生成する．
    #[inline(always)]
    pub(super) fn encode_remaining_flags(remaining: u16, path_dependent: bool) -> u16 {
        debug_assert!(
            remaining <= REMAINING_MASK,
            "remaining value {remaining} exceeds 15-bit limit"
        );
        (remaining & REMAINING_MASK) | if path_dependent { PATH_DEPENDENT_BIT } else { 0 }
    }

    /// 新しいエントリを構築する．
    #[inline(always)]
    pub(super) fn new(
        source: u32,
        pn: u32,
        dn: u32,
        hand: [u8; HAND_KINDS],
        remaining: u16,
        path_dependent: bool,
        best_move: u16,
        amount: u8,
    ) -> Self {
        Self {
            source,
            pn,
            dn,
            hand,
            amount,
            remaining_flags: Self::encode_remaining_flags(remaining, path_dependent),
            best_move,
            mate_distance: 0,
        }
    }

    /// 詰み手数付きで新しいエントリを構築する (proven entry 用)．
    #[inline(always)]
    pub(super) fn new_with_distance(
        source: u32,
        pn: u32,
        dn: u32,
        hand: [u8; HAND_KINDS],
        remaining: u16,
        path_dependent: bool,
        best_move: u16,
        amount: u8,
        mate_distance: u16,
    ) -> Self {
        Self {
            source,
            pn,
            dn,
            hand,
            amount,
            remaining_flags: Self::encode_remaining_flags(remaining, path_dependent),
            best_move,
            mate_distance,
        }
    }

    /// remaining 値を更新する(path_dependent は維持)．
    #[inline(always)]
    pub(super) fn set_remaining(&mut self, remaining: u16) {
        self.remaining_flags =
            (remaining & REMAINING_MASK) | (self.remaining_flags & PATH_DEPENDENT_BIT);
    }

    /// path_dependent フラグをクリアする(remaining は維持)．
    #[inline(always)]
    pub(super) fn clear_path_dependent(&mut self) {
        self.remaining_flags &= REMAINING_MASK;
    }
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
