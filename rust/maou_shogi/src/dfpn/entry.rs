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
/// - amount: dual-purpose フィールド (案 B):
///   - **proven entry (pn=0)**: mate_distance (詰み手数, u8 saturated 0-255)
///     を格納．PV 抽出時の longest resistance 判定に使用．
///     ※ 元々の hint_ply ベース priority は失う代わりに，より長い詰み筋
///     (distance 大) を eviction で優先保持する形になる．
///   - **non-proven entry**: 探索投資量メトリック(KomoringHeights の amount\_)．
///     GC / 置換時に大きい amount のエントリを優先保持する．
///
/// v0.24.0: エントリ圧縮(source u64→u32, amount u16→u8,
/// path_dependent を remaining_flags に pack)で 32→24 bytes に削減．
///
/// v0.24.24: amount を proven entry 用 mate_distance として再利用．
/// TT entry サイズは 24 bytes に維持しつつ PV 抽出 fast path を実現．
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
    /// dual-purpose: proven entry では mate_distance, non-proven では
    /// 探索投資量．アクセスは `proven_distance()` / `working_amount()` を
    /// 経由する．
    pub(super) amount: u8,
    /// ビット 0-14: remaining depth, ビット 15: path_dependent flag.
    remaining_flags: u16,
    /// TT Best Move: この局面で最も有望だった手の Move16 エンコーディング．
    pub(super) best_move: u16,
}

// コンパイル時にサイズを検証(TTFlatEntry が 32 bytes になることを保証)
const _: () = assert!(
    std::mem::size_of::<DfPnEntry>() == 24,
    "DfPnEntry must be 24 bytes for 32-byte TTFlatEntry"
);

/// 案 D: ProvenTT 専用のエントリ構造体．
///
/// proven entry (proof / confirmed disproof) は以下のフィールドが不要:
/// - `source`: SNDA は active search 用 (proven は不要)
/// - `pn`: proof は常に 0, disproof は常に > 0 → 1 bit で表現可
/// - `dn`: proof は常に INF, disproof は常に 0 → 1 bit で表現可
/// - `amount`: 案 B で mate_distance に置き換え済
///
/// 必要なフィールド:
/// - `hand`: hand_gte_forward_chain 比較
/// - `proof_flag`: proof or confirmed disproof
/// - `mate_distance`: PV 抽出の longest resistance 判定
/// - `best_move`: PV 表示用 attacker move
/// - `remaining`: confirmed disproof の depth-limited 判定 (proof は常に INFINITE)
///
/// レイアウト (12 bytes):
/// ```
/// hand: [u8; 7]       (7 bytes, offset 0)
/// flags: u8           (1 byte, offset 7)
///   bit 0: is_proof (1) or confirmed_disproof (0)
///   bit 7: distance_set (mate_distance is valid)
///   bits 1-6: mate_distance (0-63 plies)
/// best_move: u16      (2 bytes, offset 8)
/// remaining: u16      (2 bytes, offset 10)
/// ```
/// 合計 12 bytes (alignment は u16 = 2)．
///
/// `mate_distance` が 0-63 に制限されることに注意 (案 B の 0-127 より狭い)．
/// 64+ プライの詰みは distance unknown 扱いで slow path にフォールバック．
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub(super) struct ProvenEntry {
    pub(super) hand: [u8; HAND_KINDS],
    /// bit 0: is_proof, bit 7: distance_set, bits 1-6: distance (0-63)
    pub(super) flags: u8,
    pub(super) best_move: u16,
    pub(super) remaining: u16,
}

const _: () = assert!(
    std::mem::size_of::<ProvenEntry>() == 12,
    "ProvenEntry must be 12 bytes"
);

impl ProvenEntry {
    pub(super) const ZERO: Self = ProvenEntry {
        hand: [0; HAND_KINDS],
        flags: 0,
        best_move: 0,
        remaining: 0,
    };

    /// proof エントリかどうか (bit 0)．
    #[inline(always)]
    pub(super) fn is_proof(&self) -> bool {
        (self.flags & 0x01) != 0
    }

    /// mate_distance を取得する (proof + distance_set のときのみ Some)．
    /// bits 1-6 に格納された 0-63 の distance を返す．
    #[inline(always)]
    pub(super) fn mate_distance(&self) -> Option<u16> {
        if self.is_proof() && (self.flags & 0x80) != 0 {
            Some(((self.flags >> 1) & 0x3F) as u16)
        } else {
            None
        }
    }

    /// proof エントリ用 flags を encode する．
    /// distance > 63 は distance_set フラグなしで encode (None 扱い)．
    #[inline(always)]
    pub(super) fn encode_proof_flags(distance: u16) -> u8 {
        let bit0 = 0x01u8; // is_proof
        if distance > 0 && distance <= 63 {
            0x80 | (((distance & 0x3F) as u8) << 1) | bit0
        } else {
            bit0
        }
    }

    /// confirmed disproof エントリ用 flags．
    #[inline(always)]
    pub(super) fn encode_disproof_flags() -> u8 {
        0x00 // is_proof=0, no distance
    }
}

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
    };

    /// proven entry の mate_distance を取得する．
    ///
    /// 案 B のエンコーディング: amount フィールドは dual-purpose:
    /// - **bit 7 = 1**: distance-aware store (bits 0-6 = mate_distance, 0-127)
    /// - **bit 7 = 0**: legacy store (priority value, distance unknown)
    ///
    /// pn=0 で bit 7 がセットされていれば distance を返し，それ以外は None．
    #[inline(always)]
    pub(super) fn proven_distance(&self) -> Option<u16> {
        if self.pn == 0 && (self.amount & 0x80) != 0 {
            Some((self.amount & 0x7F) as u16)
        } else {
            None
        }
    }

    /// distance を amount フィールドに encode する (bit 7 をセット)．
    /// distance > 127 は 127 で saturate する．
    #[inline(always)]
    pub(super) fn encode_distance_amount(distance: u16) -> u8 {
        0x80 | (distance.min(127) as u8)
    }

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
        }
    }

    /// 詰み手数付きで新しいエントリを構築する (proven entry 用)．
    /// 案 B: amount フィールドに mate_distance を bit 7 set 形式で格納する．
    #[inline(always)]
    pub(super) fn new_with_distance(
        source: u32,
        pn: u32,
        dn: u32,
        hand: [u8; HAND_KINDS],
        remaining: u16,
        path_dependent: bool,
        best_move: u16,
        _amount: u8,
        mate_distance: u16,
    ) -> Self {
        // proven entry: amount に distance を bit 7 set 形式で encode
        // non-proven: amount は通常通り (mate_distance パラメータは無視)
        let amount = if pn == 0 && mate_distance > 0 {
            Self::encode_distance_amount(mate_distance)
        } else {
            _amount
        };
        Self {
            source,
            pn,
            dn,
            hand,
            amount,
            remaining_flags: Self::encode_remaining_flags(remaining, path_dependent),
            best_move,
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
