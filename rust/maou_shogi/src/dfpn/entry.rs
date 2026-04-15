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
/// - amount: 探索投資量メトリック(KomoringHeights の amount\_)．
///   GC / 置換時に大きい amount のエントリを優先保持する．
///
/// v0.24.0: エントリ圧縮(source u64→u32, amount u16→u8,
/// path_dependent を remaining_flags に pack)で 32→24 bytes に削減．
///
/// v0.24.26: 案 D (Plan D) で proven entry は `ProvenEntry` に分離．
/// DfPnEntry は WorkingTT 専用(intermediate + depth-limited disproof)．
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
    /// 探索投資量メトリック．GC / 置換で保持優先度として使う．
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
/// - `flags`: proof / confirmed disproof の種別 + mate_distance / disproof depth
/// - `best_move`: PV 表示用 attacker move
/// - `meta`: proof tag + tag_depth (施策 X, v0.24.53+)
///
/// レイアウト (12 bytes):
/// ```text
/// hand: [u8; 7]       (7 bytes, offset 0)
/// flags: u8           (1 byte, offset 7)
///   bit 0: is_proof (1) or confirmed_disproof (0)
///   bit 7: distance_set (mate_distance is valid)
///   bits 1-6: mate_distance (0-63 plies) OR disproof confirmed depth
/// best_move: u16      (2 bytes, offset 8)
/// meta: u16           (2 bytes, offset 10)
///   proof 時 (is_proof=1):
///     bits 0-3:   proof_tag (0-15; 0=ABSOLUTE, 1=FILTER_DEPENDENT,
///                            2=PROVISIONAL, 3=DEPTH_LIMITED, 15=INVALID)
///     bits 4-9:   tag_depth (0-63; tag が設定された IDS depth)
///     bits 10-15: reserved
///   disproof 時 (is_proof=0):
///     常に 0 (disproof の depth 情報は flags bits 1-6 に格納)
/// ```
/// 合計 12 bytes (alignment は u16 = 2)．
///
/// `mate_distance` が 0-63 に制限されることに注意 (案 B の 0-127 より狭い)．
/// 64+ プライの詰みは distance unknown 扱いで slow path にフォールバック．
///
/// # proof_tag の役割 (v0.24.53+)
///
/// ProvenTT は永続エントリのストアであり従来 `REMAINING_INFINITE` 不変条件
/// (depth 非依存) を前提としていた．施策 α/A-5/A-6 は全てこの不変条件と
/// 矛盾し汚染を引き起こした．proof_tag は以下を可能にする:
///
/// - **ABSOLUTE (0)**: 完全 df-pn proof．既存の永続 proof と同じ
/// - **FILTER_DEPENDENT (1)**: 施策 α 等の heuristic filter 下で生成．
///   IDS depth 境界で選択除去される
/// - **PROVISIONAL (2)**: PNS 局所 resolver / 境界 sub-solver 結果．
///   Frontier サイクル境界で除去される
/// - **DEPTH_LIMITED (3)**: 予約 (特定 depth でのみ valid な proof)
/// - **INVALID (15)**: sentinel (GC で除去)
///
/// non-ABSOLUTE proof は `clear_proven_disproofs_below(min_depth)` で
/// tag_depth < min_depth の場合に除去される (confirmed disproof と同じ
/// 選択的除去メカニズム．v0.24.53 で統合)．
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub(super) struct ProvenEntry {
    pub(super) hand: [u8; HAND_KINDS],
    /// bit 0: is_proof, bit 7: distance_set, bits 1-6: distance (0-63)
    /// or disproof_depth (0-63).
    pub(super) flags: u8,
    pub(super) best_move: u16,
    /// proof 時: tag + tag_depth + reserved を pack．
    /// disproof 時: 常に 0 (disproof は flags bits 1-6 に depth 情報を持つ)．
    pub(super) meta: u16,
}

/// proof_tag の定数定義．`ProvenEntry::proof_tag()` が返す値．
pub(super) const PROOF_TAG_ABSOLUTE: u8 = 0;
pub(super) const PROOF_TAG_FILTER_DEPENDENT: u8 = 1;
pub(super) const PROOF_TAG_PROVISIONAL: u8 = 2;
#[allow(dead_code)]
pub(super) const PROOF_TAG_DEPTH_LIMITED: u8 = 3;
#[allow(dead_code)]
pub(super) const PROOF_TAG_INVALID: u8 = 15;

/// proof_tag を meta に pack するマスク．
const PROOF_TAG_MASK: u16 = 0x000F;
/// tag_depth を meta に pack するマスク (シフト後)．
const TAG_DEPTH_MASK: u16 = 0x003F;
/// tag_depth のビット位置．
const TAG_DEPTH_SHIFT: u16 = 4;

const _: () = assert!(
    std::mem::size_of::<ProvenEntry>() == 12,
    "ProvenEntry must be 12 bytes"
);

impl ProvenEntry {
    pub(super) const ZERO: Self = ProvenEntry {
        hand: [0; HAND_KINDS],
        flags: 0,
        best_move: 0,
        meta: 0,
    };

    /// proof エントリかどうか (bit 0)．
    #[inline(always)]
    pub(super) fn is_proof(&self) -> bool {
        (self.flags & 0x01) != 0
    }

    /// pn synthesized value: proof=0, disproof=INF.
    #[inline(always)]
    pub(super) fn pn(&self) -> u32 {
        if self.is_proof() { 0 } else { u32::MAX }
    }

    /// dn synthesized value: proof=INF, disproof=0.
    #[inline(always)]
    pub(super) fn dn(&self) -> u32 {
        if self.is_proof() { u32::MAX } else { 0 }
    }

    /// source synthesized value: proven entries don't use SNDA → 0.
    #[inline(always)]
    pub(super) fn source(&self) -> u32 {
        0
    }

    /// proof_tag を取得する (v0.24.53+)．
    ///
    /// - proof 時: meta bits 0-3 (ABSOLUTE/FILTER_DEPENDENT/PROVISIONAL/...)
    /// - disproof 時: 常に ABSOLUTE 扱い (disproof は tag 非対応)
    #[inline(always)]
    pub(super) fn proof_tag(&self) -> u8 {
        if self.is_proof() {
            (self.meta & PROOF_TAG_MASK) as u8
        } else {
            PROOF_TAG_ABSOLUTE
        }
    }

    /// tag_depth を取得する (v0.24.53+)．
    ///
    /// - proof 時: meta bits 4-9 (tag が設定された IDS depth, 0-63)
    /// - disproof 時: 0 (disproof は flags 経由の disproof_depth() を使う)
    #[inline(always)]
    pub(super) fn tag_depth(&self) -> u32 {
        if self.is_proof() {
            ((self.meta >> TAG_DEPTH_SHIFT) & TAG_DEPTH_MASK) as u32
        } else {
            0
        }
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

    /// エントリの eviction priority (高いほど保持優先)．
    ///
    /// - ABSOLUTE proof with distance: `0x80 | min(distance, 63)` → 128..191
    ///   (distance が大きい = 長い詰み筋ほど高 priority)
    /// - ABSOLUTE proof without distance: 64 (mid)
    /// - non-ABSOLUTE proof (FILTER_DEPENDENT/PROVISIONAL): 48 (mid-low)
    /// - confirmed disproof: 32 (low)
    ///
    /// non-ABSOLUTE proof は ABSOLUTE proof より低い priority で eviction
    /// 対象になりやすい．tag 付き proof は一時的であり失っても再計算可能．
    ///
    /// `DfPnEntry::amount` と意味が異なることに注意(DfPnEntry は ply ベース)．
    /// Plan D では proven entries の eviction は距離/種別ベース．
    #[inline(always)]
    pub(super) fn amount(&self) -> u8 {
        if let Some(d) = self.mate_distance() {
            0x80 | (d.min(63) as u8)
        } else if self.is_proof() {
            if self.proof_tag() == PROOF_TAG_ABSOLUTE {
                64
            } else {
                48
            }
        } else {
            32
        }
    }

    /// ABSOLUTE proof エントリを構築する (default tag)．
    ///
    /// 既存コードはこの経路を使用する．tag=ABSOLUTE, tag_depth=0．
    #[inline(always)]
    pub(super) fn new_proof(
        hand: [u8; HAND_KINDS],
        best_move: u16,
        mate_distance: u16,
    ) -> Self {
        Self::new_tagged_proof(hand, best_move, mate_distance, PROOF_TAG_ABSOLUTE, 0)
    }

    /// tag 付き proof エントリを構築する (v0.24.53+)．
    ///
    /// - `tag`: PROOF_TAG_* 定数のいずれか (0-15)
    /// - `tag_depth`: tag が設定された IDS depth (0-63，63 超過は clamp)
    ///
    /// non-ABSOLUTE tag を指定すると，後続 IDS depth 境界で
    /// `clear_proven_disproofs_below(min_depth)` により選択除去される
    /// (tag_depth < min_depth の場合)．
    #[inline(always)]
    pub(super) fn new_tagged_proof(
        hand: [u8; HAND_KINDS],
        best_move: u16,
        mate_distance: u16,
        tag: u8,
        tag_depth: u32,
    ) -> Self {
        let meta = (tag as u16 & PROOF_TAG_MASK)
            | ((tag_depth.min(63) as u16) << TAG_DEPTH_SHIFT);
        Self {
            hand,
            flags: Self::encode_proof_flags(mate_distance),
            best_move,
            meta,
        }
    }

    /// confirmed disproof エントリを構築する．
    ///
    /// `ids_depth` は NM 昇格が確認された IDS depth で，
    /// `clear_proven_disproofs_below()` での選択的除去に使用する．
    /// disproof の flags bits 1-6 に 0-63 にクランプして格納する．
    #[inline(always)]
    pub(super) fn new_disproof(
        hand: [u8; HAND_KINDS],
        ids_depth: u32,
    ) -> Self {
        Self {
            hand,
            flags: Self::encode_disproof_flags(ids_depth),
            best_move: 0,
            meta: 0,
        }
    }

    /// proof エントリ用 flags を encode する．
    ///
    /// distance の扱い:
    /// - `1..=63`: `distance_set` (bit 7) をセットし，`mate_distance()` で復元可能
    /// - `> 63`: 6-bit フィールドに収まらないため distance_set なしで encode．
    ///   `mate_distance()` は `None` を返し，PV 抽出は slow path の
    ///   effective_len 計算にフォールバックする
    /// - `0`: distance 未知 (caller が実 distance を計算できなかった場合) を
    ///   意味する sentinel．`mate_distance()` は `None` を返す．
    ///   MID store 時に `compute_dist()` の子 lookup が失敗した場合に 0 が
    ///   渡されうる．有効な mate は必ず 1 手以上あるため，distance=0 を
    ///   「未設定」として扱っても 1-手詰めとの衝突はない
    ///   (1-手詰めは `distance=1` で encode される)．
    #[inline(always)]
    pub(super) fn encode_proof_flags(distance: u16) -> u8 {
        let bit0 = 0x01u8; // is_proof
        if distance > 0 && distance <= 63 {
            0x80 | (((distance & 0x3F) as u8) << 1) | bit0
        } else {
            // distance = 0 (未知) または > 63 (範囲外): distance_set なし
            bit0
        }
    }

    /// confirmed disproof エントリ用 flags．
    ///
    /// bits 1-6 に確認時の IDS depth を格納する(0-63, クランプ)．
    /// bit 0 = 0 (is_proof=false), bit 7 = 0 (refutable_disproof=false)．
    #[inline(always)]
    pub(super) fn encode_disproof_flags(ids_depth: u32) -> u8 {
        // is_proof=0, bits 1-6 に ids_depth (0-63)
        let clamped = ids_depth.min(63) as u8;
        clamped << 1
    }

    /// refutable check で確認された disproof 用 flags (v0.24.75)．
    ///
    /// confirmed disproof と同じ bits 1-6 に ids_depth を格納し，
    /// bit 7 = 1 で refutable disproof マークを付加する．
    /// TT レベルでは confirmed と区別せず，`look_up_proven` は両方を
    /// 返す．PNS 側で arena-limited false NM を防ぐ場合のみ
    /// `look_up_proven_skip_refutable` を使い本フラグで読み飛ばす．
    #[inline(always)]
    pub(super) fn encode_refutable_disproof_flags(ids_depth: u32) -> u8 {
        Self::encode_disproof_flags(ids_depth) | 0x80
    }

    /// refutable check 由来の disproof かどうか (bit 7, v0.24.75)．
    ///
    /// true: `all_checks_refutable_recursive` で格納された NM エントリ．
    /// PNS の interior 探索で使用すると arena-limited false NM を
    /// 誘発するため，PNS 経路では `look_up_proven_skip_refutable` が
    /// 本フラグをチェックしてスキップする．MID 経路からは通常の
    /// disproof と同様に可視．
    #[inline(always)]
    pub(super) fn is_refutable_disproof(&self) -> bool {
        !self.is_proof() && (self.flags & 0x80) != 0
    }

    /// confirmed disproof の確認時 IDS depth を取得する (bits 1-6)．
    #[inline(always)]
    pub(super) fn disproof_depth(&self) -> u32 {
        debug_assert!(!self.is_proof(), "disproof_depth called on proof entry");
        ((self.flags >> 1) & 0x3F) as u32
    }
}

/// `REMAINING_INFINITE` は `tt.rs` で定義されるが，entry.rs からも参照する．
/// v0.24.53 で ProvenEntry の remaining フィールドが meta にリネームされた
/// 後，現在の用途は DfPnEntry 用のみ．
#[allow(dead_code)]
const REMAINING_INFINITE: u16 = 0x7FFF;

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

/// PNS アリーナの**デフォルト**最大ノード数(メモリ上限)．
///
/// 1ノード ≈ 80〜120 bytes(children Vec 含む)．
/// 5M ノードで約 400〜600 MB を使用する．
///
/// v0.25.0 以降は `DfPnSolver::param_pns_arena_max` で実行時に変更可能．
/// この定数はデフォルト値および初期 `Vec::with_capacity` の上限として使用する．
pub(super) const PNS_MAX_ARENA_NODES: usize = 5_000_000;

/// PNS アリーナの初期確保容量上限 (1M ノード)．
///
/// `param_pns_arena_max` を大きく設定しても起動時の物理メモリ確保は
/// この値で抑えて成長は Vec 自動拡張に任せる．
pub(super) const PNS_INIT_CAPACITY_CAP: usize = 1024 * 1024;
