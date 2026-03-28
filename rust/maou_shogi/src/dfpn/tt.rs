//! 転置表(Transposition Table)．

use rustc_hash::FxHashMap;

use crate::types::HAND_KINDS;

use super::entry::DfPnEntry;
use super::{hand_gte_forward_chain, INF, REMAINING_INFINITE};

/// 同一盤面ハッシュあたりの TT エントリ上限．
///
/// 支配関係(パレートフロンティア)による圧縮により，証明・反証エントリは
/// 互いに比較不能な持ち駒構成のみ保持される．そのため上限に達することは
/// 稀であり，主に中間エントリの蓄積を制限する安全弁として機能する．
pub(super) const MAX_TT_ENTRIES_PER_POSITION: usize = 16;

/// HashMap ベースの転置表(証明駒/反証駒対応)．
///
/// キーは盤面のみのハッシュ(持ち駒除外)を使用し，
/// 同一盤面・異なる持ち駒のエントリを同一クラスタに格納する．
/// cshogi と同様のアプローチで，証明駒/反証駒を正確に保持する．
///
/// 参照時に持ち駒の優越関係を利用して TT ヒット率を向上させる:
/// - 証明済み(pn=0): 現在の持ち駒 >= 登録時の持ち駒 → 再利用
/// - 反証済み(dn=0): 登録時の持ち駒 >= 現在の持ち駒 → 再利用
pub(super) struct TranspositionTable {
    pub(super) tt: FxHashMap<u64, Vec<DfPnEntry>>,
    /// TT エントリ溢れ(置換)の発生回数．
    #[cfg(feature = "profile")]
    pub(super) overflow_count: u64,
    /// TT エントリ溢れで置換対象が見つからなかった回数(全て証明/反証済み)．
    #[cfg(feature = "profile")]
    pub(super) overflow_no_victim_count: u64,
    /// 1局面あたりのエントリ数の最大値．
    #[cfg(feature = "profile")]
    pub(super) max_entries_per_position: usize,
    // --- TT 増加診断カウンタ ---
    /// 証明エントリ(pn=0)の挿入回数．
    pub(super) diag_proof_inserts: u64,
    /// 反証エントリ(dn=0)の挿入回数．
    pub(super) diag_disproof_inserts: u64,
    /// 中間エントリの新規追加(同一 hand なし)回数．
    pub(super) diag_intermediate_new: u64,
    /// 中間エントリの既存更新(同一 hand あり)回数．
    pub(super) diag_intermediate_update: u64,
    /// 支配チェックによるスキップ回数．
    pub(super) diag_dominated_skip: u64,
    /// remaining 値別の挿入回数(0..=31 + 32=INFINITE)．
    pub(super) diag_remaining_dist: [u64; 33],
}

impl TranspositionTable {
    /// 転置表を生成する．
    pub(super) fn new() -> Self {
        TranspositionTable {
            tt: FxHashMap::with_capacity_and_hasher(
                65536,
                Default::default(),
            ),
            #[cfg(feature = "profile")]
            overflow_count: 0,
            #[cfg(feature = "profile")]
            overflow_no_victim_count: 0,
            #[cfg(feature = "profile")]
            max_entries_per_position: 0,
            diag_proof_inserts: 0,
            diag_disproof_inserts: 0,
            diag_intermediate_new: 0,
            diag_intermediate_update: 0,
            diag_dominated_skip: 0,
            diag_remaining_dist: [0; 33],
        }
    }

    /// 転置表を参照する(証明駒/反証駒の優越関係を利用)．
    ///
    /// 1. 証明済みエントリ: 現在の持ち駒 >= 登録時 → (0, dn) を返す
    /// 2. 反証済みエントリ: 登録時の持ち駒 >= 現在 かつ 十分な探索深さ → (pn, 0) を返す
    /// 3. 持ち駒完全一致: そのまま返す
    /// 4. 該当なし: (1, 1) を返す
    ///
    /// # 引数
    ///
    /// - `remaining`: 呼び出し元の残り探索深さ．反証済みエントリの有効性判定に使用．
    ///   `0` を指定すると全ての反証済みエントリを受け入れる(事後クエリ用)．
    /// 返り値: `(pn, dn, source)`.
    /// `source` は SNDA 用のソースノードハッシュ．
    /// TT ミス時は `source = 0`(独立ノード: SNDA グルーピング対象外)．
    #[inline(always)]
    pub(super) fn look_up(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
        remaining: u16,
    ) -> (u32, u32, u64) {
        let entries = match self.tt.get(&pos_key) {
            Some(e) => e,
            None => return (1, 1, 0),
        };

        let mut exact_match: Option<(u32, u32, u64)> = None;

        // 証明(pn=0)を反証(dn=0)より常に優先する．
        // IDS の浅い反復で仮反証が保存された後，深い反復で同一局面が
        // 証明されると証明と反証が共存しうる(retain_proofs で除去されるが
        // 同一反復内では共存する可能性がある)．
        // 単一パスでの early return では entries の格納順に依存して
        // 反証が先に返される場合があるため，証明を先にスキャンする．
        for e in entries {
            if e.pn == 0 && hand_gte_forward_chain(hand, &e.hand) {
                return (0, e.dn, e.source);
            }
        }
        for e in entries {
            // 反証済み: 持ち駒が少ない(以下)かつ十分な深さなら再利用．
            // 経路依存反証(path_dependent)は remaining チェックを免除する．
            // GHI ループ検出に由来する反証は propagate_nm_remaining で
            // remaining が極端に小さくなりがちで，同一 IDS 深さの再訪時に
            // マッチせず無限再入(スラッシング)を引き起こすため．
            // IDS 反復間では retain_proofs_only で経路依存反証は除去されるため，
            // 深い反復で古い反証が不正に使われる心配はない．
            if e.dn == 0
                && hand_gte_forward_chain(&e.hand, hand)
                && (e.remaining >= remaining || e.path_dependent)
            {
                return (e.pn, 0, e.source);
            }
            // 完全一致(pn=0/dn=0 は上で個別に処理済みなのでスキップ)
            if exact_match.is_none()
                && e.hand == *hand
                && e.pn != 0
                && e.dn != 0
            {
                exact_match = Some((e.pn, e.dn, e.source));
            }
        }

        exact_match.unwrap_or((1, 1, 0))
    }

    /// TT Best Move を参照する．
    ///
    /// 指定局面の完全一致エントリからベストムーブ(Move16)を返す．
    /// 該当なし，または best_move 未記録の場合は `0` を返す．
    #[inline(always)]
    pub(super) fn look_up_best_move(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
    ) -> u16 {
        let entries = match self.tt.get(&pos_key) {
            Some(e) => e,
            None => return 0,
        };
        for e in entries {
            if e.hand == *hand && e.best_move != 0 {
                return e.best_move;
            }
        }
        0
    }

    /// 転置表を更新する(支配関係によるパレートフロンティア維持)．
    ///
    /// 証明済み・反証済みエントリは持ち駒の半順序(`hand_gte`)に基づく
    /// 支配関係を利用し，冗長なエントリを自動的に除去する:
    ///
    /// - **証明(pn=0)**: 持ち駒が少ないほど強い証明(パレート最小集合を保持)．
    ///   `hand_new ≤ hand_existing` なら既存は不要．
    /// - **反証(dn=0)**: 持ち駒が多いほど強い反証(パレート最大集合を保持)．
    ///   `hand_new ≥ hand_existing` かつ `remaining_new ≥ remaining_existing` なら既存は不要．
    ///
    /// # 引数
    ///
    /// - `remaining`: 登録時の残り探索深さ．
    ///   `REMAINING_INFINITE` は深さ制限なし(真の証明/反証)を示す．
    #[inline(always)]
    pub(super) fn store(
        &mut self,
        pos_key: u64,
        hand: [u8; HAND_KINDS],
        pn: u32,
        dn: u32,
        remaining: u16,
        source: u64,
    ) {
        self.store_impl(pos_key, hand, pn, dn, remaining, source, false, 0);
    }

    /// ベストムーブ付きで転置表を更新する．
    ///
    /// MID ループの中間結果保存時に，最善子ノードの手を記録する．
    /// 次回同一局面に到達した際の手順改善(Dynamic Move Ordering)に使用する．
    #[inline(always)]
    pub(super) fn store_with_best_move(
        &mut self,
        pos_key: u64,
        hand: [u8; HAND_KINDS],
        pn: u32,
        dn: u32,
        remaining: u16,
        source: u64,
        best_move: u16,
    ) {
        self.store_impl(pos_key, hand, pn, dn, remaining, source, false, best_move);
    }

    /// 経路依存フラグ付きで転置表を更新する．
    ///
    /// `path_dependent = true` の反証エントリは，ループ検出に由来し
    /// 異なる探索経路では無効になる可能性がある．
    /// `remaining` を有限値に制限して保存することで，
    /// より深い探索で自動的に再評価される．
    #[inline(always)]
    pub(super) fn store_path_dep(
        &mut self,
        pos_key: u64,
        hand: [u8; HAND_KINDS],
        pn: u32,
        dn: u32,
        remaining: u16,
        source: u64,
        path_dependent: bool,
    ) {
        self.store_impl(pos_key, hand, pn, dn, remaining, source, path_dependent, 0);
    }

    #[inline(always)]
    pub(super) fn store_impl(
        &mut self,
        pos_key: u64,
        hand: [u8; HAND_KINDS],
        pn: u32,
        dn: u32,
        remaining: u16,
        source: u64,
        path_dependent: bool,
        best_move: u16,
    ) {
        // remaining 分布カウンタ用インデックス
        let rem_idx = if remaining == REMAINING_INFINITE { 32 } else { (remaining as usize).min(31) };

        let entries =
            self.tt.entry(pos_key).or_default();

        // === 共通: 既存の証明/反証に支配されているなら挿入不要 ===
        for e in entries.iter() {
            // 証明済みエントリが支配: hand ≥ e.hand → 新エントリの持ち駒で詰み確定
            if e.pn == 0 && hand_gte_forward_chain(&hand, &e.hand) {
                self.diag_dominated_skip += 1;
                return;
            }
            // 反証済みエントリが支配: e.hand ≥ hand かつ十分な深さ → 不詰確定
            // GHI: 経路依存の反証は経路非依存の反証に支配されない
            // 経路依存の新エントリ(remaining 免除)は経路非依存の既存エントリに支配されない
            if e.dn == 0
                && !e.path_dependent
                && !path_dependent
                && hand_gte_forward_chain(&e.hand, &hand)
                && e.remaining >= remaining
            {
                self.diag_dominated_skip += 1;
                return;
            }
        }

        if pn == 0 {
            // === 証明済みエントリの挿入 ===
            // パレートフロンティア(最小持ち駒集合)を維持:
            // 新証明に支配される既存エントリを除去する．
            // - 証明済み: e.hand ≥ hand → より多い持ち駒での証明は冗長
            // - 中間: e.hand ≥ hand → lookup 時に新証明がヒットするため不要
            entries.retain(|e| {
                // 反証済みは保護(証明と反証は異なる持ち駒領域で共存しうる)
                if e.dn == 0 {
                    return true;
                }
                !hand_gte_forward_chain(&e.hand, &hand)
            });
            entries.push(DfPnEntry { hand, pn, dn, remaining, best_move, path_dependent: false, source });
            self.diag_proof_inserts += 1;
            self.diag_remaining_dist[rem_idx] += 1;
            return;
        }

        if dn == 0 {
            // === 反証済みエントリの挿入 ===
            // パレートフロンティア(最大持ち駒 × 最大remaining 集合)を維持:
            // 新反証に支配される既存エントリを除去する．
            // GHI: 経路依存の反証は remaining 免除で lookup に使えるため保護する
            //
            entries.retain(|e| {
                // 証明済みは保護
                if e.pn == 0 {
                    return true;
                }
                if e.dn == 0 {
                    // 経路依存の反証は remaining チェック免除で lookup に使えるため，
                    // 経路非依存の反証では置換しない(remaining 不足で使えなくなる)
                    if e.path_dependent && !path_dependent {
                        return true;
                    }
                    // 反証: e.hand ≤ hand かつ e.remaining ≤ remaining → 冗長
                    return !(hand_gte_forward_chain(&hand, &e.hand)
                        && remaining >= e.remaining);
                }
                // 中間エントリは保護(remaining の不一致で必要になりうる)
                true
            });
            entries.push(DfPnEntry { hand, pn, dn, remaining, best_move, path_dependent, source });
            self.diag_disproof_inserts += 1;
            self.diag_remaining_dist[rem_idx] += 1;
            return;
        }

        // === 中間エントリ(pn > 0, dn > 0)の挿入 ===

        // 同一持ち駒の既存エントリを更新
        for e in entries.iter_mut() {
            if e.hand == hand {
                // 証明済み(pn=0)は上の共通チェックで return 済み
                if e.dn == 0 {
                    // 反証済みエントリの remaining が新エントリの remaining を
                    // カバーしている場合のみ中間値の上書きをブロックする．
                    // カバーしていない場合(e.remaining < remaining)は，
                    // look_up でも使えない「死んだ」反証であるため，
                    // 中間値で上書きして探索の進行を保証する．
                    if e.remaining >= remaining || e.path_dependent {
                        return;
                    }
                    // remaining 不足の反証: 中間値で上書き
                }
                e.pn = pn;
                e.dn = dn;
                e.remaining = remaining;
                e.source = source;
                e.path_dependent = false;
                if best_move != 0 {
                    e.best_move = best_move;
                }
                self.diag_intermediate_update += 1;
                return;
            }
        }

        // 新規エントリを追加
        if entries.len() < MAX_TT_ENTRIES_PER_POSITION {
            entries.push(DfPnEntry {
                hand,
                pn,
                dn,
                remaining,
                best_move,
                path_dependent: false,
                source,
            });
            self.diag_intermediate_new += 1;
            self.diag_remaining_dist[rem_idx] += 1;
            #[cfg(feature = "profile")]
            {
                if entries.len() > self.max_entries_per_position {
                    self.max_entries_per_position = entries.len();
                }
            }
        } else {
            #[cfg(feature = "profile")]
            { self.overflow_count += 1; }
            // 上限到達時: 証明/反証済みエントリを保護しつつ，
            // 「最も未解決な」(|pn - dn| が最小の)エントリを置換する．
            let mut worst_idx: Option<usize> = None;
            let mut worst_score: u64 = u64::MAX;
            for (i, e) in entries.iter().enumerate() {
                if e.pn == 0 || e.dn == 0 {
                    continue;
                }
                let score = if e.pn > e.dn {
                    (e.pn - e.dn) as u64
                } else {
                    (e.dn - e.pn) as u64
                };
                if score < worst_score {
                    worst_score = score;
                    worst_idx = Some(i);
                }
            }
            if let Some(idx) = worst_idx {
                entries[idx] = DfPnEntry { hand, pn, dn, remaining, best_move, path_dependent: false, source };
            } else {
                #[cfg(feature = "profile")]
                { self.overflow_no_victim_count += 1; }
            }
        }
    }

    /// 証明済みエントリの証明駒(登録時の持ち駒)を返す．
    ///
    /// 持ち駒優越で一致する証明済みエントリの hand を返す．
    /// 見つからない場合は渡された hand をそのまま返す．
    #[inline(always)]
    pub(super) fn get_proof_hand(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
    ) -> [u8; HAND_KINDS] {
        if let Some(entries) = self.tt.get(&pos_key) {
            for e in entries {
                if e.pn == 0 && hand_gte_forward_chain(hand, &e.hand) {
                    return e.hand;
                }
            }
        }
        *hand
    }

    /// 反証エントリが経路依存(path_dependent)かどうかを返す．
    ///
    /// OR ノードで子の反証を集約する際，経路依存の子反証が含まれるなら
    /// 親の反証も経路依存として保存する必要がある．
    /// GHI 由来のループ反証は IDS 間で経路が変わると無効になりうるため．
    pub(super) fn has_path_dependent_disproof(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
    ) -> bool {
        if let Some(entries) = self.tt.get(&pos_key) {
            for e in entries {
                if e.dn == 0 && hand_gte_forward_chain(&e.hand, hand) {
                    return e.path_dependent;
                }
            }
        }
        false
    }

    /// 反証エントリの remaining を返す．
    ///
    /// NM の remaining 伝播に使用: 子の NM の remaining が
    /// REMAINING_INFINITE なら親も REMAINING_INFINITE にできる．
    pub(super) fn get_disproof_remaining(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
    ) -> u16 {
        if let Some(entries) = self.tt.get(&pos_key) {
            for e in entries {
                if e.dn == 0 && hand_gte_forward_chain(&e.hand, hand) {
                    return e.remaining;
                }
            }
        }
        0
    }

    /// lookup が実際に使用する反証エントリの (remaining, path_dependent) を返す．
    ///
    /// `has_path_dependent_disproof` / `get_disproof_remaining` は
    /// 最初にマッチした反証エントリの値を返すが，`look_up` は
    /// `e.remaining >= remaining || e.path_dependent` を追加でチェックする．
    /// このため，lookup が使うエントリと情報取得関数が返すエントリが
    /// 食い違う場合がある．この関数は lookup と同じ条件でマッチした
    /// エントリの情報を返す．
    pub(super) fn get_effective_disproof_info(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS],
        remaining: u16,
    ) -> Option<(u16, bool)> {
        if let Some(entries) = self.tt.get(&pos_key) {
            for e in entries {
                if e.dn == 0
                    && hand_gte_forward_chain(&e.hand, hand)
                    && (e.remaining >= remaining || e.path_dependent)
                {
                    return Some((e.remaining, e.path_dependent));
                }
            }
        }
        None
    }

    /// 全エントリをクリアする．
    pub(super) fn clear(&mut self) {
        self.tt.clear();
    }

    /// 確定済みエントリ(証明・確定反証)を保持し，中間エントリを除去する．
    ///
    /// 反復深化 df-pn で使用: 浅い深さの中間エントリや
    /// 深さ制限による仮反証エントリを除去しつつ，
    /// 確定した証明・反証エントリは再利用する．
    pub(super) fn retain_proofs(&mut self) {
        self.tt.retain(|_key, entries| {
            entries.retain(|e| {
                // 証明(pn=0): 常に保持
                // 確定反証(dn=0 かつ path_dependent=false): 経路非依存の真の不詰
                //   → どのパスからアクセスしても同じ結果になるため IDS 間で再利用安全
                // 経路依存反証(dn=0 かつ path_dependent=true): ループ由来
                //   → 異なる IDS 反復では経路が変わり無効になりうるため破棄
                e.pn == 0 || (e.dn == 0 && !e.path_dependent)
            });
            !entries.is_empty()
        });
    }

    /// 経路依存の反証エントリを除去する．
    ///
    /// IDS 反復間で使用: ループ由来の経路依存反証は異なる反復では
    /// 無効になりうるため，次の反復の前に除去する．
    /// 証明・非経路依存反証・中間エントリは保持する．
    pub(super) fn remove_path_dependent_disproofs(&mut self) {
        for entries in self.tt.values_mut() {
            entries.retain(|e| !(e.dn == 0 && e.path_dependent));
        }
    }

    /// 証明エントリ(pn=0)のみを保持し，NM を含む他の全エントリを除去する．
    ///
    /// PNS → IDS 切替時に使用: PNS で蓄積された深さ制限由来の仮 NM エントリを
    /// 除去し，IDS が独立して NM を検出できるようにする．
    /// 証明(pn=0)は常に正しいため保持する．
    pub(super) fn retain_proofs_only(&mut self) {
        self.tt.retain(|_key, entries| {
            entries.retain(|e| e.pn == 0);
            !entries.is_empty()
        });
    }

    /// 浅い反復で remaining が不足する中間・反証エントリを除去する．
    ///
    /// IDS 反復間で使用: スラッシング防止用の中間エントリ
    /// (pn >= INF-1, dn > 0)を除去し，深い反復で再評価させる．
    ///
    /// 反証エントリは除去しない: PNS で蓄積された深い ply の反証は
    /// remaining が小さい(saved_depth - ply)が，full depth で有効であり，
    /// 除去すると root_pn が大幅に増加する．
    pub(super) fn remove_stale_for_ids(&mut self) {
        self.tt.retain(|_, entries| {
            entries.retain(|e| {
                // 証明は常に保持
                if e.pn == 0 { return true; }
                // スラッシング防止エントリ(pn >= INF-1, dn > 0)は除去
                if e.pn >= INF - 1 && e.dn > 0 { return false; }
                // remaining=0 の反証は除去．
                // 同一 depth 内でしか再利用できず，IDS depth が進むと
                // remaining > 0 の検索にヒットしないため不要．
                // IDS 反復間でメモリを解放する．
                if e.dn == 0 && e.remaining == 0 { return false; }
                // 反証・その他は保持
                true
            });
            !entries.is_empty()
        });
    }



    /// TT のポジション数を返す．
    pub(super) fn len(&self) -> usize {
        self.tt.len()
    }

    /// 反復内 periodic GC: 低価値エントリを除去してメモリを解放する．
    ///
    /// IDS 反復間 GC(`remove_stale_for_ids`)と異なり，反復内で TT が肥大化した際に
    /// 呼び出される．`remaining_threshold` 以下のエントリを除去対象とする．
    ///
    /// 除去対象:
    /// - remaining ≤ threshold の反証(dn=0): 浅い深さ制限到達時の反証
    /// - remaining ≤ threshold の中間(pn>0, dn>0): 浅い探索の中間結果
    /// - 証明(pn=0)は常に保持
    ///
    /// 返り値: 除去されたエントリ数．
    pub(super) fn gc_shallow_entries(&mut self, remaining_threshold: u16) -> usize {
        let mut removed = 0usize;
        self.tt.retain(|_, entries| {
            let before = entries.len();
            entries.retain(|e| {
                // 証明は常に保持
                if e.pn == 0 { return true; }
                // REMAINING_INFINITE の反証は常に保持(真の不詰)
                if e.dn == 0 && e.remaining == REMAINING_INFINITE { return true; }
                // remaining ≤ threshold のエントリを除去
                e.remaining > remaining_threshold
            });
            removed += before - entries.len();
            !entries.is_empty()
        });
        removed
    }

    /// TT の全エントリ数(全ポジションの Vec 長の合計)を返す．
    ///
    /// 同一盤面・異なる持ち駒のエントリを含む総数．
    /// `len()` がポジション数(HashMap キー数)を返すのに対し，
    /// `total_entries()` は実際のメモリ消費に比例する値を返す．
    #[cfg(feature = "tt_diag")]
    pub(super) fn total_entries(&self) -> usize {
        self.tt.values().map(|v| v.len()).sum()
    }

    /// 指定ポジションのエントリ数を返す．
    #[cfg(feature = "tt_diag")]
    #[allow(dead_code)]
    pub(super) fn entries_for_position(&self, pos_key: u64) -> usize {
        self.tt.get(&pos_key).map_or(0, |v| v.len())
    }

    /// 指定局面の全エントリをダンプする(診断用)．
    #[cfg(feature = "tt_diag")]
    pub(super) fn dump_entries(&self, pos_key: u64) {
        if let Some(entries) = self.tt.get(&pos_key) {
            for (i, e) in entries.iter().enumerate() {
                verbose_eprintln!(
                    "[tt_dump] entry[{}]: pn={} dn={} remaining={} path_dep={} hand={:?}",
                    i, e.pn, e.dn, e.remaining, e.path_dependent, &e.hand
                );
            }
        }
    }

    /// 証明済み(pn=0)のエントリ数を返す．
    #[cfg(feature = "tt_diag")]
    pub(super) fn count_proven(&self) -> usize {
        self.tt.values().flat_map(|v| v.iter()).filter(|e| e.pn == 0).count()
    }

    /// 反証済み(dn=0)のエントリ数を返す．
    #[cfg(feature = "tt_diag")]
    pub(super) fn count_disproven(&self) -> usize {
        self.tt.values().flat_map(|v| v.iter()).filter(|e| e.dn == 0).count()
    }

    /// 中間(pn>0 かつ dn>0)のエントリ数を返す．
    #[cfg(feature = "tt_diag")]
    pub(super) fn count_intermediate(&self) -> usize {
        self.tt.values().flat_map(|v| v.iter())
            .filter(|e| e.pn > 0 && e.dn > 0).count()
    }

    /// TT コンテンツの詳細分析(診断用)．
    #[cfg(feature = "verbose")]
    pub(super) fn dump_content_analysis(&self) {
        let mut proof_count: u64 = 0;
        let mut disproof_count: u64 = 0;
        let mut intermediate_count: u64 = 0;
        // 反証の remaining 分布
        let mut disproof_rem: [u64; 33] = [0; 33];
        // 中間エントリの pn 分布
        let mut inter_pn_buckets: [u64; 8] = [0; 8]; // [1], [2-5], [6-20], [21-100], [101-1K], [1K-10K], [10K-100K], [100K+]
        // 中間エントリの remaining 分布
        let mut inter_rem: [u64; 33] = [0; 33];
        // 中間エントリの dn 分布
        let mut inter_dn_buckets: [u64; 5] = [0; 5]; // [1], [2-5], [6-20], [21-100], [100+]
        // 局面あたりのエントリ構成
        let mut pos_proof_only: u64 = 0;
        let mut pos_disproof_only: u64 = 0;
        let mut pos_inter_only: u64 = 0;
        let mut pos_mixed: u64 = 0;

        for entries in self.tt.values() {
            let mut has_proof = false;
            let mut has_disproof = false;
            let mut has_inter = false;
            for e in entries {
                if e.pn == 0 {
                    proof_count += 1;
                    has_proof = true;
                } else if e.dn == 0 {
                    disproof_count += 1;
                    has_disproof = true;
                    let ri = if e.remaining == REMAINING_INFINITE { 32 } else { (e.remaining as usize).min(31) };
                    disproof_rem[ri] += 1;
                } else {
                    intermediate_count += 1;
                    has_inter = true;
                    let ri = if e.remaining == REMAINING_INFINITE { 32 } else { (e.remaining as usize).min(31) };
                    inter_rem[ri] += 1;
                    // pn バケット
                    let pb = match e.pn {
                        1 => 0,
                        2..=5 => 1,
                        6..=20 => 2,
                        21..=100 => 3,
                        101..=1000 => 4,
                        1001..=10000 => 5,
                        10001..=100000 => 6,
                        _ => 7,
                    };
                    inter_pn_buckets[pb] += 1;
                    // dn バケット
                    let db = match e.dn {
                        1 => 0,
                        2..=5 => 1,
                        6..=20 => 2,
                        21..=100 => 3,
                        _ => 4,
                    };
                    inter_dn_buckets[db] += 1;
                }
            }
            match (has_proof, has_disproof, has_inter) {
                (true, false, false) => pos_proof_only += 1,
                (false, true, false) => pos_disproof_only += 1,
                (false, false, true) => pos_inter_only += 1,
                _ => pos_mixed += 1,
            }
        }

        verbose_eprintln!("\n=== TT Content Analysis ===");
        verbose_eprintln!("positions: {}  entries: proof={} disproof={} intermediate={}",
            self.tt.len(), proof_count, disproof_count, intermediate_count);
        verbose_eprintln!("pos composition: proof_only={} disproof_only={} inter_only={} mixed={}",
            pos_proof_only, pos_disproof_only, pos_inter_only, pos_mixed);

        // 反証 remaining 分布
        let dr: Vec<String> = disproof_rem.iter().enumerate()
            .filter(|(_, &c)| c > 0)
            .map(|(r, &c)| if r == 32 { format!("INF:{}", c) } else { format!("{}:{}", r, c) })
            .collect();
        verbose_eprintln!("disproof remaining: [{}]", dr.join(", "));

        // 中間 remaining 分布
        let ir: Vec<String> = inter_rem.iter().enumerate()
            .filter(|(_, &c)| c > 0)
            .map(|(r, &c)| if r == 32 { format!("INF:{}", c) } else { format!("{}:{}", r, c) })
            .collect();
        verbose_eprintln!("intermediate remaining: [{}]", ir.join(", "));

        // 中間 pn 分布
        let pn_labels = ["pn=1", "pn=2-5", "pn=6-20", "pn=21-100", "pn=101-1K", "pn=1K-10K", "pn=10K-100K", "pn=100K+"];
        let pb: Vec<String> = inter_pn_buckets.iter().enumerate()
            .filter(|(_, &c)| c > 0)
            .map(|(i, &c)| format!("{}:{}", pn_labels[i], c))
            .collect();
        verbose_eprintln!("intermediate pn dist: [{}]", pb.join(", "));

        // 中間 dn 分布
        let dn_labels = ["dn=1", "dn=2-5", "dn=6-20", "dn=21-100", "dn=100+"];
        let db: Vec<String> = inter_dn_buckets.iter().enumerate()
            .filter(|(_, &c)| c > 0)
            .map(|(i, &c)| format!("{}:{}", dn_labels[i], c))
            .collect();
        verbose_eprintln!("intermediate dn dist: [{}]", db.join(", "));
    }

    /// TT ガベージコレクション: メモリ使用量を抑制する．
    ///
    /// 2段階の GC を実行する:
    /// 1. 中間エントリ(pn>0 かつ dn>0)のうち，remaining が閾値以下のものを除去
    /// 2. それでも閾値を超える場合，全中間エントリを除去(retain_proofs 相当)
    ///
    /// 証明済み(pn=0)と確定反証(dn=0, remaining=∞)は常に保持する．
    pub(super) fn gc(&mut self, target_size: usize) {
        if self.tt.len() <= target_size {
            return;
        }

        // Phase 1: remaining が小さい中間エントリを除去
        // (浅い探索の仮結果は再計算可能)
        let median_remaining = 8u16;
        self.tt.retain(|_key, entries| {
            entries.retain(|e| {
                e.pn == 0
                    || e.dn == 0
                    || e.remaining > median_remaining
            });
            !entries.is_empty()
        });

        if self.tt.len() <= target_size {
            return;
        }

        // Phase 2: 全中間エントリを除去(確定結果のみ保持)
        self.retain_proofs();
    }

    /// 他の TT から確定エントリ(証明・反証)をマージする．
    ///
    /// `other` の全エントリを走査し，証明済み(pn=0)および
    /// 確定反証(dn=0)のエントリのみを `self` に `store()` する．
    /// 中間エントリは破棄される．
    /// プロファイル統計をリセットする．
    #[cfg(feature = "profile")]
    pub(super) fn reset_profile(&mut self) {
        self.overflow_count = 0;
        self.overflow_no_victim_count = 0;
        self.max_entries_per_position = 0;
    }
}

