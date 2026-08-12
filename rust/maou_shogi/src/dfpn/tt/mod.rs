//! 置換表本体 (RegularTable + Query + RepetitionTable)．
//!
//! [`entry::Entry`] (len-aware/cross-hand) を循環配列で管理し，cluster を走査して
//! 複数 entry 横断で pn/dn を合成する．
//!
//! ## single-thread 適応 (並列化は対象外)
//! - Query は entry への生ポインタを保持せず，**cluster 先頭 index を保持し table 参照を都度渡す**
//!   形を採る (借用検査との相性のため)．
//! - thread ノイズ (`tt_noise`) は single-thread では載らないため除外する．atomic/lock も plain 化．

pub(crate) mod entry;

use super::mate_len::{MateLen, MINUS1_MATE_LEN};
use super::search_result::{BitSet64, Depth, Hand, PnDn, SearchAmount, SearchResult};
use entry::{Entry, NULL_HAND};

/// `target + (diff_dst - diff_src)` を駒種別に `[0, MAX_HAND_COUNT]` へ clamp する
/// (cross-hand 親持駒の補正)．
#[inline]
fn apply_delta_hand(target: Hand, diff_src: Hand, diff_dst: Hand) -> Hand {
    let mut res = [0u8; crate::types::HAND_KINDS];
    for i in 0..crate::types::HAND_KINDS {
        let v = target[i] as i32 + diff_dst[i] as i32 - diff_src[i] as i32;
        res[i] = v.clamp(0, crate::types::PieceType::MAX_HAND_COUNT[i] as i32) as u8;
    }
    res
}

/// プールが保持し続けるバイト数の上限 (既定 1GiB)．
///
/// **プールごとの上限**である．主 TT ([`Entry`]) と千日手表 (`RepEntry`) は別の
/// プールなので，実際に保持されうる合計は最大 2 倍になる．既定構成での実測は
/// 両者合わせて約 440MB で上限には遠い．
///
/// `DFPN_TT_POOL_BYTES` で上書きでき，`0` を渡すとプールを完全に無効化して
/// 従来どおり毎回新規確保する (A/B とメモリ制約環境向けの逃げ道)．
fn max_pooled_bytes() -> usize {
    static CAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("DFPN_TT_POOL_BYTES")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(1 << 30)
    })
}

/// `DFPN_TT_POOL_STATS=1` でプールの取りこぼしを stderr へ報告するか．
///
/// [`BufferPool::MAX_PER_SIZE`] / [`BufferPool::MAX_BUCKETS`] が実ワークロードで
/// 足りているかは机上では決まらない．**足りなかったときだけ鳴る**ようにして，
/// 静かなら足りている，と読めるようにする．
fn stats_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("DFPN_TT_POOL_STATS")
            .map(|s| s != "0" && !s.is_empty())
            .unwrap_or(false)
    })
}

/// 確保済みバッファを要素数ごとに使い回すプール．
///
/// **なぜ要るか**: TT は `solve` ごとに新規確保される ([`super::search`] の
/// `TranspositionTable::new`)．[`Entry::null`] も `RepEntry` の初期値も全ゼロでは
/// ないので `vec![v; n]` は `alloc_zeroed` に落ちず，**毎回フレッシュな mmap へ
/// 全バイトを書き込む**．既定 `root_dfpn_nodes = 2_000_000` では 1 探索あたり
/// 主 TT 256MB + 千日手表 96MB になり，first-touch のページフォルトが探索そのものを
/// 覆い隠す (実測: 1 局面 184ms が playout 数と無相関で，変動係数は 1.4%)．
///
/// **探索は不変**: 貸し出し時に初期値で `fill` するので「確保直後のテーブルは全 null」
/// という不変条件は変わらない．変わるのはページが常駐のままという点だけで，
/// lookup/insert の意味論には触れていない．
struct BufferPool<T> {
    /// 要素数ごとのバケット `(要素数, バッファ列)`．
    ///
    /// **サイズを跨いだ単一の列にしてはいけない．** leaf-mate は探索中に小さい TT を
    /// 何度も作って捨てるので，本数だけで上限を掛けると枠が小バッファで埋まり，
    /// solve の最後に drop される **root dfpn の 352MB が弾かれて次の局面で
    /// また新規確保になる** (= 削ったはずの固定費がそのまま戻る)．
    /// 利用箇所ごとに要素数が違う (root dfpn 4M / defensive・filter 1M /
    /// leaf-mate 512〜65536) ため，サイズで分けることが用途で分けることになる．
    buckets: Vec<(usize, Vec<Vec<T>>)>,
    pooled_bytes: usize,
    /// 再利用できた回数 / 新規確保に落ちた回数 (テストと診断用)．
    hits: u64,
    misses: u64,
    /// 既に報告済みの `(要素数, 事由)`．同じ事象で stderr を溢れさせないため．
    warned: Vec<(usize, &'static str)>,
}

impl<T: Clone> BufferPool<T> {
    /// 1 サイズあたりのバイト予算．[`Self::max_per_size`] が本数へ換算する．
    const BYTES_PER_SIZE: usize = 64 << 20;

    /// 要素 1 本が `bytes` のとき，そのサイズで保持する本数．
    ///
    /// **走査コストではなくメモリで決める．** [`Self::take`] が線形に見るのは
    /// バケット (サイズ種) だけで，同サイズ内は `pop` の O(1) なので，本数を
    /// 増やしても探索は遅くならない．
    ///
    /// 小さい TT (leaf-mate 32KB) は **mate スレッド数だけ同時生存する**ので
    /// 本数が要り (攻め + 受けで `leaf_mate_threads + defensive_mate_threads` 本)，
    /// 大きい TT (root dfpn 256MB) は 1〜2 本で足りる．サイズ別のバイト予算で
    /// 割ると両方を自動的に満たす．
    ///
    /// 下限 2 は，予算を超える大きい TT でも最低限プールされることを保証する
    /// (ここが 0 や 1 だと root dfpn の 352MB が毎回新規確保に戻る)．
    fn max_per_size(bytes: usize) -> usize {
        (Self::BYTES_PER_SIZE / bytes.max(1)).clamp(2, 64)
    }
    /// 保持するサイズ種の上限．実際に現れるのは 5 種程度 (`min_tt_entries` の
    /// clamp と leaf-mate の 2 冪化でサイズは離散化される)ので，走査は常に軽い．
    const MAX_BUCKETS: usize = 16;

    const fn new() -> Self {
        BufferPool {
            buckets: Vec::new(),
            pooled_bytes: 0,
            hits: 0,
            misses: 0,
            warned: Vec::new(),
        }
    }

    /// 上限に当たってプールが機能しなかったことを 1 事象につき 1 回だけ報告する．
    ///
    /// 冷えたプールでの初回 miss は当然なので鳴らさない．鳴るのは
    /// **一度プールに入ったサイズを再要求して外した**場合と，返却を拒否した場合
    /// だけで，どちらも上限が実ワークロードに足りていない証拠になる．
    fn warn_once(&mut self, n: usize, reason: &'static str) {
        if !stats_enabled() || self.warned.iter().any(|(s, r)| *s == n && *r == reason) {
            return;
        }
        self.warned.push((n, reason));
        eprintln!(
            "[dfpn-tt-pool] {reason}: entries={n} bytes={} hits={} misses={} \
             buckets={} pooled_bytes={} (limits: per_size={} buckets={} bytes={})",
            n * std::mem::size_of::<T>(),
            self.hits,
            self.misses,
            self.buckets.len(),
            self.pooled_bytes,
            Self::max_per_size(n * std::mem::size_of::<T>()),
            Self::MAX_BUCKETS,
            max_pooled_bytes(),
        );
    }

    /// 要素数がちょうど `n` のバッファを取り出し `fill_value` で初期化して返す．
    ///
    /// 要素数が違うバッファは再利用しない (`pointer_of` が `len` に依存するので
    /// 長さを変えると索引が変わる)．
    fn take(&mut self, n: usize, fill_value: &T) -> Option<Vec<T>> {
        let found = self
            .buckets
            .iter()
            .position(|(size, bufs)| *size == n && !bufs.is_empty());
        let Some(idx) = found else {
            self.misses += 1;
            // 一度プールに載ったサイズを再要求して外した．
            //
            // ただし**起動直後は偽陽性になる**: mate スレッドが一斉に起動すると
            // 誰もまだ返却していないので全員が外す．これは上限不足ではなく，
            // 返却が始まれば解消する．ウォームアップを過ぎてなお外す場合だけ
            // 「同時生存数に対して保持本数が足りない」と読める．
            const WARMUP_TAKES: u64 = 64;
            if self.hits + self.misses >= WARMUP_TAKES
                && self.buckets.iter().any(|(size, _)| *size == n)
            {
                self.warn_once(n, "repeat miss after warmup");
            }
            return None;
        };
        let mut buf = self.buckets[idx].1.pop()?;
        self.pooled_bytes -= buf.len() * std::mem::size_of::<T>();
        buf.fill(fill_value.clone());
        self.hits += 1;
        Some(buf)
    }

    /// 使い終わったバッファを返却する (上限超過分は素直に捨てる)．
    fn put(&mut self, buf: Vec<T>) {
        let n = buf.len();
        let bytes = n * std::mem::size_of::<T>();
        if bytes == 0 {
            return;
        }
        if self.pooled_bytes + bytes > max_pooled_bytes() {
            self.warn_once(n, "put refused: DFPN_TT_POOL_BYTES");
            return;
        }
        // `iter_mut().find()` の借用を跨いで `warn_once` (&mut self) を呼べないので，
        // 添字を先に確定させてから触る．
        match self.buckets.iter().position(|(size, _)| *size == n) {
            Some(i) => {
                if self.buckets[i].1.len() >= Self::max_per_size(bytes) {
                    self.warn_once(n, "put refused: per-size budget");
                    return;
                }
                self.buckets[i].1.push(buf);
            }
            None => {
                if self.buckets.len() >= Self::MAX_BUCKETS {
                    self.warn_once(n, "put refused: MAX_BUCKETS");
                    return;
                }
                self.buckets.push((n, vec![buf]));
            }
        }
        self.pooled_bytes += bytes;
    }

    /// 要素数 `n` で保持しているバッファ本数 (テスト用)．
    #[cfg(test)]
    fn pooled_count(&self, n: usize) -> usize {
        self.buckets
            .iter()
            .find(|(size, _)| *size == n)
            .map_or(0, |(_, bufs)| bufs.len())
    }
}

static REGULAR_POOL: std::sync::Mutex<BufferPool<Entry>> = std::sync::Mutex::new(BufferPool::new());
static REP_POOL: std::sync::Mutex<BufferPool<RepEntry>> = std::sync::Mutex::new(BufferPool::new());

/// プールから借りる (lock が poison していれば借りずに新規確保へ落とす)．
fn pool_take<T: Clone>(
    pool: &std::sync::Mutex<BufferPool<T>>,
    n: usize,
    fill_value: &T,
) -> Option<Vec<T>> {
    pool.lock().ok()?.take(n, fill_value)
}

/// プールへ返す (poison していれば捨てる)．
fn pool_put<T: Clone>(pool: &std::sync::Mutex<BufferPool<T>>, buf: Vec<T>) {
    if let Ok(mut p) = pool.lock() {
        p.put(buf);
    }
}

/// 要素数 `n` で主 TT プールが保持している本数 (テスト用)．
#[cfg(test)]
fn regular_pooled_count(n: usize) -> usize {
    REGULAR_POOL.lock().map_or(0, |p| p.pooled_count(n))
}

/// 循環配列でエントリを管理する通常テーブル．
pub(super) struct RegularTable {
    entries: Vec<Entry>,
}

impl Drop for RegularTable {
    fn drop(&mut self) {
        pool_put(&REGULAR_POOL, std::mem::take(&mut self.entries));
    }
}

impl RegularTable {
    /// 要素数 `num_entries` (最低 1) でテーブルを確保し全 null 初期化する．
    ///
    /// 確保は [`BufferPool`] 経由で使い回す．返る内容は `vec![Entry::zeroed(); n]` と
    /// 同一なので探索は不変．
    pub(super) fn new(num_entries: usize) -> Self {
        let n = num_entries.max(1);
        // 全ゼロ値で初期化する — memset に畳まれ，`Entry::null()` の
        // 64 バイトパターン書き込みより実測 7.6 倍速い (537MB で 2.769s → 0.366s)．
        // 全ゼロを空 slot と読んでよい根拠は `Entry::zeroed` を参照．
        let null = Entry::zeroed();
        let entries = pool_take(&REGULAR_POOL, n, &null).unwrap_or_else(|| vec![null; n]);
        RegularTable { entries }
    }
    /// 保存可能要素数．
    pub(super) fn capacity(&self) -> usize {
        self.entries.len()
    }
    /// `board_key` の cluster 先頭 index (上位積算 + 右シフトで mod を回避する)．
    #[inline]
    pub(super) fn pointer_of(&self, board_key: u64) -> usize {
        let hash_low = board_key & 0xffff_ffff;
        ((hash_low as u128 * self.entries.len() as u128) >> 32) as usize
    }

    /// 10000 個を stride 334 でサンプルし使用率を見積もる (全走査を避ける近似)．
    pub(super) fn calculate_hash_rate(&self) -> f64 {
        const SAMPLES: usize = 10_000;
        let n = self.entries.len();
        if n == 0 {
            return 1.0;
        }
        let samples = SAMPLES.min(n);
        let mut used = 0usize;
        let mut idx = 1usize % n;
        for _ in 0..samples {
            if !self.entries[idx].is_null() {
                used += 1;
            }
            idx += 334;
            if idx >= n {
                idx -= n;
            }
        }
        used as f64 / samples as f64
    }

    /// 20000 個の amount をサンプルし，下位 `ratio` 分位を閾値に，それ以下の amount の entry を
    /// null 化して間引く．max_amount が飽和に近ければ全 amount を半減する．
    /// 最後に compaction で穴を詰める．
    pub(super) fn collect_garbage(&mut self, ratio: f64) {
        const SAMPLES: usize = 20_000;
        let n = self.entries.len();
        if n == 0 {
            return;
        }
        let mut amounts: Vec<SearchAmount> = Vec::with_capacity(SAMPLES.min(n));
        let mut idx = 0usize;
        let mut scanned = 0usize;
        while amounts.len() < SAMPLES && scanned < n {
            if !self.entries[idx].is_null() {
                amounts.push(self.entries[idx].amount());
            }
            idx += 334;
            if idx >= n {
                idx -= n;
            }
            scanned += 1;
        }
        if amounts.is_empty() {
            return;
        }
        // 下位 ratio 分位を閾値 pivot とする．
        let pivot = ((amounts.len() as f64 * ratio) as usize)
            .max(1)
            .min(amounts.len() - 1);
        amounts.sort_unstable();
        let amount_threshold = amounts[pivot];
        let max_amount = *amounts.last().unwrap();
        let should_cut = max_amount > SearchAmount::MAX / 8;
        for e in &mut self.entries {
            if e.is_null() {
                continue;
            }
            // 証明済/反証済 (final) エントリは GC で消さない (amount cut もしない)．
            // これらは証明木 = 探索後の PV 復元 (STRICT VERIFY) が辿る経路であり，小さい TT で
            // eviction が走っても偽証明 (STRICT None) を出さないための健全性保護．
            // (default size では gc 自体が発火しないため挙動不変; 小 TT でのみ効く．)
            if e.is_final() {
                continue;
            }
            if e.amount() <= amount_threshold {
                e.set_null();
            } else if should_cut {
                e.cut_amount();
            }
        }
        self.compact_entries();
    }

    /// null 穴を詰めて各 entry を `pointer_of` 起点からなるべく手前へ再配置する
    /// (open addressing の probe chain 整合; 先頭周辺は wrap で目を瞑る)．
    fn compact_entries(&mut self) {
        let n = self.entries.len();
        if n == 0 {
            return;
        }
        for i in 0..n {
            if self.entries[i].is_null() {
                continue;
            }
            let start = self.pointer_of(self.entries[i].board_key());
            let mut j = start;
            while j != i {
                if self.entries[j].is_null() {
                    self.entries[j] = self.entries[i];
                    self.entries[i].set_null();
                    break;
                }
                j += 1;
                if j == n {
                    j = 0;
                }
            }
        }
    }
}

/// 千日手手順 (経路ハッシュ) を記録する **固定サイズ** 置換表 (KomoringHeights 移植)．
///
/// open-addressing + 線形走査．**generation ベース GC** で古い経路エントリを間引き，
/// メモリを bound する (旧実装は `FxHashMap` で無制限に膨張し，超長手数探索で OOM
/// していた)．eviction/compaction が不完全でも健全性は保たれる: rep 表の miss は
/// look_up で `make_unknown` にフォールスルーし **再探索** されるだけで (千日手の正しさは
/// on-path 検出 `path_depths`/`dom_path` が独立に担保する)，誤結果は生じない．
///
/// `path_key == 0` を空スロット sentinel とする (root の path_key は 0 だが root 局面が
/// 千日手化することはない; 稀な XOR 衝突での memo 損失は再探索で sound)．
#[derive(Clone, Copy)]
struct RepEntry {
    key: u64,
    depth: Depth,
    len: MateLen,
    generation: u16,
}

pub(super) struct RepetitionTable {
    entries: Vec<RepEntry>,
    generation: u16,
    entry_count: u64,
    next_generation_update: u64,
    next_gc: u16,
    entries_per_generation: usize,
}

impl Drop for RepetitionTable {
    fn drop(&mut self) {
        pool_put(&REP_POOL, std::mem::take(&mut self.entries));
    }
}

impl RepetitionTable {
    /// テーブル全体を何 generation で管理するか (KH と同値)．
    const GEN_PER_TABLE: u64 = 20;
    const INITIAL_GC_DURATION: u16 = 6;
    const GC_DURATION: u16 = 3;
    const GC_KEEP_GENERATION: u16 = 3;

    pub(super) fn new(size: usize) -> Self {
        let size = size.max(1);
        let epg = ((size as u64 / Self::GEN_PER_TABLE).max(1)) as usize;
        // 全ゼロ値 — 空判定は `key == 0` なので `len` は空 slot では読まれない
        // (`MINUS1_MATE_LEN` は len_plus_1 = 0 で全ゼロ)．主 TT と同じ理由で
        // memset に畳むために揃えてある．
        let empty = RepEntry {
            key: 0,
            depth: 0,
            len: MINUS1_MATE_LEN,
            generation: 0,
        };
        RepetitionTable {
            entries: pool_take(&REP_POOL, size, &empty).unwrap_or_else(|| vec![empty; size]),
            generation: 0,
            entry_count: 0,
            next_generation_update: epg as u64,
            next_gc: Self::INITIAL_GC_DURATION,
            entries_per_generation: epg,
        }
    }

    #[inline]
    fn start_index(&self, key: u64) -> usize {
        let key_low = (key & 0xffff_ffff) as u128;
        ((key_low * self.entries.len() as u128) >> 32) as usize
    }
    #[inline]
    fn next_index(&self, idx: usize) -> usize {
        if idx + 1 >= self.entries.len() {
            0
        } else {
            idx + 1
        }
    }

    /// path_key へ (depth, len) を記録する．
    pub(super) fn insert(&mut self, path_key: u64, depth: Depth, len: MateLen) {
        if path_key == 0 {
            return; // 空 sentinel と衝突するので memo しない (sound)．
        }
        let cap = self.entries.len();
        let mut index = self.start_index(path_key);
        let mut probes = 0usize;
        while self.entries[index].key != 0 && self.entries[index].key != path_key {
            index = self.next_index(index);
            probes += 1;
            if probes >= cap {
                return; // テーブル満杯 (GC 前の窓)．memo を諦める (再探索で sound)．
            }
        }
        if self.entries[index].key == 0 {
            self.entries[index] = RepEntry {
                key: path_key,
                depth,
                len,
                generation: self.generation,
            };
            self.entry_count += 1;
            if self.entry_count >= self.next_generation_update {
                self.generation = self.generation.wrapping_add(1);
                self.next_generation_update = self.entry_count + self.entries_per_generation as u64;
                if self.generation >= self.next_gc {
                    self.collect_garbage();
                    self.next_gc = self.generation.wrapping_add(Self::GC_DURATION);
                }
            }
        } else {
            // 既存更新: len が変われば上書き，同 len なら深い方を採る．
            if self.entries[index].len != len {
                self.entries[index].depth = depth;
                self.entries[index].len = len;
                self.entries[index].generation = self.generation;
            } else if self.entries[index].depth <= depth {
                self.entries[index].depth = depth;
                self.entries[index].generation = self.generation;
            }
        }
    }

    /// 記録された table_len が探索 len 以上なら，その千日手はこの探索にも適用される．
    pub(super) fn contains(&self, path_key: u64, len: MateLen) -> Option<(Depth, MateLen)> {
        if path_key == 0 {
            return None;
        }
        let cap = self.entries.len();
        let mut index = self.start_index(path_key);
        let mut probes = 0usize;
        while self.entries[index].key != 0 {
            let e = self.entries[index];
            if e.key == path_key && e.len >= len {
                return Some((e.depth, e.len));
            }
            index = self.next_index(index);
            probes += 1;
            if probes >= cap {
                break;
            }
        }
        None
    }

    /// generation ベース GC: 直近 `GC_KEEP_GENERATION` 世代のエントリだけ残し，
    /// 古い世代を消してから compaction する (KH 移植)．compaction が不完全で一部
    /// エントリが到達不能になっても健全 (miss = 再探索)．
    fn collect_garbage(&mut self) {
        let gen = self.generation;
        let erased = gen.wrapping_sub(Self::GC_KEEP_GENERATION);
        let should_erase = |g: u16| -> bool {
            if erased < gen {
                g < erased || gen < g
            } else {
                gen < g && g < erased
            }
        };
        for e in self.entries.iter_mut() {
            if e.key != 0 && should_erase(e.generation) {
                e.key = 0;
            }
        }
        // compaction: 各エントリを start_index 方向の空きへ手前詰めする．
        let cap = self.entries.len();
        for i in 0..cap {
            if self.entries[i].key == 0 {
                continue;
            }
            let entry = self.entries[i];
            let mut index = self.start_index(entry.key);
            while index != i {
                if self.entries[index].key == 0 {
                    self.entries[index] = entry;
                    self.entries[i].key = 0;
                    break;
                }
                index = self.next_index(index);
            }
        }
    }
}

/// LookUp/SetResult の文脈 (生ポインタを保持しない Query)．
/// 1 局面 (board_key, hand, depth, path_key) に対応し，cluster 先頭 index を cache する．
#[derive(Clone, Copy)]
pub(super) struct TtContext {
    start_idx: usize,
    path_key: u64,
    board_key: u64,
    hand: Hand,
    depth: Depth,
}

impl TtContext {
    #[inline]
    pub(super) fn board_key_hand(&self) -> (u64, Hand) {
        (self.board_key, self.hand)
    }
}

/// regular + repetition を束ねる置換表．
pub(super) struct TranspositionTable {
    regular: RegularTable,
    rep: RepetitionTable,
    gc_count: u64,
}

impl TranspositionTable {
    pub(super) fn new(num_entries: usize, rep_entries: usize) -> Self {
        TranspositionTable {
            regular: RegularTable::new(num_entries),
            rep: RepetitionTable::new(rep_entries),
            gc_count: 0,
        }
    }
    pub(super) fn capacity(&self) -> usize {
        self.regular.capacity()
    }

    /// これまでに実行した GC 回数 (診断用)．
    pub(super) fn gc_count(&self) -> u64 {
        self.gc_count
    }

    /// hashfull が 0.5 以上なら GC を実行する (通常テーブル)．
    /// 千日手テーブルは自前の generation GC で独立に bound される．
    /// テーブル満杯による `look_up` の O(cap) probe 退化を防ぐ．戻り値 = GC を実行したか．
    pub(super) fn maybe_collect_garbage(&mut self) -> bool {
        if self.regular.calculate_hash_rate() >= 0.5 {
            self.regular.collect_garbage(0.5);
            self.gc_count += 1;
            true
        } else {
            false
        }
    }

    /// cluster の board_key 一致 entry を全走査し，優等/劣等局面から pn/dn 境界を合成する．
    /// 返す**親**は exact-hand entry を優先し，無い場合のみ cross-hand 推論 (`apply_delta_hand`) に
    /// fall back する (dominant bound の親で常に上書きすると false な DAG 辺を生むため，exact 優先 +
    /// cross-hand fallback とする)．pn/dn 境界は cross-hand 合成のまま採る．
    pub(super) fn look_up_parent(
        &self,
        board_key: u64,
        hand: Hand,
    ) -> Option<(u64, Hand, PnDn, PnDn)> {
        use entry::hand_is_equal_or_superior;
        let mut pn: PnDn = 1;
        let mut dn: PnDn = 1;
        let mut cross_bk: u64 = 0;
        let mut cross_hand: Hand = NULL_HAND;
        let cap = self.regular.entries.len();
        let mut idx = self.regular.pointer_of(board_key);
        for _ in 0..cap {
            let e = &self.regular.entries[idx];
            if e.is_null() {
                break;
            }
            if e.is_for(board_key) {
                let entry_hand = e.get_hand();
                let eph = e.get_parent_hand();
                let is_inferior = hand_is_equal_or_superior(entry_hand, hand);
                let is_superior = hand_is_equal_or_superior(hand, entry_hand);
                if is_inferior && e.pn() > pn {
                    pn = e.pn();
                    if eph != NULL_HAND && (cross_hand == NULL_HAND || pn > dn) {
                        cross_bk = e.parent_board_key();
                        cross_hand = apply_delta_hand(eph, entry_hand, hand);
                    }
                }
                if is_superior && e.dn() > dn {
                    dn = e.dn();
                    if eph != NULL_HAND && (cross_hand == NULL_HAND || dn > pn) {
                        cross_bk = e.parent_board_key();
                        cross_hand = apply_delta_hand(eph, entry_hand, hand);
                    }
                }
            }
            idx += 1;
            if idx == cap {
                idx = 0;
            }
        }
        if cross_hand == NULL_HAND {
            None
        } else {
            Some((cross_bk, cross_hand, pn, dn))
        }
    }

    /// 1 局面の Query 文脈を構築する．
    pub(super) fn build_query(
        &self,
        path_key: u64,
        board_key: u64,
        hand: Hand,
        depth: Depth,
    ) -> TtContext {
        TtContext {
            start_idx: self.regular.pointer_of(board_key),
            path_key,
            board_key,
            hand,
            depth,
        }
    }

    /// `ctx` の cluster 先頭 cache line を投機的に prefetch する (memory latency hiding)．
    ///
    /// TT は ~8M entry × 64B = ~512MB と L3 を遥かに超え，`look_up` の cluster 先頭アクセスは
    /// ほぼ DRAM miss になる (= tt_lookup phase が memory-bound な理由)．[`entry::Entry`]
    /// は `#[repr(C, align(64))]` で **ちょうど 1 cache line** に収めてあるため，この 1 line の
    /// prefetch で cluster 先頭 entry 全体が載る．child loop では `build_query` 直後・`look_up` 前に
    /// dom_path / path_depths の flat path 走査が入るため，prefetch を発行すると DRAM fetch が
    /// それらと重なり latency を一部隠せる．**純粋な hint で探索結果に影響しない** (search-invariant)．
    /// x86_64 以外では no-op．
    #[inline]
    pub(super) fn prefetch(&self, ctx: &TtContext) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            // start_idx は pointer_of で [0, len) に収まる (build_query が算出)．
            let ptr = self.regular.entries.as_ptr().add(ctx.start_idx) as *const i8;
            core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(ptr);
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = ctx;
    }

    /// **置換表の読み出し**．
    /// cluster を走査し，一致/優等/劣等局面から pn/dn を合成して `SearchResult` を返す．
    /// エントリが無ければ `eval` (initial pn/dn) を seed として first visit を作る (遅延評価)．
    pub(super) fn look_up<F: FnOnce() -> (PnDn, PnDn)>(
        &mut self,
        ctx: &TtContext,
        len: MateLen,
        does_have_old_child: &mut bool,
        eval: F,
    ) -> SearchResult {
        let mut pn: PnDn = 1;
        let mut dn: PnDn = 1;
        let mut amount: SearchAmount = 1;
        let mut found_exact = false;
        let mut sum_mask = BitSet64::full();

        let cap = self.regular.entries.len();
        let mut idx = ctx.start_idx;
        for _ in 0..cap {
            if self.regular.entries[idx].is_null() {
                break; // cluster 終端 (null slot に達した)
            }
            if self.regular.entries[idx].is_for(ctx.board_key) {
                let matched = self.regular.entries[idx].look_up(
                    ctx.hand,
                    ctx.depth,
                    len,
                    &mut pn,
                    &mut dn,
                    does_have_old_child,
                );
                if matched {
                    let e = &self.regular.entries[idx];
                    amount = amount.max(e.amount());
                    if pn == 0 {
                        return SearchResult::make_final(
                            true,
                            e.get_hand(),
                            e.proven_len(),
                            amount,
                        );
                    } else if dn == 0 {
                        return SearchResult::make_final(
                            false,
                            e.get_hand(),
                            e.disproven_len(),
                            amount,
                        );
                    } else if e.get_hand() == ctx.hand {
                        // 一致局面 (exact)
                        if e.is_possible_repetition() {
                            if let Some((depth, table_len)) = self.rep.contains(ctx.path_key, len) {
                                return SearchResult::make_repetition(
                                    ctx.hand, table_len, amount, depth,
                                );
                            }
                        }
                        found_exact = true;
                        sum_mask = e.sum_mask();
                    }
                }
            }
            idx += 1;
            if idx == cap {
                idx = 0;
            }
        }

        // single-thread: ノイズなし．
        if found_exact {
            return SearchResult::make_unknown(pn, dn, len, amount, sum_mask);
        }
        let (init_pn, init_dn) = eval();
        pn = pn.max(init_pn);
        dn = dn.max(init_dn);
        SearchResult::make_first_visit(pn, dn, len, amount)
    }

    /// **置換表の書き込み**．
    pub(super) fn set_result(
        &mut self,
        ctx: &TtContext,
        result: SearchResult,
        parent: (u64, Hand),
    ) {
        if result.pn() == 0 {
            self.set_final(ctx, true, result);
        } else if result.dn() == 0 {
            if result.is_repetition() {
                self.set_repetition(ctx, result);
            } else {
                self.set_final(ctx, false, result);
            }
        } else {
            self.set_unknown(ctx, result, parent);
        }
    }

    /// `board_key`+`hand` のエントリを探し，無ければ null slot を init して返す．
    fn find_or_create(&mut self, board_key: u64, hand: Hand) -> usize {
        let start = self.regular.pointer_of(board_key);
        let cap = self.regular.entries.len();
        let mut idx = start;
        for _ in 0..cap {
            if self.regular.entries[idx].is_null() {
                self.regular.entries[idx].init(board_key, hand);
                return idx;
            }
            if self.regular.entries[idx].is_for_hand(board_key, hand) {
                return idx;
            }
            idx += 1;
            if idx == cap {
                idx = 0;
            }
        }
        // テーブル満杯時の fallback (通常は GC で回避される)．先頭を上書き．
        self.regular.entries[start].init(board_key, hand);
        start
    }

    /// 詰み/不詰の書き込み．**proof/disproof hand 下に格納** (cross-hand)．
    fn set_final(&mut self, ctx: &TtContext, proven: bool, result: SearchResult) {
        let hand = result.hand();
        let idx = self.find_or_create(ctx.board_key, hand);
        let len = result.len();
        let amount = result.amount();
        if proven {
            self.regular.entries[idx].update_proven(len, amount);
        } else {
            self.regular.entries[idx].update_disproven(len, amount);
        }
    }

    /// 千日手の書き込み．rep flag を立て path_key を rep table へ記録．
    fn set_repetition(&mut self, ctx: &TtContext, result: SearchResult) {
        let idx = self.find_or_create(ctx.board_key, ctx.hand);
        self.regular.entries[idx].set_possible_repetition();
        self.rep
            .insert(ctx.path_key, result.repetition_start(), result.len());
    }

    /// 探索中 (unknown) の書き込み．現局面 hand 下に格納．
    fn set_unknown(&mut self, ctx: &TtContext, result: SearchResult, parent: (u64, Hand)) {
        let idx = self.find_or_create(ctx.board_key, ctx.hand);
        self.regular.entries[idx].update_unknown(
            ctx.depth,
            result.pn(),
            result.dn(),
            result.amount(),
            result.sum_mask(),
            parent.0,
            parent.1,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::entry::NULL_HAND;
    use super::*;
    use crate::types::HAND_KINDS;

    fn hand(pawns: u8) -> Hand {
        let mut h = [0u8; HAND_KINDS];
        h[0] = pawns;
        h
    }
    fn null_parent() -> (u64, Hand) {
        (0, NULL_HAND)
    }
    const BK: u64 = 0xABCD_1234_5678_9ABC;

    /// プールから再利用したバッファが **全 null で返る**ことを保証する．
    ///
    /// ここが破れると前の局面の探索結果が次の局面の TT に生き残り，`look_up` が
    /// 他局面の pn/dn を返して**偽証明を出しうる** (soundness 違反)．確保経路を
    /// 使い回しへ変えた以上，これは回帰テストで固定しておく必要がある．
    #[test]
    fn pooled_buffer_comes_back_null() {
        const N: usize = 1024;
        // 1 本目: エントリを書き込んでから drop してプールへ返す．
        {
            let mut tt = TranspositionTable::new(N, N);
            let q = tt.build_query(0x33, BK, hand(3), 2);
            tt.set_result(
                &q,
                SearchResult::make_unknown(11, 13, MateLen::from_len(6), 4, BitSet64::full()),
                null_parent(),
            );
            let mut old = false;
            let r = tt.look_up(&q, MateLen::from_len(6), &mut old, || (1, 1));
            assert_eq!((r.pn(), r.dn()), (11, 13), "前提: 1 本目には書けている");
        }
        // 2 本目: 同サイズなのでプールのバッファが回ってくる．
        let mut tt = TranspositionTable::new(N, N);
        assert!(
            tt.regular.entries.iter().all(|e| e.is_null()),
            "再利用したバッファに前の探索の entry が残っている"
        );
        // 同じ query が「初回訪問」として seed 値を返す (= 前回の 11/13 が消えている)．
        let q = tt.build_query(0x33, BK, hand(3), 2);
        let mut old = false;
        let r = tt.look_up(&q, MateLen::from_len(6), &mut old, || (2, 5));
        assert_eq!((r.pn(), r.dn()), (2, 5));
        assert!(r.is_first_visit());
    }

    /// leaf-mate 相当の小さい TT を大量に作っても，root dfpn 相当の大きい TT が
    /// プールから追い出されないことを保証する．
    ///
    /// **なぜ要るか**: 本数だけで上限を掛けた実装では，探索中に何度も作られる小
    /// バッファが枠を埋め，solve の最後に drop される大 TT が弾かれる．探索結果は
    /// 正しいままなので既存のテストは一切落ちず，**確保コストだけが静かに戻る** —
    /// 壁時計を見ない限り検出できない退行になるので，ここで固定する．
    #[test]
    fn small_allocations_do_not_evict_the_large_buffer() {
        // 他のテストとバケットを共有しないサイズを使う (プールはプロセス共有)．
        const LARGE: usize = 8191;
        const SMALL: usize = 97;

        // production の順序を再現する: leaf-mate が探索の**途中**で小 TT を大量に
        // 作って捨て，root dfpn の大 TT が返るのは solve の**最後**．したがって
        // 大 TT は「すでに埋まったプール」へ返ることになる．
        for _ in 0..64 {
            drop(TranspositionTable::new(SMALL, SMALL));
        }
        drop(TranspositionTable::new(LARGE, LARGE));

        assert_eq!(
            regular_pooled_count(LARGE),
            1,
            "小バッファが大 TT をプールから追い出した (固定費が戻る)"
        );
        assert!(
            regular_pooled_count(SMALL)
                <= BufferPool::<Entry>::max_per_size(SMALL * std::mem::size_of::<Entry>()),
            "1 サイズあたりの保持本数を超えている"
        );
    }

    #[test]
    fn unknown_roundtrip() {
        let mut tt = TranspositionTable::new(1024, 1024);
        let q = tt.build_query(0x11, BK, hand(2), 5);
        // 初回 LookUp: エントリ無し => eval seed．
        let mut old = false;
        let r0 = tt.look_up(&q, MateLen::from_len(10), &mut old, || (4, 6));
        assert_eq!((r0.pn(), r0.dn()), (4, 6));
        assert!(r0.is_first_visit());

        // 書き込み後の LookUp で pn/dn/sum_mask を復元．
        let mut sm = BitSet64::full();
        sm.reset(2);
        let stored = SearchResult::make_unknown(7, 9, MateLen::from_len(10), 3, sm);
        tt.set_result(&q, stored, null_parent());

        let mut old2 = false;
        let r1 = tt.look_up(&q, MateLen::from_len(10), &mut old2, || (1, 1));
        assert_eq!((r1.pn(), r1.dn()), (7, 9));
        assert!(!r1.is_first_visit());
        assert!(!r1.sum_mask().test(2));
    }

    #[test]
    fn proven_len_aware_roundtrip() {
        let mut tt = TranspositionTable::new(1024, 1024);
        let q = tt.build_query(0x22, BK, hand(1), 3);
        // まず探索中 (unknown pn=3,dn=5) を格納し，その後 len=8 で詰みを格納する (実探索の順序)．
        tt.set_result(
            &q,
            SearchResult::make_unknown(3, 5, MateLen::from_len(8), 1, BitSet64::full()),
            null_parent(),
        );
        tt.set_result(
            &q,
            SearchResult::make_final(true, hand(1), MateLen::from_len(8), 2),
            null_parent(),
        );

        // len>=8 で LookUp => proven (len-aware: 探索 len が proven_len 以上)．
        let mut old = false;
        let r = tt.look_up(&q, MateLen::from_len(9), &mut old, || (1, 1));
        assert!(r.is_final());
        assert_eq!(r.pn(), 0);

        // len<8 では proven にならず，格納済 unknown pn/dn (3,5) を返す (eval seed は呼ばれない)．
        let mut old2 = false;
        let r2 = tt.look_up(&q, MateLen::from_len(5), &mut old2, || (99, 99));
        assert!(!r2.is_final());
        assert_eq!((r2.pn(), r2.dn()), (3, 5));
    }

    #[test]
    fn cross_hand_superior_lookup() {
        let mut tt = TranspositionTable::new(1024, 1024);
        // hand=1 で len=8 詰みを格納．
        let q1 = tt.build_query(0x33, BK, hand(1), 3);
        tt.set_result(
            &q1,
            SearchResult::make_final(true, hand(1), MateLen::from_len(8), 1),
            null_parent(),
        );
        // 同 board・hand=3 (優等) で LookUp => 優等局面として詰み．
        let q3 = tt.build_query(0x33, BK, hand(3), 3);
        let mut old = false;
        let r = tt.look_up(&q3, MateLen::from_len(10), &mut old, || (1, 1));
        assert!(r.is_final());
        assert_eq!(r.pn(), 0); // cross-hand proven
    }

    #[test]
    fn repetition_roundtrip() {
        let mut tt = TranspositionTable::new(1024, 1024);
        let q = tt.build_query(0x44, BK, hand(2), 7);
        // 千日手結果 (dn=0, rep_start=2) を格納．
        let rep = SearchResult::make_repetition(hand(2), MateLen::from_len(0), 1, 2);
        tt.set_result(&q, rep, null_parent());
        // LookUp は rep flag + rep_table.Contains で千日手を返す (len <= 記録 len)．
        let mut old = false;
        let r = tt.look_up(&q, MateLen::from_len(0), &mut old, || (1, 1));
        assert!(r.is_final());
        assert_eq!(r.dn(), 0);
        assert!(r.is_repetition());
        assert_eq!(r.repetition_start(), 2);
    }

    #[test]
    fn distinct_boards_do_not_collide() {
        let mut tt = TranspositionTable::new(1024, 1024);
        let qa = tt.build_query(0x1, 0xAAAA, hand(1), 1);
        let qb = tt.build_query(0x2, 0xBBBB, hand(1), 1);
        tt.set_result(
            &qa,
            SearchResult::make_unknown(5, 5, MateLen::from_len(10), 1, BitSet64::full()),
            null_parent(),
        );
        let mut old = false;
        // 別 board は first visit のまま (eval seed)．
        let rb = tt.look_up(&qb, MateLen::from_len(10), &mut old, || (2, 3));
        assert!(rb.is_first_visit());
        assert_eq!((rb.pn(), rb.dn()), (2, 3));
    }
}

#[cfg(test)]
mod warm_tests {
    use super::regular_pooled_count;

    /// [`crate::dfpn::warm_tt_pool`] が該当サイズのバケットを埋める．
    ///
    /// USI `isready` から呼んで初手の first-touch を前払いする機構なので，
    /// **温めたサイズと実際に要求されるサイズが一致すること**が要 —
    /// プールはサイズ別バケットなので，式がずれると warm は静かに空振りし
    /// 初手のコストがそのまま残る (症状が出ないので気づけない)．
    ///
    /// プールはプロセス global なので，他テストと衝突しない要素数を選ぶ．
    #[test]
    fn test_warm_tt_pool_populates_matching_bucket() {
        // size = clamp(NODES * 2, 1<<18, 1<<23) = 600_000
        const NODES: u64 = 300_000;
        const SIZE: usize = 600_000;
        const {
            assert!(
                SIZE > (1 << 18) && SIZE < (1 << 23),
                "clamp に掛からない前提"
            )
        };

        assert_eq!(
            regular_pooled_count(SIZE),
            0,
            "前提: このサイズはまだプールに無い",
        );

        crate::dfpn::warm_tt_pool(NODES);

        assert!(
            regular_pooled_count(SIZE) > 0,
            "warm 後は同じサイズのバケットがプールに載っていること",
        );
    }
}
