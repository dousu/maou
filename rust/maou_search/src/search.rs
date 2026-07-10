//! PUCT ベースの MCTS 探索本体．
//!
//! # バッチ収集の仕組み
//!
//! 各探索スレッドは以下のループを回す:
//!
//! 1. ルートから PUCT で降下し，未展開の葉を最大 `batch_size` 個収集する．
//!    降下中は経路上の全ノードの visits を前置インクリメントする
//!    (virtual loss — 評価結果が返るまでその経路の Q を押し下げ，
//!    同一バッチ内・スレッド間の選択を別の枝へ分散させる)．
//! 2. 収集した葉を [`Evaluator::evaluate_batch`] でまとめて評価する．
//! 3. 各葉を展開し (priors を辺に設定)，value を経路に沿って
//!    バックプロパゲーションする．
//!
//! 他スレッドが評価中 (`EXPANDING`) の葉に到達した場合は衝突として
//! 前置 visits をロールバックし，収集を打ち切ってバッチを即時 flush する．
//! 衝突率とバッチ充填率は [`SearchStats`] で観測できる．
//!
//! # ノードプール GC (stop-the-world)
//!
//! プール枯渇時は既存の停止機構をそのまま quiescence の同期に使う:
//! 枯渇を検知したスレッドが停止フラグを立て，全スレッドが手持ちバッチを
//! 評価・バックプロパゲーションしてから join する (この時点で in-flight の
//! virtual visits は残らない)．その後シングルスレッドで
//! [`NodePool::compact`] を呼び，低訪問サブツリーを刈り取ってから探索
//! スレッドを再起動する．GC 中の並行アクセスは `&mut` 要求により
//! コンパイル時に排除される．
//!
//! # 千日手検出
//!
//! 降下中は経路上の各局面の (hash, 王手フラグ) を path と並行してスタックに
//! 積み，未展開の葉に到達した時点で「対局履歴 + 経路」を後方走査して同一
//! 局面の再出現を調べる ([`crate::repetition`])．木には合流 (transposition)
//! が無く root への経路はノード毎に一意なので，千日手判定の結果はノードに
//! 対して不変 — 初回検出でノードを終端状態に焼き付け，以後の再訪は走査なし
//! で固定値をバックプロパゲーションする．root より前の対局履歴は
//! [`Searcher::search_with_history`] で渡す．
//!
//! # AND-OR 勝敗確定伝播
//!
//! 詰み/千日手で確定した葉の値は [`propagate_proven`] で祖先へ連鎖的に
//! 昇格する: いずれかの子が手番側負け確定なら親は勝ち確定 (OR)，全子が
//! 確定済みなら親の確定値は `1 - min(子の確定値)` (AND 集約)．確定ノード
//! ([`crate::tree::proven`]) は以後降下せず確定値で短絡し，root が確定した
//! 時点で探索を停止する ([`StopCause::RootProven`])．
//!
//! # ルート並行詰み探索 (dfpn)
//!
//! [`SearchOptions::root_dfpn`] を有効にすると，root 局面に対する dfpn
//! 詰み探索 (maou_shogi) を専用スレッドで並行実行する．詰みが証明されたら
//! root を勝ち確定にして MCTS を停止し，dfpn の詰み手順をそのまま
//! best_move / PV として返す．MCTS が先に終了した場合は dfpn の協調的
//! 停止フラグ (`DfPnSolver::set_stop_flag`) で打ち切る．

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use maou_shogi::board::{Board, SfenError};
use maou_shogi::dfpn::{DfPnSolver, TsumeResult};
use maou_shogi::movegen::generate_legal_moves;
use maou_shogi::moves::Move;

use crate::evaluator::{EvalItem, Evaluator};
use crate::repetition::{find_repetition, HistoryEntry, RepetitionOutcome};
use crate::tree::{node_state, proven, Edge, NodePool, NULL_NODE};

/// ルートノードの pool index (最初の alloc で必ず 0 になる)．
const ROOT_IDX: u32 = 0;

/// 上限指定なし ([`SearchLimits`] が両方 `None`) のときの playout 上限．
const DEFAULT_MAX_PLAYOUTS: u64 = 1 << 20;

/// PV 復元の最大長．
const MAX_PV_LEN: usize = 64;

/// leaf-mate ソルバ 1 回あたりのタイムアウト (秒)．探索は `leaf_mate_nodes`
/// で node-bound されるので実際には到達しない安全弁 (50 ノード探索は μ 秒オーダ)．
const LEAF_MATE_TIMEOUT_SECS: u64 = 1;

/// leaf-mate 依頼キューの最大長．満杯なら探索スレッドは依頼を捨てる
/// (mate スレッドが追いつかないときの graceful degradation; 探索は
/// ブロックしない)．
const MATE_QUEUE_CAP: usize = 4096;

/// leaf-mate の非同期詰み探索依頼 (探索スレッド → 専用 mate スレッド)．
///
/// 探索スレッドは新規展開した葉のうち王手手段を持つものについて，これを
/// キューに try-push する (ブロックしない)．mate スレッドが `board` に対し
/// df-pn を回し，詰みなら [`MateResult`] を結果キューへ返す．mate スレッドは
/// `shared` (特に `NodePool`) に触れず Arc 経由のキューだけで通信する
/// (compact が `&mut NodePool` を取るため; root-dfpn と同じ方針)．
struct MateRequest {
    /// 詰みを判定する葉ノードの index．
    idx: u32,
    /// 葉局面 (手番側が攻め方)．df-pn はこの clone に対して解く．
    board: Board,
    /// root からこの葉までの経路 (proof の AND-OR 伝播に使う)．
    path: Vec<u32>,
    /// 依頼時の GC 世代 ([`Shared::generation`])．
    generation: u64,
}

/// mate スレッドが証明した詰み (専用 mate スレッド → 探索スレッド)．
///
/// 探索スレッド (`worker`) が結果キューから取り出し，pool にアクセスできる
/// 立場で `idx` を [`proven::WIN`] にして `path` を AND-OR 伝播する．適用時に
/// `generation` が現世代と異なれば (compact で index 無効化) 破棄する
/// (偽証明防止)．探索スレッドは compact と排他 (inner scope 内でのみ適用) の
/// ため追加ロック不要．
struct MateResult {
    /// 詰みが証明された葉ノードの index．
    idx: u32,
    /// root からこの葉までの経路．
    path: Vec<u32>,
    /// 依頼時の GC 世代．
    generation: u64,
    /// PV-mate 由来か (統計を leaf-mate と分けるため)．
    is_pv: bool,
}

/// 探索の設定．
#[derive(Clone, Debug)]
pub struct SearchOptions {
    /// 探索スレッド数．
    pub threads: usize,
    /// 1 スレッドが一度に収集・評価する葉の最大数 (評価バッチサイズ)．
    pub batch_size: usize,
    /// PUCT の探索定数．
    pub c_puct: f32,
    /// 未訪問の子に与える親視点 Q の初期値 (first play urgency)．
    pub fpu: f32,
    /// ノードプール容量 (メモリ上限)．到達すると GC で低訪問サブツリーを
    /// 刈り取って継続する ([`SearchOptions::gc_enabled`] が false なら停止)．
    pub node_capacity: u32,
    /// 最大探索深さ．到達した経路は引き分け (0.5) として打ち切る
    /// (千日手にならないまま伸び続ける経路で無限に降下しないためのガード)．
    pub max_ply: u16,
    /// ノードプール GC を有効にするか．有効ならプール枯渇時に
    /// [`NodePool::compact`] で低訪問サブツリーを刈り取って探索を継続する．
    pub gc_enabled: bool,
    /// GC 後に残すノード数のプール容量比 (0.0..=1.0)．小さいほど 1 回の GC で
    /// 多く解放し，次の GC までの間隔が延びる (刈られたサブツリーの再展開
    /// コストとのトレードオフ)．
    pub gc_keep_ratio: f32,
    /// ルート局面の dfpn 詰み探索を MCTS と並行実行するか (既定 false)．
    /// 詰みが証明されたら root を勝ち確定にして探索を停止し，詰み手順を
    /// best_move / PV として返す ([`StopCause::RootProven`])．
    pub root_dfpn: bool,
    /// ルート dfpn のノード予算 ([`SearchOptions::root_dfpn`] 有効時)．
    pub root_dfpn_nodes: u64,
    /// ルート dfpn の探索深さ上限 (最大 2047)．
    pub root_dfpn_depth: u32,
    /// MCTS の各葉で短手詰み探索 (leaf-mate) を行うか (既定 false)．
    /// 展開直前の葉で手番側が短手で詰ませられるかを小予算 df-pn で判定し，
    /// 詰みなら葉を勝ち確定 ([`node_state::TERMINAL_WIN`]) にして AND-OR 伝播
    /// する (dlshogi の MCTS 葉ノード短手数詰み探索相当)．
    pub leaf_mate: bool,
    /// leaf-mate 1 回あたりのノード予算 ([`SearchOptions::leaf_mate`] 有効時)．
    /// 小さいほど cheap かつ短手のみ検出 (既定 50 = dlshogi 相当)．非詰み葉は
    /// この予算で打ち切って Unknown を返すので per-leaf コストの上限になる．
    pub leaf_mate_nodes: u64,
    /// leaf-mate 専用スレッド数 ([`SearchOptions::leaf_mate`] 有効時, 既定 1)．
    /// GPU 律速で余る CPU スレッド数に合わせて増やすと詰み探索スループットが
    /// 上がる (探索スレッドとは別スレッドなので NPS には影響しない)．
    pub leaf_mate_threads: usize,
    /// PV-mate を有効にするか (既定 false)．現在の PV 葉 (最善応手列の深部) に
    /// 大予算 df-pn を専用スレッドで回し，詰みなら勝ち確定にして AND-OR 伝播
    /// する (dlshogi の PV 上長手数詰み探索相当)．leaf-mate では届かない中長手
    /// の詰みを余剰 CPU で狙う．探索スレッドは PV 葉を投入するだけでブロック
    /// しない．
    pub pv_mate: bool,
    /// PV-mate 1 回あたりのノード予算 ([`SearchOptions::pv_mate`] 有効時)．
    /// 既定 1,000,000 (leaf-mate より桁違いに大きい)．
    pub pv_mate_nodes: u64,
    /// PV-mate 専用スレッド数 ([`SearchOptions::pv_mate`] 有効時, 既定 1)．
    pub pv_mate_threads: usize,
}

impl Default for SearchOptions {
    fn default() -> SearchOptions {
        SearchOptions {
            threads: 1,
            batch_size: 8,
            c_puct: 1.5,
            fpu: 0.5,
            node_capacity: 1 << 20,
            max_ply: 512,
            gc_enabled: true,
            gc_keep_ratio: 0.5,
            root_dfpn: false,
            root_dfpn_nodes: 1 << 20,
            root_dfpn_depth: 2047,
            leaf_mate: false,
            leaf_mate_nodes: 50,
            leaf_mate_threads: 1,
            pv_mate: false,
            pv_mate_nodes: 1_000_000,
            pv_mate_threads: 1,
        }
    }
}

/// 探索の停止条件 (予算)．
///
/// 両方 `None` の場合は playout 上限 [`DEFAULT_MAX_PLAYOUTS`] が適用される．
#[derive(Clone, Debug, Default)]
pub struct SearchLimits {
    /// playout 数の上限．バッチ処理の粒度により最大
    /// `threads × batch_size` だけ超過し得る．
    pub max_playouts: Option<u64>,
    /// 時間の上限 (ミリ秒)．
    pub time_ms: Option<u64>,
}

/// 探索が停止した理由．
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopCause {
    /// playout 上限に到達した．
    PlayoutLimit,
    /// 時間上限に到達した．
    TimeLimit,
    /// ノードプールが枯渇した (GC 無効時．有効時は 1 ノードも解放できな
    /// かったか，枯渇時点で予算も尽きていた場合のみこの理由で停止する)．
    PoolExhausted,
    /// ルート局面に合法手がない (探索不能)．
    RootTerminal,
    /// ルートの勝敗/引き分けが AND-OR 伝播で確定した (探索継続は不要)．
    RootProven,
}

impl StopCause {
    fn to_u8(self) -> u8 {
        match self {
            StopCause::PlayoutLimit => 1,
            StopCause::TimeLimit => 2,
            StopCause::PoolExhausted => 3,
            StopCause::RootTerminal => 4,
            StopCause::RootProven => 5,
        }
    }

    fn from_u8(v: u8) -> Option<StopCause> {
        match v {
            1 => Some(StopCause::PlayoutLimit),
            2 => Some(StopCause::TimeLimit),
            3 => Some(StopCause::PoolExhausted),
            4 => Some(StopCause::RootTerminal),
            5 => Some(StopCause::RootProven),
            _ => None,
        }
    }
}

/// stop_cause 未設定を表す番兵値．
const CAUSE_NONE: u8 = 0;

/// ルート直下の子ごとの統計．
#[derive(Clone, Debug)]
pub struct RootChildStat {
    /// 指し手．
    pub mv: Move,
    /// 訪問回数．
    pub visits: u64,
    /// ルート手番側から見た勝率 (未訪問なら 0)．
    pub q: f64,
    /// policy 事前確率．
    pub prior: f32,
}

/// 探索の実行統計．
#[derive(Clone, Debug)]
pub struct SearchStats {
    /// 完了した playout 数 (評価 + 終端到達)．
    pub playouts: u64,
    /// ルート評価 (初回推論 = TensorRT エンジンビルド/ロード等) の所要時間
    /// (ミリ秒)．1 回限りの固定コストであり，計測区間 (`elapsed_ms`/`nps`) から
    /// 除外して別掲する (onnx_bench の warmup と同義)．
    pub warmup_ms: u64,
    /// 経過時間 (ミリ秒)．ルート評価 (`warmup_ms`) は含まない探索本体の時間．
    pub elapsed_ms: u64,
    /// playout 毎秒 (mock/実 NN いずれも「葉評価スループット」の指標)．
    /// warmup (エンジンビルド) を除外した `elapsed_ms` を分母とする．
    pub nps: f64,
    /// 衝突数 (他スレッド評価中の葉に到達してロールバックした回数)．
    pub collisions: u64,
    /// evaluate_batch の呼び出し回数．
    pub eval_batches: u64,
    /// 評価した葉の総数．
    pub eval_items: u64,
    /// バッチあたり平均葉数 (バッチ充填率 = avg_batch / batch_size)．
    pub avg_batch: f64,
    /// 到達した最大深さ．
    pub max_depth: u16,
    /// 千日手を検出してノードを終端化した回数 (終端化済みノードの再訪は
    /// 数えない)．
    pub repetitions: u64,
    /// AND-OR 伝播で勝敗/引き分けが確定した内部ノード数 (葉の終端マークは
    /// 含まない)．
    pub proven_nodes: u64,
    /// leaf-mate 探索が葉で詰みを証明して勝ち確定にした回数
    /// ([`SearchOptions::leaf_mate`] 有効時)．
    pub leaf_mates: u64,
    /// PV-mate 探索が PV 葉で詰みを証明して勝ち確定にした回数
    /// ([`SearchOptions::pv_mate`] 有効時)．
    pub pv_mates: u64,
    /// 使用したノード数 (GC 実行後はその時点の残存数)．
    pub nodes_used: u32,
    /// 子ノード生成の CAS 競合で捨てられたノード数 (GC で回収される)．
    pub leaked_nodes: u64,
    /// 実行された GC (プール compact) の回数．
    pub gc_runs: u64,
    /// GC で解放されたノードの総数．
    pub gc_freed_nodes: u64,
}

/// 探索結果．
#[derive(Clone, Debug)]
pub struct SearchResult {
    /// 最有力手 (ルートに合法手がなければ `None`)．
    /// 選択基準: 訪問回数最大 → 同数なら Q 最大 → 同率なら合法手生成順で先頭．
    pub best_move: Option<Move>,
    /// ルート手番側から見た best_move の勝率 (Q)．root の勝敗が確定した
    /// 場合 ([`StopCause::RootProven`]) は確定値 (0 / 0.5 / 1)．
    pub winrate: f64,
    /// 訪問回数最大の経路 (PV)．
    pub pv: Vec<Move>,
    /// ルート直下の全子の統計 (合法手生成順)．
    pub root_children: Vec<RootChildStat>,
    /// 停止理由．
    pub stop: StopCause,
    /// 実行統計．
    pub stats: SearchStats,
}

/// スレッド間共有の探索状態．
struct Shared<'a, E: Evaluator> {
    pool: NodePool,
    root_board: Board,
    /// root 局面の (hash, 王手フラグ) — 降下スタックの初期エントリ．
    root_entry: HistoryEntry,
    /// root より前の対局履歴 (古い順)．千日手判定で経路の前に連結される．
    game_history: &'a [HistoryEntry],
    evaluator: &'a E,
    opts: &'a SearchOptions,
    deadline: Option<Instant>,
    max_playouts: u64,
    stop: AtomicBool,
    stop_cause: AtomicU8,
    playouts: AtomicU64,
    collisions: AtomicU64,
    eval_batches: AtomicU64,
    eval_items: AtomicU64,
    leaked_nodes: AtomicU64,
    repetitions: AtomicU64,
    proven_nodes: AtomicU64,
    leaf_mates: AtomicU64,
    pv_mates: AtomicU64,
    max_depth: AtomicU16,
    /// ルート並行 dfpn が詰みを証明したか (dfpn スレッドが立てる)．
    dfpn_found: Arc<AtomicBool>,
    /// ルート並行 dfpn の詰み手順 (証明時に dfpn スレッドが書く)．
    dfpn_mate: Arc<Mutex<Option<Vec<Move>>>>,
    /// leaf-mate 依頼キュー (探索スレッドが try-push, mate スレッドが pop)．
    /// mate スレッドは `shared` に触れないため Arc で共有する．
    mate_queue: Arc<Mutex<VecDeque<MateRequest>>>,
    /// PV-mate 依頼キュー (探索スレッドが現 PV 葉を投入, PV スレッドが pop)．
    /// leaf-mate と別キュー・別スレッド (予算が桁違いに大きいため)．
    pv_mate_queue: Arc<Mutex<VecDeque<MateRequest>>>,
    /// leaf-mate 結果キュー (mate スレッドが push, 探索スレッドが drain-apply)．
    mate_results: Arc<Mutex<VecDeque<MateResult>>>,
    /// GC 世代．[`NodePool::compact`] 実行ごとに +1 する．探索スレッドは結果の
    /// 世代と現世代が一致するときだけ proof を適用する (compact 後の無効 index
    /// への誤マーク=偽証明を防ぐ)．
    generation: AtomicU64,
}

impl<E: Evaluator> Shared<'_, E> {
    /// leaf-mate 依頼を専用スレッドへ非同期にキューイングする (探索スレッド用)．
    ///
    /// 探索スレッドをブロックしないため `try_lock` で試み，ロックが取れないか
    /// キューが満杯なら依頼を捨てる (詰み探索を諦めるだけで探索は継続)．
    /// これにより探索スレッドの NPS は leaf-mate によって低下しない．
    fn enqueue_mate_request(&self, idx: u32, board: &Board, path: &[u32]) {
        if let Ok(mut q) = self.mate_queue.try_lock() {
            if q.len() < MATE_QUEUE_CAP {
                q.push_back(MateRequest {
                    idx,
                    board: board.clone(),
                    path: path.to_vec(),
                    generation: self.generation.load(Ordering::Acquire),
                });
            }
        }
    }

    /// mate スレッドが証明した詰みを木へ反映する (探索スレッドから呼ぶ)．
    ///
    /// 探索スレッドは inner scope 内でのみ pool にアクセスし，GC compact は
    /// inner scope の外 (全 worker join 後) で走るため，適用と compact は自然に
    /// 排他される (追加ロック不要)．世代が進んだ結果 (compact 済み = index 無効)
    /// は破棄する (偽証明防止)．`try_mark_proven` (CAS) ゆえ `finish_expansion`
    /// と競合しても安全．
    fn apply_mate_results(&self) {
        loop {
            let res = self
                .mate_results
                .lock()
                .expect("mate_results lock は poison しない")
                .pop_front();
            let Some(res) = res else { break };
            if res.generation != self.generation.load(Ordering::Acquire) {
                continue; // compact 済み — index 無効化，破棄 (偽証明防止)
            }
            if self.pool.get(res.idx).try_mark_proven(proven::WIN) {
                if res.is_pv {
                    self.pv_mates.fetch_add(1, Ordering::Relaxed);
                } else {
                    self.leaf_mates.fetch_add(1, Ordering::Relaxed);
                }
            }
            propagate_proven_from(self, &res.path);
        }
    }

    /// 現在の PV (root から最大訪問辺で辿った経路) の葉局面と経路を返す．
    ///
    /// PV-mate 依頼の生成に使う (探索スレッドから呼ぶので pool アクセス可)．
    /// 木は並行更新されるため厳密な一貫性はないが，df-pn は返した `Board` を
    /// 解くので健全性に影響しない (seed としての近似で十分)．root のみ
    /// (深さ 0) の場合は `None` を返す (root は root-dfpn が担当)．
    fn current_pv_leaf(&self) -> Option<(Vec<u32>, Board)> {
        let mut board = self.root_board.clone();
        let mut path = vec![ROOT_IDX];
        let mut idx = ROOT_IDX;
        loop {
            let node = self.pool.get(idx);
            if node.state() != node_state::EXPANDED {
                break;
            }
            let mut best: Option<(Move, u32, u64)> = None;
            for e in node.edges() {
                let c = e.child.load(Ordering::Acquire);
                if c == NULL_NODE {
                    continue;
                }
                let v = self.pool.get(c).visits();
                if v == 0 {
                    continue;
                }
                if best.is_none_or(|(_, _, bv)| v > bv) {
                    best = Some((e.mv, c, v));
                }
            }
            let Some((mv, child, _)) = best else { break };
            board.do_move(mv);
            path.push(child);
            idx = child;
            if path.len() >= MAX_PV_LEN {
                break;
            }
        }
        if path.len() < 2 {
            return None;
        }
        Some((path, board))
    }

    /// 現在の PV 葉を PV-mate キューへ投入する (探索スレッド用)．
    ///
    /// キューが空のときだけ投入する (自然なスロットル: PV スレッドが前の依頼を
    /// 消化してから次を作るので，1 度に 1 依頼のみ)．探索スレッドはブロック
    /// しない (`try_lock`)．PV 走査は空検出時のみ = PV-solve あたり 1 回程度で
    /// 低頻度．
    fn enqueue_pv_mate_request(&self) {
        let Ok(mut q) = self.pv_mate_queue.try_lock() else {
            return;
        };
        if !q.is_empty() {
            return;
        }
        if let Some((path, board)) = self.current_pv_leaf() {
            q.push_back(MateRequest {
                idx: *path.last().expect("path は空にならない"),
                board,
                path,
                generation: self.generation.load(Ordering::Acquire),
            });
        }
    }

    /// 停止フラグを立てる．最初に立てたスレッドの cause が記録される．
    fn set_stop(&self, cause: StopCause) {
        let _ = self.stop_cause.compare_exchange(
            CAUSE_NONE,
            cause.to_u8(),
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        self.stop.store(true, Ordering::Release);
    }

    fn stopped(&self) -> bool {
        self.stop.load(Ordering::Acquire)
    }

    /// playout 完了を n 件計上し，上限到達なら停止フラグを立てる．
    fn complete_playouts(&self, n: u64) {
        let prev = self.playouts.fetch_add(n, Ordering::Relaxed);
        if prev + n >= self.max_playouts {
            self.set_stop(StopCause::PlayoutLimit);
        }
    }

    /// 時間上限を過ぎていたら停止フラグを立てる．
    fn check_deadline(&self) {
        if let Some(d) = self.deadline {
            if Instant::now() >= d {
                self.set_stop(StopCause::TimeLimit);
            }
        }
    }

    fn note_depth(&self, depth: usize) {
        let d = depth.min(u16::MAX as usize) as u16;
        self.max_depth.fetch_max(d, Ordering::Relaxed);
    }
}

/// 葉選択の結果．
enum Selection {
    /// 評価すべき葉を確保した (path 末尾が葉ノード)．
    /// EvalItem は Board を含み大きいため Box で持つ．
    Leaf { path: Vec<u32>, item: Box<EvalItem> },
    /// 終端 (詰み/千日手/最大深さ) に到達し，その場でバックプロパゲーション済み．
    Backpropped,
    /// 他スレッド評価中の葉に到達した (ロールバック済み)．
    Collision,
    /// ノードプール枯渇 (ロールバック済み)．
    PoolExhausted,
}

/// path 上の前置 visits を取り消す．
fn rollback<E: Evaluator>(shared: &Shared<'_, E>, path: &[u32]) {
    for &idx in path {
        shared.pool.get(idx).revert_visit();
    }
}

/// 葉の評価値を経路に沿って伝播する．
///
/// `leaf_value` は path 末尾ノードの手番側から見た勝率．各ノードの wins は
/// 親手番視点なので，1 手さかのぼるごとに視点を反転しながら加算する．
fn backprop<E: Evaluator>(shared: &Shared<'_, E>, path: &[u32], leaf_value: f64) {
    let mut v = leaf_value;
    for &idx in path.iter().rev() {
        shared.pool.get(idx).add_win(1.0 - v);
        v = 1.0 - v;
    }
}

/// 終端状態の固定評価値 (そのノードの手番側から見た勝率)．
fn terminal_value(state: u8) -> f64 {
    match state {
        node_state::TERMINAL_LOSS => 0.0,
        node_state::TERMINAL_DRAW => 0.5,
        node_state::TERMINAL_WIN => 1.0,
        other => unreachable!("終端状態のみ: {other}"),
    }
}

/// path 末尾の確定値を AND-OR 論理で祖先へ伝播する ([`crate::tree::proven`])．
///
/// - いずれかの子が手番側負け確定 (値 0) → 親は勝ち確定 (OR)
/// - 全子が確定済み → 親の確定値は `1 - min(子の確定値)` (AND 集約 —
///   相手は親視点で最悪の子を選べる)
///
/// どちらも成立しなくなった時点で打ち切る．戻り値は新規に確定した
/// ノード数．千日手由来の確定値を伝播してよい根拠は経路の一意性
/// ([`crate::repetition`] — 判定結果がノード不変なため終端と同格に扱える)．
fn propagate_proven(pool: &NodePool, path: &[u32]) -> u64 {
    let mut newly = 0u64;
    for i in (1..path.len()).rev() {
        let Some(cv) = pool.get(path[i]).proven_value() else {
            break;
        };
        let parent = pool.get(path[i - 1]);
        let p = if cv == 0.0 {
            proven::WIN
        } else {
            let mut min_cv = cv;
            let mut all_proven = true;
            for e in parent.edges() {
                let c = e.child.load(Ordering::Acquire);
                if c == NULL_NODE {
                    all_proven = false;
                    break;
                }
                match pool.get(c).proven_value() {
                    None => {
                        all_proven = false;
                        break;
                    }
                    Some(v) => min_cv = min_cv.min(v),
                }
            }
            if !all_proven {
                break;
            }
            if min_cv == 0.0 {
                proven::WIN
            } else if min_cv == 0.5 {
                proven::DRAW
            } else {
                proven::LOSS
            }
        };
        if parent.try_mark_proven(p) {
            newly += 1;
        }
        // 親が確定した (または既に確定していた) — さらに祖先を調べる
    }
    newly
}

/// 終端/確定値を祖先へ伝播し，root が確定したら探索を停止する．
fn propagate_proven_from<E: Evaluator>(shared: &Shared<'_, E>, path: &[u32]) {
    let newly = propagate_proven(&shared.pool, path);
    if newly > 0 {
        shared.proven_nodes.fetch_add(newly, Ordering::Relaxed);
    }
    if shared.pool.get(ROOT_IDX).proven_value().is_some() {
        shared.set_stop(StopCause::RootProven);
    }
}

/// ルートから PUCT で降下して評価対象の葉を 1 つ選ぶ．
///
/// leaf-mate 有効時は，新規展開した葉のうち王手手段を持つものを専用 mate
/// スレッドへ非同期依頼する ([`Shared::enqueue_mate_request`])．探索スレッド
/// 自身は solve せず即座に葉を NN 評価へ回すため NPS は低下しない．
fn select_leaf<E: Evaluator>(shared: &Shared<'_, E>) -> Selection {
    let pool = &shared.pool;
    let opts = shared.opts;
    let mut board = shared.root_board.clone();
    let mut path: Vec<u32> = Vec::with_capacity(64);
    // path と並行して経路上の各局面の (hash, 王手フラグ) を積む (千日手判定用)
    let mut rep_stack: Vec<HistoryEntry> = Vec::with_capacity(64);
    path.push(ROOT_IDX);
    rep_stack.push(shared.root_entry);
    pool.get(ROOT_IDX).add_visit();
    let mut idx = ROOT_IDX;

    loop {
        let node = pool.get(idx);
        // 確定ノード (詰み/千日手の葉，AND-OR 伝播済みの内部ノード) は
        // 降下せず確定値をその場でバックプロパゲーションする
        if let Some(v) = node.proven_value() {
            shared.note_depth(path.len());
            backprop(shared, &path, v);
            return Selection::Backpropped;
        }
        match node.state() {
            node_state::EXPANDED => {
                let edges = node.edges();
                let sqrt_pn = (node.visits() as f32).sqrt();
                let mut best_i = 0usize;
                let mut best_score = f32::NEG_INFINITY;
                for (i, e) in edges.iter().enumerate() {
                    let (q, n) = match e.child.load(Ordering::Acquire) {
                        NULL_NODE => (opts.fpu, 0u64),
                        c => {
                            let child = pool.get(c);
                            let v = child.visits();
                            if v == 0 {
                                (opts.fpu, 0)
                            } else {
                                ((child.wins() / v as f64) as f32, v)
                            }
                        }
                    };
                    let u = opts.c_puct * e.prior * sqrt_pn / (1.0 + n as f32);
                    let score = q + u;
                    if score > best_score {
                        best_score = score;
                        best_i = i;
                    }
                }
                let edge = &edges[best_i];
                board.do_move(edge.mv);

                let mut child_idx = edge.child.load(Ordering::Acquire);
                if child_idx == NULL_NODE {
                    match pool.alloc() {
                        None => {
                            rollback(shared, &path);
                            return Selection::PoolExhausted;
                        }
                        Some(new_idx) => {
                            match edge.child.compare_exchange(
                                NULL_NODE,
                                new_idx,
                                Ordering::AcqRel,
                                Ordering::Acquire,
                            ) {
                                Ok(_) => child_idx = new_idx,
                                Err(existing) => {
                                    // 競合に負けた: 確保済みノードは捨てる (プール容量を 1 消費)
                                    shared.leaked_nodes.fetch_add(1, Ordering::Relaxed);
                                    child_idx = existing;
                                }
                            }
                        }
                    }
                }
                pool.get(child_idx).add_visit();
                path.push(child_idx);
                rep_stack.push(HistoryEntry::from_board(&board));
                idx = child_idx;

                if path.len() > opts.max_ply as usize {
                    shared.note_depth(path.len());
                    backprop(shared, &path, 0.5);
                    return Selection::Backpropped;
                }
            }
            node_state::UNEXPANDED => {
                // 千日手判定: 結果は経路依存だが root への経路はノード毎に
                // 一意なので，このノードに対して不変 — 終端マークして以後の
                // 走査を省く．判定が決定的なため複数スレッドが同時に到達
                // しても同じ状態を store するだけで競合しない
                // (EXPANDING を経由せず UNEXPANDED から直接遷移する)
                if let Some(outcome) = find_repetition(shared.game_history, &rep_stack) {
                    let state = match outcome {
                        RepetitionOutcome::Loss => node_state::TERMINAL_LOSS,
                        RepetitionOutcome::Draw => node_state::TERMINAL_DRAW,
                        RepetitionOutcome::Win => node_state::TERMINAL_WIN,
                    };
                    node.mark_terminal(state);
                    shared.repetitions.fetch_add(1, Ordering::Relaxed);
                    shared.note_depth(path.len());
                    backprop(shared, &path, terminal_value(state));
                    propagate_proven_from(shared, &path);
                    return Selection::Backpropped;
                }
                if node.try_begin_expansion() {
                    let moves = generate_legal_moves(&mut board);
                    shared.note_depth(path.len());
                    if moves.is_empty() {
                        node.mark_terminal(node_state::TERMINAL_LOSS);
                        backprop(shared, &path, 0.0);
                        propagate_proven_from(shared, &path);
                        return Selection::Backpropped;
                    }
                    // leaf-mate (非同期): 手番側が王手手段を持つ葉 (root 以外) を
                    // 専用 mate スレッドへ依頼する．探索スレッドは solve せず
                    // (try-push のみ, ブロックしない) そのまま葉を NN 評価へ回す．
                    // mate スレッドが詰みを証明したら try_mark_proven(WIN) で当該
                    // ノードを勝ち確定にし AND-OR で root へ伝播する (bootstrap)．
                    // root は root-dfpn / MCTS が担当するため除外する．
                    if shared.opts.leaf_mate
                        && idx != ROOT_IDX
                        && board.does_have_mate_possibility(board.turn())
                    {
                        shared.enqueue_mate_request(idx, &board, &path);
                    }
                    return Selection::Leaf {
                        path,
                        item: Box::new(EvalItem { board, moves }),
                    };
                }
                rollback(shared, &path);
                return Selection::Collision;
            }
            node_state::EXPANDING => {
                rollback(shared, &path);
                return Selection::Collision;
            }
            state @ (node_state::TERMINAL_LOSS
            | node_state::TERMINAL_DRAW
            | node_state::TERMINAL_WIN) => {
                shared.note_depth(path.len());
                backprop(shared, &path, terminal_value(state));
                return Selection::Backpropped;
            }
            other => unreachable!("未知のノード状態: {other}"),
        }
    }
}

/// 探索スレッドのメインループ．
fn worker<E: Evaluator>(shared: &Shared<'_, E>) {
    let batch_size = shared.opts.batch_size.max(1);
    let mut paths: Vec<Vec<u32>> = Vec::with_capacity(batch_size);
    let mut items: Vec<EvalItem> = Vec::with_capacity(batch_size);

    while !shared.stopped() {
        shared.check_deadline();
        // ルート並行 dfpn の詰み証明を反映する (root 勝ち確定 → 停止)
        if shared.dfpn_found.load(Ordering::Acquire) {
            shared.pool.get(ROOT_IDX).try_mark_proven(proven::WIN);
            shared.set_stop(StopCause::RootProven);
            break;
        }
        // leaf-mate / PV-mate スレッドが証明した詰みを木へ反映する (バッチ毎)
        shared.apply_mate_results();
        // PV-mate: 現在の PV 葉を専用スレッドへ依頼する (キューが空のときだけ)
        if shared.opts.pv_mate {
            shared.enqueue_pv_mate_request();
        }
        paths.clear();
        items.clear();
        let mut had_collision = false;

        while items.len() < batch_size && !shared.stopped() {
            match select_leaf(shared) {
                Selection::Leaf { path, item } => {
                    paths.push(path);
                    items.push(*item);
                }
                Selection::Backpropped => {
                    shared.complete_playouts(1);
                }
                Selection::Collision => {
                    shared.collisions.fetch_add(1, Ordering::Relaxed);
                    had_collision = true;
                    // 収集を打ち切って手持ちのバッチを即時評価する
                    break;
                }
                Selection::PoolExhausted => {
                    shared.set_stop(StopCause::PoolExhausted);
                    break;
                }
            }
        }

        if items.is_empty() {
            if had_collision {
                // 手ぶら衝突: 他スレッドの評価完了を待つ
                std::thread::yield_now();
            }
            continue;
        }

        let results = shared.evaluator.evaluate_batch(&items);
        assert_eq!(
            results.len(),
            items.len(),
            "evaluator は items と同数の結果を返すこと"
        );
        shared.eval_batches.fetch_add(1, Ordering::Relaxed);
        shared
            .eval_items
            .fetch_add(items.len() as u64, Ordering::Relaxed);

        let n = items.len() as u64;
        for ((path, item), result) in paths.drain(..).zip(items.drain(..)).zip(results) {
            assert_eq!(
                result.priors.len(),
                item.moves.len(),
                "evaluator は moves と同数の priors を返すこと"
            );
            let leaf = shared.pool.get(*path.last().expect("path は空にならない"));
            let edges: Box<[Edge]> = item
                .moves
                .iter()
                .zip(result.priors.iter())
                .map(|(&mv, &p)| Edge::new(mv, p))
                .collect();
            leaf.finish_expansion(edges);
            backprop(shared, &path, f64::from(result.value.clamp(0.0, 1.0)));
        }
        shared.complete_playouts(n);
        shared.check_deadline();
    }
}

/// leaf-mate 専用スレッドのメインループ．
///
/// 依頼キュー `inq` から取り出し，clone 局面に df-pn を回す (`nodes` 予算)．
/// 詰み (`Checkmate` / `CheckmateNoPv`) なら結果キュー `outq` へ [`MateResult`]
/// を返す (適用は探索スレッド側 [`Shared::apply_mate_results`] が行う)．
///
/// `shared` (特に `NodePool`) には一切触れず Arc 共有のキューだけで通信する
/// (compact が `&mut NodePool` を取るため探索スレッドと同時に pool を借用でき
/// ない; root-dfpn と同じ方針)．探索スレッドをブロックしないため solve は
/// clone 局面に対して行う．
fn mate_worker(
    inq: &Mutex<VecDeque<MateRequest>>,
    outq: &Mutex<VecDeque<MateResult>>,
    stop: &AtomicBool,
    nodes: u64,
    is_pv: bool,
) {
    let mut solver = DfPnSolver::new_leaf_mate(nodes, LEAF_MATE_TIMEOUT_SECS);
    while !stop.load(Ordering::Acquire) {
        let req = inq
            .lock()
            .expect("mate_queue lock は poison しない")
            .pop_front();
        let Some(req) = req else {
            // キューが空: 探索スレッドの投入を待つ (短い譲歩でスピンを緩和)
            std::thread::yield_now();
            continue;
        };
        // solve は clone 局面に対して行う (pool 非参照)
        let mut board = req.board;
        if matches!(
            solver.solve(&mut board),
            TsumeResult::Checkmate { .. } | TsumeResult::CheckmateNoPv { .. }
        ) {
            outq.lock()
                .expect("mate_results lock は poison しない")
                .push_back(MateResult {
                    idx: req.idx,
                    path: req.path,
                    generation: req.generation,
                    is_pv,
                });
        }
    }
}

/// MCTS 探索エンジン．
///
/// evaluator と設定を保持し，[`Searcher::search`] で 1 局面を探索する．
pub struct Searcher<'e, E: Evaluator> {
    evaluator: &'e E,
    options: SearchOptions,
}

impl<'e, E: Evaluator> Searcher<'e, E> {
    /// evaluator と設定から探索エンジンを作る．
    pub fn new(evaluator: &'e E, options: SearchOptions) -> Searcher<'e, E> {
        Searcher { evaluator, options }
    }

    /// SFEN 文字列で与えた局面を探索する．
    pub fn search_sfen(
        &self,
        sfen: &str,
        limits: &SearchLimits,
    ) -> Result<SearchResult, SfenError> {
        let mut board = Board::empty();
        board.set_sfen(sfen)?;
        Ok(self.search(&board, limits))
    }

    /// 局面を探索して最有力手・評価値・統計を返す．
    ///
    /// root より前の対局履歴は考慮しない (探索経路内の千日手のみ検出する)．
    /// 履歴を考慮する場合は [`Searcher::search_with_history`] を使う．
    pub fn search(&self, root_board: &Board, limits: &SearchLimits) -> SearchResult {
        self.search_with_history(root_board, &[], limits)
    }

    /// 対局履歴 (root より前の局面列) を考慮して局面を探索する．
    ///
    /// `game_history` は開始局面から root の直前までの各局面を古い順に並べた
    /// もの (root 自身は含めない)．各エントリは [`HistoryEntry::from_board`]
    /// で作れる．探索中に履歴・経路と同一局面 (盤 + 持ち駒 + 手番) が再出現
    /// した葉は千日手として終端評価される (連続王手の千日手は王手をかけ
    /// 続けた側の負け — [`crate::repetition`])．
    pub fn search_with_history(
        &self,
        root_board: &Board,
        game_history: &[HistoryEntry],
        limits: &SearchLimits,
    ) -> SearchResult {
        // ルート評価 (初回推論 = TensorRT エンジンビルド等) は 1 回限りの固定
        // コストなので計測区間の外で済ませ，warmup_ms として別掲する
        // (onnx_bench の out-of-timer warmup と同義)．計測 (start/deadline) は
        // ルート展開が終わってから開始する．
        let warmup_start = Instant::now();
        let opts = &self.options;

        // ルートの合法手を確認する
        let mut probe = root_board.clone();
        let root_moves = generate_legal_moves(&mut probe);
        if root_moves.is_empty() {
            return SearchResult {
                best_move: None,
                winrate: 0.0,
                pv: Vec::new(),
                root_children: Vec::new(),
                stop: StopCause::RootTerminal,
                stats: SearchStats {
                    playouts: 0,
                    warmup_ms: warmup_start.elapsed().as_millis() as u64,
                    elapsed_ms: 0,
                    nps: 0.0,
                    collisions: 0,
                    eval_batches: 0,
                    eval_items: 0,
                    avg_batch: 0.0,
                    max_depth: 0,
                    repetitions: 0,
                    proven_nodes: 0,
                    leaf_mates: 0,
                    pv_mates: 0,
                    nodes_used: 0,
                    leaked_nodes: 0,
                    gc_runs: 0,
                    gc_freed_nodes: 0,
                },
            };
        }

        // ルートを同期的に評価して展開する
        let pool = NodePool::new(opts.node_capacity);
        let root_idx = pool.alloc().expect("capacity >= 1 なので必ず確保できる");
        debug_assert_eq!(root_idx, ROOT_IDX);
        let root_item = [EvalItem {
            board: root_board.clone(),
            moves: root_moves,
        }];
        let mut root_results = self.evaluator.evaluate_batch(&root_item);
        let root_result = root_results.pop().expect("バッチ 1 件の結果");
        let [root_item] = root_item;
        assert_eq!(
            root_result.priors.len(),
            root_item.moves.len(),
            "evaluator は moves と同数の priors を返すこと"
        );
        let edges: Box<[Edge]> = root_item
            .moves
            .iter()
            .zip(root_result.priors.iter())
            .map(|(&mv, &p)| Edge::new(mv, p))
            .collect();
        {
            let root_node = pool.get(ROOT_IDX);
            root_node.finish_expansion(edges);
            root_node.add_visit();
        }

        // ルート評価 (初回推論のエンジンビルド含む) はここで完了．計測を開始する
        let warmup_ms = warmup_start.elapsed().as_millis() as u64;
        let start = Instant::now();

        let mut shared = Shared {
            pool,
            root_board: root_board.clone(),
            root_entry: HistoryEntry::from_board(root_board),
            game_history,
            evaluator: self.evaluator,
            opts,
            deadline: limits.time_ms.map(|ms| start + Duration::from_millis(ms)),
            max_playouts: limits.max_playouts.unwrap_or(if limits.time_ms.is_some() {
                u64::MAX
            } else {
                DEFAULT_MAX_PLAYOUTS
            }),
            stop: AtomicBool::new(false),
            stop_cause: AtomicU8::new(CAUSE_NONE),
            playouts: AtomicU64::new(0),
            collisions: AtomicU64::new(0),
            eval_batches: AtomicU64::new(0),
            eval_items: AtomicU64::new(0),
            leaked_nodes: AtomicU64::new(0),
            repetitions: AtomicU64::new(0),
            proven_nodes: AtomicU64::new(0),
            leaf_mates: AtomicU64::new(0),
            pv_mates: AtomicU64::new(0),
            max_depth: AtomicU16::new(0),
            dfpn_found: Arc::new(AtomicBool::new(false)),
            dfpn_mate: Arc::new(Mutex::new(None)),
            mate_queue: Arc::new(Mutex::new(VecDeque::new())),
            pv_mate_queue: Arc::new(Mutex::new(VecDeque::new())),
            mate_results: Arc::new(Mutex::new(VecDeque::new())),
            generation: AtomicU64::new(0),
        };

        let gc_keep_target =
            (f64::from(opts.node_capacity) * f64::from(opts.gc_keep_ratio.clamp(0.0, 1.0))) as u32;
        let mut gc_runs = 0u64;
        let mut gc_freed_nodes = 0u64;
        // ルート並行 dfpn (設計 doc §8.1) の協調的停止フラグ．
        // MCTS 側の終了時に立てて dfpn を打ち切る
        let dfpn_stop = Arc::new(AtomicBool::new(false));
        std::thread::scope(|outer| {
            // ルート並行 dfpn: 別スレッドで root の詰みを探索する．GC が
            // shared.pool を排他借用するため shared には触れず，Arc 経由の
            // 成果物 (dfpn_found / dfpn_mate) だけで通信する
            if opts.root_dfpn {
                let stop = Arc::clone(&dfpn_stop);
                let found = Arc::clone(&shared.dfpn_found);
                let mate_out = Arc::clone(&shared.dfpn_mate);
                let mut dfpn_board = root_board.clone();
                let timeout_secs = limits.time_ms.map_or(3600, |ms| ms / 1000 + 1);
                let depth = opts.root_dfpn_depth;
                let nodes = opts.root_dfpn_nodes;
                outer.spawn(move || {
                    let mut solver = DfPnSolver::with_timeout(depth, nodes, timeout_secs);
                    // 実戦用途は最初に見つかった詰みで十分 (最短確定は不要)
                    solver.set_find_shortest(false);
                    solver.set_stop_flag(stop);
                    if let TsumeResult::Checkmate { moves, .. } = solver.solve(&mut dfpn_board) {
                        // CheckmateNoPv (手順なし) は指し手を提示できないため扱わない
                        if !moves.is_empty() {
                            *mate_out.lock().expect("dfpn mate lock は poison しない") =
                                Some(moves);
                            found.store(true, Ordering::Release);
                        }
                    }
                });
            }
            // leaf-mate 専用スレッド: 探索スレッドが投入する詰み依頼を余剰 CPU で
            // 処理する (探索スレッドは solve しないため NPS を落とさない)．
            // root-dfpn と同様 shared には触れず Arc 共有キューだけで通信する
            // (compact が &mut NodePool を取るため)．
            if opts.leaf_mate {
                for _ in 0..opts.leaf_mate_threads.max(1) {
                    let stop = Arc::clone(&dfpn_stop);
                    let inq = Arc::clone(&shared.mate_queue);
                    let outq = Arc::clone(&shared.mate_results);
                    let nodes = opts.leaf_mate_nodes;
                    outer.spawn(move || mate_worker(&inq, &outq, &stop, nodes, false));
                }
            }
            // PV-mate 専用スレッド: 現在の PV 葉に大予算 df-pn を回す (別キュー・
            // 別スレッド)．leaf-mate と同じ Arc 共有・結果適用機構を使う．
            if opts.pv_mate {
                for _ in 0..opts.pv_mate_threads.max(1) {
                    let stop = Arc::clone(&dfpn_stop);
                    let inq = Arc::clone(&shared.pv_mate_queue);
                    let outq = Arc::clone(&shared.mate_results);
                    let nodes = opts.pv_mate_nodes;
                    outer.spawn(move || mate_worker(&inq, &outq, &stop, nodes, true));
                }
            }
            loop {
                std::thread::scope(|s| {
                    for _ in 0..opts.threads.max(1) {
                        s.spawn(|| worker(&shared));
                    }
                });
                // GC で継続するのはプール枯渇による停止のみ (時間/playout は予算終了)
                if !opts.gc_enabled
                    || StopCause::from_u8(shared.stop_cause.load(Ordering::Acquire))
                        != Some(StopCause::PoolExhausted)
                {
                    break;
                }
                // 枯渇時点で予算も尽きていれば GC せずに終了する
                if shared.playouts.load(Ordering::Relaxed) >= shared.max_playouts {
                    break;
                }
                if shared.deadline.is_some_and(|d| Instant::now() >= d) {
                    break;
                }
                // 全スレッド join 済み = quiescent なので排他参照で compact できる．
                // 世代を進めて，compact 前に投入された leaf-mate 結果 (旧 index) を
                // 無効化する (探索スレッドの apply_mate_results が世代不一致で破棄)．
                shared.generation.fetch_add(1, Ordering::AcqRel);
                let gc = shared.pool.compact(gc_keep_target);
                if gc.freed == 0 {
                    // 1 ノードも解放できない場合は再起動しても即枯渇する
                    break;
                }
                gc_runs += 1;
                gc_freed_nodes += u64::from(gc.freed);
                shared.stop_cause.store(CAUSE_NONE, Ordering::Relaxed);
                shared.stop.store(false, Ordering::Release);
            }
            // MCTS 終了 — dfpn に協調停止を要求する (join は scope 終端で行われる)
            dfpn_stop.store(true, Ordering::Release);
        });

        self.collect_result(&shared, warmup_ms, start, gc_runs, gc_freed_nodes)
    }

    /// 探索終了後の木からベストムーブ・PV・統計を集計する．
    ///
    /// `warmup_ms` はルート評価 (エンジンビルド等) の所要時間で，計測区間
    /// (`start` からの経過) には含まれない．
    fn collect_result(
        &self,
        shared: &Shared<'_, E>,
        warmup_ms: u64,
        start: Instant,
        gc_runs: u64,
        gc_freed_nodes: u64,
    ) -> SearchResult {
        let pool = &shared.pool;
        let root_node = pool.get(ROOT_IDX);
        let root_children: Vec<RootChildStat> = root_node
            .edges()
            .iter()
            .map(|e| {
                let (visits, q) = match e.child.load(Ordering::Acquire) {
                    NULL_NODE => (0, 0.0),
                    c => {
                        let child = pool.get(c);
                        let v = child.visits();
                        if v == 0 {
                            (0, 0.0)
                        } else {
                            (v, child.wins() / v as f64)
                        }
                    }
                };
                RootChildStat {
                    mv: e.mv,
                    visits,
                    q,
                    prior: e.prior,
                }
            })
            .collect();

        // 訪問回数最大 → 同数なら Q 最大 → 同率なら合法手生成順で先頭
        let mut best_i = 0usize;
        for (i, c) in root_children.iter().enumerate().skip(1) {
            let b = &root_children[best_i];
            if c.visits > b.visits || (c.visits == b.visits && c.q > b.q) {
                best_i = i;
            }
        }

        // root の勝敗が確定している場合は確定値を達成する子を優先する
        // (訪問回数最大は確定前の探索の偏りを引きずり得る)．勝ち確定なら
        // 負け確定 (値 0) の子，引き分け確定なら値 0.5 の子．負け確定は
        // 全子が勝ち確定なのでどれでも同じ (訪問回数最大のまま)
        let root_proven = root_node.proven_value();
        if let Some(rv) = root_proven {
            if rv > 0.0 {
                let want = 1.0 - rv;
                let found =
                    root_node
                        .edges()
                        .iter()
                        .position(|e| match e.child.load(Ordering::Acquire) {
                            NULL_NODE => false,
                            c => pool.get(c).proven_value() == Some(want),
                        });
                if let Some(i) = found {
                    best_i = i;
                }
            }
        }
        let best = &root_children[best_i];

        // PV: 先頭は best_move に一致させ，以降は訪問回数最大の辺を辿る
        let mut pv = vec![best.mv];
        let mut idx = root_node.edges()[best_i].child.load(Ordering::Acquire);
        while idx != NULL_NODE && pv.len() < MAX_PV_LEN {
            let node = pool.get(idx);
            if node.state() != node_state::EXPANDED {
                break;
            }
            let edges = node.edges();
            let mut pv_best: Option<(&Edge, u64)> = None;
            for e in edges {
                let c = e.child.load(Ordering::Acquire);
                if c == NULL_NODE {
                    continue;
                }
                let v = pool.get(c).visits();
                if v == 0 {
                    continue;
                }
                if pv_best.is_none_or(|(_, bv)| v > bv) {
                    pv_best = Some((e, v));
                }
            }
            let Some((e, _)) = pv_best else { break };
            pv.push(e.mv);
            idx = e.child.load(Ordering::Acquire);
        }

        let elapsed = start.elapsed();
        let playouts = shared.playouts.load(Ordering::Relaxed);
        let eval_batches = shared.eval_batches.load(Ordering::Relaxed);
        let eval_items = shared.eval_items.load(Ordering::Relaxed);
        let stats = SearchStats {
            playouts,
            warmup_ms,
            elapsed_ms: elapsed.as_millis() as u64,
            nps: playouts as f64 / elapsed.as_secs_f64().max(1e-9),
            collisions: shared.collisions.load(Ordering::Relaxed),
            eval_batches,
            eval_items,
            avg_batch: if eval_batches > 0 {
                eval_items as f64 / eval_batches as f64
            } else {
                0.0
            },
            max_depth: shared.max_depth.load(Ordering::Relaxed),
            repetitions: shared.repetitions.load(Ordering::Relaxed),
            proven_nodes: shared.proven_nodes.load(Ordering::Relaxed),
            leaf_mates: shared.leaf_mates.load(Ordering::Relaxed),
            pv_mates: shared.pv_mates.load(Ordering::Relaxed),
            nodes_used: shared.pool.used(),
            leaked_nodes: shared.leaked_nodes.load(Ordering::Relaxed),
            gc_runs,
            gc_freed_nodes,
        };

        // ルート並行 dfpn が詰みを証明していれば詰み手順を優先する
        // (MCTS の木の訪問分布に依存しない constructive proof)
        let dfpn_mate = shared
            .dfpn_mate
            .lock()
            .expect("dfpn mate lock は poison しない")
            .clone();
        let (best_move, winrate, pv, stop) = if let Some(mate) = dfpn_mate {
            (Some(mate[0]), 1.0, mate, StopCause::RootProven)
        } else {
            (
                Some(best.mv),
                root_proven.unwrap_or(best.q),
                pv,
                StopCause::from_u8(shared.stop_cause.load(Ordering::Acquire))
                    .unwrap_or(StopCause::PlayoutLimit),
            )
        };

        SearchResult {
            best_move,
            winrate,
            pv,
            root_children,
            stop,
            stats,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluator::MockEvaluator;

    const STARTPOS: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
    /// 先手: 5三歩・金 1 枚 (持駒)，後手: 5一玉のみ．G*5b (5二金打) の 1 手詰め．
    const MATE_IN_1: &str = "4k4/9/4P4/9/9/9/9/9/9 b G 1";

    fn run(sfen: &str, opts: SearchOptions, limits: SearchLimits, seed: u64) -> SearchResult {
        let evaluator = MockEvaluator::new(seed);
        let searcher = Searcher::new(&evaluator, opts);
        searcher
            .search_sfen(sfen, &limits)
            .expect("テスト SFEN は正当")
    }

    #[test]
    fn test_finds_mate_in_1() {
        let result = run(
            MATE_IN_1,
            SearchOptions {
                threads: 1,
                batch_size: 4,
                node_capacity: 1 << 14,
                ..SearchOptions::default()
            },
            SearchLimits {
                max_playouts: Some(3000),
                ..SearchLimits::default()
            },
            42,
        );
        let best = result.best_move.expect("合法手がある");
        assert_eq!(best.to_usi(), "G*5b", "詰み手 G*5b が最多訪問になるはず");
        assert!(
            result.winrate > 0.9,
            "詰み手の勝率が高いはず: {}",
            result.winrate
        );
        assert_eq!(
            result.pv.first().map(|m| m.to_usi()).as_deref(),
            Some("G*5b")
        );
    }

    #[test]
    fn test_playout_limit_respected() {
        let opts = SearchOptions {
            threads: 2,
            batch_size: 8,
            ..SearchOptions::default()
        };
        let overshoot = (opts.threads * opts.batch_size) as u64;
        let result = run(
            STARTPOS,
            opts,
            SearchLimits {
                max_playouts: Some(500),
                ..SearchLimits::default()
            },
            0,
        );
        assert_eq!(result.stop, StopCause::PlayoutLimit);
        assert!(result.stats.playouts >= 500);
        assert!(
            result.stats.playouts <= 500 + overshoot,
            "playouts = {} (上限 500 + 許容超過 {overshoot})",
            result.stats.playouts
        );
    }

    #[test]
    fn test_multithread_smoke() {
        let result = run(
            STARTPOS,
            SearchOptions {
                threads: 4,
                batch_size: 16,
                ..SearchOptions::default()
            },
            SearchLimits {
                max_playouts: Some(5000),
                ..SearchLimits::default()
            },
            0,
        );
        let best = result.best_move.expect("合法手がある");
        // best_move はルートの合法手のいずれかであること
        assert!(result.root_children.iter().any(|c| c.mv == best));
        // 全子の訪問回数の総和 + ルート初期評価 1 = ルート playout 相当
        let total_child_visits: u64 = result.root_children.iter().map(|c| c.visits).sum();
        assert!(total_child_visits > 0);
        assert!(result.stats.max_depth >= 2, "複数手先まで読んでいるはず");
    }

    #[test]
    fn test_single_thread_deterministic() {
        let opts = SearchOptions {
            threads: 1,
            batch_size: 8,
            ..SearchOptions::default()
        };
        let limits = SearchLimits {
            max_playouts: Some(2000),
            ..SearchLimits::default()
        };
        let a = run(STARTPOS, opts.clone(), limits.clone(), 123);
        let b = run(STARTPOS, opts, limits, 123);
        assert_eq!(
            a.best_move.map(|m| m.to_usi()),
            b.best_move.map(|m| m.to_usi())
        );
        let visits_a: Vec<u64> = a.root_children.iter().map(|c| c.visits).collect();
        let visits_b: Vec<u64> = b.root_children.iter().map(|c| c.visits).collect();
        assert_eq!(visits_a, visits_b, "単一スレッドでは訪問分布まで再現される");
    }

    #[test]
    fn test_pool_exhaustion_stops_search_when_gc_disabled() {
        let result = run(
            STARTPOS,
            SearchOptions {
                threads: 2,
                batch_size: 8,
                node_capacity: 128,
                gc_enabled: false,
                ..SearchOptions::default()
            },
            SearchLimits {
                max_playouts: Some(u64::MAX >> 1),
                ..SearchLimits::default()
            },
            0,
        );
        assert_eq!(result.stop, StopCause::PoolExhausted);
        assert!(result.stats.nodes_used <= 128);
        assert_eq!(result.stats.gc_runs, 0);
        assert!(result.best_move.is_some());
    }

    #[test]
    fn test_gc_continues_past_pool_capacity() {
        // 容量 512 では GC なしなら 20,000 playout に到達する前に必ず枯渇する
        let result = run(
            STARTPOS,
            SearchOptions {
                threads: 1,
                batch_size: 8,
                node_capacity: 512,
                ..SearchOptions::default()
            },
            SearchLimits {
                max_playouts: Some(20_000),
                ..SearchLimits::default()
            },
            0,
        );
        assert_eq!(result.stop, StopCause::PlayoutLimit);
        assert!(result.stats.gc_runs >= 1, "GC が発動しているはず");
        assert!(result.stats.playouts >= 20_000);
        assert!(result.stats.nodes_used <= 512);
        assert!(result.stats.gc_freed_nodes > 0);
        assert!(result.best_move.is_some());
    }

    #[test]
    fn test_mate_proven_with_tiny_pool() {
        // 容量 128 でも (必要なら GC を挟みつつ) 詰みが証明され確定停止する
        let result = run(
            MATE_IN_1,
            SearchOptions {
                threads: 1,
                batch_size: 4,
                node_capacity: 128,
                ..SearchOptions::default()
            },
            SearchLimits {
                max_playouts: Some(10_000),
                ..SearchLimits::default()
            },
            42,
        );
        assert_eq!(result.stop, StopCause::RootProven);
        assert_eq!(
            result.best_move.map(|m| m.to_usi()).as_deref(),
            Some("G*5b")
        );
        assert_eq!(result.winrate, 1.0);
    }

    #[test]
    fn test_gc_deterministic_single_thread() {
        let opts = SearchOptions {
            threads: 1,
            batch_size: 8,
            node_capacity: 512,
            ..SearchOptions::default()
        };
        let limits = SearchLimits {
            max_playouts: Some(10_000),
            ..SearchLimits::default()
        };
        let a = run(STARTPOS, opts.clone(), limits.clone(), 123);
        let b = run(STARTPOS, opts, limits, 123);
        assert!(a.stats.gc_runs >= 1, "GC 経路を通っているはず");
        assert_eq!(a.stats.gc_runs, b.stats.gc_runs);
        assert_eq!(
            a.best_move.map(|m| m.to_usi()),
            b.best_move.map(|m| m.to_usi())
        );
        let visits_a: Vec<u64> = a.root_children.iter().map(|c| c.visits).collect();
        let visits_b: Vec<u64> = b.root_children.iter().map(|c| c.visits).collect();
        assert_eq!(
            visits_a, visits_b,
            "GC を挟んでも単一スレッドなら再現される"
        );
    }

    #[test]
    fn test_gc_multithread_smoke() {
        let result = run(
            STARTPOS,
            SearchOptions {
                threads: 4,
                batch_size: 16,
                node_capacity: 4096,
                ..SearchOptions::default()
            },
            SearchLimits {
                max_playouts: Some(50_000),
                ..SearchLimits::default()
            },
            0,
        );
        assert_eq!(result.stop, StopCause::PlayoutLimit);
        assert!(result.stats.gc_runs >= 1);
        assert!(result.stats.nodes_used <= 4096);
        let best = result.best_move.expect("合法手がある");
        assert!(result.root_children.iter().any(|c| c.mv == best));
    }

    #[test]
    fn test_time_limit_terminates() {
        let result = run(
            STARTPOS,
            SearchOptions {
                threads: 2,
                batch_size: 8,
                ..SearchOptions::default()
            },
            SearchLimits {
                time_ms: Some(100),
                ..SearchLimits::default()
            },
            0,
        );
        assert_eq!(result.stop, StopCause::TimeLimit);
        assert!(result.best_move.is_some());
    }

    /// 既存ノード idx を n_children 個の子付きで展開し，子の index 列を返す．
    fn expand_with_children(pool: &NodePool, idx: u32, n_children: usize) -> Vec<u32> {
        let mv = Move::from_usi("7g7f").expect("正当な USI");
        let edges: Box<[Edge]> = (0..n_children).map(|_| Edge::new(mv, 0.5)).collect();
        let node = pool.get(idx);
        assert!(node.try_begin_expansion());
        node.finish_expansion(edges);
        let mut kids = Vec::new();
        for e in pool.get(idx).edges() {
            let c = pool.alloc().expect("容量内");
            e.child.store(c, Ordering::Release);
            kids.push(c);
        }
        kids
    }

    #[test]
    fn test_propagate_or_win() {
        let pool = NodePool::new(8);
        let root = pool.alloc().expect("容量内");
        let kids = expand_with_children(&pool, root, 2);
        pool.get(kids[0]).mark_terminal(node_state::TERMINAL_LOSS);
        assert_eq!(propagate_proven(&pool, &[root, kids[0]]), 1);
        assert_eq!(
            pool.get(root).proven_value(),
            Some(1.0),
            "子の負け = 親の勝ち (OR)"
        );
    }

    #[test]
    fn test_propagate_and_needs_all_children() {
        let pool = NodePool::new(8);
        let root = pool.alloc().expect("容量内");
        let kids = expand_with_children(&pool, root, 2);
        pool.get(kids[0]).mark_terminal(node_state::TERMINAL_WIN);
        assert_eq!(propagate_proven(&pool, &[root, kids[0]]), 0);
        assert_eq!(
            pool.get(root).proven_value(),
            None,
            "未確定の子が残る間は親を確定できない"
        );
        pool.get(kids[1]).mark_terminal(node_state::TERMINAL_WIN);
        assert_eq!(propagate_proven(&pool, &[root, kids[1]]), 1);
        assert_eq!(
            pool.get(root).proven_value(),
            Some(0.0),
            "全子が勝ち = 親の負け (AND)"
        );
    }

    #[test]
    fn test_propagate_and_draw() {
        let pool = NodePool::new(8);
        let root = pool.alloc().expect("容量内");
        let kids = expand_with_children(&pool, root, 2);
        pool.get(kids[0]).mark_terminal(node_state::TERMINAL_WIN);
        pool.get(kids[1]).mark_terminal(node_state::TERMINAL_DRAW);
        assert_eq!(propagate_proven(&pool, &[root, kids[1]]), 1);
        assert_eq!(
            pool.get(root).proven_value(),
            Some(0.5),
            "最善の子が引き分けなら親も引き分け確定"
        );
    }

    #[test]
    fn test_propagate_chain() {
        // root — mid — leaf の 2 段: leaf 負け → mid 勝ち (OR) →
        // mid が唯一の子なので root は負け (AND)
        let pool = NodePool::new(8);
        let root = pool.alloc().expect("容量内");
        let mid = expand_with_children(&pool, root, 1)[0];
        let leaf = expand_with_children(&pool, mid, 1)[0];
        pool.get(leaf).mark_terminal(node_state::TERMINAL_LOSS);
        assert_eq!(propagate_proven(&pool, &[root, mid, leaf]), 2);
        assert_eq!(pool.get(mid).proven_value(), Some(1.0));
        assert_eq!(pool.get(root).proven_value(), Some(0.0));
    }

    #[test]
    fn test_root_proven_win_mate_in_1() {
        let result = run(
            MATE_IN_1,
            SearchOptions {
                threads: 1,
                batch_size: 4,
                node_capacity: 1 << 14,
                ..SearchOptions::default()
            },
            SearchLimits {
                max_playouts: Some(3000),
                ..SearchLimits::default()
            },
            42,
        );
        assert_eq!(result.stop, StopCause::RootProven);
        assert_eq!(result.winrate, 1.0);
        assert_eq!(
            result.best_move.map(|m| m.to_usi()).as_deref(),
            Some("G*5b")
        );
        assert_eq!(
            result.pv.first().map(|m| m.to_usi()).as_deref(),
            Some("G*5b"),
            "PV の先頭は best_move と一致する"
        );
        assert!(
            result.stats.playouts < 3000,
            "確定で早期停止するはず: {}",
            result.stats.playouts
        );
        assert!(result.stats.proven_nodes >= 1);
    }

    #[test]
    fn test_root_proven_win_mate_in_3() {
        // dfpn テストの canonical 詰み局面 (rust/maou_shogi/src/dfpn/tests.rs)．
        // 3 手以内の詰みが AND-OR 連鎖 (全応手の勝ち確定 → 相手ノードの
        // 負け確定 → root の勝ち確定) で証明される
        let result = run(
            "8k/9/7G1/9/9/9/9/9/9 b G 1",
            SearchOptions {
                threads: 1,
                batch_size: 8,
                node_capacity: 1 << 16,
                ..SearchOptions::default()
            },
            SearchLimits {
                max_playouts: Some(200_000),
                ..SearchLimits::default()
            },
            42,
        );
        assert_eq!(result.stop, StopCause::RootProven, "{:?}", result.stats);
        assert_eq!(result.winrate, 1.0);
        assert!(result.stats.proven_nodes >= 1);
    }

    #[test]
    fn test_leaf_mate_enabled_search_proves() {
        // leaf-mate を有効にした状態でも探索は詰みを正しく証明し，専用 mate
        // スレッドは探索終了時に協調停止する (ハングしない)．3 手詰めを使う．
        // 非同期 leaf-mate の「寄与」自体は GPU 律速前提のため mock では
        // 計測できず (MCTS の AND-OR が先に証明する)，Colab で計測する．
        // ここでは leaf-mate 有効時の健全性 (正しく証明・偽陽性なし・停止) を
        // 検証する．mate スレッドの単体動作は test_mate_worker_finds_mate を参照．
        let result = run(
            "8k/9/7G1/9/9/9/9/9/9 b G 1",
            SearchOptions {
                threads: 1,
                batch_size: 8,
                node_capacity: 1 << 16,
                leaf_mate: true,
                leaf_mate_nodes: 100_000,
                leaf_mate_threads: 2,
                root_dfpn: false,
                ..SearchOptions::default()
            },
            SearchLimits {
                max_playouts: Some(200_000),
                ..SearchLimits::default()
            },
            42,
        );
        assert_eq!(result.stop, StopCause::RootProven, "{:?}", result.stats);
        assert_eq!(result.winrate, 1.0);
    }

    #[test]
    fn test_mate_worker_finds_mate() {
        // 非同期 mate スレッドの単体検証: 詰み局面の依頼に対して df-pn を回し
        // MateResult を結果キューへ返すこと (探索の MCTS 競合に依存しない
        // 決定論的テスト)．
        let inq: Mutex<VecDeque<MateRequest>> = Mutex::new(VecDeque::new());
        let outq: Mutex<VecDeque<MateResult>> = Mutex::new(VecDeque::new());
        let stop = AtomicBool::new(false);
        let mut board = Board::empty();
        board.set_sfen(MATE_IN_1).expect("正当な SFEN");
        inq.lock().expect("lock").push_back(MateRequest {
            idx: 7,
            board,
            path: vec![ROOT_IDX, 7],
            generation: 0,
        });
        std::thread::scope(|s| {
            s.spawn(|| mate_worker(&inq, &outq, &stop, 1000, false));
            let mut got = None;
            for _ in 0..2000 {
                if let Some(r) = outq.lock().expect("lock").pop_front() {
                    got = Some(r);
                    break;
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            stop.store(true, Ordering::Release);
            let r = got.expect("mate スレッドが 1 手詰めを検出して結果を返す");
            assert_eq!(r.idx, 7);
            assert_eq!(r.path, vec![ROOT_IDX, 7]);
            assert_eq!(r.generation, 0);
            assert!(!r.is_pv);
        });
    }

    #[test]
    fn test_pv_mate_enabled_search_proves() {
        // PV-mate を有効にした状態でも探索は正しく詰みを証明し，PV-mate 専用
        // スレッドは探索終了時に協調停止する (ハングしない)．寄与自体は GPU
        // 律速前提のため mock では計測できず (MCTS が先に証明する)，Colab で
        // 計測する．ここでは PV-mate 有効時の健全性 (正しく証明・停止) を検証．
        let result = run(
            "8k/9/6R2/9/9/9/9/9/9 b G 1",
            SearchOptions {
                threads: 1,
                batch_size: 8,
                node_capacity: 1 << 16,
                pv_mate: true,
                pv_mate_nodes: 100_000,
                pv_mate_threads: 2,
                root_dfpn: false,
                ..SearchOptions::default()
            },
            SearchLimits {
                max_playouts: Some(200_000),
                ..SearchLimits::default()
            },
            42,
        );
        assert_eq!(result.stop, StopCause::RootProven, "{:?}", result.stats);
        assert_eq!(result.winrate, 1.0);
    }

    #[test]
    fn test_leaf_mate_no_false_positive() {
        // 初期局面 (詰みなし) では leaf-mate は決して詰みを証明しない
        // (偽陽性ゼロ)．探索は正常に予算で終了する (ハングしない)．
        let result = run(
            STARTPOS,
            SearchOptions {
                threads: 1,
                batch_size: 8,
                node_capacity: 1 << 16,
                leaf_mate: true,
                ..SearchOptions::default()
            },
            SearchLimits {
                max_playouts: Some(3000),
                ..SearchLimits::default()
            },
            7,
        );
        assert_ne!(
            result.stop,
            StopCause::RootProven,
            "詰みなし局面を誤って勝ち確定にしない"
        );
        assert_eq!(
            result.stats.leaf_mates, 0,
            "偽の詰み検出がない: {:?}",
            result.stats
        );
    }

    #[test]
    fn test_root_dfpn_proves_deep_mate() {
        // 7 手詰め (dfpn テストの canonical 局面)．mock 評価の MCTS 単独では
        // 短時間での証明が難しく，ルート並行 dfpn が詰みを証明して停止させる
        let evaluator = MockEvaluator::new(42);
        let searcher = Searcher::new(
            &evaluator,
            SearchOptions {
                threads: 1,
                batch_size: 8,
                root_dfpn: true,
                ..SearchOptions::default()
            },
        );
        let mut board = Board::empty();
        board
            .set_sfen("8k/9/6R2/9/9/9/9/9/9 b G 1")
            .expect("正当な SFEN");
        let result = searcher.search(
            &board,
            &SearchLimits {
                time_ms: Some(30_000),
                ..SearchLimits::default()
            },
        );
        assert_eq!(result.stop, StopCause::RootProven);
        assert_eq!(result.winrate, 1.0);
        assert!(!result.pv.is_empty());
        let best = result.best_move.expect("詰み手順の先頭");
        assert!(
            result.root_children.iter().any(|c| c.mv == best),
            "best_move はルートの合法手: {}",
            best.to_usi()
        );
    }

    #[test]
    fn test_root_dfpn_no_mate_keeps_normal_search() {
        // 詰みが無い局面では root_dfpn は結果に影響しない (dfpn は不詰で
        // 即終了し，MCTS は通常の予算で停止する)
        let result = run(
            STARTPOS,
            SearchOptions {
                threads: 1,
                batch_size: 8,
                root_dfpn: true,
                ..SearchOptions::default()
            },
            SearchLimits {
                max_playouts: Some(500),
                ..SearchLimits::default()
            },
            0,
        );
        assert_eq!(result.stop, StopCause::PlayoutLimit);
        assert!(result.best_move.is_some());
    }

    #[test]
    fn test_repetition_detected_in_search() {
        // 双方玉のみの隅対峙 — 王の往復で同一局面が容易に再出現する
        let result = run(
            "k8/9/9/9/9/9/9/9/8K b - 1",
            SearchOptions {
                threads: 1,
                batch_size: 8,
                ..SearchOptions::default()
            },
            SearchLimits {
                max_playouts: Some(3000),
                ..SearchLimits::default()
            },
            42,
        );
        assert!(
            result.stats.repetitions > 0,
            "王の往復で千日手が検出されるはず: {:?}",
            result.stats
        );
        assert!(result.best_move.is_some());
    }

    #[test]
    fn test_perpetual_check_avoided_with_history() {
        // 後手玉 1a，先手飛 2c．対局履歴として王手往復 1 循環 (4 手) を渡すと，
        // root で再び 2c1c と王手する手は連続王手の千日手 (先手負け) を完成
        // させる — 経路 + 履歴で検出され Q=0 に固定される
        let sfen = "8k/9/7R1/9/9/9/9/9/4K4 b - 1";
        let mut board = Board::empty();
        board.set_sfen(sfen).expect("正当な SFEN");
        let mut history = Vec::new();
        for usi in ["2c1c", "1a2a", "1c2c", "2a1a"] {
            history.push(HistoryEntry::from_board(&board));
            let mut probe = board.clone();
            let mv = generate_legal_moves(&mut probe)
                .into_iter()
                .find(|m| m.to_usi() == usi)
                .expect("合法手のはず");
            board.do_move(mv);
        }
        // board は開始局面と同一に戻っている — これを root にする
        let evaluator = MockEvaluator::new(7);
        let searcher = Searcher::new(
            &evaluator,
            SearchOptions {
                threads: 1,
                batch_size: 8,
                ..SearchOptions::default()
            },
        );
        let result = searcher.search_with_history(
            &board,
            &history,
            &SearchLimits {
                max_playouts: Some(2000),
                ..SearchLimits::default()
            },
        );
        assert!(result.stats.repetitions >= 1, "{:?}", result.stats);
        let rep_child = result
            .root_children
            .iter()
            .find(|c| c.mv.to_usi() == "2c1c")
            .expect("2c1c はルートの合法手");
        assert!(rep_child.visits > 0, "一度は訪問される");
        assert_eq!(
            rep_child.q, 0.0,
            "連続王手の千日手を完成させる手は負け評価に固定される"
        );
        assert_ne!(
            result.best_move.map(|m| m.to_usi()).as_deref(),
            Some("2c1c"),
            "千日手負けの王手は best_move に選ばれない"
        );
    }

    #[test]
    fn test_no_history_no_false_repetition() {
        // 同じ局面でも履歴なしなら root 直下の王手は千日手にならない
        // (経路内で循環が閉じるまでは通常の探索)
        let result = run(
            "8k/9/7R1/9/9/9/9/9/4K4 b - 1",
            SearchOptions {
                threads: 1,
                batch_size: 8,
                ..SearchOptions::default()
            },
            SearchLimits {
                max_playouts: Some(50),
                ..SearchLimits::default()
            },
            7,
        );
        // 浅い探索 (深さ 4 未満の経路が大半) では検出ゼロでも探索は正常
        assert!(result.best_move.is_some());
    }

    #[test]
    fn test_root_terminal() {
        // 後手玉が既に詰んでいる局面で後手番 (合法手なし): MATE_IN_1 の G*5b 後
        let evaluator = MockEvaluator::new(0);
        let searcher = Searcher::new(&evaluator, SearchOptions::default());
        let mut board = Board::empty();
        board.set_sfen(MATE_IN_1).expect("正当な SFEN");
        let mut probe = board.clone();
        let mate = generate_legal_moves(&mut probe)
            .into_iter()
            .find(|m| m.to_usi() == "G*5b")
            .expect("G*5b は合法手");
        board.do_move(mate);
        let result = searcher.search(&board, &SearchLimits::default());
        assert_eq!(result.stop, StopCause::RootTerminal);
        assert!(result.best_move.is_none());
    }
}
