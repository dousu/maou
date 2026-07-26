//! in-process 自己対局 driver (設計 docs/design/usi-engine/index.md §9)．
//!
//! 1 対局 = [`Agent`] 2 個 (先後) を stdio/プロセスなしで直接駆動する．
//! 評価器 (ONNX session + TRT キャッシュ) はプロセス内 1 個を全対局で共有し，
//! モデルロード/warmup は 1 回だけ行う．終局判定 (宣言/千日手/最大手数/投了)
//! は USI 対局と同じ実装 (`Board::nyugyoku_declarable` /
//! `maou_search::find_repetition` / agent の resign 判断) を使う = 意味論一致．
//!
//! 並列度は「同時対局数」([`SelfplayConfig::parallel`])．評価器の
//! `Mutex<Session>` 直列化が並列時のスループット上限になり得る — バッチ
//! aggregator の採否はこの driver での計測後に判断する (設計 §12 未決 5)．

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use maou_search::repetition::find_repetition;
use maou_search::{build_board_and_history, HistoryEntry, RepetitionOutcome};
use maou_shogi::movegen::generate_legal_moves;
use maou_shogi::types::Color;

use crate::agent::{Agent, EngineConfig, STARTPOS_SFEN};
use crate::backend::{
    build_evaluator, search_options, warmup_evaluator, EngineEvaluator, MaouSearchBackend,
};
use crate::protocol::{BestMoveKind, EngineCommand, GoParams, GuiCommand};

/// 自己対局の設定．
#[derive(Clone, Debug)]
pub struct SelfplayConfig {
    /// プレイヤー A のエンジン設定 (モデル・探索オプション・戦略)．
    /// `usi_ponder` は無視される (ponder は GUI 対局の概念)．
    /// [`SelfplayConfig::sync_max_moves_to_draw`] が true (既定) なら
    /// `max_moves_to_draw` は [`SelfplayConfig::max_moves`] で上書きされる．
    pub engine: EngineConfig,
    /// プレイヤー B のエンジン設定 (`None` = A と同一 = 純粋自己対局)．
    /// Some なら A/B 対戦モード．**モデルは A と同一であること** (評価器を
    /// プロセス内 1 個で共有するため．設定レバーの A/B 比較用)．
    pub engine_b: Option<EngineConfig>,
    /// 対局ごとに A/B の先後を入れ替える (既定 false)．true なら偶数対局で
    /// A が先手，奇数対局で B が先手になり，ペア (2n, 2n+1) は同じ序盤
    /// ランダム系列を共有する (paired 比較で開局サンプリング分散を相殺)．
    pub alternate_colors: bool,
    /// エージェントの `max_moves_to_draw` を [`SelfplayConfig::max_moves`] に
    /// 同期する (既定 true — driver の終局判定と in-search 終端化を一致させる)．
    /// in-search 終端化そのものを A/B するときだけ false にして
    /// engine/engine_b の値を呼び出し側が管理する．
    pub sync_max_moves_to_draw: bool,
    /// 基準局面 SFEN (`None` = 平手初期局面)．
    pub sfen: Option<String>,
    /// 対局数．
    pub games: u32,
    /// 同時対局数 (スレッド数．各対局は探索 `engine.threads` を別途使う)．
    pub parallel: usize,
    /// 1 手あたりの playout 予算 (`go nodes` 相当)．
    pub playouts: Option<u64>,
    /// 1 手あたりの思考時間ミリ秒 (`go movetime` 相当)．
    pub movetime_ms: Option<u64>,
    /// プレイヤー B の playout 予算 (`None` = A と同じ)．A/B 対戦で
    /// **予算差**を作るためのもの — 「強い予算が勝つ」ことの確認は driver が
    /// 棋力差を測れているかの検証になる (設計 §9 の自己対局を計測基盤として
    /// 使う前提の健全性確認)．
    pub playouts_b: Option<u64>,
    /// プレイヤー B の思考時間ミリ秒 (`None` = A と同じ)．
    pub movetime_ms_b: Option<u64>,
    /// 持ち時間モード (`Some` = 実際の時計を回して TimeStrategy に予算を
    /// 決めさせる)．playouts/movetime とは排他，`parallel > 1` とも排他
    /// (壁時計を測るため)．TimeStrategy の定数調整に使う (設計 §12 未決 1)．
    pub clock: Option<ClockSetting>,
    /// 最大手数 (到達で引き分け．到達局面で宣言可能なら手番の勝ち — 電竜戦
    /// ルール)．エージェントの `MaxMovesToDraw` にも同じ値が渡る．
    pub max_moves: u32,
    /// 序盤の driver 直指しランダム手数 (対局多様化用．0 = 無効)．最初の
    /// この手数は探索せず合法手から一様に選ぶ (シード決定的)．
    pub opening_random_plies: u32,
    /// 乱数シード (対局 index と混合するため全対局で異なる系列になる)．
    pub seed: u64,
}

impl Default for SelfplayConfig {
    fn default() -> SelfplayConfig {
        SelfplayConfig {
            engine: EngineConfig::default(),
            engine_b: None,
            alternate_colors: false,
            sync_max_moves_to_draw: true,
            sfen: None,
            games: 1,
            parallel: 1,
            playouts: Some(800),
            movetime_ms: None,
            playouts_b: None,
            movetime_ms_b: None,
            clock: None,
            max_moves: 512,
            opening_random_plies: 0,
            seed: 0,
        }
    }
}

/// 自己対局の持ち時間設定 (USI の `go` に渡す clock をシミュレートする)．
///
/// 実測の消費時間を残時間から引くため，**壁時計に依存する** — 同時対局
/// (`parallel > 1`) では CPU 競合で消費時間が歪むので併用できない．
/// TimeStrategy の定数調整 (設計 §12 未決 1) はこのモードで行う．
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClockSetting {
    /// 各側の初期持ち時間 (ミリ秒)．
    pub initial_ms: u64,
    /// 秒読み (ミリ秒，0 = なし)．
    pub byoyomi_ms: u64,
    /// 1 手ごとのフィッシャー加算 (ミリ秒，0 = なし)．
    pub inc_ms: u64,
}

/// 終局理由．
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GameEndReason {
    /// 合法手なし (詰み)．
    Checkmate,
    /// 投了 (`resign_value` 閾値)．
    Resign,
    /// 入玉宣言勝ち (27 点法)．
    Declaration,
    /// 千日手 (同一局面 4 回) — 引き分け．
    Repetition,
    /// 連続王手の千日手 — 王手をかけ続けた側の負け．
    PerpetualCheck,
    /// 最大手数到達 — 引き分け．
    MaxMoves,
    /// 非合法手または不成立の宣言 (指した側の負け．エージェントのバグ指標)．
    IllegalMove,
    /// 時間切れ (持ち時間モードのみ．負けた側は時間管理が破綻している)．
    Timeout,
}

impl GameEndReason {
    /// 機械可読な名前 (Python 側の出力用)．
    pub fn as_str(&self) -> &'static str {
        match self {
            GameEndReason::Checkmate => "checkmate",
            GameEndReason::Resign => "resign",
            GameEndReason::Declaration => "declaration",
            GameEndReason::Repetition => "repetition",
            GameEndReason::PerpetualCheck => "perpetual_check",
            GameEndReason::MaxMoves => "max_moves",
            GameEndReason::IllegalMove => "illegal_move",
            GameEndReason::Timeout => "timeout",
        }
    }
}

/// 1 対局の結果 (棋譜 = 基準 SFEN + USI 指し手列 + 勝敗)．
#[derive(Clone, Debug)]
pub struct GameOutcome {
    /// 対局番号 (0 始まり)．
    pub game_index: u32,
    /// 基準局面 SFEN．
    pub sfen: String,
    /// USI 指し手列 (基準局面から)．
    pub moves: Vec<String>,
    /// 先手 (基準局面の手番側) をプレイヤー A が持ったか (A/B 対戦の色交代
    /// 記録．純粋自己対局では常に true)．
    pub black_is_a: bool,
    /// 勝者 (`None` = 引き分け)．
    pub winner: Option<Color>,
    /// 終局理由．
    pub reason: GameEndReason,
    /// 両側合計の playout 数 (探索した手のみ．script/ランダム手は 0)．
    pub playouts: u64,
    /// subtree 再利用が発火した手数 (前回木から訪問を引き継いだ手)．
    pub reused_moves: u32,
    /// 前回木から引き継いだ訪問数の合計 (再利用の実効量．`playouts` に対する
    /// 比が「実質何 % 予算が増えたか」— 設計 §12 未決 6 の計測)．
    pub carried_visits: u64,
    /// 対局の壁時計時間 (ミリ秒)．
    pub elapsed_ms: u64,
    /// 終局時の残り持ち時間 [先手, 後手] (持ち時間モードのみ)．
    ///
    /// 時間配分の診断に使う: 使い残しが多ければ配分が保守的すぎ，
    /// 0 付近に張り付いていれば攻めすぎ (設計 §12 未決 1)．
    pub remaining_ms: Option<[u64; 2]>,
}

/// SplitMix64 (シード決定的な軽量乱数．依存を増やさない)．
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// 自己対局を実行して全対局の結果を返す (対局 index 順)．
///
/// `progress` は各対局の終了ごとに worker スレッドから呼ばれる (`Sync` 必須)．
/// 対局中のエラー (バグ指標) は全体のエラーとして返す．
pub fn run_selfplay(
    config: &SelfplayConfig,
    progress: Option<&(dyn Fn(&GameOutcome) + Sync)>,
) -> Result<Vec<GameOutcome>, String> {
    if config.games == 0 {
        return Err("games は 1 以上".to_string());
    }
    if config.parallel == 0 {
        return Err("parallel は 1 以上".to_string());
    }
    if config.max_moves == 0 {
        return Err("max_moves は 1 以上".to_string());
    }
    match (config.playouts, config.movetime_ms, config.clock) {
        (Some(_), None, None) | (None, Some(_), None) | (None, None, Some(_)) => {}
        _ => return Err("playouts / movetime_ms / clock のいずれか 1 つだけを指定".to_string()),
    }
    if config.clock.is_some() && config.parallel > 1 {
        // 消費時間を壁時計で測るため，同時対局の CPU 競合で歪む
        return Err("持ち時間モードは parallel=1 でのみ使える".to_string());
    }
    let sfen = config
        .sfen
        .clone()
        .unwrap_or_else(|| STARTPOS_SFEN.to_string());
    // 基準局面の検証 (不正 SFEN は対局前に弾く)
    build_board_and_history(&sfen, &[]).map_err(|e| e.to_string())?;

    // エージェント設定の正規化: GUI 対局の概念 (ponder) を無効化し，既定では
    // driver の終局判定と in-search 終端化を一致させる (sync_max_moves_to_draw)
    let normalize = |src: &EngineConfig| {
        let mut e = src.clone();
        e.usi_ponder = false;
        if config.sync_max_moves_to_draw {
            e.max_moves_to_draw = config.max_moves;
        }
        e
    };
    let engines: [EngineConfig; 2] = [
        normalize(&config.engine),
        normalize(config.engine_b.as_ref().unwrap_or(&config.engine)),
    ];
    if engines[0].model_path != engines[1].model_path {
        return Err("A/B 対戦は同一モデル前提 (評価器をプロセス内 1 個で共有するため)".to_string());
    }
    // ペア開局 (2n, 2n+1 が同じ序盤ランダム系列): A/B + 色交代のときのみ
    let paired = config.engine_b.is_some() && config.alternate_colors;

    // 評価器はプロセス内 1 個 (モデルロード/warmup 1 回) を全対局で共有する
    let evaluator = build_evaluator(&engines[0])?;
    warmup_evaluator(&evaluator)?;
    let evaluator = Arc::new(evaluator);

    // 探索予算 (A/B で別々にできる — 予算差で棋力差を作る検証用)
    let go_params: [GoParams; 2] = [
        GoParams {
            nodes: config.playouts,
            movetime: config.movetime_ms,
            ..GoParams::default()
        },
        GoParams {
            nodes: config.playouts_b.or(config.playouts),
            movetime: config.movetime_ms_b.or(config.movetime_ms),
            ..GoParams::default()
        },
    ];

    let next = AtomicU32::new(0);
    let results: Mutex<Vec<(u32, Result<GameOutcome, String>)>> =
        Mutex::new(Vec::with_capacity(config.games as usize));
    std::thread::scope(|s| {
        for _ in 0..config.parallel.min(config.games as usize) {
            s.spawn(|| loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                if index >= config.games {
                    break;
                }
                // 色交代: 偶数対局は A が先手，奇数対局は B が先手．ペア時は
                // 開局乱数系列をペア (index/2) で共有する
                let black_is_a = !(config.alternate_colors && index % 2 == 1);
                let rng_index = if paired { index / 2 } else { index };
                let result = play_game(
                    &engines,
                    black_is_a,
                    &evaluator,
                    &sfen,
                    &go_params,
                    config.clock,
                    config.max_moves,
                    config.opening_random_plies,
                    config.seed,
                    rng_index,
                    index,
                );
                if let (Ok(outcome), Some(cb)) = (&result, progress) {
                    cb(outcome);
                }
                results
                    .lock()
                    .expect("results lock は poison しない")
                    .push((index, result));
            });
        }
    });

    let mut results = results.into_inner().expect("全 worker 終了済み");
    results.sort_by_key(|(i, _)| *i);
    results
        .into_iter()
        .map(|(_, r)| r)
        .collect::<Result<Vec<_>, _>>()
}

/// 1 対局を最後まで駆動する．
///
/// `engines` は [A, B] (純粋自己対局では同一設定 2 つ)，`black_is_a` で
/// どちらが先手 (基準局面の手番側) を持つかを決める．`rng_index` は開局
/// ランダム系列の番号 (ペア比較では 2 対局が共有する)．
#[allow(clippy::too_many_arguments)]
fn play_game(
    engines: &[EngineConfig; 2],
    black_is_a: bool,
    evaluator: &Arc<EngineEvaluator>,
    sfen: &str,
    go_params: &[GoParams; 2],
    clock: Option<ClockSetting>,
    max_moves: u32,
    opening_random_plies: u32,
    seed: u64,
    rng_index: u32,
    game_index: u32,
) -> Result<GameOutcome, String> {
    let start = Instant::now();
    // 対局ごとにエージェント 2 個 (先後で独立の探索木 = 実対局と同じ構図)．
    // backend は共有評価器から安価に作られる (モデル再ロードなし)
    let mk_agent = |engine: &EngineConfig| {
        let ev = Arc::clone(evaluator);
        let mut agent = Agent::new(engine.clone(), move |cfg: &EngineConfig| {
            Ok(MaouSearchBackend::from_shared(
                Arc::clone(&ev),
                search_options(cfg),
                cfg.subtree_reuse,
            ))
        });
        agent
            .handle(GuiCommand::IsReady)
            .map(|_| ())
            .map_err(|e| format!("game {game_index}: isready 失敗: {e}"))?;
        Ok::<_, String>(agent)
    };
    // 先後への A/B 割り当て (設定と探索予算は同じ側から取る)
    let (ia, ib) = if black_is_a { (0, 1) } else { (1, 0) };
    let mut black = mk_agent(&engines[ia])?;
    let mut white = mk_agent(&engines[ib])?;
    let (go_black, go_white) = (&go_params[ia], &go_params[ib]);

    let (mut board, _) = build_board_and_history(sfen, &[]).map_err(|e| e.to_string())?;
    let mut entries = vec![HistoryEntry::from_board(&board)];
    let mut counts: HashMap<u64, u32> = HashMap::new();
    counts.insert(board.hash(), 1);
    // 開局系列ごとに異なる決定的シード (SplitMix64 で系列番号を撹拌)
    let mut rng = SplitMix64(seed ^ SplitMix64(rng_index as u64 + 1).next());

    let mut moves: Vec<String> = Vec::new();
    let mut playouts: u64 = 0;
    let mut reused_moves: u32 = 0;
    let mut carried_visits: u64 = 0;
    // 持ち時間モードの残時間 (先手, 後手)．非該当モードでは使わない
    let mut remaining = clock.map(|c| [c.initial_ms, c.initial_ms]);
    let (winner, reason) = loop {
        // 最大手数: 到達局面で宣言可能なら手番の勝ち，さもなくば引き分け
        // (電竜戦ルール．最大手数時の詰みも引き分け)
        if moves.len() as u32 >= max_moves {
            if board.nyugyoku_declarable() {
                break (Some(board.turn()), GameEndReason::Declaration);
            }
            break (None, GameEndReason::MaxMoves);
        }
        let side = board.turn();
        let usi = if (moves.len() as u32) < opening_random_plies {
            // 序盤ランダム手 (driver 直指し — 対局多様化)
            let legal = generate_legal_moves(&mut board.clone());
            if legal.is_empty() {
                break (Some(side.opponent()), GameEndReason::Checkmate);
            }
            let pick = (rng.next() % legal.len() as u64) as usize;
            legal[pick].to_usi()
        } else {
            let (agent, go) = match side {
                Color::Black => (&mut black, go_black),
                Color::White => (&mut white, go_white),
            };
            agent
                .handle(GuiCommand::Position {
                    sfen: Some(sfen.to_string()),
                    moves: moves.clone(),
                })
                .map_err(|e| format!("game {game_index}: position 失敗: {e}"))?;
            // 持ち時間モードでは現在の残時間を go に載せ，消費を実測する
            let go = match (clock, remaining) {
                (Some(c), Some(rem)) => GoParams {
                    clock: crate::protocol::ClockParams {
                        btime: Some(rem[0]),
                        wtime: Some(rem[1]),
                        byoyomi: (c.byoyomi_ms > 0).then_some(c.byoyomi_ms),
                        binc: (c.inc_ms > 0).then_some(c.inc_ms),
                        winc: (c.inc_ms > 0).then_some(c.inc_ms),
                    },
                    ..GoParams::default()
                },
                _ => go.clone(),
            };
            let move_started = Instant::now();
            let out = agent
                .handle(GuiCommand::Go(go))
                .map_err(|e| format!("game {game_index}: go 失敗: {e}"))?;
            // 時計の更新 (消費 → 秒読み/加算)．使い切ったら時間切れ負け
            if let (Some(c), Some(rem)) = (clock, remaining.as_mut()) {
                let idx = usize::from(side == Color::White);
                let spent = move_started.elapsed().as_millis() as u64;
                if spent <= rem[idx] {
                    rem[idx] -= spent;
                } else if spent <= rem[idx] + c.byoyomi_ms {
                    // 秒読みで賄えた (持ち時間は使い切る)
                    rem[idx] = 0;
                } else {
                    break (Some(side.opponent()), GameEndReason::Timeout);
                }
                rem[idx] += c.inc_ms;
            }
            // 探索サマリ info (最後の nodes 付き info) から playout を集計する
            playouts += out
                .iter()
                .rev()
                .find_map(|c| match c {
                    EngineCommand::Info(info) => info.nodes,
                    _ => None,
                })
                .unwrap_or(0);
            // subtree 再利用の実効量 (前回木から引き継いだ訪問数．0 = fresh)
            let carried = agent.last_carried_visits();
            if carried > 0 {
                reused_moves += 1;
                carried_visits += carried;
            }
            let best = out
                .iter()
                .rev()
                .find_map(|c| match c {
                    EngineCommand::BestMove { mv, .. } => Some(mv.clone()),
                    _ => None,
                })
                .ok_or_else(|| format!("game {game_index}: bestmove が返らない"))?;
            match best {
                BestMoveKind::Move(usi) => usi,
                BestMoveKind::Resign => {
                    // 合法手なし = 詰み，あり = 閾値投了 (理由を区別して記録)
                    let mated = generate_legal_moves(&mut board.clone()).is_empty();
                    let reason = if mated {
                        GameEndReason::Checkmate
                    } else {
                        GameEndReason::Resign
                    };
                    break (Some(side.opponent()), reason);
                }
                BestMoveKind::Win => {
                    // 宣言の再検証 (不成立の宣言は宣言側の負け — CSA ルール)
                    if board.nyugyoku_declarable() {
                        break (Some(side), GameEndReason::Declaration);
                    }
                    break (Some(side.opponent()), GameEndReason::IllegalMove);
                }
            }
        };
        // 指し手の検証と適用 (合法手列挙 + USI 表記照合 = USI 対局と同じ規約)
        let Some(mv) = generate_legal_moves(&mut board.clone())
            .into_iter()
            .find(|m| m.to_usi() == usi)
        else {
            break (Some(side.opponent()), GameEndReason::IllegalMove);
        };
        board.do_move(mv);
        moves.push(usi);
        entries.push(HistoryEntry::from_board(&board));
        // 千日手: 実ルールの同一局面 4 回で終局し，探索と同じ分類器
        // (find_repetition) で通常/連続王手を判定する (意味論一致)
        let count = counts.entry(board.hash()).or_insert(0);
        *count += 1;
        if *count >= 4 {
            match find_repetition(&[], &entries) {
                Some(RepetitionOutcome::Loss) => {
                    // 手番側 (次に指す側) が王手をかけ続けた → 手番側の負け
                    break (Some(board.turn().opponent()), GameEndReason::PerpetualCheck);
                }
                Some(RepetitionOutcome::Win) => {
                    break (Some(board.turn()), GameEndReason::PerpetualCheck);
                }
                // Draw (通常の千日手)．None は 4 回目の一致がある以上
                // 理論上到達しないが，安全側で引き分けに倒す
                Some(RepetitionOutcome::Draw) | None => {
                    break (None, GameEndReason::Repetition);
                }
            }
        }
    };

    Ok(GameOutcome {
        game_index,
        sfen: sfen.to_string(),
        moves,
        black_is_a,
        winner,
        reason,
        playouts,
        reused_moves,
        carried_visits,
        elapsed_ms: start.elapsed().as_millis() as u64,
        remaining_ms: remaining,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// テスト用の軽量エンジン設定 (mock 評価器 + 小さい木 + 詰み探索 off)．
    fn test_engine() -> EngineConfig {
        EngineConfig {
            root_dfpn: Some(false),
            leaf_mate: Some(false),
            node_capacity: Some(1 << 12),
            ..EngineConfig::default()
        }
    }

    fn config(games: u32, playouts: u64, max_moves: u32) -> SelfplayConfig {
        SelfplayConfig {
            engine: test_engine(),
            games,
            playouts: Some(playouts),
            max_moves,
            ..SelfplayConfig::default()
        }
    }

    #[test]
    fn test_selfplay_smoke_single_game() {
        let outcomes = run_selfplay(
            &SelfplayConfig {
                opening_random_plies: 4,
                seed: 42,
                ..config(1, 16, 32)
            },
            None,
        )
        .expect("mock 自己対局は成功する");
        assert_eq!(outcomes.len(), 1);
        let o = &outcomes[0];
        assert_eq!(o.game_index, 0);
        assert!(o.moves.len() as u32 <= 32);
        assert!(o.playouts > 0, "探索した手があるはず");
        if o.reason == GameEndReason::MaxMoves || o.reason == GameEndReason::Repetition {
            assert!(o.winner.is_none() || o.reason != GameEndReason::MaxMoves);
        }
    }

    #[test]
    fn test_selfplay_max_moves_draw() {
        let outcomes = run_selfplay(&config(1, 16, 4), None).expect("成功");
        let o = &outcomes[0];
        assert_eq!(o.reason, GameEndReason::MaxMoves);
        assert_eq!(o.winner, None);
        assert_eq!(o.moves.len(), 4);
    }

    #[test]
    fn test_selfplay_resign_threshold() {
        // mock の root 勝率 ~0.5 < 0.9 → 初手で投了 (consecutive 1)
        let mut cfg = config(1, 16, 64);
        cfg.engine.resign_value = 900;
        cfg.engine.resign_consecutive = 1;
        let outcomes = run_selfplay(&cfg, None).expect("成功");
        let o = &outcomes[0];
        assert_eq!(o.reason, GameEndReason::Resign);
        assert_eq!(o.winner, Some(Color::White), "先手が投了 → 後手勝ち");
        assert!(o.moves.is_empty());
    }

    #[test]
    fn test_selfplay_declaration_immediate() {
        // 先手が即宣言可能な局面 (board.rs の 28 点 golden fixture)
        let mut cfg = config(1, 16, 64);
        cfg.sfen = Some("K+R+BGGGGSS/PPPP5/9/9/9/9/9/9/8k b B3P 1".to_string());
        let outcomes = run_selfplay(&cfg, None).expect("成功");
        let o = &outcomes[0];
        assert_eq!(o.reason, GameEndReason::Declaration);
        assert_eq!(o.winner, Some(Color::Black));
        assert!(o.moves.is_empty());
    }

    #[test]
    fn test_selfplay_checkmate_immediate() {
        // 手番 (後手) が詰まされている局面 → 即 Checkmate で先手勝ち
        let mut cfg = config(1, 16, 64);
        cfg.sfen = Some("4k4/4G4/4P4/9/9/9/9/9/9 w - 1".to_string());
        let outcomes = run_selfplay(&cfg, None).expect("成功");
        let o = &outcomes[0];
        assert_eq!(o.reason, GameEndReason::Checkmate);
        assert_eq!(o.winner, Some(Color::Black));
        assert!(o.moves.is_empty());
    }

    #[test]
    fn test_selfplay_opening_script_followed_by_both_sides() {
        // M4 完了条件「強制手順対局」: 両側が script どおり指す
        let script = "7g7f 3c3d 2g2f 8c8d";
        let mut cfg = config(1, 16, 8);
        cfg.engine.opening_script = Some(script.to_string());
        let outcomes = run_selfplay(&cfg, None).expect("成功");
        let o = &outcomes[0];
        let played: Vec<&str> = o.moves.iter().take(4).map(String::as_str).collect();
        assert_eq!(played.join(" "), script, "最初の 4 手は script どおり");
        assert_eq!(o.reason, GameEndReason::MaxMoves);
    }

    #[test]
    fn test_selfplay_parallel_multiple_games() {
        let done = Mutex::new(0u32);
        let outcomes = run_selfplay(
            &SelfplayConfig {
                parallel: 2,
                opening_random_plies: 2,
                seed: 7,
                ..config(4, 8, 12)
            },
            Some(&|_o: &GameOutcome| {
                *done.lock().unwrap() += 1;
            }),
        )
        .expect("成功");
        assert_eq!(outcomes.len(), 4);
        // index 順に整列して返る
        let indices: Vec<u32> = outcomes.iter().map(|o| o.game_index).collect();
        assert_eq!(indices, vec![0, 1, 2, 3]);
        assert_eq!(*done.lock().unwrap(), 4, "progress は対局ごとに呼ばれる");
    }

    #[test]
    fn test_selfplay_bare_kings_terminates() {
        // 裸玉同士: 千日手 (同一局面 4 回) か最大手数で必ず終局する
        let mut cfg = config(1, 8, 200);
        cfg.sfen = Some("4k4/9/9/9/9/9/9/9/4K4 b - 1".to_string());
        let outcomes = run_selfplay(&cfg, None).expect("成功");
        let o = &outcomes[0];
        assert!(
            o.reason == GameEndReason::Repetition || o.reason == GameEndReason::MaxMoves,
            "裸玉は引き分けで終わる: {:?}",
            o.reason
        );
    }

    #[test]
    fn test_selfplay_ab_alternates_colors_and_maps_winner() {
        // B は即投了する設定 → 色がどちらでも勝者はプレイヤー A になる
        let mut cfg = config(2, 16, 64);
        let mut b = cfg.engine.clone();
        b.resign_value = 900;
        b.resign_consecutive = 1;
        cfg.engine_b = Some(b);
        cfg.alternate_colors = true;
        let outcomes = run_selfplay(&cfg, None).expect("成功");
        assert_eq!(outcomes.len(), 2);
        // 偶数対局: A が先手，B (後手) が自分の初手で投了 → 先手勝ち
        assert!(outcomes[0].black_is_a);
        assert_eq!(outcomes[0].winner, Some(Color::Black));
        assert_eq!(outcomes[0].reason, GameEndReason::Resign);
        // 奇数対局: B が先手で即投了 → 後手 (A) 勝ち
        assert!(!outcomes[1].black_is_a);
        assert_eq!(outcomes[1].winner, Some(Color::White));
        assert_eq!(outcomes[1].reason, GameEndReason::Resign);
    }

    #[test]
    fn test_selfplay_ab_paired_openings_share_random_series() {
        // ペア (2n, 2n+1) は同じ開局ランダム系列を共有する
        let mut cfg = config(2, 8, 12);
        cfg.engine_b = Some(cfg.engine.clone());
        cfg.alternate_colors = true;
        cfg.opening_random_plies = 4;
        cfg.seed = 99;
        let outcomes = run_selfplay(&cfg, None).expect("成功");
        assert_eq!(outcomes[0].moves[..4], outcomes[1].moves[..4]);
    }

    #[test]
    fn test_selfplay_subtree_reuse_toggle_changes_carry() {
        // on: 手が進むと reroot が成功して訪問を引き継ぐ手が現れる
        let on = run_selfplay(&config(1, 64, 12), None).expect("成功");
        assert_eq!(on[0].reason, GameEndReason::MaxMoves);
        assert!(
            on[0].reused_moves > 0 && on[0].carried_visits > 0,
            "再利用 on では引き継ぎが発生するはず: {} moves / {} visits",
            on[0].reused_moves,
            on[0].carried_visits
        );
        // off: 常に fresh なので引き継ぎは 0 (計測トグルの効きの検証)
        let mut cfg = config(1, 64, 12);
        cfg.engine.subtree_reuse = false;
        let off = run_selfplay(&cfg, None).expect("成功");
        assert_eq!(off[0].reason, GameEndReason::MaxMoves);
        assert_eq!(off[0].reused_moves, 0);
        assert_eq!(off[0].carried_visits, 0);
    }

    #[test]
    fn test_selfplay_ab_per_side_budget() {
        // A に 4 倍の予算を与えると，その分だけ playout が積まれる
        // (予算差で棋力差を作れる = ハーネス健全性確認の前提)
        let mut cfg = config(1, 64, 6);
        cfg.engine_b = Some(cfg.engine.clone());
        cfg.playouts_b = Some(16);
        let outcomes = run_selfplay(&cfg, None).expect("成功");
        let o = &outcomes[0];
        assert_eq!(o.moves.len(), 6);
        // 先手 3 手 × 64 + 後手 3 手 × 16 = 240 前後 (バッチ粒度で超過し得る)
        assert!(
            (200..400).contains(&o.playouts),
            "A/B 予算差が playout 合計に出るはず: {}",
            o.playouts
        );
    }

    #[test]
    fn test_selfplay_clock_mode_runs_and_consumes_time() {
        // 持ち時間モード: 時計から予算が決まり，時間切れせずに終局する
        let mut cfg = config(1, 16, 10);
        cfg.playouts = None;
        cfg.clock = Some(ClockSetting {
            initial_ms: 2_000,
            byoyomi_ms: 0,
            inc_ms: 50,
        });
        let outcomes = run_selfplay(&cfg, None).expect("成功");
        let o = &outcomes[0];
        assert_eq!(o.reason, GameEndReason::MaxMoves, "{:?}", o.reason);
        assert!(o.playouts > 0, "時計由来の予算で探索している");
        // 終局時の残り時間が記録される (時間配分の診断に使う)
        let rem = o.remaining_ms.expect("持ち時間モードでは記録される");
        assert!(
            rem.iter().all(|&v| v <= 2_000 + 10 * 50),
            "残り時間が初期値 + 加算を超えている: {rem:?}"
        );
    }

    #[test]
    fn test_selfplay_clock_mode_rejects_parallel_and_budget_mix() {
        let mut cfg = config(2, 16, 10);
        cfg.playouts = None;
        cfg.parallel = 2;
        cfg.clock = Some(ClockSetting {
            initial_ms: 1_000,
            byoyomi_ms: 0,
            inc_ms: 10,
        });
        assert!(
            run_selfplay(&cfg, None).is_err(),
            "壁時計計測なので並列は不可"
        );
        // playouts と clock の同時指定も不可 (予算の出所が二重になる)
        let mut cfg = config(1, 16, 10);
        cfg.clock = Some(ClockSetting {
            initial_ms: 1_000,
            byoyomi_ms: 0,
            inc_ms: 10,
        });
        assert!(run_selfplay(&cfg, None).is_err());
    }

    #[test]
    fn test_selfplay_clock_timeout_is_a_loss() {
        // 実質ゼロ秒の持ち時間 + 秒読みなし → 最低思考時間を使った時点で
        // 時間切れになり，相手の勝ちになる (時間管理破綻の検出経路)
        let mut cfg = config(1, 16, 40);
        cfg.playouts = None;
        cfg.engine.time.min_think_ms = 50;
        cfg.clock = Some(ClockSetting {
            initial_ms: 1,
            byoyomi_ms: 0,
            inc_ms: 0,
        });
        let outcomes = run_selfplay(&cfg, None).expect("成功");
        let o = &outcomes[0];
        assert_eq!(o.reason, GameEndReason::Timeout);
        assert_eq!(o.winner, Some(Color::White), "先手が時間切れ");
    }

    #[test]
    fn test_selfplay_ab_rejects_different_models() {
        let mut cfg = config(1, 16, 8);
        let mut b = cfg.engine.clone();
        b.model_path = Some("other.onnx".to_string());
        cfg.engine_b = Some(b);
        assert!(run_selfplay(&cfg, None).is_err(), "異モデルはエラー");
    }

    #[test]
    fn test_selfplay_config_validation() {
        assert!(run_selfplay(&config(0, 16, 32), None).is_err(), "games 0");
        let mut cfg = config(1, 16, 32);
        cfg.playouts = None;
        assert!(run_selfplay(&cfg, None).is_err(), "予算未指定");
        let mut cfg = config(1, 16, 32);
        cfg.sfen = Some("invalid".to_string());
        assert!(run_selfplay(&cfg, None).is_err(), "不正 SFEN");
    }
}
