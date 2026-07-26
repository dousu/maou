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
    /// 両側エージェント共通のエンジン設定 (モデル・探索オプション・戦略)．
    /// `usi_ponder` は無視され (ponder は GUI 対局の概念)，`max_moves_to_draw`
    /// は [`SelfplayConfig::max_moves`] で上書きされる (driver の終局判定と
    /// エージェントの in-search 終端化を一致させる)．
    pub engine: EngineConfig,
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
            sfen: None,
            games: 1,
            parallel: 1,
            playouts: Some(800),
            movetime_ms: None,
            max_moves: 512,
            opening_random_plies: 0,
            seed: 0,
        }
    }
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
    /// 勝者 (`None` = 引き分け)．
    pub winner: Option<Color>,
    /// 終局理由．
    pub reason: GameEndReason,
    /// 両側合計の playout 数 (探索した手のみ．script/ランダム手は 0)．
    pub playouts: u64,
    /// 対局の壁時計時間 (ミリ秒)．
    pub elapsed_ms: u64,
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
    match (config.playouts, config.movetime_ms) {
        (Some(_), None) | (None, Some(_)) => {}
        _ => return Err("playouts か movetime_ms のどちらか一方を指定".to_string()),
    }
    let sfen = config
        .sfen
        .clone()
        .unwrap_or_else(|| STARTPOS_SFEN.to_string());
    // 基準局面の検証 (不正 SFEN は対局前に弾く)
    build_board_and_history(&sfen, &[]).map_err(|e| e.to_string())?;

    // エージェント設定の正規化: driver の終局判定と in-search 終端化を一致させ，
    // GUI 対局の概念 (ponder) を無効化する
    let mut engine = config.engine.clone();
    engine.max_moves_to_draw = config.max_moves;
    engine.usi_ponder = false;

    // 評価器はプロセス内 1 個 (モデルロード/warmup 1 回) を全対局で共有する
    let evaluator = build_evaluator(&engine)?;
    warmup_evaluator(&evaluator)?;
    let evaluator = Arc::new(evaluator);

    let go_params = GoParams {
        nodes: config.playouts,
        movetime: config.movetime_ms,
        ..GoParams::default()
    };

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
                let result = play_game(
                    &engine,
                    &evaluator,
                    &sfen,
                    &go_params,
                    config.max_moves,
                    config.opening_random_plies,
                    config.seed,
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
#[allow(clippy::too_many_arguments)]
fn play_game(
    engine: &EngineConfig,
    evaluator: &Arc<EngineEvaluator>,
    sfen: &str,
    go_params: &GoParams,
    max_moves: u32,
    opening_random_plies: u32,
    seed: u64,
    game_index: u32,
) -> Result<GameOutcome, String> {
    let start = Instant::now();
    // 対局ごとにエージェント 2 個 (先後で独立の探索木 = 実対局と同じ構図)．
    // backend は共有評価器から安価に作られる (モデル再ロードなし)
    let mk_agent = || {
        let ev = Arc::clone(evaluator);
        let mut agent = Agent::new(engine.clone(), move |cfg: &EngineConfig| {
            Ok(MaouSearchBackend::from_shared(
                Arc::clone(&ev),
                search_options(cfg),
            ))
        });
        agent
            .handle(GuiCommand::IsReady)
            .map(|_| ())
            .map_err(|e| format!("game {game_index}: isready 失敗: {e}"))?;
        Ok::<_, String>(agent)
    };
    let mut black = mk_agent()?;
    let mut white = mk_agent()?;

    let (mut board, _) = build_board_and_history(sfen, &[]).map_err(|e| e.to_string())?;
    let mut entries = vec![HistoryEntry::from_board(&board)];
    let mut counts: HashMap<u64, u32> = HashMap::new();
    counts.insert(board.hash(), 1);
    // 対局ごとに異なる決定的シード (SplitMix64 で index を撹拌)
    let mut rng = SplitMix64(seed ^ SplitMix64(game_index as u64 + 1).next());

    let mut moves: Vec<String> = Vec::new();
    let mut playouts: u64 = 0;
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
            let agent = match side {
                Color::Black => &mut black,
                Color::White => &mut white,
            };
            agent
                .handle(GuiCommand::Position {
                    sfen: Some(sfen.to_string()),
                    moves: moves.clone(),
                })
                .map_err(|e| format!("game {game_index}: position 失敗: {e}"))?;
            let out = agent
                .handle(GuiCommand::Go(go_params.clone()))
                .map_err(|e| format!("game {game_index}: go 失敗: {e}"))?;
            // 探索サマリ info (最後の nodes 付き info) から playout を集計する
            playouts += out
                .iter()
                .rev()
                .find_map(|c| match c {
                    EngineCommand::Info(info) => info.nodes,
                    _ => None,
                })
                .unwrap_or(0);
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
        winner,
        reason,
        playouts,
        elapsed_ms: start.elapsed().as_millis() as u64,
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
