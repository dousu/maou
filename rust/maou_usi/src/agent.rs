//! 対局エージェント — USI コマンド (型付き) を受けて応答を返す状態機械．
//!
//! transport (stdio/自己対局 driver) 非依存 (docs/design/usi-engine/index.md §4)．
//! 探索は [`SearchBackend`] trait 越しに呼ぶため，fake backend で状態機械を
//! 単体テストできる．時間管理は [`crate::time`] (戦略レイヤー) に委譲する．

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use maou_shogi::types::Color;

use crate::protocol::{
    BestMoveKind, EngineCommand, GameResult, GoParams, GuiCommand, Info, OptionDecl, OptionKind,
    Score,
};
use crate::time::{allocate, TimeStrategyConfig};

/// 平手初期局面 (USI `position startpos`)．
pub const STARTPOS_SFEN: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

/// `USI_Hash` (MB) → ノードプール容量の換算に使う 1 ノードあたりの概算バイト数
/// (Node 本体 + Edge 配列の平均的な合計．将来 NodePool の実測で較正する —
/// 設計 doc §12 未決事項 3)．
const APPROX_BYTES_PER_NODE: u64 = 512;

/// エンジン設定 (CLI 初期値 + `setoption` 上書き)．
#[derive(Clone, Debug)]
pub struct EngineConfig {
    /// `id name` に出す名前．
    pub engine_name: String,
    /// `id author` に出す作者名．
    pub engine_author: String,
    /// ONNX モデルパス (`None`/空 = mock 評価器)．
    pub model_path: Option<String>,
    /// 探索スレッド数．
    pub threads: usize,
    /// 評価バッチサイズ．
    pub batch_size: usize,
    /// ノードプール容量 (`None` = `usi_hash_mb` から換算，どちらもなければ既定)．
    pub node_capacity: Option<u32>,
    /// `USI_Hash` (MB)．`node_capacity` 未指定時のみ換算に使う．
    pub usi_hash_mb: Option<u64>,
    /// CUDA Execution Provider を使う (onnx-cuda feature 必要)．
    pub use_cuda: bool,
    /// TensorRT Execution Provider を使う (onnx-tensorrt feature 必要)．
    pub use_tensorrt: bool,
    /// TensorRT エンジンキャッシュ保存先．
    pub trt_cache_dir: Option<String>,
    /// 時間戦略の設定．
    pub time: TimeStrategyConfig,
    /// ルート並行 dfpn (None = maou_search 既定)．
    pub root_dfpn: Option<bool>,
    /// ルート dfpn ノード予算．
    pub root_dfpn_nodes: Option<u64>,
    /// ルート dfpn 深さ上限．
    pub root_dfpn_depth: Option<u32>,
    /// leaf-mate (None = maou_search 既定)．
    pub leaf_mate: Option<bool>,
    /// leaf-mate ノード予算．
    pub leaf_mate_nodes: Option<u64>,
    /// leaf-mate スレッド数．
    pub leaf_mate_threads: Option<usize>,
}

impl Default for EngineConfig {
    fn default() -> EngineConfig {
        EngineConfig {
            engine_name: format!("maou {}", env!("CARGO_PKG_VERSION")),
            engine_author: "dousu".to_string(),
            model_path: None,
            threads: 1,
            batch_size: 8,
            node_capacity: None,
            usi_hash_mb: None,
            use_cuda: false,
            use_tensorrt: false,
            trt_cache_dir: None,
            time: TimeStrategyConfig::default(),
            root_dfpn: None,
            root_dfpn_nodes: None,
            root_dfpn_depth: None,
            leaf_mate: None,
            leaf_mate_nodes: None,
            leaf_mate_threads: None,
        }
    }
}

impl EngineConfig {
    /// ノードプール容量の実効値 (優先: NodeCapacity > USI_Hash 換算 > 既定)．
    pub fn effective_node_capacity(&self) -> Option<u32> {
        if let Some(v) = self.node_capacity {
            return Some(v);
        }
        self.usi_hash_mb.map(|mb| {
            let nodes = (mb * 1024 * 1024 / APPROX_BYTES_PER_NODE).max(1024);
            u32::try_from(nodes).unwrap_or(u32::MAX)
        })
    }
}

/// 1 回の探索予算 (エージェント → バックエンド)．
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SearchBudget {
    /// 時間予算 (ミリ秒)．
    pub time_ms: Option<u64>,
    /// playout 予算．
    pub max_playouts: Option<u64>,
    /// 無期限 (stop でのみ停止．`go ponder`/`go infinite`)．
    pub unbounded: bool,
}

/// 探索結果 (バックエンド → エージェント)．
#[derive(Clone, Debug, Default)]
pub struct SearchOutcome {
    /// 最有力手 (USI 表記．`None` = 合法手なし = 詰まされている)．
    pub best_usi: Option<String>,
    /// root 手番側から見た勝率．
    pub winrate: f64,
    /// 読み筋 (USI 表記)．
    pub pv: Vec<String>,
    /// 完了 playout 数．
    pub playouts: u64,
    /// 所要時間 (ミリ秒，warmup 込みの壁時計)．
    pub elapsed_ms: u64,
    /// playout 毎秒．
    pub nps: u64,
    /// 到達最大深さ．
    pub max_depth: u16,
    /// root の確定値 (1.0 = 勝ち確定/詰み発見，0.0 = 負け確定，0.5 = 引き分け
    /// 確定)．未確定は `None`．
    pub proven: Option<f64>,
}

/// 探索バックエンド (実装: [`MaouSearchBackend`]，テスト: fake)．
pub trait SearchBackend {
    /// 局面 (基準 SFEN + USI 経路 = 千日手履歴規約) を予算内で探索する．
    ///
    /// `stop` が立てられたら途中で打ち切ってその時点の最有力手を返すこと．
    fn search(
        &mut self,
        sfen: &str,
        moves: &[String],
        budget: &SearchBudget,
        stop: &Arc<AtomicBool>,
    ) -> Result<SearchOutcome, String>;

    /// mock 評価器か (isready 時の明示に使う)．
    fn is_mock(&self) -> bool;
}

/// 対局状態．
#[derive(Clone, Debug)]
struct GameState {
    /// 基準局面 SFEN．
    sfen: String,
    /// 基準局面からの USI 指し手列．
    moves: Vec<String>,
}

impl Default for GameState {
    fn default() -> GameState {
        GameState {
            sfen: STARTPOS_SFEN.to_string(),
            moves: Vec::new(),
        }
    }
}

impl GameState {
    /// 現局面の手番 (基準 SFEN の手番を指し手数で反転)．
    fn side_to_move(&self) -> Color {
        let base = match self.sfen.split_whitespace().nth(1) {
            Some("w") => Color::White,
            _ => Color::Black,
        };
        if self.moves.len().is_multiple_of(2) {
            base
        } else {
            base.opponent()
        }
    }
}

/// 勝率 → 評価値 (センチポーン) 変換．
///
/// Ponanza 定数系: `winrate = 1 / (1 + exp(-cp/600))` の逆関数．
pub fn winrate_to_cp(winrate: f64) -> i32 {
    let w = winrate.clamp(1e-6, 1.0 - 1e-6);
    (600.0 * (w / (1.0 - w)).ln()).round() as i32
}

/// 対局エージェント．
///
/// `factory` はバックエンド (評価器) の構築関数 — `isready` 時に呼ばれる
/// (モデルロード/warmup はそこで走る)．評価器に関わるオプションが後から
/// 変更されたらバックエンドを破棄し，次の `isready` で再構築する．
pub struct Agent<B, F>
where
    B: SearchBackend,
    F: Fn(&EngineConfig) -> Result<B, String>,
{
    config: EngineConfig,
    factory: F,
    backend: Option<B>,
    game: GameState,
    /// 現在の探索の協調停止フラグ．transport (stdio reader) が `go` 行で
    /// false，`stop`/`quit` 行で true にする ([`Agent::stop_handle`])．
    stop: Arc<AtomicBool>,
}

impl<B, F> Agent<B, F>
where
    B: SearchBackend,
    F: Fn(&EngineConfig) -> Result<B, String>,
{
    /// エージェントを作る (バックエンドは未構築 — `isready` で構築される)．
    pub fn new(config: EngineConfig, factory: F) -> Agent<B, F> {
        Agent {
            config,
            factory,
            backend: None,
            game: GameState::default(),
            stop: Arc::new(AtomicBool::new(false)),
        }
    }

    /// 探索停止フラグのハンドル．
    ///
    /// transport 側の規約 (行の読み取り順で更新することで race を避ける):
    /// `go` 行を読んだら false，`stop`/`quit` 行を読んだら true を store する．
    pub fn stop_handle(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.stop)
    }

    /// コマンドを 1 つ処理して応答列を返す．
    ///
    /// `Go` は探索が終わるまでブロックする (探索の中断は [`Agent::stop_handle`]
    /// を別スレッドから立てる)．`Quit` の後始末は transport 側 (ループ終了)．
    pub fn handle(&mut self, cmd: GuiCommand) -> Result<Vec<EngineCommand>, String> {
        match cmd {
            GuiCommand::Usi => Ok(self.handle_usi()),
            GuiCommand::IsReady => self.handle_isready(),
            GuiCommand::SetOption { name, value } => {
                self.handle_setoption(&name, value.as_deref());
                Ok(Vec::new())
            }
            GuiCommand::UsiNewGame => {
                self.game = GameState::default();
                Ok(Vec::new())
            }
            GuiCommand::Position { sfen, moves } => {
                self.game = GameState {
                    sfen: sfen.unwrap_or_else(|| STARTPOS_SFEN.to_string()),
                    moves,
                };
                Ok(Vec::new())
            }
            GuiCommand::Go(params) => self.handle_go(&params),
            // M1: 同期探索のため Stop 処理時点で探索は終わっている
            // (bestmove は Go の応答として送信済み)．ここでは何もしない
            GuiCommand::Stop => Ok(Vec::new()),
            // M1: ponder 未対応 (USI_Ponder を宣言しないため通常来ない)
            GuiCommand::PonderHit(_) => Ok(Vec::new()),
            GuiCommand::GameOver(result) => {
                let _: GameResult = result;
                self.game = GameState::default();
                Ok(Vec::new())
            }
            GuiCommand::Quit => Ok(Vec::new()),
            // 未知コマンドは無視 (方言・拡張への耐性)
            GuiCommand::Unknown(_) => Ok(Vec::new()),
        }
    }

    fn handle_usi(&self) -> Vec<EngineCommand> {
        let mut out = vec![EngineCommand::Id {
            name: self.config.engine_name.clone(),
            author: self.config.engine_author.clone(),
        }];
        out.extend(
            self.option_decls()
                .into_iter()
                .map(EngineCommand::OptionDecl),
        );
        out.push(EngineCommand::UsiOk);
        out
    }

    /// オプション宣言 (default は現在の設定値 = CLI 初期値を反映する)．
    fn option_decls(&self) -> Vec<OptionDecl> {
        let c = &self.config;
        vec![
            OptionDecl {
                name: "ModelPath",
                kind: OptionKind::Filename {
                    default: c.model_path.clone().unwrap_or_default(),
                },
            },
            OptionDecl {
                name: "Threads",
                kind: OptionKind::Spin {
                    default: c.threads as i64,
                    min: 1,
                    max: 256,
                },
            },
            OptionDecl {
                name: "BatchSize",
                kind: OptionKind::Spin {
                    default: c.batch_size as i64,
                    min: 1,
                    max: 4096,
                },
            },
            OptionDecl {
                name: "NodeCapacity",
                kind: OptionKind::Spin {
                    default: i64::from(c.node_capacity.unwrap_or(1 << 20)),
                    min: 1 << 10,
                    max: i64::from(u32::MAX),
                },
            },
            OptionDecl {
                name: "USI_Hash",
                kind: OptionKind::Spin {
                    default: c.usi_hash_mb.map_or(0, |v| v as i64),
                    min: 0,
                    max: 1 << 20,
                },
            },
            OptionDecl {
                name: "UseCuda",
                kind: OptionKind::Check {
                    default: c.use_cuda,
                },
            },
            OptionDecl {
                name: "UseTensorRT",
                kind: OptionKind::Check {
                    default: c.use_tensorrt,
                },
            },
            OptionDecl {
                name: "TrtCacheDir",
                kind: OptionKind::String {
                    default: c.trt_cache_dir.clone().unwrap_or_default(),
                },
            },
            OptionDecl {
                name: "NetworkDelay",
                kind: OptionKind::Spin {
                    default: c.time.network_delay_ms as i64,
                    min: 0,
                    max: 60_000,
                },
            },
            OptionDecl {
                name: "MinimumThinkingTime",
                kind: OptionKind::Spin {
                    default: c.time.min_think_ms as i64,
                    min: 0,
                    max: 60_000,
                },
            },
            OptionDecl {
                name: "RootDfpn",
                kind: OptionKind::Check {
                    default: c.root_dfpn.unwrap_or(true),
                },
            },
            OptionDecl {
                name: "LeafMate",
                kind: OptionKind::Check {
                    default: c.leaf_mate.unwrap_or(true),
                },
            },
        ]
    }

    fn handle_isready(&mut self) -> Result<Vec<EngineCommand>, String> {
        let mut out = Vec::new();
        if self.backend.is_none() {
            let backend = (self.factory)(&self.config)?;
            if backend.is_mock() {
                // モデル未指定は mock 評価器 + UI 明示 (analyze-gui と同じ慣例)
                out.push(EngineCommand::Info(Info {
                    string: Some(
                        "mock evaluator (development only) - set ModelPath for real play"
                            .to_string(),
                    ),
                    ..Info::default()
                }));
            }
            self.backend = Some(backend);
        }
        out.push(EngineCommand::ReadyOk);
        Ok(out)
    }

    fn handle_setoption(&mut self, name: &str, value: Option<&str>) {
        let v = value.unwrap_or("");
        let parse_bool = || v.eq_ignore_ascii_case("true");
        // 評価器/探索リソースに関わる変更はバックエンドを破棄して次の
        // isready で再構築する (将棋所は対局ごとに isready を送る)
        let mut invalidate = true;
        match name {
            "ModelPath" => {
                self.config.model_path = if v.is_empty() {
                    None
                } else {
                    Some(v.to_string())
                };
            }
            "Threads" => {
                if let Ok(n) = v.parse() {
                    self.config.threads = n;
                }
            }
            "BatchSize" => {
                if let Ok(n) = v.parse() {
                    self.config.batch_size = n;
                }
            }
            "NodeCapacity" => {
                if let Ok(n) = v.parse() {
                    self.config.node_capacity = Some(n);
                }
            }
            "USI_Hash" => {
                if let Ok(n) = v.parse::<u64>() {
                    self.config.usi_hash_mb = if n == 0 { None } else { Some(n) };
                }
            }
            "UseCuda" => self.config.use_cuda = parse_bool(),
            "UseTensorRT" => self.config.use_tensorrt = parse_bool(),
            "TrtCacheDir" => {
                self.config.trt_cache_dir = if v.is_empty() {
                    None
                } else {
                    Some(v.to_string())
                };
            }
            "RootDfpn" => self.config.root_dfpn = Some(parse_bool()),
            "LeafMate" => self.config.leaf_mate = Some(parse_bool()),
            "NetworkDelay" => {
                if let Ok(n) = v.parse() {
                    self.config.time.network_delay_ms = n;
                }
                invalidate = false;
            }
            "MinimumThinkingTime" => {
                if let Ok(n) = v.parse() {
                    self.config.time.min_think_ms = n;
                }
                invalidate = false;
            }
            // 未知のオプションは無視 (GUI 側の拡張への耐性)
            _ => invalidate = false,
        }
        if invalidate {
            self.backend = None;
        }
    }

    fn handle_go(&mut self, params: &GoParams) -> Result<Vec<EngineCommand>, String> {
        if params.mate.is_some() {
            // 詰み探索モードは M1 では未対応 (dfpn 接続は将来対応)
            return Ok(vec![EngineCommand::CheckmateNotImplemented]);
        }

        // isready を送らない GUI への防御 (通常は isready で構築済み)
        if self.backend.is_none() {
            let backend = (self.factory)(&self.config)?;
            self.backend = Some(backend);
        }

        let budget = self.decide_budget(params);
        let backend = self.backend.as_mut().expect("直前で構築済み");
        let outcome = backend.search(&self.game.sfen, &self.game.moves, &budget, &self.stop)?;

        let mut out = Vec::new();
        out.push(EngineCommand::Info(build_info(&outcome)));
        let mv = match outcome.best_usi {
            // 合法手なし = 詰まされている → 投了
            None => BestMoveKind::Resign,
            Some(usi) => BestMoveKind::Move(usi),
        };
        // M1: ponder 非対応のため bestmove に ponder は付けない
        out.push(EngineCommand::BestMove { mv, ponder: None });
        Ok(out)
    }

    /// go パラメータ → 探索予算 (時間戦略は crate::time)．
    fn decide_budget(&self, params: &GoParams) -> SearchBudget {
        if params.infinite || params.ponder {
            return SearchBudget {
                time_ms: None,
                max_playouts: None,
                unbounded: true,
            };
        }
        if let Some(nodes) = params.nodes {
            return SearchBudget {
                time_ms: None,
                max_playouts: Some(nodes),
                unbounded: false,
            };
        }
        if let Some(movetime) = params.movetime {
            let ms = movetime
                .saturating_sub(self.config.time.network_delay_ms)
                .max(self.config.time.min_think_ms);
            return SearchBudget {
                time_ms: Some(ms),
                max_playouts: None,
                unbounded: false,
            };
        }
        let budget = allocate(&self.config.time, &params.clock, self.game.side_to_move());
        SearchBudget {
            time_ms: Some(budget.hard_ms),
            max_playouts: None,
            unbounded: false,
        }
    }
}

/// 探索結果 → `info` 1 行 (探索サマリ)．
fn build_info(outcome: &SearchOutcome) -> Info {
    let score = match outcome.proven {
        // 勝ち確定 (詰み発見含む): PV 長を詰み手数として報告する
        Some(v) if v >= 1.0 => Some(Score::Mate(outcome.pv.len().max(1) as i32)),
        Some(v) if v <= 0.0 => Some(Score::Mate(-(outcome.pv.len().max(1) as i32))),
        Some(_) => Some(Score::Cp(0)),
        None => Some(Score::Cp(winrate_to_cp(outcome.winrate))),
    };
    Info {
        depth: Some(u32::from(outcome.max_depth)),
        time_ms: Some(outcome.elapsed_ms),
        nodes: Some(outcome.playouts),
        nps: Some(outcome.nps),
        score,
        pv: outcome.pv.clone(),
        ..Info::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    /// fake バックエンド: 呼び出しを記録し，決めた応答を返す．
    struct FakeBackend {
        calls: Rc<RefCell<Vec<(String, Vec<String>, SearchBudget)>>>,
        outcome: SearchOutcome,
    }

    impl SearchBackend for FakeBackend {
        fn search(
            &mut self,
            sfen: &str,
            moves: &[String],
            budget: &SearchBudget,
            _stop: &Arc<AtomicBool>,
        ) -> Result<SearchOutcome, String> {
            self.calls
                .borrow_mut()
                .push((sfen.to_string(), moves.to_vec(), budget.clone()));
            Ok(self.outcome.clone())
        }

        fn is_mock(&self) -> bool {
            true
        }
    }

    type Calls = Rc<RefCell<Vec<(String, Vec<String>, SearchBudget)>>>;

    fn agent_with_fake(
        outcome: SearchOutcome,
    ) -> (
        Agent<FakeBackend, impl Fn(&EngineConfig) -> Result<FakeBackend, String>>,
        Calls,
    ) {
        let calls: Calls = Rc::new(RefCell::new(Vec::new()));
        let calls_for_factory = Rc::clone(&calls);
        let agent = Agent::new(EngineConfig::default(), move |_config| {
            Ok(FakeBackend {
                calls: Rc::clone(&calls_for_factory),
                outcome: outcome.clone(),
            })
        });
        (agent, calls)
    }

    fn default_outcome() -> SearchOutcome {
        SearchOutcome {
            best_usi: Some("7g7f".to_string()),
            winrate: 0.6,
            pv: vec!["7g7f".to_string(), "3c3d".to_string()],
            playouts: 1000,
            elapsed_ms: 900,
            nps: 1100,
            max_depth: 8,
            proven: None,
        }
    }

    #[test]
    fn test_usi_handshake() {
        let (mut agent, _) = agent_with_fake(default_outcome());
        let out = agent.handle(GuiCommand::Usi).unwrap();
        assert!(matches!(&out[0], EngineCommand::Id { name, .. } if name.starts_with("maou")));
        assert_eq!(out.last(), Some(&EngineCommand::UsiOk));
        // ModelPath オプションが宣言されている
        assert!(out.iter().any(
            |c| matches!(c, EngineCommand::OptionDecl(OptionDecl { name, .. }) if *name == "ModelPath")
        ));
    }

    #[test]
    fn test_isready_builds_backend_and_marks_mock() {
        let (mut agent, _) = agent_with_fake(default_outcome());
        let out = agent.handle(GuiCommand::IsReady).unwrap();
        // mock 明示の info string + readyok
        assert!(
            matches!(&out[0], EngineCommand::Info(i) if i.string.as_deref().is_some_and(|s| s.contains("mock")))
        );
        assert_eq!(out.last(), Some(&EngineCommand::ReadyOk));
        // 2 回目の isready は即 readyok のみ
        let out2 = agent.handle(GuiCommand::IsReady).unwrap();
        assert_eq!(out2, vec![EngineCommand::ReadyOk]);
    }

    #[test]
    fn test_full_game_turn() {
        let (mut agent, calls) = agent_with_fake(default_outcome());
        agent.handle(GuiCommand::IsReady).unwrap();
        agent.handle(GuiCommand::UsiNewGame).unwrap();
        agent
            .handle(GuiCommand::Position {
                sfen: None,
                moves: vec!["7g7f".to_string(), "3c3d".to_string()],
            })
            .unwrap();
        let out = agent
            .handle(GuiCommand::Go(GoParams {
                clock: crate::protocol::ClockParams {
                    btime: Some(60_000),
                    wtime: Some(60_000),
                    byoyomi: Some(10_000),
                    ..Default::default()
                },
                ..GoParams::default()
            }))
            .unwrap();
        // info (pv 付き) → bestmove の順
        assert!(matches!(&out[0], EngineCommand::Info(i) if !i.pv.is_empty()));
        assert_eq!(
            out.last(),
            Some(&EngineCommand::BestMove {
                mv: BestMoveKind::Move("7g7f".to_string()),
                ponder: None
            })
        );
        // バックエンドへは SFEN + USI 経路 (千日手履歴規約) が渡る
        let recorded = calls.borrow();
        let (sfen, moves, budget) = &recorded[0];
        assert_eq!(sfen, STARTPOS_SFEN);
        assert_eq!(moves.len(), 2);
        // 60s/40 + 10s − 1s = 10.5s
        assert_eq!(budget.time_ms, Some(10_500));
    }

    #[test]
    fn test_clock_uses_side_to_move() {
        let (mut agent, calls) = agent_with_fake(default_outcome());
        agent.handle(GuiCommand::IsReady).unwrap();
        // 1 手進んだ局面 = 後手番 → wtime 側が使われる
        agent
            .handle(GuiCommand::Position {
                sfen: None,
                moves: vec!["7g7f".to_string()],
            })
            .unwrap();
        agent
            .handle(GuiCommand::Go(GoParams {
                clock: crate::protocol::ClockParams {
                    btime: Some(400_000),
                    wtime: Some(80_000),
                    byoyomi: Some(2_000),
                    ..Default::default()
                },
                ..GoParams::default()
            }))
            .unwrap();
        let recorded = calls.borrow();
        // 80s/40 + 2s − 1s = 3s (wtime 基準)．btime 基準なら 11s になるはず
        assert_eq!(recorded[0].2.time_ms, Some(3_000));
    }

    #[test]
    fn test_go_infinite_is_unbounded() {
        let (mut agent, calls) = agent_with_fake(default_outcome());
        agent.handle(GuiCommand::IsReady).unwrap();
        agent
            .handle(GuiCommand::Go(GoParams {
                infinite: true,
                ..GoParams::default()
            }))
            .unwrap();
        assert!(calls.borrow()[0].2.unbounded);
    }

    #[test]
    fn test_go_mate_not_implemented() {
        let (mut agent, _) = agent_with_fake(default_outcome());
        let out = agent
            .handle(GuiCommand::Go(GoParams {
                mate: Some(crate::protocol::MateLimit::TimeMs(1000)),
                ..GoParams::default()
            }))
            .unwrap();
        assert_eq!(out, vec![EngineCommand::CheckmateNotImplemented]);
    }

    #[test]
    fn test_no_legal_move_resigns() {
        let (mut agent, _) = agent_with_fake(SearchOutcome {
            best_usi: None,
            ..SearchOutcome::default()
        });
        agent.handle(GuiCommand::IsReady).unwrap();
        let out = agent.handle(GuiCommand::Go(GoParams::default())).unwrap();
        assert_eq!(
            out.last(),
            Some(&EngineCommand::BestMove {
                mv: BestMoveKind::Resign,
                ponder: None
            })
        );
    }

    #[test]
    fn test_position_without_gameover_recovers() {
        // gameover を送らない GUI (ShogiGUI 連続対局) でも次の position で
        // 状態が入れ替わる
        let (mut agent, calls) = agent_with_fake(default_outcome());
        agent.handle(GuiCommand::IsReady).unwrap();
        agent
            .handle(GuiCommand::Position {
                sfen: None,
                moves: vec!["7g7f".to_string()],
            })
            .unwrap();
        agent.handle(GuiCommand::Go(GoParams::default())).unwrap();
        // gameover なしで新しい対局の position が来る
        agent
            .handle(GuiCommand::Position {
                sfen: None,
                moves: vec![],
            })
            .unwrap();
        agent.handle(GuiCommand::Go(GoParams::default())).unwrap();
        let recorded = calls.borrow();
        assert_eq!(recorded[1].1.len(), 0, "新しい対局の経路に更新されている");
    }

    #[test]
    fn test_setoption_updates_config_and_invalidates_backend() {
        let (mut agent, _) = agent_with_fake(default_outcome());
        agent.handle(GuiCommand::IsReady).unwrap();
        agent
            .handle(GuiCommand::SetOption {
                name: "Threads".to_string(),
                value: Some("4".to_string()),
            })
            .unwrap();
        assert_eq!(agent.config.threads, 4);
        // 評価器系オプションはバックエンドを破棄 → isready で mock 明示が再度出る
        let out = agent.handle(GuiCommand::IsReady).unwrap();
        assert_eq!(out.len(), 2, "再構築 (info + readyok) される");
        // 時間系オプションは破棄しない
        agent
            .handle(GuiCommand::SetOption {
                name: "NetworkDelay".to_string(),
                value: Some("500".to_string()),
            })
            .unwrap();
        assert_eq!(agent.config.time.network_delay_ms, 500);
        let out = agent.handle(GuiCommand::IsReady).unwrap();
        assert_eq!(out, vec![EngineCommand::ReadyOk]);
    }

    #[test]
    fn test_usi_hash_conversion() {
        let mut config = EngineConfig::default();
        config.usi_hash_mb = Some(512);
        // 512MB / 512B = 1M ノード
        assert_eq!(config.effective_node_capacity(), Some(1 << 20));
        // NodeCapacity 明示が優先
        config.node_capacity = Some(4096);
        assert_eq!(config.effective_node_capacity(), Some(4096));
    }

    #[test]
    fn test_winrate_to_cp() {
        assert_eq!(winrate_to_cp(0.5), 0);
        assert!(winrate_to_cp(0.9) > 0);
        assert!(winrate_to_cp(0.1) < 0);
        // 対称性
        assert_eq!(winrate_to_cp(0.75), -winrate_to_cp(0.25));
    }

    #[test]
    fn test_unknown_command_ignored() {
        let (mut agent, _) = agent_with_fake(default_outcome());
        let out = agent
            .handle(GuiCommand::Unknown("mystery".to_string()))
            .unwrap();
        assert!(out.is_empty());
    }
}
