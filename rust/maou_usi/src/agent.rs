//! 対局エージェント — USI コマンド (型付き) を受けて応答を返す状態機械．
//!
//! transport (stdio/自己対局 driver) 非依存 (docs/design/usi-engine/index.md §4)．
//! 探索は [`SearchBackend`] trait 越しに呼ぶため，fake backend で状態機械を
//! 単体テストできる．時間管理は [`crate::time`] (戦略レイヤー) に委譲する．

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use maou_shogi::types::Color;

use crate::protocol::{
    BestMoveKind, EngineCommand, GameResult, GoParams, GuiCommand, Info, OptionDecl, OptionKind,
    Score,
};
use crate::time::{allocate, should_stop, TimeBudget, TimeStrategyConfig};

/// エンジン応答の逐次出力先．`info` を探索中に随時流すため，`go` 系ハンドラは
/// 応答を貯めず [`Emit`] へ push する (stdio では書き込み + flush する closure)．
pub type Emit<'a> = dyn FnMut(EngineCommand) + 'a;

/// 平手初期局面 (USI `position startpos`)．
pub const STARTPOS_SFEN: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

/// `USI_Hash` (MB) → ノードプール容量の換算に使う 1 ノードあたりの概算バイト数
/// (Node 本体 + Edge 配列の平均的な合計．将来 NodePool の実測で較正する —
/// 設計 doc §12 未決事項 3)．
const APPROX_BYTES_PER_NODE: u64 = 512;

/// 探索中の `info` 随時出力の最小間隔 (ミリ秒)．過剰な流量を抑える (設計 §10)．
const INFO_INTERVAL_MS: u64 = 1000;

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
    /// 先手番の引き分け価値 (千分率，既定 500)．千日手・最大手数の引き分けを
    /// 手番視点で `draw_value` に変換して探索へ渡す (電竜戦の先手 0.4 勝 = 400)．
    pub draw_value_black: u32,
    /// 後手番の引き分け価値 (千分率，既定 500．電竜戦の後手 0.6 勝 = 600)．
    pub draw_value_white: u32,
    /// 投了する root 勝率の閾値 (千分率，既定 0 = 投了しない)．root 勝率がこの
    /// 値未満の手が `resign_consecutive` 手続いたら `bestmove resign`．
    pub resign_value: u32,
    /// 投了に必要な連続手数 (`resign_value` > 0 のとき有効)．
    pub resign_consecutive: u32,
    /// 引き分けになる最大手数 (既定 0 = 無効．電竜戦は 512)．到達局面では入玉
    /// 宣言を必ず確認し，可能なら宣言勝ちする．
    pub max_moves_to_draw: u32,
    /// ponder (先読み) を有効にするか (既定 true)．true のとき `USI_Ponder`
    /// option を宣言し，`bestmove` に予想相手手 (自探索 PV の 2 手目) を付ける．
    /// GUI はこれを見て相手番中に `go ponder` を送る (設計 doc §8.4)．
    pub usi_ponder: bool,
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
            draw_value_black: 500,
            draw_value_white: 500,
            resign_value: 0,
            resign_consecutive: 3,
            max_moves_to_draw: 0,
            usi_ponder: true,
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SearchBudget {
    /// 時間予算 (soft/hard)．`None` = playout 制または無期限．バックエンドは
    /// `hard_ms` を探索の絶対上限に，soft 到達時の延長判断は monitor が行う．
    pub time: Option<TimeBudget>,
    /// playout 予算．
    pub max_playouts: Option<u64>,
    /// 無期限 (stop でのみ停止．`go ponder`/`go infinite`)．
    pub unbounded: bool,
}

/// 探索中の進捗スナップショット (バックエンド → [`SearchObserver`])．
/// maou_search の `RootSnapshot` を transport 非依存の USI 表記へ写したもの．
#[derive(Clone, Debug, Default)]
pub struct ProgressSnapshot {
    /// 完了 playout 数．
    pub playouts: u64,
    /// playout 毎秒．
    pub nps: u64,
    /// 到達最大深さ．
    pub max_depth: u16,
    /// 現時点の最有力手 (USI 表記)．
    pub best_usi: Option<String>,
    /// 最有力手の訪問回数．
    pub best_visits: u64,
    /// 次点手の訪問回数 (延長判断用)．
    pub second_visits: u64,
    /// 最有力手の root 手番視点勝率．
    pub winrate: f64,
    /// 読み筋 (USI 表記)．
    pub pv: Vec<String>,
    /// root 確定値 (詰み等)．未確定は `None`．
    pub proven: Option<f64>,
}

/// 探索中の進捗を受け取る観測者 (エージェント実装)．
///
/// バックエンドが探索スレッドと並行に，一定間隔で最新スナップショットを渡す．
/// 実装は `info` の随時出力などの副作用を行い，早期停止したいときに `true` を
/// 返す (停止フラグを立てる責務はバックエンド側)．
pub trait SearchObserver {
    /// 最新スナップショットと計測経過 (ミリ秒) を受け取る．`true` を返すと
    /// バックエンドが停止フラグを立てて探索を打ち切る．
    fn on_progress(&mut self, snapshot: &ProgressSnapshot, elapsed_ms: u64) -> bool;
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
    /// `draw_value` は root 手番視点の引き分け価値 (千日手戦略)．`observer` は
    /// 探索中の進捗を受け取り早期停止を要求できる (`info` 随時出力・時間延長)．
    /// `stop` が立てられたら途中で打ち切ってその時点の最有力手を返すこと．
    fn search(
        &mut self,
        sfen: &str,
        moves: &[String],
        budget: &SearchBudget,
        draw_value: f64,
        stop: &Arc<AtomicBool>,
        observer: &mut dyn SearchObserver,
    ) -> Result<SearchOutcome, String>;

    /// 現局面 (基準 SFEN + USI 経路) で手番側の入玉宣言 (27 点法) が成立するか．
    fn nyugyoku_declarable(&self, sfen: &str, moves: &[String]) -> Result<bool, String>;

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

    /// 現局面の手数 (基準 SFEN の手数 + 経路長)．SFEN 第 4 要素が手数
    /// (startpos は 1)．MaxMovesToDraw 判定に使う．
    fn move_number(&self) -> u64 {
        let base: u64 = self
            .sfen
            .split_whitespace()
            .nth(3)
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        base + self.moves.len() as u64
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
    /// root 勝率が投了閾値未満だった連続手数 (`resign_value` 用)．usinewgame /
    /// gameover でリセットする．
    resign_streak: u32,
    /// ponder 的中シグナル．transport (stdio reader) が行の到着順で更新する:
    /// `go` 行で false，`ponderhit` 行で true (stop フラグと同じ race-free 規約)．
    /// 探索中の [`GoObserver`] がポーリングし，的中したら無期限の ponder 探索を
    /// 時間予算へ切り替える (探索木はそのまま引き継がれる — ponder の主利得)．
    ponderhit: Arc<AtomicBool>,
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
            resign_streak: 0,
            ponderhit: Arc::new(AtomicBool::new(false)),
        }
    }

    /// 探索停止フラグのハンドル．
    ///
    /// transport 側の規約 (行の読み取り順で更新することで race を避ける):
    /// `go` 行を読んだら false，`stop`/`quit` 行を読んだら true を store する．
    pub fn stop_handle(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.stop)
    }

    /// ponder 的中フラグのハンドル ([`Agent::ponderhit`] の規約参照)．
    ///
    /// transport 側の規約 (stop フラグと同じ行順更新): `go` 行で false，
    /// `ponderhit` 行で true を store する．探索中の observer がこれを拾う．
    pub fn ponderhit_handle(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.ponderhit)
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
                self.resign_streak = 0;
                Ok(Vec::new())
            }
            GuiCommand::Position { sfen, moves } => {
                self.game = GameState {
                    sfen: sfen.unwrap_or_else(|| STARTPOS_SFEN.to_string()),
                    moves,
                };
                Ok(Vec::new())
            }
            // Go は探索中に info を随時流すため sink 版へ委譲し，収集して返す
            // (stdio transport は Go を handle_go_stream で直接ストリームする)
            GuiCommand::Go(params) => {
                let mut out = Vec::new();
                self.handle_go_stream(&params, &mut |c| out.push(c))?;
                Ok(out)
            }
            // Stop は探索中に transport (reader スレッド) が stop フラグへ即
            // 反映する (行の到着順規約)．dispatcher が Go でブロック中に処理
            // されるのはブロック解除後 = 探索終了後で，bestmove は Go の応答
            // として送信済み．ここでは何もしない
            GuiCommand::Stop => Ok(Vec::new()),
            // PonderHit も reader スレッドが ponderhit フラグへ反映し，探索中の
            // GoObserver が拾って時間予算へ切り替える (agent 内で完結する)．
            // dispatcher へ届く頃には探索終了後なので，ここでは何もしない
            GuiCommand::PonderHit(_) => Ok(Vec::new()),
            GuiCommand::GameOver(result) => {
                let _: GameResult = result;
                self.game = GameState::default();
                self.resign_streak = 0;
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
                name: "USI_Ponder",
                kind: OptionKind::Check {
                    default: c.usi_ponder,
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
            OptionDecl {
                name: "DrawValueBlack",
                kind: OptionKind::Spin {
                    default: i64::from(c.draw_value_black),
                    min: 0,
                    max: 1000,
                },
            },
            OptionDecl {
                name: "DrawValueWhite",
                kind: OptionKind::Spin {
                    default: i64::from(c.draw_value_white),
                    min: 0,
                    max: 1000,
                },
            },
            OptionDecl {
                name: "ResignValue",
                kind: OptionKind::Spin {
                    default: i64::from(c.resign_value),
                    min: 0,
                    max: 1000,
                },
            },
            OptionDecl {
                name: "ResignConsecutive",
                kind: OptionKind::Spin {
                    default: i64::from(c.resign_consecutive),
                    min: 1,
                    max: 1000,
                },
            },
            OptionDecl {
                name: "MaxMovesToDraw",
                kind: OptionKind::Spin {
                    default: i64::from(c.max_moves_to_draw),
                    min: 0,
                    max: 100_000,
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
            "DrawValueBlack" => {
                if let Ok(n) = v.parse() {
                    self.config.draw_value_black = n;
                }
                invalidate = false;
            }
            "DrawValueWhite" => {
                if let Ok(n) = v.parse() {
                    self.config.draw_value_white = n;
                }
                invalidate = false;
            }
            "ResignValue" => {
                if let Ok(n) = v.parse() {
                    self.config.resign_value = n;
                }
                invalidate = false;
            }
            "ResignConsecutive" => {
                if let Ok(n) = v.parse::<u32>() {
                    self.config.resign_consecutive = n.max(1);
                }
                invalidate = false;
            }
            "MaxMovesToDraw" => {
                if let Ok(n) = v.parse() {
                    self.config.max_moves_to_draw = n;
                }
                invalidate = false;
            }
            "USI_Ponder" => {
                self.config.usi_ponder = parse_bool();
                invalidate = false;
            }
            // 未知のオプションは無視 (GUI 側の拡張への耐性)
            _ => invalidate = false,
        }
        if invalidate {
            self.backend = None;
        }
    }

    /// `go` を処理し，探索中の `info` を随時 `emit` へ流して bestmove を返す．
    ///
    /// - `go mate` は未対応 (`checkmate notimplemented`)．
    /// - SpecialRules 前処理: 入玉宣言 (27 点法 + 持ち時間) が成立すれば探索
    ///   せず `bestmove win`．
    /// - 千日手戦略: 手番の引き分け価値を探索の `draw_value` へ変換して渡す．
    /// - 投了: root 勝率が閾値未満の手が `resign_consecutive` 手続いたら
    ///   `bestmove resign` (`resign_value` = 0 なら投了しない)．
    pub fn handle_go_stream(&mut self, params: &GoParams, emit: &mut Emit) -> Result<(), String> {
        if params.mate.is_some() {
            // 詰み探索モードは未対応 (dfpn 接続は将来対応)
            emit(EngineCommand::CheckmateNotImplemented);
            return Ok(());
        }

        // isready を送らない GUI への防御 (通常は isready で構築済み)
        if self.backend.is_none() {
            self.backend = Some((self.factory)(&self.config)?);
        }

        // SpecialRules: 入玉宣言 (27 点法 + 持ち時間が残っている)
        if self.should_declare_win(&params.clock)? {
            emit(EngineCommand::BestMove {
                mv: BestMoveKind::Win,
                ponder: None,
            });
            return Ok(());
        }

        let draw_value = self.draw_value_for(self.game.side_to_move());
        let budget = self.decide_budget(params);
        // ponder (`go ponder`) は無期限探索で始め，`ponderhit` を拾ったら時間
        // 予算へ切り替える (探索木は引き継がれる)．的中後に使う予算 = この局面を
        // 通常 go したときと同じ配分をここで計算して observer へ渡す．
        let ponder_switch = params.ponder.then(|| {
            (
                Arc::clone(&self.ponderhit),
                self.timed_budget(&params.clock),
            )
        });
        let sfen = self.game.sfen.clone();
        let moves = self.game.moves.clone();
        let stop = Arc::clone(&self.stop);
        let outcome = {
            let backend = self.backend.as_mut().expect("直前で構築済み");
            let mut observer = GoObserver::new(emit, budget.time, ponder_switch);
            backend.search(&sfen, &moves, &budget, draw_value, &stop, &mut observer)?
        };

        // 探索サマリ info (最終) → bestmove (予想相手手 = 自探索 PV の 2 手目)
        emit(EngineCommand::Info(build_info(&outcome)));
        let mv = self.decide_bestmove(&outcome);
        let ponder = self.ponder_move(&mv, &outcome);
        emit(EngineCommand::BestMove { mv, ponder });
        Ok(())
    }

    /// 入玉宣言 (27 点法) が成立し，かつ持ち時間が残っているか．
    fn should_declare_win(&self, clock: &crate::protocol::ClockParams) -> Result<bool, String> {
        let backend = self.backend.as_ref().expect("直前で構築済み");
        if !backend.nyugyoku_declarable(&self.game.sfen, &self.game.moves)? {
            return Ok(false);
        }
        // 盤面 5 条件は backend が判定済み．残る「持ち時間が残っている」条件を確認
        Ok(has_time_remaining(clock, self.game.side_to_move()))
    }

    /// 手番側の引き分け価値 (千分率 → 0.0..=1.0)．
    fn draw_value_for(&self, side: Color) -> f64 {
        let permille = match side {
            Color::Black => self.config.draw_value_black,
            Color::White => self.config.draw_value_white,
        };
        f64::from(permille) / 1000.0
    }

    /// 探索結果から bestmove を決める (投了判断込み，`resign_streak` を更新)．
    fn decide_bestmove(&mut self, outcome: &SearchOutcome) -> BestMoveKind {
        let Some(usi) = &outcome.best_usi else {
            // 合法手なし = 詰まされている → 投了 (対局終了なので streak リセット)
            self.resign_streak = 0;
            return BestMoveKind::Resign;
        };
        // 投了: 閾値 > 0 かつ勝ち確定でない局面で，root 勝率が閾値未満の手が
        // resign_consecutive 手続いたら投了する
        let win_proven = matches!(outcome.proven, Some(v) if v >= 1.0);
        if self.config.resign_value > 0 && !win_proven {
            let threshold = f64::from(self.config.resign_value) / 1000.0;
            if outcome.winrate < threshold {
                self.resign_streak += 1;
                if self.resign_streak >= self.config.resign_consecutive {
                    return BestMoveKind::Resign;
                }
            } else {
                self.resign_streak = 0;
            }
        } else {
            self.resign_streak = 0;
        }
        BestMoveKind::Move(usi.clone())
    }

    /// bestmove に付ける予想相手手 (ponder 対象 = 自探索 PV の 2 手目)．
    ///
    /// `USI_Ponder` が無効・指し手が resign/win・PV が 2 手未満なら `None`
    /// (設計 doc §8.4)．GUI はこの手を相手が指すと仮定して相手番中に
    /// `go ponder` を送る．
    fn ponder_move(&self, mv: &BestMoveKind, outcome: &SearchOutcome) -> Option<String> {
        if !self.config.usi_ponder || !matches!(mv, BestMoveKind::Move(_)) {
            return None;
        }
        outcome.pv.get(1).cloned()
    }

    /// go パラメータ → 探索予算 (時間戦略は crate::time)．
    fn decide_budget(&self, params: &GoParams) -> SearchBudget {
        if params.infinite || params.ponder {
            return SearchBudget {
                time: None,
                max_playouts: None,
                unbounded: true,
            };
        }
        if let Some(nodes) = params.nodes {
            return SearchBudget {
                time: None,
                max_playouts: Some(nodes),
                unbounded: false,
            };
        }
        if let Some(movetime) = params.movetime {
            let ms = movetime
                .saturating_sub(self.config.time.network_delay_ms)
                .max(self.config.time.min_think_ms);
            return SearchBudget {
                // movetime は固定 (延長なし): soft == hard
                time: Some(TimeBudget {
                    soft_ms: ms,
                    hard_ms: ms,
                }),
                max_playouts: None,
                unbounded: false,
            };
        }
        SearchBudget {
            time: Some(self.timed_budget(&params.clock)),
            max_playouts: None,
            unbounded: false,
        }
    }

    /// 持ち時間 → 時間予算 (soft/hard)．`allocate` に MaxMovesToDraw 目前の絞り
    /// (§8.3) を重ねたもの．通常 go と ponderhit 後の予算計算で共有する．
    fn timed_budget(&self, clock: &crate::protocol::ClockParams) -> TimeBudget {
        let mut budget = allocate(&self.config.time, clock, self.game.side_to_move());
        // MaxMovesToDraw が近ければ予算を絞る (引き分け目前で深追いしない．§8.3)
        if let Some(cap) = self.max_moves_draw_cap() {
            budget.soft_ms = budget.soft_ms.min(cap);
            budget.hard_ms = budget.hard_ms.min(cap).max(budget.soft_ms);
        }
        budget
    }

    /// MaxMovesToDraw リミットが目前 (残り 4 手以内) なら思考予算の上限
    /// (ミリ秒) を返す．無効 (0) または十分手数があるなら `None`．
    /// in-search でのリミット以降 Draw 終端化は M4 (効果計測後)．
    fn max_moves_draw_cap(&self) -> Option<u64> {
        let limit = u64::from(self.config.max_moves_to_draw);
        if limit == 0 {
            return None;
        }
        if self.game.move_number() + 4 >= limit {
            Some(self.config.time.min_think_ms.max(1))
        } else {
            None
        }
    }
}

/// 手番側に持ち時間 (残時間・加算・秒読み) が残っているか — 入玉宣言の時間条件．
/// clock 情報が皆無 (検討モード等) なら true (時間条件を課さない)．
fn has_time_remaining(clock: &crate::protocol::ClockParams, side: Color) -> bool {
    if clock.is_empty() {
        return true;
    }
    let (t, inc) = match side {
        Color::Black => (clock.btime.unwrap_or(0), clock.binc.unwrap_or(0)),
        Color::White => (clock.wtime.unwrap_or(0), clock.winc.unwrap_or(0)),
    };
    t + inc + clock.byoyomi.unwrap_or(0) > 0
}

/// `info` を時間 gate で随時出力し，soft 到達時の時間延長を判断する観測者
/// ([`crate::time::should_stop`])．
struct GoObserver<'a, 'e> {
    emit: &'a mut Emit<'e>,
    /// 現在有効な時間予算 (`None` = 無期限: `go infinite` / ponder 中で未的中)．
    budget: Option<TimeBudget>,
    /// 次に info を出す経過時刻 (ミリ秒)．
    next_info_ms: u64,
    /// ponder (`go ponder`) 用の的中スイッチ: (的中フラグ, 的中後の時間予算)．
    /// 非 ponder go は `None`．的中フラグが立ったら `budget` を的中後予算へ
    /// 切り替える (探索木はそのまま引き継がれる — ponder の主利得)．
    ponder: Option<(Arc<AtomicBool>, TimeBudget)>,
    /// 時間計測の起点 (ミリ秒)．ponder 的中時に「今」へ更新する — 的中後の
    /// 予算は的中時点から計測する (ponder 中の思考は相手の持ち時間ゆえ無料)．
    baseline_ms: u64,
}

impl<'a, 'e> GoObserver<'a, 'e> {
    fn new(
        emit: &'a mut Emit<'e>,
        budget: Option<TimeBudget>,
        ponder: Option<(Arc<AtomicBool>, TimeBudget)>,
    ) -> GoObserver<'a, 'e> {
        GoObserver {
            emit,
            budget,
            next_info_ms: 0,
            ponder,
            baseline_ms: 0,
        }
    }
}

impl SearchObserver for GoObserver<'_, '_> {
    fn on_progress(&mut self, snapshot: &ProgressSnapshot, elapsed_ms: u64) -> bool {
        // info 随時出力 (時間 gate — 過剰な流量を抑える，設計 §10)
        if elapsed_ms >= self.next_info_ms {
            (self.emit)(EngineCommand::Info(info_from_snapshot(
                snapshot, elapsed_ms,
            )));
            self.next_info_ms = elapsed_ms + INFO_INTERVAL_MS;
        }
        // ponder 的中: 無期限探索を時間予算へ切り替え，計測起点を的中時点に置く
        // (それまでに積んだ playout = 探索木はそのまま活きる = ponder の主利得)
        if self.budget.is_none() {
            let hit = self
                .ponder
                .as_ref()
                .is_some_and(|(flag, _)| flag.load(Ordering::Acquire));
            if hit {
                self.budget = self.ponder.as_ref().map(|(_, b)| *b);
                self.baseline_ms = elapsed_ms;
            }
        }
        // 時間延長判断 (無期限探索なら停止要求しない = 外部 stop のみで止まる)．
        // 経過は計測起点 (通常 0，ponder 的中後は的中時点) からの差分で見る
        match self.budget {
            Some(b) => should_stop(
                &b,
                elapsed_ms.saturating_sub(self.baseline_ms),
                snapshot.best_visits,
                snapshot.second_visits,
                snapshot.proven.is_some(),
            ),
            None => false,
        }
    }
}

/// 勝率・確定値・PV 長から `info score` を作る (サマリ/随時出力で共有)．
fn score_of(proven: Option<f64>, winrate: f64, pv_len: usize) -> Option<Score> {
    match proven {
        // 勝ち確定 (詰み発見含む): PV 長を詰み手数として報告する
        Some(v) if v >= 1.0 => Some(Score::Mate(pv_len.max(1) as i32)),
        Some(v) if v <= 0.0 => Some(Score::Mate(-(pv_len.max(1) as i32))),
        Some(_) => Some(Score::Cp(0)),
        None => Some(Score::Cp(winrate_to_cp(winrate))),
    }
}

/// 探索結果 → `info` 1 行 (探索サマリ)．
fn build_info(outcome: &SearchOutcome) -> Info {
    Info {
        depth: Some(u32::from(outcome.max_depth)),
        time_ms: Some(outcome.elapsed_ms),
        nodes: Some(outcome.playouts),
        nps: Some(outcome.nps),
        score: score_of(outcome.proven, outcome.winrate, outcome.pv.len()),
        pv: outcome.pv.clone(),
        ..Info::default()
    }
}

/// 進捗スナップショット → `info` 1 行 (探索中の随時出力)．
fn info_from_snapshot(snap: &ProgressSnapshot, elapsed_ms: u64) -> Info {
    Info {
        depth: Some(u32::from(snap.max_depth)),
        time_ms: Some(elapsed_ms),
        nodes: Some(snap.playouts),
        nps: Some(snap.nps),
        score: score_of(snap.proven, snap.winrate, snap.pv.len()),
        pv: snap.pv.clone(),
        ..Info::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    /// fake バックエンド: 呼び出しを記録し，決めた応答を返す．observer を
    /// 1 回駆動して info 随時出力の経路も通す．
    struct FakeBackend {
        calls: Calls,
        draw_values: DrawValues,
        outcome: SearchOutcome,
        /// `nyugyoku_declarable` の返り値 (入玉宣言テスト用)．
        nyugyoku: bool,
    }

    impl SearchBackend for FakeBackend {
        fn search(
            &mut self,
            sfen: &str,
            moves: &[String],
            budget: &SearchBudget,
            draw_value: f64,
            _stop: &Arc<AtomicBool>,
            observer: &mut dyn SearchObserver,
        ) -> Result<SearchOutcome, String> {
            self.calls
                .borrow_mut()
                .push((sfen.to_string(), moves.to_vec(), *budget));
            self.draw_values.borrow_mut().push(draw_value);
            // observer を 1 回駆動 (info 随時出力の経路を検証可能にする)
            let snap = ProgressSnapshot {
                playouts: self.outcome.playouts,
                nps: self.outcome.nps,
                max_depth: self.outcome.max_depth,
                best_usi: self.outcome.best_usi.clone(),
                best_visits: 10,
                second_visits: 1,
                winrate: self.outcome.winrate,
                pv: self.outcome.pv.clone(),
                proven: self.outcome.proven,
            };
            let _ = observer.on_progress(&snap, 0);
            Ok(self.outcome.clone())
        }

        fn nyugyoku_declarable(&self, _sfen: &str, _moves: &[String]) -> Result<bool, String> {
            Ok(self.nyugyoku)
        }

        fn is_mock(&self) -> bool {
            true
        }
    }

    type Calls = Rc<RefCell<Vec<(String, Vec<String>, SearchBudget)>>>;
    type DrawValues = Rc<RefCell<Vec<f64>>>;

    fn agent_with_fake(
        outcome: SearchOutcome,
    ) -> (
        Agent<FakeBackend, impl Fn(&EngineConfig) -> Result<FakeBackend, String>>,
        Calls,
    ) {
        let (agent, calls, _draws) = agent_with_fake_full(outcome, false);
        (agent, calls)
    }

    /// nyugyoku フラグと draw_value 記録も取れる版．
    fn agent_with_fake_full(
        outcome: SearchOutcome,
        nyugyoku: bool,
    ) -> (
        Agent<FakeBackend, impl Fn(&EngineConfig) -> Result<FakeBackend, String>>,
        Calls,
        DrawValues,
    ) {
        let calls: Calls = Rc::new(RefCell::new(Vec::new()));
        let draws: DrawValues = Rc::new(RefCell::new(Vec::new()));
        let calls_for_factory = Rc::clone(&calls);
        let draws_for_factory = Rc::clone(&draws);
        let agent = Agent::new(EngineConfig::default(), move |_config| {
            Ok(FakeBackend {
                calls: Rc::clone(&calls_for_factory),
                draw_values: Rc::clone(&draws_for_factory),
                outcome: outcome.clone(),
                nyugyoku,
            })
        });
        (agent, calls, draws)
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
        // info (pv 付き) → bestmove の順．PV = [7g7f, 3c3d] → 予想相手手 3c3d
        assert!(matches!(&out[0], EngineCommand::Info(i) if !i.pv.is_empty()));
        assert_eq!(
            out.last(),
            Some(&EngineCommand::BestMove {
                mv: BestMoveKind::Move("7g7f".to_string()),
                ponder: Some("3c3d".to_string())
            })
        );
        // バックエンドへは SFEN + USI 経路 (千日手履歴規約) が渡る
        let recorded = calls.borrow();
        let (sfen, moves, budget) = &recorded[0];
        assert_eq!(sfen, STARTPOS_SFEN);
        assert_eq!(moves.len(), 2);
        // 60s/40 + 10s − 1s = 10.5s (soft)．hard は延長上限 (soft×2)
        assert_eq!(budget.time.unwrap().soft_ms, 10_500);
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
        assert_eq!(recorded[0].2.time.unwrap().soft_ms, 3_000);
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

    fn set(
        agent: &mut Agent<FakeBackend, impl Fn(&EngineConfig) -> Result<FakeBackend, String>>,
        name: &str,
        value: &str,
    ) {
        agent
            .handle(GuiCommand::SetOption {
                name: name.to_string(),
                value: Some(value.to_string()),
            })
            .unwrap();
    }

    #[test]
    fn test_draw_value_converted_by_side_to_move() {
        let (mut agent, _calls, draws) = agent_with_fake_full(default_outcome(), false);
        agent.handle(GuiCommand::IsReady).unwrap();
        // 電竜戦: 先手 0.4 勝 / 後手 0.6 勝
        set(&mut agent, "DrawValueBlack", "400");
        set(&mut agent, "DrawValueWhite", "600");
        // startpos = 先手番 → draw_value 0.4
        agent
            .handle(GuiCommand::Position {
                sfen: None,
                moves: vec![],
            })
            .unwrap();
        agent.handle(GuiCommand::Go(GoParams::default())).unwrap();
        // 1 手進めて後手番 → draw_value 0.6
        agent
            .handle(GuiCommand::Position {
                sfen: None,
                moves: vec!["7g7f".to_string()],
            })
            .unwrap();
        agent.handle(GuiCommand::Go(GoParams::default())).unwrap();
        let d = draws.borrow();
        assert!((d[0] - 0.4).abs() < 1e-9, "先手番は 0.4 (got {})", d[0]);
        assert!((d[1] - 0.6).abs() < 1e-9, "後手番は 0.6 (got {})", d[1]);
    }

    #[test]
    fn test_nyugyoku_declares_win_with_time() {
        let (mut agent, calls, _) = agent_with_fake_full(default_outcome(), true);
        agent.handle(GuiCommand::IsReady).unwrap();
        let out = agent
            .handle(GuiCommand::Go(GoParams {
                clock: crate::protocol::ClockParams {
                    btime: Some(60_000),
                    ..Default::default()
                },
                ..GoParams::default()
            }))
            .unwrap();
        assert_eq!(
            out,
            vec![EngineCommand::BestMove {
                mv: BestMoveKind::Win,
                ponder: None
            }]
        );
        // 宣言勝ちのため探索は呼ばれない
        assert!(calls.borrow().is_empty());
    }

    #[test]
    fn test_nyugyoku_not_declared_without_time() {
        let (mut agent, calls, _) = agent_with_fake_full(default_outcome(), true);
        agent.handle(GuiCommand::IsReady).unwrap();
        // 持ち時間ゼロ → 宣言の時間条件を満たさず通常探索
        let out = agent
            .handle(GuiCommand::Go(GoParams {
                clock: crate::protocol::ClockParams {
                    btime: Some(0),
                    wtime: Some(0),
                    ..Default::default()
                },
                ..GoParams::default()
            }))
            .unwrap();
        assert!(matches!(
            out.last(),
            Some(EngineCommand::BestMove {
                mv: BestMoveKind::Move(_),
                ..
            })
        ));
        assert_eq!(calls.borrow().len(), 1, "探索が実行される");
    }

    #[test]
    fn test_resign_after_consecutive_low_winrate() {
        let losing = SearchOutcome {
            best_usi: Some("7g7f".to_string()),
            winrate: 0.05,
            pv: vec!["7g7f".to_string()],
            proven: None,
            ..SearchOutcome::default()
        };
        let (mut agent, _calls, _) = agent_with_fake_full(losing, false);
        agent.handle(GuiCommand::IsReady).unwrap();
        set(&mut agent, "ResignValue", "100"); // 勝率 0.1 未満
        set(&mut agent, "ResignConsecutive", "2");
        // 1 手目: streak 1 → まだ指す
        let out1 = agent.handle(GuiCommand::Go(GoParams::default())).unwrap();
        assert!(matches!(
            out1.last(),
            Some(EngineCommand::BestMove {
                mv: BestMoveKind::Move(_),
                ..
            })
        ));
        // 2 手目: streak 2 → 投了
        let out2 = agent.handle(GuiCommand::Go(GoParams::default())).unwrap();
        assert_eq!(
            out2.last(),
            Some(&EngineCommand::BestMove {
                mv: BestMoveKind::Resign,
                ponder: None
            })
        );
    }

    #[test]
    fn test_resign_disabled_by_default() {
        let losing = SearchOutcome {
            best_usi: Some("7g7f".to_string()),
            winrate: 0.01,
            pv: vec!["7g7f".to_string()],
            proven: None,
            ..SearchOutcome::default()
        };
        let (mut agent, _calls, _) = agent_with_fake_full(losing, false);
        agent.handle(GuiCommand::IsReady).unwrap();
        // ResignValue 既定 0 → 何手続いても投了しない
        for _ in 0..5 {
            let out = agent.handle(GuiCommand::Go(GoParams::default())).unwrap();
            assert!(matches!(
                out.last(),
                Some(EngineCommand::BestMove {
                    mv: BestMoveKind::Move(_),
                    ..
                })
            ));
        }
    }

    #[test]
    fn test_max_moves_to_draw_narrows_budget() {
        let (mut agent, calls, _) = agent_with_fake_full(default_outcome(), false);
        agent.handle(GuiCommand::IsReady).unwrap();
        set(&mut agent, "MaxMovesToDraw", "10");
        // move_number = 1 (startpos) + 6 = 7; 7 + 4 = 11 >= 10 → 予算を絞る
        agent
            .handle(GuiCommand::Position {
                sfen: None,
                moves: vec![
                    "7g7f".to_string(),
                    "3c3d".to_string(),
                    "2g2f".to_string(),
                    "8c8d".to_string(),
                    "2f2e".to_string(),
                    "8d8e".to_string(),
                ],
            })
            .unwrap();
        agent
            .handle(GuiCommand::Go(GoParams {
                clock: crate::protocol::ClockParams {
                    btime: Some(600_000),
                    wtime: Some(600_000),
                    byoyomi: Some(10_000),
                    ..Default::default()
                },
                ..GoParams::default()
            }))
            .unwrap();
        // 通常なら soft ~ 24s だが，リミット目前で min_think (100ms) に絞られる
        assert_eq!(calls.borrow()[0].2.time.unwrap().soft_ms, 100);
    }

    #[test]
    fn test_streaming_info_before_bestmove() {
        let (mut agent, _calls, _) = agent_with_fake_full(default_outcome(), false);
        agent.handle(GuiCommand::IsReady).unwrap();
        let out = agent.handle(GuiCommand::Go(GoParams::default())).unwrap();
        // 随時 info (observer 駆動) + 最終サマリ info の 2 本以上
        let infos = out
            .iter()
            .filter(|c| matches!(c, EngineCommand::Info(_)))
            .count();
        assert!(infos >= 2, "随時 info と最終 info が出る (got {infos})");
        assert!(matches!(out.last(), Some(EngineCommand::BestMove { .. })));
    }

    #[test]
    fn test_m2_options_declared() {
        let (mut agent, _) = agent_with_fake(default_outcome());
        let out = agent.handle(GuiCommand::Usi).unwrap();
        for name in [
            "DrawValueBlack",
            "DrawValueWhite",
            "ResignValue",
            "ResignConsecutive",
            "MaxMovesToDraw",
        ] {
            assert!(
                out.iter().any(|c| matches!(
                    c,
                    EngineCommand::OptionDecl(OptionDecl { name: n, .. }) if *n == name
                )),
                "{name} オプションが宣言される"
            );
        }
    }

    #[test]
    fn test_usi_ponder_option_declared() {
        let (mut agent, _) = agent_with_fake(default_outcome());
        let out = agent.handle(GuiCommand::Usi).unwrap();
        // M3: USI_Ponder は check 型・既定 true で宣言される (GUI が ponder を
        // 送る trigger)．M1/M2 では宣言していなかった
        assert!(out.iter().any(|c| matches!(
            c,
            EngineCommand::OptionDecl(OptionDecl {
                name: "USI_Ponder",
                kind: OptionKind::Check { default: true }
            })
        )));
    }

    #[test]
    fn test_bestmove_carries_ponder_move() {
        // default_outcome の PV = [7g7f, 3c3d] → 予想相手手 = 3c3d
        let (mut agent, _) = agent_with_fake(default_outcome());
        agent.handle(GuiCommand::IsReady).unwrap();
        let out = agent.handle(GuiCommand::Go(GoParams::default())).unwrap();
        assert_eq!(
            out.last(),
            Some(&EngineCommand::BestMove {
                mv: BestMoveKind::Move("7g7f".to_string()),
                ponder: Some("3c3d".to_string()),
            })
        );
    }

    #[test]
    fn test_no_ponder_when_disabled() {
        let (mut agent, _) = agent_with_fake(default_outcome());
        agent.handle(GuiCommand::IsReady).unwrap();
        set(&mut agent, "USI_Ponder", "false");
        let out = agent.handle(GuiCommand::Go(GoParams::default())).unwrap();
        assert_eq!(
            out.last(),
            Some(&EngineCommand::BestMove {
                mv: BestMoveKind::Move("7g7f".to_string()),
                ponder: None,
            })
        );
    }

    #[test]
    fn test_no_ponder_when_pv_too_short() {
        // PV が 1 手のみ → 予想相手手なし
        let short = SearchOutcome {
            best_usi: Some("7g7f".to_string()),
            winrate: 0.6,
            pv: vec!["7g7f".to_string()],
            ..SearchOutcome::default()
        };
        let (mut agent, _) = agent_with_fake(short);
        agent.handle(GuiCommand::IsReady).unwrap();
        let out = agent.handle(GuiCommand::Go(GoParams::default())).unwrap();
        assert!(matches!(
            out.last(),
            Some(EngineCommand::BestMove {
                mv: BestMoveKind::Move(_),
                ponder: None,
            })
        ));
    }

    #[test]
    fn test_go_ponder_is_unbounded() {
        // go ponder は無期限探索 (的中/stop まで止まらない)
        let (mut agent, calls) = agent_with_fake(default_outcome());
        agent.handle(GuiCommand::IsReady).unwrap();
        agent
            .handle(GuiCommand::Go(GoParams {
                ponder: true,
                clock: crate::protocol::ClockParams {
                    btime: Some(60_000),
                    wtime: Some(60_000),
                    byoyomi: Some(10_000),
                    ..Default::default()
                },
                ..GoParams::default()
            }))
            .unwrap();
        assert!(calls.borrow()[0].2.unbounded);
    }

    #[test]
    fn test_go_observer_ponderhit_switches_to_timed() {
        // ponder observer の核心: 未的中は無期限 (停止しない)，的中したら以後は
        // 「的中時点からの経過」で時間予算を判定する (それまでの木は活きる)
        let hit = Arc::new(AtomicBool::new(false));
        let budget = TimeBudget {
            soft_ms: 1_000,
            hard_ms: 2_000,
        };
        let mut sink = |_c: EngineCommand| {};
        let mut obs = GoObserver::new(&mut sink, None, Some((Arc::clone(&hit), budget)));
        let snap = ProgressSnapshot {
            best_visits: 100,
            second_visits: 90,
            ..ProgressSnapshot::default()
        };
        // 未的中: 経過が長くても無期限のまま停止しない
        assert!(!obs.on_progress(&snap, 5_000));
        // 的中: 計測起点が的中時点 (5000ms) に更新される
        hit.store(true, Ordering::Release);
        // 的中直後 (since = 0) はまだ停止しない
        assert!(!obs.on_progress(&snap, 5_000));
        // since = 7000 − 5000 = 2000 ≥ hard → 停止する
        assert!(obs.on_progress(&snap, 7_000));
    }
}
