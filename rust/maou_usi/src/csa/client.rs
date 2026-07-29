//! CSA サーバ接続のセッション driver (IO 層)．
//!
//! [`crate::stdio`] が stdin/stdout を担うのと同じ位置づけで，TCP ソケットを
//! 担う transport．[`crate::agent::Agent`] は無変更で駆動する．
//!
//! floodgate (wdoor) の運用に合わせた設計:
//!
//! - **1 局ごとにログアウト状態へ戻る**ため，連続対局は「接続 → ログイン →
//!   対局 → 切断」を繰り返す (公式の案内どおり)．
//! - 評価器 (ONNX セッション) は**プロセス内 1 個**を全対局で共有する．
//!   モデルロードと warmup は連続対局を通じて 1 回だけ (自己対局 driver と同じ構図)．
//! - 待機中は空行 1 つを keep-alive として送る．CSA 仕様は「サーバが当該
//!   クライアントから何かを受信してから 30 秒を経ずに再送してはならない
//!   (違反は反則負けにできる)」と定めるので，最短間隔を機構的に守る．
//!
//! 持ち時間は**サーバが指し手に付ける消費時間 (`,T<n>`) が正**であり，
//! クライアントの実測ではない (遅延時間の控除等をサーバが行うため)．

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::atomic::Ordering;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use maou_search::build_board_and_history;
use maou_shogi::board::Board;
use maou_shogi::movegen::generate_legal_moves;
use maou_shogi::types::Color;

use crate::agent::{Agent, EngineConfig, SearchBackend};
use crate::backend::{build_evaluator, search_options, warmup_evaluator, MaouSearchBackend};
use crate::protocol::{BestMoveKind, ClockParams, EngineCommand, GoParams, GuiCommand};
use crate::selfplay::clock_margin_ms;

use super::protocol::{
    is_terminal_status, move_from_csa, move_to_csa, parse_server_line, GameSummary,
    GameSummaryBuilder, ServerMessage, TimeRule,
};

/// CSA 仕様が定める keep-alive の最短間隔 (秒)．
///
/// 「サーバが当該クライアントから何らかの文字列を受信してから 30 秒を経ずして
/// 空行を送ってはならない」(仕様 3.4)．違反はサーバが反則負けにできるため，
/// 設定値がこれを下回っても機構的に切り上げる．
const KEEP_ALIVE_MIN_SEC: u64 = 30;

/// 持ち時間の指定が一切ない対局で 1 手に使う時間 (ミリ秒)．
///
/// CSA 仕様では `BEGIN Time` 階層の省略 = 時間無制限だが，無制限に探索すると
/// 対局が進まないため既定の思考時間を置く．
const NO_LIMIT_MOVETIME_MS: u64 = 10_000;

/// CSA クライアントの設定．
#[derive(Clone, Debug)]
pub struct CsaConfig {
    /// エンジン設定 (モデル・探索・戦略)．
    pub engine: EngineConfig,
    /// 接続先ホスト．
    pub host: String,
    /// 接続先ポート．
    pub port: u16,
    /// ログイン名 (レーティングの単位となる識別子)．
    pub login_name: String,
    /// パスワード欄．floodgate では `<ゲーム名>,<trip>` 形式．
    pub password: String,
    /// 対局数 (0 = 無制限)．
    pub games: u32,
    /// keep-alive の送信間隔 (秒)．[`KEEP_ALIVE_MIN_SEC`] 未満は切り上げる．
    pub keep_alive_sec: u64,
    /// 接続タイムアウト (秒)．
    pub connect_timeout_sec: u64,
    /// 対局条件の待機タイムアウト (秒．0 = 無制限)．
    pub game_wait_sec: u64,
    /// 対局間の再接続待ち (秒)．
    pub reconnect_wait_sec: u64,
    /// 通信ログを stderr に出す．
    pub verbose: bool,
}

impl Default for CsaConfig {
    fn default() -> Self {
        Self {
            engine: EngineConfig::default(),
            host: "wdoor.c.u-tokyo.ac.jp".to_string(),
            port: 4081,
            login_name: String::new(),
            password: String::new(),
            games: 1,
            keep_alive_sec: 60,
            connect_timeout_sec: 30,
            // floodgate は毎時 0 分/30 分に対局が組まれるので既定は 40 分待つ
            game_wait_sec: 2400,
            reconnect_wait_sec: 5,
            verbose: true,
        }
    }
}

/// 1 局の結果．
#[derive(Clone, Debug, Default)]
pub struct CsaGameResult {
    /// サーバの対局 ID．
    pub game_id: String,
    /// 相手の対局者名．
    pub opponent: String,
    /// 自分の手番 (`"black"` / `"white"`)．
    pub my_color: String,
    /// 総手数．
    pub moves: u32,
    /// 結果 (`"win"` / `"lose"` / `"draw"` / `"censored"` / `"chudan"`)．
    pub result: String,
    /// 終局理由 (`"#RESIGN"` / `"#SENNICHITE"` / `"#TIME_UP"` 等)．
    pub reason: String,
    /// 対局の所要時間 (ミリ秒)．
    pub elapsed_ms: u64,
}

/// セッション全体の結果．
#[derive(Clone, Debug, Default)]
pub struct CsaSessionSummary {
    /// 完了した対局．
    pub games: Vec<CsaGameResult>,
}

/// セッションのエラー (再試行可否を区別する)．
enum SessionError {
    /// 設定の誤り等，再接続しても直らないもの．
    Fatal(String),
    /// 通信断など，次の接続で回復し得るもの．
    Retryable(String),
}

impl SessionError {
    fn message(&self) -> &str {
        match self {
            SessionError::Fatal(m) | SessionError::Retryable(m) => m,
        }
    }
}

/// 連続した接続失敗をこの回数まで許容する (超えたら中止)．
const MAX_CONSECUTIVE_FAILURES: u32 = 3;

/// CSA サーバへ接続して対局する (`games` 局が終わるまで戻らない)．
pub fn run_csa(config: CsaConfig) -> Result<CsaSessionSummary, String> {
    if config.login_name.trim().is_empty() {
        return Err("ログイン名が空".to_string());
    }
    // 評価器はプロセス内 1 個 (モデルロード/warmup を連続対局で 1 回だけ)
    let evaluator = build_evaluator(&config.engine)?;
    warmup_evaluator(&evaluator)?;
    let evaluator = Arc::new(evaluator);

    let mut summary = CsaSessionSummary::default();
    let mut failures = 0u32;
    let mut played = 0u32;
    while config.games == 0 || played < config.games {
        let game_no = played + 1;
        eprintln!(
            "[csa] 対局 {game_no}{} — {}:{} へ接続",
            if config.games == 0 {
                String::new()
            } else {
                format!("/{}", config.games)
            },
            config.host,
            config.port
        );
        match play_one_connection(&config, &evaluator) {
            Ok(result) => {
                failures = 0;
                played += 1;
                eprintln!(
                    "[csa] 対局 {game_no} 終了: {} ({}) vs {} — {} 手 / {:.1} 秒",
                    result.result,
                    result.reason,
                    result.opponent,
                    result.moves,
                    result.elapsed_ms as f64 / 1000.0
                );
                summary.games.push(result);
            }
            Err(SessionError::Fatal(message)) => {
                return Err(message);
            }
            Err(err) => {
                failures += 1;
                eprintln!("[csa] 対局 {game_no} 失敗 ({failures}): {}", err.message());
                if failures >= MAX_CONSECUTIVE_FAILURES {
                    return Err(format!(
                        "{MAX_CONSECUTIVE_FAILURES} 回連続で失敗したため中止: {}",
                        err.message()
                    ));
                }
            }
        }
        let remaining = config.games == 0 || played < config.games;
        if remaining && config.reconnect_wait_sec > 0 {
            std::thread::sleep(Duration::from_secs(config.reconnect_wait_sec));
        }
    }
    Ok(summary)
}

/// 1 接続 = 1 対局 (floodgate は対局後にログアウト状態へ戻る)．
fn play_one_connection(
    config: &CsaConfig,
    evaluator: &Arc<crate::backend::EngineEvaluator>,
) -> Result<CsaGameResult, SessionError> {
    let mut session = Session::connect(config)?;
    session.login(config)?;
    let summary = session.wait_game_summary(config)?;
    eprintln!(
        "[csa] 対局条件: id={} 先手={} 後手={} 自分={} 持ち時間={}秒 秒読み={}秒 加算={}秒 最大手数={}",
        summary.game_id,
        summary.name_black,
        summary.name_white,
        color_name(summary.my_color),
        summary.time.total_ms.unwrap_or(0) / 1000,
        summary.time.byoyomi_ms / 1000,
        summary.time.increment_ms / 1000,
        summary.max_moves.unwrap_or(0),
    );

    // 対局条件が確定してからエンジンを組む (打ち切り手数を探索へ渡すため)
    let mut engine = config.engine.clone();
    // 消費時間はサーバが計測する = 通信往復と driver の遅れを予算から引く必要がある
    // (自己対局 driver の clock_margin_ms と同じ手当て．USI/自己対局に続く 3 本目の経路)
    engine.time.network_delay_ms = clock_margin_ms(engine.time.network_delay_ms);
    if let Some(max_moves) = summary.max_moves {
        engine.max_moves_to_draw = max_moves;
    }
    let ev = Arc::clone(evaluator);
    let mut agent = Agent::new(engine, move |cfg: &EngineConfig| {
        Ok(MaouSearchBackend::from_shared(
            Arc::clone(&ev),
            search_options(cfg),
            cfg.subtree_reuse,
        ))
    });
    agent
        .handle(GuiCommand::UsiNewGame)
        .map_err(|e| SessionError::Retryable(format!("usinewgame 失敗: {e}")))?;
    agent
        .handle(GuiCommand::IsReady)
        .map_err(|e| SessionError::Retryable(format!("isready 失敗: {e}")))?;

    session.send(&format!("AGREE {}", summary.game_id))?;
    session.wait_start(&summary)?;

    let result = play_game(&mut session, &mut agent, &summary, config)?;
    // 対局後は対局待ち状態へ戻る．floodgate は接続も切るが，明示的に落とす
    let _ = session.send("LOGOUT");
    Ok(result)
}

/// 手番の表示名．
fn color_name(color: Color) -> &'static str {
    match color {
        Color::Black => "black",
        Color::White => "white",
    }
}

/// 手番を残り時間配列の添字へ．
fn color_index(color: Color) -> usize {
    usize::from(color == Color::White)
}

/// 対局中の時計 (サーバ通知の持ち時間規定を追跡する)．
struct GameClock {
    rule: TimeRule,
    /// 残り持ち時間 [先手, 後手]．
    remaining: [u64; 2],
    /// 持ち時間・秒読み・加算のいずれかが定義されているか．
    limited: bool,
}

impl GameClock {
    /// `consumed_units` は再開局面までに消費された時間 (`Time_Unit` 単位)．
    /// 途中再開の対局で持ち時間を過大評価しないために差し引く．
    fn new(rule: &TimeRule, consumed_units: [u64; 2]) -> Self {
        let total = rule.total_ms.unwrap_or(0);
        let spent = |units: u64| units.saturating_mul(rule.unit_ms);
        Self {
            rule: *rule,
            remaining: [
                total.saturating_sub(spent(consumed_units[0])),
                total.saturating_sub(spent(consumed_units[1])),
            ],
            limited: rule.total_ms.is_some() || rule.byoyomi_ms > 0 || rule.increment_ms > 0,
        }
    }

    /// 手番開始直前の加算 (フィッシャー)．
    fn add_increment(&mut self, color: Color) {
        let i = color_index(color);
        self.remaining[i] = self.remaining[i].saturating_add(self.rule.increment_ms);
    }

    /// サーバが計測した消費時間を残り時間から引く．
    ///
    /// 残り時間を超える消費は秒読みで賄われたものとして 0 で止める
    /// (時間切れの判定はサーバが行う)．
    fn consume(&mut self, color: Color, consumed_units: u64) {
        let i = color_index(color);
        let spent = consumed_units.saturating_mul(self.rule.unit_ms);
        self.remaining[i] = self.remaining[i].saturating_sub(spent);
    }

    /// 1 手の探索指示を組み立てる．
    ///
    /// 予算配分そのものは [`crate::time`] (TimeStrategy) が行う — ここは
    /// CSA の持ち時間規定を USI と同じ [`ClockParams`] へ写すだけ
    /// (「持ち時間の消費計画は別レイヤー」の分界)．
    fn go_params(&self) -> GoParams {
        if !self.limited {
            return GoParams {
                movetime: Some(NO_LIMIT_MOVETIME_MS),
                ..GoParams::default()
            };
        }
        GoParams {
            clock: ClockParams {
                btime: Some(self.remaining[0]),
                wtime: Some(self.remaining[1]),
                byoyomi: (self.rule.byoyomi_ms > 0).then_some(self.rule.byoyomi_ms),
                binc: (self.rule.increment_ms > 0).then_some(self.rule.increment_ms),
                winc: (self.rule.increment_ms > 0).then_some(self.rule.increment_ms),
            },
            ..GoParams::default()
        }
    }
}

/// TCP セッション (reader スレッド + 書き込み)．
struct Session {
    writer: TcpStream,
    rx: Receiver<String>,
    last_send: Instant,
    keep_alive: Duration,
    verbose: bool,
}

impl Session {
    fn connect(config: &CsaConfig) -> Result<Session, SessionError> {
        let addr = format!("{}:{}", config.host, config.port);
        let target = addr
            .to_socket_addrs()
            .map_err(|e| SessionError::Retryable(format!("名前解決に失敗 ({addr}): {e}")))?
            .next()
            .ok_or_else(|| SessionError::Retryable(format!("接続先が解決できない: {addr}")))?;
        let stream =
            TcpStream::connect_timeout(&target, Duration::from_secs(config.connect_timeout_sec))
                .map_err(|e| SessionError::Retryable(format!("接続に失敗 ({addr}): {e}")))?;
        // 指し手 1 行は即座に送る (Nagle で結合させない)
        let _ = stream.set_nodelay(true);
        let reader = stream
            .try_clone()
            .map_err(|e| SessionError::Retryable(format!("ソケットの複製に失敗: {e}")))?;
        let (tx, rx) = mpsc::channel::<String>();
        std::thread::spawn(move || {
            // EOF・受信エラーでは tx を落とす = 受信側は切断として観測する
            for line in BufReader::new(reader).lines() {
                match line {
                    Ok(line) => {
                        if tx.send(line).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        });
        Ok(Session {
            writer: stream,
            rx,
            last_send: Instant::now(),
            keep_alive: Duration::from_secs(config.keep_alive_sec.max(KEEP_ALIVE_MIN_SEC)),
            verbose: config.verbose,
        })
    }

    /// 1 行送る (末尾に LF を付ける)．空文字列は keep-alive の LF 単体になる．
    fn send(&mut self, line: &str) -> Result<(), SessionError> {
        if self.verbose && !line.is_empty() {
            eprintln!("[csa] > {line}");
        }
        let mut buf = String::with_capacity(line.len() + 1);
        buf.push_str(line);
        buf.push('\n');
        self.writer
            .write_all(buf.as_bytes())
            .and_then(|()| self.writer.flush())
            .map_err(|e| SessionError::Retryable(format!("送信に失敗: {e}")))?;
        self.last_send = Instant::now();
        Ok(())
    }

    /// 次の行を待つ．待機中は keep-alive を送る．`deadline` 到達で `None`．
    fn recv_line(&mut self, deadline: Option<Instant>) -> Result<Option<String>, SessionError> {
        loop {
            let now = Instant::now();
            if let Some(limit) = deadline {
                if now >= limit {
                    return Ok(None);
                }
            }
            // keep-alive の期日と deadline のうち近い方まで待つ
            let mut wait = self.keep_alive.saturating_sub(self.last_send.elapsed());
            if let Some(limit) = deadline {
                wait = wait.min(limit.saturating_duration_since(now));
            }
            let wait = wait.max(Duration::from_millis(50));
            match self.rx.recv_timeout(wait) {
                Ok(line) => {
                    if self.verbose {
                        eprintln!("[csa] < {line}");
                    }
                    return Ok(Some(line));
                }
                Err(RecvTimeoutError::Timeout) => {
                    if self.last_send.elapsed() >= self.keep_alive {
                        // 空行 1 つ = CSA の keep-alive
                        self.send("")?;
                    }
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(SessionError::Retryable("サーバから切断された".to_string()));
                }
            }
        }
    }

    /// ログインする．
    fn login(&mut self, config: &CsaConfig) -> Result<(), SessionError> {
        // パスワードは平文で流れるため通信ログには出さない
        if self.verbose {
            eprintln!("[csa] > LOGIN {} <password>", config.login_name);
        }
        let verbose = self.verbose;
        self.verbose = false;
        let sent = self.send(&format!("LOGIN {} {}", config.login_name, config.password));
        self.verbose = verbose;
        sent?;
        let deadline = Instant::now() + Duration::from_secs(config.connect_timeout_sec);
        loop {
            let Some(line) = self.recv_line(Some(deadline))? else {
                return Err(SessionError::Retryable(
                    "ログイン応答がタイムアウト".to_string(),
                ));
            };
            match parse_server_line(&line) {
                ServerMessage::LoginOk(name) => {
                    eprintln!("[csa] ログイン成功: {name}");
                    return Ok(());
                }
                ServerMessage::LoginIncorrect => {
                    return Err(SessionError::Fatal(
                        "ログインを拒否された (LOGIN:incorrect)．ログイン名とパスワードの書式を確認する \
                         (floodgate のパスワード欄は `<ゲーム名>,<trip>`)"
                            .to_string(),
                    ));
                }
                _ => continue,
            }
        }
    }

    /// 対局条件 (`Game_Summary`) を待つ．
    fn wait_game_summary(&mut self, config: &CsaConfig) -> Result<GameSummary, SessionError> {
        let deadline = (config.game_wait_sec > 0)
            .then(|| Instant::now() + Duration::from_secs(config.game_wait_sec));
        eprintln!("[csa] 対局条件を待機中 (floodgate は毎時 0 分/30 分に対局が組まれる)");
        let mut builder: Option<GameSummaryBuilder> = None;
        loop {
            let Some(line) = self.recv_line(deadline)? else {
                return Err(SessionError::Retryable(
                    "対局条件の待機がタイムアウト".to_string(),
                ));
            };
            if let Some(active) = builder.as_mut() {
                match active.push(&line) {
                    Ok(Some(summary)) => return Ok(summary),
                    Ok(None) => {}
                    Err(e) => {
                        return Err(SessionError::Retryable(format!(
                            "対局条件のパースに失敗: {e}"
                        )))
                    }
                }
                continue;
            }
            if matches!(parse_server_line(&line), ServerMessage::GameSummaryBegin) {
                builder = Some(GameSummaryBuilder::default());
            }
        }
    }

    /// `START:` を待つ (`REJECT:` は失敗扱い)．
    fn wait_start(&mut self, summary: &GameSummary) -> Result<(), SessionError> {
        let deadline = Instant::now() + Duration::from_secs(120);
        loop {
            let Some(line) = self.recv_line(Some(deadline))? else {
                return Err(SessionError::Retryable(
                    "START の待機がタイムアウト".to_string(),
                ));
            };
            match parse_server_line(&line) {
                ServerMessage::Start(id) => {
                    if !id.is_empty() && !summary.game_id.is_empty() && id != summary.game_id {
                        eprintln!("[csa] 警告: START の対局 ID が対局条件と異なる ({id})");
                    }
                    return Ok(());
                }
                ServerMessage::Reject { game_id, by } => {
                    return Err(SessionError::Retryable(format!(
                        "対局が拒否された: {game_id} by {by}"
                    )));
                }
                _ => continue,
            }
        }
    }
}

/// 1 対局を最後まで駆動する．
fn play_game<B, F>(
    session: &mut Session,
    agent: &mut Agent<B, F>,
    summary: &GameSummary,
    config: &CsaConfig,
) -> Result<CsaGameResult, SessionError>
where
    B: SearchBackend,
    F: Fn(&EngineConfig) -> Result<B, String>,
{
    let start = Instant::now();
    let mut moves = summary.init_moves.clone();
    let (mut board, _) = build_board_and_history(&summary.start_sfen, &moves)
        .map_err(|e| SessionError::Retryable(format!("再開局面の構築に失敗: {e}")))?;
    let mut clock = GameClock::new(&summary.time, summary.init_consumed_units);
    let my_color = summary.my_color;
    let opponent = match my_color {
        Color::Black => summary.name_white.clone(),
        Color::White => summary.name_black.clone(),
    };
    let mut reason = String::new();

    // 自分が先に指す場合はサーバからの通知を待たずに思考する
    if board.turn() == my_color {
        think_and_send(session, agent, &board, &moves, summary, &mut clock)?;
    }
    let result = loop {
        let Some(line) = session.recv_line(None)? else {
            return Err(SessionError::Retryable("対局中に待機が終了".to_string()));
        };
        match parse_server_line(&line) {
            ServerMessage::Move { token, consumed } => {
                let side = board.turn();
                let Some(mv) = move_from_csa(&board, &token) else {
                    return Err(SessionError::Retryable(format!(
                        "サーバの指し手を解決できない: {token}"
                    )));
                };
                board.do_move(mv);
                moves.push(mv.to_usi());
                clock.consume(side, consumed);
                if board.turn() == my_color {
                    think_and_send(session, agent, &board, &moves, summary, &mut clock)?;
                }
            }
            ServerMessage::Special { token, consumed } => {
                // %TORYO / %KACHI / %CHUDAN のエコー．勝敗は続く `#` 行で確定する
                clock.consume(board.turn(), consumed);
                reason = token;
            }
            ServerMessage::Status(status) => {
                if is_terminal_status(&status) {
                    break status.trim_start_matches('#').to_lowercase();
                }
                // 終局理由 (#RESIGN / #SENNICHITE / #TIME_UP / #MAX_MOVES 等)
                reason = status;
            }
            ServerMessage::LogoutCompleted => {
                return Err(SessionError::Retryable(
                    "対局中にログアウトされた".to_string(),
                ));
            }
            _ => {}
        }
    };
    if config.verbose {
        eprintln!("[csa] 棋譜 (USI): {}", moves.join(" "));
    }
    Ok(CsaGameResult {
        game_id: summary.game_id.clone(),
        opponent,
        my_color: color_name(my_color).to_string(),
        moves: moves.len() as u32,
        result,
        reason,
        elapsed_ms: start.elapsed().as_millis() as u64,
    })
}

/// 自分の手番: 思考して指し手 (または投了・宣言) を送る．
fn think_and_send<B, F>(
    session: &mut Session,
    agent: &mut Agent<B, F>,
    board: &Board,
    moves: &[String],
    summary: &GameSummary,
    clock: &mut GameClock,
) -> Result<(), SessionError>
where
    B: SearchBackend,
    F: Fn(&EngineConfig) -> Result<B, String>,
{
    let my_color = summary.my_color;
    // 加算は手番側の計測開始直前に行われる (仕様 3.2.2)
    clock.add_increment(my_color);
    agent
        .handle(GuiCommand::Position {
            sfen: Some(summary.start_sfen.clone()),
            moves: moves.to_vec(),
        })
        .map_err(|e| SessionError::Retryable(format!("position 失敗: {e}")))?;
    // transport 規約: `go` の直前に停止フラグをクリアする (stdio の reader と
    // 同じ役割を driver が担う)．怠ると前の手の時間切れ停止が残り，以降の
    // 探索が即座に打ち切られる
    agent.stop_handle().store(false, Ordering::Release);
    let out = agent
        .handle(GuiCommand::Go(clock.go_params()))
        .map_err(|e| SessionError::Retryable(format!("go 失敗: {e}")))?;
    let best = out
        .iter()
        .rev()
        .find_map(|c| match c {
            EngineCommand::BestMove { mv, .. } => Some(mv.clone()),
            _ => None,
        })
        .ok_or_else(|| SessionError::Retryable("bestmove が返らない".to_string()))?;
    let line = match best {
        BestMoveKind::Move(usi) => {
            let mv = generate_legal_moves(&mut board.clone())
                .into_iter()
                .find(|m| m.to_usi() == usi)
                .ok_or_else(|| {
                    SessionError::Retryable(format!("エンジンが非合法手を返した: {usi}"))
                })?;
            move_to_csa(my_color, mv)
        }
        BestMoveKind::Resign => "%TORYO".to_string(),
        BestMoveKind::Win => "%KACHI".to_string(),
    };
    session.send(&line)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rule() -> TimeRule {
        TimeRule {
            unit_ms: 1000,
            total_ms: Some(300_000),
            byoyomi_ms: 0,
            increment_ms: 10_000,
            delay_ms: 0,
            least_per_move_ms: 0,
        }
    }

    #[test]
    fn clock_tracks_increment_and_consumption() {
        let mut clock = GameClock::new(&rule(), [0, 0]);
        assert_eq!(clock.remaining, [300_000, 300_000]);
        // 先手番の開始: 加算が乗る
        clock.add_increment(Color::Black);
        assert_eq!(clock.remaining[0], 310_000);
        // サーバが 12 秒と計測した
        clock.consume(Color::Black, 12);
        assert_eq!(clock.remaining[0], 298_000);
        assert_eq!(clock.remaining[1], 300_000);
    }

    #[test]
    fn clock_floors_at_zero() {
        let mut clock = GameClock::new(
            &TimeRule {
                increment_ms: 0,
                ..rule()
            },
            [0, 0],
        );
        clock.consume(Color::White, 9999);
        assert_eq!(clock.remaining[1], 0);
    }

    #[test]
    fn go_params_map_to_usi_clock() {
        let mut clock = GameClock::new(&rule(), [0, 0]);
        clock.add_increment(Color::Black);
        let go = clock.go_params();
        assert_eq!(go.clock.btime, Some(310_000));
        assert_eq!(go.clock.wtime, Some(300_000));
        assert_eq!(go.clock.binc, Some(10_000));
        assert_eq!(go.clock.winc, Some(10_000));
        assert_eq!(go.clock.byoyomi, None);
        assert_eq!(go.movetime, None);
    }

    #[test]
    fn byoyomi_rule_maps_to_byoyomi() {
        let clock = GameClock::new(
            &TimeRule {
                increment_ms: 0,
                byoyomi_ms: 10_000,
                ..rule()
            },
            [0, 0],
        );
        let go = clock.go_params();
        assert_eq!(go.clock.byoyomi, Some(10_000));
        assert_eq!(go.clock.binc, None);
    }

    #[test]
    fn unlimited_rule_falls_back_to_movetime() {
        let clock = GameClock::new(
            &TimeRule {
                total_ms: None,
                byoyomi_ms: 0,
                increment_ms: 0,
                ..rule()
            },
            [0, 0],
        );
        let go = clock.go_params();
        assert_eq!(go.movetime, Some(NO_LIMIT_MOVETIME_MS));
        assert!(go.clock.is_empty());
    }

    #[test]
    fn time_unit_scales_consumption() {
        // Time_Unit:1min の対局では T1 = 60 秒
        let mut clock = GameClock::new(
            &TimeRule {
                unit_ms: 60_000,
                total_ms: Some(600_000),
                increment_ms: 0,
                ..rule()
            },
            [0, 0],
        );
        clock.consume(Color::Black, 1);
        assert_eq!(clock.remaining[0], 540_000);
    }

    #[test]
    fn resumed_game_starts_from_reduced_clock() {
        // 途中再開: Position 階層の指し手で消費された分を引いて始める
        // (引かないと持ち時間を過大評価して時間切れ負けし得る)
        let clock = GameClock::new(&rule(), [12, 6]);
        assert_eq!(clock.remaining, [288_000, 294_000]);
    }

    #[test]
    fn resumed_consumption_cannot_go_negative() {
        let clock = GameClock::new(&rule(), [9999, 0]);
        assert_eq!(clock.remaining[0], 0);
    }
}
