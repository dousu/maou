//! USI プロトコルの行 ⇔ 型付きコマンド変換 (pure — IO・戦略なし)．
//!
//! GUI 方言 (docs/design/usi-engine/index.md §2) をここで吸収する:
//! - `position startpos` は `moves` を含まないことがある
//! - `setoption` は `value` なし (button 型) があり得る
//! - `ponderhit` はやねうら王拡張で残り時間を伴うことがある
//! - 未知のコマンド・空行は [`GuiCommand::Unknown`] として無害化する
//! - `info` の `pv` は行末尾に置く (シリアライザが構造的に保証する)

/// 持ち時間パラメータ (`go` / 拡張 `ponderhit` に付く．全てミリ秒)．
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct ClockParams {
    /// 先手の残り持ち時間．
    pub btime: Option<u64>,
    /// 後手の残り持ち時間．
    pub wtime: Option<u64>,
    /// 秒読み (1 手ごとの追加時間，持ち時間消費後)．
    pub byoyomi: Option<u64>,
    /// 先手のフィッシャー加算．
    pub binc: Option<u64>,
    /// 後手のフィッシャー加算．
    pub winc: Option<u64>,
}

impl ClockParams {
    /// 時間パラメータを 1 つも含まないか．
    pub fn is_empty(&self) -> bool {
        self.btime.is_none()
            && self.wtime.is_none()
            && self.byoyomi.is_none()
            && self.binc.is_none()
            && self.winc.is_none()
    }
}

/// `go mate` の予算．
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MateLimit {
    /// 時間予算 (ミリ秒)．
    TimeMs(u64),
    /// 無制限．
    Infinite,
}

/// `go` コマンドのパラメータ．
#[derive(Clone, Debug, PartialEq, Default)]
pub struct GoParams {
    /// 持ち時間．
    pub clock: ClockParams,
    /// 先読み探索 (`go ponder`)．
    pub ponder: bool,
    /// 無制限探索 (`go infinite`，検討モード)．
    pub infinite: bool,
    /// 詰み探索 (`go mate <ms|infinite>`)．
    pub mate: Option<MateLimit>,
    /// 1 手の思考時間固定 (原案の `go movetime`．将棋所は未使用)．
    pub movetime: Option<u64>,
    /// ノード数予算 (原案の `go nodes`．ShogiGUI 検討が使う)．
    pub nodes: Option<u64>,
    /// 深さ予算 (原案の `go depth`．MCTS では未対応 — 受理して無視)．
    pub depth: Option<u32>,
}

/// `gameover` の結果．
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GameResult {
    /// 勝ち．
    Win,
    /// 負け．
    Lose,
    /// 引き分け．
    Draw,
    /// 不明 (中断等．USI2.0 案の `gameover unknown` を含む)．
    Unknown,
}

/// GUI → エンジンのコマンド．
#[derive(Clone, Debug, PartialEq)]
pub enum GuiCommand {
    /// `usi` — 初期化要求．
    Usi,
    /// `isready` — 対局準備要求 (重い初期化はここで行う)．
    IsReady,
    /// `setoption name <name> [value <value>]`．
    SetOption {
        /// オプション名．
        name: String,
        /// 値 (button 型は `None`)．
        value: Option<String>,
    },
    /// `usinewgame` — 新規対局開始通知．
    UsiNewGame,
    /// `position startpos|sfen <sfen> [moves ...]`．
    Position {
        /// 基準局面 SFEN (`None` = 平手初期局面 startpos)．
        sfen: Option<String>,
        /// 基準局面からの USI 指し手列．
        moves: Vec<String>,
    },
    /// `go ...` — 思考開始．
    Go(GoParams),
    /// `stop` — 思考中断 (即時に bestmove を返す)．
    Stop,
    /// `ponderhit [時間パラメータ]` — 先読み的中 (時刻付きはやねうら王拡張)．
    PonderHit(ClockParams),
    /// `gameover win|lose|draw`．
    GameOver(GameResult),
    /// `quit` — 終了．
    Quit,
    /// 認識できない行 (無視する)．
    Unknown(String),
}

/// 1 行をパースする．空行・空白のみの行は `None`．
pub fn parse_line(line: &str) -> Option<GuiCommand> {
    let mut tokens = line.split_whitespace();
    let head = tokens.next()?;
    let rest: Vec<&str> = tokens.collect();
    Some(match head {
        "usi" => GuiCommand::Usi,
        "isready" => GuiCommand::IsReady,
        "setoption" => parse_setoption(&rest),
        "usinewgame" => GuiCommand::UsiNewGame,
        "position" => parse_position(&rest),
        "go" => GuiCommand::Go(parse_go(&rest)),
        "stop" => GuiCommand::Stop,
        "ponderhit" => GuiCommand::PonderHit(parse_clock(&rest)),
        "gameover" => GuiCommand::GameOver(match rest.first().copied() {
            Some("win") => GameResult::Win,
            Some("lose") => GameResult::Lose,
            Some("draw") => GameResult::Draw,
            _ => GameResult::Unknown,
        }),
        "quit" => GuiCommand::Quit,
        _ => GuiCommand::Unknown(line.trim().to_string()),
    })
}

fn parse_setoption(rest: &[&str]) -> GuiCommand {
    // "name <トークン列> [value <トークン列>]"．name/value とも空白を含み得る
    let mut name_parts: Vec<&str> = Vec::new();
    let mut value_parts: Vec<&str> = Vec::new();
    let mut mode = 0u8; // 0: 先頭, 1: name 読み取り中, 2: value 読み取り中
    for t in rest {
        match (*t, mode) {
            ("name", 0 | 1) => mode = 1,
            ("value", 1) => mode = 2,
            (_, 1) => name_parts.push(t),
            (_, 2) => value_parts.push(t),
            _ => {}
        }
    }
    let name = name_parts.join(" ");
    if name.is_empty() {
        return GuiCommand::Unknown(format!("setoption {}", rest.join(" ")));
    }
    let value = if mode == 2 {
        Some(value_parts.join(" "))
    } else {
        None
    };
    GuiCommand::SetOption { name, value }
}

fn parse_position(rest: &[&str]) -> GuiCommand {
    let mut it = rest.iter().copied().peekable();
    let sfen = match it.next() {
        Some("startpos") => None,
        Some("sfen") => {
            // "moves" が来るまでのトークンが SFEN (盤・手番・持駒・手数の 4 要素)
            let mut sfen_parts: Vec<&str> = Vec::new();
            while let Some(&t) = it.peek() {
                if t == "moves" {
                    break;
                }
                sfen_parts.push(t);
                it.next();
            }
            if sfen_parts.is_empty() {
                return GuiCommand::Unknown(format!("position {}", rest.join(" ")));
            }
            Some(sfen_parts.join(" "))
        }
        _ => return GuiCommand::Unknown(format!("position {}", rest.join(" "))),
    };
    let mut moves: Vec<String> = Vec::new();
    if let Some("moves") = it.next() {
        moves.extend(it.map(str::to_string));
    }
    GuiCommand::Position { sfen, moves }
}

fn parse_u64(it: &mut std::slice::Iter<'_, &str>) -> Option<u64> {
    it.next().and_then(|t| t.parse().ok())
}

fn parse_clock(rest: &[&str]) -> ClockParams {
    let mut clock = ClockParams::default();
    let mut it = rest.iter();
    while let Some(&t) = it.next() {
        match t {
            "btime" => clock.btime = it.next().and_then(|t| t.parse().ok()),
            "wtime" => clock.wtime = it.next().and_then(|t| t.parse().ok()),
            "byoyomi" => clock.byoyomi = it.next().and_then(|t| t.parse().ok()),
            "binc" => clock.binc = it.next().and_then(|t| t.parse().ok()),
            "winc" => clock.winc = it.next().and_then(|t| t.parse().ok()),
            _ => {}
        }
    }
    clock
}

fn parse_go(rest: &[&str]) -> GoParams {
    let mut p = GoParams {
        clock: parse_clock(rest),
        ..GoParams::default()
    };
    let mut it = rest.iter();
    while let Some(&t) = it.next() {
        match t {
            "ponder" => p.ponder = true,
            "infinite" => p.infinite = true,
            "mate" => {
                p.mate = Some(match it.next() {
                    Some(&"infinite") => MateLimit::Infinite,
                    Some(ms) => match ms.parse() {
                        Ok(v) => MateLimit::TimeMs(v),
                        Err(_) => MateLimit::Infinite,
                    },
                    None => MateLimit::Infinite,
                });
            }
            "movetime" => p.movetime = parse_u64(&mut it),
            "nodes" => p.nodes = parse_u64(&mut it),
            "depth" => p.depth = it.next().and_then(|t| t.parse().ok()),
            _ => {}
        }
    }
    p
}

/// エンジンオプション宣言の型 (`option name ... type ...`)．
#[derive(Clone, Debug, PartialEq)]
pub enum OptionKind {
    /// 真偽値．
    Check {
        /// 既定値．
        default: bool,
    },
    /// 整数 (範囲付き)．
    Spin {
        /// 既定値．
        default: i64,
        /// 最小値．
        min: i64,
        /// 最大値．
        max: i64,
    },
    /// 文字列．
    String {
        /// 既定値 (空は `<empty>` として宣言される)．
        default: String,
    },
    /// ファイルパス．
    Filename {
        /// 既定値 (空は `<empty>` として宣言される)．
        default: String,
    },
}

/// エンジンオプション宣言．
#[derive(Clone, Debug, PartialEq)]
pub struct OptionDecl {
    /// オプション名．
    pub name: &'static str,
    /// 型と既定値．
    pub kind: OptionKind,
}

/// `info score` の値．
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Score {
    /// 評価値 (センチポーン相当，手番視点)．
    Cp(i32),
    /// 詰み (正 = 手番側の勝ち，値は手数．負 = 手番側の負け)．
    Mate(i32),
}

/// `info` の内容．`pv` は必ず行末尾にシリアライズされる (フィールド順は
/// 構造体定義でなくシリアライザが保証する)．
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Info {
    /// 探索深さ．
    pub depth: Option<u32>,
    /// 到達最大深さ．
    pub seldepth: Option<u32>,
    /// 経過時間 (ミリ秒)．
    pub time_ms: Option<u64>,
    /// 探索ノード数 (playout 数)．
    pub nodes: Option<u64>,
    /// ノード毎秒．
    pub nps: Option<u64>,
    /// 評価値．
    pub score: Option<Score>,
    /// 任意文字列 (ASCII 安全に保つこと — GUI のエンコーディング差のため)．
    pub string: Option<String>,
    /// 読み筋 (USI 指し手列)．非空なら必ず行末尾に出力される．
    pub pv: Vec<String>,
}

/// `bestmove` の指し手．
#[derive(Clone, Debug, PartialEq)]
pub enum BestMoveKind {
    /// 通常の指し手 (USI 表記)．
    Move(String),
    /// 投了．
    Resign,
    /// 入玉宣言勝ち．
    Win,
}

/// エンジン → GUI のコマンド．
#[derive(Clone, Debug, PartialEq)]
pub enum EngineCommand {
    /// `id name <name>` + `id author <author>` (2 行)．
    Id {
        /// エンジン名．
        name: String,
        /// 作者名．
        author: String,
    },
    /// `usiok`．
    UsiOk,
    /// `readyok`．
    ReadyOk,
    /// オプション宣言 1 件．
    OptionDecl(OptionDecl),
    /// `bestmove <move|resign|win> [ponder <move>]`．
    BestMove {
        /// 指し手．
        mv: BestMoveKind,
        /// 相手の予想手 (ponder 対象)．
        ponder: Option<String>,
    },
    /// `info ...`．
    Info(Info),
    /// `checkmate notimplemented` (`go mate` 未対応の応答)．
    CheckmateNotImplemented,
}

/// コマンドを USI の行 (改行なし) へシリアライズする．
///
/// [`EngineCommand::Id`] のみ 2 行 (`\n` 区切り) を返す．
pub fn serialize(cmd: &EngineCommand) -> String {
    match cmd {
        EngineCommand::Id { name, author } => {
            format!("id name {name}\nid author {author}")
        }
        EngineCommand::UsiOk => "usiok".to_string(),
        EngineCommand::ReadyOk => "readyok".to_string(),
        EngineCommand::OptionDecl(decl) => serialize_option_decl(decl),
        EngineCommand::BestMove { mv, ponder } => {
            let mv = match mv {
                BestMoveKind::Move(m) => m.as_str(),
                BestMoveKind::Resign => "resign",
                BestMoveKind::Win => "win",
            };
            match ponder {
                Some(p) => format!("bestmove {mv} ponder {p}"),
                None => format!("bestmove {mv}"),
            }
        }
        EngineCommand::Info(info) => serialize_info(info),
        EngineCommand::CheckmateNotImplemented => "checkmate notimplemented".to_string(),
    }
}

fn serialize_option_decl(decl: &OptionDecl) -> String {
    let name = decl.name;
    match &decl.kind {
        OptionKind::Check { default } => {
            format!("option name {name} type check default {default}")
        }
        OptionKind::Spin { default, min, max } => {
            format!("option name {name} type spin default {default} min {min} max {max}")
        }
        OptionKind::String { default } => {
            let d = if default.is_empty() {
                "<empty>"
            } else {
                default
            };
            format!("option name {name} type string default {d}")
        }
        OptionKind::Filename { default } => {
            let d = if default.is_empty() {
                "<empty>"
            } else {
                default
            };
            format!("option name {name} type filename default {d}")
        }
    }
}

fn serialize_info(info: &Info) -> String {
    let mut parts: Vec<String> = vec!["info".to_string()];
    if let Some(v) = info.depth {
        parts.push(format!("depth {v}"));
    }
    if let Some(v) = info.seldepth {
        parts.push(format!("seldepth {v}"));
    }
    if let Some(v) = info.time_ms {
        parts.push(format!("time {v}"));
    }
    if let Some(v) = info.nodes {
        parts.push(format!("nodes {v}"));
    }
    if let Some(v) = info.nps {
        parts.push(format!("nps {v}"));
    }
    match info.score {
        Some(Score::Cp(v)) => parts.push(format!("score cp {v}")),
        Some(Score::Mate(v)) => parts.push(format!("score mate {v}")),
        None => {}
    }
    if let Some(s) = &info.string {
        parts.push(format!("string {s}"));
    }
    // pv は必ず末尾 (末尾でないと正しくパースできない GUI/ホストがある)
    if !info.pv.is_empty() {
        parts.push(format!("pv {}", info.pv.join(" ")));
    }
    parts.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_commands() {
        assert_eq!(parse_line("usi"), Some(GuiCommand::Usi));
        assert_eq!(parse_line("isready"), Some(GuiCommand::IsReady));
        assert_eq!(parse_line("usinewgame"), Some(GuiCommand::UsiNewGame));
        assert_eq!(parse_line("stop"), Some(GuiCommand::Stop));
        assert_eq!(parse_line("quit"), Some(GuiCommand::Quit));
        assert_eq!(parse_line("  "), None);
        assert_eq!(parse_line(""), None);
    }

    #[test]
    fn test_parse_unknown_is_harmless() {
        assert_eq!(
            parse_line("hoge fuga"),
            Some(GuiCommand::Unknown("hoge fuga".to_string()))
        );
    }

    #[test]
    fn test_parse_position_startpos_without_moves() {
        // 将棋所は開始局面で moves を付けない (方言)
        assert_eq!(
            parse_line("position startpos"),
            Some(GuiCommand::Position {
                sfen: None,
                moves: vec![]
            })
        );
    }

    #[test]
    fn test_parse_position_startpos_moves() {
        assert_eq!(
            parse_line("position startpos moves 7g7f 3c3d"),
            Some(GuiCommand::Position {
                sfen: None,
                moves: vec!["7g7f".to_string(), "3c3d".to_string()]
            })
        );
    }

    #[test]
    fn test_parse_position_sfen() {
        let line = "position sfen lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1 moves 7g7f";
        assert_eq!(
            parse_line(line),
            Some(GuiCommand::Position {
                sfen: Some(
                    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1".to_string()
                ),
                moves: vec!["7g7f".to_string()]
            })
        );
    }

    #[test]
    fn test_parse_go_byoyomi() {
        let cmd = parse_line("go btime 60000 wtime 50000 byoyomi 10000").unwrap();
        assert_eq!(
            cmd,
            GuiCommand::Go(GoParams {
                clock: ClockParams {
                    btime: Some(60000),
                    wtime: Some(50000),
                    byoyomi: Some(10000),
                    ..ClockParams::default()
                },
                ..GoParams::default()
            })
        );
    }

    #[test]
    fn test_parse_go_fischer() {
        let cmd = parse_line("go btime 60000 wtime 50000 binc 2000 winc 2000").unwrap();
        let GuiCommand::Go(p) = cmd else {
            panic!("go を期待")
        };
        assert_eq!(p.clock.binc, Some(2000));
        assert_eq!(p.clock.winc, Some(2000));
        assert!(!p.infinite && !p.ponder);
    }

    #[test]
    fn test_parse_go_ponder_and_infinite() {
        let GuiCommand::Go(p) = parse_line("go ponder btime 1000 wtime 1000 byoyomi 500").unwrap()
        else {
            panic!()
        };
        assert!(p.ponder);
        let GuiCommand::Go(p) = parse_line("go infinite").unwrap() else {
            panic!()
        };
        assert!(p.infinite);
        assert!(p.clock.is_empty());
    }

    #[test]
    fn test_parse_go_mate() {
        let GuiCommand::Go(p) = parse_line("go mate 30000").unwrap() else {
            panic!()
        };
        assert_eq!(p.mate, Some(MateLimit::TimeMs(30000)));
        let GuiCommand::Go(p) = parse_line("go mate infinite").unwrap() else {
            panic!()
        };
        assert_eq!(p.mate, Some(MateLimit::Infinite));
    }

    #[test]
    fn test_parse_go_extended_budgets() {
        let GuiCommand::Go(p) = parse_line("go nodes 100000 depth 20 movetime 5000").unwrap()
        else {
            panic!()
        };
        assert_eq!(p.nodes, Some(100000));
        assert_eq!(p.depth, Some(20));
        assert_eq!(p.movetime, Some(5000));
    }

    #[test]
    fn test_parse_setoption() {
        assert_eq!(
            parse_line("setoption name ModelPath value /path/to model.onnx"),
            Some(GuiCommand::SetOption {
                name: "ModelPath".to_string(),
                // value は空白を含み得る
                value: Some("/path/to model.onnx".to_string())
            })
        );
        // button 型は value なし
        assert_eq!(
            parse_line("setoption name ClearHash"),
            Some(GuiCommand::SetOption {
                name: "ClearHash".to_string(),
                value: None
            })
        );
    }

    #[test]
    fn test_parse_ponderhit_with_times() {
        // やねうら王拡張: ponderhit に残り時間が付くことがある
        assert_eq!(
            parse_line("ponderhit btime 30000 wtime 20000 byoyomi 5000"),
            Some(GuiCommand::PonderHit(ClockParams {
                btime: Some(30000),
                wtime: Some(20000),
                byoyomi: Some(5000),
                ..ClockParams::default()
            }))
        );
        assert_eq!(
            parse_line("ponderhit"),
            Some(GuiCommand::PonderHit(ClockParams::default()))
        );
    }

    #[test]
    fn test_parse_gameover() {
        assert_eq!(
            parse_line("gameover win"),
            Some(GuiCommand::GameOver(GameResult::Win))
        );
        assert_eq!(
            parse_line("gameover draw"),
            Some(GuiCommand::GameOver(GameResult::Draw))
        );
        // USI2.0 案の unknown / 引数なしも安全に受ける
        assert_eq!(
            parse_line("gameover unknown"),
            Some(GuiCommand::GameOver(GameResult::Unknown))
        );
        assert_eq!(
            parse_line("gameover"),
            Some(GuiCommand::GameOver(GameResult::Unknown))
        );
    }

    #[test]
    fn test_serialize_bestmove() {
        assert_eq!(
            serialize(&EngineCommand::BestMove {
                mv: BestMoveKind::Move("7g7f".to_string()),
                ponder: Some("3c3d".to_string())
            }),
            "bestmove 7g7f ponder 3c3d"
        );
        assert_eq!(
            serialize(&EngineCommand::BestMove {
                mv: BestMoveKind::Resign,
                ponder: None
            }),
            "bestmove resign"
        );
        assert_eq!(
            serialize(&EngineCommand::BestMove {
                mv: BestMoveKind::Win,
                ponder: None
            }),
            "bestmove win"
        );
    }

    #[test]
    fn test_serialize_info_pv_is_last() {
        let info = Info {
            depth: Some(12),
            time_ms: Some(987),
            nodes: Some(12345),
            nps: Some(12000),
            score: Some(Score::Cp(153)),
            string: None,
            pv: vec!["7g7f".to_string(), "3c3d".to_string()],
            ..Info::default()
        };
        let line = serialize(&EngineCommand::Info(info));
        assert_eq!(
            line,
            "info depth 12 time 987 nodes 12345 nps 12000 score cp 153 pv 7g7f 3c3d"
        );
        // pv が末尾であること (末尾でないとパースできないホストがある)
        assert!(line.ends_with("pv 7g7f 3c3d"));
    }

    #[test]
    fn test_serialize_info_score_mate() {
        let line = serialize(&EngineCommand::Info(Info {
            score: Some(Score::Mate(-7)),
            ..Info::default()
        }));
        assert_eq!(line, "info score mate -7");
    }

    #[test]
    fn test_serialize_option_decls() {
        assert_eq!(
            serialize(&EngineCommand::OptionDecl(OptionDecl {
                name: "UseTensorRT",
                kind: OptionKind::Check { default: false }
            })),
            "option name UseTensorRT type check default false"
        );
        assert_eq!(
            serialize(&EngineCommand::OptionDecl(OptionDecl {
                name: "NetworkDelay",
                kind: OptionKind::Spin {
                    default: 1000,
                    min: 0,
                    max: 60000
                }
            })),
            "option name NetworkDelay type spin default 1000 min 0 max 60000"
        );
        assert_eq!(
            serialize(&EngineCommand::OptionDecl(OptionDecl {
                name: "ModelPath",
                kind: OptionKind::Filename {
                    default: String::new()
                }
            })),
            "option name ModelPath type filename default <empty>"
        );
    }

    #[test]
    fn test_serialize_id() {
        assert_eq!(
            serialize(&EngineCommand::Id {
                name: "maou 0.47.0".to_string(),
                author: "dousu".to_string()
            }),
            "id name maou 0.47.0\nid author dousu"
        );
    }
}
