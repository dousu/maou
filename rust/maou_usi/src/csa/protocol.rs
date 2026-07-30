//! CSA サーバプロトコル (ver 1.2.1) の行 ⇔ 型付きメッセージ変換 (pure — IO なし)．
//!
//! 仕様: <http://www.computer-shogi.org/protocol/tcp_ip_server_121.html>．
//! USI 側の [`crate::protocol`] と同じ役割を CSA 側で担う (parse/serialize のみ．
//! 戦略・IO は持たない)．
//!
//! 実装上の要点:
//!
//! - 指し手表記は `<符号><移動元 2 桁><移動先 2 桁><駒種 2 文字>` (駒打ちは移動元 `00`)．
//!   成りの場合の駒種は**成った後**の駒 (`+8822UM`)．
//! - サーバは指し手に消費時間を付けて返す (`+7776FU,T12`)．この消費時間が
//!   持ち時間計算の正 (クライアントの実測ではない — 遅延時間の控除等をサーバが行う)．
//! - 局面 (`BEGIN Position`) は既存の CSA 棋譜パーサ
//!   ([`maou_shogi::kifu::parse_csa_str`]) に委譲する (golden 検証済みの実装を再利用し，
//!   CSA 局面表記の 2 つ目の実装を作らない)．

use maou_shogi::board::Board;
use maou_shogi::kifu::parse_csa_str;
use maou_shogi::movegen::generate_legal_moves;
use maou_shogi::moves::Move;
use maou_shogi::types::Color;

/// 駒種の CSA 2 文字コード / 指し手の CSA 表記．
///
/// 実装は棋譜パーサと同じ [`maou_shogi::kifu`] にあり，ここでは再輸出する
/// だけにする (パーサと writer と transport で 3 つ目の表記実装を持たない)．
pub use maou_shogi::kifu::{move_to_csa, piece_type_to_csa as piece_type_csa};

/// CSA 表記の指し手を合法手の中から解決する．
///
/// 合法手を列挙して表記一致で選ぶ (USI 経路の
/// `generate_legal_moves(...).find(|m| m.to_usi() == usi)` と同じ規約)．
/// 非合法手・未知表記は `None`．
pub fn move_from_csa(board: &Board, token: &str) -> Option<Move> {
    let turn = board.turn();
    generate_legal_moves(&mut board.clone())
        .into_iter()
        .find(|m| move_to_csa(turn, *m) == token)
}

/// 対局の持ち時間規定 (`BEGIN Time` 階層)．全てミリ秒に正規化して保持する．
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimeRule {
    /// 単位時間 (`Time_Unit`) のミリ秒換算．指し手に付く `,T<n>` の換算に使う．
    pub unit_ms: u64,
    /// 通算持ち時間 (`Total_Time`)．`None` = 無制限．
    pub total_ms: Option<u64>,
    /// 秒読み (`Byoyomi`)．
    pub byoyomi_ms: u64,
    /// フィッシャー加算 (`Increment`)．
    pub increment_ms: u64,
    /// 遅延時間 (`Delay`)．消費時間の控除はサーバが行うので，こちらは記録のみ．
    pub delay_ms: u64,
    /// 1 手の最少消費時間 (`Least_Time_Per_Move`)．
    pub least_per_move_ms: u64,
}

impl Default for TimeRule {
    fn default() -> Self {
        Self {
            unit_ms: 1000,
            total_ms: None,
            byoyomi_ms: 0,
            increment_ms: 0,
            delay_ms: 0,
            least_per_move_ms: 0,
        }
    }
}

/// 対局条件 (`BEGIN Game_Summary` ～ `END Game_Summary`)．
#[derive(Clone, Debug, PartialEq)]
pub struct GameSummary {
    /// 対局 ID (`Game_ID`)．
    pub game_id: String,
    /// 先手の対局者名 (`Name+`)．
    pub name_black: String,
    /// 後手の対局者名 (`Name-`)．
    pub name_white: String,
    /// 自分の手番 (`Your_Turn`)．
    pub my_color: Color,
    /// 開始 (再開) 直後に指す手番 (`To_Move`)．
    pub to_move: Color,
    /// 打ち切り手数 (`Max_Moves`)．`None` = 無制限．
    pub max_moves: Option<u32>,
    /// 持ち時間規定．
    pub time: TimeRule,
    /// 基準局面の SFEN (`BEGIN Position` の盤面部)．
    pub start_sfen: String,
    /// 基準局面から再開局面までの指し手 (USI 表記)．
    pub init_moves: Vec<String>,
    /// 再開局面までに消費された時間 [先手, 後手] (`Time_Unit` 単位)．
    ///
    /// 再開局面から始まる対局で残り持ち時間を出すために使う．
    pub init_consumed_units: [u64; 2],
}

/// `Game_Summary` を行単位で組み立てるパーサ．
///
/// `BEGIN Game_Summary` から `END Game_Summary` までを受け取り，完成した
/// [`GameSummary`] を返す．
#[derive(Debug, Default)]
pub struct GameSummaryBuilder {
    fields: Vec<(String, String)>,
    time_fields: Vec<(String, String)>,
    position_lines: Vec<String>,
    section: Section,
}

/// `Game_Summary` の入れ子階層．
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
enum Section {
    /// 一般情報 (最上位)．
    #[default]
    Root,
    /// `BEGIN Time` / `Time+` / `Time-` の内部．
    Time,
    /// `BEGIN Position` の内部．
    Position,
}

impl GameSummaryBuilder {
    /// 1 行を投入する．`END Game_Summary` で完成品を返す．
    pub fn push(&mut self, line: &str) -> Result<Option<GameSummary>, String> {
        let trimmed = line.trim_end_matches(['\r', '\n']);
        let t = trimmed.trim();
        if t.is_empty() {
            return Ok(None);
        }
        if let Some(name) = t.strip_prefix("BEGIN ") {
            match name.trim() {
                "Time" | "Time+" | "Time-" => self.section = Section::Time,
                "Position" => self.section = Section::Position,
                _ => {}
            }
            return Ok(None);
        }
        if let Some(name) = t.strip_prefix("END ") {
            match name.trim() {
                "Game_Summary" => return self.finish().map(Some),
                _ => self.section = Section::Root,
            }
            return Ok(None);
        }
        match self.section {
            // 局面階層は行の形をそのまま保つ (CSA 棋譜パーサへ渡すため)
            Section::Position => self.position_lines.push(trimmed.to_string()),
            Section::Time | Section::Root => {
                if let Some((k, v)) = t.split_once(':') {
                    let entry = (k.trim().to_string(), v.trim().to_string());
                    if self.section == Section::Time {
                        self.time_fields.push(entry);
                    } else {
                        self.fields.push(entry);
                    }
                }
            }
        }
        Ok(None)
    }

    fn field(&self, key: &str) -> Option<&str> {
        self.fields
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    fn time_field(&self, key: &str) -> Option<&str> {
        self.time_fields
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    fn finish(&mut self) -> Result<GameSummary, String> {
        let my_color = self
            .field("Your_Turn")
            .and_then(parse_color)
            .ok_or_else(|| "Game_Summary に Your_Turn がない".to_string())?;
        let to_move = self
            .field("To_Move")
            .and_then(parse_color)
            .unwrap_or(Color::Black);
        let unit_ms = self
            .time_field("Time_Unit")
            .map_or(1000, parse_time_unit_ms);
        let scale = |key: &str| -> Option<u64> {
            self.time_field(key)
                .and_then(|v| v.parse::<u64>().ok())
                .map(|n| n.saturating_mul(unit_ms))
        };
        let time = TimeRule {
            unit_ms,
            total_ms: scale("Total_Time"),
            byoyomi_ms: scale("Byoyomi").unwrap_or(0),
            increment_ms: scale("Increment").unwrap_or(0),
            delay_ms: scale("Delay").unwrap_or(0),
            least_per_move_ms: scale("Least_Time_Per_Move").unwrap_or(0),
        };
        let (start_sfen, init_moves, init_consumed_units) =
            parse_position_block(&self.position_lines)?;
        Ok(GameSummary {
            game_id: self.field("Game_ID").unwrap_or_default().to_string(),
            name_black: self.field("Name+").unwrap_or_default().to_string(),
            name_white: self.field("Name-").unwrap_or_default().to_string(),
            my_color,
            to_move,
            max_moves: self.field("Max_Moves").and_then(|v| v.parse().ok()),
            time,
            start_sfen,
            init_moves,
            init_consumed_units,
        })
    }
}

/// `+` / `-` を手番へ．
fn parse_color(s: &str) -> Option<Color> {
    match s.trim() {
        "+" => Some(Color::Black),
        "-" => Some(Color::White),
        _ => None,
    }
}

/// `Time_Unit` (`1sec` / `min` / `msec` 等) をミリ秒へ．
fn parse_time_unit_ms(s: &str) -> u64 {
    let s = s.trim();
    let digits: String = s.chars().take_while(|c| c.is_ascii_digit()).collect();
    let count: u64 = digits.parse().unwrap_or(1);
    let unit = s[digits.len()..].trim();
    let base = match unit {
        "min" => 60_000,
        "msec" => 1,
        // 既定は秒 ("sec"，未知の単位も秒に倒す)
        _ => 1000,
    };
    count.saturating_mul(base)
}

/// `BEGIN Position` 階層を (基準局面 SFEN, USI 指し手列, 手番別の消費時間) へ．
///
/// 盤面部は CSA 棋譜として組み立て直して [`parse_csa_str`] に渡す．指し手行は
/// 合法手照合で解決する (`move_from_csa`)．
///
/// 消費時間は再開局面での残り時間を出すために手番別に合計する (単位時間のまま)．
/// これを引かないと再開時に持ち時間を過大評価して時間切れ負けし得る．
type PositionBlock = (String, Vec<String>, [u64; 2]);

fn parse_position_block(lines: &[String]) -> Result<PositionBlock, String> {
    let mut board_lines: Vec<&str> = Vec::new();
    let mut move_lines: Vec<&str> = Vec::new();
    for line in lines {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        // 手番行 ("+" / "-") は盤面部の締め，指し手行は 2 文字以上
        let is_move = (t.starts_with('+') || t.starts_with('-')) && t.len() > 1;
        if is_move {
            move_lines.push(t);
        } else {
            board_lines.push(t);
        }
    }
    // 手番行が盤面部の最後に来ていない棋譜は CSA として不正
    let record = format!("V2.2\n{}\n", board_lines.join("\n"));
    let parsed = parse_csa_str(&record).map_err(|e| format!("局面のパースに失敗: {e}"))?;
    let start_sfen = parsed.sfen;

    let mut board = Board::empty();
    board
        .set_sfen(&start_sfen)
        .map_err(|e| format!("局面 SFEN の適用に失敗: {e:?}"))?;
    let mut moves = Vec::with_capacity(move_lines.len());
    let mut consumed = [0u64; 2];
    for line in move_lines {
        // "+2726FU,T12" → 指し手部と消費時間に分ける
        let (token, spent) = split_consumed(line);
        let mv = move_from_csa(&board, &token)
            .ok_or_else(|| format!("再開局面の指し手が非合法: {token}"))?;
        consumed[usize::from(board.turn() == Color::White)] += spent;
        moves.push(mv.to_usi());
        board.do_move(mv);
    }
    Ok((start_sfen, moves, consumed))
}

/// サーバ → クライアントのメッセージ．
#[derive(Clone, Debug, PartialEq)]
pub enum ServerMessage {
    /// `LOGIN:<name> OK`．
    LoginOk(String),
    /// `LOGIN:incorrect`．
    LoginIncorrect,
    /// `LOGOUT:completed`．
    LogoutCompleted,
    /// `BEGIN Game_Summary` (以降の行は [`GameSummaryBuilder`] へ)．
    GameSummaryBegin,
    /// `START:<GameID>`．
    Start(String),
    /// `REJECT:<GameID> by <name>`．
    Reject {
        /// 対局 ID．
        game_id: String,
        /// 拒否したクライアント名．
        by: String,
    },
    /// 指し手 (`+7776FU,T12`)．
    Move {
        /// 指し手表記 (`+7776FU`)．
        token: String,
        /// サーバが計測した消費時間 (`Time_Unit` 単位．無しは 0)．
        consumed: u64,
    },
    /// 特殊手のエコー (`%TORYO,T4` / `%KACHI,T8` / `%CHUDAN`)．
    Special {
        /// 表記 (`%TORYO`)．
        token: String,
        /// 消費時間．
        consumed: u64,
    },
    /// `#` で始まる状態通知 (`#WIN` / `#SENNICHITE` 等)．
    Status(String),
    /// 空行 (サーバからの keep-alive 応答)．
    KeepAlive,
    /// 解釈しない行 (拡張モードの `##[LOGIN]` 等)．
    Other(String),
}

/// サーバからの 1 行をパースする．
pub fn parse_server_line(line: &str) -> ServerMessage {
    let t = line.trim_end_matches(['\r', '\n']);
    let s = t.trim();
    if s.is_empty() {
        return ServerMessage::KeepAlive;
    }
    if let Some(rest) = s.strip_prefix("LOGIN:") {
        if rest == "incorrect" {
            return ServerMessage::LoginIncorrect;
        }
        let name = rest.strip_suffix(" OK").unwrap_or(rest);
        return ServerMessage::LoginOk(name.to_string());
    }
    if s == "LOGOUT:completed" {
        return ServerMessage::LogoutCompleted;
    }
    if s == "BEGIN Game_Summary" {
        return ServerMessage::GameSummaryBegin;
    }
    if let Some(rest) = s.strip_prefix("START:") {
        return ServerMessage::Start(rest.trim().to_string());
    }
    if let Some(rest) = s.strip_prefix("REJECT:") {
        let (game_id, by) = match rest.split_once(" by ") {
            Some((g, b)) => (g.trim().to_string(), b.trim().to_string()),
            None => (rest.trim().to_string(), String::new()),
        };
        return ServerMessage::Reject { game_id, by };
    }
    // `##` 始まりは shogi-server 拡張モード (x1) の通知で CSA の状態通知ではない
    if s.starts_with("##") {
        return ServerMessage::Other(s.to_string());
    }
    if s.starts_with('#') {
        return ServerMessage::Status(s.to_string());
    }
    if s.starts_with('%') {
        let (token, consumed) = split_consumed(s);
        return ServerMessage::Special { token, consumed };
    }
    if (s.starts_with('+') || s.starts_with('-')) && s.len() >= 7 {
        let (token, consumed) = split_consumed(s);
        return ServerMessage::Move { token, consumed };
    }
    ServerMessage::Other(s.to_string())
}

/// `"<手>,T<n>"` を (手, 消費時間) へ分解する．
fn split_consumed(s: &str) -> (String, u64) {
    match s.split_once(',') {
        Some((head, rest)) => {
            let n = rest
                .trim()
                .strip_prefix('T')
                .and_then(|v| v.trim().parse::<u64>().ok())
                .unwrap_or(0);
            (head.trim().to_string(), n)
        }
        None => (s.trim().to_string(), 0),
    }
}

/// 状態通知が対局の終了を表すか (勝敗行 = 2 行目)．
pub fn is_terminal_status(status: &str) -> bool {
    matches!(status, "#WIN" | "#LOSE" | "#DRAW" | "#CENSORED" | "#CHUDAN")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::STARTPOS_SFEN;

    fn startpos() -> Board {
        let mut b = Board::empty();
        b.set_sfen(STARTPOS_SFEN).unwrap();
        b
    }

    #[test]
    fn encodes_normal_move() {
        let board = startpos();
        let mv = move_from_csa(&board, "+7776FU").expect("7776FU は合法");
        assert_eq!(mv.to_usi(), "7g7f");
        assert_eq!(move_to_csa(Color::Black, mv), "+7776FU");
    }

    #[test]
    fn encodes_promotion_with_promoted_piece_code() {
        // 8 八角が 2 二へ成る局面 (角道が開いた状態から)
        let mut board = startpos();
        for usi in ["7g7f", "3c3d"] {
            let mv = generate_legal_moves(&mut board.clone())
                .into_iter()
                .find(|m| m.to_usi() == usi)
                .unwrap();
            board.do_move(mv);
        }
        let mv = move_from_csa(&board, "+8822UM").expect("8822UM は合法");
        assert_eq!(mv.to_usi(), "8h2b+");
        // 成った後の駒種 (UM) で表記される
        assert_eq!(move_to_csa(Color::Black, mv), "+8822UM");
    }

    #[test]
    fn encodes_drop_with_zero_origin() {
        // 後手が 5 五へ歩を打つ局面を作る (先手が歩を取らせる形は面倒なので
        // 持ち駒つき SFEN を直接与える)
        let mut board = Board::empty();
        // SFEN の持ち駒は小文字 = 後手 (後手番に歩を 1 枚持たせる)
        board.set_sfen("4k4/9/9/9/9/9/9/9/4K4 w p 1").unwrap();
        let mv = move_from_csa(&board, "-0055FU").expect("0055FU は合法");
        assert_eq!(mv.to_usi(), "P*5e");
        assert_eq!(move_to_csa(Color::White, mv), "-0055FU");
    }

    #[test]
    fn rejects_illegal_move_token() {
        let board = startpos();
        assert!(move_from_csa(&board, "+7775FU").is_none());
        assert!(move_from_csa(&board, "garbage").is_none());
    }

    #[test]
    fn parses_server_lines() {
        assert_eq!(
            parse_server_line("LOGIN:maou_test OK"),
            ServerMessage::LoginOk("maou_test".to_string())
        );
        assert_eq!(
            parse_server_line("LOGIN:incorrect"),
            ServerMessage::LoginIncorrect
        );
        assert_eq!(
            parse_server_line("START:20260728-floodgate-1"),
            ServerMessage::Start("20260728-floodgate-1".to_string())
        );
        assert_eq!(
            parse_server_line("REJECT:game1 by opponent"),
            ServerMessage::Reject {
                game_id: "game1".to_string(),
                by: "opponent".to_string()
            }
        );
        assert_eq!(
            parse_server_line("+7776FU,T12"),
            ServerMessage::Move {
                token: "+7776FU".to_string(),
                consumed: 12
            }
        );
        assert_eq!(
            parse_server_line("%TORYO,T4"),
            ServerMessage::Special {
                token: "%TORYO".to_string(),
                consumed: 4
            }
        );
        assert_eq!(
            parse_server_line("#SENNICHITE"),
            ServerMessage::Status("#SENNICHITE".to_string())
        );
        assert_eq!(parse_server_line(""), ServerMessage::KeepAlive);
        assert_eq!(
            parse_server_line("##[LOGIN] +OK x1"),
            ServerMessage::Other("##[LOGIN] +OK x1".to_string())
        );
    }

    #[test]
    fn parses_time_units() {
        assert_eq!(parse_time_unit_ms("1sec"), 1000);
        assert_eq!(parse_time_unit_ms("sec"), 1000);
        assert_eq!(parse_time_unit_ms("1min"), 60_000);
        assert_eq!(parse_time_unit_ms("msec"), 1);
        assert_eq!(parse_time_unit_ms("5sec"), 5000);
    }

    /// floodgate が実際に送ってくる形の Game_Summary．
    const FLOODGATE_SUMMARY: &str = "\
Protocol_Version:1.2
Protocol_Mode:Server
Format:Shogi 1.0
Declaration:Jishogi 1.1
Game_ID:20260728-floodgate-300-10F-1
Name+:maou_test
Name-:opponent
Your_Turn:+
Rematch_On_Draw:NO
To_Move:+
Max_Moves:512
BEGIN Time
Time_Unit:1sec
Total_Time:300
Increment:10
Least_Time_Per_Move:0
END Time
BEGIN Position
P1-KY-KE-GI-KI-OU-KI-GI-KE-KY
P2 * -HI *  *  *  *  * -KA *
P3-FU-FU-FU-FU-FU-FU-FU-FU-FU
P4 *  *  *  *  *  *  *  *  *
P5 *  *  *  *  *  *  *  *  *
P6 *  *  *  *  *  *  *  *  *
P7+FU+FU+FU+FU+FU+FU+FU+FU+FU
P8 * +KA *  *  *  *  * +HI *
P9+KY+KE+GI+KI+OU+KI+GI+KE+KY
P+
P-
+
END Position
END Game_Summary";

    fn build(text: &str) -> GameSummary {
        let mut b = GameSummaryBuilder::default();
        for line in text.lines() {
            if let Some(summary) = b.push(line).expect("パース成功") {
                return summary;
            }
        }
        panic!("END Game_Summary に到達しなかった");
    }

    #[test]
    fn parses_floodgate_game_summary() {
        let s = build(FLOODGATE_SUMMARY);
        assert_eq!(s.game_id, "20260728-floodgate-300-10F-1");
        assert_eq!(s.name_black, "maou_test");
        assert_eq!(s.name_white, "opponent");
        assert_eq!(s.my_color, Color::Black);
        assert_eq!(s.to_move, Color::Black);
        assert_eq!(s.max_moves, Some(512));
        // 300 秒 + 10 秒加算 (floodgate-300-10F)
        assert_eq!(s.time.total_ms, Some(300_000));
        assert_eq!(s.time.increment_ms, 10_000);
        assert_eq!(s.time.byoyomi_ms, 0);
        assert_eq!(s.start_sfen, STARTPOS_SFEN);
        assert!(s.init_moves.is_empty());
    }

    #[test]
    fn parses_resumed_position_with_moves() {
        let text = FLOODGATE_SUMMARY.replace(
            "+\nEND Position",
            "+\n+2726FU,T12\n-3334FU,T6\nEND Position",
        );
        let s = build(&text);
        assert_eq!(s.init_moves, vec!["2g2f".to_string(), "3c3d".to_string()]);
        assert_eq!(s.start_sfen, STARTPOS_SFEN);
        // 再開局面の残り持ち時間を出すため手番別に消費時間を積む
        assert_eq!(s.init_consumed_units, [12, 6]);
    }

    #[test]
    fn parses_byoyomi_rule() {
        let text = FLOODGATE_SUMMARY.replace("Increment:10", "Byoyomi:10");
        let s = build(&text);
        assert_eq!(s.time.byoyomi_ms, 10_000);
        assert_eq!(s.time.increment_ms, 0);
    }

    #[test]
    fn your_turn_gote() {
        let text = FLOODGATE_SUMMARY.replace("Your_Turn:+", "Your_Turn:-");
        let s = build(&text);
        assert_eq!(s.my_color, Color::White);
    }
}
