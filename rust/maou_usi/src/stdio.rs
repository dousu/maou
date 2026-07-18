//! 標準入出力の USI transport．
//!
//! - **reader スレッド**: stdin を行単位で読み，パースして dispatcher へ渡す．
//!   `go`/`stop`/`quit` の行を読んだ瞬間に停止フラグを更新する — フラグの
//!   変化が行の到着順と一致するため race がなく，dispatcher が探索で
//!   ブロックしていても `stop`/`quit` が即応する (設計 doc §4/§7)．
//! - **dispatcher (呼び出しスレッド)**: コマンドを順に [`Agent::handle`] へ
//!   渡し，応答を stdout へ書く (行バッファ + flush)．stdout はプロトコル
//!   専用 — ログ・診断は stderr へ．

use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;

use crate::agent::{Agent, EngineConfig, SearchBackend};
use crate::backend::MaouSearchBackend;
use crate::protocol::{parse_line, serialize, EngineCommand, GuiCommand};

/// reader: 入力を行単位でパースして送る．`go`/`stop`/`quit` は行順で
/// 停止フラグへ反映する．EOF (GUI 死亡) では Quit を合成して終了する．
fn read_commands<R: BufRead>(input: R, stop: &Arc<AtomicBool>, tx: &Sender<GuiCommand>) {
    for line in input.lines() {
        let Ok(line) = line else { break };
        let Some(cmd) = parse_line(&line) else {
            continue;
        };
        match &cmd {
            GuiCommand::Go(_) => stop.store(false, Ordering::Release),
            GuiCommand::Stop | GuiCommand::Quit => stop.store(true, Ordering::Release),
            _ => {}
        }
        let quit = matches!(cmd, GuiCommand::Quit);
        if tx.send(cmd).is_err() || quit {
            return;
        }
    }
    // EOF: 実行中の探索を止め，dispatcher を終了させる
    stop.store(true, Ordering::Release);
    let _ = tx.send(GuiCommand::Quit);
}

/// dispatcher: コマンドを順に処理して応答を書く．`Quit` で終了する．
fn run_loop<B, F, W>(
    rx: &Receiver<GuiCommand>,
    out: &mut W,
    agent: &mut Agent<B, F>,
) -> Result<(), String>
where
    B: SearchBackend,
    F: Fn(&EngineConfig) -> Result<B, String>,
    W: Write,
{
    loop {
        let Ok(cmd) = rx.recv() else {
            return Ok(()); // reader 消滅 (EOF 相当)
        };
        if matches!(cmd, GuiCommand::Quit) {
            return Ok(());
        }
        // Go は探索中に info を随時流すため handle_go_stream で直接ストリームする．
        // それ以外は応答をまとめて書く．
        if let GuiCommand::Go(params) = cmd {
            let res = {
                let mut emit = |c: EngineCommand| {
                    let _ = writeln!(out, "{}", serialize(&c));
                    let _ = out.flush();
                };
                agent.handle_go_stream(&params, &mut emit)
            };
            if let Err(e) = res {
                return fatal(out, &e);
            }
        } else {
            match agent.handle(cmd) {
                Ok(responses) => {
                    for r in &responses {
                        writeln!(out, "{}", serialize(r)).map_err(|e| e.to_string())?;
                    }
                    if !responses.is_empty() {
                        out.flush().map_err(|e| e.to_string())?;
                    }
                }
                Err(e) => return fatal(out, &e),
            }
        }
    }
}

/// 致命的エラー (モデルロード失敗・不正局面)．詳細は stderr，GUI へは理由を
/// ASCII 化した `info string` で通知して終了する (subprocess の stderr が
/// 見えない環境 — Colab 等 — でも原因が stdout 側から分かるように)．
fn fatal<W: Write>(out: &mut W, e: &str) -> Result<(), String> {
    eprintln!("maou usi fatal: {e}");
    let _ = writeln!(out, "info string ERROR: {}", sanitize_ascii(e));
    let _ = out.flush();
    Err(e.to_string())
}

/// `info string` 用にエラーメッセージを 1 行の ASCII へ丸める
/// (GUI のエンコーディング差と行指向プロトコルの保護)．
fn sanitize_ascii(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii() && !c.is_ascii_control() {
                c
            } else if c.is_whitespace() {
                ' '
            } else {
                '?'
            }
        })
        .collect()
}

/// 標準入出力で USI エンジンを実行する (quit / EOF まで戻らない)．
///
/// 呼び出しスレッドが dispatcher になる．stdin を専有するため，プロセスに
/// つき 1 回だけ呼ぶこと．
pub fn run_stdio(config: EngineConfig) -> Result<(), String> {
    let mut agent = Agent::new(config, MaouSearchBackend::build);
    let stop = agent.stop_handle();
    let (tx, rx) = std::sync::mpsc::channel::<GuiCommand>();
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        read_commands(stdin.lock(), &stop, &tx);
    });
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    run_loop(&rx, &mut out, &mut agent)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::STARTPOS_SFEN;
    use maou_search::build_board_and_history;
    use maou_shogi::movegen::generate_legal_moves;
    use std::io::Cursor;

    /// USI セッション台本を実バックエンド (mock 評価器) で流す E2E．
    fn run_session(script: &str) -> Vec<String> {
        let config = EngineConfig {
            root_dfpn: Some(false),
            leaf_mate: Some(false),
            node_capacity: Some(1 << 14),
            ..EngineConfig::default()
        };
        let mut agent = Agent::new(config, MaouSearchBackend::build);
        let stop = agent.stop_handle();
        let (tx, rx) = std::sync::mpsc::channel::<GuiCommand>();
        let input = Cursor::new(script.to_string());
        let reader = std::thread::spawn(move || {
            read_commands(input, &stop, &tx);
        });
        let mut out: Vec<u8> = Vec::new();
        run_loop(&rx, &mut out, &mut agent).expect("セッションは正常終了する");
        reader.join().expect("reader 正常終了");
        String::from_utf8(out)
            .expect("プロトコル出力は UTF-8")
            .lines()
            .map(str::to_string)
            .collect()
    }

    #[test]
    fn test_full_usi_session() {
        let lines = run_session(
            "usi\nisready\nusinewgame\nposition startpos moves 7g7f\ngo btime 0 wtime 0 byoyomi 1200\nquit\n",
        );
        assert!(lines.iter().any(|l| l.starts_with("id name maou")));
        assert!(lines.iter().any(|l| l == "usiok"));
        assert!(lines.iter().any(|l| l == "readyok"));
        // mock 明示 (ModelPath 未指定)
        assert!(lines
            .iter()
            .any(|l| l.starts_with("info string") && l.contains("mock")));
        // bestmove は現局面の合法手
        let bestmove = lines
            .iter()
            .find(|l| l.starts_with("bestmove "))
            .expect("bestmove が出力される");
        let mv = bestmove.split_whitespace().nth(1).expect("指し手がある");
        let (board, _) =
            build_board_and_history(STARTPOS_SFEN, &["7g7f".to_string()]).expect("正当");
        let legal: Vec<String> = generate_legal_moves(&mut board.clone())
            .into_iter()
            .map(|m| m.to_usi())
            .collect();
        assert!(legal.contains(&mv.to_string()), "{mv} は合法手であるべき");
        // info サマリの pv は行末尾 (score 等より後)
        let info = lines
            .iter()
            .find(|l| l.starts_with("info ") && l.contains(" pv "))
            .expect("探索サマリ info がある");
        let pv_pos = info.find(" pv ").expect("pv がある");
        assert!(!info[pv_pos..].contains(" score "), "pv は末尾に置く");
    }

    #[test]
    fn test_sanitize_ascii() {
        assert_eq!(sanitize_ascii("plain error 123"), "plain error 123");
        // 非 ASCII は '?'，改行は空白 (1 行の info string を壊さない)
        assert_eq!(sanitize_ascii("不正な SFEN\nrow2"), "??? SFEN row2");
    }

    #[test]
    fn test_eof_terminates_cleanly() {
        // quit なしで入力が尽きても (GUI 死亡相当) ループは終了する
        let lines = run_session("usi\n");
        assert!(lines.iter().any(|l| l == "usiok"));
    }

    #[test]
    fn test_stop_flag_follows_line_order() {
        // reader が go → stop の行順でフラグを false → true にする
        let stop = Arc::new(AtomicBool::new(false));
        let (tx, rx) = std::sync::mpsc::channel::<GuiCommand>();
        let input = Cursor::new("go infinite\nstop\n".to_string());
        read_commands(input, &stop, &tx);
        // 最終状態は stop = true (stop 行が最後)
        assert!(stop.load(Ordering::Acquire));
        let mut received = Vec::new();
        while let Ok(c) = rx.try_recv() {
            received.push(c);
        }
        assert!(matches!(received[0], GuiCommand::Go(_)));
        assert!(matches!(received[1], GuiCommand::Stop));
    }
}
