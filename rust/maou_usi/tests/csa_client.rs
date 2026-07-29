//! CSA クライアントの E2E テスト (モックサーバ相手)．
//!
//! floodgate の運用に合わせ「1 局ごとに接続が切れる」構成を再現し，
//! **2 局連続**が再接続込みで回ることを検証する．実サーバに繋ぐ前に
//! プロトコル適合をここで確認する．

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;

use maou_usi::csa::{run_csa, CsaConfig};
use maou_usi::EngineConfig;

/// 対局条件 (floodgate と同形．持ち時間はテストが速く終わる値にする)．
fn game_summary(game_id: &str) -> String {
    format!(
        "BEGIN Game_Summary
Protocol_Version:1.2
Protocol_Mode:Server
Format:Shogi 1.0
Declaration:Jishogi 1.1
Game_ID:{game_id}
Name+:maou_test
Name-:mock_opponent
Your_Turn:+
Rematch_On_Draw:NO
To_Move:+
Max_Moves:512
BEGIN Time
Time_Unit:1sec
Total_Time:10
Byoyomi:1
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
END Game_Summary"
    )
}

/// 内容のある行を 1 つ読む (keep-alive の空行は読み飛ばす)．
fn read_line(reader: &mut BufReader<TcpStream>) -> String {
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).expect("読み取り") == 0 {
            return String::new();
        }
        let trimmed = line.trim().to_string();
        if !trimmed.is_empty() {
            return trimmed;
        }
    }
}

/// 1 接続 = 1 対局を演じる．終局後は接続を閉じる (floodgate と同じ挙動)．
fn serve_one_game(stream: TcpStream, game_id: &str) {
    let mut writer = stream.try_clone().expect("ソケット複製");
    let mut reader = BufReader::new(stream);

    let login = read_line(&mut reader);
    assert!(login.starts_with("LOGIN "), "LOGIN が来る: {login}");
    writeln!(writer, "LOGIN:maou_test OK").unwrap();
    writeln!(writer, "{}", game_summary(game_id)).unwrap();
    writer.flush().unwrap();

    let agree = read_line(&mut reader);
    assert!(agree.starts_with("AGREE"), "AGREE が来る: {agree}");
    writeln!(writer, "START:{game_id}").unwrap();
    writer.flush().unwrap();

    // 先手 (クライアント) の 1 手目 → サーバが消費時間を付けて返す
    let first = read_line(&mut reader);
    assert!(first.starts_with('+'), "先手の指し手が来る: {first}");
    writeln!(writer, "{first},T1").unwrap();
    // 後手の応手 (先手の初手では 3d に到達できないので常に合法)
    writeln!(writer, "-3334FU,T1").unwrap();
    writer.flush().unwrap();

    let second = read_line(&mut reader);
    assert!(second.starts_with('+'), "先手の 2 手目が来る: {second}");
    writeln!(writer, "{second},T1").unwrap();
    // 後手の投了で終局 (指し手・理由・勝敗の 3 行)
    writeln!(writer, "%TORYO,T1").unwrap();
    writeln!(writer, "#RESIGN").unwrap();
    writeln!(writer, "#WIN").unwrap();
    writer.flush().unwrap();
}

/// テスト用のエンジン設定 (mock 評価器 + メモリ節約)．
fn test_engine() -> EngineConfig {
    EngineConfig {
        model_path: None,
        // 8GB DevContainer で走らせるため確保を小さくする
        node_capacity: Some(1 << 14),
        root_dfpn: Some(false),
        leaf_mate: Some(false),
        ..EngineConfig::default()
    }
}

#[test]
fn plays_two_consecutive_games_reconnecting_each_time() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().unwrap().port();
    let server = thread::spawn(move || {
        for (index, conn) in listener.incoming().take(2).enumerate() {
            serve_one_game(conn.expect("accept"), &format!("mockgame-{index}"));
        }
    });

    let summary = run_csa(CsaConfig {
        engine: test_engine(),
        host: "127.0.0.1".to_string(),
        port,
        login_name: "maou_test".to_string(),
        password: "mock-300-10F,trip".to_string(),
        games: 2,
        keep_alive_sec: 60,
        connect_timeout_sec: 10,
        game_wait_sec: 30,
        reconnect_wait_sec: 0,
        verbose: false,
    })
    .expect("2 局が完了する");

    assert_eq!(summary.games.len(), 2, "2 局とも完了する");
    for (index, game) in summary.games.iter().enumerate() {
        assert_eq!(game.game_id, format!("mockgame-{index}"));
        assert_eq!(game.result, "win");
        assert_eq!(game.reason, "#RESIGN");
        assert_eq!(game.my_color, "black");
        assert_eq!(game.opponent, "mock_opponent");
        // 先手 2 手 + 後手 1 手
        assert_eq!(game.moves, 3);
    }
    server.join().expect("サーバスレッド");
}

/// ログイン拒否は再試行せず即座に失敗する (設定の誤りを黙って握り潰さない)．
#[test]
fn login_rejection_is_fatal() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().unwrap().port();
    let server = thread::spawn(move || {
        let (stream, _) = listener.accept().expect("accept");
        let mut writer = stream.try_clone().expect("ソケット複製");
        let mut reader = BufReader::new(stream);
        let _ = read_line(&mut reader);
        writeln!(writer, "LOGIN:incorrect").unwrap();
        writer.flush().unwrap();
    });

    let err = run_csa(CsaConfig {
        engine: test_engine(),
        host: "127.0.0.1".to_string(),
        port,
        login_name: "maou_test".to_string(),
        password: "bad".to_string(),
        games: 2,
        keep_alive_sec: 60,
        connect_timeout_sec: 5,
        game_wait_sec: 5,
        reconnect_wait_sec: 0,
        verbose: false,
    })
    .expect_err("ログイン拒否は失敗する");
    assert!(err.contains("LOGIN:incorrect"), "拒否理由が伝わる: {err}");
    server.join().expect("サーバスレッド");
}
