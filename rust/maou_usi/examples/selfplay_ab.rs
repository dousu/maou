//! 自己対局 A/B ハーネス (Rust 単体版) — 設定レバーの棋力効果を対戦で測る．
//!
//! **配布 wheel を使う環境 (Colab など) では `maou selfplay --ab-mode ...` を
//! 使う** (同じ [`maou_usi::ab`] を呼ぶので数値の定義は一致する)．この example
//! は Python 拡張をビルドせずに Rust だけで回したいときの入口．
//!
//! ```bash
//! cargo run --release -p maou_usi --example selfplay_ab --features onnx -- \
//!   --mode subtree --model model.onnx --games 30 --playouts 64 \
//!   --random-plies 8 --seed 1 --out /tmp/ab.jsonl
//! ```
//!
//! - `--mode subtree`: A = subtree 再利用 on / B = off
//! - `--mode maxmoves`: A = MaxMovesToDraw の in-search 終端化 on / B = off
//! - `--mode budget`: A = `--playouts` / B = `--playouts-b` (既定 A の 1/8)．
//!   **ハーネスの健全性確認**
//! - `--mode horizon`: 持ち時間モードで TimeStrategy の想定残り手数を A/B
//!   (`--horizon` vs `--horizon-b`)．時計は `--clock-ms` / `--byoyomi-ms` /
//!   `--inc-ms`．壁時計を測るので parallel=1
//! - `--mode batch`: A = `--batch-size` / B = `--batch-size-b` (既定 A の 4 倍)．
//!   **持ち時間モードで回すこと** — 固定 playout 予算では速度差が棋力差に
//!   ならない

use maou_usi::ab::{build_ab, summarize, AbMode, AbOptions, SummaryOptions};
use maou_usi::selfplay::{run_selfplay, ClockSetting, GameOutcome, SelfplayConfig};
use maou_usi::EngineConfig;

fn arg_value<T: std::str::FromStr>(args: &[String], key: &str) -> Option<T> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode_name: String = arg_value(&args, "--mode").unwrap_or_else(|| "subtree".to_string());
    let Some(mode) = AbMode::parse(&mode_name) else {
        eprintln!("unknown --mode {mode_name} (subtree | maxmoves | budget | horizon)");
        std::process::exit(2);
    };
    let model: Option<String> = arg_value(&args, "--model");
    let games: u32 = arg_value(&args, "--games").unwrap_or(20);
    let parallel: usize = arg_value(&args, "--parallel").unwrap_or(1);
    let playouts: u64 = arg_value(&args, "--playouts").unwrap_or(64);
    let max_moves: u32 = arg_value(&args, "--max-moves").unwrap_or(256);
    let random_plies: u32 = arg_value(&args, "--random-plies").unwrap_or(8);
    let seed: u64 = arg_value(&args, "--seed").unwrap_or(1);
    let node_capacity: u32 = arg_value(&args, "--node-capacity").unwrap_or(1 << 16);
    let resign_value: u32 = arg_value(&args, "--resign-value").unwrap_or(0);
    let resign_consecutive: u32 = arg_value(&args, "--resign-consecutive").unwrap_or(4);
    let draw_black: u32 = arg_value(&args, "--draw-value-black").unwrap_or(500);
    let draw_white: u32 = arg_value(&args, "--draw-value-white").unwrap_or(500);
    let out: Option<String> = arg_value(&args, "--out");

    // 持ち時間モード (--mode horizon) の時計設定
    let clock_ms: u64 = arg_value(&args, "--clock-ms").unwrap_or(30_000);
    let byoyomi_ms: u64 = arg_value(&args, "--byoyomi-ms").unwrap_or(0);
    let inc_ms: u64 = arg_value(&args, "--inc-ms").unwrap_or(500);
    let horizon_a: u64 = arg_value(&args, "--horizon").unwrap_or(40);
    let horizon_b: u64 = arg_value(&args, "--horizon-b").unwrap_or(25);

    // 共通ベース: 詰み探索 off (両者同条件の純 MCTS で速く回す)
    let mut base = EngineConfig {
        model_path: model,
        node_capacity: Some(node_capacity),
        resign_value,
        resign_consecutive,
        draw_value_black: draw_black,
        draw_value_white: draw_white,
        root_dfpn: Some(false),
        leaf_mate: Some(false),
        ..EngineConfig::default()
    };
    // 自己対局に伝送遅延はない (持ち時間モードで margin を引かない)
    base.time.network_delay_ms = 0;

    let setup = build_ab(
        &base,
        &AbOptions {
            mode,
            playouts_b: arg_value(&args, "--playouts-b"),
            max_moves,
            horizon_a,
            horizon_b,
            batch_size_b: arg_value(&args, "--batch-size-b"),
        },
        Some(playouts),
    );
    let clock = mode.needs_clock().then_some(ClockSetting {
        initial_ms: clock_ms,
        byoyomi_ms,
        inc_ms,
    });

    let config = SelfplayConfig {
        engine: setup.engine_a,
        engine_b: Some(setup.engine_b),
        alternate_colors: true,
        sync_max_moves_to_draw: setup.sync_max_moves_to_draw,
        sfen: None,
        games,
        parallel,
        // 持ち時間モードでは playout 予算を渡さない (時計から算出させる)
        playouts: clock.is_none().then_some(playouts),
        movetime_ms: None,
        playouts_b: setup.playouts_b,
        movetime_ms_b: None,
        clock,
        max_moves,
        opening_random_plies: random_plies,
        seed,
    };

    let total = games;
    let progress = |o: &GameOutcome| {
        let a_color = if o.black_is_a { "A=black" } else { "A=white" };
        let winner = match o.winner {
            Some(c) => format!("{c:?} wins"),
            None => "draw".to_string(),
        };
        eprintln!(
            "[ab] game {}/{}: {} ({}) by {} — {} plies, {:.1}s",
            o.game_index + 1,
            total,
            winner,
            a_color,
            o.reason.as_str(),
            o.moves.len(),
            o.elapsed_ms as f64 / 1000.0,
        );
    };

    let outcomes = match run_selfplay(&config, Some(&progress)) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("selfplay failed: {e}");
            std::process::exit(1);
        }
    };

    if let Some(path) = out {
        use std::io::Write;
        let mut f = std::fs::File::create(&path).expect("JSONL 出力先を作成できる");
        for o in &outcomes {
            let winner = match o.winner {
                Some(maou_shogi::types::Color::Black) => "\"black\"",
                Some(maou_shogi::types::Color::White) => "\"white\"",
                None => "null",
            };
            let moves: Vec<String> = o.moves.iter().map(|m| format!("\"{m}\"")).collect();
            writeln!(
                f,
                "{{\"game_index\":{},\"black_player\":\"{}\",\"winner\":{},\"reason\":\"{}\",\"plies\":{},\"playouts\":{},\"elapsed_ms\":{},\"moves\":[{}]}}",
                o.game_index,
                if o.black_is_a { "a" } else { "b" },
                winner,
                o.reason.as_str(),
                o.moves.len(),
                o.playouts,
                o.elapsed_ms,
                moves.join(","),
            )
            .expect("JSONL 書き込み成功");
        }
        eprintln!("[ab] wrote {} records to {path}", outcomes.len());
    }

    println!("mode: {mode_name} (A = lever on, B = off)");
    match clock {
        Some(c) => println!(
            "games: {games}, clock: {}ms + {}ms/move (byoyomi {}ms), horizon: A {horizon_a} / B {horizon_b}, random plies: {random_plies}, max moves: {max_moves}, seed: {seed}",
            c.initial_ms, c.inc_ms, c.byoyomi_ms,
        ),
        None => println!(
            "games: {games}, playouts/move: A {playouts} / B {}, random plies: {random_plies}, max moves: {max_moves}, seed: {seed}",
            setup.playouts_b.unwrap_or(playouts),
        ),
    }
    // 勝率・ペア差分・機構の発火量をまとめて出す (集計は maou_usi::ab)
    print!(
        "{}",
        summarize(
            &outcomes,
            SummaryOptions {
                ab: true,
                paired: config.alternate_colors,
            }
        )
    );
}
