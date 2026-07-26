//! 自己対局 A/B ハーネス — 設定レバーの棋力効果を対戦で測る (設計 §12 未決 4/6)．
//!
//! プレイヤー A (レバー on) vs B (レバー off) を色交代 + ペア開局で対戦させ，
//! A の勝率と Wilson 95% 信頼区間を報告する．
//!
//! ```bash
//! cargo run --release -p maou_usi --example selfplay_ab --features onnx -- \
//!   --mode subtree --model model.onnx --games 30 --playouts 64 \
//!   --random-plies 8 --seed 1 --out /tmp/ab.jsonl
//! ```
//!
//! - `--mode subtree`: A = subtree 再利用 on / B = off
//! - `--mode maxmoves`: A = MaxMovesToDraw の in-search 終端化 on / B = off
//!   (両者とも driver の最大手数で引き分けになる点は同一．知識の有無だけが違う)
//! - `--mode budget`: A = `--playouts` / B = `--playouts-b` (既定 A の 1/8)．
//!   **ハーネスの健全性確認** — 予算の多い側が勝たなければ，この driver で
//!   棋力差を測ること自体が成立していない (レバーの A/B より前に通すべき検証)
//! - `--mode horizon`: 持ち時間モードで TimeStrategy の想定残り手数を A/B
//!   (`--horizon` vs `--horizon-b`)．設計 §12 未決 1 の調整用．時計は
//!   `--clock-ms` / `--byoyomi-ms` / `--inc-ms`．壁時計を測るので parallel=1

use maou_usi::selfplay::{run_selfplay, GameOutcome, SelfplayConfig};
use maou_usi::EngineConfig;

fn arg_value<T: std::str::FromStr>(args: &[String], key: &str) -> Option<T> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
}

/// Wilson score interval (95%) — 少数対局でも壊れない勝率の区間推定．
fn wilson_95(score: f64, n: f64) -> (f64, f64) {
    if n <= 0.0 {
        return (0.0, 1.0);
    }
    let z = 1.96_f64;
    let p = score / n;
    let denom = 1.0 + z * z / n;
    let center = (p + z * z / (2.0 * n)) / denom;
    let half = (z / denom) * (p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt();
    ((center - half).max(0.0), (center + half).min(1.0))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode: String = arg_value(&args, "--mode").unwrap_or_else(|| "subtree".to_string());
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

    // 予算差モードのみ B の playout を変える (既定は A の 1/8)
    let playouts_b_opt: Option<u64> = if mode == "budget" {
        Some(arg_value(&args, "--playouts-b").unwrap_or((playouts / 8).max(1)))
    } else {
        None
    };

    let (engine_a, engine_b, sync) = match mode.as_str() {
        // A = subtree 再利用 on (現行 default) / B = off
        "subtree" => {
            let a = base.clone();
            let mut b = base.clone();
            b.subtree_reuse = false;
            (a, b, true)
        }
        // A = in-search 終端化 on / B = off (max_moves_to_draw を per-side 管理
        // するため sync は切る．driver の終局判定は両者共通で max_moves)
        "maxmoves" => {
            let mut a = base.clone();
            a.max_moves_to_draw = max_moves;
            let mut b = base.clone();
            b.max_moves_to_draw = 0;
            (a, b, false)
        }
        // A/B の設定は同一で探索予算だけを変える (ハーネスの健全性確認)
        "budget" => (base.clone(), base.clone(), true),
        // 持ち時間モードで TimeStrategy の想定残り手数だけを変える (未決 1)
        "horizon" => {
            let mut a = base.clone();
            a.time.horizon_moves = horizon_a;
            let mut b = base.clone();
            b.time.horizon_moves = horizon_b;
            (a, b, true)
        }
        other => {
            eprintln!("unknown --mode {other} (subtree | maxmoves | budget | horizon)");
            std::process::exit(2);
        }
    };
    let clock = (mode == "horizon").then_some(maou_usi::selfplay::ClockSetting {
        initial_ms: clock_ms,
        byoyomi_ms,
        inc_ms,
    });

    let config = SelfplayConfig {
        engine: engine_a,
        engine_b: Some(engine_b),
        alternate_colors: true,
        sync_max_moves_to_draw: sync,
        sfen: None,
        games,
        parallel,
        // 持ち時間モードでは playout 予算を渡さない (時計から算出させる)
        playouts: clock.is_none().then_some(playouts),
        movetime_ms: None,
        playouts_b: playouts_b_opt,
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

    // 集計: A 視点の W/D/L (勝ち 1 点，引き分け 0.5 点)
    let mut wins = 0u32;
    let mut draws = 0u32;
    let mut losses = 0u32;
    let mut reasons: std::collections::BTreeMap<&'static str, u32> = Default::default();
    let mut total_plies = 0usize;
    let mut total_ms = 0u64;
    let mut total_playouts = 0u64;
    let mut reused_moves = 0u32;
    let mut carried_visits = 0u64;
    for o in &outcomes {
        use maou_shogi::types::Color;
        let a_is_winner = match (o.winner, o.black_is_a) {
            (None, _) => None,
            (Some(Color::Black), black_a) => Some(black_a),
            (Some(Color::White), black_a) => Some(!black_a),
        };
        match a_is_winner {
            Some(true) => wins += 1,
            Some(false) => losses += 1,
            None => draws += 1,
        }
        *reasons.entry(o.reason.as_str()).or_default() += 1;
        total_plies += o.moves.len();
        total_ms += o.elapsed_ms;
        total_playouts += o.playouts;
        reused_moves += o.reused_moves;
        carried_visits += o.carried_visits;
    }
    let n = outcomes.len() as f64;
    let score = f64::from(wins) + 0.5 * f64::from(draws);
    let (lo, hi) = wilson_95(score, n);

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

    println!("mode: {mode} (A = lever on, B = off)");
    match clock {
        Some(c) => println!(
            "games: {games}, clock: {}ms + {}ms/move (byoyomi {}ms), horizon: A {horizon_a} / B {horizon_b}, random plies: {random_plies}, max moves: {max_moves}, seed: {seed}",
            c.initial_ms, c.inc_ms, c.byoyomi_ms,
        ),
        None => println!(
            "games: {games}, playouts/move: A {playouts} / B {}, random plies: {random_plies}, max moves: {max_moves}, seed: {seed}",
            playouts_b_opt.unwrap_or(playouts),
        ),
    }
    println!("A result: {wins}W {draws}D {losses}L");
    println!(
        "A score: {:.1}/{} = {:.1}% (Wilson 95% CI [{:.1}%, {:.1}%])",
        score,
        n,
        100.0 * score / n,
        100.0 * lo,
        100.0 * hi,
    );
    println!(
        "reasons: {}",
        reasons
            .iter()
            .map(|(k, v)| format!("{k} {v}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "plies: {} total, wall: {:.1}s summed",
        total_plies,
        total_ms as f64 / 1000.0,
    );
    // 時間配分の診断 (持ち時間モード): 終局時の残り時間 = 使い残し．
    // 勝率より少ないサンプルで「配分が保守的すぎないか」が分かる
    if let Some(c) = clock {
        let mut left = [0u64; 2]; // [A, B]
        let mut n = 0u32;
        for o in &outcomes {
            if let Some(rem) = o.remaining_ms {
                // rem は [先手, 後手]．A/B 視点へ写す
                let (a, b) = if o.black_is_a {
                    (rem[0], rem[1])
                } else {
                    (rem[1], rem[0])
                };
                left[0] += a;
                left[1] += b;
                n += 1;
            }
        }
        if n > 0 {
            let avg = |v: u64| v as f64 / f64::from(n);
            println!(
                "time left at end (avg): A {:.1}s / B {:.1}s (initial {:.1}s + {}ms/move)",
                avg(left[0]) / 1000.0,
                avg(left[1]) / 1000.0,
                c.initial_ms as f64 / 1000.0,
                c.inc_ms,
            );
        }
    }
    // 機構レベルの計測: 勝率が有意でなくても「レバーが実対局で発火したか」
    // は決着する (subtree 再利用は A 側のみ on なので carry は全て A の分)
    println!(
        "subtree reuse engagement: {} moves carried {} visits ({:.1}% of {} total playouts)",
        reused_moves,
        carried_visits,
        100.0 * carried_visits as f64 / total_playouts.max(1) as f64,
        total_playouts,
    );
}
