//! NPS ベンチマーク (MockEvaluator)．
//!
//! 探索コア + バッチ収集機構のスループットを計測する．評価器は決定論的 mock
//! なので，ここでの NPS は「NN 推論を除いた探索側スループットの上限」を示す．
//!
//! 使用例:
//!
//! ```bash
//! cargo run --release -p maou_search --example nps_bench -- \
//!     --threads 4 --batch 16 --time-ms 5000
//! ```
//!
//! 注意: NPS 計測は必ず `--release` で行うこと (dev profile は約 6 倍遅い)．

use maou_search::{MockEvaluator, SearchLimits, SearchOptions, Searcher};

const STARTPOS: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

/// コマンドライン引数から値を取り出す簡易パーサ．
fn arg_value<T: std::str::FromStr>(args: &[String], key: &str) -> Option<T> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        eprintln!(
            "usage: nps_bench [--sfen SFEN] [--threads N] [--batch N] \
             [--playouts N] [--time-ms N] [--capacity N] [--cpuct F] \
             [--fpu F] [--seed N] [--keep-ratio F] [--no-gc]"
        );
        return;
    }

    let sfen = args
        .iter()
        .position(|a| a == "--sfen")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| STARTPOS.to_string());

    let options = SearchOptions {
        threads: arg_value(&args, "--threads").unwrap_or(1),
        batch_size: arg_value(&args, "--batch").unwrap_or(8),
        c_puct: arg_value(&args, "--cpuct").unwrap_or(1.5),
        fpu: arg_value(&args, "--fpu").unwrap_or(0.5),
        node_capacity: arg_value(&args, "--capacity").unwrap_or(1 << 21),
        gc_enabled: !args.iter().any(|a| a == "--no-gc"),
        gc_keep_ratio: arg_value(&args, "--keep-ratio").unwrap_or(0.5),
        ..SearchOptions::default()
    };
    let limits = SearchLimits {
        max_playouts: arg_value(&args, "--playouts"),
        time_ms: arg_value(&args, "--time-ms").or(Some(5000)),
    };
    let seed: u64 = arg_value(&args, "--seed").unwrap_or(42);

    #[cfg(debug_assertions)]
    eprintln!("warning: debug ビルドでの計測値は参考値 (--release を使うこと)");

    println!("=== maou_search NPS bench (MockEvaluator) ===");
    println!("sfen: {sfen}");
    println!(
        "threads={} batch={} cpuct={} fpu={} capacity={} gc={} keep_ratio={} seed={seed}",
        options.threads,
        options.batch_size,
        options.c_puct,
        options.fpu,
        options.node_capacity,
        options.gc_enabled,
        options.gc_keep_ratio
    );

    let evaluator = MockEvaluator::new(seed);
    let searcher = Searcher::new(&evaluator, options.clone());
    let result = match searcher.search_sfen(&sfen, &limits) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("SFEN parse error: {e}");
            std::process::exit(1);
        }
    };

    let s = &result.stats;
    println!(
        "playouts={} elapsed={}ms NPS={:.0}",
        s.playouts, s.elapsed_ms, s.nps
    );
    println!(
        "collisions={} ({:.2}% of playouts) eval_batches={} avg_batch={:.2} (fill {:.1}%)",
        s.collisions,
        100.0 * s.collisions as f64 / (s.playouts.max(1)) as f64,
        s.eval_batches,
        s.avg_batch,
        100.0 * s.avg_batch / options.batch_size as f64
    );
    println!(
        "max_depth={} repetitions={} proven_nodes={} nodes_used={} leaked_nodes={} gc_runs={} gc_freed_nodes={} stop={:?}",
        s.max_depth,
        s.repetitions,
        s.proven_nodes,
        s.nodes_used,
        s.leaked_nodes,
        s.gc_runs,
        s.gc_freed_nodes,
        result.stop
    );
    match result.best_move {
        Some(m) => println!("bestmove={} winrate={:.4}", m.to_usi(), result.winrate),
        None => println!("bestmove=none (root terminal)"),
    }
    let pv: Vec<String> = result.pv.iter().map(|m| m.to_usi()).collect();
    println!("pv={}", pv.join(" "));

    let mut children = result.root_children.clone();
    children.sort_by_key(|c| std::cmp::Reverse(c.visits));
    println!("top children (visits desc):");
    for c in children.iter().take(8) {
        println!(
            "  {:<6} visits={:<8} q={:.4} prior={:.4}",
            c.mv.to_usi(),
            c.visits,
            c.q,
            c.prior
        );
    }
}
