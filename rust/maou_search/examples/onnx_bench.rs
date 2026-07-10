//! ONNX evaluator による NPS ベンチマーク (`onnx` feature 必須)．
//!
//! North-star 計測 (GPU 実測 NPS) の実行口．CUDA を使う場合は
//! `onnx-cuda` feature でビルドして `--cuda` を付ける．
//!
//! 使用例 (CPU):
//!
//! ```bash
//! cargo run --release -p maou_search --features onnx --example onnx_bench -- \
//!     --model model.onnx --threads 2 --batch 32 --time-ms 10000
//! ```
//!
//! 使用例 (GPU / Colab):
//!
//! ```bash
//! cargo run --release -p maou_search --features onnx-cuda --example onnx_bench -- \
//!     --model model.onnx --cuda --threads 2 --batch 128 --time-ms 30000
//! ```
//!
//! 注意: NPS 計測は必ず `--release` で行うこと．

use maou_search::onnx::{OnnxEvaluator, OnnxOptions};
use maou_search::{EvalItem, Evaluator, SearchLimits, SearchOptions, Searcher};
use maou_shogi::board::Board;
use maou_shogi::movegen::generate_legal_moves;

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
            "usage: onnx_bench --model PATH [--sfen SFEN] [--threads N] [--batch N] \
             [--playouts N] [--time-ms N] [--capacity N] [--cpuct F] [--fpu F] \
             [--keep-ratio F] [--no-gc] [--ort-threads N] [--cuda] [--tensorrt] \
             [--trt-cache DIR] [--pad N] [--root-dfpn]"
        );
        return;
    }

    let Some(model): Option<String> = arg_value(&args, "--model") else {
        eprintln!("--model PATH は必須 (--help 参照)");
        std::process::exit(2);
    };
    let sfen = args
        .iter()
        .position(|a| a == "--sfen")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| STARTPOS.to_string());

    let options = SearchOptions {
        threads: arg_value(&args, "--threads").unwrap_or(2),
        batch_size: arg_value(&args, "--batch").unwrap_or(32),
        c_puct: arg_value(&args, "--cpuct").unwrap_or(1.5),
        fpu: arg_value(&args, "--fpu").unwrap_or(0.5),
        node_capacity: arg_value(&args, "--capacity").unwrap_or(1 << 21),
        gc_enabled: !args.iter().any(|a| a == "--no-gc"),
        gc_keep_ratio: arg_value(&args, "--keep-ratio").unwrap_or(0.5),
        // ルート並行 dfpn を有効にすると，MCTS と CPU/メモリを競合する dfpn
        // スレッドが NPS に与える影響を計測できる (実配備と同条件)．ノード予算は
        // SearchOptions::default() の root_dfpn_nodes (1<<20 ≈ 冒頭 1 秒の詰み
        // スキャン) を使う — 詰みのない局面では予算消化後に dfpn は終了する
        root_dfpn: args.iter().any(|a| a == "--root-dfpn"),
        // NPS ベンチは leaf-mate を off にして純粋な MCTS(+root-dfpn) を測る
        // (production default は leaf-mate on だが計測ノイズになるため)．
        leaf_mate: false,
        ..SearchOptions::default()
    };
    let limits = SearchLimits {
        max_playouts: arg_value(&args, "--playouts"),
        time_ms: arg_value(&args, "--time-ms").or(Some(10000)),
    };
    let use_tensorrt = args.iter().any(|a| a == "--tensorrt");
    let onnx_options = OnnxOptions {
        intra_threads: arg_value(&args, "--ort-threads").unwrap_or(1),
        use_cuda: args.iter().any(|a| a == "--cuda"),
        use_tensorrt,
        trt_engine_cache_dir: arg_value(&args, "--trt-cache"),
        // TensorRT は shape 固定が前提のため，未指定なら探索バッチサイズに合わせる
        pad_to: arg_value(&args, "--pad").or(if use_tensorrt {
            Some(options.batch_size)
        } else {
            None
        }),
    };

    #[cfg(debug_assertions)]
    eprintln!("warning: debug ビルドでの計測値は参考値 (--release を使うこと)");

    println!("=== maou_search NPS bench (OnnxEvaluator) ===");
    println!("model: {model}");
    println!("sfen: {sfen}");
    println!(
        "threads={} batch={} cpuct={} fpu={} capacity={} gc={} keep_ratio={} \
         ort_threads={} cuda={} tensorrt={} pad={:?} root_dfpn={}",
        options.threads,
        options.batch_size,
        options.c_puct,
        options.fpu,
        options.node_capacity,
        options.gc_enabled,
        options.gc_keep_ratio,
        onnx_options.intra_threads,
        onnx_options.use_cuda,
        onnx_options.use_tensorrt,
        onnx_options.pad_to,
        options.root_dfpn
    );

    let evaluator = match OnnxEvaluator::from_file(&model, &onnx_options) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("ONNX モデルのロードに失敗: {e}");
            std::process::exit(1);
        }
    };

    // TensorRT のエンジンビルド/ロード (shape ごとに初回数十秒〜数分) を
    // 計測時間の外で済ませる warmup 評価
    {
        let warmup_start = std::time::Instant::now();
        let mut board = Board::empty();
        if board.set_sfen(&sfen).is_ok() {
            let moves = generate_legal_moves(&mut board);
            if !moves.is_empty() {
                let _ = evaluator.evaluate_batch(&[EvalItem { board, moves }]);
            }
        }
        println!(
            "warmup_ms={} (エンジンビルド/ロード込み，計測時間外)",
            warmup_start.elapsed().as_millis()
        );
    }

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
