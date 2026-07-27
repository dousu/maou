//! maou_usi (USI 対局エージェント) の PyO3 バインディング．

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use maou_usi::ab::{build_ab, summarize, AbMode, AbOptions, RunSummary, SummaryOptions};
use maou_usi::selfplay::{ClockSetting, GameOutcome, SelfplayConfig};
use maou_usi::{EngineConfig, TimeStrategyConfig};

/// 評価器・探索・戦略の共通引数を [`EngineConfig`] へ反映する
/// (`run_usi` / `run_selfplay` で共有する単一実装)．
#[allow(clippy::too_many_arguments)]
fn apply_common_engine_args(
    config: &mut EngineConfig,
    model_path: Option<String>,
    threads: Option<usize>,
    batch_size: Option<usize>,
    node_capacity: Option<u32>,
    use_cuda: Option<bool>,
    use_tensorrt: Option<bool>,
    trt_engine_cache_dir: Option<String>,
    draw_value_black: Option<u32>,
    draw_value_white: Option<u32>,
    resign_value: Option<u32>,
    resign_consecutive: Option<u32>,
    opening_script: Option<String>,
    root_dfpn: Option<bool>,
    root_dfpn_nodes: Option<u64>,
    root_dfpn_depth: Option<u32>,
    leaf_mate: Option<bool>,
    leaf_mate_nodes: Option<u64>,
    leaf_mate_threads: Option<usize>,
) {
    config.model_path = model_path.filter(|p| !p.is_empty());
    if let Some(v) = threads {
        config.threads = v;
    }
    if let Some(v) = batch_size {
        config.batch_size = v;
    }
    config.node_capacity = node_capacity;
    config.use_cuda = use_cuda.unwrap_or(false);
    config.use_tensorrt = use_tensorrt.unwrap_or(false);
    config.trt_cache_dir = trt_engine_cache_dir;
    if let Some(v) = draw_value_black {
        config.draw_value_black = v;
    }
    if let Some(v) = draw_value_white {
        config.draw_value_white = v;
    }
    if let Some(v) = resign_value {
        config.resign_value = v;
    }
    if let Some(v) = resign_consecutive {
        config.resign_consecutive = v.max(1);
    }
    config.opening_script = opening_script.filter(|s| !s.trim().is_empty());
    config.root_dfpn = root_dfpn;
    config.root_dfpn_nodes = root_dfpn_nodes;
    config.root_dfpn_depth = root_dfpn_depth;
    config.leaf_mate = leaf_mate;
    config.leaf_mate_nodes = leaf_mate_nodes;
    config.leaf_mate_threads = leaf_mate_threads;
}

/// USI エンジンを標準入出力で実行する (`quit`/EOF まで戻らない)．
///
/// GIL を解放して Rust の USI ループ (reader スレッド + dispatcher) が
/// stdin/stdout を専有する — Python 側は起動前に logging を stderr へ向けて
/// おくこと (stdout はプロトコル専用)．
///
/// # 引数 (全て keyword-only．CLI フラグ = 初期値で，USI `setoption` が上書きする)
///
/// - `engine_name` (str, optional): `id name` に出す名前 (デフォルト
///   "maou <maou_usi crate version>")．
/// - `engine_author` (str, optional): `id author` に出す作者名．
/// - `model_path` (str, optional): ONNX モデルパス．未指定なら mock 評価器
///   (開発検証用 — `isready` 時に `info string` で明示される)．
/// - `threads` / `batch_size` (int, optional): 探索スレッド数/評価バッチサイズ．
/// - `node_capacity` (int, optional): ノードプール容量．
/// - `use_cuda` / `use_tensorrt` (bool, optional): Execution Provider
///   (対応 feature 付き wheel が必要)．
/// - `trt_engine_cache_dir` (str, optional): TensorRT エンジンキャッシュ保存先．
/// - `network_delay_ms` (int, optional): 通信マージン (デフォルト 1000)．
/// - `min_think_ms` (int, optional): 最低思考時間 (デフォルト 100)．
/// - `draw_value_black` / `draw_value_white` (int, optional): 先手/後手の引き分け
///   価値 (千分率，デフォルト 500．電竜戦 0.4/0.6 勝 = 400/600)．
/// - `resign_value` (int, optional): 投了する root 勝率 (千分率，デフォルト
///   0 = 投了しない)．`resign_consecutive` (int, optional): 投了に必要な連続手数．
/// - `keep_alive_ms` (int, optional): `isready` 応答待ち中に空行を送る間隔
///   (デフォルト 0 = 送らない)．TensorRT の初回ビルドが GUI のタイムアウトを
///   超える場合の生存通知．
/// - `max_moves_to_draw` (int, optional): 引き分け最大手数 (デフォルト 0 = 無効)．
///   > 0 なら探索内でもリミット以降を引き分け終端として扱う．
/// - `usi_ponder` (bool, optional): ponder (先読み) を有効にするか (デフォルト
///   true)．true のとき `USI_Ponder` option を宣言し `bestmove` に予想相手手を
///   付ける (`setoption name USI_Ponder` で上書き可)．
/// - `opening_script` (str, optional): 強制序盤手順 (USI 指し手の空白区切り，
///   例 "5i5h 5a5b 5h5i 5b5a")．経路が prefix 一致する間は探索なしで即指し．
/// - `root_dfpn` / `root_dfpn_nodes` / `root_dfpn_depth`: ルート並行 dfpn．
/// - `leaf_mate` / `leaf_mate_nodes` / `leaf_mate_threads`: leaf-mate．
///
/// # エラー
///
/// モデルロード失敗・不正局面などの致命的エラーは `RuntimeError`．
#[pyfunction]
#[pyo3(signature = (*, engine_name=None, engine_author=None, model_path=None, threads=None, batch_size=None, node_capacity=None, use_cuda=None, use_tensorrt=None, trt_engine_cache_dir=None, network_delay_ms=None, min_think_ms=None, keep_alive_ms=None, draw_value_black=None, draw_value_white=None, resign_value=None, resign_consecutive=None, max_moves_to_draw=None, usi_ponder=None, opening_script=None, root_dfpn=None, root_dfpn_nodes=None, root_dfpn_depth=None, leaf_mate=None, leaf_mate_nodes=None, leaf_mate_threads=None))]
#[allow(clippy::too_many_arguments)]
fn run_usi(
    py: Python<'_>,
    engine_name: Option<String>,
    engine_author: Option<String>,
    model_path: Option<String>,
    threads: Option<usize>,
    batch_size: Option<usize>,
    node_capacity: Option<u32>,
    use_cuda: Option<bool>,
    use_tensorrt: Option<bool>,
    trt_engine_cache_dir: Option<String>,
    network_delay_ms: Option<u64>,
    min_think_ms: Option<u64>,
    keep_alive_ms: Option<u64>,
    draw_value_black: Option<u32>,
    draw_value_white: Option<u32>,
    resign_value: Option<u32>,
    resign_consecutive: Option<u32>,
    max_moves_to_draw: Option<u32>,
    usi_ponder: Option<bool>,
    opening_script: Option<String>,
    root_dfpn: Option<bool>,
    root_dfpn_nodes: Option<u64>,
    root_dfpn_depth: Option<u32>,
    leaf_mate: Option<bool>,
    leaf_mate_nodes: Option<u64>,
    leaf_mate_threads: Option<usize>,
) -> PyResult<()> {
    let mut config = EngineConfig::default();
    if let Some(v) = engine_name {
        config.engine_name = v;
    }
    if let Some(v) = engine_author {
        config.engine_author = v;
    }
    let time_defaults = TimeStrategyConfig::default();
    config.time = TimeStrategyConfig {
        network_delay_ms: network_delay_ms.unwrap_or(time_defaults.network_delay_ms),
        min_think_ms: min_think_ms.unwrap_or(time_defaults.min_think_ms),
        horizon_moves: time_defaults.horizon_moves,
    };
    if let Some(v) = keep_alive_ms {
        config.keep_alive_ms = v;
    }
    if let Some(v) = max_moves_to_draw {
        config.max_moves_to_draw = v;
    }
    if let Some(v) = usi_ponder {
        config.usi_ponder = v;
    }
    apply_common_engine_args(
        &mut config,
        model_path,
        threads,
        batch_size,
        node_capacity,
        use_cuda,
        use_tensorrt,
        trt_engine_cache_dir,
        draw_value_black,
        draw_value_white,
        resign_value,
        resign_consecutive,
        opening_script,
        root_dfpn,
        root_dfpn_nodes,
        root_dfpn_depth,
        leaf_mate,
        leaf_mate_nodes,
        leaf_mate_threads,
    );

    py.detach(move || maou_usi::run_stdio(config))
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

/// [`GameOutcome`] → Python dict．
fn outcome_to_dict<'py>(py: Python<'py>, o: &GameOutcome) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("game_index", o.game_index)?;
    d.set_item("sfen", &o.sfen)?;
    d.set_item("moves", PyList::new(py, &o.moves)?)?;
    d.set_item(
        "winner",
        o.winner.map(|c| match c {
            maou_shogi::types::Color::Black => "black",
            maou_shogi::types::Color::White => "white",
        }),
    )?;
    d.set_item("reason", o.reason.as_str())?;
    d.set_item("black_player", if o.black_is_a { "a" } else { "b" })?;
    d.set_item("plies", o.moves.len())?;
    d.set_item("playouts", o.playouts)?;
    d.set_item("terminal_backprops", o.terminal_backprops)?;
    d.set_item("reused_moves", o.reused_moves)?;
    d.set_item("carried_visits", o.carried_visits)?;
    d.set_item("elapsed_ms", o.elapsed_ms)?;
    // 終局時の残り持ち時間 [先手, 後手]．持ち時間モード以外は None
    d.set_item("remaining_ms", o.remaining_ms.map(|r| r.to_vec()))?;
    Ok(d)
}

/// [`RunSummary`] → Python dict (`ab` は A/B 対戦時のみ dict，他は None)．
fn summary_to_dict<'py>(py: Python<'py>, s: &RunSummary) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("games", s.games)?;
    d.set_item("black_wins", s.black_wins)?;
    d.set_item("white_wins", s.white_wins)?;
    d.set_item("draws", s.draws)?;
    let reasons = PyDict::new(py);
    for (name, count) in &s.reasons {
        reasons.set_item(name, count)?;
    }
    d.set_item("reasons", reasons)?;
    d.set_item("total_plies", s.total_plies)?;
    d.set_item("total_playouts", s.total_playouts)?;
    d.set_item("total_terminal_backprops", s.total_terminal_backprops)?;
    d.set_item("total_game_ms", s.total_game_ms)?;
    d.set_item("reused_moves", s.reused_moves)?;
    d.set_item("carried_visits", s.carried_visits)?;
    match &s.ab {
        None => d.set_item("ab", py.None())?,
        Some(ab) => {
            let a = PyDict::new(py);
            a.set_item("wins", ab.wins)?;
            a.set_item("draws", ab.draws)?;
            a.set_item("losses", ab.losses)?;
            a.set_item("score", ab.score)?;
            a.set_item("score_rate", ab.score_rate)?;
            a.set_item("ci_lo", ab.ci_lo)?;
            a.set_item("ci_hi", ab.ci_hi)?;
            a.set_item("elo", ab.elo)?;
            a.set_item("elo_lo", ab.elo_lo)?;
            a.set_item("elo_hi", ab.elo_hi)?;
            a.set_item("pairs", ab.pairs)?;
            a.set_item("pairs_a_ahead", ab.pairs_a_ahead)?;
            a.set_item("pairs_tied", ab.pairs_tied)?;
            a.set_item("pairs_b_ahead", ab.pairs_b_ahead)?;
            a.set_item("paired_mean", ab.paired_mean)?;
            a.set_item("paired_se", ab.paired_se)?;
            a.set_item("paired_t", ab.paired_t)?;
            a.set_item("time_left_a_ms", ab.time_left_a_ms)?;
            a.set_item("time_left_b_ms", ab.time_left_b_ms)?;
            d.set_item("ab", a)?
        }
    }
    Ok(d)
}

/// in-process 自己対局を実行し，対局結果 (dict) のリストを返す．
///
/// 1 対局 = エージェント 2 個 (先後独立の探索木) を stdio なしで直接駆動する．
/// 評価器はプロセス内 1 個を全対局で共有する (モデルロード/warmup 1 回)．
/// 終局判定 (宣言/千日手/最大手数/投了) は USI 対局と同一実装．
///
/// # 引数 (全て keyword-only)
///
/// - `model_path` (str, optional): ONNX モデルパス (未指定 = mock 評価器)．
/// - `games` (int, optional): 対局数 (デフォルト 1)．
/// - `parallel` (int, optional): 同時対局数 (デフォルト 1)．
/// - `playouts` (int, optional): 1 手の playout 予算 (`movetime_ms` 未指定なら
///   デフォルト 800)．
/// - `movetime_ms` (int, optional): 1 手の思考時間ミリ秒 (`playouts` と排他)．
/// - `max_moves` (int, optional): 最大手数 (デフォルト 512 = 電竜戦)．到達で
///   引き分け (宣言成立は勝ち)．
/// - `sfen` (str, optional): 基準局面 (デフォルト平手)．
/// - `opening_random_plies` (int, optional): 序盤ランダム手数 (対局多様化，
///   デフォルト 0)．`seed` (int, optional): 乱数シード (デフォルト 0)．
/// - `verbose` (bool, optional): 対局ごとの進捗を stderr へ出す (デフォルト
///   true)．
/// - `min_think_ms` (int, optional): 最低思考時間 (持ち時間モードの下限)．
/// - 残りは `run_usi` と同名の評価器・探索・戦略引数
///   (`opening_script` は両側エージェントに適用される)．
///
/// # A/B 対戦 (レバーの効果計測．設計 §12)
///
/// - `ab_mode` (str, optional): `"subtree"` / `"maxmoves"` / `"budget"` /
///   `"spin"` (空回りを予算から外す; 固定 playout 予算専用) /
///   `"horizon"`．指定するとプレイヤー B (レバー off 側) が作られ，A 視点の
///   勝率統計が summary に載る．レバー以外の設定は A と同一になる．
/// - `playouts_b` (int, optional): `"budget"` の B 側予算 (未指定 = A の 1/8)．
/// - `horizon_moves` / `horizon_moves_b` (int, optional): `"horizon"` の
///   A/B 想定残り手数 (未指定 = 40 / 25)．
/// - `alternate_colors` (bool, optional): 対局ごとに先後を入れ替え，ペア
///   (2n, 2n+1) で同じ開局系列を共有する (`ab_mode` 指定時のデフォルト true)．
/// - `clock_ms` / `byoyomi_ms` / `inc_ms` (int, optional): 持ち時間モード
///   (`clock_ms` > 0 で有効)．`playouts`/`movetime_ms` と排他，`parallel=1`
///   限定 (壁時計で消費を測るため)．`"horizon"` では必須．
///
/// # 返り値
///
/// `{"records": [...], "summary": {...}}`．`records` は対局 index 順の dict
/// `{game_index, sfen, moves, winner ("black"/"white"/None), reason,
/// black_player, plies, playouts, reused_moves, carried_visits, elapsed_ms,
/// remaining_ms}`，`summary` は集計 (`ab` キーは A/B 対戦時のみ dict)．
///
/// # エラー
///
/// 設定不正・モデルロード失敗・対局中の内部エラーは `RuntimeError`，
/// 未知の `ab_mode` は `ValueError`．
#[pyfunction]
#[pyo3(signature = (*, model_path=None, games=None, parallel=None, playouts=None, movetime_ms=None, max_moves=None, sfen=None, opening_random_plies=None, seed=None, verbose=None, threads=None, batch_size=None, node_capacity=None, use_cuda=None, use_tensorrt=None, trt_engine_cache_dir=None, draw_value_black=None, draw_value_white=None, resign_value=None, resign_consecutive=None, opening_script=None, root_dfpn=None, root_dfpn_nodes=None, root_dfpn_depth=None, leaf_mate=None, leaf_mate_nodes=None, leaf_mate_threads=None, spin_budget_relief=None, skip_proven_children=None, min_think_ms=None, ab_mode=None, playouts_b=None, horizon_moves=None, horizon_moves_b=None, alternate_colors=None, clock_ms=None, byoyomi_ms=None, inc_ms=None))]
#[allow(clippy::too_many_arguments)]
fn run_selfplay(
    py: Python<'_>,
    model_path: Option<String>,
    games: Option<u32>,
    parallel: Option<usize>,
    playouts: Option<u64>,
    movetime_ms: Option<u64>,
    max_moves: Option<u32>,
    sfen: Option<String>,
    opening_random_plies: Option<u32>,
    seed: Option<u64>,
    verbose: Option<bool>,
    threads: Option<usize>,
    batch_size: Option<usize>,
    node_capacity: Option<u32>,
    use_cuda: Option<bool>,
    use_tensorrt: Option<bool>,
    trt_engine_cache_dir: Option<String>,
    draw_value_black: Option<u32>,
    draw_value_white: Option<u32>,
    resign_value: Option<u32>,
    resign_consecutive: Option<u32>,
    opening_script: Option<String>,
    root_dfpn: Option<bool>,
    root_dfpn_nodes: Option<u64>,
    root_dfpn_depth: Option<u32>,
    leaf_mate: Option<bool>,
    leaf_mate_nodes: Option<u64>,
    leaf_mate_threads: Option<usize>,
    spin_budget_relief: Option<bool>,
    skip_proven_children: Option<bool>,
    min_think_ms: Option<u64>,
    ab_mode: Option<String>,
    playouts_b: Option<u64>,
    horizon_moves: Option<u64>,
    horizon_moves_b: Option<u64>,
    alternate_colors: Option<bool>,
    clock_ms: Option<u64>,
    byoyomi_ms: Option<u64>,
    inc_ms: Option<u64>,
) -> PyResult<Py<PyDict>> {
    let mut engine = EngineConfig::default();
    apply_common_engine_args(
        &mut engine,
        model_path,
        threads,
        batch_size,
        node_capacity,
        use_cuda,
        use_tensorrt,
        trt_engine_cache_dir,
        draw_value_black,
        draw_value_white,
        resign_value,
        resign_consecutive,
        opening_script,
        root_dfpn,
        root_dfpn_nodes,
        root_dfpn_depth,
        leaf_mate,
        leaf_mate_nodes,
        leaf_mate_threads,
    );
    if let Some(v) = spin_budget_relief {
        engine.spin_budget_relief = v;
    }
    if let Some(v) = skip_proven_children {
        engine.skip_proven_children = v;
    }
    // 自己対局に伝送遅延はない (movetime をそのまま思考時間にする)
    engine.time.network_delay_ms = 0;
    if let Some(v) = min_think_ms {
        engine.time.min_think_ms = v;
    }

    // 持ち時間モード (clock_ms > 0)．playouts/movetime との排他は
    // maou_usi::selfplay::run_selfplay が検証する
    let clock = match clock_ms {
        Some(v) if v > 0 => Some(ClockSetting {
            initial_ms: v,
            byoyomi_ms: byoyomi_ms.unwrap_or(0),
            inc_ms: inc_ms.unwrap_or(0),
        }),
        _ => None,
    };
    let max_moves = max_moves.unwrap_or(512);

    // A/B 対戦: レバーの on/off を A/B へ割る (maou_usi::ab が単一実装)
    let mode = match &ab_mode {
        None => None,
        Some(name) => Some(AbMode::parse(name).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "未知の ab_mode: {name} \
                 (subtree | maxmoves | budget | horizon | spin | proven)"
            ))
        })?),
    };
    if let Some(m) = mode {
        if m.needs_clock() && clock.is_none() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "ab_mode=horizon は持ち時間モード (clock_ms > 0) が必要",
            ));
        }
        // 空回りの会計は固定 playout 予算でのみ効く．持ち時間モードは時計が
        // 拘束条件なので，レバーを入れても消費 wall clock は変わらない
        // (「効果なし」と誤結論する regime を最初から塞ぐ)
        if m == AbMode::Spin && clock.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "ab_mode=spin は固定 playout 予算専用 (clock_ms は指定不可)                  — 持ち時間モードでは予算の拘束条件が時計なので効果が出ない",
            ));
        }
    }
    let alternate_colors = alternate_colors.unwrap_or(mode.is_some());
    let setup = mode.map(|mode| {
        build_ab(
            &engine,
            &AbOptions {
                mode,
                playouts_b,
                max_moves,
                horizon_a: horizon_moves.unwrap_or(engine.time.horizon_moves),
                horizon_b: horizon_moves_b.unwrap_or(25),
            },
            playouts,
        )
    });

    let config = SelfplayConfig {
        engine: setup.as_ref().map_or(engine, |s| s.engine_a.clone()),
        engine_b: setup.as_ref().map(|s| s.engine_b.clone()),
        alternate_colors,
        sync_max_moves_to_draw: setup.as_ref().is_none_or(|s| s.sync_max_moves_to_draw),
        sfen,
        games: games.unwrap_or(1),
        parallel: parallel.unwrap_or(1),
        playouts_b: setup.as_ref().and_then(|s| s.playouts_b),
        movetime_ms_b: None,
        clock,
        // playouts/movetime_ms/clock がどれも未指定なら playout 予算の
        // デフォルトを使う (持ち時間モードでは時計から算出させる)
        playouts: match (playouts, movetime_ms, clock) {
            (None, None, None) => SelfplayConfig::default().playouts,
            _ => playouts,
        },
        movetime_ms,
        max_moves,
        opening_random_plies: opening_random_plies.unwrap_or(0),
        seed: seed.unwrap_or(0),
    };
    let verbose = verbose.unwrap_or(true);
    let total = config.games;

    let progress = move |o: &GameOutcome| {
        if verbose {
            let winner = match o.winner {
                Some(maou_shogi::types::Color::Black) => "black wins",
                Some(maou_shogi::types::Color::White) => "white wins",
                None => "draw",
            };
            // stderr へ (stdout を汚さない — logging 規約と同じ)
            eprintln!(
                "[selfplay] game {}/{}: {} by {} ({} plies, {:.1}s, {} playouts)",
                o.game_index + 1,
                total,
                winner,
                o.reason.as_str(),
                o.moves.len(),
                o.elapsed_ms as f64 / 1000.0,
                o.playouts,
            );
        }
    };

    let ab = mode.is_some();
    let outcomes = py
        .detach(move || maou_usi::selfplay::run_selfplay(&config, Some(&progress)))
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

    let list = PyList::empty(py);
    for o in &outcomes {
        list.append(outcome_to_dict(py, o)?)?;
    }
    let summary = summarize(
        &outcomes,
        SummaryOptions {
            ab,
            paired: ab && alternate_colors,
        },
    );
    let result = PyDict::new(py);
    result.set_item("records", list)?;
    result.set_item("summary", summary_to_dict(py, &summary)?)?;
    Ok(result.unbind())
}

/// Create maou_usi submodule
pub fn create_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "maou_usi")?;
    m.add_function(wrap_pyfunction!(run_usi, &m)?)?;
    m.add_function(wrap_pyfunction!(run_selfplay, &m)?)?;
    Ok(m)
}
