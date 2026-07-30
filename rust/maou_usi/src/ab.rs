//! 自己対局 A/B 計測 — 設定レバーの効果を対戦で測る (設計 §12 の未決事項)．
//!
//! 2 つの責務だけを持つ:
//!
//! - [`build_ab`]: レバーの on/off を プレイヤー A/B の [`EngineConfig`] へ割る
//! - [`summarize`]: 対局結果 ([`GameOutcome`]) を A 視点の統計へ集計する
//!
//! `maou selfplay --ab-mode` (PyO3 経由) と Rust example `selfplay_ab` が
//! 同じ実装を使うため，どちらから回しても数値の定義は一致する．
//!
//! 集計が勝率だけでなく **機構の発火量** (subtree 再利用の引き継ぎ訪問数) と
//! **ペア比較の t 値** を必ず出すのは，n=40 級の勝率だけでは「効果なし」と
//! 「機構が動いていない」を区別できないため (worklog 2026-07-26: 40 局の
//! Wilson CI は ±15% 程度あり，予算 +18% 相当 = 約 +50 Elo は原理的に埋もれる)．

use std::collections::BTreeMap;
use std::fmt;

use crate::selfplay::GameOutcome;
use crate::EngineConfig;

/// A/B で比較するレバー．
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AbMode {
    /// A = subtree 再利用 on (現行の既定) / B = off (設計 §12 未決 6)．
    Subtree,
    /// A = `MaxMovesToDraw` の in-search 終端化 on / B = off (未決 4)．
    /// 両者とも driver の最大手数で引き分けになる点は同一で，探索が
    /// リミットを知っているかどうかだけが違う．
    MaxMoves,
    /// A/B 同一設定で探索予算だけを変える (ハーネスの健全性確認)．
    /// 予算の多い側が勝たなければ，この driver で棋力差を測ること自体が
    /// 成立していない — レバーの A/B より前に通すべき検証．
    Budget,
    /// 持ち時間モードで `TimeStrategy` の想定残り手数を A/B (未決 1)．
    Horizon,
    /// A = 手数カーブ (中盤重み付け) on / B = 一律配分 (現行の既定)．
    ///
    /// 序盤は定跡化された分岐で探索時間の限界効用が低く，終盤は dfpn の詰み
    /// 判定が効くという想定を，裁量枠の中盤寄せとして検証する．**持ち時間
    /// モードでのみ意味を持つ** — 固定 playout 予算では配分そのものが無い．
    TimeCurve,
    /// A = 確定済みの子を選択候補から外す (MCTS-Solver) / B = 従来どおり．
    ///
    /// 空回りの降下自体が消えるので，固定予算でも持ち時間モードでも効く．
    Proven,
    /// A = 空回りを playout 予算から外す / B = 従来どおり合算．
    ///
    /// **固定 playout 予算でのみ意味を持つ** (`--playouts`)．持ち時間モードは
    /// 時計が拘束条件なので，会計を変えても消費 wall clock は変わらない．
    Spin,
    /// A/B で評価バッチサイズだけを変える．
    ///
    /// バッチサイズは**速度と探索の質の両方**に効く: TensorRT のコストは
    /// padding 後の長さにほぼ比例する (固定費はほぼ無い) ので，充填しきれない
    /// 大きなバッチは払った padding を捨てている．一方で in-flight の葉が
    /// 増えるほど virtual loss が PUCT を歪める．**持ち時間モードで測ること**
    /// — 固定 playout 予算では速度差が棋力差にならない．
    Batch,
}

impl AbMode {
    /// CLI 文字列 → モード (未知の文字列は `None`)．
    pub fn parse(s: &str) -> Option<AbMode> {
        match s {
            "subtree" => Some(AbMode::Subtree),
            "maxmoves" => Some(AbMode::MaxMoves),
            "budget" => Some(AbMode::Budget),
            "horizon" => Some(AbMode::Horizon),
            "timecurve" => Some(AbMode::TimeCurve),
            "spin" => Some(AbMode::Spin),
            "proven" => Some(AbMode::Proven),
            "batch" => Some(AbMode::Batch),
            _ => None,
        }
    }

    /// CLI 文字列表現．
    pub fn as_str(&self) -> &'static str {
        match self {
            AbMode::Subtree => "subtree",
            AbMode::MaxMoves => "maxmoves",
            AbMode::Budget => "budget",
            AbMode::Horizon => "horizon",
            AbMode::TimeCurve => "timecurve",
            AbMode::Spin => "spin",
            AbMode::Proven => "proven",
            AbMode::Batch => "batch",
        }
    }

    /// 持ち時間モード (実時計) を必要とするか．
    pub fn needs_clock(&self) -> bool {
        matches!(self, AbMode::Horizon | AbMode::TimeCurve)
    }
}

/// A/B のパラメータ ([`build_ab`] の入力)．
#[derive(Clone, Copy, Debug)]
pub struct AbOptions {
    /// 比較するレバー．
    pub mode: AbMode,
    /// [`AbMode::Budget`] で B に渡す playout 予算 (`None` = A の 1/8)．
    pub playouts_b: Option<u64>,
    /// [`AbMode::MaxMoves`] で A 側が探索へ渡すリミット (driver の最大手数)．
    pub max_moves: u32,
    /// [`AbMode::Horizon`] の A 側 想定残り手数．
    pub horizon_a: u64,
    /// [`AbMode::Horizon`] の B 側 想定残り手数．
    pub horizon_b: u64,
    /// [`AbMode::Batch`] で B に渡す評価バッチサイズ (`None` = A の 4 倍)．
    pub batch_size_b: Option<usize>,
}

/// [`build_ab`] の出力 ([`crate::selfplay::SelfplayConfig`] へそのまま載せる)．
#[derive(Clone, Debug)]
pub struct AbSetup {
    /// プレイヤー A (レバー on 側) のエンジン設定．
    pub engine_a: EngineConfig,
    /// プレイヤー B (レバー off 側) のエンジン設定．
    pub engine_b: EngineConfig,
    /// driver がエージェントの `max_moves_to_draw` を同期してよいか
    /// ([`AbMode::MaxMoves`] のときだけ false = 呼び出し側が管理する)．
    pub sync_max_moves_to_draw: bool,
    /// B の playout 予算 (`None` = A と同じ)．
    pub playouts_b: Option<u64>,
}

/// レバーの on/off を A/B の [`EngineConfig`] へ割る．
///
/// `base` は共通条件 (モデル・スレッド数・詰み探索など呼び出し側の設定) で，
/// 差分はモードが決めるフィールドだけに限る — A/B の差が 1 つであることを
/// この関数が保証する．`playouts` は [`AbMode::Budget`] の既定 (A の 1/8)
/// 算出にだけ使う．
pub fn build_ab(base: &EngineConfig, opts: &AbOptions, playouts: Option<u64>) -> AbSetup {
    let mut engine_a = base.clone();
    let mut engine_b = base.clone();
    let mut sync_max_moves_to_draw = true;
    let mut playouts_b = None;
    match opts.mode {
        AbMode::Subtree => {
            engine_b.subtree_reuse = false;
        }
        AbMode::MaxMoves => {
            // per-side に管理するので driver の同期は切る (終局判定自体は
            // driver 側の max_moves で両者共通)
            engine_a.max_moves_to_draw = opts.max_moves;
            engine_b.max_moves_to_draw = 0;
            sync_max_moves_to_draw = false;
        }
        AbMode::Budget => {
            playouts_b = Some(
                opts.playouts_b
                    .unwrap_or_else(|| (playouts.unwrap_or(0) / 8).max(1)),
            );
        }
        AbMode::Horizon => {
            engine_a.time.horizon_moves = opts.horizon_a;
            engine_b.time.horizon_moves = opts.horizon_b;
        }
        AbMode::TimeCurve => {
            // パラメータ (`curve_params`) は CLI から両者に同じ値が入っている．
            // 差は「掛けるか掛けないか」だけにする
            engine_a.time.curve_enabled = true;
            engine_b.time.curve_enabled = false;
        }
        AbMode::Spin => {
            engine_a.spin_budget_relief = true;
            engine_b.spin_budget_relief = false;
        }
        AbMode::Proven => {
            engine_a.skip_proven_children = true;
            engine_b.skip_proven_children = false;
        }
        AbMode::Batch => {
            // A は base の batch_size をそのまま使い，B だけを変える
            engine_b.batch_size = opts
                .batch_size_b
                .unwrap_or_else(|| engine_a.batch_size.saturating_mul(4).max(1));
        }
    }
    AbSetup {
        engine_a,
        engine_b,
        sync_max_moves_to_draw,
        playouts_b,
    }
}

/// 集計のモード指定．
#[derive(Clone, Copy, Debug, Default)]
pub struct SummaryOptions {
    /// A/B 対戦 (`engine_b` あり) — A 視点の勝率統計を出す．
    pub ab: bool,
    /// ペア開局 + 色交代 — 対局 (2n, 2n+1) を対にした差分統計を出す．
    pub paired: bool,
}

/// A 視点の勝率統計 (A/B 対戦のときだけ作られる)．
#[derive(Clone, Debug, PartialEq)]
pub struct AbSummary {
    /// A から見た勝ち数．
    pub wins: u32,
    /// 引き分け数．
    pub draws: u32,
    /// A から見た負け数．
    pub losses: u32,
    /// A のスコア (勝ち 1 点 + 引き分け 0.5 点)．
    pub score: f64,
    /// スコア率 (`score / games`)．
    pub score_rate: f64,
    /// Wilson 95% 信頼区間の下限 (スコア率)．
    pub ci_lo: f64,
    /// Wilson 95% 信頼区間の上限 (スコア率)．
    pub ci_hi: f64,
    /// スコア率の Elo 換算 (0% / 100% では `None`)．
    pub elo: Option<f64>,
    /// Elo 換算の下限 (`ci_lo` 由来)．
    pub elo_lo: Option<f64>,
    /// Elo 換算の上限 (`ci_hi` 由来)．
    pub elo_hi: Option<f64>,
    /// ペア数 (色交代 + ペア開局のときのみ > 0)．
    pub pairs: u32,
    /// A が優位だったペア数 (ペア合計スコア > 1.0)．
    pub pairs_a_ahead: u32,
    /// 同着ペア数．
    pub pairs_tied: u32,
    /// B が優位だったペア数．
    pub pairs_b_ahead: u32,
    /// ペア差分の平均 (ペア合計スコア − 1.0．範囲 [-1, +1])．
    pub paired_mean: f64,
    /// ペア差分の標準誤差．
    pub paired_se: f64,
    /// ペア差分の t 値 (`paired_mean / paired_se`)．
    pub paired_t: f64,
    /// 終局時の A 側平均残り持ち時間 (ミリ秒．持ち時間モードのみ)．
    pub time_left_a_ms: Option<f64>,
    /// 終局時の B 側平均残り持ち時間 (ミリ秒．持ち時間モードのみ)．
    pub time_left_b_ms: Option<f64>,
}

/// 自己対局 1 回分の集計．
#[derive(Clone, Debug, PartialEq)]
pub struct RunSummary {
    /// 対局数．
    pub games: u32,
    /// 先手勝ち数．
    pub black_wins: u32,
    /// 後手勝ち数．
    pub white_wins: u32,
    /// 引き分け数．
    pub draws: u32,
    /// 終局理由のヒストグラム (理由名の昇順)．
    pub reasons: Vec<(&'static str, u32)>,
    /// 総手数．
    pub total_plies: u64,
    /// 総 playout 数 (両側合計)．
    pub total_playouts: u64,
    /// 全局合計の空回り数 (終端到達だけで折り返した backprop)．
    /// `total_playouts` との比で「予算のどれだけが空回りに消えたか」を見る．
    pub total_terminal_backprops: u64,
    /// 対局の壁時計時間の総和 (ミリ秒．並列時は wall clock と一致しない)．
    pub total_game_ms: u64,
    /// subtree 再利用が発火した手数．
    pub reused_moves: u32,
    /// 引き継いだ訪問数の合計．
    pub carried_visits: u64,
    /// A 視点の勝率統計 (A/B 対戦のときのみ)．
    pub ab: Option<AbSummary>,
}

/// 対局結果を集計する．
///
/// `opts.ab` が true のときだけ A 視点の統計を作る (純粋自己対局では
/// 「A の勝ち」に意味がないため)．`opts.paired` が true ならペア (2n, 2n+1)
/// の差分統計も作る — 開局サンプリング分散を相殺した比較になる．
pub fn summarize(outcomes: &[GameOutcome], opts: SummaryOptions) -> RunSummary {
    use maou_shogi::types::Color;

    let mut black_wins = 0;
    let mut white_wins = 0;
    let mut draws = 0;
    let mut reasons: BTreeMap<&'static str, u32> = BTreeMap::new();
    let mut total_plies = 0u64;
    let mut total_playouts = 0u64;
    let mut total_terminal_backprops = 0u64;
    let mut total_game_ms = 0u64;
    let mut reused_moves = 0u32;
    let mut carried_visits = 0u64;
    for o in outcomes {
        match o.winner {
            Some(Color::Black) => black_wins += 1,
            Some(Color::White) => white_wins += 1,
            None => draws += 1,
        }
        *reasons.entry(o.reason.as_str()).or_default() += 1;
        total_plies += o.moves.len() as u64;
        total_playouts += o.playouts;
        total_terminal_backprops += o.terminal_backprops;
        total_game_ms += o.elapsed_ms;
        reused_moves += o.reused_moves;
        carried_visits += o.carried_visits;
    }

    RunSummary {
        games: outcomes.len() as u32,
        black_wins,
        white_wins,
        draws,
        reasons: reasons.into_iter().collect(),
        total_plies,
        total_playouts,
        total_terminal_backprops,
        total_game_ms,
        reused_moves,
        carried_visits,
        ab: opts.ab.then(|| summarize_ab(outcomes, opts.paired)),
    }
}

/// A 視点の 1 局のスコア (勝ち 1.0 / 引き分け 0.5 / 負け 0.0)．
fn a_score(o: &GameOutcome) -> f64 {
    use maou_shogi::types::Color;
    match o.winner {
        None => 0.5,
        Some(Color::Black) => f64::from(u8::from(o.black_is_a)),
        Some(Color::White) => f64::from(u8::from(!o.black_is_a)),
    }
}

/// A 視点の勝率統計を作る．
fn summarize_ab(outcomes: &[GameOutcome], paired: bool) -> AbSummary {
    let mut wins = 0u32;
    let mut draws = 0u32;
    let mut losses = 0u32;
    let mut time_left = [0f64; 2];
    let mut clocked = 0u32;
    for o in outcomes {
        let s = a_score(o);
        if s > 0.75 {
            wins += 1;
        } else if s < 0.25 {
            losses += 1;
        } else {
            draws += 1;
        }
        if let Some(rem) = o.remaining_ms {
            // rem は [先手, 後手]．A/B 視点へ写す
            let (a, b) = if o.black_is_a {
                (rem[0], rem[1])
            } else {
                (rem[1], rem[0])
            };
            time_left[0] += a as f64;
            time_left[1] += b as f64;
            clocked += 1;
        }
    }
    let n = outcomes.len() as f64;
    let score = f64::from(wins) + 0.5 * f64::from(draws);
    let score_rate = if n > 0.0 { score / n } else { 0.0 };
    let (ci_lo, ci_hi) = wilson_95(score, n);

    // ペア差分: 対局 (2n, 2n+1) の A スコア合計 − 1.0 (= 引き分け均衡)．
    // 端数の 1 局は捨てる (対にならないため)
    let mut diffs: Vec<f64> = Vec::new();
    let mut pairs_a_ahead = 0u32;
    let mut pairs_tied = 0u32;
    let mut pairs_b_ahead = 0u32;
    if paired {
        for chunk in outcomes.chunks_exact(2) {
            let d = a_score(&chunk[0]) + a_score(&chunk[1]) - 1.0;
            if d > 0.0 {
                pairs_a_ahead += 1;
            } else if d < 0.0 {
                pairs_b_ahead += 1;
            } else {
                pairs_tied += 1;
            }
            diffs.push(d);
        }
    }
    let (paired_mean, paired_se, paired_t) = paired_stats(&diffs);

    AbSummary {
        wins,
        draws,
        losses,
        score,
        score_rate,
        ci_lo,
        ci_hi,
        elo: elo(score_rate),
        elo_lo: elo(ci_lo),
        elo_hi: elo(ci_hi),
        pairs: diffs.len() as u32,
        pairs_a_ahead,
        pairs_tied,
        pairs_b_ahead,
        paired_mean,
        paired_se,
        paired_t,
        time_left_a_ms: (clocked > 0).then(|| time_left[0] / f64::from(clocked)),
        time_left_b_ms: (clocked > 0).then(|| time_left[1] / f64::from(clocked)),
    }
}

/// Wilson score interval (95%) — 少数対局でも壊れない勝率の区間推定．
pub fn wilson_95(score: f64, n: f64) -> (f64, f64) {
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

/// スコア率 → Elo 差 (0% / 100% は発散するため `None`)．
pub fn elo(score_rate: f64) -> Option<f64> {
    if score_rate <= 0.0 || score_rate >= 1.0 {
        return None;
    }
    // + 0.0 は 50% ちょうどで -0.0 になるのを避けるため (表示が "-0" になる)
    Some(-400.0 * (1.0 / score_rate - 1.0).log10() + 0.0)
}

/// ペア差分の (平均, 標準誤差, t 値)．標本 0/1 個では SE と t を 0 にする．
fn paired_stats(diffs: &[f64]) -> (f64, f64, f64) {
    let n = diffs.len();
    if n == 0 {
        return (0.0, 0.0, 0.0);
    }
    let mean = diffs.iter().sum::<f64>() / n as f64;
    if n < 2 {
        return (mean, 0.0, 0.0);
    }
    // 不偏分散 → 標準誤差
    let var = diffs.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    let se = (var / n as f64).sqrt();
    let t = if se > 0.0 { mean / se } else { 0.0 };
    (mean, se, t)
}

/// 人が読むサマリ (Rust example 用．CLI は Python 側で整形する)．
impl fmt::Display for RunSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "games: {}", self.games)?;
        writeln!(
            f,
            "results: black {} / white {} / draw {}",
            self.black_wins, self.white_wins, self.draws
        )?;
        writeln!(
            f,
            "reasons: {}",
            self.reasons
                .iter()
                .map(|(k, v)| format!("{k} {v}"))
                .collect::<Vec<_>>()
                .join(", ")
        )?;
        writeln!(
            f,
            "plies: {} total, playouts: {} total, game time: {:.1}s summed",
            self.total_plies,
            self.total_playouts,
            self.total_game_ms as f64 / 1000.0,
        )?;
        writeln!(
            f,
            "subtree reuse: {} move(s) warm-started, {} visits carried over ({:.1}% of playouts)",
            self.reused_moves,
            self.carried_visits,
            100.0 * self.carried_visits as f64 / self.total_playouts.max(1) as f64,
        )?;
        if let Some(ab) = &self.ab {
            let elo = |v: Option<f64>| match v {
                Some(e) => format!("{e:+.0}"),
                None => "n/a".to_string(),
            };
            writeln!(f, "A result: {}W {}D {}L", ab.wins, ab.draws, ab.losses)?;
            writeln!(
                f,
                "A score: {:.1}/{} = {:.1}% (Wilson 95% CI [{:.1}%, {:.1}%])",
                ab.score,
                self.games,
                100.0 * ab.score_rate,
                100.0 * ab.ci_lo,
                100.0 * ab.ci_hi,
            )?;
            writeln!(
                f,
                "A Elo: {} [{}, {}]",
                elo(ab.elo),
                elo(ab.elo_lo),
                elo(ab.elo_hi)
            )?;
            if ab.pairs > 0 {
                writeln!(
                    f,
                    "paired: {} pairs, mean {:+.3} (SE {:.3}), t = {:+.2}, A ahead in {}/{} (tied {})",
                    ab.pairs,
                    ab.paired_mean,
                    ab.paired_se,
                    ab.paired_t,
                    ab.pairs_a_ahead,
                    ab.pairs,
                    ab.pairs_tied,
                )?;
            }
            if let (Some(a), Some(b)) = (ab.time_left_a_ms, ab.time_left_b_ms) {
                writeln!(
                    f,
                    "time left at end (avg): A {:.1}s / B {:.1}s",
                    a / 1000.0,
                    b / 1000.0
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::selfplay::GameEndReason;
    use maou_shogi::types::Color;

    fn outcome(game_index: u32, black_is_a: bool, winner: Option<Color>) -> GameOutcome {
        GameOutcome {
            game_index,
            sfen: "startpos".to_string(),
            moves: vec!["7g7f".to_string()],
            move_times_ms: vec![10],
            move_playouts: vec![100],
            move_scores: vec![0],
            black_is_a,
            winner,
            reason: GameEndReason::MaxMoves,
            playouts: 100,
            terminal_backprops: 0,
            reused_moves: 1,
            carried_visits: 20,
            elapsed_ms: 1000,
            remaining_ms: None,
        }
    }

    #[test]
    fn test_ab_mode_parse_roundtrip() {
        for name in ["subtree", "maxmoves", "budget", "horizon"] {
            let mode = AbMode::parse(name).expect("既知のモード");
            assert_eq!(mode.as_str(), name);
        }
        assert!(AbMode::parse("unknown").is_none());
        assert!(AbMode::Horizon.needs_clock());
        assert!(!AbMode::Subtree.needs_clock());
    }

    #[test]
    fn test_build_ab_batch_changes_only_batch_size() {
        let base = EngineConfig {
            batch_size: 64,
            ..EngineConfig::default()
        };
        let opts = AbOptions {
            mode: AbMode::Batch,
            playouts_b: None,
            max_moves: 512,
            horizon_a: 40,
            horizon_b: 25,
            batch_size_b: Some(256),
        };
        let setup = build_ab(&base, &opts, Some(64));
        assert_eq!(setup.engine_a.batch_size, 64, "A は base のまま");
        assert_eq!(setup.engine_b.batch_size, 256);
        // 差は batch_size ただ 1 つであること
        assert_eq!(setup.engine_a.threads, setup.engine_b.threads);
        assert_eq!(setup.engine_a.subtree_reuse, setup.engine_b.subtree_reuse);
        assert_eq!(
            setup.engine_a.spin_budget_relief,
            setup.engine_b.spin_budget_relief
        );
        assert!(setup.playouts_b.is_none(), "予算は動かさない");
    }

    #[test]
    fn test_build_ab_batch_defaults_b_to_four_times_a() {
        let base = EngineConfig {
            batch_size: 64,
            ..EngineConfig::default()
        };
        let opts = AbOptions {
            mode: AbMode::Batch,
            playouts_b: None,
            max_moves: 512,
            horizon_a: 40,
            horizon_b: 25,
            batch_size_b: None,
        };
        let setup = build_ab(&base, &opts, Some(64));
        assert_eq!(setup.engine_b.batch_size, 256);
    }

    #[test]
    fn test_ab_mode_batch_round_trips_through_cli_string() {
        // Rust 側の列挙と CLI 文字列がずれると「未知の ab_mode」で落ちる
        assert_eq!(AbMode::parse("batch"), Some(AbMode::Batch));
        assert_eq!(AbMode::Batch.as_str(), "batch");
        assert!(
            !AbMode::Batch.needs_clock(),
            "時計は必須にしない (計測は推奨)"
        );
    }

    #[test]
    fn test_build_ab_differs_in_exactly_one_lever() {
        let base = EngineConfig::default();
        let opts = AbOptions {
            mode: AbMode::Subtree,
            playouts_b: None,
            max_moves: 512,
            horizon_a: 40,
            horizon_b: 25,
            batch_size_b: None,
        };
        let setup = build_ab(&base, &opts, Some(64));
        assert!(setup.engine_a.subtree_reuse, "A は既定 (on) のまま");
        assert!(!setup.engine_b.subtree_reuse);
        assert!(setup.sync_max_moves_to_draw);
        assert_eq!(setup.playouts_b, None, "予算は共通");
        // 差はレバー 1 つだけ (他フィールドは base のまま)
        let mut b_as_a = setup.engine_b.clone();
        b_as_a.subtree_reuse = setup.engine_a.subtree_reuse;
        assert_eq!(format!("{b_as_a:?}"), format!("{:?}", setup.engine_a));
    }

    #[test]
    fn test_build_ab_maxmoves_unsyncs_driver() {
        let opts = AbOptions {
            mode: AbMode::MaxMoves,
            playouts_b: None,
            max_moves: 120,
            horizon_a: 40,
            horizon_b: 25,
            batch_size_b: None,
        };
        let setup = build_ab(&EngineConfig::default(), &opts, Some(64));
        assert_eq!(setup.engine_a.max_moves_to_draw, 120);
        assert_eq!(setup.engine_b.max_moves_to_draw, 0);
        assert!(
            !setup.sync_max_moves_to_draw,
            "driver に上書きさせるとレバーが消える"
        );
    }

    #[test]
    fn test_build_ab_budget_defaults_to_one_eighth() {
        let opts = AbOptions {
            mode: AbMode::Budget,
            playouts_b: None,
            max_moves: 512,
            horizon_a: 40,
            horizon_b: 25,
            batch_size_b: None,
        };
        let setup = build_ab(&EngineConfig::default(), &opts, Some(64));
        assert_eq!(setup.playouts_b, Some(8));
        // 明示指定が優先される
        let setup = build_ab(
            &EngineConfig::default(),
            &AbOptions {
                playouts_b: Some(3),
                ..opts
            },
            Some(64),
        );
        assert_eq!(setup.playouts_b, Some(3));
    }

    #[test]
    fn test_build_ab_horizon_sets_time_strategy() {
        let opts = AbOptions {
            mode: AbMode::Horizon,
            playouts_b: None,
            max_moves: 512,
            horizon_a: 40,
            horizon_b: 20,
            batch_size_b: None,
        };
        let setup = build_ab(&EngineConfig::default(), &opts, None);
        assert_eq!(setup.engine_a.time.horizon_moves, 40);
        assert_eq!(setup.engine_b.time.horizon_moves, 20);
    }

    #[test]
    fn test_build_ab_timecurve_splits_only_the_enable_flag() {
        let mut base = EngineConfig::default();
        base.time.curve_params.peak_ply = 60;
        let opts = AbOptions {
            mode: AbMode::TimeCurve,
            playouts_b: None,
            max_moves: 512,
            horizon_a: 40,
            horizon_b: 25,
            batch_size_b: None,
        };
        let setup = build_ab(&base, &opts, None);
        assert!(setup.engine_a.time.curve_enabled);
        assert!(!setup.engine_b.time.curve_enabled);
        // パラメータと想定残り手数は両者で同一 (差は on/off だけ)
        assert_eq!(setup.engine_a.time.curve_params.peak_ply, 60);
        assert_eq!(setup.engine_b.time.curve_params.peak_ply, 60);
        assert_eq!(
            setup.engine_a.time.horizon_moves,
            setup.engine_b.time.horizon_moves
        );
    }

    #[test]
    fn test_timecurve_needs_clock() {
        // 固定 playout 予算では配分そのものが無いので実時計が要る
        assert!(AbMode::TimeCurve.needs_clock());
        assert_eq!(AbMode::parse("timecurve"), Some(AbMode::TimeCurve));
        assert_eq!(AbMode::TimeCurve.as_str(), "timecurve");
    }

    #[test]
    fn test_summarize_a_view_follows_color_alternation() {
        // A が先手で勝ち / A が後手で勝ち / A が先手で負け / 引き分け
        let outcomes = vec![
            outcome(0, true, Some(Color::Black)),
            outcome(1, false, Some(Color::White)),
            outcome(2, true, Some(Color::White)),
            outcome(3, false, None),
        ];
        let s = summarize(
            &outcomes,
            SummaryOptions {
                ab: true,
                paired: false,
            },
        );
        assert_eq!(s.games, 4);
        assert_eq!((s.black_wins, s.white_wins, s.draws), (1, 2, 1));
        let ab = s.ab.expect("A/B 集計あり");
        assert_eq!((ab.wins, ab.draws, ab.losses), (2, 1, 1));
        assert!((ab.score - 2.5).abs() < 1e-9);
        assert!((ab.score_rate - 0.625).abs() < 1e-9);
        assert_eq!(ab.pairs, 0, "paired=false ではペア統計を作らない");
    }

    #[test]
    fn test_summarize_paired_statistics() {
        // ペア 0: A が両局勝ち (+1.0) / ペア 1: 1 勝 1 敗 (0.0)
        let outcomes = vec![
            outcome(0, true, Some(Color::Black)),
            outcome(1, false, Some(Color::White)),
            outcome(2, true, Some(Color::Black)),
            outcome(3, false, Some(Color::Black)),
        ];
        let s = summarize(
            &outcomes,
            SummaryOptions {
                ab: true,
                paired: true,
            },
        );
        let ab = s.ab.expect("A/B 集計あり");
        assert_eq!(ab.pairs, 2);
        assert_eq!(
            (ab.pairs_a_ahead, ab.pairs_tied, ab.pairs_b_ahead),
            (1, 1, 0)
        );
        assert!((ab.paired_mean - 0.5).abs() < 1e-9);
        // 差分 [1.0, 0.0] の不偏分散 = 0.5 → SE = sqrt(0.5/2) = 0.5
        assert!((ab.paired_se - 0.5).abs() < 1e-9);
        assert!((ab.paired_t - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_summarize_aggregates_mechanism_counters() {
        let outcomes = vec![outcome(0, true, None), outcome(1, true, None)];
        let s = summarize(&outcomes, SummaryOptions::default());
        assert!(s.ab.is_none(), "純粋自己対局では A 視点を作らない");
        assert_eq!(s.total_playouts, 200);
        assert_eq!(s.reused_moves, 2);
        assert_eq!(s.carried_visits, 40);
        assert_eq!(s.total_plies, 2);
        assert_eq!(s.reasons, vec![("max_moves", 2)]);
    }

    #[test]
    fn test_wilson_and_elo() {
        // 全勝は上限 1.0 に張り付き，下限は 1.0 未満
        let (lo, hi) = wilson_95(10.0, 10.0);
        assert!(lo > 0.6 && lo < 1.0, "lo = {lo}");
        assert!((hi - 1.0).abs() < 1e-9);
        // 五分は 0.5 を挟む
        let (lo, hi) = wilson_95(20.0, 40.0);
        assert!(lo < 0.5 && hi > 0.5);
        // n=0 は区間が全域
        assert_eq!(wilson_95(0.0, 0.0), (0.0, 1.0));

        assert_eq!(elo(0.5), Some(0.0));
        assert!((elo(0.75).expect("有限") - 190.8).abs() < 0.1);
        assert_eq!(elo(1.0), None);
        assert_eq!(elo(0.0), None);
    }

    #[test]
    fn test_time_left_maps_to_ab_sides() {
        let mut a_black = outcome(0, true, None);
        a_black.remaining_ms = Some([10_000, 4_000]);
        let mut a_white = outcome(1, false, None);
        a_white.remaining_ms = Some([6_000, 20_000]);
        let s = summarize(
            &[a_black, a_white],
            SummaryOptions {
                ab: true,
                paired: false,
            },
        );
        let ab = s.ab.expect("A/B 集計あり");
        // A = 10s と 20s / B = 4s と 6s
        assert!((ab.time_left_a_ms.expect("時計あり") - 15_000.0).abs() < 1e-9);
        assert!((ab.time_left_b_ms.expect("時計あり") - 5_000.0).abs() < 1e-9);
    }
}
