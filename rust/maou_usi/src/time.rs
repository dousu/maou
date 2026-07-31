//! 時間管理 (TimeStrategy) — 持ち時間 → 1 手予算の変換レイヤー．
//!
//! 「持ち時間の消費計画は別レイヤー — 1 局面探索は与えられた予算内まで」
//! (docs/design/usi-engine/index.md §8.1) の上位レイヤー実装．探索側は
//! ここが決めた予算を消費するだけで時間配分の知識を持たない．
//!
//! soft (通常打ち切り目標) と hard (絶対上限) を分離し，soft 到達時に root の
//! 最有力手が不安定 (上位 2 手の訪問が拮抗) なら hard まで延長する (設計 §8.1)．
//! 延長判断も本モジュール ([`should_stop`]) が持つ．
//!
//! 定数 (想定残り手数 `horizon_moves`・延長倍率・安定判定比) は自己対局で
//! 調整して worklog に記録する (設計 doc §12 未決事項 1)．

use maou_shogi::types::Color;

use crate::protocol::ClockParams;

/// soft から hard への延長倍率 (分子/分母)．soft の `EXT_NUM/EXT_DEN` 倍まで
/// 延長を許す (ただし残時間の絶対上限 ceiling は超えない)．暫定 2 倍
/// (自己対局で調整予定 — 設計 §12 未決 1)．
const EXT_NUM: u64 = 2;
const EXT_DEN: u64 = 1;

/// 最有力手が「安定」と見なす上位 2 手の訪問比 (best ≥ `STABLE_NUM/STABLE_DEN`
/// × second)．暫定 1.5 倍 (拮抗していれば延長する)．
const STABLE_NUM: u64 = 3;
const STABLE_DEN: u64 = 2;

/// 1 手の思考予算 (ミリ秒)．
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimeBudget {
    /// 通常の思考打ち切り目標．
    pub soft_ms: u64,
    /// 絶対上限 (時間切れ安全マージン込み)．M1 では soft と同値．
    pub hard_ms: u64,
}

/// 重み 1.0 を表す permille 値 (整数演算のスケール)．
const FLAT_PERMILLE: u64 = 1000;

/// 手数に応じて**裁量枠**に掛ける乗数を決めるカーブ．
///
/// 山型の折れ線で，`peak_ply` を頂点に `half_width_ply` かけて底まで線形に
/// 落ちる．頂点から `half_width_ply` 以上離れた手数は一律その側の底．
///
/// **底は序盤側と終盤側で別**にする．自己対局 + 中立な再解析で，序盤の
/// 探索時間はほぼ無価値 (winrate loss 0.0004) だが，**変換期 (優勢を勝ちに
/// 変える帯) の時間は中盤より価値が高い**ことが測れた．同じ底を使うと，
/// 切れ負け (毎手戻ってくる時間が無い) で終盤が必要以上に痩せる．
///
/// **裁量枠 (`残時間 / horizon`) にのみ掛かり，毎手戻ってくる時間
/// (秒読み・フィッシャー加算) には掛からない** ([`allocate`])．これにより
/// 秒読みでは終盤の重みを下げても秒読み分は必ず残り，フィッシャーでは
/// 終盤が加算ペースに収束したうえでバンクを中盤へ寄せられる．
///
/// 配分の土台が常に**その時点の残時間**の分数である以上，カーブは配分を
/// 歪めるだけで総消費を増やす方向には倒れない (使えば残りが減り，以降の
/// 配分が自動的に小さくなる)．
///
/// `peak_permille` が底を下回る退化した設定では，その側の山が消えて一律
/// 底の値になる (安全側)．
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimeCurve {
    /// 重みが最大になる手数 (ply．`GameState::move_number` と同じ数え方)．
    pub peak_ply: u64,
    /// 頂点から底に達するまでの手数．0 は 1 として扱う．
    pub half_width_ply: u64,
    /// 頂点での乗数 (permille．1000 = 1.0 倍)．
    pub peak_permille: u64,
    /// 序盤側 (`ply < peak_ply`) の底の乗数 (permille)．
    pub opening_floor_permille: u64,
    /// 終盤側 (`ply >= peak_ply`) の底の乗数 (permille)．
    pub endgame_floor_permille: u64,
}

impl TimeCurve {
    /// 既定プリセット — 山を**変換期** (優勢を勝ちに変える局面帯) に置く．
    ///
    /// 自己対局 A/B で決めた形 (経緯は
    /// `docs/design/usi-engine/verification.md` §4.4.1)．要点は 2 つ:
    ///
    /// - **序盤の探索時間はほぼ無価値**．中立な再解析で ply 9-30 の平均
    ///   winrate loss は 0.0004 (相手 0.0013) と両者ほぼ完璧だった．
    ///   だから序盤の底を 0.3 倍まで下げて原資にできる．
    /// - **変換期の時間は中盤より価値が高い**．中盤 +12% が loss を
    ///   Δ0.0022 しか改善しないのに対し，変換期 −10% は Δ0.0077 悪化させた．
    ///
    /// 山を中盤 (ply 55) に置いた最初の形は **50 局で 34% (−117 Elo)** と
    /// 明確に負けた．変換期へ移して **20 局で 65% (+108 Elo)** へ反転した．
    pub const DEFAULT: TimeCurve = TimeCurve {
        peak_ply: 100,
        half_width_ply: 55,
        peak_permille: 2500,
        opening_floor_permille: 300,
        endgame_floor_permille: 1200,
    };

    /// 手数に対する乗数 (permille)．
    pub fn weight_permille(&self, ply: u64) -> u64 {
        let half = self.half_width_ply.max(1);
        let dist = ply.abs_diff(self.peak_ply);
        let floor = if ply < self.peak_ply {
            self.opening_floor_permille
        } else {
            self.endgame_floor_permille
        };
        if dist >= half {
            return floor;
        }
        let span = self.peak_permille.saturating_sub(floor);
        floor + span.saturating_mul(half - dist) / half
    }
}

/// 時間戦略の設定 (USI オプション / CLI から与える)．
#[derive(Clone, Debug)]
pub struct TimeStrategyConfig {
    /// 通信・プロセスオーバーヘッドのマージン (ミリ秒)．GUI/サーバは
    /// 相手の指し手送信から bestmove 受信までを消費時間として計測するため，
    /// 探索予算はこの分だけ短くする．
    pub network_delay_ms: u64,
    /// 最低思考時間 (ミリ秒)．マージン控除後もこれを下回らない．
    pub min_think_ms: u64,
    /// 持ち時間を配分する想定残り手数．
    pub horizon_moves: u64,
    /// 手数カーブを有効にするか (**既定 on**)．
    ///
    /// 自己対局 A/B で一律配分より強い側に振れたため既定化した
    /// (20 局 65% / +108 Elo)．**t = +1.15・CI [−47, +262] で有意水準には
    /// 達していない** — 追試で覆り得ることを前提に扱うこと
    /// (経緯: `docs/design/usi-engine/verification.md` §4.4.1)．
    pub curve_enabled: bool,
    /// 手数カーブのパラメータ．`curve_enabled` と独立に**常に保持**する．
    ///
    /// USI の `setoption` は到着順が保証されない (GUI が `TimeCurve` の前後
    /// どちらでパラメータを送るか決められない) ため，パラメータを有効化フラグ
    /// と同じ `Option` に畳み込むと順序次第で設定が失われる．
    pub curve_params: TimeCurve,
}

impl TimeStrategyConfig {
    /// 実効カーブ (無効なら `None`)．
    pub fn curve(&self) -> Option<TimeCurve> {
        self.curve_enabled.then_some(self.curve_params)
    }
}

impl Default for TimeStrategyConfig {
    fn default() -> TimeStrategyConfig {
        TimeStrategyConfig {
            network_delay_ms: 1000,
            min_think_ms: 100,
            horizon_moves: 40,
            curve_enabled: true,
            curve_params: TimeCurve::DEFAULT,
        }
    }
}

/// 自分の手番の残り時間・加算を取り出して 1 手予算 (soft/hard) を計算する．
///
/// 三態 (秒読み / フィッシャー / 切れ負け) を扱う:
/// - 秒読み: `残時間 / horizon × w(ply) + byoyomi − margin`
/// - フィッシャー: `残時間 / horizon × w(ply) + inc − margin`
/// - 切れ負け: `残時間 / horizon × w(ply) − margin` (安全側)
///
/// `w(ply)` は手数カーブ ([`TimeCurve`]) の乗数で，`cfg.curve` が `None` なら
/// 一律 1.0．**裁量枠にのみ掛かり，毎手戻ってくる時間 (秒読み・加算) には
/// 掛からない** — 終盤の重みを下げても保証された床は削られない．
///
/// - `soft_ms` = 通常打ち切り目標 (上記 base 配分)．
/// - `hard_ms` = 延長上限 = `soft × EXT_NUM/EXT_DEN`．ただし今使える時間の
///   絶対上限 ceiling (残時間 + 秒読み − マージン) を超えない．
///
/// いずれも `min_think_ms` を下回らない．ceiling に張り付く局面 (秒読みのみ・
/// 残り僅少) では `soft == hard` になり延長余地はない (安全側)．
///
/// `move_number` は手番側から見た現在の手数 (`GameState::move_number`)．
pub fn allocate(
    cfg: &TimeStrategyConfig,
    clock: &ClockParams,
    my_color: Color,
    move_number: u64,
) -> TimeBudget {
    let (my_time, my_inc) = match my_color {
        Color::Black => (clock.btime.unwrap_or(0), clock.binc.unwrap_or(0)),
        Color::White => (clock.wtime.unwrap_or(0), clock.winc.unwrap_or(0)),
    };
    let byoyomi = clock.byoyomi.unwrap_or(0);
    let horizon = cfg.horizon_moves.max(1);

    // 裁量枠 (バンクの取り崩し)．手数カーブはここにだけ掛ける
    let weight = cfg
        .curve()
        .map_or(FLAT_PERMILLE, |c| c.weight_permille(move_number));
    let discretionary = (my_time / horizon).saturating_mul(weight) / FLAT_PERMILLE;

    // 1 手のベース配分 + 毎手戻ってくる時間 (秒読み or フィッシャー加算)
    let base = discretionary + byoyomi + my_inc;
    let budget = base.saturating_sub(cfg.network_delay_ms);

    // 今使える時間の絶対上限: 残時間 + 秒読み − マージン
    // (フィッシャー加算は指した後に足されるので上限には入れない)
    let ceiling = (my_time + byoyomi).saturating_sub(cfg.network_delay_ms);

    let soft = budget.min(ceiling).max(cfg.min_think_ms);
    // 延長上限: soft の EXT 倍．ただし ceiling を超えず，soft を下回らない
    let hard = soft
        .saturating_mul(EXT_NUM)
        .saturating_div(EXT_DEN)
        .min(ceiling)
        .max(soft);
    TimeBudget {
        soft_ms: soft,
        hard_ms: hard,
    }
}

/// 探索を今すぐ打ち切るべきか (monitor が一定間隔で呼ぶ延長判断)．
///
/// - root が確定 (詰み等) していれば即停止 (探索側も止まるが保険)．
/// - `hard_ms` 到達で必ず停止．
/// - `soft_ms` 到達後は，最有力手が安定していれば停止，上位 2 手が拮抗して
///   いれば `hard_ms` まで延長する ([`is_best_stable`])．
/// - `soft_ms` 未満なら継続．
///
/// `best_visits` / `second_visits` は root 直下の最有力手・次点の訪問回数
/// (進捗スナップショット由来)．`proven` は root 確定の有無．
pub fn should_stop(
    budget: &TimeBudget,
    elapsed_ms: u64,
    best_visits: u64,
    second_visits: u64,
    proven: bool,
) -> bool {
    if proven {
        return true;
    }
    if elapsed_ms >= budget.hard_ms {
        return true;
    }
    if elapsed_ms >= budget.soft_ms {
        return is_best_stable(best_visits, second_visits);
    }
    false
}

/// 最有力手が安定しているか (best が second の `STABLE_NUM/STABLE_DEN` 倍以上の
/// 訪問)．拮抗 (未確立) なら延長する側に倒す．
fn is_best_stable(best_visits: u64, second_visits: u64) -> bool {
    best_visits.saturating_mul(STABLE_DEN) >= second_visits.saturating_mul(STABLE_NUM)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// カーブ off の設定では手数が結果に影響しないことを明示する番人．
    const ANY_PLY: u64 = 1;

    fn cfg() -> TimeStrategyConfig {
        TimeStrategyConfig {
            network_delay_ms: 1000,
            min_think_ms: 100,
            horizon_moves: 40,
            curve_enabled: false,
            curve_params: TimeCurve::DEFAULT,
        }
    }

    fn cfg_curved(curve: TimeCurve) -> TimeStrategyConfig {
        TimeStrategyConfig {
            curve_enabled: true,
            curve_params: curve,
            ..cfg()
        }
    }

    #[test]
    fn test_byoyomi_only() {
        // 残時間 0 + 秒読み 10 秒 → 10s − 1s マージン = 9s
        let clock = ClockParams {
            btime: Some(0),
            wtime: Some(0),
            byoyomi: Some(10_000),
            ..ClockParams::default()
        };
        let b = allocate(&cfg(), &clock, Color::Black, ANY_PLY);
        assert_eq!(b.soft_ms, 9_000);
        assert_eq!(b.hard_ms, 9_000);
    }

    #[test]
    fn test_main_time_plus_byoyomi() {
        // 残 400s / 40 手 = 10s + 秒読み 10s − 1s = 19s
        let clock = ClockParams {
            btime: Some(400_000),
            wtime: Some(400_000),
            byoyomi: Some(10_000),
            ..ClockParams::default()
        };
        let b = allocate(&cfg(), &clock, Color::Black, ANY_PLY);
        assert_eq!(b.soft_ms, 19_000);
    }

    #[test]
    fn test_fischer_uses_my_side_inc() {
        // 電竜戦系: 先手 5 分+2 秒/後手 10 分+2 秒 (非対称)
        let clock = ClockParams {
            btime: Some(300_000),
            wtime: Some(600_000),
            binc: Some(2_000),
            winc: Some(2_000),
            ..ClockParams::default()
        };
        let black = allocate(&cfg(), &clock, Color::Black, ANY_PLY);
        let white = allocate(&cfg(), &clock, Color::White, ANY_PLY);
        // 先手: 300s/40 + 2s − 1s = 8.5s / 後手: 600s/40 + 2s − 1s = 16s
        assert_eq!(black.soft_ms, 8_500);
        assert_eq!(white.soft_ms, 16_000);
    }

    #[test]
    fn test_sudden_death_keeps_reserve() {
        // 切れ負け: 残 40s → 1s/手 − 1s = 0 → 最低思考時間まで切り上げ
        let clock = ClockParams {
            btime: Some(40_000),
            wtime: Some(40_000),
            ..ClockParams::default()
        };
        let b = allocate(&cfg(), &clock, Color::Black, ANY_PLY);
        assert_eq!(b.soft_ms, 100);
    }

    #[test]
    fn test_ceiling_when_time_nearly_exhausted() {
        // フィッシャー加算 5s は指した後に足されるため上限に入らない:
        // ベース = 2s/40 + 5s − 1s = 4.05s だが，今使えるのは残 2s − 1s = 1s
        let clock = ClockParams {
            btime: Some(2_000),
            wtime: Some(2_000),
            binc: Some(5_000),
            winc: Some(5_000),
            ..ClockParams::default()
        };
        let b = allocate(&cfg(), &clock, Color::Black, ANY_PLY);
        assert_eq!(b.soft_ms, 1_000);
    }

    #[test]
    fn test_min_think_floor() {
        let clock = ClockParams {
            btime: Some(0),
            wtime: Some(0),
            byoyomi: Some(500),
            ..ClockParams::default()
        };
        // 500 − 1000 は負 → 最低思考時間 100ms
        let b = allocate(&cfg(), &clock, Color::Black, ANY_PLY);
        assert_eq!(b.soft_ms, 100);
    }

    #[test]
    fn test_hard_allows_extension_with_main_time() {
        // 主時間十分: hard = soft × 2 (ceiling 未達で延長余地あり)
        let clock = ClockParams {
            btime: Some(400_000),
            wtime: Some(400_000),
            byoyomi: Some(10_000),
            ..ClockParams::default()
        };
        let b = allocate(&cfg(), &clock, Color::Black, ANY_PLY);
        assert_eq!(b.soft_ms, 19_000);
        assert_eq!(b.hard_ms, 38_000);
    }

    #[test]
    fn test_hard_clamped_to_ceiling_in_byoyomi() {
        // 秒読みのみ: soft == hard == ceiling (延長余地なし)
        let clock = ClockParams {
            btime: Some(0),
            wtime: Some(0),
            byoyomi: Some(10_000),
            ..ClockParams::default()
        };
        let b = allocate(&cfg(), &clock, Color::Black, ANY_PLY);
        assert_eq!(b.soft_ms, 9_000);
        assert_eq!(b.hard_ms, 9_000);
    }

    #[test]
    fn test_should_stop_before_soft_continues() {
        let b = TimeBudget {
            soft_ms: 1_000,
            hard_ms: 2_000,
        };
        assert!(!should_stop(&b, 500, 100, 90, false));
    }

    #[test]
    fn test_should_stop_at_soft_when_stable() {
        let b = TimeBudget {
            soft_ms: 1_000,
            hard_ms: 2_000,
        };
        // best 200 vs second 100 → 安定 (200×2 ≥ 100×3) → soft で停止
        assert!(should_stop(&b, 1_000, 200, 100, false));
    }

    #[test]
    fn test_should_stop_extends_when_contested() {
        let b = TimeBudget {
            soft_ms: 1_000,
            hard_ms: 2_000,
        };
        // best 110 vs second 100 → 拮抗 (110×2 < 100×3) → soft でも延長 (継続)
        assert!(!should_stop(&b, 1_000, 110, 100, false));
        // hard 到達なら拮抗でも停止
        assert!(should_stop(&b, 2_000, 110, 100, false));
    }

    #[test]
    fn test_should_stop_on_proven() {
        let b = TimeBudget {
            soft_ms: 1_000,
            hard_ms: 2_000,
        };
        // root 確定は soft 未満でも即停止
        assert!(should_stop(&b, 10, 1, 0, true));
    }

    #[test]
    fn test_curve_weight_shape() {
        let c = TimeCurve::DEFAULT;
        // 頂点で peak，頂点から half_width 以上離れればその側の底
        assert_eq!(c.weight_permille(100), 2500);
        assert_eq!(c.weight_permille(45), 300);
        assert_eq!(c.weight_permille(1), 300);
        assert_eq!(c.weight_permille(155), 1200);
        assert_eq!(c.weight_permille(400), 1200);
        // 中間は線形 (頂点から半分の距離なら振幅の半分)
        assert_eq!(c.weight_permille(100 - 55 / 2), 300 + 2200 * 28 / 55);
        assert_eq!(c.weight_permille(100 + 55 / 2), 1200 + 1300 * 28 / 55);
        // 底が非対称なので山も非対称 — 終盤側が必ず厚い
        for d in 1..60 {
            assert!(
                c.weight_permille(100 + d) > c.weight_permille(100 - d),
                "ply 100±{d} で終盤側が序盤側を上回らない"
            );
        }
    }

    #[test]
    fn test_curve_default_shape_matches_the_measured_conclusion() {
        // A/B の結論をコードに固定する: 序盤は削ってよい / 変換期は厚くする．
        // 逆向き (序盤 ≥ 1.0 や 終盤 ≤ 1.0) に倒すと 50 局 34% の側へ戻る
        let c = TimeCurve::DEFAULT;
        const { assert!(TimeCurve::DEFAULT.opening_floor_permille < FLAT_PERMILLE) };
        const { assert!(TimeCurve::DEFAULT.endgame_floor_permille > FLAT_PERMILLE) };
        // 山は変換期 (優勢を勝ちに変える帯) にある
        assert!((90..=115).contains(&c.peak_ply), "peak_ply={}", c.peak_ply);
    }

    #[test]
    fn test_curve_is_on_by_default() {
        // 既定 on (A/B で一律配分より強い側に振れたため)．
        // 明示的に切ればカーブ導入前と同じ配分に戻ることも固定する
        let d = TimeStrategyConfig::default();
        assert!(d.curve().is_some());
        assert_eq!(d.curve_params, TimeCurve::DEFAULT);

        let clock = ClockParams {
            btime: Some(400_000),
            wtime: Some(400_000),
            byoyomi: Some(10_000),
            ..ClockParams::default()
        };
        for ply in [1, 30, 55, 80, 150] {
            let b = allocate(&cfg(), &clock, Color::Black, ply);
            assert_eq!(b.soft_ms, 19_000, "off にすれば ply {ply} で一律配分");
        }
    }

    #[test]
    fn test_curve_weights_only_discretionary_part() {
        // 秒読み 10s + 残 400s．裁量枠 400s/40 = 10s にのみ乗数が掛かる．
        // 頂点 (×2.5): 10s×2.5 + 10s − 1s = 34s
        // 序盤の底 (×0.3): 10s×0.3 + 10s − 1s = 12s
        let clock = ClockParams {
            btime: Some(400_000),
            wtime: Some(400_000),
            byoyomi: Some(10_000),
            ..ClockParams::default()
        };
        let cfg = cfg_curved(TimeCurve::DEFAULT);
        assert_eq!(allocate(&cfg, &clock, Color::Black, 100).soft_ms, 34_000);
        assert_eq!(allocate(&cfg, &clock, Color::Black, 1).soft_ms, 12_000);
    }

    #[test]
    fn test_curve_never_cuts_into_byoyomi_floor() {
        // 主時間を使い切った秒読み局面では，どの手数でも秒読み分が残る
        // (乗数は裁量枠にしか掛からないため)．
        let clock = ClockParams {
            btime: Some(0),
            wtime: Some(0),
            byoyomi: Some(10_000),
            ..ClockParams::default()
        };
        let cfg = cfg_curved(TimeCurve::DEFAULT);
        for ply in [1, 55, 120, 300] {
            let b = allocate(&cfg, &clock, Color::Black, ply);
            assert_eq!(b.soft_ms, 9_000, "ply {ply} で秒読みの床が削られた");
        }
    }

    #[test]
    fn test_curve_endgame_keeps_fischer_increment() {
        // フィッシャー 10s 加算．終盤の底 (×1.2) でも加算分は丸ごと残る:
        // 200s/40 × 1.2 + 10s − 1s = 15s
        let clock = ClockParams {
            btime: Some(200_000),
            wtime: Some(200_000),
            binc: Some(10_000),
            winc: Some(10_000),
            ..ClockParams::default()
        };
        let b = allocate(&cfg_curved(TimeCurve::DEFAULT), &clock, Color::Black, 160);
        assert_eq!(b.soft_ms, 15_000);
    }

    #[test]
    fn test_curve_gives_the_endgame_more_than_flat() {
        // 終盤の底は 1.0 を上回る (>flat)．中盤へ寄せてバンクを削った前の形は
        // ここが痩せて 50 局 34% になった — その逆を固定する
        let clock = ClockParams {
            btime: Some(120_000),
            wtime: Some(120_000),
            ..ClockParams::default()
        };
        let flat = allocate(&cfg(), &clock, Color::Black, 160);
        let curved = allocate(&cfg_curved(TimeCurve::DEFAULT), &clock, Color::Black, 160);
        assert!(
            curved.soft_ms > flat.soft_ms,
            "終盤 curved {} <= flat {}",
            curved.soft_ms,
            flat.soft_ms
        );
    }

    #[test]
    fn test_curve_respects_ceiling_and_min_think() {
        // 残り僅少ではカーブがあっても ceiling / min_think の保護が先に効く
        let clock = ClockParams {
            btime: Some(2_000),
            wtime: Some(2_000),
            binc: Some(5_000),
            winc: Some(5_000),
            ..ClockParams::default()
        };
        let b = allocate(&cfg_curved(TimeCurve::DEFAULT), &clock, Color::Black, 100);
        assert_eq!(b.soft_ms, 1_000);

        let broke = ClockParams {
            btime: Some(40_000),
            wtime: Some(40_000),
            ..ClockParams::default()
        };
        // 序盤の底 (×0.3) では 40s/40×0.3 − 1s が負 → 最低思考時間まで切り上げ
        let b = allocate(&cfg_curved(TimeCurve::DEFAULT), &broke, Color::Black, 1);
        assert_eq!(b.soft_ms, 100);
    }

    /// 残時間を実際に減らしながら 1 局分の soft 配分を追う
    /// (消費 = 思考 + 通信マージン．フィッシャーは指した後に加算)．
    fn simulate_soft(
        cfg: &TimeStrategyConfig,
        main_ms: u64,
        inc_ms: u64,
        plies: u64,
    ) -> Vec<(u64, u64)> {
        let mut remaining = main_ms;
        let mut out = Vec::new();
        let mut ply = 1;
        while ply <= plies {
            let clock = ClockParams {
                btime: Some(remaining),
                wtime: Some(remaining),
                binc: Some(inc_ms),
                winc: Some(inc_ms),
                ..ClockParams::default()
            };
            let b = allocate(cfg, &clock, Color::Black, ply);
            let consumed = (b.soft_ms + cfg.network_delay_ms).min(remaining);
            out.push((ply, b.soft_ms));
            remaining = remaining.saturating_sub(consumed) + inc_ms;
            ply += 2;
        }
        out
    }

    #[test]
    fn test_curve_moves_the_peak_into_the_conversion_phase() {
        // この機能の存在理由そのもの: 一律配分は `残時間 / horizon` の等比
        // 減衰なので**初手が最も厚い**．カーブはその山を変換期へ移す．
        // (300s 切れ負け / 180s + 5s フィッシャー，130 手で決着)
        for (main_ms, inc_ms) in [(300_000, 0), (180_000, 5_000)] {
            let flat = simulate_soft(&cfg(), main_ms, inc_ms, 130);
            let curved = simulate_soft(&cfg_curved(TimeCurve::DEFAULT), main_ms, inc_ms, 130);

            assert_eq!(
                flat.iter().max_by_key(|(_, ms)| *ms).unwrap().0,
                1,
                "一律配分の最厚手が初手でない (前提が変わった)"
            );
            // **実効ピークは `peak_ply` より手前に来る**: 重みが上がる一方で
            // 掛ける先の残り時間が減るので，積が先に頂点を迎える．GPU 実測でも
            // 配分比の最大は ply 61-90 帯だった (peak_ply = 100 に対して)．
            // ここを `peak_ply` ちょうどで固定すると，パラメータを動かすたびに
            // 落ちる脆いテストになる
            let peak_ply = curved.iter().max_by_key(|(_, ms)| *ms).unwrap().0;
            assert!(
                (70..=115).contains(&peak_ply),
                "カーブの山が変換期に無い: ply {peak_ply}"
            );
            // 中盤へ寄せるのであって節約ではない — 総思考は減らない
            let total = |v: &[(u64, u64)]| v.iter().map(|(_, ms)| ms).sum::<u64>();
            assert!(
                total(&curved) > total(&flat),
                "カーブ有効で総思考時間が減った"
            );
        }
    }

    #[test]
    fn test_curve_params_survive_toggle_order() {
        // setoption の到着順に依存しない: パラメータを先に入れても後から
        // 有効化して効く / 無効化してもパラメータは残る
        let mut c = cfg();
        c.curve_params.peak_ply = 70;
        assert!(c.curve().is_none());
        c.curve_enabled = true;
        assert_eq!(c.curve().unwrap().peak_ply, 70);
        c.curve_enabled = false;
        assert_eq!(c.curve_params.peak_ply, 70);
    }

    #[test]
    fn test_curve_degenerate_peak_below_floor_is_flat() {
        // peak が底を下回る退化設定は山が消えて一律 底 (安全側)
        let c = TimeCurve {
            peak_ply: 55,
            half_width_ply: 35,
            peak_permille: 500,
            opening_floor_permille: 1000,
            endgame_floor_permille: 1200,
        };
        assert_eq!(c.weight_permille(1), 1000);
        assert_eq!(c.weight_permille(54), 1000);
        assert_eq!(c.weight_permille(55), 1200);
        assert_eq!(c.weight_permille(200), 1200);
    }

    #[test]
    fn test_curve_zero_half_width_is_flat_floor_off_peak() {
        // half_width 0 は 1 として扱う (0 除算なし)
        let c = TimeCurve {
            half_width_ply: 0,
            ..TimeCurve::DEFAULT
        };
        assert_eq!(c.weight_permille(100), 2500);
        assert_eq!(c.weight_permille(99), 300);
        assert_eq!(c.weight_permille(101), 1200);
    }
}
