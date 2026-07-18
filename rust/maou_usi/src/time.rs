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

/// 時間戦略の設定 (USI オプション / CLI から与える)．
#[derive(Clone, Debug)]
pub struct TimeStrategyConfig {
    /// 通信・プロセスオーバーヘッドのマージン (ミリ秒)．GUI/サーバは
    /// 相手の指し手送信から bestmove 受信までを消費時間として計測するため，
    /// 探索予算はこの分だけ短くする．
    pub network_delay_ms: u64,
    /// 最低思考時間 (ミリ秒)．マージン控除後もこれを下回らない．
    pub min_think_ms: u64,
    /// 持ち時間を配分する想定残り手数 (M1 は固定．M2 で手数カーブ化)．
    pub horizon_moves: u64,
}

impl Default for TimeStrategyConfig {
    fn default() -> TimeStrategyConfig {
        TimeStrategyConfig {
            network_delay_ms: 1000,
            min_think_ms: 100,
            horizon_moves: 40,
        }
    }
}

/// 自分の手番の残り時間・加算を取り出して 1 手予算 (soft/hard) を計算する．
///
/// 三態 (秒読み / フィッシャー / 切れ負け) を扱う:
/// - 秒読み: `残時間 / horizon + byoyomi − margin`
/// - フィッシャー: `残時間 / horizon + inc − margin`
/// - 切れ負け: `残時間 / horizon − margin` (安全側)
///
/// - `soft_ms` = 通常打ち切り目標 (上記 base 配分)．
/// - `hard_ms` = 延長上限 = `soft × EXT_NUM/EXT_DEN`．ただし今使える時間の
///   絶対上限 ceiling (残時間 + 秒読み − マージン) を超えない．
///
/// いずれも `min_think_ms` を下回らない．ceiling に張り付く局面 (秒読みのみ・
/// 残り僅少) では `soft == hard` になり延長余地はない (安全側)．
pub fn allocate(cfg: &TimeStrategyConfig, clock: &ClockParams, my_color: Color) -> TimeBudget {
    let (my_time, my_inc) = match my_color {
        Color::Black => (clock.btime.unwrap_or(0), clock.binc.unwrap_or(0)),
        Color::White => (clock.wtime.unwrap_or(0), clock.winc.unwrap_or(0)),
    };
    let byoyomi = clock.byoyomi.unwrap_or(0);
    let horizon = cfg.horizon_moves.max(1);

    // 1 手のベース配分 + 毎手戻ってくる時間 (秒読み or フィッシャー加算)
    let base = my_time / horizon + byoyomi + my_inc;
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

    fn cfg() -> TimeStrategyConfig {
        TimeStrategyConfig {
            network_delay_ms: 1000,
            min_think_ms: 100,
            horizon_moves: 40,
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
        let b = allocate(&cfg(), &clock, Color::Black);
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
        let b = allocate(&cfg(), &clock, Color::Black);
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
        let black = allocate(&cfg(), &clock, Color::Black);
        let white = allocate(&cfg(), &clock, Color::White);
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
        let b = allocate(&cfg(), &clock, Color::Black);
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
        let b = allocate(&cfg(), &clock, Color::Black);
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
        let b = allocate(&cfg(), &clock, Color::Black);
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
        let b = allocate(&cfg(), &clock, Color::Black);
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
        let b = allocate(&cfg(), &clock, Color::Black);
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
}
