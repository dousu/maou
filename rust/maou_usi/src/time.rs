//! 時間管理 (TimeStrategy) — 持ち時間 → 1 手予算の変換レイヤー．
//!
//! 「持ち時間の消費計画は別レイヤー — 1 局面探索は与えられた予算内まで」
//! (docs/design/usi-engine/index.md §8.1) の上位レイヤー実装．探索側は
//! ここが決めた予算を消費するだけで時間配分の知識を持たない．
//!
//! M1 は固定 horizon の簡易版 (soft == hard)．想定残り手数カーブ・
//! ベスト不安定時の延長は M2 (設計 §8.1)．

use maou_shogi::types::Color;

use crate::protocol::ClockParams;

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

/// 自分の手番の残り時間・加算を取り出して 1 手予算を計算する．
///
/// 三態 (秒読み / フィッシャー / 切れ負け) を扱う:
/// - 秒読み: `残時間 / horizon + byoyomi − margin`
/// - フィッシャー: `残時間 / horizon + inc − margin`
/// - 切れ負け: `残時間 / horizon − margin` (安全側)
///
/// いずれも「今使える時間の上限 (残時間 + byoyomi − margin)」でクランプし，
/// `min_think_ms` を下回らない．
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

    let ms = budget.min(ceiling).max(cfg.min_think_ms);
    TimeBudget {
        soft_ms: ms,
        hard_ms: ms,
    }
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
}
