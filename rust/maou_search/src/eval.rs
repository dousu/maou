//! 勝率と評価値スコアの相互変換 (表示用)．
//!
//! 評価値スコアは Ponanza 由来の係数 600 を用いた `eval = 600 * ln(w / (1 - w))`
//! (逆関数は `w = sigmoid(eval / 600)`)．探索の意思決定はすべて勝率 [0, 1] で
//! 行い，本モジュールの変換は USI の `score cp`・`maou search`・`maou evaluate`
//! の**表示にのみ**使う．

/// Ponanza 由来の評価値スケーリング係数．
const EVAL_SCALE: f64 = 600.0;

/// 勝率が 0 / 1 のとき評価値が発散するため，この量だけ内側にクランプする．
///
/// 飽和値は `600 * ln(1e12) ≒ ±16570`．
const WINRATE_CLAMP: f64 = 1e-12;

/// 手番側の勝率 [0, 1] を評価値スコアに変換する．
///
/// `winrate` が 0 / 1 でも発散しないよう [`WINRATE_CLAMP`] だけ内側に丸める．
///
/// # 例
///
/// ```
/// use maou_search::eval::winrate_to_eval;
/// assert_eq!(winrate_to_eval(0.5), 0.0);
/// assert!(winrate_to_eval(0.9) > 0.0);
/// assert!(winrate_to_eval(0.1) < 0.0);
/// ```
pub fn winrate_to_eval(winrate: f64) -> f64 {
    let w = winrate.clamp(WINRATE_CLAMP, 1.0 - WINRATE_CLAMP);
    EVAL_SCALE * (w / (1.0 - w)).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_winrate_to_eval_midpoint_is_zero() {
        assert_eq!(winrate_to_eval(0.5), 0.0);
    }

    #[test]
    fn test_winrate_to_eval_is_antisymmetric() {
        // logit は 0.5 を中心に奇関数なので符号だけが反転する
        for w in [0.1, 0.25, 0.4, 0.75, 0.9] {
            let lhs = winrate_to_eval(w);
            let rhs = winrate_to_eval(1.0 - w);
            assert!(
                (lhs + rhs).abs() < 1e-9,
                "winrate_to_eval({w}) = {lhs} と winrate_to_eval({}) = {rhs} が対称でない",
                1.0 - w
            );
        }
    }

    #[test]
    fn test_winrate_to_eval_matches_ponanza_scale() {
        // sigmoid(600 / 600) = 0.7310585786... が eval = 600 に対応する
        let winrate = 1.0 / (1.0 + (-1.0f64).exp());
        assert!((winrate_to_eval(winrate) - 600.0).abs() < 1e-9);
    }

    #[test]
    fn test_winrate_to_eval_saturates_at_certainty() {
        // 0 / 1 でも発散せず，符号だけが反対の有限値になる．
        // クランプ端では `1.0 - 1e-12` の桁落ちで約 0.05 の非対称が残るため
        // 許容幅を持たせる (置き換え元の Python 実装も同じ丸めをしていた)．
        let win = winrate_to_eval(1.0);
        let loss = winrate_to_eval(0.0);
        assert!(win.is_finite() && loss.is_finite());
        assert!(
            (win + loss).abs() < 0.1,
            "飽和値が符号反転していない: {win} / {loss}"
        );
        assert!(win > 16000.0 && win < 17000.0, "飽和値が想定範囲外: {win}");
    }

    #[test]
    fn test_winrate_to_eval_is_monotonic() {
        let mut prev = f64::NEG_INFINITY;
        for i in 0..=100 {
            let v = winrate_to_eval(f64::from(i) / 100.0);
            assert!(v > prev, "単調増加でない: {v} <= {prev}");
            prev = v;
        }
    }
}
