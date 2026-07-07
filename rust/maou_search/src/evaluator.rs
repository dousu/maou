//! 局面評価の抽象境界．
//!
//! MCTS 本体は評価器の実装 (NN 推論・mock 等) を知らない．
//! 実運用では ONNX モデルの I/O 契約 (入力: board int32 (B,9,9) + hand f32 (B,14)，
//! 出力: policy (B,1496) logits + value (B,1) logit) への適合 — 特徴量エンコード，
//! move→policy label 変換，logit→勝率変換 — はすべて evaluator 実装側の責務とし，
//! 探索側には「合法手ごとの事前確率 + 手番側勝率」だけを渡す．

use maou_shogi::board::Board;
use maou_shogi::moves::Move;

/// 評価リクエスト 1 件 (葉局面とその合法手)．
pub struct EvalItem {
    /// 評価対象の局面．
    pub board: Board,
    /// `board` における合法手．[`EvalResult::priors`] はこの並び順に対応する．
    pub moves: Vec<Move>,
}

/// 評価結果 1 件．
pub struct EvalResult {
    /// 合法手それぞれの事前確率．[`EvalItem::moves`] と同数・同順で，総和はほぼ 1．
    pub priors: Vec<f32>,
    /// 手番側から見た勝率 [0, 1]．
    pub value: f32,
}

/// バッチ評価器の trait．
///
/// 複数の探索スレッドから並行に呼ばれるため `Send + Sync` を要求する．
/// 実装は items と同数・同順の結果を返さなければならない．
pub trait Evaluator: Send + Sync {
    /// 複数局面をまとめて評価する．
    fn evaluate_batch(&self, items: &[EvalItem]) -> Vec<EvalResult>;
}

/// splitmix64 (決定論的擬似乱数生成)．
fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// u64 乱数を [0, 1) の f64 に写す (上位 53 bit を使用)．
fn to_unit_f64(r: u64) -> f64 {
    (r >> 11) as f64 / (1u64 << 53) as f64
}

/// 決定論的な擬似乱数評価器 (ベンチ・テスト用)．
///
/// 局面の zobrist ハッシュと seed から splitmix64 で priors / value を生成する．
/// 同一局面には常に同じ評価を返すため，「局面→評価」の対応は実行間・
/// スレッド数間で再現可能 (探索結果自体はスレッドの競合順で変わり得る)．
pub struct MockEvaluator {
    seed: u64,
}

impl MockEvaluator {
    /// seed を指定して生成する．
    pub fn new(seed: u64) -> Self {
        MockEvaluator { seed }
    }
}

impl Default for MockEvaluator {
    fn default() -> Self {
        MockEvaluator::new(0)
    }
}

impl Evaluator for MockEvaluator {
    fn evaluate_batch(&self, items: &[EvalItem]) -> Vec<EvalResult> {
        items
            .iter()
            .map(|item| {
                let h = splitmix64(item.board.hash() ^ self.seed);
                let mut priors: Vec<f32> = item
                    .moves
                    .iter()
                    .map(|m| {
                        let r = splitmix64(h ^ u64::from(m.raw_u32()));
                        // 1..=2^16 の正の重み (ゼロ確率を作らない)
                        ((r & 0xFFFF) + 1) as f32
                    })
                    .collect();
                let sum: f32 = priors.iter().sum();
                for p in &mut priors {
                    *p /= sum;
                }
                let value = to_unit_f64(splitmix64(h ^ 0xA5A5_A5A5_A5A5_A5A5)) as f32;
                EvalResult { priors, value }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use maou_shogi::movegen::generate_legal_moves;

    const STARTPOS: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

    fn startpos_item() -> EvalItem {
        let mut board = Board::empty();
        board.set_sfen(STARTPOS).expect("startpos は正当な SFEN");
        let moves = generate_legal_moves(&mut board);
        EvalItem { board, moves }
    }

    #[test]
    fn test_mock_priors_normalized() {
        let item = startpos_item();
        let n_moves = item.moves.len();
        let results = MockEvaluator::new(42).evaluate_batch(&[item]);
        assert_eq!(results.len(), 1);
        let r = &results[0];
        assert_eq!(r.priors.len(), n_moves);
        let sum: f32 = r.priors.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "priors 総和 = {sum}");
        assert!(r.priors.iter().all(|&p| p > 0.0));
        assert!((0.0..=1.0).contains(&r.value));
    }

    #[test]
    fn test_mock_deterministic() {
        let ev = MockEvaluator::new(7);
        let a = ev.evaluate_batch(&[startpos_item()]);
        let b = ev.evaluate_batch(&[startpos_item()]);
        assert_eq!(a[0].priors, b[0].priors);
        assert_eq!(a[0].value, b[0].value);
    }
}
