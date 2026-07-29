//! 局面評価の抽象境界．
//!
//! MCTS 本体は評価器の実装 (NN 推論・mock 等) を知らない．
//! 実運用では ONNX モデルの I/O 契約 (入力: board int32 (B,9,9) + hand f32 (B,14)，
//! 出力: policy (B,1496) logits + value (B,1) logit) への適合 — 特徴量エンコード，
//! move→policy label 変換，logit→勝率変換 — はすべて evaluator 実装側の責務とし，
//! 探索側には「合法手ごとの事前確率 + 手番側勝率」だけを渡す．

use maou_shogi::board::Board;
use maou_shogi::movegen::generate_legal_moves;
use maou_shogi::moves::Move;

/// 平手初期局面の SFEN．
pub const STARTPOS_SFEN: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

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

/// 平手初期局面を 1 回評価して初回推論の固定費を前払いする (warmup)．
///
/// ONNX Runtime の TensorRT EP は**最初の推論時に**エンジンをビルドする
/// (キャッシュがなければ数十秒かかる)．探索の時間予算は `go` 受領時刻を
/// 起点に張るため，この固定費を予算の内側で払うと短い予算では playout が
/// 0 になる．そこで**予算を張る前に**本関数で 1 回推論を通しておく．
///
/// 呼び出し箇所は「1 手ぶんの予算の外側」— USI は `isready`，CSA は対局
/// 開始前，CLI (`maou search`) と `SearchEngine` は探索開始前．
///
/// 結果は捨てる (副作用としての初期化だけが目的)．
pub fn warmup<E: Evaluator + ?Sized>(evaluator: &E) {
    let mut board = Board::empty();
    board
        .set_sfen(STARTPOS_SFEN)
        .expect("STARTPOS_SFEN は正当な SFEN");
    let moves = generate_legal_moves(&mut board.clone());
    let _ = evaluator.evaluate_batch(&[EvalItem { board, moves }]);
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
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    fn startpos_item() -> EvalItem {
        let mut board = Board::empty();
        board
            .set_sfen(STARTPOS_SFEN)
            .expect("startpos は正当な SFEN");
        let moves = generate_legal_moves(&mut board);
        EvalItem { board, moves }
    }

    /// 呼び出し回数とバッチ内容を記録する評価器 (warmup の検証用)．
    struct CountingEvaluator {
        calls: AtomicUsize,
        last_batch: Mutex<Vec<usize>>,
    }

    impl Evaluator for CountingEvaluator {
        fn evaluate_batch(&self, items: &[EvalItem]) -> Vec<EvalResult> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            *self.last_batch.lock().expect("poisoned") =
                items.iter().map(|i| i.moves.len()).collect();
            items
                .iter()
                .map(|i| EvalResult {
                    priors: vec![0.0; i.moves.len()],
                    value: 0.5,
                })
                .collect()
        }
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

    #[test]
    fn test_warmup_runs_exactly_one_inference() {
        let ev = CountingEvaluator {
            calls: AtomicUsize::new(0),
            last_batch: Mutex::new(Vec::new()),
        };
        warmup(&ev);
        assert_eq!(
            ev.calls.load(Ordering::SeqCst),
            1,
            "warmup は推論をちょうど 1 回だけ走らせる"
        );
    }

    #[test]
    fn test_warmup_evaluates_startpos_with_all_legal_moves() {
        // 平手初期局面の合法手は 30 手．TensorRT のエンジンは shape ごとに
        // ビルドされるため，warmup が実運用と同じ 1 局面バッチを通すことが重要
        let ev = CountingEvaluator {
            calls: AtomicUsize::new(0),
            last_batch: Mutex::new(Vec::new()),
        };
        warmup(&ev);
        let batch = ev.last_batch.lock().expect("poisoned").clone();
        assert_eq!(batch, vec![30], "1 局面 × 合法手 30 手のバッチ");
    }

    #[test]
    fn test_startpos_sfen_parses() {
        let mut board = Board::empty();
        board
            .set_sfen(STARTPOS_SFEN)
            .expect("STARTPOS_SFEN は正当な SFEN");
        assert_eq!(generate_legal_moves(&mut board).len(), 30);
    }
}
