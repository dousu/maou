//! ONNX Runtime による NN 評価器 (`onnx` feature)．
//!
//! モデルの I/O 契約 (Python 側 `model_io.py` の torch.onnx.export と同一):
//!
//! - 入力 `board`: int32 (B, 9, 9) — 手番視点の駒 ID 盤面 ([`crate::feature`])
//! - 入力 `hand`: float32 (B, 14) — 手番側先頭の持ち駒枚数
//! - 出力 `policy`: float32 (B, 1496) — 指し手ラベル ([`crate::label`]) の logits
//! - 出力 `value`: float32 (B, 1) — 手番側勝率の logit (sigmoid で勝率化)
//!
//! priors は合法手のラベルに対応する logits だけを取り出して softmax を取る
//! (非合法手ラベルは分布から除外される)．
//!
//! # スレッド安全性
//!
//! ort の `Session::run` は `&mut self` を要求するため，現状は `Mutex` で
//! 直列化している (PoC)．GPU 推論が律速の場合は影響が小さいが，
//! 複数スレッドの CPU 前処理と推論を重ねる最適化は今後の課題．

use std::path::Path;
use std::sync::Mutex;

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

use crate::evaluator::{EvalItem, EvalResult, Evaluator};
use crate::feature::{encode_board, encode_hand, BOARD_FEATURE_LEN, HAND_FEATURE_LEN};
use crate::label::{move_label, MOVE_LABELS_NUM};

/// ONNX evaluator の設定．
#[derive(Clone, Debug)]
pub struct OnnxOptions {
    /// ONNX Runtime の intra-op スレッド数．
    pub intra_threads: usize,
    /// CUDA Execution Provider を使う (`onnx-cuda` feature が必要)．
    /// 失敗時はエラーにする (静かな CPU フォールバックで計測を誤らせない)．
    pub use_cuda: bool,
}

impl Default for OnnxOptions {
    fn default() -> Self {
        OnnxOptions {
            intra_threads: 1,
            use_cuda: false,
        }
    }
}

/// ONNX Runtime による [`Evaluator`] 実装．
pub struct OnnxEvaluator {
    session: Mutex<Session>,
}

impl OnnxEvaluator {
    /// ONNX モデルファイルから評価器を作る．
    pub fn from_file(path: impl AsRef<Path>, options: &OnnxOptions) -> ort::Result<Self> {
        let builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(options.intra_threads.max(1))?;
        let builder = if options.use_cuda {
            #[cfg(feature = "onnx-cuda")]
            {
                use ort::execution_providers::CUDAExecutionProvider;
                builder.with_execution_providers([CUDAExecutionProvider::default()
                    .build()
                    .error_on_failure()])?
            }
            #[cfg(not(feature = "onnx-cuda"))]
            {
                return Err(ort::Error::new(
                    "use_cuda には onnx-cuda feature でのビルドが必要",
                ));
            }
        } else {
            builder
        };
        let session = builder.commit_from_file(path)?;
        Ok(OnnxEvaluator {
            session: Mutex::new(session),
        })
    }
}

/// 数値安定な softmax (最大値シフト)．
fn softmax_in_place(xs: &mut [f32]) {
    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in xs.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    if sum > 0.0 {
        for x in xs.iter_mut() {
            *x /= sum;
        }
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl Evaluator for OnnxEvaluator {
    fn evaluate_batch(&self, items: &[EvalItem]) -> Vec<EvalResult> {
        let batch = items.len();
        let mut board_data = vec![0i32; batch * BOARD_FEATURE_LEN];
        let mut hand_data = vec![0f32; batch * HAND_FEATURE_LEN];
        for (i, item) in items.iter().enumerate() {
            let board_out: &mut [i32; BOARD_FEATURE_LEN] = (&mut board_data
                [i * BOARD_FEATURE_LEN..(i + 1) * BOARD_FEATURE_LEN])
                .try_into()
                .expect("スライス長は BOARD_FEATURE_LEN");
            encode_board(&item.board, board_out);
            let hand_out: &mut [f32; HAND_FEATURE_LEN] = (&mut hand_data
                [i * HAND_FEATURE_LEN..(i + 1) * HAND_FEATURE_LEN])
                .try_into()
                .expect("スライス長は HAND_FEATURE_LEN");
            encode_hand(&item.board, hand_out);
        }

        let board_tensor = Tensor::from_array(([batch, 9, 9], board_data))
            .expect("board テンソル作成は失敗しない");
        let hand_tensor = Tensor::from_array(([batch, HAND_FEATURE_LEN], hand_data))
            .expect("hand テンソル作成は失敗しない");

        let mut session = self.session.lock().expect("poisoned session mutex");
        let outputs = session
            .run(ort::inputs![
                "board" => board_tensor,
                "hand" => hand_tensor,
            ])
            .expect("ONNX 推論は失敗しない (モデル I/O 契約の不一致はここで落ちる)");

        let (policy_shape, policy) = outputs["policy"]
            .try_extract_tensor::<f32>()
            .expect("policy は f32 テンソル");
        assert_eq!(
            policy_shape.as_ref(),
            &[batch as i64, MOVE_LABELS_NUM as i64],
            "policy 出力形状がモデル契約と一致すること"
        );
        let (value_shape, value) = outputs["value"]
            .try_extract_tensor::<f32>()
            .expect("value は f32 テンソル");
        assert_eq!(value_shape.as_ref()[0], batch as i64);

        items
            .iter()
            .enumerate()
            .map(|(i, item)| {
                let logits = &policy[i * MOVE_LABELS_NUM..(i + 1) * MOVE_LABELS_NUM];
                let turn = item.board.turn();
                let mut priors: Vec<f32> = item
                    .moves
                    .iter()
                    .map(|&m| logits[move_label(turn, m) as usize])
                    .collect();
                softmax_in_place(&mut priors);
                EvalResult {
                    priors,
                    value: sigmoid(value[i]),
                }
            })
            .collect()
    }
}
