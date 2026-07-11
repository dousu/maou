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
//! # Execution Provider
//!
//! 実行時フラグと feature の二段 opt-in (デフォルトビルドは CPU のみ):
//!
//! - CUDA: feature `onnx-cuda` + [`OnnxOptions::use_cuda`]
//! - TensorRT: feature `onnx-tensorrt` + [`OnnxOptions::use_tensorrt`]．
//!   FP16 + エンジンキャッシュを有効にする．**可変バッチのままだと shape ごとに
//!   エンジンビルドが走る**ため，[`OnnxOptions::pad_to`] で shape を固定すること
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
    /// TensorRT Execution Provider を使う (`onnx-tensorrt` feature が必要)．
    /// FP16 とエンジンキャッシュを有効にする．TensorRT が扱えないノードは
    /// 後続の EP (use_cuda 併用時は CUDA) にフォールバックする．
    pub use_tensorrt: bool,
    /// TensorRT エンジンキャッシュの保存先 (未指定時はカレントの `trt_cache/`)．
    /// 初回実行のエンジンビルドは数分かかるため，キャッシュ必須．
    pub trt_engine_cache_dir: Option<String>,
    /// バッチを常にこのサイズまでゼロ局面で padding する．
    /// TensorRT は入力 shape ごとにエンジンを選択/構築するため，可変バッチの
    /// まま渡すと shape 数だけビルドが走る．探索側の batch_size と同値を
    /// 指定して shape を固定するのが推奨 (余剰スロットの出力は捨てる)．
    pub pad_to: Option<usize>,
}

impl Default for OnnxOptions {
    fn default() -> Self {
        OnnxOptions {
            intra_threads: 1,
            use_cuda: false,
            use_tensorrt: false,
            trt_engine_cache_dir: None,
            pad_to: None,
        }
    }
}

/// ONNX Runtime による [`Evaluator`] 実装．
pub struct OnnxEvaluator {
    session: Mutex<Session>,
    pad_to: Option<usize>,
}

impl OnnxEvaluator {
    /// ONNX モデルファイルから評価器を作る．
    pub fn from_file(path: impl AsRef<Path>, options: &OnnxOptions) -> ort::Result<Self> {
        #[cfg(not(feature = "onnx-cuda"))]
        if options.use_cuda {
            return Err(ort::Error::new(
                "use_cuda には onnx-cuda feature でのビルドが必要",
            ));
        }
        #[cfg(not(feature = "onnx-tensorrt"))]
        if options.use_tensorrt {
            return Err(ort::Error::new(
                "use_tensorrt には onnx-tensorrt feature でのビルドが必要",
            ));
        }

        #[allow(unused_mut)]
        let mut builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(options.intra_threads.max(1))?;

        // EP は登録順に試行される: TensorRT → CUDA → (暗黙の) CPU．
        // いずれも初期化失敗は即エラー (静かなフォールバックで計測を誤らせない)
        #[cfg(feature = "onnx-tensorrt")]
        if options.use_tensorrt {
            use ort::execution_providers::TensorRTExecutionProvider;
            let cache_dir = options
                .trt_engine_cache_dir
                .clone()
                .unwrap_or_else(|| "trt_cache".to_string());
            let ep = TensorRTExecutionProvider::default()
                .with_fp16(true)
                .with_engine_cache(true)
                .with_engine_cache_path(cache_dir);
            builder = builder.with_execution_providers([ep.build().error_on_failure()])?;
        }
        #[cfg(feature = "onnx-cuda")]
        if options.use_cuda {
            use ort::execution_providers::CUDAExecutionProvider;
            builder = builder.with_execution_providers([CUDAExecutionProvider::default()
                .build()
                .error_on_failure()])?;
        }

        let session = builder.commit_from_file(path)?;
        Ok(OnnxEvaluator {
            session: Mutex::new(session),
            pad_to: options.pad_to,
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
        // pad_to 指定時は shape を固定する (余剰スロットは空盤面のダミー入力．
        // ゼロ初期化がそのまま EMPTY 盤面 + 持ち駒なしに一致する)
        let padded = self.pad_to.map_or(batch, |p| p.max(batch));
        let mut board_data = vec![0i32; padded * BOARD_FEATURE_LEN];
        let mut hand_data = vec![0f32; padded * HAND_FEATURE_LEN];
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

        let board_tensor = Tensor::from_array(([padded, 9, 9], board_data))
            .expect("board テンソル作成は失敗しない");
        let hand_tensor = Tensor::from_array(([padded, HAND_FEATURE_LEN], hand_data))
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
            &[padded as i64, MOVE_LABELS_NUM as i64],
            "policy 出力形状がモデル契約と一致すること"
        );
        let (value_shape, value) = outputs["value"]
            .try_extract_tensor::<f32>()
            .expect("value は f32 テンソル");
        assert_eq!(value_shape.as_ref()[0], padded as i64);

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
