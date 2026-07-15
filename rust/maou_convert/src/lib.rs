//! 棋譜 (CSA / KIF) → HCPE 一括変換パイプライン．
//!
//! Python 実装 `src/maou/app/converter/hcpe_converter.py` の
//! `_process_single_file` + `game_filter` + `convert` (ProcessPoolExecutor) を
//! Rust に移植したもの．maou_shogi (パーサ + 局面再生) と maou_io (Arrow IPC
//! 出力) を合成し，ファイル直読み (UTF-8→cp932 fallback)・フィルタ・
//! partitioningKey 付与・RecordBatch 組み立て・rayon 並列を担う．
//!
//! maou_shogi は arrow 非依存，maou_io は shogi 非依存という依存分離を保つため
//! 両者を合成する層として独立 crate にしている (bindings-only の maou_rust とは
//! 別に，単体テスト可能にする目的も兼ねる)．
//!
//! 移植の正しさは HCPE 変換 golden fixture
//! (`tests/maou/app/converter/resources/golden/`) との bit-exact 一致で検証．

mod batch;
mod date;
mod decode;
mod pipeline;

pub use batch::build_record_batch;
pub use pipeline::{
    convert_content, convert_files, ConvertOptions, ConvertOutcome, HcpeRecord, InputFormat,
    PipelineError,
};
