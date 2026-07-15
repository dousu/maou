//! `HcpeRecord` 列 → Arrow RecordBatch 組み立て．
//!
//! スキーマは `maou_io::schema::hcpe_schema` (現行 Polars 出力と bit-exact に
//! 一致: large_binary / large_utf8 / large_list<uint16> / date32，全 nullable) を
//! 単一の真実とする．

use std::sync::Arc;

use arrow::array::{ArrayRef, RecordBatch};
use arrow::array::{
    Date32Array, Int16Array, Int8Array, LargeBinaryBuilder, LargeListBuilder, LargeStringBuilder,
    UInt16Builder,
};
use maou_io::schema::hcpe_schema;
use maou_io::MaouIOError;

use crate::pipeline::HcpeRecord;

/// `HcpeRecord` の列を HCPE スキーマの RecordBatch に組み立てる．
pub fn build_record_batch(records: &[HcpeRecord]) -> Result<RecordBatch, MaouIOError> {
    let n = records.len();

    let mut hcp_b = LargeBinaryBuilder::with_capacity(n, n * 32);
    for r in records {
        hcp_b.append_value(r.hcp);
    }

    let eval = Int16Array::from_iter_values(records.iter().map(|r| r.eval));
    let best_move16 = Int16Array::from_iter_values(records.iter().map(|r| r.best_move16));
    let game_result = Int8Array::from_iter_values(records.iter().map(|r| r.game_result));

    let mut id_b = LargeStringBuilder::new();
    for r in records {
        id_b.append_value(&r.id);
    }

    let partitioning_key = Date32Array::from_iter(records.iter().map(|r| r.partitioning_key));

    let mut ratings_b = LargeListBuilder::new(UInt16Builder::new());
    for r in records {
        for &v in &r.ratings {
            ratings_b.values().append_value(v);
        }
        ratings_b.append(true);
    }

    let mut endgame_b = LargeStringBuilder::new();
    for r in records {
        match &r.endgame_status {
            Some(s) => endgame_b.append_value(s),
            None => endgame_b.append_null(),
        }
    }

    let moves = Int16Array::from_iter_values(records.iter().map(|r| r.moves));

    let columns: Vec<ArrayRef> = vec![
        Arc::new(hcp_b.finish()),
        Arc::new(eval),
        Arc::new(best_move16),
        Arc::new(game_result),
        Arc::new(id_b.finish()),
        Arc::new(partitioning_key),
        Arc::new(ratings_b.finish()),
        Arc::new(endgame_b.finish()),
        Arc::new(moves),
    ];

    RecordBatch::try_new(Arc::new(hcpe_schema()), columns).map_err(MaouIOError::ArrowError)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> HcpeRecord {
        HcpeRecord {
            hcp: [7u8; 32],
            eval: -123,
            best_move16: 7739,
            game_result: 1,
            id: "test.hcpe_0".to_string(),
            partitioning_key: Some(20097),
            ratings: vec![3495, 3300],
            endgame_status: Some("%TORYO".to_string()),
            moves: 163,
        }
    }

    #[test]
    fn test_build_matches_schema() {
        let batch = build_record_batch(&[sample()]).expect("build ok");
        assert_eq!(batch.schema().as_ref(), &hcpe_schema());
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.num_columns(), 9);
    }

    #[test]
    fn test_empty_ratings_and_null_fields() {
        let mut rec = sample();
        rec.ratings = vec![]; // KIF: 空リスト
        rec.partitioning_key = None; // 開始日時なし
        rec.endgame_status = None; // まで行なし
        let batch = build_record_batch(&[rec]).expect("build ok");
        assert_eq!(batch.num_rows(), 1);
    }
}
