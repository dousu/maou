use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;

/// HCPE schema (game records)．
///
/// HCPEフォーマットは棋譜データを効率的に保存するための構造．
/// 各レコードは1つの局面とその評価値，指し手，対局情報を含む．
///
/// 型は現行の Polars 由来出力 (`domain/data/schema.py` の
/// `get_hcpe_polars_schema` → Arrow) と bit-exact に一致させている:
/// large_binary / large_utf8 / large_list<uint16> と全フィールド nullable．
/// これにより Rust 側 (`maou_convert`) で組み立てた RecordBatch が
/// Polars 出力の feather と同一 schema になる (parity gate)．
/// `ratings` は可変長 (KIF はレーティング欠損で空リスト，CSA は 2 要素)．
pub fn hcpe_schema() -> Schema {
    Schema::new(vec![
        Field::new("hcp", DataType::LargeBinary, true),
        Field::new("eval", DataType::Int16, true),
        Field::new("bestMove16", DataType::Int16, true),
        Field::new("gameResult", DataType::Int8, true),
        Field::new("id", DataType::LargeUtf8, true),
        Field::new("partitioningKey", DataType::Date32, true),
        Field::new(
            "ratings",
            DataType::LargeList(Arc::new(Field::new("item", DataType::UInt16, true))),
            true,
        ),
        Field::new("endgameStatus", DataType::LargeUtf8, true),
        Field::new("moves", DataType::Int16, true),
    ])
}

/// Preprocessing schema (training features)．
///
/// 前処理済みデータスキーマ．ニューラルネットワーク学習用の特徴量を含む．
/// boardIdPositions: 9x9盤面の駒配置
/// piecesInHand: 持ち駒情報（14要素）
/// moveLabel: 指し手ラベル（1496要素）
/// resultValue: 対局結果値（0.0-1.0）
pub fn preprocessing_schema() -> Schema {
    Schema::new(vec![
        Field::new("id", DataType::UInt64, false),
        Field::new(
            "boardIdPositions",
            DataType::FixedSizeList(
                Arc::new(Field::new(
                    "row",
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::UInt8, false)),
                        9,
                    ),
                    false,
                )),
                9,
            ),
            false,
        ),
        Field::new(
            "piecesInHand",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::UInt8, false)), 14),
            false,
        ),
        Field::new(
            "moveLabel",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 1496),
            false,
        ),
        Field::new(
            "moveWinRate",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 1496),
            false,
        ),
        Field::new("bestMoveWinRate", DataType::Float32, false),
        Field::new("resultValue", DataType::Float32, false),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hcpe_schema() {
        let schema = hcpe_schema();
        assert_eq!(schema.fields().len(), 9);
        assert_eq!(schema.field(0).name(), "hcp");
        assert_eq!(schema.field(1).name(), "eval");
    }

    #[test]
    fn test_preprocessing_schema() {
        let schema = preprocessing_schema();
        assert_eq!(schema.fields().len(), 7);
        assert_eq!(schema.field(0).name(), "id");
        assert_eq!(schema.field(1).name(), "boardIdPositions");
    }
}
