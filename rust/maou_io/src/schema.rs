use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;

/// HCPE schema (game records)．
///
/// HCPEフォーマットは棋譜データを効率的に保存するための構造．
/// 各レコードは1つの局面とその評価値，指し手，対局情報を含む．
pub fn hcpe_schema() -> Schema {
    Schema::new(vec![
        Field::new("hcp", DataType::FixedSizeBinary(32), false),
        Field::new("eval", DataType::Int16, false),
        Field::new("bestMove16", DataType::Int16, false),
        Field::new("gameResult", DataType::Int8, false),
        Field::new("id", DataType::Utf8, false),
        Field::new("partitioningKey", DataType::Date32, false),
        Field::new(
            "ratings",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::UInt16, false)),
                2,
            ),
            false,
        ),
        Field::new("endgameStatus", DataType::Utf8, false),
        Field::new("moves", DataType::Int16, false),
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
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::UInt8, false)),
                14,
            ),
            false,
        ),
        Field::new(
            "moveLabel",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, false)),
                1496,
            ),
            false,
        ),
        Field::new(
            "moveWinRate",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, false)),
                1496,
            ),
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
