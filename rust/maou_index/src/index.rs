//! コアインデックス実装．
//!
//! Hash mapとB-treeを組み合わせたハイブリッドインデックス構造．
//! - Hash map: O(1)でID検索
//! - B-tree: O(log n)で評価値範囲検索

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

use polars::prelude::*;

use crate::error::IndexError;

/// レコードの位置情報．
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordLocation {
    /// ファイルインデックス（file_pathsへのインデックス）
    pub file_index: u32,
    /// ファイル内の行番号
    pub row_number: u32,
}

/// データ型の種類．
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrayType {
    HCPE,
    Preprocessing,
    Stage1,
    Stage2,
}

impl ArrayType {
    /// 文字列からArrayTypeを解析．
    pub fn from_str(s: &str) -> Result<Self, IndexError> {
        match s.to_lowercase().as_str() {
            "hcpe" => Ok(ArrayType::HCPE),
            "preprocessing" => Ok(ArrayType::Preprocessing),
            "stage1" => Ok(ArrayType::Stage1),
            "stage2" => Ok(ArrayType::Stage2),
            _ => Err(IndexError::InvalidParameter(format!(
                "Unknown array type: {}",
                s
            ))),
        }
    }
}

/// ハイブリッドインデックス構造．
pub struct DataIndex {
    /// ID → レコード位置のハッシュマップ（O(1)検索）
    id_index: HashMap<String, RecordLocation>,

    /// 評価値 → レコード位置リストのB-tree（O(log n)範囲検索）
    eval_index: BTreeMap<i16, Vec<RecordLocation>>,

    /// 総レコード数
    total_records: usize,

    /// データ型
    array_type: ArrayType,

    /// ファイルパスリスト
    file_paths: Vec<PathBuf>,
}

impl DataIndex {
    /// 新しいインデックスを作成．
    pub fn new(
        array_type: ArrayType,
        file_paths: Vec<PathBuf>,
    ) -> Self {
        Self {
            id_index: HashMap::new(),
            eval_index: BTreeMap::new(),
            total_records: 0,
            array_type,
            file_paths,
        }
    }

    /// レコードをインデックスに追加．
    pub fn add_record(
        &mut self,
        id: String,
        eval: i16,
        location: RecordLocation,
    ) {
        // IDインデックスに追加
        self.id_index.insert(id, location);

        // 評価値インデックスに追加
        self.eval_index
            .entry(eval)
            .or_insert_with(Vec::new)
            .push(location);

        self.total_records += 1;
    }

    /// IDでレコード位置を検索（O(1)）．
    pub fn search_by_id(
        &self,
        id: &str,
    ) -> Option<RecordLocation> {
        self.id_index.get(id).copied()
    }

    /// 評価値範囲でレコード位置を検索（O(log n)）．
    ///
    /// # Arguments
    /// * `min_eval` - 最小評価値（None = -∞）
    /// * `max_eval` - 最大評価値（None = +∞）
    /// * `offset` - スキップする件数（ページネーション用）
    /// * `limit` - 取得する最大件数
    ///
    /// # Returns
    /// レコード位置のベクター
    pub fn search_by_eval_range(
        &self,
        min_eval: Option<i16>,
        max_eval: Option<i16>,
        offset: usize,
        limit: usize,
    ) -> Vec<RecordLocation> {
        let min = min_eval.unwrap_or(i16::MIN);
        let max = max_eval.unwrap_or(i16::MAX);

        // B-treeで範囲検索
        self.eval_index
            .range(min..=max)
            .flat_map(|(_, locations)| locations.iter().copied())
            .skip(offset)
            .take(limit)
            .collect()
    }

    /// 評価値範囲内のレコード総数をカウント．
    pub fn count_eval_range(
        &self,
        min_eval: Option<i16>,
        max_eval: Option<i16>,
    ) -> usize {
        let min = min_eval.unwrap_or(i16::MIN);
        let max = max_eval.unwrap_or(i16::MAX);

        self.eval_index
            .range(min..=max)
            .flat_map(|(_, locations)| locations.iter())
            .count()
    }

    /// 総レコード数を取得．
    pub fn total_records(&self) -> usize {
        self.total_records
    }

    /// データ型を取得．
    pub fn array_type(&self) -> ArrayType {
        self.array_type
    }

    /// ファイルパスリストを取得．
    pub fn file_paths(&self) -> &[PathBuf] {
        &self.file_paths
    }

    /// .featherファイルをスキャンしてインデックスを構築．
    ///
    /// # Returns
    /// 成功時はOk(())，エラー時はErr
    pub fn build_from_files(&mut self) -> Result<(), IndexError> {
        for (file_idx, file_path) in self.file_paths.clone().iter().enumerate() {
            // Featherファイルを読み込み
            let df = LazyFrame::scan_ipc(
                file_path,
                ScanArgsIpc::default(),
            )
            .map_err(|e| {
                IndexError::BuildFailed(format!(
                    "Failed to read {}: {}",
                    file_path.display(),
                    e
                ))
            })?
            .collect()
            .map_err(|e| {
                IndexError::BuildFailed(format!(
                    "Failed to collect DataFrame from {}: {}",
                    file_path.display(),
                    e
                ))
            })?;

            let num_rows = df.height();

            // IDカラムを取得
            let id_column = df
                .column("id")
                .map_err(|e| {
                    IndexError::InvalidFormat(format!(
                        "Missing 'id' column: {}",
                        e
                    ))
                })?;

            // IDインデックスを構築
            for row_num in 0..num_rows {
                let id_value = id_column.get(row_num).map_err(|e| {
                    IndexError::BuildFailed(format!(
                        "Failed to get id at row {}: {}",
                        row_num, e
                    ))
                })?;

                let id_str = id_value.to_string();

                let location = RecordLocation {
                    file_index: file_idx as u32,
                    row_number: row_num as u32,
                };

                // HCPEの場合はeval値も登録
                if matches!(self.array_type, ArrayType::HCPE) {
                    if let Ok(eval_column) = df.column("eval") {
                        if let Ok(eval_value) = eval_column.get(row_num)
                        {
                            if let Ok(eval_i16) =
                                eval_value.try_extract::<i16>()
                            {
                                self.add_record(
                                    id_str,
                                    eval_i16,
                                    location,
                                );
                                continue;
                            }
                        }
                    }
                }

                // HCPE以外，またはeval取得失敗時はeval=0で登録
                self.add_record(id_str, 0, location);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_search_by_id() {
        let mut index = DataIndex::new(
            ArrayType::HCPE,
            vec![PathBuf::from("test.feather")],
        );

        let loc = RecordLocation {
            file_index: 0,
            row_number: 10,
        };

        index.add_record("id_123".to_string(), 500, loc);

        let result = index.search_by_id("id_123");
        assert_eq!(result, Some(loc));

        let not_found = index.search_by_id("id_999");
        assert_eq!(not_found, None);
    }

    #[test]
    fn test_search_by_eval_range() {
        let mut index = DataIndex::new(
            ArrayType::HCPE,
            vec![PathBuf::from("test.feather")],
        );

        // 複数レコード追加
        for i in 0..100 {
            let loc = RecordLocation {
                file_index: 0,
                row_number: i,
            };
            let eval = (i as i16) * 10 - 500; // -500 ~ 490
            index.add_record(format!("id_{}", i), eval, loc);
        }

        // 範囲検索: -100 ~ 100
        let results =
            index.search_by_eval_range(Some(-100), Some(100), 0, 50);

        // -100 ~ 100の範囲には21件（-100, -90, ..., 90, 100）
        assert_eq!(results.len(), 21);

        // ページネーション: offset=10, limit=5
        let page_results =
            index.search_by_eval_range(Some(-100), Some(100), 10, 5);
        assert_eq!(page_results.len(), 5);
    }

    #[test]
    fn test_count_eval_range() {
        let mut index = DataIndex::new(
            ArrayType::HCPE,
            vec![PathBuf::from("test.feather")],
        );

        for i in 0..100 {
            let loc = RecordLocation {
                file_index: 0,
                row_number: i,
            };
            let eval = (i as i16) * 10 - 500;
            index.add_record(format!("id_{}", i), eval, loc);
        }

        let count = index.count_eval_range(Some(-100), Some(100));
        assert_eq!(count, 21);

        let all_count = index.count_eval_range(None, None);
        assert_eq!(all_count, 100);
    }

    #[test]
    fn test_array_type_from_str() {
        assert_eq!(
            ArrayType::from_str("hcpe").unwrap(),
            ArrayType::HCPE
        );
        assert_eq!(
            ArrayType::from_str("PREPROCESSING").unwrap(),
            ArrayType::Preprocessing
        );
        assert!(ArrayType::from_str("invalid").is_err());
    }
}
