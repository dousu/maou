//! Maou検索インデックスモジュール（PyO3バインディング）．
//!
//! PythonからRustベースの高速インデックスを使用するためのインターフェース．

use pyo3::prelude::*;
use std::path::PathBuf;

mod error;
mod index;

use index::{ArrayType, DataIndex, RecordLocation};

/// Python公開用の検索インデックスクラス．
#[pyclass]
pub struct SearchIndex {
    /// 内部インデックス
    index: DataIndex,
}

#[pymethods]
impl SearchIndex {
    /// 新しいSearchIndexを作成（モック実装）．
    ///
    /// Note: 現在はモック実装．実際のファイル読み込みは後のPhaseで実装．
    ///
    /// # Arguments
    /// * `file_paths` - データファイルパスのリスト
    /// * `array_type` - データ型（"hcpe", "preprocessing", "stage1", "stage2"）
    #[new]
    #[pyo3(signature = (file_paths, array_type))]
    pub fn new(
        file_paths: Vec<String>,
        array_type: String,
    ) -> PyResult<Self> {
        // ArrayType解析
        let arr_type = ArrayType::from_str(&array_type)?;

        // PathBufに変換
        let paths: Vec<PathBuf> =
            file_paths.iter().map(PathBuf::from).collect();

        // インデックス作成
        let index = DataIndex::new(arr_type, paths);

        Ok(Self { index })
    }

    /// モックデータでインデックスを構築（テスト用）．
    ///
    /// # Arguments
    /// * `num_records` - 生成するモックレコード数
    pub fn build_mock(&mut self, num_records: usize) {
        for i in 0..num_records {
            let location = RecordLocation {
                file_index: 0,
                row_number: i as u32,
            };
            let id = format!("mock_id_{}", i);
            let eval = ((i as i16) % 2000) - 1000; // -1000 ~ 999

            self.index.add_record(id, eval, location);
        }
    }

    /// .featherファイルをスキャンしてインデックスを構築．
    ///
    /// # Returns
    /// 成功時はOk(())，エラー時はErr
    pub fn build_from_files(&mut self) -> PyResult<()> {
        self.index
            .build_from_files()
            .map_err(|e| PyErr::from(e))
    }

    /// IDでレコードを検索．
    ///
    /// # Arguments
    /// * `id` - 検索するレコードID
    ///
    /// # Returns
    /// `(file_index, row_number)`のタプル，または`None`
    pub fn search_by_id(
        &self,
        id: String,
    ) -> Option<(u32, u32)> {
        self.index
            .search_by_id(&id)
            .map(|loc| (loc.file_index, loc.row_number))
    }

    /// 評価値範囲でレコードを検索．
    ///
    /// # Arguments
    /// * `min_eval` - 最小評価値（Noneで-∞）
    /// * `max_eval` - 最大評価値（Noneで+∞）
    /// * `offset` - スキップする件数
    /// * `limit` - 取得する最大件数
    ///
    /// # Returns
    /// `[(file_index, row_number), ...]`のリスト
    #[pyo3(signature = (min_eval=None, max_eval=None, offset=0, limit=20))]
    pub fn search_by_eval_range(
        &self,
        min_eval: Option<i16>,
        max_eval: Option<i16>,
        offset: usize,
        limit: usize,
    ) -> Vec<(u32, u32)> {
        self.index
            .search_by_eval_range(min_eval, max_eval, offset, limit)
            .iter()
            .map(|loc| (loc.file_index, loc.row_number))
            .collect()
    }

    /// 評価値範囲内のレコード総数をカウント．
    ///
    /// # Arguments
    /// * `min_eval` - 最小評価値（Noneで-∞）
    /// * `max_eval` - 最大評価値（Noneで+∞）
    ///
    /// # Returns
    /// レコード数
    #[pyo3(signature = (min_eval=None, max_eval=None))]
    pub fn count_eval_range(
        &self,
        min_eval: Option<i16>,
        max_eval: Option<i16>,
    ) -> usize {
        self.index.count_eval_range(min_eval, max_eval)
    }

    /// 総レコード数を取得．
    pub fn total_records(&self) -> usize {
        self.index.total_records()
    }

    /// ファイルパスリストを取得．
    pub fn file_paths(&self) -> Vec<String> {
        self.index
            .file_paths()
            .iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect()
    }
}

/// Pythonモジュール定義．
#[pymodule]
fn maou_index(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SearchIndex>()?;
    Ok(())
}
