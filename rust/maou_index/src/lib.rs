//! Maou検索インデックスモジュール（PyO3バインディング）．
//!
//! PythonからRustベースの高速インデックスを使用するためのインターフェース．

use pyo3::prelude::*;
use std::path::PathBuf;

mod error;
mod index;
mod path_scanner;

use index::{ArrayType, DataIndex, RecordLocation};
use path_scanner::PathScanner as RustPathScanner;

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

    /// IDプレフィックスで候補を検索．
    ///
    /// # Arguments
    /// * `prefix` - 検索プレフィックス
    /// * `limit` - 最大取得件数（デフォルト: 50）
    ///
    /// # Returns
    /// マッチするIDのリスト（ソート済み）
    #[pyo3(signature = (prefix, limit=50))]
    pub fn search_id_prefix(&self, prefix: String, limit: usize) -> Vec<String> {
        self.index.search_id_prefix(&prefix, limit)
    }

    /// 全IDリストを取得（ソート済み）．
    ///
    /// # Arguments
    /// * `limit` - 最大取得件数（Noneで全件，デフォルト: None）
    ///
    /// # Returns
    /// IDのリスト（ソート済み）
    #[pyo3(signature = (limit=None))]
    pub fn get_all_ids(&self, limit: Option<usize>) -> Vec<String> {
        self.index.get_all_ids(limit)
    }
}

/// Python公開用のパススキャナークラス．
#[pyclass]
pub struct PathScanner {
    /// 内部スキャナー
    scanner: RustPathScanner,
}

#[pymethods]
impl PathScanner {
    /// 新しいPathScannerを作成．
    ///
    /// # Arguments
    /// * `ttl_seconds` - キャッシュのTTL（秒，デフォルト: 60）
    #[new]
    #[pyo3(signature = (ttl_seconds=60))]
    pub fn new(ttl_seconds: u64) -> Self {
        Self {
            scanner: RustPathScanner::new(ttl_seconds),
        }
    }

    /// キャッシュが古いかチェック．
    ///
    /// # Returns
    /// キャッシュが古い場合はTrue
    pub fn is_stale(&self) -> bool {
        self.scanner.is_stale()
    }

    /// ディレクトリをスキャンしてキャッシュを構築．
    ///
    /// # Arguments
    /// * `base_path` - スキャンするベースディレクトリ
    /// * `max_depth` - 最大スキャン深度（デフォルト: 5）
    ///
    /// # Returns
    /// スキャンしたディレクトリ数
    #[pyo3(signature = (base_path, max_depth=5))]
    pub fn scan_directories(&mut self, base_path: String, max_depth: usize) -> PyResult<usize> {
        let path = PathBuf::from(base_path);
        self.scanner
            .scan_directories(&path, max_depth)
            .map_err(|e| PyErr::from(e))
    }

    /// .featherファイルをスキャンしてキャッシュを構築．
    ///
    /// # Arguments
    /// * `base_path` - スキャンするベースディレクトリ
    /// * `recursive` - 再帰的にスキャンするか（デフォルト: True）
    ///
    /// # Returns
    /// スキャンしたファイル数
    #[pyo3(signature = (base_path, recursive=true))]
    pub fn scan_feather_files(&mut self, base_path: String, recursive: bool) -> PyResult<usize> {
        let path = PathBuf::from(base_path);
        self.scanner
            .scan_feather_files(&path, recursive)
            .map_err(|e| PyErr::from(e))
    }

    /// ディレクトリパスのプレフィックス検索．
    ///
    /// # Arguments
    /// * `prefix` - 検索するプレフィックス
    /// * `limit` - 最大取得件数（デフォルト: 50）
    ///
    /// # Returns
    /// マッチするディレクトリパスのリスト（ソート済み）
    #[pyo3(signature = (prefix, limit=50))]
    pub fn search_directory_prefix(&self, prefix: String, limit: usize) -> Vec<String> {
        self.scanner.search_directory_prefix(&prefix, limit)
    }

    /// ファイルパスのプレフィックス検索．
    ///
    /// # Arguments
    /// * `prefix` - 検索するプレフィックス
    /// * `limit` - 最大取得件数（デフォルト: 100）
    ///
    /// # Returns
    /// マッチする.featherファイルパスのリスト（ソート済み）
    #[pyo3(signature = (prefix, limit=100))]
    pub fn search_file_prefix(&self, prefix: String, limit: usize) -> Vec<String> {
        self.scanner.search_file_prefix(&prefix, limit)
    }

    /// 現在のキャッシュ統計を取得．
    ///
    /// # Returns
    /// (ディレクトリ数, ファイル数)のタプル
    pub fn cache_stats(&self) -> (usize, usize) {
        self.scanner.cache_stats()
    }
}

/// Pythonモジュール定義．
#[pymodule]
fn maou_index(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SearchIndex>()?;
    m.add_class::<PathScanner>()?;
    Ok(())
}
