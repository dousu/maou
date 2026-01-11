//! パススキャナーモジュール（高速パス検索）．
//!
//! ファイルシステムをスキャンしてBTreeMapベースのプレフィックス検索を提供．

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};
use walkdir::WalkDir;

use crate::error::IndexError;

/// パスキャッシュ（TTL付き）．
#[derive(Debug)]
pub struct PathScanner {
    /// ディレクトリパスのマップ（key: パス文字列, value: PathBuf）
    directories: BTreeMap<String, PathBuf>,
    /// .featherファイルのマップ（key: パス文字列, value: PathBuf）
    feather_files: BTreeMap<String, PathBuf>,
    /// 最終スキャン時刻
    last_scan: Option<SystemTime>,
    /// キャッシュTTL
    ttl: Duration,
}

impl PathScanner {
    /// 新しいPathScannerを作成．
    ///
    /// # Arguments
    /// * `ttl_seconds` - キャッシュのTTL（秒）
    pub fn new(ttl_seconds: u64) -> Self {
        Self {
            directories: BTreeMap::new(),
            feather_files: BTreeMap::new(),
            last_scan: None,
            ttl: Duration::from_secs(ttl_seconds),
        }
    }

    /// キャッシュが古いかチェック．
    ///
    /// # Returns
    /// キャッシュが古い場合はtrue
    pub fn is_stale(&self) -> bool {
        match self.last_scan {
            None => true,
            Some(last_time) => {
                match SystemTime::now().duration_since(last_time) {
                    Ok(elapsed) => elapsed > self.ttl,
                    Err(_) => true, // システム時刻が巻き戻った場合
                }
            }
        }
    }

    /// ディレクトリをスキャンしてキャッシュを構築．
    ///
    /// # Arguments
    /// * `base_path` - スキャンするベースディレクトリ
    /// * `max_depth` - 最大スキャン深度（1 = 直下のみ）
    ///
    /// # Returns
    /// スキャンしたディレクトリ数
    pub fn scan_directories(
        &mut self,
        base_path: &Path,
        max_depth: usize,
    ) -> Result<usize, IndexError> {
        self.directories.clear();

        // ベースパスが存在しない場合はエラー
        if !base_path.exists() {
            return Err(IndexError::NotFound(
                format!("Directory not found: {}", base_path.to_string_lossy()),
            ));
        }

        // walkdirで再帰的にディレクトリを探索
        let walker = WalkDir::new(base_path)
            .max_depth(max_depth)
            .follow_links(false);

        let mut count = 0;
        for entry in walker.into_iter().filter_map(|e| e.ok()) {
            if entry.file_type().is_dir() {
                let path = entry.path();
                let path_str = path.to_string_lossy().to_string();
                self.directories.insert(path_str, path.to_path_buf());
                count += 1;
            }
        }

        self.last_scan = Some(SystemTime::now());
        Ok(count)
    }

    /// .featherファイルをスキャンしてキャッシュを構築．
    ///
    /// # Arguments
    /// * `base_path` - スキャンするベースディレクトリ
    /// * `recursive` - 再帰的にスキャンするか
    ///
    /// # Returns
    /// スキャンしたファイル数
    pub fn scan_feather_files(
        &mut self,
        base_path: &Path,
        recursive: bool,
    ) -> Result<usize, IndexError> {
        self.feather_files.clear();

        // ベースパスが存在しない場合はエラー
        if !base_path.exists() {
            return Err(IndexError::NotFound(
                format!("Directory not found: {}", base_path.to_string_lossy()),
            ));
        }

        // walkdirで.featherファイルを探索
        let walker = if recursive {
            WalkDir::new(base_path).follow_links(false)
        } else {
            WalkDir::new(base_path).max_depth(1).follow_links(false)
        };

        let mut count = 0;
        for entry in walker.into_iter().filter_map(|e| e.ok()) {
            if entry.file_type().is_file() {
                let path = entry.path();
                if let Some(ext) = path.extension() {
                    if ext == "feather" {
                        let path_str = path.to_string_lossy().to_string();
                        self.feather_files.insert(path_str, path.to_path_buf());
                        count += 1;
                    }
                }
            }
        }

        self.last_scan = Some(SystemTime::now());
        Ok(count)
    }

    /// ディレクトリパスのプレフィックス検索．
    ///
    /// # Arguments
    /// * `prefix` - 検索するプレフィックス
    /// * `limit` - 最大取得件数
    ///
    /// # Returns
    /// マッチするディレクトリパスのリスト（ソート済み）
    pub fn search_directory_prefix(&self, prefix: &str, limit: usize) -> Vec<String> {
        self.directories
            .range(prefix.to_string()..)
            .take_while(|(k, _)| k.starts_with(prefix))
            .take(limit)
            .map(|(k, _)| k.clone())
            .collect()
    }

    /// ファイルパスのプレフィックス検索．
    ///
    /// # Arguments
    /// * `prefix` - 検索するプレフィックス
    /// * `limit` - 最大取得件数
    ///
    /// # Returns
    /// マッチする.featherファイルパスのリスト（ソート済み）
    pub fn search_file_prefix(&self, prefix: &str, limit: usize) -> Vec<String> {
        self.feather_files
            .range(prefix.to_string()..)
            .take_while(|(k, _)| k.starts_with(prefix))
            .take(limit)
            .map(|(k, _)| k.clone())
            .collect()
    }

    /// 現在のキャッシュ統計を取得．
    ///
    /// # Returns
    /// (ディレクトリ数, ファイル数)
    pub fn cache_stats(&self) -> (usize, usize) {
        (self.directories.len(), self.feather_files.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_path_scanner_new() {
        let scanner = PathScanner::new(60);
        assert!(scanner.is_stale());
        assert_eq!(scanner.cache_stats(), (0, 0));
    }

    #[test]
    fn test_scan_directories() {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path();

        // テスト用ディレクトリ構造を作成
        fs::create_dir_all(base_path.join("dir1/subdir1")).unwrap();
        fs::create_dir_all(base_path.join("dir2")).unwrap();

        let mut scanner = PathScanner::new(60);
        let count = scanner.scan_directories(base_path, 3).unwrap();

        // ベースディレクトリ + dir1 + dir1/subdir1 + dir2 = 4
        assert!(count >= 3); // 少なくとも3つ以上
        assert!(!scanner.is_stale());
    }

    #[test]
    fn test_scan_feather_files() {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path();

        // テスト用ファイルを作成
        fs::write(base_path.join("data1.feather"), b"test").unwrap();
        fs::write(base_path.join("data2.feather"), b"test").unwrap();
        fs::write(base_path.join("other.txt"), b"test").unwrap();

        let mut scanner = PathScanner::new(60);
        let count = scanner.scan_feather_files(base_path, false).unwrap();

        // .featherファイル2つ（.txtは除外）
        assert_eq!(count, 2);
    }

    #[test]
    fn test_prefix_search() {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path();

        // テスト用ファイルを作成
        fs::write(base_path.join("test_data1.feather"), b"test").unwrap();
        fs::write(base_path.join("test_data2.feather"), b"test").unwrap();
        fs::write(base_path.join("other.feather"), b"test").unwrap();

        let mut scanner = PathScanner::new(60);
        scanner.scan_feather_files(base_path, false).unwrap();

        // プレフィックス "test_" で検索
        let prefix = format!("{}/test_", base_path.to_string_lossy());
        let results = scanner.search_file_prefix(&prefix, 10);

        assert_eq!(results.len(), 2);
        assert!(results[0].contains("test_data"));
    }
}
