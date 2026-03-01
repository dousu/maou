use arrow::array::RecordBatch;
use arrow::ipc::reader::{FileReader, StreamReader};
use arrow::ipc::writer::{FileWriter, IpcWriteOptions};
use arrow::ipc::CompressionType;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use crate::error::MaouIOError;

/// Save Arrow RecordBatch to .feather file with LZ4 compression．
///
/// LZ4_FRAME圧縮を使用して高速かつ効率的なファイルI/Oを実現する．
///
/// Args:
///     record_batch: Arrow RecordBatch to save
///     file_path: Output file path (.feather extension recommended)
///
/// Returns:
///     Ok(()) on success，Err(MaouIOError) on failure
pub fn save_feather(record_batch: &RecordBatch, file_path: &str) -> Result<(), MaouIOError> {
    let path = Path::new(file_path);
    let file = File::create(path)?;
    let writer = BufWriter::new(file);

    // Enable LZ4 compression for efficient storage
    let write_options = IpcWriteOptions::default()
        .try_with_compression(Some(CompressionType::LZ4_FRAME))?;

    let mut writer = FileWriter::try_new_with_options(writer, &record_batch.schema(), write_options)?;

    writer.write(record_batch)?;
    writer.finish()?;

    Ok(())
}

/// Load Arrow RecordBatch from .feather file．
///
/// Automatically detects and supports both IPC formats:
/// - File format (starts with "ARROW1")
/// - Stream format (starts with 0xFFFFFFFF)
///
/// Args:
///     file_path: Input file path (.feather extension)
///
/// Returns:
///     Ok(RecordBatch) on success，Err(MaouIOError) on failure
pub fn load_feather(file_path: &str) -> Result<RecordBatch, MaouIOError> {
    let path = Path::new(file_path);
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Peek at the first 4 bytes to detect format
    use std::io::{Read, Seek, SeekFrom};
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    reader.seek(SeekFrom::Start(0))?;

    // Check if it's Stream format (starts with 0xFFFFFFFF = -1)
    if magic == [0xFF, 0xFF, 0xFF, 0xFF] {
        // Stream format
        let mut stream_reader = StreamReader::try_new(reader, None)?;

        // Read all batches and concatenate them
        let mut batches = Vec::new();
        while let Some(batch_result) = stream_reader.next() {
            batches.push(batch_result?);
        }

        if batches.is_empty() {
            return Err(MaouIOError::IOError(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "Empty file: no record batches found",
            )));
        }

        // If there's only one batch, return it directly
        if batches.len() == 1 {
            return Ok(batches.into_iter().next().unwrap());
        }

        // Concatenate multiple batches into one
        use arrow::compute::concat_batches;
        let schema = batches[0].schema();
        concat_batches(&schema, &batches)
            .map_err(|e| MaouIOError::ArrowError(e))
    } else {
        // File format (starts with "ARROW1")
        let mut file_reader = FileReader::try_new(reader, None)?;

        // Read first batch
        match file_reader.next() {
            Some(Ok(batch)) => Ok(batch),
            Some(Err(e)) => Err(MaouIOError::ArrowError(e)),
            None => Err(MaouIOError::IOError(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "Empty file: no record batches found",
            ))),
        }
    }
}

/// Save multiple record batches to a single .feather file with LZ4 compression．
///
/// For large datasets that don't fit in a single batch．
/// Uses LZ4_FRAME compression for efficient storage．
pub fn save_feather_batches(
    batches: &[RecordBatch],
    file_path: &str,
) -> Result<(), MaouIOError> {
    if batches.is_empty() {
        return Err(MaouIOError::SchemaError(
            "Cannot save empty batch list".to_string(),
        ));
    }

    let path = Path::new(file_path);
    let file = File::create(path)?;
    let writer = BufWriter::new(file);

    // Enable LZ4 compression for efficient storage
    let write_options = IpcWriteOptions::default()
        .try_with_compression(Some(CompressionType::LZ4_FRAME))?;

    let mut writer = FileWriter::try_new_with_options(writer, &batches[0].schema(), write_options)?;

    for batch in batches {
        writer.write(batch)?;
    }

    writer.finish()?;

    Ok(())
}

/// Load all record batches from a .feather file．
///
/// Returns a vector of all batches in the file．
pub fn load_feather_batches(file_path: &str) -> Result<Vec<RecordBatch>, MaouIOError> {
    let path = Path::new(file_path);
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut reader = FileReader::try_new(reader, None)?;

    let mut batches = Vec::new();
    while let Some(batch_result) = reader.next() {
        batches.push(batch_result?);
    }

    if batches.is_empty() {
        return Err(MaouIOError::IOError(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "Empty file: no record batches found",
        )));
    }

    Ok(batches)
}

/// Merge multiple .feather files into chunked output files．
///
/// 複数の小さなfeatherファイルを指定行数ごとにまとめ，
/// チャンクされた出力ファイルとして保存する．
/// LZ4圧縮を維持したまま，ファイル粒度を最適化する．
///
/// Args:
///     file_paths: Input .feather file paths
///     output_dir: Directory for output chunked files
///     rows_per_chunk: Target number of rows per output file
///     output_prefix: Prefix for output file names
///
/// Returns:
///     Ok(Vec<String>) - List of output file paths on success
pub fn merge_feather_files(
    file_paths: &[String],
    output_dir: &str,
    rows_per_chunk: usize,
    output_prefix: &str,
) -> Result<Vec<String>, MaouIOError> {
    if rows_per_chunk == 0 {
        return Err(MaouIOError::SchemaError(
            "rows_per_chunk must be > 0".to_string(),
        ));
    }
    if file_paths.is_empty() {
        return Ok(Vec::new());
    }

    // Create output directory
    let out_path = Path::new(output_dir);
    std::fs::create_dir_all(out_path)?;

    // Load all files and track their row counts
    let mut all_batches: Vec<RecordBatch> = Vec::new();
    for fp in file_paths {
        let batch = load_feather(fp)?;
        all_batches.push(batch);
    }

    // Calculate total rows
    let total_rows: usize = all_batches.iter().map(|b| b.num_rows()).sum();

    // If total is small enough for a single file, merge all
    if total_rows <= rows_per_chunk {
        let schema = all_batches[0].schema();
        let merged = arrow::compute::concat_batches(&schema, &all_batches)
            .map_err(MaouIOError::ArrowError)?;

        let chunk_path = out_path.join(format!("{}_chunk0000.feather", output_prefix));
        let chunk_path_str = chunk_path.to_str().ok_or_else(|| {
            MaouIOError::IOError(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Invalid output path",
            ))
        })?;
        save_feather(&merged, chunk_path_str)?;
        return Ok(vec![chunk_path_str.to_string()]);
    }

    // Merge files into chunks of approximately rows_per_chunk
    let mut output_paths = Vec::new();
    let mut chunk_idx = 0;
    let mut current_batches: Vec<RecordBatch> = Vec::new();
    let mut current_rows = 0;

    for batch in all_batches {
        let batch_rows = batch.num_rows();

        // If adding this batch exceeds the target, flush current chunk first
        if current_rows > 0 && current_rows + batch_rows > rows_per_chunk {
            let schema = current_batches[0].schema();
            let merged = arrow::compute::concat_batches(&schema, &current_batches)
                .map_err(MaouIOError::ArrowError)?;

            let chunk_path = out_path.join(format!("{}_chunk{:04}.feather", output_prefix, chunk_idx));
            let chunk_path_str = chunk_path.to_str().ok_or_else(|| {
                MaouIOError::IOError(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Invalid output path",
                ))
            })?;
            save_feather(&merged, chunk_path_str)?;
            output_paths.push(chunk_path_str.to_string());

            current_batches.clear();
            current_rows = 0;
            chunk_idx += 1;
        }

        current_batches.push(batch);
        current_rows += batch_rows;
    }

    // Flush remaining
    if !current_batches.is_empty() {
        let schema = current_batches[0].schema();
        let merged = arrow::compute::concat_batches(&schema, &current_batches)
            .map_err(MaouIOError::ArrowError)?;

        let chunk_path = out_path.join(format!("{}_chunk{:04}.feather", output_prefix, chunk_idx));
        let chunk_path_str = chunk_path.to_str().ok_or_else(|| {
            MaouIOError::IOError(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Invalid output path",
            ))
        })?;
        save_feather(&merged, chunk_path_str)?;
        output_paths.push(chunk_path_str.to_string());
    }

    Ok(output_paths)
}

/// Split a .feather file into multiple smaller files．
///
/// 大きなfeatherファイルを指定行数ごとに分割し，
/// 各チャンクをLZ4圧縮付きのfeatherファイルとして保存する．
/// RecordBatch::slice()によるゼロコピー分割で高速に処理する．
///
/// Args:
///     file_path: Input .feather file path
///     output_dir: Directory for output split files
///     rows_per_file: Maximum number of rows per output file
///
/// Returns:
///     Ok(Vec<String>) - List of output file paths on success
pub fn split_feather(
    file_path: &str,
    output_dir: &str,
    rows_per_file: usize,
) -> Result<Vec<String>, MaouIOError> {
    if rows_per_file == 0 {
        return Err(MaouIOError::SchemaError(
            "rows_per_file must be > 0".to_string(),
        ));
    }

    // Load the entire file
    let batch = load_feather(file_path)?;
    let total_rows = batch.num_rows();

    if total_rows == 0 {
        return Ok(Vec::new());
    }

    // If the file is already small enough, return it as-is
    if total_rows <= rows_per_file {
        return Ok(vec![file_path.to_string()]);
    }

    // Create output directory if it doesn't exist
    let out_path = Path::new(output_dir);
    std::fs::create_dir_all(out_path)?;

    // Extract base name from input file
    let input_path = Path::new(file_path);
    let stem = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("split");

    let mut output_paths = Vec::new();
    let mut offset = 0;
    let mut chunk_idx = 0;

    while offset < total_rows {
        let length = std::cmp::min(rows_per_file, total_rows - offset);
        let chunk = batch.slice(offset, length);

        let chunk_path = out_path.join(format!("{}_split{:04}.feather", stem, chunk_idx));
        let chunk_path_str = chunk_path.to_str().ok_or_else(|| {
            MaouIOError::IOError(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Invalid output path",
            ))
        })?;

        save_feather(&chunk, chunk_path_str)?;
        output_paths.push(chunk_path_str.to_string());

        offset += length;
        chunk_idx += 1;
    }

    Ok(output_paths)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{ArrayRef, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;
    use tempfile::tempdir;

    fn create_test_batch() -> RecordBatch {
        let schema = Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Int32, false),
        ]);

        let id_array = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let value_array = Int32Array::from(vec![10, 20, 30, 40, 50]);

        RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(id_array) as ArrayRef, Arc::new(value_array) as ArrayRef],
        )
        .unwrap()
    }

    #[test]
    fn test_save_load_feather() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.feather");

        let batch = create_test_batch();

        // Save
        save_feather(&batch, file_path.to_str().unwrap()).unwrap();

        // Load
        let loaded_batch = load_feather(file_path.to_str().unwrap()).unwrap();

        // Verify
        assert_eq!(batch.num_rows(), loaded_batch.num_rows());
        assert_eq!(batch.num_columns(), loaded_batch.num_columns());
    }

    #[test]
    fn test_save_load_multiple_batches() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_multi.feather");

        let batch1 = create_test_batch();
        let batch2 = create_test_batch();

        // Save
        save_feather_batches(&[batch1, batch2], file_path.to_str().unwrap()).unwrap();

        // Load
        let loaded_batches = load_feather_batches(file_path.to_str().unwrap()).unwrap();

        // Verify
        assert_eq!(loaded_batches.len(), 2);
        assert_eq!(loaded_batches[0].num_rows(), 5);
        assert_eq!(loaded_batches[1].num_rows(), 5);
    }

    #[test]
    fn test_split_feather_no_split_needed() {
        let dir = tempdir().unwrap();
        let input_path = dir.path().join("small.feather");
        let output_dir = dir.path().join("output");

        let batch = create_test_batch(); // 5 rows
        save_feather(&batch, input_path.to_str().unwrap()).unwrap();

        // rows_per_file >= total rows: no split needed
        let result = split_feather(
            input_path.to_str().unwrap(),
            output_dir.to_str().unwrap(),
            10,
        )
        .unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0], input_path.to_str().unwrap());
    }

    #[test]
    fn test_split_feather_splits_correctly() {
        let dir = tempdir().unwrap();
        let input_path = dir.path().join("large.feather");
        let output_dir = dir.path().join("output");

        // Create a batch with 10 rows
        let schema = Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Int32, false),
        ]);
        let id_array = Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let value_array = Int32Array::from(vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(id_array) as ArrayRef, Arc::new(value_array) as ArrayRef],
        )
        .unwrap();

        save_feather(&batch, input_path.to_str().unwrap()).unwrap();

        // Split into chunks of 3 rows
        let result = split_feather(
            input_path.to_str().unwrap(),
            output_dir.to_str().unwrap(),
            3,
        )
        .unwrap();

        // 10 rows / 3 per file = 4 files (3+3+3+1)
        assert_eq!(result.len(), 4);

        // Verify each file
        let batch0 = load_feather(&result[0]).unwrap();
        assert_eq!(batch0.num_rows(), 3);

        let batch1 = load_feather(&result[1]).unwrap();
        assert_eq!(batch1.num_rows(), 3);

        let batch2 = load_feather(&result[2]).unwrap();
        assert_eq!(batch2.num_rows(), 3);

        let batch3 = load_feather(&result[3]).unwrap();
        assert_eq!(batch3.num_rows(), 1);
    }

    #[test]
    fn test_split_feather_zero_rows_per_file() {
        let dir = tempdir().unwrap();
        let input_path = dir.path().join("test.feather");
        let output_dir = dir.path().join("output");

        let batch = create_test_batch();
        save_feather(&batch, input_path.to_str().unwrap()).unwrap();

        let result = split_feather(
            input_path.to_str().unwrap(),
            output_dir.to_str().unwrap(),
            0,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_merge_feather_files_single_chunk() {
        let dir = tempdir().unwrap();
        let output_dir = dir.path().join("merged");

        // Create 3 small files (5 rows each = 15 total)
        let mut file_paths = Vec::new();
        for i in 0..3 {
            let path = dir.path().join(format!("small_{}.feather", i));
            let batch = create_test_batch(); // 5 rows
            save_feather(&batch, path.to_str().unwrap()).unwrap();
            file_paths.push(path.to_str().unwrap().to_string());
        }

        // Merge with chunk size 20 → all fit in one chunk
        let result = merge_feather_files(
            &file_paths,
            output_dir.to_str().unwrap(),
            20,
            "test",
        )
        .unwrap();

        assert_eq!(result.len(), 1);
        let merged = load_feather(&result[0]).unwrap();
        assert_eq!(merged.num_rows(), 15);
    }

    #[test]
    fn test_merge_feather_files_multiple_chunks() {
        let dir = tempdir().unwrap();
        let output_dir = dir.path().join("merged");

        // Create 5 small files (5 rows each = 25 total)
        let mut file_paths = Vec::new();
        for i in 0..5 {
            let path = dir.path().join(format!("small_{}.feather", i));
            let batch = create_test_batch(); // 5 rows
            save_feather(&batch, path.to_str().unwrap()).unwrap();
            file_paths.push(path.to_str().unwrap().to_string());
        }

        // Merge with chunk size 12 → should produce 3 chunks
        // (5+5=10, 5+5=10, 5=5)
        let result = merge_feather_files(
            &file_paths,
            output_dir.to_str().unwrap(),
            12,
            "test",
        )
        .unwrap();

        assert_eq!(result.len(), 3);

        let chunk0 = load_feather(&result[0]).unwrap();
        assert_eq!(chunk0.num_rows(), 10);

        let chunk1 = load_feather(&result[1]).unwrap();
        assert_eq!(chunk1.num_rows(), 10);

        let chunk2 = load_feather(&result[2]).unwrap();
        assert_eq!(chunk2.num_rows(), 5);
    }

    #[test]
    fn test_merge_feather_files_empty_input() {
        let dir = tempdir().unwrap();
        let output_dir = dir.path().join("merged");

        let result = merge_feather_files(
            &[],
            output_dir.to_str().unwrap(),
            10,
            "test",
        )
        .unwrap();

        assert!(result.is_empty());
    }

    #[test]
    fn test_merge_feather_files_zero_rows_per_chunk() {
        let dir = tempdir().unwrap();
        let output_dir = dir.path().join("merged");
        let path = dir.path().join("small.feather");
        let batch = create_test_batch();
        save_feather(&batch, path.to_str().unwrap()).unwrap();

        let result = merge_feather_files(
            &[path.to_str().unwrap().to_string()],
            output_dir.to_str().unwrap(),
            0,
            "test",
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_merge_feather_files_preserves_schema() {
        let dir = tempdir().unwrap();
        let output_dir = dir.path().join("merged");

        let mut file_paths = Vec::new();
        for i in 0..2 {
            let path = dir.path().join(format!("data_{}.feather", i));
            let batch = create_test_batch();
            save_feather(&batch, path.to_str().unwrap()).unwrap();
            file_paths.push(path.to_str().unwrap().to_string());
        }

        let result = merge_feather_files(
            &file_paths,
            output_dir.to_str().unwrap(),
            100,
            "test",
        )
        .unwrap();

        let merged = load_feather(&result[0]).unwrap();
        let original = create_test_batch();
        assert_eq!(merged.schema(), original.schema());
    }
}
