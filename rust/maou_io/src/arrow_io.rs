use arrow::array::RecordBatch;
use arrow::compute::{cast, concat_batches};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::ipc::reader::{FileReader, StreamReader};
use arrow::ipc::writer::{FileWriter, IpcWriteOptions};
use arrow::ipc::CompressionType;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use crate::error::MaouIOError;

/// Arrow IPC File 形式のマジックバイト (先頭 8 バイト)．
const ARROW_FILE_MAGIC: &[u8; 8] = b"ARROW1\0\0";

/// 複数のRecordBatchを単一のRecordBatchに統合する．
///
/// 空の場合はエラー，単一バッチの場合はゼロコピーで返し，
/// 複数バッチの場合は `concat_batches` で結合する．
fn consolidate_batches(batches: Vec<RecordBatch>) -> Result<RecordBatch, MaouIOError> {
    if batches.is_empty() {
        return Err(MaouIOError::IOError(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "Empty file: no record batches found",
        )));
    }

    if batches.len() == 1 {
        return Ok(batches.into_iter().next().unwrap());
    }

    let batches = normalize_view_types(batches)?;
    let schema = batches[0].schema();
    concat_batches(&schema, &batches).map_err(MaouIOError::ArrowError)
}

/// view 型を対応する非 view 型へ読み替える．
///
/// polars 1.38 は `Binary` 列を **BinaryView** で書き，`maou_io` の
/// [`save_feather`] (arrow-rs) は **LargeBinary** で書く．中身は同じでも
/// Arrow の型としては別物なので，混ざると `concat_batches` が
/// `It is not possible to concatenate arrays of different data types` で
/// 落ちる．寄せ先を非 view 側にするのは，このクレートが書き出すのが
/// そちらだからで，入力の順序に依らず出力の型が決まる．
fn canonical_type(data_type: &DataType) -> DataType {
    match data_type {
        DataType::BinaryView => DataType::LargeBinary,
        DataType::Utf8View => DataType::LargeUtf8,
        other => other.clone(),
    }
}

/// writer 違いで view / 非 view が混ざった batch 列をそろえる．
///
/// **スキーマが完全に一致していれば何もしない** (同じ writer だけで
/// 書かれた入力の出力は従来とバイト単位で同じ)．食い違うときだけ，
/// **view / 非 view の差に限って** cast する．それ以外の型の食い違い
/// (Int32 と Int64 など) は黙って寄せると値が壊れ得るので手を出さず，
/// これまでどおり `concat_batches` のエラーに任せる．
///
/// 背景: `pre-process --input-split-rows` は入力を chunk するため，
/// polars 書きと Rust 書きの `.feather` が同じディレクトリに混在すると
/// マージが停止していた (backlog 行 N-1)．
fn normalize_view_types(batches: Vec<RecordBatch>) -> Result<Vec<RecordBatch>, MaouIOError> {
    let first = batches[0].schema();
    if batches.iter().all(|b| b.schema() == first) {
        return Ok(batches);
    }

    // 列数・列名が違うのは view 型の問題ではないので触らない．
    let same_shape = batches.iter().all(|b| {
        let s = b.schema();
        s.fields().len() == first.fields().len()
            && s.fields()
                .iter()
                .zip(first.fields())
                .all(|(a, b)| a.name() == b.name())
    });
    if !same_shape {
        return Ok(batches);
    }

    // 寄せ先: 各列とも「1 本目の型を非 view 化したもの」．
    // ただし，非 view 化しても一致しない列が 1 つでもあれば，それは
    // view の問題ではない食い違いなので normalize しない．
    let target_fields: Vec<Field> = first
        .fields()
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let target = canonical_type(f.data_type());
            let nullable = batches.iter().any(|b| b.schema().field(i).is_nullable());
            Field::new(f.name(), target, nullable)
        })
        .collect();

    let normalizable = batches.iter().all(|b| {
        b.schema()
            .fields()
            .iter()
            .zip(&target_fields)
            .all(|(f, t)| canonical_type(f.data_type()) == *t.data_type())
    });
    if !normalizable {
        return Ok(batches);
    }

    let target: SchemaRef = std::sync::Arc::new(Schema::new(target_fields));

    batches
        .into_iter()
        .map(|batch| {
            let columns = batch
                .columns()
                .iter()
                .zip(target.fields())
                .map(|(col, field)| {
                    if col.data_type() == field.data_type() {
                        Ok(col.clone())
                    } else {
                        cast(col, field.data_type())
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            RecordBatch::try_new(target.clone(), columns).map_err(MaouIOError::ArrowError)
        })
        .collect()
}

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
    let write_options =
        IpcWriteOptions::default().try_with_compression(Some(CompressionType::LZ4_FRAME))?;

    let mut writer =
        FileWriter::try_new_with_options(writer, &record_batch.schema(), write_options)?;

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

    // Peek at the first 8 bytes to detect format.
    //
    // 判定は「File 形式か?」を問い，そうでなければ Stream として読む．
    // Python 側 (`domain/data/arrow_format.py`) と同じ向きである．
    // File 形式のマジック `ARROW1\0\0` は全バージョンで不変なのに対し，
    // Stream 形式の先頭は Arrow 0.15 で `0xFFFFFFFF` の continuation
    // marker が入る形へ変わっているため，Stream 側を条件にすると
    // 0.15 以前の Stream ファイルを File と誤認して footer エラーになる．
    use std::io::{Read, Seek, SeekFrom};
    let mut magic = [0u8; 8];
    // 8 バイト未満なら File 形式ではありえない (Stream reader が
    // 「壊れたファイル」として同じエラーを返す)．
    let is_file_format = match reader.read_exact(&mut magic) {
        Ok(()) => &magic == ARROW_FILE_MAGIC,
        Err(_) => false,
    };
    reader.seek(SeekFrom::Start(0))?;

    let batches: Vec<RecordBatch> = if is_file_format {
        FileReader::try_new(reader, None)?
            .collect::<Result<_, _>>()
            .map_err(MaouIOError::ArrowError)?
    } else {
        StreamReader::try_new(reader, None)?
            .collect::<Result<_, _>>()
            .map_err(MaouIOError::ArrowError)?
    };

    consolidate_batches(batches)
}

/// Save multiple record batches to a single .feather file with LZ4 compression．
///
/// For large datasets that don't fit in a single batch．
/// Uses LZ4_FRAME compression for efficient storage．
pub fn save_feather_batches(batches: &[RecordBatch], file_path: &str) -> Result<(), MaouIOError> {
    if batches.is_empty() {
        return Err(MaouIOError::SchemaError(
            "Cannot save empty batch list".to_string(),
        ));
    }

    let path = Path::new(file_path);
    let file = File::create(path)?;
    let writer = BufWriter::new(file);

    // Enable LZ4 compression for efficient storage
    let write_options =
        IpcWriteOptions::default().try_with_compression(Some(CompressionType::LZ4_FRAME))?;

    let mut writer = FileWriter::try_new_with_options(writer, &batches[0].schema(), write_options)?;

    for batch in batches {
        writer.write(batch)?;
    }

    writer.finish()?;

    Ok(())
}

/// Load all record batches from a .feather file．
///
/// `load_feather` と同じく File / Stream の両形式を受け付ける
/// (同じ拡張子の同じ入力を，関数によって読めたり読めなかったりさせない)．
///
/// Returns a vector of all batches in the file．
pub fn load_feather_batches(file_path: &str) -> Result<Vec<RecordBatch>, MaouIOError> {
    let path = Path::new(file_path);
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    use std::io::{Read, Seek, SeekFrom};
    let mut magic = [0u8; 8];
    let is_file_format = match reader.read_exact(&mut magic) {
        Ok(()) => &magic == ARROW_FILE_MAGIC,
        Err(_) => false,
    };
    reader.seek(SeekFrom::Start(0))?;

    let mut batches = Vec::new();
    if is_file_format {
        for batch_result in FileReader::try_new(reader, None)? {
            batches.push(batch_result?);
        }
    } else {
        for batch_result in StreamReader::try_new(reader, None)? {
            batches.push(batch_result?);
        }
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
        let merged = consolidate_batches(all_batches)?;

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
            let merged = consolidate_batches(std::mem::take(&mut current_batches))?;

            let chunk_path =
                out_path.join(format!("{}_chunk{:04}.feather", output_prefix, chunk_idx));
            let chunk_path_str = chunk_path.to_str().ok_or_else(|| {
                MaouIOError::IOError(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Invalid output path",
                ))
            })?;
            save_feather(&merged, chunk_path_str)?;
            output_paths.push(chunk_path_str.to_string());

            current_rows = 0;
            chunk_idx += 1;
        }

        current_batches.push(batch);
        current_rows += batch_rows;
    }

    // Flush remaining
    if !current_batches.is_empty() {
        let merged = consolidate_batches(current_batches)?;

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
    use arrow::array::{BinaryArray, BinaryViewArray, Int64Array, LargeBinaryArray};
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
            vec![
                Arc::new(id_array) as ArrayRef,
                Arc::new(value_array) as ArrayRef,
            ],
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
    fn test_load_feather_reads_all_batches_from_file_format() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_multi_load.feather");

        let batch1 = create_test_batch(); // 5 rows
        let batch2 = create_test_batch(); // 5 rows
        let batch3 = create_test_batch(); // 5 rows

        // Save 3 batches using File format (save_feather_batches uses FileWriter)
        save_feather_batches(&[batch1, batch2, batch3], file_path.to_str().unwrap()).unwrap();

        // load_feather must return ALL rows, not just the first batch
        let loaded = load_feather(file_path.to_str().unwrap()).unwrap();
        assert_eq!(
            loaded.num_rows(),
            15,
            "load_feather should concatenate all batches (5+5+5=15 rows)"
        );
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
            vec![
                Arc::new(id_array) as ArrayRef,
                Arc::new(value_array) as ArrayRef,
            ],
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
        let result =
            merge_feather_files(&file_paths, output_dir.to_str().unwrap(), 20, "test").unwrap();

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
        let result =
            merge_feather_files(&file_paths, output_dir.to_str().unwrap(), 12, "test").unwrap();

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

        let result = merge_feather_files(&[], output_dir.to_str().unwrap(), 10, "test").unwrap();

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

        let result =
            merge_feather_files(&file_paths, output_dir.to_str().unwrap(), 100, "test").unwrap();

        let merged = load_feather(&result[0]).unwrap();
        let original = create_test_batch();
        assert_eq!(merged.schema(), original.schema());
    }

    /// `/audit-backlog` 2026-08-12 backlog 行 O7 の回帰テスト．
    ///
    /// 判定の向きが Python と逆 (「Stream か?」を問い既定 File) だったため，
    /// `0xFFFFFFFF` の continuation marker を持たない古い Stream ファイルが
    /// File 形式と誤認され footer エラーになっていた．
    #[test]
    fn test_load_feather_defaults_to_stream_when_magic_is_not_arrow1() {
        use arrow::ipc::writer::StreamWriter;

        let dir = tempdir().unwrap();
        let path = dir.path().join("stream.feather");
        let batch = create_test_batch();

        {
            let file = File::create(&path).unwrap();
            let mut writer = StreamWriter::try_new(file, &batch.schema()).unwrap();
            writer.write(&batch).unwrap();
            writer.finish().unwrap();
        }

        // 前提: File 形式のマジックでは*ない*こと
        let head = std::fs::read(&path).unwrap();
        assert_ne!(&head[..8], ARROW_FILE_MAGIC);

        let loaded = load_feather(path.to_str().unwrap()).unwrap();
        assert_eq!(loaded.num_rows(), batch.num_rows());
        assert_eq!(loaded.schema(), batch.schema());
    }

    /// File 形式は従来どおりマジック一致で File reader に入ること．
    #[test]
    fn test_load_feather_file_format_still_detected_by_magic() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("file.feather");
        let batch = create_test_batch();
        save_feather(&batch, path.to_str().unwrap()).unwrap();

        let head = std::fs::read(&path).unwrap();
        assert_eq!(&head[..8], ARROW_FILE_MAGIC);

        let loaded = load_feather(path.to_str().unwrap()).unwrap();
        assert_eq!(loaded.num_rows(), batch.num_rows());
    }

    /// `load_feather_batches` も同じ入力を受け付けること．
    #[test]
    fn test_load_feather_batches_accepts_stream_format() {
        use arrow::ipc::writer::StreamWriter;

        let dir = tempdir().unwrap();
        let path = dir.path().join("stream_batches.feather");
        let batch = create_test_batch();

        {
            let file = File::create(&path).unwrap();
            let mut writer = StreamWriter::try_new(file, &batch.schema()).unwrap();
            writer.write(&batch).unwrap();
            writer.write(&batch).unwrap();
            writer.finish().unwrap();
        }

        let batches = load_feather_batches(path.to_str().unwrap()).unwrap();
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].num_rows(), batch.num_rows());
    }

    // ========================================================================
    // /audit-backlog 2026-08-13 — backlog 行 N-1 の回帰テスト
    // ========================================================================
    //
    // polars 1.38 は Binary 列を BinaryView で書き，`save_feather`
    // (arrow-rs) は LargeBinary で書く．`pre-process --input-split-rows`
    // は入力を chunk するので，writer の違う `.feather` が同じ入力
    // ディレクトリに混在するとマージが停止していた．

    fn binary_batch(data_type: DataType, values: &[&[u8]]) -> RecordBatch {
        let schema = Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("payload", data_type.clone(), false),
        ]);
        let ids = Int32Array::from((0..values.len() as i32).collect::<Vec<_>>());
        let payload: ArrayRef = match data_type {
            DataType::BinaryView => Arc::new(BinaryViewArray::from(values.to_vec())),
            DataType::LargeBinary => Arc::new(LargeBinaryArray::from(values.to_vec())),
            DataType::Binary => Arc::new(BinaryArray::from(values.to_vec())),
            other => panic!("unsupported test type: {other}"),
        };
        RecordBatch::try_new(Arc::new(schema), vec![Arc::new(ids) as ArrayRef, payload]).unwrap()
    }

    #[test]
    fn consolidate_merges_binary_view_with_large_binary() {
        // 回帰: 以前は
        // "It is not possible to concatenate arrays of different data types
        //  (BinaryView, LargeBinary)" で落ちていた．
        let batches = vec![
            binary_batch(DataType::BinaryView, &[b"aa", b"bb"]),
            binary_batch(DataType::LargeBinary, &[b"cc"]),
        ];

        let merged = consolidate_batches(batches).unwrap();

        assert_eq!(merged.num_rows(), 3);
        assert_eq!(
            merged.schema().field(1).data_type(),
            &DataType::LargeBinary,
            "寄せ先は save_feather が書く非 view 側であること"
        );
    }

    #[test]
    fn consolidate_target_type_does_not_depend_on_input_order() {
        // 1 本目が view でも非 view でも出力の型が同じであること．
        let a = consolidate_batches(vec![
            binary_batch(DataType::BinaryView, &[b"aa"]),
            binary_batch(DataType::LargeBinary, &[b"bb"]),
        ])
        .unwrap();
        let b = consolidate_batches(vec![
            binary_batch(DataType::LargeBinary, &[b"aa"]),
            binary_batch(DataType::BinaryView, &[b"bb"]),
        ])
        .unwrap();

        assert_eq!(a.schema(), b.schema());
    }

    #[test]
    fn consolidate_preserves_payload_bytes_across_writers() {
        let batches = vec![
            binary_batch(DataType::BinaryView, &[b"first", b"second"]),
            binary_batch(DataType::LargeBinary, &[b"third"]),
        ];

        let merged = consolidate_batches(batches).unwrap();
        let payload = merged
            .column(1)
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .expect("寄せ先は LargeBinary");

        assert_eq!(payload.value(0), b"first");
        assert_eq!(payload.value(1), b"second");
        assert_eq!(payload.value(2), b"third");
    }

    #[test]
    fn consolidate_leaves_matching_schemas_untouched() {
        // **trap**: スキーマが一致している入力まで normalize すると，
        // これまで BinaryView のまま出ていた出力が LargeBinary に
        // 変わってしまう．同じ writer だけの入力は従来どおりであること．
        let batches = vec![
            binary_batch(DataType::BinaryView, &[b"aa"]),
            binary_batch(DataType::BinaryView, &[b"bb"]),
        ];

        let merged = consolidate_batches(batches).unwrap();

        assert_eq!(merged.schema().field(1).data_type(), &DataType::BinaryView);
    }

    #[test]
    fn consolidate_still_rejects_unrelated_type_mismatches() {
        // view の話ではない食い違いは黙って寄せない (値が壊れ得る)．
        let int32 = create_test_batch();
        let schema = Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("value", DataType::Int64, false),
        ]);
        let int64 = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(Int64Array::from(vec![1_i64])) as ArrayRef,
                Arc::new(Int64Array::from(vec![10_i64])) as ArrayRef,
            ],
        )
        .unwrap();

        assert!(consolidate_batches(vec![int32, int64]).is_err());
    }

    #[test]
    fn merge_feather_files_accepts_mixed_writers() {
        // 経路まるごと: 型の違う `.feather` が混ざったディレクトリを
        // マージできること．
        let dir = tempdir().unwrap();
        let view_path = dir.path().join("view.feather");
        let large_path = dir.path().join("large.feather");
        save_feather(
            &binary_batch(DataType::BinaryView, &[b"aa", b"bb"]),
            view_path.to_str().unwrap(),
        )
        .unwrap();
        save_feather(
            &binary_batch(DataType::LargeBinary, &[b"cc"]),
            large_path.to_str().unwrap(),
        )
        .unwrap();

        let out_dir = dir.path().join("out");
        let outputs = merge_feather_files(
            &[
                view_path.to_str().unwrap().to_string(),
                large_path.to_str().unwrap().to_string(),
            ],
            out_dir.to_str().unwrap(),
            100,
            "mixed",
        )
        .unwrap();

        assert_eq!(outputs.len(), 1);
        let merged = load_feather(&outputs[0]).unwrap();
        assert_eq!(merged.num_rows(), 3);
    }
}
