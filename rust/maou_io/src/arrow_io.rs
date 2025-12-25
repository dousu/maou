use arrow::array::RecordBatch;
use arrow::ipc::reader::FileReader;
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
/// Args:
///     file_path: Input file path (.feather extension)
///
/// Returns:
///     Ok(RecordBatch) on success，Err(MaouIOError) on failure
pub fn load_feather(file_path: &str) -> Result<RecordBatch, MaouIOError> {
    let path = Path::new(file_path);
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut reader = FileReader::try_new(reader, None)?;

    // Read first (and assumed only) batch
    match reader.next() {
        Some(Ok(batch)) => Ok(batch),
        Some(Err(e)) => Err(MaouIOError::ArrowError(e)),
        None => Err(MaouIOError::IOError(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "Empty file: no record batches found",
        ))),
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
}
