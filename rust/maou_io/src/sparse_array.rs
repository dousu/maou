//! Sparse array compression utilities for intermediate data storage.
//!
//! This module provides high-performance sparse array operations for move label
//! count arrays which are typically 99% zeros (only ~20 non-zero elements out of 1496).

use pyo3::prelude::*;

/// Compress a dense integer array to sparse format (indices + values).
///
/// Returns a tuple of (indices, values) where indices are u16 (max 65535)
/// and values are i32．Only non-zero elements are stored．
///
/// # Arguments
/// * `dense` - Dense array with mostly zeros
///
/// # Returns
/// * `(Vec<u16>, Vec<i32>)` - (non-zero indices, corresponding values)
///
/// # Example
/// ```
/// let dense = vec![0, 5, 0, 0, 10, 0];
/// let (indices, values) = compress_sparse_array_rust(dense);
/// assert_eq!(indices, vec![1, 4]);
/// assert_eq!(values, vec![5, 10]);
/// ```
#[pyfunction]
pub fn compress_sparse_array_rust(dense: Vec<i32>) -> PyResult<(Vec<u16>, Vec<i32>)> {
    let mut indices = Vec::new();
    let mut values = Vec::new();

    for (idx, &val) in dense.iter().enumerate() {
        if val != 0 {
            // Check if index fits in u16 (max 65535)
            if idx > u16::MAX as usize {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("Index {} exceeds u16::MAX (65535)", idx)
                ));
            }
            indices.push(idx as u16);
            values.push(val);
        }
    }

    Ok((indices, values))
}

/// Expand sparse format (indices + values) back to dense array.
///
/// Creates a zero-filled array of given size and fills in non-zero values
/// at the specified indices．
///
/// # Arguments
/// * `indices` - Non-zero element positions (u16)
/// * `values` - Corresponding values (i32)
/// * `size` - Target dense array size
///
/// # Returns
/// * `Vec<i32>` - Dense array with zeros and filled values
///
/// # Example
/// ```
/// let indices = vec![1, 4];
/// let values = vec![5, 10];
/// let dense = expand_sparse_array_rust(indices, values, 6);
/// assert_eq!(dense, vec![0, 5, 0, 0, 10, 0]);
/// ```
#[pyfunction]
pub fn expand_sparse_array_rust(indices: Vec<u16>, values: Vec<i32>, size: usize) -> PyResult<Vec<i32>> {
    if indices.len() != values.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("indices length ({}) != values length ({})", indices.len(), values.len())
        ));
    }

    let mut result = vec![0i32; size];

    for (idx, val) in indices.iter().zip(values.iter()) {
        let idx_usize = *idx as usize;
        if idx_usize >= size {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                format!("Index {} out of bounds for array of size {}", idx_usize, size)
            ));
        }
        result[idx_usize] = *val;
    }

    Ok(result)
}

/// Add two sparse arrays efficiently without converting to dense format.
///
/// Merges two sparse arrays by combining their non-zero elements．
/// If both arrays have a non-zero value at the same index，values are summed．
///
/// This is used for UPSERT aggregation in the intermediate store．
///
/// # Arguments
/// * `indices1` - First array's non-zero indices
/// * `values1` - First array's non-zero values
/// * `indices2` - Second array's non-zero indices
/// * `values2` - Second array's non-zero values
///
/// # Returns
/// * `(Vec<u16>, Vec<i32>)` - Merged sparse array (indices, values)
///
/// # Example
/// ```
/// // Array 1: [0, 5, 0, 10] → indices=[1,3], values=[5,10]
/// // Array 2: [0, 3, 7, 0]  → indices=[1,2], values=[3,7]
/// // Sum:     [0, 8, 7, 10] → indices=[1,2,3], values=[8,7,10]
/// let (indices, values) = add_sparse_arrays_rust(
///     vec![1, 3], vec![5, 10],
///     vec![1, 2], vec![3, 7]
/// );
/// assert_eq!(indices, vec![1, 2, 3]);
/// assert_eq!(values, vec![8, 7, 10]);
/// ```
#[pyfunction]
pub fn add_sparse_arrays_rust(
    indices1: Vec<u16>,
    values1: Vec<i32>,
    indices2: Vec<u16>,
    values2: Vec<i32>,
) -> PyResult<(Vec<u16>, Vec<i32>)> {
    if indices1.len() != values1.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "indices1 and values1 must have same length"
        ));
    }
    if indices2.len() != values2.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "indices2 and values2 must have same length"
        ));
    }

    // Use HashMap for efficient merging
    use std::collections::HashMap;
    let mut merged: HashMap<u16, i32> = HashMap::new();

    // Add first array
    for (idx, val) in indices1.iter().zip(values1.iter()) {
        *merged.entry(*idx).or_insert(0) += val;
    }

    // Add second array
    for (idx, val) in indices2.iter().zip(values2.iter()) {
        *merged.entry(*idx).or_insert(0) += val;
    }

    // Convert back to sorted (indices, values)
    let mut result: Vec<(u16, i32)> = merged.into_iter().collect();
    result.sort_by_key(|(idx, _)| *idx);

    // Remove zeros (in case values cancelled out)
    result.retain(|(_, val)| *val != 0);

    let (result_indices, result_values): (Vec<u16>, Vec<i32>) = result.into_iter().unzip();

    Ok((result_indices, result_values))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compress_sparse_array() {
        let dense = vec![0, 5, 0, 0, 10, 0, -3];
        let (indices, values) = compress_sparse_array_rust(dense).unwrap();
        assert_eq!(indices, vec![1, 4, 6]);
        assert_eq!(values, vec![5, 10, -3]);
    }

    #[test]
    fn test_compress_all_zeros() {
        let dense = vec![0, 0, 0, 0];
        let (indices, values) = compress_sparse_array_rust(dense).unwrap();
        assert_eq!(indices.len(), 0);
        assert_eq!(values.len(), 0);
    }

    #[test]
    fn test_expand_sparse_array() {
        let indices = vec![1, 4, 6];
        let values = vec![5, 10, -3];
        let dense = expand_sparse_array_rust(indices, values, 8).unwrap();
        assert_eq!(dense, vec![0, 5, 0, 0, 10, 0, -3, 0]);
    }

    #[test]
    fn test_expand_empty() {
        let indices: Vec<u16> = vec![];
        let values: Vec<i32> = vec![];
        let dense = expand_sparse_array_rust(indices, values, 5).unwrap();
        assert_eq!(dense, vec![0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_expand_index_out_of_bounds() {
        let indices = vec![1, 10];
        let values = vec![5, 10];
        let result = expand_sparse_array_rust(indices, values, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_add_sparse_arrays_no_overlap() {
        let (indices, values) = add_sparse_arrays_rust(
            vec![1, 3],
            vec![5, 10],
            vec![2, 4],
            vec![7, 20],
        ).unwrap();
        assert_eq!(indices, vec![1, 2, 3, 4]);
        assert_eq!(values, vec![5, 7, 10, 20]);
    }

    #[test]
    fn test_add_sparse_arrays_with_overlap() {
        let (indices, values) = add_sparse_arrays_rust(
            vec![1, 3],
            vec![5, 10],
            vec![1, 2],
            vec![3, 7],
        ).unwrap();
        assert_eq!(indices, vec![1, 2, 3]);
        assert_eq!(values, vec![8, 7, 10]);
    }

    #[test]
    fn test_add_sparse_arrays_cancel_out() {
        // Values cancel out at index 1
        let (indices, values) = add_sparse_arrays_rust(
            vec![1, 3],
            vec![5, 10],
            vec![1, 2],
            vec![-5, 7],
        ).unwrap();
        assert_eq!(indices, vec![2, 3]);
        assert_eq!(values, vec![7, 10]);
    }

    #[test]
    fn test_add_empty_arrays() {
        let (indices, values) = add_sparse_arrays_rust(
            vec![],
            vec![],
            vec![],
            vec![],
        ).unwrap();
        assert_eq!(indices.len(), 0);
        assert_eq!(values.len(), 0);
    }

    #[test]
    fn test_roundtrip() {
        // Compress → Expand should give original array
        let original = vec![0, 5, 0, 0, 10, 0, -3, 0, 0, 0];
        let (indices, values) = compress_sparse_array_rust(original.clone()).unwrap();
        let expanded = expand_sparse_array_rust(indices, values, original.len()).unwrap();
        assert_eq!(expanded, original);
    }
}
