//! Sparse array compression utilities for intermediate data storage.
//!
//! This module provides high-performance sparse array operations for move label
//! count arrays which are typically 99% zeros (only ~20 non-zero elements out of 1496).

use crate::error::MaouIOError;

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
/// use maou_io::sparse_array::compress_sparse_array;
///
/// let dense = vec![0, 5, 0, 0, 10, 0];
/// let (indices, values) = compress_sparse_array(&dense).unwrap();
/// assert_eq!(indices, vec![1, 4]);
/// assert_eq!(values, vec![5, 10]);
/// ```
pub fn compress_sparse_array(dense: &[i32]) -> Result<(Vec<u16>, Vec<i32>), MaouIOError> {
    let mut indices = Vec::new();
    let mut values = Vec::new();

    for (idx, &val) in dense.iter().enumerate() {
        if val != 0 {
            // Check if index fits in u16 (max 65535)
            if idx > u16::MAX as usize {
                return Err(MaouIOError::CompressionError(
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
/// use maou_io::sparse_array::expand_sparse_array;
///
/// let indices = vec![1, 4];
/// let values = vec![5, 10];
/// let dense = expand_sparse_array(&indices, &values, 6).unwrap();
/// assert_eq!(dense, vec![0, 5, 0, 0, 10, 0]);
/// ```
pub fn expand_sparse_array(indices: &[u16], values: &[i32], size: usize) -> Result<Vec<i32>, MaouIOError> {
    if indices.len() != values.len() {
        return Err(MaouIOError::CompressionError(
            format!("indices length ({}) != values length ({})", indices.len(), values.len())
        ));
    }

    let mut result = vec![0i32; size];

    for (idx, val) in indices.iter().zip(values.iter()) {
        let idx_usize = *idx as usize;
        if idx_usize >= size {
            return Err(MaouIOError::CompressionError(
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
/// use maou_io::sparse_array::add_sparse_arrays;
///
/// // Array 1: [0, 5, 0, 10] → indices=[1,3], values=[5,10]
/// // Array 2: [0, 3, 7, 0]  → indices=[1,2], values=[3,7]
/// // Sum:     [0, 8, 7, 10] → indices=[1,2,3], values=[8,7,10]
/// let (indices, values) = add_sparse_arrays(
///     &[1, 3], &[5, 10],
///     &[1, 2], &[3, 7]
/// ).unwrap();
/// assert_eq!(indices, vec![1, 2, 3]);
/// assert_eq!(values, vec![8, 7, 10]);
/// ```
pub fn add_sparse_arrays(
    indices1: &[u16],
    values1: &[i32],
    indices2: &[u16],
    values2: &[i32],
) -> Result<(Vec<u16>, Vec<i32>), MaouIOError> {
    if indices1.len() != values1.len() {
        return Err(MaouIOError::CompressionError(
            "indices1 and values1 must have same length".to_string()
        ));
    }
    if indices2.len() != values2.len() {
        return Err(MaouIOError::CompressionError(
            "indices2 and values2 must have same length".to_string()
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

/// Add two dual-track sparse arrays efficiently.
///
/// Merges two sparse arrays that share indices but have two value tracks
/// (label counts and win counts). Uses a single HashMap and sort pass
/// instead of two separate `add_sparse_arrays` calls.
///
/// Filtering: entries are retained when label_value != 0 (even if win_value == 0).
/// This preserves moves that were played but never won.
///
/// # Arguments
/// * `indices1` - First array's non-zero indices
/// * `label_values1` - First array's label count values
/// * `win_values1` - First array's win count values
/// * `indices2` - Second array's non-zero indices
/// * `label_values2` - Second array's label count values
/// * `win_values2` - Second array's win count values
///
/// # Returns
/// * `(Vec<u16>, Vec<i32>, Vec<f32>)` - Merged sparse array (indices, label_values, win_values)
pub fn add_sparse_arrays_dual(
    indices1: &[u16],
    label_values1: &[i32],
    win_values1: &[f32],
    indices2: &[u16],
    label_values2: &[i32],
    win_values2: &[f32],
) -> Result<(Vec<u16>, Vec<i32>, Vec<f32>), MaouIOError> {
    if indices1.len() != label_values1.len() || indices1.len() != win_values1.len() {
        return Err(MaouIOError::CompressionError(
            format!(
                "Array 1 length mismatch: indices={}, label_values={}, win_values={}",
                indices1.len(), label_values1.len(), win_values1.len()
            )
        ));
    }
    if indices2.len() != label_values2.len() || indices2.len() != win_values2.len() {
        return Err(MaouIOError::CompressionError(
            format!(
                "Array 2 length mismatch: indices={}, label_values={}, win_values={}",
                indices2.len(), label_values2.len(), win_values2.len()
            )
        ));
    }

    use std::collections::HashMap;
    let mut merged: HashMap<u16, (i32, f32)> = HashMap::new();

    // Add first array
    for i in 0..indices1.len() {
        let entry = merged.entry(indices1[i]).or_insert((0, 0.0));
        entry.0 += label_values1[i];
        entry.1 += win_values1[i];
    }

    // Add second array
    for i in 0..indices2.len() {
        let entry = merged.entry(indices2[i]).or_insert((0, 0.0));
        entry.0 += label_values2[i];
        entry.1 += win_values2[i];
    }

    // Convert back to sorted arrays, filtering by label_value != 0
    let mut result: Vec<(u16, i32, f32)> = merged
        .into_iter()
        .filter(|(_, (lv, _))| *lv != 0)
        .map(|(idx, (lv, wv))| (idx, lv, wv))
        .collect();
    result.sort_by_key(|(idx, _, _)| *idx);

    let mut result_indices = Vec::with_capacity(result.len());
    let mut result_label_values = Vec::with_capacity(result.len());
    let mut result_win_values = Vec::with_capacity(result.len());
    for (idx, lv, wv) in result {
        result_indices.push(idx);
        result_label_values.push(lv);
        result_win_values.push(wv);
    }

    Ok((result_indices, result_label_values, result_win_values))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compress_sparse_array() {
        let dense = vec![0, 5, 0, 0, 10, 0, -3];
        let (indices, values) = compress_sparse_array(&dense).unwrap();
        assert_eq!(indices, vec![1, 4, 6]);
        assert_eq!(values, vec![5, 10, -3]);
    }

    #[test]
    fn test_compress_all_zeros() {
        let dense = vec![0, 0, 0, 0];
        let (indices, values) = compress_sparse_array(&dense).unwrap();
        assert_eq!(indices.len(), 0);
        assert_eq!(values.len(), 0);
    }

    #[test]
    fn test_expand_sparse_array() {
        let indices = vec![1, 4, 6];
        let values = vec![5, 10, -3];
        let dense = expand_sparse_array(&indices, &values, 8).unwrap();
        assert_eq!(dense, vec![0, 5, 0, 0, 10, 0, -3, 0]);
    }

    #[test]
    fn test_expand_empty() {
        let indices: Vec<u16> = vec![];
        let values: Vec<i32> = vec![];
        let dense = expand_sparse_array(&indices, &values, 5).unwrap();
        assert_eq!(dense, vec![0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_expand_index_out_of_bounds() {
        let indices = vec![1, 10];
        let values = vec![5, 10];
        let result = expand_sparse_array(&indices, &values, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_add_sparse_arrays_no_overlap() {
        let (indices, values) = add_sparse_arrays(
            &[1, 3],
            &[5, 10],
            &[2, 4],
            &[7, 20],
        ).unwrap();
        assert_eq!(indices, vec![1, 2, 3, 4]);
        assert_eq!(values, vec![5, 7, 10, 20]);
    }

    #[test]
    fn test_add_sparse_arrays_with_overlap() {
        let (indices, values) = add_sparse_arrays(
            &[1, 3],
            &[5, 10],
            &[1, 2],
            &[3, 7],
        ).unwrap();
        assert_eq!(indices, vec![1, 2, 3]);
        assert_eq!(values, vec![8, 7, 10]);
    }

    #[test]
    fn test_add_sparse_arrays_cancel_out() {
        // Values cancel out at index 1
        let (indices, values) = add_sparse_arrays(
            &[1, 3],
            &[5, 10],
            &[1, 2],
            &[-5, 7],
        ).unwrap();
        assert_eq!(indices, vec![2, 3]);
        assert_eq!(values, vec![7, 10]);
    }

    #[test]
    fn test_add_empty_arrays() {
        let (indices, values) = add_sparse_arrays(
            &[],
            &[],
            &[],
            &[],
        ).unwrap();
        assert_eq!(indices.len(), 0);
        assert_eq!(values.len(), 0);
    }

    #[test]
    fn test_roundtrip() {
        // Compress → Expand should give original array
        let original = vec![0, 5, 0, 0, 10, 0, -3, 0, 0, 0];
        let (indices, values) = compress_sparse_array(&original).unwrap();
        let expanded = expand_sparse_array(&indices, &values, original.len()).unwrap();
        assert_eq!(expanded, original);
    }

    #[test]
    fn test_add_sparse_arrays_dual_no_overlap() {
        let (indices, label_values, win_values) = add_sparse_arrays_dual(
            &[1, 3], &[5, 10], &[1.0, 2.0],
            &[2, 4], &[7, 20], &[3.0, 4.0],
        ).unwrap();
        assert_eq!(indices, vec![1, 2, 3, 4]);
        assert_eq!(label_values, vec![5, 7, 10, 20]);
        assert_eq!(win_values, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_add_sparse_arrays_dual_with_overlap() {
        let (indices, label_values, win_values) = add_sparse_arrays_dual(
            &[1, 3], &[5, 10], &[1.0, 2.0],
            &[1, 2], &[3, 7], &[0.5, 3.0],
        ).unwrap();
        assert_eq!(indices, vec![1, 2, 3]);
        assert_eq!(label_values, vec![8, 7, 10]);
        assert_eq!(win_values, vec![1.5, 3.0, 2.0]);
    }

    #[test]
    fn test_add_sparse_arrays_dual_win_zero_kept() {
        // win_value=0 should be kept when label_value > 0
        let (indices, label_values, win_values) = add_sparse_arrays_dual(
            &[1, 3], &[5, 10], &[0.0, 0.0],
            &[1], &[3], &[0.0],
        ).unwrap();
        assert_eq!(indices, vec![1, 3]);
        assert_eq!(label_values, vec![8, 10]);
        assert_eq!(win_values, vec![0.0, 0.0]);
    }

    #[test]
    fn test_add_sparse_arrays_dual_label_cancel_out() {
        // label values cancel to 0 → entry removed (win value also removed)
        let (indices, label_values, win_values) = add_sparse_arrays_dual(
            &[1, 3], &[5, 10], &[1.0, 2.0],
            &[1, 2], &[-5, 7], &[0.5, 3.0],
        ).unwrap();
        assert_eq!(indices, vec![2, 3]);
        assert_eq!(label_values, vec![7, 10]);
        assert_eq!(win_values, vec![3.0, 2.0]);
    }

    #[test]
    fn test_add_sparse_arrays_dual_empty() {
        let (indices, label_values, win_values) = add_sparse_arrays_dual(
            &[], &[], &[],
            &[], &[], &[],
        ).unwrap();
        assert_eq!(indices.len(), 0);
        assert_eq!(label_values.len(), 0);
        assert_eq!(win_values.len(), 0);
    }

    #[test]
    fn test_add_sparse_arrays_dual_mismatched_lengths() {
        let result = add_sparse_arrays_dual(
            &[1, 3], &[5], &[1.0, 2.0],
            &[], &[], &[],
        );
        assert!(result.is_err());
    }
}
