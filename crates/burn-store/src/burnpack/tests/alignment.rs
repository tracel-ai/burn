//! Tests for tensor data alignment in burnpack format.
//!
//! These tests verify that tensor data is properly aligned for mmap zero-copy access.

use crate::TensorSnapshot;
use crate::burnpack::{
    base::{
        BurnpackHeader, BurnpackMetadata, HEADER_SIZE, TENSOR_ALIGNMENT, aligned_data_section_start,
    },
    reader::BurnpackReader,
    writer::BurnpackWriter,
};
use burn_core::module::ParamId;
use burn_tensor::{DType, TensorData};

/// Verify that aligned_data_section_start always returns 256-byte aligned values
#[test]
fn test_aligned_data_section_start_is_always_aligned() {
    // Test various metadata sizes
    for metadata_size in 0..1024 {
        let result = aligned_data_section_start(metadata_size);
        assert_eq!(
            result % TENSOR_ALIGNMENT as usize,
            0,
            "aligned_data_section_start({}) = {} is not aligned to {}",
            metadata_size,
            result,
            TENSOR_ALIGNMENT
        );
    }
}

/// Verify data section starts at correct aligned position
#[test]
fn test_data_section_alignment() {
    // Create a tensor
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    let snapshot = TensorSnapshot::from_data(
        TensorData::from_bytes_vec(bytes, vec![4], DType::F32),
        vec!["tensor".to_string()],
        vec![],
        ParamId::new(),
    );

    let writer = BurnpackWriter::new(vec![snapshot]);
    let file_bytes = writer.to_bytes().unwrap();

    // Parse header to get metadata size
    let header = BurnpackHeader::from_bytes(&file_bytes[..HEADER_SIZE]).unwrap();
    let data_section_start = aligned_data_section_start(header.metadata_size as usize);

    // Verify data section starts at 256-byte aligned position
    assert_eq!(
        data_section_start % TENSOR_ALIGNMENT as usize,
        0,
        "Data section start {} is not 256-byte aligned",
        data_section_start
    );

    // Verify the file is large enough
    assert!(
        file_bytes.len() >= data_section_start,
        "File too small: {} < {}",
        file_bytes.len(),
        data_section_start
    );
}

/// Verify that first tensor's absolute file position is 256-byte aligned
#[test]
fn test_first_tensor_absolute_position_aligned() {
    let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let snapshot = TensorSnapshot::from_data(
        TensorData::from_bytes_vec(data, vec![8], DType::U8),
        vec!["first".to_string()],
        vec![],
        ParamId::new(),
    );

    let writer = BurnpackWriter::new(vec![snapshot]);
    let file_bytes = writer.to_bytes().unwrap();

    let header = BurnpackHeader::from_bytes(&file_bytes[..HEADER_SIZE]).unwrap();
    let metadata_end = HEADER_SIZE + header.metadata_size as usize;
    let metadata: BurnpackMetadata =
        ciborium::de::from_reader(&file_bytes[HEADER_SIZE..metadata_end]).unwrap();

    let tensor_desc = metadata.tensors.get("first").unwrap();
    let data_section_start = aligned_data_section_start(header.metadata_size as usize);

    // Absolute file position of first tensor
    let absolute_pos = data_section_start + tensor_desc.data_offsets.0 as usize;

    assert_eq!(
        absolute_pos % TENSOR_ALIGNMENT as usize,
        0,
        "First tensor absolute position {} is not 256-byte aligned",
        absolute_pos
    );
}

/// Verify that all tensors in a multi-tensor file have 256-byte aligned absolute positions
#[test]
fn test_all_tensors_absolute_positions_aligned() {
    // Create multiple tensors of different sizes (all U8 to simplify shape calculation)
    let tensors = vec![
        ("tensor_a", vec![1u8, 2, 3]), // 3 bytes
        ("tensor_b", vec![0u8; 16]),   // 16 bytes
        ("tensor_c", vec![0u8; 64]),   // 64 bytes
        ("tensor_d", vec![42u8]),      // 1 byte
        ("tensor_e", vec![0u8; 400]),  // 400 bytes
    ];

    let snapshots: Vec<TensorSnapshot> = tensors
        .into_iter()
        .map(|(name, data)| {
            let len = data.len();
            TensorSnapshot::from_data(
                TensorData::from_bytes_vec(data, vec![len], DType::U8),
                vec![name.to_string()],
                vec![],
                ParamId::new(),
            )
        })
        .collect();

    let writer = BurnpackWriter::new(snapshots);
    let file_bytes = writer.to_bytes().unwrap();

    let header = BurnpackHeader::from_bytes(&file_bytes[..HEADER_SIZE]).unwrap();
    let metadata_end = HEADER_SIZE + header.metadata_size as usize;
    let metadata: BurnpackMetadata =
        ciborium::de::from_reader(&file_bytes[HEADER_SIZE..metadata_end]).unwrap();

    let data_section_start = aligned_data_section_start(header.metadata_size as usize);

    // Check every tensor has aligned absolute position
    for (name, desc) in &metadata.tensors {
        let absolute_pos = data_section_start + desc.data_offsets.0 as usize;
        assert_eq!(
            absolute_pos % TENSOR_ALIGNMENT as usize,
            0,
            "Tensor '{}' at absolute position {} is not 256-byte aligned (offset in data section: {})",
            name,
            absolute_pos,
            desc.data_offsets.0
        );
    }
}

/// Test edge case: metadata size that results in no padding needed
#[test]
fn test_alignment_with_minimal_padding() {
    // We can't control metadata size directly, but we can verify the math works
    // When HEADER_SIZE + metadata_size is already a multiple of 256, no padding needed
    let aligned_metadata_size = TENSOR_ALIGNMENT as usize - HEADER_SIZE; // 256 - 10 = 246

    let result = aligned_data_section_start(aligned_metadata_size);
    assert_eq!(result, TENSOR_ALIGNMENT as usize); // Should be exactly 256

    // One byte more should still round up to 256
    let result_plus_one = aligned_data_section_start(aligned_metadata_size + 1);
    assert_eq!(result_plus_one, 2 * TENSOR_ALIGNMENT as usize); // Should be 512
}

/// Verify padding bytes in the file are zeros
#[test]
fn test_padding_bytes_are_zeros() {
    let data: Vec<u8> = vec![0xAA; 16]; // Distinctive pattern
    let snapshot = TensorSnapshot::from_data(
        TensorData::from_bytes_vec(data.clone(), vec![16], DType::U8),
        vec!["tensor".to_string()],
        vec![],
        ParamId::new(),
    );

    let writer = BurnpackWriter::new(vec![snapshot]);
    let file_bytes = writer.to_bytes().unwrap();

    let header = BurnpackHeader::from_bytes(&file_bytes[..HEADER_SIZE]).unwrap();
    let metadata_end = HEADER_SIZE + header.metadata_size as usize;
    let data_section_start = aligned_data_section_start(header.metadata_size as usize);

    // Check padding between metadata and data section
    if data_section_start > metadata_end {
        let padding = &file_bytes[metadata_end..data_section_start];
        assert!(
            padding.iter().all(|&b| b == 0),
            "Padding bytes between metadata and data section contain non-zero values"
        );
    }
}

/// Verify alignment is sufficient for all primitive types
/// 256-byte alignment is a multiple of all primitive type alignments:
/// - f64/i64/u64: 8 bytes
/// - f32/i32/u32: 4 bytes
/// - f16/bf16/i16/u16: 2 bytes
/// - i8/u8/bool: 1 byte
#[test]
#[allow(clippy::modulo_one)]
fn test_alignment_covers_all_primitive_types() {
    // 256 must be divisible by all common alignments
    assert_eq!(
        TENSOR_ALIGNMENT % 8,
        0,
        "256 not divisible by 8 (f64 alignment)"
    );
    assert_eq!(
        TENSOR_ALIGNMENT % 4,
        0,
        "256 not divisible by 4 (f32 alignment)"
    );
    assert_eq!(
        TENSOR_ALIGNMENT % 2,
        0,
        "256 not divisible by 2 (f16 alignment)"
    );
    assert_eq!(
        TENSOR_ALIGNMENT % 1,
        0,
        "256 not divisible by 1 (u8 alignment)"
    );
}

/// Verify that tensor data can be read correctly after alignment
#[test]
fn test_aligned_tensor_data_readable() {
    // Create f32 tensor
    let f32_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let f32_bytes: Vec<u8> = f32_data.iter().flat_map(|f| f.to_le_bytes()).collect();

    let snapshot = TensorSnapshot::from_data(
        TensorData::from_bytes_vec(f32_bytes.clone(), vec![4], DType::F32),
        vec!["floats".to_string()],
        vec![],
        ParamId::new(),
    );

    let writer = BurnpackWriter::new(vec![snapshot]);
    let file_bytes = writer.to_bytes().unwrap();

    let header = BurnpackHeader::from_bytes(&file_bytes[..HEADER_SIZE]).unwrap();
    let metadata_end = HEADER_SIZE + header.metadata_size as usize;
    let metadata: BurnpackMetadata =
        ciborium::de::from_reader(&file_bytes[HEADER_SIZE..metadata_end]).unwrap();

    let tensor_desc = metadata.tensors.get("floats").unwrap();
    let data_section_start = aligned_data_section_start(header.metadata_size as usize);

    let start = data_section_start + tensor_desc.data_offsets.0 as usize;
    let end = data_section_start + tensor_desc.data_offsets.1 as usize;
    let tensor_bytes = &file_bytes[start..end];

    // Verify the bytes match what we wrote
    assert_eq!(tensor_bytes, f32_bytes.as_slice());

    // Verify we can interpret them as floats
    let mut floats = Vec::new();
    for chunk in tensor_bytes.chunks_exact(4) {
        floats.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    assert_eq!(floats, f32_data);
}

/// Verify alignment works with f64 data
#[test]
fn test_aligned_f64_tensor_data_readable() {
    let f64_data = vec![1.0f64, 2.0, 3.0, 4.0];
    let f64_bytes: Vec<u8> = f64_data.iter().flat_map(|f| f.to_le_bytes()).collect();

    let snapshot = TensorSnapshot::from_data(
        TensorData::from_bytes_vec(f64_bytes.clone(), vec![4], DType::F64),
        vec!["doubles".to_string()],
        vec![],
        ParamId::new(),
    );

    let writer = BurnpackWriter::new(vec![snapshot]);
    let file_bytes = writer.to_bytes().unwrap();

    let header = BurnpackHeader::from_bytes(&file_bytes[..HEADER_SIZE]).unwrap();
    let metadata_end = HEADER_SIZE + header.metadata_size as usize;
    let metadata: BurnpackMetadata =
        ciborium::de::from_reader(&file_bytes[HEADER_SIZE..metadata_end]).unwrap();

    let tensor_desc = metadata.tensors.get("doubles").unwrap();
    let data_section_start = aligned_data_section_start(header.metadata_size as usize);

    let start = data_section_start + tensor_desc.data_offsets.0 as usize;
    let end = data_section_start + tensor_desc.data_offsets.1 as usize;
    let tensor_bytes = &file_bytes[start..end];

    // Verify the bytes match
    assert_eq!(tensor_bytes, f64_bytes.as_slice());

    // Verify we can interpret them as doubles
    let mut doubles = Vec::new();
    for chunk in tensor_bytes.chunks_exact(8) {
        doubles.push(f64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]));
    }
    assert_eq!(doubles, f64_data);
}

/// Test round-trip preserves alignment (write then read)
#[test]
fn test_round_trip_maintains_alignment() {
    let f32_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let f32_bytes: Vec<u8> = f32_data.iter().flat_map(|f| f.to_le_bytes()).collect();

    let snapshot = TensorSnapshot::from_data(
        TensorData::from_bytes_vec(f32_bytes, vec![2, 4], DType::F32),
        vec!["matrix".to_string()],
        vec![],
        ParamId::new(),
    );

    // Write
    let writer = BurnpackWriter::new(vec![snapshot]);
    let file_bytes = writer.to_bytes().unwrap();

    // Read back
    let reader = BurnpackReader::from_bytes(file_bytes.clone()).unwrap();
    let snapshots = reader.get_snapshots().unwrap();

    assert_eq!(snapshots.len(), 1);
    let loaded = &snapshots[0];
    assert_eq!(loaded.full_path(), "matrix");

    // Verify the loaded data is correct
    let tensor_data = loaded.to_data().unwrap();
    let mut loaded_floats = Vec::new();
    for chunk in tensor_data.bytes.chunks_exact(4) {
        loaded_floats.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    assert_eq!(loaded_floats, f32_data);
}

/// Test that tensor offsets within data section are also aligned
#[test]
fn test_tensor_relative_offsets_are_aligned() {
    // Create several small tensors to force multiple alignment padding
    let tensors: Vec<_> = (0..5)
        .map(|i| {
            let data = vec![i as u8; 7]; // 7 bytes each - not aligned
            TensorSnapshot::from_data(
                TensorData::from_bytes_vec(data, vec![7], DType::U8),
                vec![format!("tensor_{}", i)],
                vec![],
                ParamId::new(),
            )
        })
        .collect();

    let writer = BurnpackWriter::new(tensors);
    let file_bytes = writer.to_bytes().unwrap();

    let header = BurnpackHeader::from_bytes(&file_bytes[..HEADER_SIZE]).unwrap();
    let metadata_end = HEADER_SIZE + header.metadata_size as usize;
    let metadata: BurnpackMetadata =
        ciborium::de::from_reader(&file_bytes[HEADER_SIZE..metadata_end]).unwrap();

    // All tensor start offsets within data section should be multiples of 256
    for (name, desc) in &metadata.tensors {
        assert_eq!(
            desc.data_offsets.0 % TENSOR_ALIGNMENT,
            0,
            "Tensor '{}' relative offset {} is not 256-byte aligned",
            name,
            desc.data_offsets.0
        );
    }
}

#[cfg(feature = "std")]
mod file_tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    /// Test alignment is preserved when writing to and reading from file
    #[test]
    fn test_file_io_preserves_alignment() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("aligned.bpk");

        let f32_data = [1.0f32, 2.0, 3.0, 4.0];
        let f32_bytes: Vec<u8> = f32_data.iter().flat_map(|f| f.to_le_bytes()).collect();

        let snapshot = TensorSnapshot::from_data(
            TensorData::from_bytes_vec(f32_bytes, vec![4], DType::F32),
            vec!["floats".to_string()],
            vec![],
            ParamId::new(),
        );

        // Write to file
        let writer = BurnpackWriter::new(vec![snapshot]);
        writer.write_to_file(&file_path).unwrap();

        // Read file bytes directly
        let file_bytes = fs::read(&file_path).unwrap();

        let header = BurnpackHeader::from_bytes(&file_bytes[..HEADER_SIZE]).unwrap();
        let metadata_end = HEADER_SIZE + header.metadata_size as usize;
        let metadata: BurnpackMetadata =
            ciborium::de::from_reader(&file_bytes[HEADER_SIZE..metadata_end]).unwrap();

        let tensor_desc = metadata.tensors.get("floats").unwrap();
        let data_section_start = aligned_data_section_start(header.metadata_size as usize);
        let absolute_pos = data_section_start + tensor_desc.data_offsets.0 as usize;

        assert_eq!(
            absolute_pos % TENSOR_ALIGNMENT as usize,
            0,
            "Tensor absolute position in file {} is not 256-byte aligned",
            absolute_pos
        );

        // Verify data is correct
        let start = data_section_start + tensor_desc.data_offsets.0 as usize;
        let end = data_section_start + tensor_desc.data_offsets.1 as usize;
        let tensor_bytes = &file_bytes[start..end];

        let mut floats = Vec::new();
        for chunk in tensor_bytes.chunks_exact(4) {
            floats.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        assert_eq!(floats, vec![1.0f32, 2.0, 3.0, 4.0]);
    }
}
