use crate::burnpack::{
    base::{
        BurnpackError, FORMAT_VERSION, HEADER_SIZE, MAGIC_NUMBER, magic_range, metadata_size_range,
        version_range,
    },
    reader::BurnpackReader,
    writer::BurnpackWriter,
};

use super::*;
use burn_tensor::{Bytes, DType, TensorData};

#[test]
fn test_reader_from_bytes_empty() {
    // Create empty burnpack data
    let writer = BurnpackWriter::new(Vec::new());
    let bytes = writer.to_bytes().unwrap();

    // Read it back
    let reader = BurnpackReader::from_bytes(bytes).unwrap();

    assert_eq!(reader.metadata().tensors.len(), 0);
    assert!(reader.metadata().metadata.is_empty());
}

#[test]
fn test_reader_from_bytes_with_data() {
    // Create test data
    let snapshot = TensorSnapshot::from_data(
        TensorData::from_bytes_vec(vec![1, 2, 3, 4], vec![2, 2], DType::U8),
        vec!["test_tensor".to_string()],
        vec![],
        burn_core::module::ParamId::new(),
    );

    let writer = BurnpackWriter::new(vec![snapshot]).with_metadata("test", "value");

    let bytes = writer.to_bytes().unwrap();

    // Read it back
    let reader = BurnpackReader::from_bytes(bytes).unwrap();

    assert_eq!(reader.metadata().tensors.len(), 1);
    assert_eq!(
        reader.metadata().metadata.get("test"),
        Some(&"value".to_string())
    );

    // Get tensor data
    let tensor_data = reader.get_tensor_data("test_tensor").unwrap();
    assert_eq!(tensor_data, &[1, 2, 3, 4]);
}

#[test]
fn test_reader_invalid_magic_number() {
    let mut bytes = vec![0u8; 100];
    // Write invalid magic number
    bytes[magic_range()].copy_from_slice(b"NOPE");

    let result = BurnpackReader::from_bytes(Bytes::from_bytes_vec(bytes));
    assert!(matches!(result, Err(BurnpackError::InvalidMagicNumber)));
}

#[test]
fn test_reader_invalid_version() {
    let mut bytes = vec![0u8; 100];
    // Write correct magic but invalid version
    bytes[magic_range()].copy_from_slice(&MAGIC_NUMBER.to_le_bytes());
    bytes[version_range()].copy_from_slice(&999u16.to_le_bytes()); // Invalid version
    bytes[metadata_size_range()].copy_from_slice(&10u32.to_le_bytes()); // Metadata size

    let result = BurnpackReader::from_bytes(Bytes::from_bytes_vec(bytes));
    assert!(matches!(result, Err(BurnpackError::InvalidVersion)));
}

#[test]
fn test_reader_header_too_short() {
    let bytes = vec![0u8; 5]; // Less than HEADER_SIZE

    let result = BurnpackReader::from_bytes(Bytes::from_bytes_vec(bytes));
    assert!(matches!(result, Err(BurnpackError::InvalidHeader)));
}

#[test]
fn test_reader_metadata_truncated() {
    let mut bytes = vec![0u8; HEADER_SIZE + 10];
    // Write valid header
    bytes[magic_range()].copy_from_slice(&MAGIC_NUMBER.to_le_bytes());
    bytes[version_range()].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
    bytes[metadata_size_range()].copy_from_slice(&100u32.to_le_bytes()); // Claims 100 bytes of metadata

    // But only provide 10 bytes after header
    let result = BurnpackReader::from_bytes(Bytes::from_bytes_vec(bytes));
    assert!(matches!(result, Err(BurnpackError::InvalidHeader)));
}

#[test]
fn test_reader_get_tensor_not_found() {
    let writer = BurnpackWriter::new(Vec::new());
    let bytes = writer.to_bytes().unwrap();
    let reader = BurnpackReader::from_bytes(bytes).unwrap();

    let result = reader.get_tensor_data("non_existent");
    assert!(matches!(result, Err(BurnpackError::TensorNotFound(_))));
}

#[test]
fn test_reader_get_tensor_snapshot() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    let snapshot = TensorSnapshot::from_data(
        TensorData::from_bytes_vec(bytes, vec![2, 2], DType::F32),
        vec!["weights".to_string()],
        vec![],
        burn_core::module::ParamId::new(),
    );

    let writer = BurnpackWriter::new(vec![snapshot]);
    let writer_bytes = writer.to_bytes().unwrap();
    let reader = BurnpackReader::from_bytes(writer_bytes).unwrap();

    // Get tensor as snapshot
    let loaded_snapshot = reader.get_tensor_snapshot("weights").unwrap();

    // Verify snapshot metadata
    assert_eq!(loaded_snapshot.full_path(), "weights");
    assert_eq!(loaded_snapshot.dtype, DType::F32);
    assert_eq!(loaded_snapshot.shape, vec![2, 2]);

    // Verify data through closure
    let tensor_data = loaded_snapshot.to_data().unwrap();
    assert_eq!(tensor_data.shape, vec![2, 2]);
}

#[test]
fn test_reader_multiple_tensors() {
    // Add multiple tensors
    let mut snapshots = Vec::new();
    for i in 0..10 {
        let name = format!("tensor_{}", i);
        let data = vec![i as u8; 100];
        let snapshot = TensorSnapshot::from_data(
            TensorData::from_bytes_vec(data, vec![100], DType::U8),
            vec![name.clone()],
            vec![],
            burn_core::module::ParamId::new(),
        );
        snapshots.push(snapshot);
    }

    let writer = BurnpackWriter::new(snapshots);
    let bytes = writer.to_bytes().unwrap();
    let reader = BurnpackReader::from_bytes(bytes).unwrap();

    // Verify all tensors can be read
    for i in 0..10 {
        let name = format!("tensor_{}", i);
        let data = reader.get_tensor_data(&name).unwrap();
        assert_eq!(data.len(), 100);
        assert!(data.iter().all(|&b| b == i as u8));
    }
}

#[test]
fn test_reader_lazy_loading() {
    // Create large tensor
    let size = 1024 * 1024; // 1MB
    let data = vec![42u8; size];
    let snapshot = TensorSnapshot::from_data(
        TensorData::from_bytes_vec(data.clone(), vec![size], DType::U8),
        vec!["large".to_string()],
        vec![],
        burn_core::module::ParamId::new(),
    );

    let writer = BurnpackWriter::new(vec![snapshot]);
    let bytes = writer.to_bytes().unwrap();
    let reader = BurnpackReader::from_bytes(bytes).unwrap();

    // Get snapshot (should be lazy)
    let snapshot = reader.get_tensor_snapshot("large").unwrap();

    // Data should only be accessed when to_data is called
    let tensor_data = snapshot.to_data().unwrap();
    assert_eq!(tensor_data.bytes.len(), size);
    assert!(tensor_data.bytes.iter().all(|&b| b == 42));
}

#[test]
fn test_reader_all_dtypes() {
    // Test all data types
    let test_data = vec![
        (DType::F32, vec![1.0f32.to_le_bytes().to_vec()].concat()),
        (DType::F64, vec![2.0f64.to_le_bytes().to_vec()].concat()),
        (DType::I32, vec![3i32.to_le_bytes().to_vec()].concat()),
        (DType::I64, vec![4i64.to_le_bytes().to_vec()].concat()),
        (DType::U32, vec![5u32.to_le_bytes().to_vec()].concat()),
        (DType::U64, vec![6u64.to_le_bytes().to_vec()].concat()),
        (DType::U8, vec![7u8]),
        (DType::Bool, vec![1u8]),
    ];

    let mut snapshots = Vec::new();
    for (i, (dtype, data)) in test_data.iter().enumerate() {
        let name = format!("tensor_{}", i);
        let snapshot = TensorSnapshot::from_data(
            TensorData::from_bytes_vec(data.clone(), vec![1], *dtype),
            vec![name.clone()],
            vec![],
            burn_core::module::ParamId::new(),
        );
        snapshots.push(snapshot);
    }

    let writer = BurnpackWriter::new(snapshots);
    let bytes = writer.to_bytes().unwrap();
    let reader = BurnpackReader::from_bytes(bytes).unwrap();

    // Verify all dtypes are preserved
    for (i, (expected_dtype, expected_data)) in test_data.iter().enumerate() {
        let name = format!("tensor_{}", i);
        let snapshot = reader.get_tensor_snapshot(&name).unwrap();
        assert_eq!(snapshot.dtype, *expected_dtype);

        let data = reader.get_tensor_data(&name).unwrap();
        assert_eq!(data, expected_data.as_slice());
    }
}

#[test]
fn test_reader_empty_tensor() {
    let snapshot = TensorSnapshot::from_data(
        TensorData::from_bytes_vec(vec![], vec![0], DType::F32),
        vec!["empty".to_string()],
        vec![],
        burn_core::module::ParamId::new(),
    );

    let writer = BurnpackWriter::new(vec![snapshot]);
    let bytes = writer.to_bytes().unwrap();
    let reader = BurnpackReader::from_bytes(bytes).unwrap();

    let data = reader.get_tensor_data("empty").unwrap();
    assert_eq!(data.len(), 0);

    let snapshot = reader.get_tensor_snapshot("empty").unwrap();
    assert_eq!(snapshot.shape, vec![0]);
}

#[cfg(feature = "std")]
#[test]
fn test_reader_from_file() {
    use tempfile::tempdir;

    let dir = tempdir().unwrap();
    let file_path = dir.path().join("test.bpk");

    // Create test file
    let snapshot = TensorSnapshot::from_data(
        TensorData::from_bytes_vec(vec![10, 20, 30], vec![3], DType::U8),
        vec!["file_tensor".to_string()],
        vec![],
        burn_core::module::ParamId::new(),
    );

    let writer = BurnpackWriter::new(vec![snapshot]).with_metadata("from_file_test", "true");

    writer.write_to_file(&file_path).unwrap();

    // Read from file
    let reader = BurnpackReader::from_file(&file_path).unwrap();

    assert_eq!(
        reader.metadata().metadata.get("from_file_test"),
        Some(&"true".to_string())
    );

    let data = reader.get_tensor_data("file_tensor").unwrap();
    assert_eq!(data, &[10, 20, 30]);
}

#[cfg(all(feature = "std", feature = "memmap"))]
#[test]
fn test_reader_from_file_mmap() {
    use tempfile::tempdir;

    let dir = tempdir().unwrap();
    let file_path = dir.path().join("test_mmap.bpk");

    // Create large test file
    let size = 1024 * 1024; // 1MB
    let data = vec![99u8; size];
    let snapshot = TensorSnapshot::from_data(
        TensorData::from_bytes_vec(data, vec![size], DType::U8),
        vec!["large_mmap".to_string()],
        vec![],
        burn_core::module::ParamId::new(),
    );

    let writer = BurnpackWriter::new(vec![snapshot]);
    writer.write_to_file(&file_path).unwrap();

    // Read using mmap
    let reader = BurnpackReader::from_file_mmap(&file_path).unwrap();

    let data = reader.get_tensor_data("large_mmap").unwrap();
    assert_eq!(data.len(), size);
    assert!(data.iter().all(|&b| b == 99));
}

#[cfg(feature = "std")]
#[test]
fn test_reader_from_file_buffered() {
    use tempfile::tempdir;

    let dir = tempdir().unwrap();
    let file_path = dir.path().join("test_buffered.bpk");

    // Create test file
    let snapshot = TensorSnapshot::from_data(
        TensorData::from_bytes_vec(vec![5, 10, 15], vec![3], DType::U8),
        vec!["buffered_tensor".to_string()],
        vec![],
        burn_core::module::ParamId::new(),
    );

    let writer = BurnpackWriter::new(vec![snapshot]);
    writer.write_to_file(&file_path).unwrap();

    // Read using buffered reader
    let reader = BurnpackReader::from_file_buffered(&file_path).unwrap();

    let data = reader.get_tensor_data("buffered_tensor").unwrap();
    assert_eq!(data, &[5, 10, 15]);
}

#[test]
fn test_reader_metadata_access() {
    // Add various metadata using builder pattern
    let writer = BurnpackWriter::new(Vec::new())
        .with_metadata("model_name", "test_model")
        .with_metadata("version", "1.2.3")
        .with_metadata("author", "test_author")
        .with_metadata("description", "A test model");

    let bytes = writer.to_bytes().unwrap();
    let reader = BurnpackReader::from_bytes(bytes).unwrap();

    let metadata = reader.metadata();
    assert_eq!(metadata.metadata.len(), 4);
    assert_eq!(
        metadata.metadata.get("model_name"),
        Some(&"test_model".to_string())
    );
    assert_eq!(metadata.metadata.get("version"), Some(&"1.2.3".to_string()));
    assert_eq!(
        metadata.metadata.get("author"),
        Some(&"test_author".to_string())
    );
    assert_eq!(
        metadata.metadata.get("description"),
        Some(&"A test model".to_string())
    );
}

#[test]
fn test_reader_tensor_iteration() {
    // Add tensors
    let tensor_names = vec!["weights", "bias", "running_mean", "running_var"];
    let mut snapshots = Vec::new();
    for name in &tensor_names {
        let snapshot = TensorSnapshot::from_data(
            TensorData::from_bytes_vec(vec![1, 2, 3, 4], vec![4], DType::U8),
            vec![name.to_string()],
            vec![],
            burn_core::module::ParamId::new(),
        );
        snapshots.push(snapshot);
    }

    let writer = BurnpackWriter::new(snapshots);
    let bytes = writer.to_bytes().unwrap();
    let reader = BurnpackReader::from_bytes(bytes).unwrap();

    // Iterate through all tensors
    let metadata = reader.metadata();
    assert_eq!(metadata.tensors.len(), 4);

    // Check that all expected tensor names are present
    for name in &tensor_names {
        let tensor_desc = metadata.tensors.get(*name).unwrap();
        assert_eq!(tensor_desc.shape, vec![4u64]);
        assert_eq!(tensor_desc.dtype, DType::U8);
    }

    // Verify the keys match the expected names
    let mut actual_names: Vec<_> = metadata.tensors.keys().cloned().collect();
    actual_names.sort();
    let mut expected_names = tensor_names
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    expected_names.sort();
    assert_eq!(actual_names, expected_names);
}

#[test]
fn test_reader_corrupt_metadata() {
    let mut bytes = vec![0u8; 100];

    // Write valid header
    bytes[magic_range()].copy_from_slice(&MAGIC_NUMBER.to_le_bytes());
    bytes[version_range()].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
    bytes[metadata_size_range()].copy_from_slice(&50u32.to_le_bytes()); // 50 bytes of metadata

    // Write garbage as metadata
    for i in HEADER_SIZE..HEADER_SIZE + 50 {
        bytes[i] = 0xFF;
    }

    let result = BurnpackReader::from_bytes(Bytes::from_bytes_vec(bytes));
    assert!(result.is_err());
}

#[test]
fn test_reader_data_offsets_validation() {
    // Add two tensors
    let snapshot1 = TensorSnapshot::from_data(
        TensorData::from_bytes_vec(vec![1, 2, 3, 4], vec![4], DType::U8),
        vec!["tensor1".to_string()],
        vec![],
        burn_core::module::ParamId::new(),
    );
    let snapshot2 = TensorSnapshot::from_data(
        TensorData::from_bytes_vec(vec![5, 6, 7, 8], vec![4], DType::U8),
        vec!["tensor2".to_string()],
        vec![],
        burn_core::module::ParamId::new(),
    );

    let writer = BurnpackWriter::new(vec![snapshot1, snapshot2]);
    let bytes = writer.to_bytes().unwrap();
    let reader = BurnpackReader::from_bytes(bytes).unwrap();

    // Verify offsets don't overlap
    let metadata = reader.metadata();
    let tensor1_desc = metadata.tensors.get("tensor1").unwrap();
    let tensor2_desc = metadata.tensors.get("tensor2").unwrap();

    assert_eq!(tensor1_desc.data_offsets, (0, 4));
    assert_eq!(tensor2_desc.data_offsets, (4, 8));
}

#[test]
fn test_reader_out_of_bounds_error() {
    use crate::burnpack::reader::StorageBackend;
    use alloc::rc::Rc;

    // Create a small data buffer
    let data = Bytes::from_bytes_vec(vec![1, 2, 3, 4, 5]);
    let backend = StorageBackend::Memory(Rc::new(data));

    // Try to read beyond the available data
    let mut buffer = vec![0u8; 10];
    let result = backend.read_into(&mut buffer, 0);

    // Should return an error
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("out of bounds"));
}

#[test]
fn test_reader_offset_overflow_error() {
    use crate::burnpack::reader::StorageBackend;
    use alloc::rc::Rc;

    let data = Bytes::from_bytes_vec(vec![1, 2, 3, 4, 5]);
    let backend = StorageBackend::Memory(Rc::new(data));

    // Try to read with an offset that would overflow
    let mut buffer = vec![0u8; 10];
    let result = backend.read_into(&mut buffer, usize::MAX - 5);

    // Should return an error about overflow
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("overflow"));
}
