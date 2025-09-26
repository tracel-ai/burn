use super::super::store::{
    BurnpackError, BurnpackHeader, BurnpackReader, BurnpackWriter, FORMAT_VERSION, MAGIC_NUMBER,
};
use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;

#[test]
fn test_header_serialization() {
    let header = BurnpackHeader::new(12345);

    // Check fields
    assert_eq!(header.magic, MAGIC_NUMBER);
    assert_eq!(header.version, FORMAT_VERSION);
    assert_eq!(header.metadata_size, 12345);

    // Serialize to bytes
    let bytes = header.to_bytes();
    assert_eq!(bytes.len(), 10);

    // Deserialize back
    let header2 = BurnpackHeader::from_bytes(&bytes).unwrap();
    assert_eq!(header2.magic, header.magic);
    assert_eq!(header2.version, header.version);
    assert_eq!(header2.metadata_size, header.metadata_size);
}

#[test]
fn test_header_invalid_magic() {
    let mut bytes = [0u8; 10];
    // Write wrong magic number
    bytes[0..4].copy_from_slice(&[0x00, 0x00, 0x00, 0x00]);

    let result = BurnpackHeader::from_bytes(&bytes);
    match result {
        Err(BurnpackError::InvalidMagicNumber) => {}
        _ => panic!("Expected InvalidMagicNumber error"),
    }
}

#[test]
fn test_header_insufficient_bytes() {
    let bytes = [0u8; 5]; // Too short

    let result = BurnpackHeader::from_bytes(&bytes);
    match result {
        Err(BurnpackError::InvalidHeader) => {}
        _ => panic!("Expected InvalidHeader error"),
    }
}

#[test]
fn test_writer_single_tensor() {
    let mut writer = BurnpackWriter::new();

    // Add a single tensor
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let data_bytes =
        unsafe { core::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };

    writer.add_tensor(
        "test_tensor".to_string(),
        burn_tensor::DType::F32,
        vec![2, 2],
        data_bytes,
    );

    // Add metadata
    writer.add_metadata("format".to_string(), "burnpack".to_string());
    writer.add_metadata("version".to_string(), "1.0".to_string());

    // Convert to bytes
    let bytes = writer.to_bytes().unwrap();

    // Verify we can read it back
    let reader = BurnpackReader::from_bytes(bytes).unwrap();
    assert_eq!(reader.tensor_names(), vec!["test_tensor"]);

    let tensor_data = reader.get_tensor_data("test_tensor").unwrap();
    assert_eq!(tensor_data.len(), 16); // 4 floats * 4 bytes

    // Check metadata
    assert_eq!(reader.metadata().metadata["format"], "burnpack");
    assert_eq!(reader.metadata().metadata["version"], "1.0");
}

#[test]
fn test_writer_multiple_tensors() {
    let mut writer = BurnpackWriter::new();

    // Add multiple tensors
    let data1 = vec![1.0f32, 2.0];
    let data1_bytes =
        unsafe { core::slice::from_raw_parts(data1.as_ptr() as *const u8, data1.len() * 4) };

    let data2 = vec![3.0f64, 4.0, 5.0];
    let data2_bytes =
        unsafe { core::slice::from_raw_parts(data2.as_ptr() as *const u8, data2.len() * 8) };

    writer.add_tensor(
        "weights".to_string(),
        burn_tensor::DType::F32,
        vec![2],
        data1_bytes,
    );

    writer.add_tensor(
        "bias".to_string(),
        burn_tensor::DType::F64,
        vec![3],
        data2_bytes,
    );

    // Convert to bytes
    let bytes = writer.to_bytes().unwrap();

    // Read back
    let reader = BurnpackReader::from_bytes(bytes).unwrap();

    let names = reader.tensor_names();
    assert_eq!(names.len(), 2);
    assert!(names.contains(&"weights"));
    assert!(names.contains(&"bias"));

    // Check tensor data
    let weights_data = reader.get_tensor_data("weights").unwrap();
    assert_eq!(weights_data.len(), 8); // 2 f32s * 4 bytes

    let bias_data = reader.get_tensor_data("bias").unwrap();
    assert_eq!(bias_data.len(), 24); // 3 f64s * 8 bytes

    // Check metadata structure
    let metadata = reader.metadata();
    assert_eq!(metadata.tensors.len(), 2);

    // Find weights descriptor
    let weights_desc = metadata
        .tensors
        .iter()
        .find(|t| t.name == "weights")
        .unwrap();
    assert_eq!(weights_desc.dtype, burn_tensor::DType::F32);
    assert_eq!(weights_desc.shape, vec![2]);
    assert_eq!(weights_desc.data_offsets.1 - weights_desc.data_offsets.0, 8);

    // Find bias descriptor
    let bias_desc = metadata.tensors.iter().find(|t| t.name == "bias").unwrap();
    assert_eq!(bias_desc.dtype, burn_tensor::DType::F64);
    assert_eq!(bias_desc.shape, vec![3]);
    assert_eq!(bias_desc.data_offsets.1 - bias_desc.data_offsets.0, 24);
}

#[test]
fn test_tensor_not_found() {
    let mut writer = BurnpackWriter::new();
    writer.add_tensor(
        "existing".to_string(),
        burn_tensor::DType::F32,
        vec![1],
        &[0, 0, 0, 0],
    );

    let bytes = writer.to_bytes().unwrap();
    let reader = BurnpackReader::from_bytes(bytes).unwrap();

    match reader.get_tensor_data("nonexistent") {
        Err(BurnpackError::TensorNotFound(name)) => {
            assert_eq!(name, "nonexistent");
        }
        _ => panic!("Expected TensorNotFound error"),
    }
}

#[test]
fn test_empty_file() {
    let writer = BurnpackWriter::new();
    let bytes = writer.to_bytes().unwrap();

    let reader = BurnpackReader::from_bytes(bytes).unwrap();
    assert_eq!(reader.tensor_names().len(), 0);
    assert!(reader.metadata().metadata.is_empty());
}

#[test]
fn test_metadata_only() {
    let mut writer = BurnpackWriter::new();
    writer.add_metadata("model_name".to_string(), "test_model".to_string());
    writer.add_metadata("author".to_string(), "test_author".to_string());

    let bytes = writer.to_bytes().unwrap();
    let reader = BurnpackReader::from_bytes(bytes).unwrap();

    assert_eq!(reader.tensor_names().len(), 0);
    assert_eq!(reader.metadata().metadata["model_name"], "test_model");
    assert_eq!(reader.metadata().metadata["author"], "test_author");
}

#[test]
fn test_large_metadata() {
    let mut writer = BurnpackWriter::new();

    // Add many metadata entries
    for i in 0..1000 {
        writer.add_metadata(format!("key_{}", i), format!("value_{}", i));
    }

    let bytes = writer.to_bytes().unwrap();
    let reader = BurnpackReader::from_bytes(bytes).unwrap();

    assert_eq!(reader.metadata().metadata.len(), 1000);
    assert_eq!(reader.metadata().metadata["key_500"], "value_500");
}

#[test]
fn test_various_tensor_shapes() {
    let mut writer = BurnpackWriter::new();

    // Scalar
    writer.add_tensor(
        "scalar".to_string(),
        burn_tensor::DType::F32,
        vec![],
        &[0, 0, 0, 0],
    );

    // 1D tensor
    writer.add_tensor(
        "vector".to_string(),
        burn_tensor::DType::F32,
        vec![5],
        &vec![0u8; 20],
    );

    // 2D tensor
    writer.add_tensor(
        "matrix".to_string(),
        burn_tensor::DType::F32,
        vec![3, 4],
        &vec![0u8; 48],
    );

    // 4D tensor (common in CNNs)
    writer.add_tensor(
        "conv_weights".to_string(),
        burn_tensor::DType::F32,
        vec![64, 3, 7, 7],
        &vec![0u8; 64 * 3 * 7 * 7 * 4],
    );

    let bytes = writer.to_bytes().unwrap();
    let reader = BurnpackReader::from_bytes(bytes).unwrap();

    let metadata = reader.metadata();

    // Check scalar
    let scalar = metadata
        .tensors
        .iter()
        .find(|t| t.name == "scalar")
        .unwrap();
    assert_eq!(scalar.shape, Vec::<u64>::new());

    // Check vector
    let vector = metadata
        .tensors
        .iter()
        .find(|t| t.name == "vector")
        .unwrap();
    assert_eq!(vector.shape, vec![5]);

    // Check matrix
    let matrix = metadata
        .tensors
        .iter()
        .find(|t| t.name == "matrix")
        .unwrap();
    assert_eq!(matrix.shape, vec![3, 4]);

    // Check 4D tensor
    let conv = metadata
        .tensors
        .iter()
        .find(|t| t.name == "conv_weights")
        .unwrap();
    assert_eq!(conv.shape, vec![64, 3, 7, 7]);
}

#[cfg(feature = "std")]
#[test]
fn test_file_operations() {
    use std::fs;
    use tempfile::tempdir;

    let dir = tempdir().unwrap();
    let file_path = dir.path().join("test.burnpack");

    // Write to file
    {
        let mut writer = BurnpackWriter::new();
        writer.add_tensor(
            "test".to_string(),
            burn_tensor::DType::I32,
            vec![2, 2],
            &vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        );
        writer.add_metadata("test".to_string(), "metadata".to_string());

        writer.write_to_file(&file_path).unwrap();
    }

    // Verify file exists
    assert!(file_path.exists());

    // Read from file
    {
        let reader = BurnpackReader::from_file(&file_path).unwrap();
        assert_eq!(reader.tensor_names(), vec!["test"]);
        assert_eq!(reader.metadata().metadata["test"], "metadata");

        let data = reader.get_tensor_data("test").unwrap();
        assert_eq!(data.len(), 16);
    }

    // Clean up
    fs::remove_file(file_path).unwrap();
}

#[test]
fn test_version_compatibility() {
    // Create a header with current version
    let header = BurnpackHeader::new(100);
    let bytes = header.to_bytes();

    // Should succeed with current version
    let result = BurnpackHeader::from_bytes(&bytes);
    assert!(result.is_ok());

    // Test with future version (should fail in real implementation)
    // For now, we just verify the version field is correctly set
    let header = result.unwrap();
    assert_eq!(header.version, FORMAT_VERSION);
}
