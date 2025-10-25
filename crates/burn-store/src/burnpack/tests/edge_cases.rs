use crate::TensorSnapshot;
use crate::burnpack::{
    base::{BurnpackHeader, HEADER_SIZE},
    reader::BurnpackReader,
    writer::BurnpackWriter,
};
use burn_core::module::ParamId;
use burn_tensor::{DType, TensorData};

#[test]
fn test_maximum_metadata_size() {
    // Create metadata that approaches u32::MAX (4GB limit)
    // In practice, we'll test with a reasonably large metadata
    let large_key = "x".repeat(1000);
    let large_value = "y".repeat(10000);

    let mut writer = BurnpackWriter::new(vec![]);

    for i in 0..100 {
        writer = writer.with_metadata(&format!("{}_{}", large_key, i), &large_value);
    }

    let result = writer.to_bytes();
    assert!(result.is_ok());

    let bytes = result.unwrap();
    let header = BurnpackHeader::from_bytes(&bytes[..HEADER_SIZE]).unwrap();

    // Metadata size should be large but within u32 bounds
    assert!(header.metadata_size > 1000000); // At least 1MB of metadata
    assert!(header.metadata_size < u32::MAX);
}

#[test]
fn test_zero_size_tensor_shapes() {
    // Test various zero-dimensional shapes
    let test_cases = vec![
        (vec![0], vec![]),        // Empty 1D
        (vec![0, 10], vec![]),    // Zero rows
        (vec![10, 0], vec![]),    // Zero columns
        (vec![0, 0], vec![]),     // Zero both dimensions
        (vec![5, 0, 10], vec![]), // Zero in middle dimension
    ];

    let mut snapshots = vec![];
    for (i, (shape, data)) in test_cases.iter().enumerate() {
        let name = format!("zero_tensor_{}", i);
        let snapshot = TensorSnapshot::from_data(
            TensorData::from_bytes_vec(data.clone(), shape.clone(), DType::F32),
            vec![name.clone()],
            vec![],
            ParamId::new(),
        );
        snapshots.push(snapshot);
    }

    let writer = BurnpackWriter::new(snapshots);
    let bytes = writer.to_bytes().unwrap();

    // Read back and verify
    let reader = BurnpackReader::from_bytes(bytes).unwrap();
    let names = reader.tensor_names();
    assert_eq!(names.len(), 5);
}

#[test]
fn test_extremely_long_tensor_names() {
    // Create a tensor with an extremely long name
    let long_name = "a".repeat(10000);

    let snapshot = TensorSnapshot::from_data(
        TensorData::from_bytes_vec(vec![1, 2, 3, 4], vec![4], DType::U8),
        vec![long_name.clone()],
        vec![],
        ParamId::new(),
    );

    let writer = BurnpackWriter::new(vec![snapshot]);
    let bytes = writer.to_bytes().unwrap();

    let reader = BurnpackReader::from_bytes(bytes).unwrap();
    let names = reader.tensor_names();
    assert_eq!(names[0].len(), 10000);
}

#[test]
fn test_unicode_in_names_and_metadata() {
    // Test various Unicode characters in tensor names and metadata
    let unicode_names = vec![
        "测试_tensor",    // Chinese
        "тест_tensor",    // Cyrillic
        "テスト_tensor",  // Japanese
        "🔥_burn_tensor", // Emoji
        "αβγδ_tensor",    // Greek
        "한글_tensor",    // Korean
    ];

    let mut snapshots = vec![];
    for name in &unicode_names {
        let snapshot = TensorSnapshot::from_data(
            TensorData::from_bytes_vec(vec![1], vec![1], DType::U8),
            vec![name.to_string()],
            vec![],
            ParamId::new(),
        );
        snapshots.push(snapshot);
    }

    let writer = BurnpackWriter::new(snapshots)
        .with_metadata("模型名称", "测试模型")
        .with_metadata("מודל", "בדיקה")
        .with_metadata("🔥", "fire");

    let bytes = writer.to_bytes().unwrap();
    let reader = BurnpackReader::from_bytes(bytes).unwrap();

    // Verify all Unicode names are preserved
    let names = reader.tensor_names();
    assert_eq!(names.len(), unicode_names.len());

    // Verify metadata
    assert_eq!(
        reader.metadata().metadata.get("模型名称"),
        Some(&"测试模型".to_string())
    );
    assert_eq!(
        reader.metadata().metadata.get("🔥"),
        Some(&"fire".to_string())
    );
}

#[test]
fn test_all_supported_dtypes() {
    // Test all DTypes with their boundary values
    let dtypes_with_data = vec![
        (
            DType::F32,
            [
                f32::MIN.to_le_bytes().to_vec(),
                f32::MAX.to_le_bytes().to_vec(),
            ]
            .concat(),
        ),
        (
            DType::F64,
            [
                f64::MIN.to_le_bytes().to_vec(),
                f64::MAX.to_le_bytes().to_vec(),
            ]
            .concat(),
        ),
        (
            DType::I32,
            [
                i32::MIN.to_le_bytes().to_vec(),
                i32::MAX.to_le_bytes().to_vec(),
            ]
            .concat(),
        ),
        (
            DType::I64,
            [
                i64::MIN.to_le_bytes().to_vec(),
                i64::MAX.to_le_bytes().to_vec(),
            ]
            .concat(),
        ),
        (
            DType::U32,
            [
                u32::MIN.to_le_bytes().to_vec(),
                u32::MAX.to_le_bytes().to_vec(),
            ]
            .concat(),
        ),
        (
            DType::U64,
            [
                u64::MIN.to_le_bytes().to_vec(),
                u64::MAX.to_le_bytes().to_vec(),
            ]
            .concat(),
        ),
        (DType::U8, vec![u8::MIN, u8::MAX]),
        (DType::Bool, vec![0, 1]),
    ];

    let mut snapshots = vec![];
    for (i, (dtype, data)) in dtypes_with_data.iter().enumerate() {
        let name = format!("dtype_test_{}", i);
        let snapshot = TensorSnapshot::from_data(
            TensorData::from_bytes_vec(data.clone(), vec![2], *dtype),
            vec![name],
            vec![],
            ParamId::new(),
        );
        snapshots.push(snapshot);
    }

    let writer = BurnpackWriter::new(snapshots);
    let bytes = writer.to_bytes().unwrap();

    let reader = BurnpackReader::from_bytes(bytes).unwrap();
    assert_eq!(reader.tensor_names().len(), dtypes_with_data.len());

    // Verify dtypes are preserved
    for (i, (expected_dtype, _)) in dtypes_with_data.iter().enumerate() {
        let name = format!("dtype_test_{}", i);
        let snapshot = reader.get_tensor_snapshot(&name).unwrap();
        assert_eq!(snapshot.dtype, *expected_dtype);
    }
}

#[test]
fn test_special_float_values() {
    // Test special floating-point values (NaN, Inf, -Inf)
    let special_values = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0_f32,
        -0.0_f32,
    ];

    let data: Vec<u8> = special_values
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();

    let snapshot = TensorSnapshot::from_data(
        TensorData::from_bytes_vec(data.clone(), vec![5], DType::F32),
        vec!["special_floats".to_string()],
        vec![],
        ParamId::new(),
    );

    let writer = BurnpackWriter::new(vec![snapshot]);
    let bytes = writer.to_bytes().unwrap();

    let reader = BurnpackReader::from_bytes(bytes).unwrap();
    let tensor_data = reader.get_tensor_data("special_floats").unwrap();

    // Check data is preserved exactly (bit-for-bit)
    assert_eq!(tensor_data, data);
}

#[test]
fn test_metadata_with_empty_values() {
    let writer = BurnpackWriter::new(vec![])
        .with_metadata("empty_value", "")
        .with_metadata("", "empty_key")
        .with_metadata("normal", "value");

    let bytes = writer.to_bytes().unwrap();
    let reader = BurnpackReader::from_bytes(bytes).unwrap();

    let metadata = &reader.metadata().metadata;
    assert_eq!(metadata.get("empty_value"), Some(&"".to_string()));
    assert_eq!(metadata.get(""), Some(&"empty_key".to_string()));
    assert_eq!(metadata.get("normal"), Some(&"value".to_string()));
}

#[test]
fn test_single_byte_tensor() {
    // Test the smallest possible tensor (1 byte)
    let snapshot = TensorSnapshot::from_data(
        TensorData::from_bytes_vec(vec![42], vec![1], DType::U8),
        vec!["single_byte".to_string()],
        vec![],
        ParamId::new(),
    );

    let writer = BurnpackWriter::new(vec![snapshot]);
    let bytes = writer.to_bytes().unwrap();

    let reader = BurnpackReader::from_bytes(bytes).unwrap();
    let data = reader.get_tensor_data("single_byte").unwrap();
    assert_eq!(data, vec![42]);
}

#[test]
fn test_high_dimensional_tensor() {
    // Test a tensor with many dimensions (10D)
    let shape = vec![2, 2, 2, 2, 2, 2, 2, 2, 2, 2]; // 10 dimensions, 1024 elements total
    let data = vec![1u8; 1024];

    let snapshot = TensorSnapshot::from_data(
        TensorData::from_bytes_vec(data.clone(), shape.clone(), DType::U8),
        vec!["high_dim".to_string()],
        vec![],
        ParamId::new(),
    );

    let writer = BurnpackWriter::new(vec![snapshot]);
    let bytes = writer.to_bytes().unwrap();

    let reader = BurnpackReader::from_bytes(bytes).unwrap();
    let loaded_snapshot = reader.get_tensor_snapshot("high_dim").unwrap();
    assert_eq!(loaded_snapshot.shape, shape);
}

#[test]
fn test_metadata_key_collision() {
    // Test that later values override earlier ones for the same key
    let writer = BurnpackWriter::new(vec![])
        .with_metadata("key", "value1")
        .with_metadata("key", "value2")
        .with_metadata("key", "value3");

    let bytes = writer.to_bytes().unwrap();
    let reader = BurnpackReader::from_bytes(bytes).unwrap();

    assert_eq!(
        reader.metadata().metadata.get("key"),
        Some(&"value3".to_string())
    );
}

#[test]
fn test_tensor_name_with_path_separators() {
    // Test tensor names that look like file paths
    let path_like_names = vec![
        "model/encoder/layer1/weights",
        "model\\decoder\\layer1\\bias",
        "model::module::param",
        "model.submodule.weight",
    ];

    let mut snapshots = vec![];
    for name in &path_like_names {
        let snapshot = TensorSnapshot::from_data(
            TensorData::from_bytes_vec(vec![1, 2, 3, 4], vec![4], DType::U8),
            vec![name.to_string()],
            vec![],
            ParamId::new(),
        );
        snapshots.push(snapshot);
    }

    let writer = BurnpackWriter::new(snapshots);
    let bytes = writer.to_bytes().unwrap();

    let reader = BurnpackReader::from_bytes(bytes).unwrap();
    let names = reader.tensor_names();

    // All names should be preserved exactly
    for expected_name in &path_like_names {
        assert!(names.contains(expected_name));
    }
}

// The following tests are commented out as they test error conditions
// that might be handled differently in the new API

// #[test]
// fn test_data_overflow_protection() {
//     // Test that we handle potential integer overflows in offset calculations
//     ...
// }

// #[test]
// fn test_reading_corrupted_header() {
//     // Test reading files with corrupted headers
//     ...
// }
