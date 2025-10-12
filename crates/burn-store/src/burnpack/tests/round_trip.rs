use crate::burnpack::{reader::BurnpackReader, writer::BurnpackWriter};

use super::*;
use alloc::collections::BTreeMap;
use alloc::string::String;
use burn_tensor::{DType, TensorData};

/// Helper function to perform round-trip test
fn round_trip_test<F>(setup: F)
where
    F: FnOnce(&mut Vec<TensorSnapshot>, &mut BTreeMap<String, String>),
{
    // Collect snapshots and metadata
    let mut snapshots = Vec::new();
    let mut metadata = BTreeMap::new();
    setup(&mut snapshots, &mut metadata);

    // Sort snapshots by name to ensure consistent ordering
    // This is necessary because BTreeMap will store them sorted
    snapshots.sort_by(|a, b| a.full_path().cmp(&b.full_path()));

    // Create writer with snapshots and metadata
    let mut writer = BurnpackWriter::new(snapshots);
    for (key, value) in &metadata {
        writer = writer.with_metadata(key, value);
    }

    let bytes = writer.to_bytes().unwrap();
    let reader = BurnpackReader::from_bytes(bytes.clone()).unwrap();

    // Write to bytes again from reader data
    let mut snapshots2 = Vec::new();

    // Copy tensors (metadata.tensors is now BTreeMap<String, TensorDescriptor>)
    // They will come out in sorted order from tensor_names()
    for tensor_name in reader.tensor_names() {
        let snapshot = reader.get_tensor_snapshot(tensor_name).unwrap();
        snapshots2.push(snapshot);
    }

    // Create writer2 with collected snapshots and metadata
    let mut writer2 = BurnpackWriter::new(snapshots2);
    for (key, value) in &reader.metadata().metadata {
        writer2 = writer2.with_metadata(key, value);
    }

    let bytes2 = writer2.to_bytes().unwrap();

    // Both byte representations should be identical
    assert_eq!(bytes, bytes2, "Round-trip produced different bytes");
}

#[test]
fn test_round_trip_empty() {
    round_trip_test(|_snapshots, _metadata| {
        // Empty writer
    });
}

#[test]
fn test_round_trip_metadata_only() {
    round_trip_test(|_snapshots, metadata| {
        metadata.insert("key1".to_string(), "value1".to_string());
        metadata.insert("key2".to_string(), "value2".to_string());
        metadata.insert("key3".to_string(), "value3".to_string());
    });
}

#[test]
fn test_round_trip_f32() {
    round_trip_test(|snapshots, _metadata| {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let snapshot = TensorSnapshot::from_data(
            TensorData::from_bytes_vec(bytes, vec![5], DType::F32),
            vec!["f32_tensor".to_string()],
            vec![],
            burn_core::module::ParamId::new(),
        );
        snapshots.push(snapshot);
    });
}

#[test]
fn test_round_trip_f64() {
    round_trip_test(|snapshots, _metadata| {
        let data = vec![1.0f64, 2.0, 3.0];
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let snapshot = TensorSnapshot::from_data(
            TensorData::from_bytes_vec(bytes, vec![3], DType::F64),
            vec!["f64_tensor".to_string()],
            vec![],
            burn_core::module::ParamId::new(),
        );
        snapshots.push(snapshot);
    });
}

#[test]
fn test_round_trip_i32() {
    round_trip_test(|snapshots, _metadata| {
        let data = vec![-10i32, 0, 10, 20];
        let bytes: Vec<u8> = data.iter().flat_map(|i| i.to_le_bytes()).collect();
        let snapshot = TensorSnapshot::from_data(
            TensorData::from_bytes_vec(bytes, vec![4], DType::I32),
            vec!["i32_tensor".to_string()],
            vec![],
            burn_core::module::ParamId::new(),
        );
        snapshots.push(snapshot);
    });
}

#[test]
fn test_round_trip_i64() {
    round_trip_test(|snapshots, _metadata| {
        let data = vec![i64::MIN, 0, i64::MAX];
        let bytes: Vec<u8> = data.iter().flat_map(|i| i.to_le_bytes()).collect();
        let snapshot = TensorSnapshot::from_data(
            TensorData::from_bytes_vec(bytes, vec![3], DType::I64),
            vec!["i64_tensor".to_string()],
            vec![],
            burn_core::module::ParamId::new(),
        );
        snapshots.push(snapshot);
    });
}

#[test]
fn test_round_trip_u32() {
    round_trip_test(|snapshots, _metadata| {
        let data = vec![0u32, 100, 1000, u32::MAX];
        let bytes: Vec<u8> = data.iter().flat_map(|u| u.to_le_bytes()).collect();
        let snapshot = TensorSnapshot::from_data(
            TensorData::from_bytes_vec(bytes, vec![4], DType::U32),
            vec!["u32_tensor".to_string()],
            vec![],
            burn_core::module::ParamId::new(),
        );
        snapshots.push(snapshot);
    });
}

#[test]
fn test_round_trip_u64() {
    round_trip_test(|snapshots, _metadata| {
        let data = vec![0u64, u64::MAX / 2, u64::MAX];
        let bytes: Vec<u8> = data.iter().flat_map(|u| u.to_le_bytes()).collect();
        let snapshot = TensorSnapshot::from_data(
            TensorData::from_bytes_vec(bytes, vec![3], DType::U64),
            vec!["u64_tensor".to_string()],
            vec![],
            burn_core::module::ParamId::new(),
        );
        snapshots.push(snapshot);
    });
}

#[test]
fn test_round_trip_u8() {
    round_trip_test(|snapshots, _metadata| {
        let data = vec![0u8, 127, 255];
        let snapshot = TensorSnapshot::from_data(
            TensorData::from_bytes_vec(data, vec![3], DType::U8),
            vec!["u8_tensor".to_string()],
            vec![],
            burn_core::module::ParamId::new(),
        );
        snapshots.push(snapshot);
    });
}

#[test]
fn test_round_trip_bool() {
    round_trip_test(|snapshots, _metadata| {
        let data = vec![0u8, 1, 0, 1, 1];
        let snapshot = TensorSnapshot::from_data(
            TensorData::from_bytes_vec(data, vec![5], DType::Bool),
            vec!["bool_tensor".to_string()],
            vec![],
            burn_core::module::ParamId::new(),
        );
        snapshots.push(snapshot);
    });
}

#[test]
fn test_round_trip_mixed_dtypes() {
    round_trip_test(|snapshots, _metadata| {
        // F32
        let f32_data = vec![1.0f32, 2.0];
        let f32_bytes: Vec<u8> = f32_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let f32_snapshot = TensorSnapshot::from_data(
            TensorData::from_bytes_vec(f32_bytes, vec![2], DType::F32),
            vec!["f32".to_string()],
            vec![],
            burn_core::module::ParamId::new(),
        );
        snapshots.push(f32_snapshot);

        // I64
        let i64_data = vec![100i64, 200];
        let i64_bytes: Vec<u8> = i64_data.iter().flat_map(|i| i.to_le_bytes()).collect();
        let i64_snapshot = TensorSnapshot::from_data(
            TensorData::from_bytes_vec(i64_bytes, vec![2], DType::I64),
            vec!["i64".to_string()],
            vec![],
            burn_core::module::ParamId::new(),
        );
        snapshots.push(i64_snapshot);

        // Bool
        let bool_snapshot = TensorSnapshot::from_data(
            TensorData::from_bytes_vec(vec![1, 0, 1], vec![3], DType::Bool),
            vec!["bool".to_string()],
            vec![],
            burn_core::module::ParamId::new(),
        );
        snapshots.push(bool_snapshot);
    });
}

#[test]
fn test_round_trip_multidimensional() {
    round_trip_test(|snapshots, _metadata| {
        // 2D tensor
        let data_2d = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bytes_2d: Vec<u8> = data_2d.iter().flat_map(|f| f.to_le_bytes()).collect();
        let snapshot_2d = TensorSnapshot::from_data(
            TensorData::from_bytes_vec(bytes_2d, vec![2, 3], DType::F32),
            vec!["tensor_2d".to_string()],
            vec![],
            burn_core::module::ParamId::new(),
        );
        snapshots.push(snapshot_2d);

        // 3D tensor
        let data_3d = vec![1.0f32; 24];
        let bytes_3d: Vec<u8> = data_3d.iter().flat_map(|f| f.to_le_bytes()).collect();
        let snapshot_3d = TensorSnapshot::from_data(
            TensorData::from_bytes_vec(bytes_3d, vec![2, 3, 4], DType::F32),
            vec!["tensor_3d".to_string()],
            vec![],
            burn_core::module::ParamId::new(),
        );
        snapshots.push(snapshot_3d);

        // 4D tensor (common for CNNs)
        let data_4d = vec![1.0f32; 120];
        let bytes_4d: Vec<u8> = data_4d.iter().flat_map(|f| f.to_le_bytes()).collect();
        let snapshot_4d = TensorSnapshot::from_data(
            TensorData::from_bytes_vec(bytes_4d, vec![2, 3, 4, 5], DType::F32),
            vec!["tensor_4d".to_string()],
            vec![],
            burn_core::module::ParamId::new(),
        );
        snapshots.push(snapshot_4d);
    });
}

#[test]
fn test_round_trip_with_metadata_and_tensors() {
    round_trip_test(|snapshots, metadata| {
        // Add metadata
        metadata.insert("model_name".to_string(), "test_model".to_string());
        metadata.insert("version".to_string(), "1.0.0".to_string());
        metadata.insert(
            "description".to_string(),
            "A test model for round-trip testing".to_string(),
        );

        // Add tensors
        let weights = vec![0.1f32, 0.2, 0.3, 0.4];
        let weights_bytes: Vec<u8> = weights.iter().flat_map(|f| f.to_le_bytes()).collect();
        let weights_snapshot = TensorSnapshot::from_data(
            TensorData::from_bytes_vec(weights_bytes, vec![2, 2], DType::F32),
            vec!["layer1".to_string(), "weights".to_string()],
            vec![],
            burn_core::module::ParamId::new(),
        );
        snapshots.push(weights_snapshot);

        let bias = vec![0.5f32, 0.6];
        let bias_bytes: Vec<u8> = bias.iter().flat_map(|f| f.to_le_bytes()).collect();
        let bias_snapshot = TensorSnapshot::from_data(
            TensorData::from_bytes_vec(bias_bytes, vec![2], DType::F32),
            vec!["layer1".to_string(), "bias".to_string()],
            vec![],
            burn_core::module::ParamId::new(),
        );
        snapshots.push(bias_snapshot);
    });
}

#[test]
fn test_round_trip_special_values() {
    round_trip_test(|snapshots, _metadata| {
        // Test special float values
        let special_f32 = vec![
            0.0f32,
            -0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::MIN,
            f32::MAX,
            f32::EPSILON,
        ];
        let f32_bytes: Vec<u8> = special_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
        let f32_snapshot = TensorSnapshot::from_data(
            TensorData::from_bytes_vec(f32_bytes, vec![8], DType::F32),
            vec!["special_f32".to_string()],
            vec![],
            burn_core::module::ParamId::new(),
        );
        snapshots.push(f32_snapshot);

        // Test special f64 values
        let special_f64 = vec![
            0.0f64,
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            f64::MIN,
            f64::MAX,
            f64::EPSILON,
        ];
        let f64_bytes: Vec<u8> = special_f64.iter().flat_map(|f| f.to_le_bytes()).collect();
        let f64_snapshot = TensorSnapshot::from_data(
            TensorData::from_bytes_vec(f64_bytes, vec![8], DType::F64),
            vec!["special_f64".to_string()],
            vec![],
            burn_core::module::ParamId::new(),
        );
        snapshots.push(f64_snapshot);
    });
}

#[test]
fn test_round_trip_large_tensors() {
    round_trip_test(|snapshots, _metadata| {
        // Large tensor (100KB)
        let size = 25600; // 100KB / 4 bytes per f32
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let snapshot = TensorSnapshot::from_data(
            TensorData::from_bytes_vec(bytes, vec![size], DType::F32),
            vec!["large_tensor".to_string()],
            vec![],
            burn_core::module::ParamId::new(),
        );
        snapshots.push(snapshot);
    });
}

#[cfg(feature = "std")]
#[test]
fn test_round_trip_file_io() {
    use std::fs;
    use tempfile::tempdir;

    use crate::burnpack::writer::BurnpackWriter;

    let dir = tempdir().unwrap();
    let file_path = dir.path().join("round_trip.bpk");

    // Create original data
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    let snapshot = TensorSnapshot::from_data(
        TensorData::from_bytes_vec(bytes, vec![2, 2], DType::F32),
        vec!["weights".to_string()],
        vec![],
        burn_core::module::ParamId::new(),
    );

    let writer = BurnpackWriter::new(vec![snapshot]).with_metadata("test", "round_trip");

    // Write to file
    writer.write_to_file(&file_path).unwrap();

    // Read from file
    let reader = BurnpackReader::from_file(&file_path).unwrap();

    // Write to another file
    let file_path2 = dir.path().join("round_trip2.bpk");

    // Collect snapshots from reader
    let mut snapshots2 = Vec::new();
    for tensor_name in reader.tensor_names() {
        let snapshot = reader.get_tensor_snapshot(tensor_name).unwrap();
        snapshots2.push(snapshot);
    }

    // Create writer2 with snapshots and metadata
    let mut writer2 = BurnpackWriter::new(snapshots2);
    for (key, value) in &reader.metadata().metadata {
        writer2 = writer2.with_metadata(key, value);
    }

    writer2.write_to_file(&file_path2).unwrap();

    // Compare files
    let bytes1 = fs::read(&file_path).unwrap();
    let bytes2 = fs::read(&file_path2).unwrap();

    assert_eq!(
        bytes1, bytes2,
        "Round-trip through files produced different content"
    );
}

#[test]
fn test_round_trip_empty_shapes() {
    round_trip_test(|snapshots, _metadata| {
        // Scalar (0-dimensional)
        let scalar = vec![42.0f32];
        let scalar_bytes: Vec<u8> = scalar.iter().flat_map(|f| f.to_le_bytes()).collect();
        let scalar_snapshot = TensorSnapshot::from_data(
            TensorData::from_bytes_vec(scalar_bytes, vec![], DType::F32),
            vec!["scalar".to_string()],
            vec![],
            burn_core::module::ParamId::new(),
        );
        snapshots.push(scalar_snapshot);

        // Empty tensor
        let empty_snapshot = TensorSnapshot::from_data(
            TensorData::from_bytes_vec(vec![], vec![0], DType::F32),
            vec!["empty".to_string()],
            vec![],
            burn_core::module::ParamId::new(),
        );
        snapshots.push(empty_snapshot);
    });
}

#[test]
fn test_param_id_persistence() {
    use burn_core::module::ParamId;

    // Create a specific ParamId with a known value
    let original_param_id = ParamId::from(123456789u64);

    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    let snapshot = TensorSnapshot::from_data(
        TensorData::from_bytes_vec(bytes, vec![2, 2], DType::F32),
        vec!["weights".to_string()],
        vec![],
        original_param_id,
    );

    // Write to burnpack
    let writer = BurnpackWriter::new(vec![snapshot]);
    let bytes = writer.to_bytes().unwrap();

    // Read back from burnpack
    let reader = BurnpackReader::from_bytes(bytes).unwrap();
    let loaded_snapshot = reader.get_tensor_snapshot("weights").unwrap();

    // Verify ParamId was preserved
    assert!(
        loaded_snapshot.tensor_id.is_some(),
        "ParamId should be present"
    );
    let loaded_param_id = loaded_snapshot.tensor_id.unwrap();
    assert_eq!(
        loaded_param_id.val(),
        original_param_id.val(),
        "ParamId value should be preserved: expected {}, got {}",
        original_param_id.val(),
        loaded_param_id.val()
    );
}

#[test]
fn test_param_id_backward_compatibility() {
    use crate::burnpack::base::{BurnpackMetadata, TensorDescriptor};
    use alloc::collections::BTreeMap;

    // Create metadata without param_id (simulating old burnpack format)
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "old_tensor".to_string(),
        TensorDescriptor {
            dtype: DType::F32,
            shape: vec![2, 2],
            data_offsets: (0, 16),
            param_id: None, // No param_id stored (old format)
        },
    );

    let metadata = BurnpackMetadata {
        tensors,
        metadata: BTreeMap::new(),
    };

    // Serialize metadata
    let mut metadata_bytes = Vec::new();
    ciborium::ser::into_writer(&metadata, &mut metadata_bytes).unwrap();

    // Create a complete burnpack with header and data
    use crate::burnpack::base::{BurnpackHeader, FORMAT_VERSION, MAGIC_NUMBER};

    let metadata_size = metadata_bytes.len() as u32;
    let header = BurnpackHeader {
        magic: MAGIC_NUMBER,
        version: FORMAT_VERSION,
        metadata_size,
    };

    let mut full_bytes = Vec::new();
    full_bytes.extend_from_slice(&header.into_bytes());
    full_bytes.extend_from_slice(&metadata_bytes);

    // Add tensor data (4 f32 values = 16 bytes)
    let tensor_data = vec![1.0f32, 2.0, 3.0, 4.0];
    for value in tensor_data {
        full_bytes.extend_from_slice(&value.to_le_bytes());
    }

    // Read the old format burnpack
    let reader =
        BurnpackReader::from_bytes(burn_tensor::Bytes::from_bytes_vec(full_bytes)).unwrap();
    let loaded_snapshot = reader.get_tensor_snapshot("old_tensor").unwrap();

    // Verify that a new ParamId was generated (backward compatibility)
    assert!(
        loaded_snapshot.tensor_id.is_some(),
        "ParamId should be generated for old format"
    );

    // The generated ParamId should be different each time (it's new), but we can't test the exact value
    // We just verify it exists and has a valid u64 value
    let generated_param_id = loaded_snapshot.tensor_id.unwrap();
    assert!(
        generated_param_id.val() > 0,
        "Generated ParamId should have a valid value"
    );
}

#[test]
fn test_multiple_tensors_preserve_distinct_param_ids() {
    use burn_core::module::ParamId;

    // Create multiple tensors with distinct ParamIds
    let param_id_1 = ParamId::from(111111u64);
    let param_id_2 = ParamId::from(222222u64);
    let param_id_3 = ParamId::from(333333u64);

    let mut snapshots = Vec::new();

    let data1 = vec![1.0f32, 2.0];
    let bytes1: Vec<u8> = data1.iter().flat_map(|f| f.to_le_bytes()).collect();
    snapshots.push(TensorSnapshot::from_data(
        TensorData::from_bytes_vec(bytes1, vec![2], DType::F32),
        vec!["tensor1".to_string()],
        vec![],
        param_id_1,
    ));

    let data2 = vec![3.0f32, 4.0];
    let bytes2: Vec<u8> = data2.iter().flat_map(|f| f.to_le_bytes()).collect();
    snapshots.push(TensorSnapshot::from_data(
        TensorData::from_bytes_vec(bytes2, vec![2], DType::F32),
        vec!["tensor2".to_string()],
        vec![],
        param_id_2,
    ));

    let data3 = vec![5.0f32, 6.0];
    let bytes3: Vec<u8> = data3.iter().flat_map(|f| f.to_le_bytes()).collect();
    snapshots.push(TensorSnapshot::from_data(
        TensorData::from_bytes_vec(bytes3, vec![2], DType::F32),
        vec!["tensor3".to_string()],
        vec![],
        param_id_3,
    ));

    // Write to burnpack
    let writer = BurnpackWriter::new(snapshots);
    let bytes = writer.to_bytes().unwrap();

    // Read back
    let reader = BurnpackReader::from_bytes(bytes).unwrap();

    let snapshot1 = reader.get_tensor_snapshot("tensor1").unwrap();
    let snapshot2 = reader.get_tensor_snapshot("tensor2").unwrap();
    let snapshot3 = reader.get_tensor_snapshot("tensor3").unwrap();

    // Verify each ParamId was preserved correctly
    assert_eq!(snapshot1.tensor_id.unwrap().val(), param_id_1.val());
    assert_eq!(snapshot2.tensor_id.unwrap().val(), param_id_2.val());
    assert_eq!(snapshot3.tensor_id.unwrap().val(), param_id_3.val());

    // Verify they are distinct
    let id1 = snapshot1.tensor_id.unwrap().val();
    let id2 = snapshot2.tensor_id.unwrap().val();
    let id3 = snapshot3.tensor_id.unwrap().val();

    assert_ne!(id1, id2, "ParamIds should be distinct");
    assert_ne!(id2, id3, "ParamIds should be distinct");
    assert_ne!(id1, id3, "ParamIds should be distinct");
}
