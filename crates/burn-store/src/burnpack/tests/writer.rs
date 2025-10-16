use crate::burnpack::{
    base::{
        BurnpackHeader, BurnpackMetadata, FORMAT_VERSION, HEADER_SIZE, MAGIC_NUMBER, magic_range,
    },
    writer::BurnpackWriter,
};

use super::*;
use burn_core::module::ParamId;
use burn_tensor::{DType, TensorData};
use std::rc::Rc;

#[test]
fn test_writer_new() {
    let writer = BurnpackWriter::new(vec![]);
    assert_eq!(writer.snapshots.len(), 0);
    assert!(writer.metadata.is_empty());
}

#[test]
fn test_writer_add_metadata() {
    let writer = BurnpackWriter::new(vec![])
        .with_metadata("model_name", "test_model")
        .with_metadata("version", "1.0.0")
        .with_metadata("author", "test_author");

    assert_eq!(writer.metadata.len(), 3);
    assert_eq!(
        writer.metadata.get("model_name"),
        Some(&"test_model".to_string())
    );
    assert_eq!(writer.metadata.get("version"), Some(&"1.0.0".to_string()));
    assert_eq!(
        writer.metadata.get("author"),
        Some(&"test_author".to_string())
    );
}

#[test]
fn test_writer_add_tensor_snapshot() {
    // Create test tensor snapshots
    let snapshot1 = TensorSnapshot::from_data(
        TensorData::from_bytes_vec(vec![1, 2, 3, 4], vec![2, 2], DType::U8),
        vec!["layer1".to_string(), "weights".to_string()],
        vec![],
        burn_core::module::ParamId::new(),
    );

    let snapshot2 = TensorSnapshot::from_data(
        TensorData::from_bytes_vec(vec![5, 6, 7, 8], vec![4], DType::U8),
        vec!["layer1".to_string(), "bias".to_string()],
        vec![],
        burn_core::module::ParamId::new(),
    );

    let writer = BurnpackWriter::new(vec![snapshot1, snapshot2]);

    assert_eq!(writer.snapshots.len(), 2);
    assert_eq!(writer.snapshots[0].full_path(), "layer1.weights");
    assert_eq!(writer.snapshots[1].full_path(), "layer1.bias");
}

#[test]
fn test_writer_to_bytes_empty() {
    let writer = BurnpackWriter::new(vec![]);
    let bytes = writer.to_bytes().unwrap();

    // Verify header
    assert!(bytes.len() >= HEADER_SIZE);
    assert_eq!(&bytes[magic_range()], &MAGIC_NUMBER.to_le_bytes());

    // Parse header
    let header = BurnpackHeader::from_bytes(&bytes[..HEADER_SIZE]).unwrap();
    assert_eq!(header.magic, MAGIC_NUMBER);
    assert_eq!(header.version, FORMAT_VERSION);

    // Verify metadata
    let metadata_end = HEADER_SIZE + header.metadata_size as usize;
    let metadata_bytes = &bytes[HEADER_SIZE..metadata_end];
    let metadata: BurnpackMetadata = ciborium::de::from_reader(metadata_bytes).unwrap();

    assert_eq!(metadata.tensors.len(), 0);
    assert!(metadata.metadata.is_empty());
}

#[test]
fn test_writer_to_bytes_with_tensors() {
    // Add tensors with different data types
    let f32_data = [1.0f32, 2.0, 3.0, 4.0];
    let f32_bytes: Vec<u8> = f32_data.iter().flat_map(|f| f.to_le_bytes()).collect();
    let snapshot_f32 = TensorSnapshot::from_data(
        TensorData::from_bytes_vec(f32_bytes.clone(), vec![2, 2], DType::F32),
        vec!["weights".to_string()],
        vec![],
        burn_core::module::ParamId::new(),
    );

    let i64_data = [10i64, 20, 30];
    let i64_bytes: Vec<u8> = i64_data.iter().flat_map(|i| i.to_le_bytes()).collect();
    let snapshot_i64 = TensorSnapshot::from_data(
        TensorData::from_bytes_vec(i64_bytes.clone(), vec![3], DType::I64),
        vec!["bias".to_string()],
        vec![],
        burn_core::module::ParamId::new(),
    );

    let writer = BurnpackWriter::new(vec![snapshot_f32, snapshot_i64])
        .with_metadata("test_key", "test_value");

    let bytes = writer.to_bytes().unwrap();

    // Parse and verify
    let header = BurnpackHeader::from_bytes(&bytes[..HEADER_SIZE]).unwrap();
    let metadata_end = HEADER_SIZE + header.metadata_size as usize;
    let metadata: BurnpackMetadata =
        ciborium::de::from_reader(&bytes[HEADER_SIZE..metadata_end]).unwrap();

    // Verify metadata
    assert_eq!(
        metadata.metadata.get("test_key"),
        Some(&"test_value".to_string())
    );

    // Verify tensors
    assert_eq!(metadata.tensors.len(), 2);

    let weights = metadata.tensors.get("weights").unwrap();
    assert_eq!(weights.dtype, DType::F32);
    assert_eq!(weights.shape, vec![2, 2]);
    assert_eq!(weights.data_offsets.1 - weights.data_offsets.0, 16); // 4 * 4 bytes

    let bias = metadata.tensors.get("bias").unwrap();
    assert_eq!(bias.dtype, DType::I64);
    assert_eq!(bias.shape, vec![3]);
    assert_eq!(bias.data_offsets.1 - bias.data_offsets.0, 24); // 3 * 8 bytes

    // Verify actual tensor data
    let weights = metadata.tensors.get("weights").unwrap();
    let bias = metadata.tensors.get("bias").unwrap();
    let weights_data = &bytes[metadata_end + weights.data_offsets.0 as usize
        ..metadata_end + weights.data_offsets.1 as usize];
    assert_eq!(weights_data, f32_bytes);

    let bias_data = &bytes
        [metadata_end + bias.data_offsets.0 as usize..metadata_end + bias.data_offsets.1 as usize];
    assert_eq!(bias_data, i64_bytes);
}

#[test]
fn test_writer_all_dtypes() {
    // Test all supported data types
    let test_cases = vec![
        (DType::F32, 4, [1.0f32.to_le_bytes().to_vec()].concat()),
        (DType::F64, 8, [1.0f64.to_le_bytes().to_vec()].concat()),
        (DType::I32, 4, [1i32.to_le_bytes().to_vec()].concat()),
        (DType::I64, 8, [1i64.to_le_bytes().to_vec()].concat()),
        (DType::U32, 4, [1u32.to_le_bytes().to_vec()].concat()),
        (DType::U64, 8, [1u64.to_le_bytes().to_vec()].concat()),
        (DType::U8, 1, vec![255u8]),
        (DType::Bool, 1, vec![1u8]),
    ];

    let mut snapshots = vec![];
    for (i, (dtype, _expected_size, data)) in test_cases.into_iter().enumerate() {
        let name = format!("tensor_{}", i);
        let snapshot = TensorSnapshot::from_data(
            TensorData::from_bytes_vec(data.clone(), vec![1], dtype),
            vec![name.clone()],
            vec![],
            burn_core::module::ParamId::new(),
        );
        snapshots.push(snapshot);
    }

    let writer = BurnpackWriter::new(snapshots);

    let bytes = writer.to_bytes().unwrap();

    // Parse and verify
    let header = BurnpackHeader::from_bytes(&bytes[..HEADER_SIZE]).unwrap();
    let metadata: BurnpackMetadata =
        ciborium::de::from_reader(&bytes[HEADER_SIZE..HEADER_SIZE + header.metadata_size as usize])
            .unwrap();

    assert_eq!(metadata.tensors.len(), 8);

    for i in 0..8 {
        let name = format!("tensor_{}", i);
        let tensor = metadata.tensors.get(&name).unwrap();
        assert_eq!(tensor.shape, vec![1]);
    }
}

#[test]
fn test_writer_large_tensor() {
    // Create a large tensor (1MB)
    let size = 256 * 1024; // 256K floats = 1MB
    let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();

    let snapshot = TensorSnapshot::from_data(
        TensorData::from_bytes_vec(bytes.clone(), vec![size], DType::F32),
        vec!["large_tensor".to_string()],
        vec![],
        burn_core::module::ParamId::new(),
    );

    let writer = BurnpackWriter::new(vec![snapshot]);

    let result = writer.to_bytes().unwrap();

    // Verify the large tensor is correctly stored
    let header = BurnpackHeader::from_bytes(&result[..HEADER_SIZE]).unwrap();
    let metadata: BurnpackMetadata = ciborium::de::from_reader(
        &result[HEADER_SIZE..HEADER_SIZE + header.metadata_size as usize],
    )
    .unwrap();

    assert_eq!(metadata.tensors.len(), 1);
    let tensor = metadata.tensors.get("large_tensor").unwrap();
    assert_eq!(tensor.shape, vec![size as u64]);
    assert_eq!(
        tensor.data_offsets.1 - tensor.data_offsets.0,
        (size * 4) as u64
    );
}

#[test]
fn test_writer_empty_tensors() {
    // Add tensor with empty data
    let snapshot = TensorSnapshot::from_data(
        TensorData::from_bytes_vec(vec![], vec![0], DType::F32),
        vec!["empty".to_string()],
        vec![],
        ParamId::new(),
    );

    let writer = BurnpackWriter::new(vec![snapshot]);

    let bytes = writer.to_bytes().unwrap();

    let header = BurnpackHeader::from_bytes(&bytes[..HEADER_SIZE]).unwrap();
    let metadata: BurnpackMetadata =
        ciborium::de::from_reader(&bytes[HEADER_SIZE..HEADER_SIZE + header.metadata_size as usize])
            .unwrap();

    assert_eq!(metadata.tensors.len(), 1);
    let tensor = metadata.tensors.get("empty").unwrap();
    assert_eq!(tensor.shape, vec![0]);
    assert_eq!(tensor.data_offsets.1 - tensor.data_offsets.0, 0);
}

#[test]
fn test_writer_special_characters_in_names() {
    // Test various special characters in tensor names
    let special_names = vec![
        "layer.0.weight",
        "model/encoder/layer1",
        "model::layer::weight",
        "layer[0].bias",
        "layer_1_weight",
        "layer-1-bias",
        "layer@1#weight",
        "emoji_😀_tensor",
        "unicode_测试_tensor",
        "spaces in name",
    ];

    let mut snapshots = vec![];
    for name in &special_names {
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

    let header = BurnpackHeader::from_bytes(&bytes[..HEADER_SIZE]).unwrap();
    let metadata: BurnpackMetadata =
        ciborium::de::from_reader(&bytes[HEADER_SIZE..HEADER_SIZE + header.metadata_size as usize])
            .unwrap();

    assert_eq!(metadata.tensors.len(), 10);
    for (tensor_name, _tensor) in metadata.tensors.iter() {
        assert!(!tensor_name.is_empty());
        // Names should be preserved exactly
        assert!(
            tensor_name.contains("layer")
                || tensor_name.contains("model")
                || tensor_name.contains("emoji")
                || tensor_name.contains("unicode")
                || tensor_name.contains("spaces")
        );
    }
}

#[test]
fn test_writer_metadata_overwrite() {
    let writer = BurnpackWriter::new(vec![])
        .with_metadata("key", "value1")
        .with_metadata("key", "value2");

    assert_eq!(writer.metadata.get("key"), Some(&"value2".to_string()));
    assert_eq!(writer.metadata.len(), 1);
}

#[test]
fn test_writer_tensor_order_preserved() {
    // Add tensors in specific order
    let names = vec!["z_tensor", "a_tensor", "m_tensor", "b_tensor"];

    let mut snapshots = vec![];
    for name in &names {
        let snapshot = TensorSnapshot::from_data(
            TensorData::from_bytes_vec(vec![1], vec![1], DType::U8),
            vec![name.to_string()],
            vec![],
            ParamId::new(),
        );
        snapshots.push(snapshot);
    }

    let writer = BurnpackWriter::new(snapshots);

    let bytes = writer.to_bytes().unwrap();

    let header = BurnpackHeader::from_bytes(&bytes[..HEADER_SIZE]).unwrap();
    let metadata: BurnpackMetadata =
        ciborium::de::from_reader(&bytes[HEADER_SIZE..HEADER_SIZE + header.metadata_size as usize])
            .unwrap();

    // Verify all tensors are present (BTreeMap stores in sorted order by key)
    let expected_sorted = vec!["a_tensor", "b_tensor", "m_tensor", "z_tensor"];
    let actual_names: Vec<_> = metadata.tensors.keys().collect();
    assert_eq!(actual_names, expected_sorted);
}

#[test]
fn test_writer_lazy_snapshot_evaluation() {
    // Create a lazy snapshot using closure
    let data = Rc::new(vec![1.0f32, 2.0, 3.0, 4.0]);
    let data_clone = data.clone();

    let snapshot = TensorSnapshot::from_closure(
        Rc::new(move || {
            let bytes: Vec<u8> = data_clone.iter().flat_map(|f| f.to_le_bytes()).collect();
            Ok(TensorData::from_bytes_vec(bytes, vec![2, 2], DType::F32))
        }),
        DType::F32,
        vec![2, 2],
        vec!["lazy".to_string()],
        vec![],
        ParamId::new(),
    );

    let writer = BurnpackWriter::new(vec![snapshot]);

    // The closure should only be evaluated when to_bytes is called
    let bytes = writer.to_bytes().unwrap();

    let header = BurnpackHeader::from_bytes(&bytes[..HEADER_SIZE]).unwrap();
    let metadata_end = HEADER_SIZE + header.metadata_size as usize;
    let metadata: BurnpackMetadata =
        ciborium::de::from_reader(&bytes[HEADER_SIZE..metadata_end]).unwrap();

    assert_eq!(metadata.tensors.len(), 1);
    let tensor = metadata.tensors.get("lazy").unwrap();
    assert_eq!(tensor.dtype, DType::F32);
    assert_eq!(tensor.shape, vec![2, 2]);

    // Verify the data was correctly written
    let tensor_data = &bytes[metadata_end..metadata_end + 16];
    let expected: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    assert_eq!(tensor_data, expected.as_slice());
}

#[cfg(feature = "std")]
#[test]
fn test_writer_write_to_file() {
    use std::fs;
    use tempfile::tempdir;

    let dir = tempdir().unwrap();
    let file_path = dir.path().join("test.bpk");

    let snapshot = TensorSnapshot::from_data(
        TensorData::from_bytes_vec(vec![1, 2, 3, 4], vec![2, 2], DType::U8),
        vec!["test".to_string()],
        vec![],
        ParamId::new(),
    );

    let writer = BurnpackWriter::new(vec![snapshot]).with_metadata("file_test", "true");

    writer.write_to_file(&file_path).unwrap();

    // Verify file exists and has correct content
    assert!(file_path.exists());

    let file_bytes = fs::read(&file_path).unwrap();
    let memory_bytes = writer.to_bytes().unwrap();

    assert_eq!(file_bytes.as_slice(), &*memory_bytes);
}

#[test]
fn test_writer_size() {
    let snapshot = TensorSnapshot::from_data(
        TensorData::from_bytes_vec(vec![1, 2, 3, 4], vec![2, 2], DType::U8),
        vec!["test".to_string()],
        vec![],
        ParamId::new(),
    );

    let writer = BurnpackWriter::new(vec![snapshot]).with_metadata("test", "value");

    let size = writer.size().unwrap();
    let bytes = writer.to_bytes().unwrap();

    // Size should match actual bytes length
    assert_eq!(size, bytes.len());
}

#[test]
fn test_writer_write_into() {
    let snapshot = TensorSnapshot::from_data(
        TensorData::from_bytes_vec(vec![1, 2, 3, 4], vec![2, 2], DType::U8),
        vec!["test".to_string()],
        vec![],
        ParamId::new(),
    );

    let writer = BurnpackWriter::new(vec![snapshot]).with_metadata("test", "value");

    // Get size and allocate buffer
    let size = writer.size().unwrap();
    let mut buffer = vec![0u8; size];

    // Write into buffer
    writer.write_into(&mut buffer).unwrap();

    // Compare with to_bytes()
    let bytes = writer.to_bytes().unwrap();
    assert_eq!(buffer.as_slice(), &*bytes);
}

#[test]
fn test_writer_write_into_buffer_too_small() {
    let snapshot = TensorSnapshot::from_data(
        TensorData::from_bytes_vec(vec![1, 2, 3, 4], vec![2, 2], DType::U8),
        vec!["test".to_string()],
        vec![],
        ParamId::new(),
    );

    let writer = BurnpackWriter::new(vec![snapshot]);

    // Allocate a buffer that's too small
    let mut buffer = vec![0u8; 10];

    // Should fail with buffer too small error
    let result = writer.write_into(&mut buffer);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Buffer too small"));
}

#[test]
fn test_writer_write_into_buffer_larger_than_needed() {
    let snapshot = TensorSnapshot::from_data(
        TensorData::from_bytes_vec(vec![1, 2, 3, 4], vec![2, 2], DType::U8),
        vec!["test".to_string()],
        vec![],
        ParamId::new(),
    );

    let writer = BurnpackWriter::new(vec![snapshot]);

    // Allocate a larger buffer
    let size = writer.size().unwrap();
    let mut buffer = vec![0u8; size + 100]; // Extra 100 bytes

    // Should succeed and only write the necessary bytes
    writer.write_into(&mut buffer).unwrap();

    // Compare the written portion with to_bytes()
    let bytes = writer.to_bytes().unwrap();
    assert_eq!(&buffer[..size], &*bytes);
}

#[test]
fn test_writer_write_into_multiple_tensors() {
    let snapshot1 = TensorSnapshot::from_data(
        TensorData::from_bytes_vec(vec![1, 2, 3, 4], vec![2, 2], DType::U8),
        vec!["tensor1".to_string()],
        vec![],
        ParamId::new(),
    );

    let snapshot2 = TensorSnapshot::from_data(
        TensorData::from_bytes_vec(vec![5, 6, 7, 8, 9, 10], vec![2, 3], DType::U8),
        vec!["tensor2".to_string()],
        vec![],
        ParamId::new(),
    );

    let writer = BurnpackWriter::new(vec![snapshot1, snapshot2]).with_metadata("test", "multiple");

    let size = writer.size().unwrap();
    let mut buffer = vec![0u8; size];
    writer.write_into(&mut buffer).unwrap();

    let bytes = writer.to_bytes().unwrap();
    assert_eq!(buffer.as_slice(), &*bytes);
}

#[test]
fn test_writer_write_into_empty() {
    let writer = BurnpackWriter::new(vec![]);

    let size = writer.size().unwrap();
    let mut buffer = vec![0u8; size];
    writer.write_into(&mut buffer).unwrap();

    let bytes = writer.to_bytes().unwrap();
    assert_eq!(buffer.as_slice(), &*bytes);
}
