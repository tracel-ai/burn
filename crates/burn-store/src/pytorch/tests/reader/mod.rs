//! Tests for PyTorch file reader functionality
//!
//! Floating-point comparison tolerances:
//! - F16/BF16: 1e-2 (~3 decimal digits precision)
//! - F32: 1e-6 (~7 decimal digits precision)
//! - F64: 1e-10 (~16 decimal digits precision)

use crate::pytorch::PytorchReader;
// Import internal types for testing only
use crate::pytorch::reader::{ByteOrder, FileFormat};
use burn_tensor::DType;
use std::path::PathBuf;

fn test_data_path(filename: &str) -> PathBuf {
    // Get the path relative to the crate root
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("pytorch")
        .join("tests")
        .join("reader")
        .join("test_data")
        .join(filename)
}

#[test]
fn test_float32_tensor() {
    let path = test_data_path("float32.pt");
    let reader = PytorchReader::new(&path).expect("Failed to load float32.pt");
    let tensor = reader.get("tensor").expect("tensor key not found");
    assert_eq!(tensor.dtype, DType::F32);
    assert_eq!(tensor.shape, vec![4]);

    let data = tensor.to_data().unwrap();
    let values = data.as_slice::<f32>().unwrap();
    assert_eq!(values.len(), 4);
    assert!((values[0] - 1.0).abs() < 1e-6);
    assert!((values[1] - 2.5).abs() < 1e-6);
    assert!((values[2] - (-3.7)).abs() < 1e-6);
    assert!((values[3] - 0.0).abs() < 1e-6);
}

#[test]
fn test_float64_tensor() {
    let path = test_data_path("float64.pt");
    let reader = PytorchReader::new(&path).expect("Failed to load float64.pt");
    let tensor = reader.get("tensor").expect("tensor key not found");
    assert_eq!(tensor.dtype, DType::F64);
    assert_eq!(tensor.shape, vec![3]);

    let data = tensor.to_data().unwrap();
    let values = data.as_slice::<f64>().unwrap();
    assert_eq!(values.len(), 3);
    assert!((values[0] - 1.1).abs() < 1e-10);
    assert!((values[1] - 2.2).abs() < 1e-10);
    assert!((values[2] - 3.3).abs() < 1e-10);
}

#[test]
fn test_int64_tensor() {
    let path = test_data_path("int64.pt");
    let reader = PytorchReader::new(&path).expect("Failed to load int64.pt");
    let tensor = reader.get("tensor").expect("tensor key not found");
    assert_eq!(tensor.dtype, DType::I64);
    assert_eq!(tensor.shape, vec![4]);

    let data = tensor.to_data().unwrap();
    let values = data.as_slice::<i64>().unwrap();
    assert_eq!(values, &[100, -200, 300, 0]);
}

#[test]
fn test_int32_tensor() {
    let path = test_data_path("int32.pt");
    let reader = PytorchReader::new(&path).expect("Failed to load int32.pt");
    let tensor = reader.get("tensor").expect("tensor key not found");
    assert_eq!(tensor.dtype, DType::I32);
    assert_eq!(tensor.shape, vec![3]);

    let data = tensor.to_data().unwrap();
    // Convert to the appropriate element type
    let data_converted = data.convert::<i32>();
    let values = data_converted.as_slice::<i32>().unwrap();
    assert_eq!(values, &[10, 20, -30]);
}

#[test]
fn test_int16_tensor() {
    let path = test_data_path("int16.pt");
    let reader = PytorchReader::new(&path).expect("Failed to load int16.pt");
    let tensor = reader.get("tensor").expect("tensor key not found");
    assert_eq!(tensor.dtype, DType::I16);
    assert_eq!(tensor.shape, vec![3]);

    let data = tensor.to_data().unwrap();
    let data_converted = data.convert::<i16>();
    let values = data_converted.as_slice::<i16>().unwrap();
    assert_eq!(values, &[1000, -2000, 3000]);
}

#[test]
fn test_int8_tensor() {
    let path = test_data_path("int8.pt");
    let reader = PytorchReader::new(&path).expect("Failed to load int8.pt");
    let tensor = reader.get("tensor").expect("tensor key not found");
    assert_eq!(tensor.dtype, DType::I8);
    assert_eq!(tensor.shape, vec![4]);

    let data = tensor.to_data().unwrap();
    let data_converted = data.convert::<i8>();
    let values = data_converted.as_slice::<i8>().unwrap();
    assert_eq!(values, &[127, -128, 0, 50]);
}

#[test]
fn test_bool_tensor() {
    let path = test_data_path("bool.pt");
    let reader = PytorchReader::new(&path).expect("Failed to load bool.pt");
    let tensor = reader.get("tensor").expect("tensor key not found");
    assert_eq!(tensor.dtype, DType::Bool);
    assert_eq!(tensor.shape, vec![5]);

    let data = tensor.to_data().unwrap();
    let values = data.as_slice::<bool>().unwrap();
    assert_eq!(values, &[true, false, true, true, false]);
}

#[test]
fn test_uint8_tensor() {
    let path = test_data_path("uint8.pt");

    let reader = PytorchReader::new(&path).expect("Failed to load uint8.pt");
    let tensor = reader.get("tensor").expect("tensor key not found");
    assert_eq!(tensor.dtype, DType::U8);
    assert_eq!(tensor.shape, vec![4]);

    // Verify actual U8 values [0, 128, 255, 42] from test_data.py
    let data = tensor.to_data().unwrap();
    let values = data.as_slice::<u8>().unwrap();
    assert_eq!(values, &[0, 128, 255, 42]);
}

#[test]
fn test_float16_tensor() {
    use half::f16;

    let path = test_data_path("float16.pt");
    let reader = PytorchReader::new(&path).expect("Failed to load float16.pt");
    let tensor = reader.get("tensor").expect("tensor key not found");
    assert_eq!(tensor.dtype, DType::F16);
    assert_eq!(tensor.shape, vec![3]);

    // Verify actual F16 values [1.5, -2.25, 3.125] from test_data.py
    let data = tensor.to_data().unwrap();
    assert_eq!(data.shape, vec![3]);
    let values = data.as_slice::<f16>().unwrap();
    assert_eq!(values.len(), 3);
    assert!((values[0].to_f32() - 1.5).abs() < 1e-2);
    assert!((values[1].to_f32() - (-2.25)).abs() < 1e-2);
    assert!((values[2].to_f32() - 3.125).abs() < 1e-2);
}

#[test]
fn test_bfloat16_tensor() {
    use half::bf16;

    let path = test_data_path("bfloat16.pt");
    let reader = PytorchReader::new(&path).expect("Failed to load bfloat16.pt");
    let tensor = reader.get("tensor").expect("tensor key not found");
    assert_eq!(tensor.dtype, DType::BF16);
    assert_eq!(tensor.shape, vec![3]);

    // Verify actual BF16 values [1.5, -2.5, 3.5] from test_data.py
    let data = tensor.to_data().unwrap();
    assert_eq!(data.shape, vec![3]);
    let values = data.as_slice::<bf16>().unwrap();
    assert_eq!(values.len(), 3);
    assert!((values[0].to_f32() - 1.5).abs() < 1e-2);
    assert!((values[1].to_f32() - (-2.5)).abs() < 1e-2);
    assert!((values[2].to_f32() - 3.5).abs() < 1e-2);
}

#[test]
fn test_2d_tensor() {
    let path = test_data_path("tensor_2d.pt");
    let reader = PytorchReader::new(&path).expect("Failed to load tensor_2d.pt");
    let tensor = reader.get("tensor").expect("tensor key not found");
    assert_eq!(tensor.dtype, DType::F32);
    assert_eq!(tensor.shape, vec![3, 2]);

    let data = tensor.to_data().unwrap();
    let values = data.as_slice::<f32>().unwrap();
    assert_eq!(values.len(), 6);
    // Check flattened values [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    for (i, expected) in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].iter().enumerate() {
        assert!((values[i] - expected).abs() < 1e-6);
    }
}

#[test]
fn test_3d_tensor() {
    let path = test_data_path("tensor_3d.pt");
    let reader = PytorchReader::new(&path).expect("Failed to load tensor_3d.pt");
    let tensor = reader.get("tensor").expect("tensor key not found");
    assert_eq!(tensor.dtype, DType::F32);
    assert_eq!(tensor.shape, vec![2, 3, 4]);

    let data = tensor.to_data().unwrap();
    assert_eq!(data.shape, vec![2, 3, 4]);
    let values = data.as_slice::<f32>().unwrap();
    assert_eq!(values.len(), 24);
}

#[test]
fn test_4d_tensor() {
    let path = test_data_path("tensor_4d.pt");
    let reader = PytorchReader::new(&path).expect("Failed to load tensor_4d.pt");
    let tensor = reader.get("tensor").expect("tensor key not found");
    assert_eq!(tensor.dtype, DType::F32);
    assert_eq!(tensor.shape, vec![2, 3, 2, 2]);

    let data = tensor.to_data().unwrap();
    assert_eq!(data.shape, vec![2, 3, 2, 2]);
    let values = data.as_slice::<f32>().unwrap();
    assert_eq!(values.len(), 24);
}

#[test]
fn test_state_dict() {
    let path = test_data_path("state_dict.pt");
    let reader = PytorchReader::new(&path).expect("Failed to load state_dict.pt");
    let keys = reader.keys();

    assert_eq!(keys.len(), 4);
    assert!(keys.contains(&"weight".to_string()));
    assert!(keys.contains(&"bias".to_string()));
    assert!(keys.contains(&"running_mean".to_string()));
    assert!(keys.contains(&"running_var".to_string()));

    // Check weight tensor
    let weight = reader.get("weight").unwrap();
    assert_eq!(weight.shape, vec![3, 4]);
    assert_eq!(weight.dtype, DType::F32);

    // Check bias tensor
    let bias = reader.get("bias").unwrap();
    assert_eq!(bias.shape, vec![3]);
    assert_eq!(bias.dtype, DType::F32);

    // Check running_mean (should be zeros)
    let running_mean = reader.get("running_mean").unwrap();
    assert_eq!(running_mean.shape, vec![3]);
    let mean_data = running_mean.to_data().unwrap();
    let mean_values = mean_data.as_slice::<f32>().unwrap();
    assert!(mean_values.iter().all(|&v| v.abs() < 1e-6));

    // Check running_var (should be ones)
    let running_var = reader.get("running_var").unwrap();
    assert_eq!(running_var.shape, vec![3]);
    let var_data = running_var.to_data().unwrap();
    let var_values = var_data.as_slice::<f32>().unwrap();
    assert!(var_values.iter().all(|&v| (v - 1.0).abs() < 1e-6));
}

#[test]
fn test_nested_dict() {
    let path = test_data_path("nested_dict.pt");
    let reader = PytorchReader::new(&path).expect("Failed to load nested_dict.pt");
    let keys = reader.keys();

    assert_eq!(keys.len(), 4);
    assert!(keys.contains(&"layer1.weight".to_string()));
    assert!(keys.contains(&"layer1.bias".to_string()));
    assert!(keys.contains(&"layer2.weight".to_string()));
    assert!(keys.contains(&"layer2.bias".to_string()));

    // Check layer1.weight and load data
    let layer1_weight = reader.get("layer1.weight").unwrap();
    assert_eq!(layer1_weight.shape, vec![2, 3]);
    assert_eq!(layer1_weight.dtype, DType::F32);
    let data = layer1_weight.to_data().unwrap();
    let values = data.as_slice::<f32>().unwrap();
    assert_eq!(values.len(), 6); // 2x3 = 6 elements

    // Check layer2.weight and load data
    let layer2_weight = reader.get("layer2.weight").unwrap();
    assert_eq!(layer2_weight.shape, vec![4, 2]);
    assert_eq!(layer2_weight.dtype, DType::F32);
    let data = layer2_weight.to_data().unwrap();
    let values = data.as_slice::<f32>().unwrap();
    assert_eq!(values.len(), 8); // 4x2 = 8 elements
}

#[test]
fn test_checkpoint() {
    let path = test_data_path("checkpoint.pt");
    let reader = PytorchReader::new(&path).expect("Failed to load checkpoint.pt");
    let keys = reader.keys();

    // Should have model_state_dict entries and optimizer entries
    assert!(keys.contains(&"model_state_dict.fc1.weight".to_string()));
    assert!(keys.contains(&"model_state_dict.fc1.bias".to_string()));
    assert!(keys.contains(&"model_state_dict.fc2.weight".to_string()));
    assert!(keys.contains(&"model_state_dict.fc2.bias".to_string()));

    // Check fc1.weight dimensions and load data
    let fc1_weight = reader.get("model_state_dict.fc1.weight").unwrap();
    assert_eq!(fc1_weight.shape, vec![10, 5]);
    let data = fc1_weight.to_data().unwrap();
    let values = data.as_slice::<f32>().unwrap();
    assert_eq!(values.len(), 50); // 10x5 = 50 elements

    // Check fc2.weight dimensions and load data
    let fc2_weight = reader.get("model_state_dict.fc2.weight").unwrap();
    assert_eq!(fc2_weight.shape, vec![3, 10]);
    let data = fc2_weight.to_data().unwrap();
    let values = data.as_slice::<f32>().unwrap();
    assert_eq!(values.len(), 30); // 3x10 = 30 elements
}

#[test]
fn test_empty_tensor() {
    let path = test_data_path("empty.pt");
    let reader = PytorchReader::new(&path).expect("Failed to load empty.pt");
    let tensor = reader.get("tensor").expect("tensor key not found");
    assert_eq!(tensor.shape, vec![0]); // Empty tensor has shape [0]
    assert_eq!(tensor.dtype, DType::F32);

    // Note: Empty tensors cannot be loaded with to_data() due to TensorData validation
    // We verify the metadata is correct, which confirms the .pt file is being read
}

#[test]
fn test_scalar_tensor() {
    let path = test_data_path("scalar.pt");
    let reader = PytorchReader::new(&path).expect("Failed to load scalar.pt");
    let tensor = reader.get("tensor").expect("tensor key not found");
    assert_eq!(tensor.shape, Vec::<usize>::new()); // Scalar has empty shape
    assert_eq!(tensor.dtype, DType::F32);

    let data = tensor.to_data().unwrap();
    let values = data.as_slice::<f32>().unwrap();
    assert_eq!(values.len(), 1);
    assert!((values[0] - 42.0).abs() < 1e-6);
}

#[test]
fn test_large_shape() {
    let path = test_data_path("large_shape.pt");
    let reader = PytorchReader::new(&path).expect("Failed to load large_shape.pt");
    let tensor = reader.get("tensor").expect("tensor key not found");
    assert_eq!(tensor.shape, vec![100, 100]);
    assert_eq!(tensor.dtype, DType::F32);

    let data = tensor.to_data().unwrap();
    let values = data.as_slice::<f32>().unwrap();
    assert_eq!(values.len(), 10000);

    // Check specific non-zero values
    assert!((values[0] - 1.0).abs() < 1e-6); // [0, 0] = 1.0
    assert!((values[5050] - 2.0).abs() < 1e-6); // [50, 50] = 2.0
    assert!((values[9999] - 3.0).abs() < 1e-6); // [99, 99] = 3.0
}

#[test]
fn test_mixed_types() {
    let path = test_data_path("mixed_types.pt");
    let reader = PytorchReader::new(&path).expect("Failed to load mixed_types.pt");
    let tensors = reader.tensors();

    assert_eq!(tensors.len(), 4);

    // Check float32 tensor [1.0, 2.0] from test_data.py
    let float32 = reader.get("float32").unwrap();
    assert_eq!(float32.dtype, DType::F32);
    assert_eq!(float32.shape, vec![2]);
    let data = float32.to_data().unwrap();
    let values = data.as_slice::<f32>().unwrap();
    assert!((values[0] - 1.0).abs() < 1e-6);
    assert!((values[1] - 2.0).abs() < 1e-6);

    // Check int64 tensor [100, 200] from test_data.py
    let int64 = reader.get("int64").unwrap();
    assert_eq!(int64.dtype, DType::I64);
    assert_eq!(int64.shape, vec![2]);
    let data = int64.to_data().unwrap();
    let values = data.as_slice::<i64>().unwrap();
    assert_eq!(values, &[100, 200]);

    // Check bool tensor [True, False] from test_data.py
    let bool_tensor = reader.get("bool").unwrap();
    assert_eq!(bool_tensor.dtype, DType::Bool);
    assert_eq!(bool_tensor.shape, vec![2]);
    let data = bool_tensor.to_data().unwrap();
    let values = data.as_slice::<bool>().unwrap();
    assert_eq!(values, &[true, false]);

    // Check float64 tensor [1.1, 2.2] from test_data.py
    let float64 = reader.get("float64").unwrap();
    assert_eq!(float64.dtype, DType::F64);
    assert_eq!(float64.shape, vec![2]);
    let data = float64.to_data().unwrap();
    let values = data.as_slice::<f64>().unwrap();
    assert!((values[0] - 1.1).abs() < 1e-10);
    assert!((values[1] - 2.2).abs() < 1e-10);
}

#[test]
fn test_special_values() {
    let path = test_data_path("special_values.pt");
    let reader = PytorchReader::new(&path).expect("Failed to load special_values.pt");
    let tensor = reader.get("tensor").expect("tensor key not found");
    assert_eq!(tensor.dtype, DType::F32);
    assert_eq!(tensor.shape, vec![5]);

    let data = tensor.to_data().unwrap();
    let values = data.as_slice::<f32>().unwrap();
    assert_eq!(values.len(), 5);

    // Check for special values
    assert!(values[0].is_nan());
    assert!(values[1].is_infinite() && values[1] > 0.0);
    assert!(values[2].is_infinite() && values[2] < 0.0);
    assert!((values[3] - 0.0).abs() < 1e-6);
    assert!((values[4] - 1.0).abs() < 1e-6);
}

#[test]
fn test_extreme_values() {
    let path = test_data_path("extreme_values.pt");
    let reader = PytorchReader::new(&path).expect("Failed to load extreme_values.pt");
    let tensor = reader.get("tensor").expect("tensor key not found");
    assert_eq!(tensor.dtype, DType::F32);
    assert_eq!(tensor.shape, vec![4]);

    let data = tensor.to_data().unwrap();
    let values = data.as_slice::<f32>().unwrap();
    assert_eq!(values.len(), 4);

    // Very small positive
    assert!(values[0] > 0.0 && values[0] < 1e-20);
    // Very large positive
    assert!(values[1] > 1e20);
    // Very small negative
    assert!(values[2] < 0.0 && values[2] > -1e-20);
    // Very large negative
    assert!(values[3] < -1e20);
}

#[test]
fn test_parameter() {
    let path = test_data_path("parameter.pt");
    let reader = PytorchReader::new(&path).expect("Failed to load parameter.pt");
    let tensors = reader.tensors();

    // nn.Parameter is typically saved as a regular tensor
    assert_eq!(tensors.len(), 1);
    let param = reader.get("param").unwrap();
    assert_eq!(param.shape, vec![3, 3]);
    assert_eq!(param.dtype, DType::F32);

    let data = param.to_data().unwrap();
    let values = data.as_slice::<f32>().unwrap();
    assert_eq!(values.len(), 9);
}

#[test]
fn test_buffers() {
    let path = test_data_path("buffers.pt");
    let reader = PytorchReader::new(&path).expect("Failed to load buffers.pt");
    let tensors = reader.tensors();

    assert_eq!(tensors.len(), 2);

    // Check buffer1 (int32)
    let buffer1 = reader.get("buffer1").unwrap();
    assert_eq!(buffer1.dtype, DType::I32);
    assert_eq!(buffer1.shape, vec![3]);
    let data1 = buffer1.to_data().unwrap();
    let data1_converted = data1.convert::<i32>();
    let values1 = data1_converted.as_slice::<i32>().unwrap();
    assert_eq!(values1, &[1, 2, 3]);

    // Check buffer2 (bool)
    let buffer2 = reader.get("buffer2").unwrap();
    assert_eq!(buffer2.dtype, DType::Bool);
    assert_eq!(buffer2.shape, vec![2]);
    let data2 = buffer2.to_data().unwrap();
    let values2 = data2.as_slice::<bool>().unwrap();
    assert_eq!(values2, &[true, false]);
}

#[test]
fn test_complex_structure() {
    let path = test_data_path("complex_structure.pt");
    let reader = PytorchReader::new(&path).expect("Failed to load complex_structure.pt");
    let keys = reader.keys();

    // Should have nested structure tensors
    assert!(keys.contains(&"state.encoder.layer_0.weight".to_string()));
    assert!(keys.contains(&"state.encoder.layer_0.bias".to_string()));
    assert!(keys.contains(&"state.encoder.layer_1.weight".to_string()));
    assert!(keys.contains(&"state.encoder.layer_1.bias".to_string()));
    assert!(keys.contains(&"state.decoder.weight".to_string()));
    assert!(keys.contains(&"state.decoder.bias".to_string()));

    // Check encoder layer_0 weight and load data
    let layer0_weight = reader.get("state.encoder.layer_0.weight").unwrap();
    assert_eq!(layer0_weight.shape, vec![4, 3]);
    let data = layer0_weight.to_data().unwrap();
    let values = data.as_slice::<f32>().unwrap();
    assert_eq!(values.len(), 12); // 4x3 = 12 elements

    // Check decoder weight and load data
    let decoder_weight = reader.get("state.decoder.weight").unwrap();
    assert_eq!(decoder_weight.shape, vec![3, 2]);
    let data = decoder_weight.to_data().unwrap();
    let values = data.as_slice::<f32>().unwrap();
    assert_eq!(values.len(), 6); // 3x2 = 6 elements
}

#[test]
fn test_read_pytorch_tensors_convenience() {
    // Test reading and materializing tensors into memory
    let path = test_data_path("state_dict.pt");
    let reader = PytorchReader::new(&path).expect("Failed to read file");

    let keys = reader.keys();
    assert_eq!(keys.len(), 4);
    assert!(keys.contains(&"weight".to_string()));
    assert!(keys.contains(&"bias".to_string()));

    // Check that data can be materialized
    let weight = reader.get("weight").unwrap();
    let weight_data = weight.to_data().unwrap();
    assert_eq!(weight_data.shape, vec![3, 4]);
    assert_eq!(weight_data.dtype, DType::F32);
}

#[test]
fn test_with_top_level_key() {
    // Test loading with a specific top-level key
    let path = test_data_path("checkpoint.pt");

    // Load only model_state_dict
    let reader = PytorchReader::with_top_level_key(&path, "model_state_dict")
        .expect("Failed to load with top-level key");

    let keys = reader.keys();
    // Should only have model weights, not optimizer state
    assert!(keys.contains(&"fc1.weight".to_string()));
    assert!(keys.contains(&"fc1.bias".to_string()));
    assert!(keys.contains(&"fc2.weight".to_string()));
    assert!(keys.contains(&"fc2.bias".to_string()));

    // Should NOT have nested paths with model_state_dict prefix
    assert!(!keys.contains(&"model_state_dict.fc1.weight".to_string()));
}

#[test]
fn test_legacy_format() {
    // Test loading PyTorch legacy format (pre-1.6)
    let path = test_data_path("simple_legacy.pt");

    // This file has the sequential pickle structure of legacy PyTorch format
    let reader = PytorchReader::new(&path).expect("Failed to load legacy format");
    let keys = reader.keys();

    // Should have the tensors from the state dict
    assert!(keys.contains(&"weight".to_string()), "Missing 'weight' key");
    assert!(keys.contains(&"bias".to_string()), "Missing 'bias' key");
    assert!(
        keys.contains(&"running_mean".to_string()),
        "Missing 'running_mean' key"
    );

    // Check weight tensor
    let weight = reader.get("weight").expect("weight not found");
    assert_eq!(weight.shape, vec![2, 3]);
    assert_eq!(weight.dtype, DType::F32);

    // Check bias tensor
    let bias = reader.get("bias").expect("bias not found");
    assert_eq!(bias.shape, vec![2]);
    assert_eq!(bias.dtype, DType::F32);

    // Verify bias values are all ones
    let bias_data = bias.to_data().unwrap();
    let bias_values = bias_data.as_slice::<f32>().unwrap();
    // Note: values in simple_legacy.pt are randomly generated, not necessarily 1.0
    assert_eq!(bias_values.len(), 2);

    // Check running_mean tensor
    let running_mean = reader.get("running_mean").expect("running_mean not found");
    assert_eq!(running_mean.shape, vec![2]);
    assert_eq!(running_mean.dtype, DType::F32);

    // Verify running_mean values are accessible
    let mean_data = running_mean.to_data().unwrap();
    let mean_values = mean_data.as_slice::<f32>().unwrap();
    assert_eq!(mean_values.len(), 2);
}

#[test]
fn test_legacy_with_offsets() {
    // Test with legacy format file that has storage offsets
    let path = test_data_path("legacy_with_offsets.pt");
    let reader = PytorchReader::new(&path).expect("Should read legacy file with offsets");

    let keys = reader.keys();
    assert_eq!(keys.len(), 3, "Should have 3 tensors");

    // Check that tensors exist
    for key in &keys {
        assert!(reader.get(key).is_some(), "Should have tensor: {}", key);
        let tensor = reader.get(key).unwrap();
        let data = tensor.to_data().unwrap();
        let values = data.as_slice::<f32>().unwrap();
        assert!(!values.is_empty(), "Tensor {} should have data", key);
    }
}

#[test]
fn test_legacy_shared_storage() {
    // Test with legacy format file that has shared storage
    let path = test_data_path("legacy_shared_storage.pt");
    let reader = PytorchReader::new(&path).expect("Should read legacy file with shared storage");

    let keys = reader.keys();
    assert!(keys.len() >= 2, "Should have at least 2 tensors");

    // Check that tensors exist and can be loaded
    for key in &keys {
        assert!(reader.get(key).is_some(), "Should have tensor: {}", key);
        let tensor = reader.get(key).unwrap();
        let data = tensor.to_data().unwrap();

        // Verify tensor data can be accessed
        match tensor.dtype {
            DType::F32 => {
                let values = data.as_slice::<f32>().unwrap();
                assert!(!values.is_empty(), "Tensor {} should have data", key);
            }
            DType::I64 => {
                let values = data.as_slice::<i64>().unwrap();
                assert!(!values.is_empty(), "Tensor {} should have data", key);
            }
            _ => {
                // For other types, just verify we can convert to data
                assert!(!data.shape.is_empty(), "Tensor {} should have shape", key);
            }
        }
    }
}

#[test]
fn test_metadata_zip_format() {
    // Test that metadata is properly populated for ZIP format files
    let path = test_data_path("float32.pt");
    let reader = PytorchReader::new(&path).expect("Failed to load float32.pt");

    // Check metadata
    let metadata = reader.metadata();
    assert_eq!(metadata.format_type, FileFormat::Zip);
    assert_eq!(metadata.byte_order, ByteOrder::LittleEndian);
    assert_eq!(metadata.tensor_count, 1);
    assert!(metadata.total_data_size.is_some());

    // Check that metadata is accessible
    assert!(metadata.is_modern_format());
    assert!(!metadata.is_legacy_format());
}

#[test]
fn test_metadata_legacy_format() {
    // Test that metadata is properly populated for legacy format files
    let path = test_data_path("simple_legacy.pt");
    let reader = PytorchReader::new(&path).expect("Failed to load legacy file");

    // Check metadata
    let metadata = reader.metadata();
    assert_eq!(metadata.format_type, FileFormat::Legacy);
    assert_eq!(metadata.byte_order, ByteOrder::LittleEndian);
    assert_eq!(metadata.tensor_count, 3); // weight, bias, running_mean
    assert!(metadata.total_data_size.is_some());
}

#[test]
fn test_legacy_metadata_detailed() {
    // Detailed test to prove we load all metadata for legacy format files
    let path = test_data_path("simple_legacy.pt");
    let reader = PytorchReader::new(&path).expect("Failed to load legacy file");

    // Get and examine metadata
    let metadata = reader.metadata();

    // Verify the metadata is correct for legacy format
    assert_eq!(
        metadata.format_type,
        FileFormat::Legacy,
        "Should be Legacy format"
    );
    assert_eq!(
        metadata.byte_order,
        ByteOrder::LittleEndian,
        "Legacy format is little-endian"
    );
    assert_eq!(
        metadata.tensor_count, 3,
        "Should have 3 tensors: weight, bias, running_mean"
    );
    assert!(
        metadata.total_data_size.is_some(),
        "Should have total data size"
    );
    assert!(
        metadata.total_data_size.unwrap() > 0,
        "Data size should be positive"
    );

    // Legacy format specifics
    assert_eq!(
        metadata.format_version, None,
        "Legacy format doesn't have version file"
    );
    assert_eq!(
        metadata.pytorch_version, None,
        "Legacy format doesn't store PyTorch version reliably"
    );
    assert!(
        !metadata.has_storage_alignment,
        "Legacy format doesn't have storage alignment"
    );

    // Also verify we can access the tensors
    let keys = reader.keys();
    assert!(
        keys.contains(&"weight".to_string()),
        "Should have weight tensor"
    );
    assert!(
        keys.contains(&"bias".to_string()),
        "Should have bias tensor"
    );
    assert!(
        keys.contains(&"running_mean".to_string()),
        "Should have running_mean tensor"
    );
}

#[test]
fn test_small_invalid_file() {
    // Test that we handle broken/invalid files gracefully
    let path = test_data_path("broken.pt");

    // Should fail gracefully with an appropriate error
    let result = PytorchReader::new(&path);
    assert!(result.is_err(), "Expected error for broken file");

    // The error should be a pickle error since the file is too small to be valid
    if let Err(e) = result {
        let err_str = format!("{}", e);
        assert!(
            err_str.contains("Pickle") || err_str.contains("Invalid"),
            "Error should mention pickle or invalid format: {}",
            err_str
        );
    }
}

#[test]
fn test_read_pickle_data_basic() {
    use crate::pytorch::reader::PickleValue;

    // Test reading pickle data from a checkpoint file
    let path = test_data_path("checkpoint.pt");

    // Read the entire pickle data
    let data = PytorchReader::read_pickle_data(&path, None).expect("Failed to read pickle data");

    // Should be a dictionary at the root
    if let PickleValue::Dict(dict) = data {
        // Check that expected keys exist
        assert!(dict.contains_key("model_state_dict"));
        assert!(dict.contains_key("optimizer_state_dict"));
        assert!(dict.contains_key("epoch"));
        assert!(dict.contains_key("loss"));

        // Check epoch value
        if let Some(PickleValue::Int(epoch)) = dict.get("epoch") {
            assert_eq!(*epoch, 42);
        } else {
            panic!("Expected epoch to be an integer");
        }

        // Check loss value
        if let Some(PickleValue::Float(loss)) = dict.get("loss") {
            assert!(*loss > 0.0 && *loss < 1.0, "Loss should be between 0 and 1");
        } else {
            panic!("Expected loss to be a float");
        }
    } else {
        panic!("Expected root to be a dictionary");
    }
}

#[test]
fn test_read_pickle_data_with_key() {
    use crate::pytorch::reader::PickleValue;

    // Test reading specific key from checkpoint
    let path = test_data_path("checkpoint.pt");

    // Read only the model_state_dict
    let data = PytorchReader::read_pickle_data(&path, Some("model_state_dict"))
        .expect("Failed to read pickle data with key");

    // Should get the model_state_dict directly
    if let PickleValue::Dict(dict) = data {
        // Should have model weights
        assert!(dict.contains_key("fc1.weight"));
        assert!(dict.contains_key("fc1.bias"));
        assert!(dict.contains_key("fc2.weight"));
        assert!(dict.contains_key("fc2.bias"));

        // Should NOT have optimizer keys
        assert!(!dict.contains_key("optimizer_state_dict"));
        assert!(!dict.contains_key("epoch"));
    } else {
        panic!("Expected model_state_dict to be a dictionary");
    }
}

#[test]
fn test_read_pickle_data_nested_structure() {
    use crate::pytorch::reader::PickleValue;

    // Test reading nested dictionary structure
    let path = test_data_path("nested_dict.pt");

    let data =
        PytorchReader::read_pickle_data(&path, None).expect("Failed to read nested structure");

    if let PickleValue::Dict(dict) = data {
        // nested_dict.pt has a nested structure, not flat keys
        // It should have layer1 and layer2 as nested dicts
        assert!(!dict.is_empty(), "Dictionary should not be empty");

        // The structure depends on how the file was saved
        // It could be flat keys like "layer1.weight" or nested dicts
        // Just verify it's a valid dict structure
        for (_key, value) in dict.iter() {
            // Values could be None (tensors), nested dicts, or other types
            assert!(
                matches!(value, PickleValue::None | PickleValue::Dict(_)),
                "Values should be None or nested dicts"
            );
        }
    } else {
        panic!("Expected nested_dict to be a dictionary");
    }
}

#[test]
fn test_read_pickle_data_types() {
    use crate::pytorch::reader::PickleValue;

    // Test various data types in mixed_types.pt
    let path = test_data_path("mixed_types.pt");

    let data = PytorchReader::read_pickle_data(&path, None).expect("Failed to read mixed types");

    if let PickleValue::Dict(dict) = data {
        // The file contains different tensor types
        assert!(dict.len() >= 3, "Should have at least 3 tensor types");

        // All tensor values should be None in pickle data
        for (_key, value) in dict.iter() {
            // All values should be None (tensors are not included in pickle data)
            assert!(
                matches!(value, PickleValue::None),
                "Tensors should be None in pickle data"
            );
        }
    } else {
        panic!("Expected mixed_types to be a dictionary");
    }
}

#[test]
fn test_read_pickle_data_key_not_found() {
    // Test error handling when key doesn't exist
    let path = test_data_path("checkpoint.pt");

    let result = PytorchReader::read_pickle_data(&path, Some("nonexistent_key"));
    assert!(result.is_err());

    if let Err(e) = result {
        let err_str = format!("{}", e);
        assert!(
            err_str.contains("not found"),
            "Error should mention key not found: {}",
            err_str
        );
    }
}

#[test]
fn test_read_pickle_data_simple_pickle() {
    use crate::pytorch::reader::PickleValue;

    // Test reading a simple pickle file (not ZIP)
    // Note: simple_legacy.pt is a legacy format file, not a simple pickle
    // It may return None because legacy format reading is different
    let path = test_data_path("state_dict.pt"); // Use a proper simple pickle file

    let data = PytorchReader::read_pickle_data(&path, None).expect("Failed to read simple pickle");

    // Should contain state dict entries
    if let PickleValue::Dict(dict) = data {
        // state_dict.pt has weight, bias, running_mean, running_var
        assert!(dict.len() >= 3);
        assert!(dict.contains_key("weight"));
        assert!(dict.contains_key("bias"));

        // All tensor values should be None in pickle data
        for (_key, value) in dict.iter() {
            assert!(matches!(value, PickleValue::None));
        }
    } else {
        panic!("Expected state_dict to contain a dictionary");
    }
}

#[test]
fn test_load_config_basic() {
    let path = test_data_path("checkpoint.pt");

    // Define a struct that matches part of the checkpoint data
    #[derive(Debug, serde::Deserialize, PartialEq)]
    struct CheckpointConfig {
        epoch: i64,
        loss: f64,
    }

    // Load config
    let config: CheckpointConfig =
        PytorchReader::load_config(&path, None).expect("Failed to load config");

    // Verify values - based on test_read_pickle_data_basic
    assert_eq!(config.epoch, 42);
    assert!((config.loss - 0.123).abs() < 1e-6);
}

#[test]
fn test_load_config_with_top_level_key() {
    // Test that we can extract a non-existent key and get an appropriate error
    let path = test_data_path("checkpoint.pt");

    #[derive(Debug, serde::Deserialize, PartialEq)]
    struct DummyConfig {
        field: String,
    }

    // Try loading with a valid top-level key that exists but has wrong structure
    let result: Result<DummyConfig, _> = PytorchReader::load_config(&path, Some("epoch"));

    // This should fail because epoch is an integer, not a struct with a field
    assert!(result.is_err());

    // Now test that we can load with a real key that has the right structure
    // Since checkpoint.pt doesn't have nested configs, let's use nested_dict.pt
    let path2 = test_data_path("nested_dict.pt");

    // Try to extract a specific nested key if it exists
    // Since nested_dict has complex structure, let's just verify we can read it
    let data = PytorchReader::read_pickle_data(&path2, None).unwrap();

    // Verify it's a dict
    if let crate::pytorch::reader::PickleValue::Dict(dict) = data {
        assert!(!dict.is_empty());
    } else {
        panic!("Expected a dict");
    }
}

#[test]
fn test_load_config_complex_types() {
    // For this test, let's create a comprehensive test using checkpoint.pt
    // which has both metadata and state_dict fields
    let path = test_data_path("checkpoint.pt");

    // Define a partial config that only captures metadata fields
    #[derive(Debug, serde::Deserialize, PartialEq)]
    struct PartialCheckpoint {
        epoch: i64,
        loss: f64,
        // We skip model_state_dict and optimizer_state_dict
        // as they contain tensor references that become None
    }

    // Load partial config
    let config: PartialCheckpoint =
        PytorchReader::load_config(&path, None).expect("Failed to load config");

    // Verify we can extract the metadata
    assert_eq!(config.epoch, 42);
    assert!((config.loss - 0.123).abs() < 1e-6);
}

#[test]
fn test_load_config_key_not_found() {
    let path = test_data_path("checkpoint.pt");

    #[derive(Debug, serde::Deserialize)]
    struct DummyConfig {
        #[allow(dead_code)]
        field: String,
    }

    // Try to load with non-existent key
    let result: Result<DummyConfig, _> = PytorchReader::load_config(&path, Some("nonexistent"));

    assert!(result.is_err());
    let error = result.unwrap_err();
    assert!(error.to_string().contains("not found") || error.to_string().contains("Key"));
}

#[test]
fn test_pickle_value_conversion() {
    use crate::pytorch::reader::PickleValue;

    // Test that PickleValue provides useful data structures
    let path = test_data_path("checkpoint.pt");
    let data = PytorchReader::read_pickle_data(&path, None).unwrap();

    // Test pattern matching and data extraction
    match data {
        PickleValue::Dict(dict) => {
            // Extract epoch as integer
            if let Some(PickleValue::Int(epoch)) = dict.get("epoch") {
                assert!(*epoch >= 0);
            }

            // Extract loss as float
            if let Some(PickleValue::Float(loss)) = dict.get("loss") {
                assert!(loss.is_finite());
            }

            // Test nested access
            if let Some(PickleValue::Dict(model_dict)) = dict.get("model_state_dict") {
                assert!(!model_dict.is_empty());
            }
        }
        _ => panic!("Unexpected root type"),
    }
}
