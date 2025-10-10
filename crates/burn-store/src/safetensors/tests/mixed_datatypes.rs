use burn::{
    module::{Module, Param, ParamId},
    nn,
    tensor::{Bool, Int, Tensor, backend::Backend},
};

use crate::{ModuleSnapshot, SafetensorsStore};

/// Simple model with different data types for testing
#[derive(Module, Debug)]
pub struct MixedDtypeModel<B: Backend> {
    // Standard neural network layers (float tensors)
    linear: nn::Linear<B>,

    // Direct tensor parameters of different types
    float_tensor: Param<Tensor<B, 2>>,

    int_tensor: Param<Tensor<B, 2, Int>>,

    bool_tensor: Param<Tensor<B, 2, Bool>>,
}

impl<B: Backend> MixedDtypeModel<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            linear: nn::LinearConfig::new(3, 3).init(device),

            // Simple float values
            float_tensor: Param::from_tensor(Tensor::from_floats(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                device,
            )),

            // Simple integer values
            int_tensor: Param::initialized(
                ParamId::new(),
                Tensor::from_ints([[1, 2, 3], [4, 5, 6]], device),
            ),

            // Simple boolean values
            bool_tensor: Param::initialized(
                ParamId::new(),
                Tensor::from_bool(
                    burn::tensor::TensorData::new(
                        vec![true, false, true, false, true, false],
                        [2, 3],
                    ),
                    device,
                ),
            ),
        }
    }
}

#[cfg(test)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;

    #[test]
    fn test_mixed_dtypes_round_trip() {
        type TestBackend = burn_ndarray::NdArray<f32>;
        let device = Default::default();

        // Create model with mixed data types
        let model = MixedDtypeModel::<TestBackend>::new(&device);

        // Save to bytes
        let mut save_store = SafetensorsStore::from_bytes(None);
        model.save_into(&mut save_store).expect("Failed to save");
        let bytes = save_store.get_bytes().expect("Failed to get bytes");

        // Load into a new model
        let mut load_store = SafetensorsStore::from_bytes(Some(bytes));
        let mut loaded_model = MixedDtypeModel::<TestBackend>::new(&device);
        loaded_model
            .load_from(&mut load_store)
            .expect("Failed to load");

        // Verify float tensor is preserved
        let orig_float = model.float_tensor.val().into_data();
        let loaded_float = loaded_model.float_tensor.val().into_data();
        assert_eq!(orig_float, loaded_float, "Float tensor not preserved");

        // Verify integer tensor is preserved
        let orig_int = model.int_tensor.val().into_data();
        let loaded_int = loaded_model.int_tensor.val().into_data();
        assert_eq!(orig_int, loaded_int, "Integer tensor not preserved");

        // Verify boolean tensor is preserved
        let orig_bool = model.bool_tensor.val().into_data();
        let loaded_bool = loaded_model.bool_tensor.val().into_data();
        assert_eq!(orig_bool, loaded_bool, "Boolean tensor not preserved");
    }

    #[test]
    fn test_dtype_detection() {
        type TestBackend = burn_ndarray::NdArray<f32>;
        let device = Default::default();

        let model = MixedDtypeModel::<TestBackend>::new(&device);
        let snapshots = model.collect(None, None);

        for snapshot in snapshots {
            let path = snapshot.full_path();
            let dtype = snapshot.dtype;

            if path.contains("float_tensor") || path.contains("linear") {
                assert_eq!(
                    dtype,
                    burn::tensor::DType::F32,
                    "Float tensor {} should have F32 dtype",
                    path
                );
            } else if path.contains("int_tensor") {
                assert!(
                    matches!(
                        dtype,
                        burn::tensor::DType::I64
                            | burn::tensor::DType::I32
                            | burn::tensor::DType::I16
                            | burn::tensor::DType::I8
                    ),
                    "Integer tensor {} should have integer dtype, got {:?}",
                    path,
                    dtype
                );
            } else if path.contains("bool_tensor") {
                assert_eq!(
                    dtype,
                    burn::tensor::DType::Bool,
                    "Boolean tensor {} should have Bool dtype",
                    path
                );
            }
        }
    }

    #[test]
    fn test_extreme_values() {
        type TestBackend = burn_ndarray::NdArray<f32>;
        let device = <TestBackend as Backend>::Device::default();

        #[derive(Module, Debug)]
        struct ExtremeValueModel<B: Backend> {
            large_floats: Param<Tensor<B, 1>>,
            small_floats: Param<Tensor<B, 1>>,
            large_ints: Param<Tensor<B, 1, Int>>,
        }

        impl<B: Backend> ExtremeValueModel<B> {
            fn new(device: &B::Device) -> Self {
                Self {
                    large_floats: Param::from_tensor(Tensor::from_floats(
                        [1e30, -1e30, f32::MAX, f32::MIN],
                        device,
                    )),
                    small_floats: Param::from_tensor(Tensor::from_floats(
                        [1e-30, -1e-30, f32::MIN_POSITIVE, f32::EPSILON],
                        device,
                    )),
                    large_ints: Param::initialized(
                        ParamId::new(),
                        Tensor::from_ints([i32::MAX, i32::MIN, 0, -1], device),
                    ),
                }
            }
        }

        let model = ExtremeValueModel::<TestBackend>::new(&device);

        // Save and load
        let mut save_store = SafetensorsStore::from_bytes(None);
        model.save_into(&mut save_store).expect("Failed to save");
        let bytes = save_store.get_bytes().expect("Failed to get bytes");

        let mut load_store = SafetensorsStore::from_bytes(Some(bytes));
        let mut loaded_model = ExtremeValueModel::<TestBackend>::new(&device);
        loaded_model
            .load_from(&mut load_store)
            .expect("Failed to load");

        // Check exact preservation
        assert_eq!(
            model.large_floats.val().into_data(),
            loaded_model.large_floats.val().into_data(),
            "Large floats not preserved"
        );
        assert_eq!(
            model.small_floats.val().into_data(),
            loaded_model.small_floats.val().into_data(),
            "Small floats not preserved"
        );
        assert_eq!(
            model.large_ints.val().into_data(),
            loaded_model.large_ints.val().into_data(),
            "Large integers not preserved"
        );
    }

    #[test]
    fn test_mixed_precision_floats() {
        // Note: While SafeTensors format supports storing tensors with different precisions
        // (F16, BF16, F32, F64, etc.) in the same file, Burn's backend architecture currently
        // requires all tensors in a model instance to share the same floating-point precision.
        // This is determined at the backend level (e.g., NdArray<f32> or NdArray<f64>).
        //
        // However, for storage purposes, SafeTensors can correctly save and load tensors
        // with their original precision, preserving the data type information in the file format.
        // This test demonstrates that different precision backends work correctly with SafeTensors.

        // Test with f32 backend
        {
            type TestBackend = burn_ndarray::NdArray<f32>;
            let device = Default::default();

            let model = MixedDtypeModel::<TestBackend>::new(&device);

            // Save to bytes
            let mut save_store = SafetensorsStore::from_bytes(None);
            model.save_into(&mut save_store).expect("Failed to save");
            let bytes = save_store.get_bytes().expect("Failed to get bytes");

            // Load and verify
            let mut load_store = SafetensorsStore::from_bytes(Some(bytes));
            let mut loaded_model = MixedDtypeModel::<TestBackend>::new(&device);
            loaded_model
                .load_from(&mut load_store)
                .expect("Failed to load");

            assert_eq!(
                model.float_tensor.val().into_data(),
                loaded_model.float_tensor.val().into_data(),
                "F32 float tensor not preserved"
            );
        }

        // Test with f64 backend
        {
            type TestBackend = burn_ndarray::NdArray<f64>;
            let device = Default::default();

            #[derive(Module, Debug)]
            struct F64Model<B: Backend> {
                linear: nn::Linear<B>,
                double_precision: Param<Tensor<B, 2>>,
            }

            let model = F64Model::<TestBackend> {
                linear: nn::LinearConfig::new(2, 2).init(&device),
                double_precision: Param::from_tensor(Tensor::from_floats(
                    [
                        [1.234567890123456789, 2.345678901234567890],
                        [3.456789012345678901, 4.567890123456789012],
                    ],
                    &device,
                )),
            };

            // Save to bytes
            let mut save_store = SafetensorsStore::from_bytes(None);
            model.save_into(&mut save_store).expect("Failed to save");
            let bytes = save_store.get_bytes().expect("Failed to get bytes");

            // Load and verify
            let mut load_store = SafetensorsStore::from_bytes(Some(bytes));
            let mut loaded_model = F64Model::<TestBackend> {
                linear: nn::LinearConfig::new(2, 2).init(&device),
                double_precision: Param::from_tensor(Tensor::zeros([2, 2], &device)),
            };
            loaded_model
                .load_from(&mut load_store)
                .expect("Failed to load");

            let orig = model.double_precision.val().into_data();
            let loaded = loaded_model.double_precision.val().into_data();
            assert_eq!(orig, loaded, "F64 double precision not preserved");
        }
    }

    #[test]
    fn test_mixed_precision_integers() {
        type TestBackend = burn_ndarray::NdArray<f32>;
        let device = Default::default();

        #[derive(Module, Debug)]
        struct MultiIntModel<B: Backend> {
            // Note: Burn's Tensor<B, D, Int> uses the backend's default int type
            // We can't directly specify i8, i16, etc. in the type system
            // But we can test with different values that would fit in different ranges
            small_ints: Param<Tensor<B, 1, Int>>, // Values that fit in i8
            medium_ints: Param<Tensor<B, 1, Int>>, // Values that fit in i16
            large_ints: Param<Tensor<B, 1, Int>>, // Values that need i32/i64
        }

        let model = MultiIntModel::<TestBackend> {
            small_ints: Param::initialized(
                ParamId::new(),
                Tensor::from_ints([127i32, -128, 0, 42], &device),
            ),
            medium_ints: Param::initialized(
                ParamId::new(),
                Tensor::from_ints([32767i32, -32768, 1000, -1000], &device),
            ),
            large_ints: Param::initialized(
                ParamId::new(),
                Tensor::from_ints([i32::MAX, i32::MIN, 1_000_000, -1_000_000], &device),
            ),
        };

        // Save to bytes
        let mut save_store = SafetensorsStore::from_bytes(None);
        model.save_into(&mut save_store).expect("Failed to save");
        let bytes = save_store.get_bytes().expect("Failed to get bytes");

        // Load and verify
        let mut load_store = SafetensorsStore::from_bytes(Some(bytes));
        let mut loaded_model = MultiIntModel::<TestBackend> {
            small_ints: Param::initialized(ParamId::new(), Tensor::zeros([4], &device)),
            medium_ints: Param::initialized(ParamId::new(), Tensor::zeros([4], &device)),
            large_ints: Param::initialized(ParamId::new(), Tensor::zeros([4], &device)),
        };
        loaded_model
            .load_from(&mut load_store)
            .expect("Failed to load");

        assert_eq!(
            model.small_ints.val().into_data(),
            loaded_model.small_ints.val().into_data(),
            "Small ints (i8 range) not preserved"
        );
        assert_eq!(
            model.medium_ints.val().into_data(),
            loaded_model.medium_ints.val().into_data(),
            "Medium ints (i16 range) not preserved"
        );
        assert_eq!(
            model.large_ints.val().into_data(),
            loaded_model.large_ints.val().into_data(),
            "Large ints (i32 range) not preserved"
        );
    }

    #[test]
    fn test_comprehensive_mixed_types() {
        type TestBackend = burn_ndarray::NdArray<f32>;
        let device = Default::default();

        #[derive(Module, Debug)]
        struct ComprehensiveModel<B: Backend> {
            // Neural network layers
            linear1: nn::Linear<B>,
            conv2d: nn::conv::Conv2d<B>,

            // Different tensor types
            float32_weights: Param<Tensor<B, 3>>,
            integer_indices: Param<Tensor<B, 2, Int>>,
            boolean_mask: Param<Tensor<B, 2, Bool>>,
        }

        let model = ComprehensiveModel::<TestBackend> {
            linear1: nn::LinearConfig::new(4, 8).init(&device),
            conv2d: nn::conv::Conv2dConfig::new([3, 16], [3, 3]).init(&device),

            float32_weights: Param::from_tensor(Tensor::from_floats(
                [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
                &device,
            )),

            integer_indices: Param::initialized(
                ParamId::new(),
                Tensor::from_ints(
                    [[0, 1, 2, 3], [10, 20, 30, 40], [100, 200, 300, 400]],
                    &device,
                ),
            ),

            boolean_mask: Param::initialized(
                ParamId::new(),
                Tensor::from_bool(
                    burn::tensor::TensorData::new(
                        vec![true, false, false, true, false, true, true, false],
                        [2, 4],
                    ),
                    &device,
                ),
            ),
        };

        // Collect all tensors
        let snapshots = model.collect(None, None);

        // Verify we have all expected tensors
        let paths: Vec<String> = snapshots.iter().map(|s| s.full_path()).collect();
        assert!(paths.iter().any(|p| p.contains("linear1")));
        assert!(paths.iter().any(|p| p.contains("conv2d")));
        assert!(paths.iter().any(|p| p.contains("float32_weights")));
        assert!(paths.iter().any(|p| p.contains("integer_indices")));
        assert!(paths.iter().any(|p| p.contains("boolean_mask")));

        // Save to bytes
        let mut save_store = SafetensorsStore::from_bytes(None);
        model.save_into(&mut save_store).expect("Failed to save");
        let bytes = save_store.get_bytes().expect("Failed to get bytes");

        // Load into fresh model
        let mut load_store = SafetensorsStore::from_bytes(Some(bytes));
        let mut loaded_model = ComprehensiveModel::<TestBackend> {
            linear1: nn::LinearConfig::new(4, 8).init(&device),
            conv2d: nn::conv::Conv2dConfig::new([3, 16], [3, 3]).init(&device),
            float32_weights: Param::from_tensor(Tensor::zeros([2, 2, 2], &device)),
            integer_indices: Param::initialized(ParamId::new(), Tensor::zeros([3, 4], &device)),
            boolean_mask: Param::initialized(
                ParamId::new(),
                Tensor::from_bool(
                    burn::tensor::TensorData::new(vec![false; 8], [2, 4]),
                    &device,
                ),
            ),
        };
        loaded_model
            .load_from(&mut load_store)
            .expect("Failed to load");

        // Verify all data is preserved
        assert_eq!(
            model.float32_weights.val().into_data(),
            loaded_model.float32_weights.val().into_data(),
            "Float32 weights not preserved"
        );
        assert_eq!(
            model.integer_indices.val().into_data(),
            loaded_model.integer_indices.val().into_data(),
            "Integer indices not preserved"
        );
        assert_eq!(
            model.boolean_mask.val().into_data(),
            loaded_model.boolean_mask.val().into_data(),
            "Boolean mask not preserved"
        );
    }
}
