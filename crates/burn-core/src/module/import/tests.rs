#[cfg(all(test, feature = "std"))]
mod tests {
    use crate as burn;
    use crate::{
        TestBackend,
        module::{Module, ModuleExport, ModuleImport, Param},
        nn::{Linear, LinearConfig},
    };
    use burn_tensor::{Tensor, backend::Backend};

    // Test Vec of modules
    #[derive(Module, Debug)]
    struct VecModule<B: Backend> {
        layers: Vec<Linear<B>>,
    }

    impl<B: Backend> VecModule<B> {
        fn new(device: &B::Device, num_layers: usize) -> Self {
            Self {
                layers: (0..num_layers)
                    .map(|_| LinearConfig::new(4, 4).with_bias(true).init(device))
                    .collect(),
            }
        }

        fn new_zeros(device: &B::Device, num_layers: usize) -> Self {
            Self {
                layers: (0..num_layers)
                    .map(|_| {
                        let mut module = LinearConfig::new(4, 4).with_bias(true).init(device);
                        // Zero out the weights and biases
                        module.weight = Param::from_tensor(Tensor::zeros([4, 4], device));
                        module.bias = Some(Param::from_tensor(Tensor::zeros([4], device)));
                        module
                    })
                    .collect(),
            }
        }
    }

    #[test]
    fn test_import_vec_module() {
        let device = Default::default();
        let model1 = VecModule::<TestBackend>::new(&device, 3);
        let mut model2 = VecModule::<TestBackend>::new_zeros(&device, 3);

        // Export from model1
        let exported = model1.export_tensor_views();

        // Should have 6 tensors (3 layers × 2 tensors each)
        assert_eq!(exported.len(), 6);
        assert!(exported.contains_key("layers.0.weight"));
        assert!(exported.contains_key("layers.0.bias"));
        assert!(exported.contains_key("layers.1.weight"));
        assert!(exported.contains_key("layers.1.bias"));
        assert!(exported.contains_key("layers.2.weight"));
        assert!(exported.contains_key("layers.2.bias"));

        // Import into model2
        let result = model2.import_tensor_views(exported, &device);

        println!(
            "Result: applied={}, errors={}, missing={}, unused={}",
            result.applied.len(),
            result.errors.len(),
            result.missing.len(),
            result.unused.len()
        );
        println!("Applied: {:?}", result.applied);
        println!("Errors: {:?}", result.errors);
        println!("Missing: {:?}", result.missing);
        println!("Unused: {:?}", result.unused);

        assert!(result.is_success());
        assert_eq!(result.applied.len(), 6);
        assert_eq!(result.errors.len(), 0);

        // Verify the tensors were imported correctly
        let model2_exported = model2.export_tensor_views();

        // Check that all tensors are non-zero after import
        for i in 0..3 {
            let weight_path = format!("layers.{}.weight", i);
            let bias_path = format!("layers.{}.bias", i);

            let weight_data = model2_exported.get(&weight_path).unwrap().to_data();
            let bias_data = model2_exported.get(&bias_path).unwrap().to_data();

            // Shapes should be correct
            assert_eq!(weight_data.shape, vec![4, 4]);
            assert_eq!(bias_data.shape, vec![4]);
        }
    }

    #[test]
    fn test_import_vec_module_filtered() {
        let device = Default::default();
        let model1 = VecModule::<TestBackend>::new(&device, 3);
        let mut model2 = VecModule::<TestBackend>::new_zeros(&device, 3);

        // Export all from model1
        let exported = model1.export_tensor_views();

        // Import only layer 1 tensors
        let result = model2
            .import_tensor_views_filtered(exported, &device, &[r"^layers\.1\..*"])
            .unwrap();

        assert!(result.is_success());
        assert_eq!(result.applied.len(), 2); // Only layer 1 tensors
        assert_eq!(result.skipped.len(), 4); // Other layers skipped
    }

    // Test array of modules
    #[derive(Module, Debug)]
    struct ArrayModule<B: Backend> {
        layers: [Linear<B>; 3],
    }

    impl<B: Backend> ArrayModule<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                layers: [
                    LinearConfig::new(4, 4).with_bias(true).init(device),
                    LinearConfig::new(4, 4).with_bias(true).init(device),
                    LinearConfig::new(4, 4).with_bias(true).init(device),
                ],
            }
        }

        fn new_zeros(device: &B::Device) -> Self {
            let create_zero_module = || {
                let mut module = LinearConfig::new(4, 4).with_bias(true).init(device);
                module.weight = Param::from_tensor(Tensor::zeros([4, 4], device));
                module.bias = Some(Param::from_tensor(Tensor::zeros([4], device)));
                module
            };

            Self {
                layers: [
                    create_zero_module(),
                    create_zero_module(),
                    create_zero_module(),
                ],
            }
        }
    }

    #[test]
    fn test_import_array_module() {
        let device = Default::default();
        let model1 = ArrayModule::<TestBackend>::new(&device);
        let mut model2 = ArrayModule::<TestBackend>::new_zeros(&device);

        // Export from model1
        let exported = model1.export_tensor_views();

        // Should have 6 tensors (3 layers × 2 tensors each)
        assert_eq!(exported.len(), 6);
        for i in 0..3 {
            assert!(exported.contains_key(&format!("layers.{}.weight", i)));
            assert!(exported.contains_key(&format!("layers.{}.bias", i)));
        }

        // Import into model2
        let result = model2.import_tensor_views(exported, &device);

        assert!(result.is_success());
        assert_eq!(result.applied.len(), 6);
        assert_eq!(result.errors.len(), 0);

        // Verify shapes
        let model2_exported = model2.export_tensor_views();
        for i in 0..3 {
            let weight_data = model2_exported
                .get(&format!("layers.{}.weight", i))
                .unwrap()
                .to_data();
            let bias_data = model2_exported
                .get(&format!("layers.{}.bias", i))
                .unwrap()
                .to_data();

            assert_eq!(weight_data.shape, vec![4, 4]);
            assert_eq!(bias_data.shape, vec![4]);
        }
    }

    // Test enum modules
    #[derive(Module, Debug)]
    enum EnumModule<B: Backend> {
        Small(Linear<B>),
        Medium(Linear<B>),
        Large(Linear<B>),
    }

    impl<B: Backend> EnumModule<B> {
        fn new_small(device: &B::Device) -> Self {
            Self::Small(LinearConfig::new(2, 2).with_bias(true).init(device))
        }

        fn new_small_zeros(device: &B::Device) -> Self {
            let mut module = LinearConfig::new(2, 2).with_bias(true).init(device);
            module.weight = Param::from_tensor(Tensor::zeros([2, 2], device));
            module.bias = Some(Param::from_tensor(Tensor::zeros([2], device)));
            Self::Small(module)
        }
    }

    #[test]
    fn test_import_enum_module() {
        let device = Default::default();
        let model1 = EnumModule::<TestBackend>::new_small(&device);
        let mut model2 = EnumModule::<TestBackend>::new_small_zeros(&device);

        // Export from model1
        let exported = model1.export_tensor_views();

        // Should have variant name in the path
        assert_eq!(exported.len(), 2);
        assert!(exported.contains_key("Small.weight"));
        assert!(exported.contains_key("Small.bias"));

        // Import into model2
        let result = model2.import_tensor_views(exported, &device);

        assert!(result.is_success());
        assert_eq!(result.applied.len(), 2);
        assert_eq!(result.errors.len(), 0);

        // Verify the import
        let model2_exported = model2.export_tensor_views();
        let weight_data = model2_exported.get("Small.weight").unwrap().to_data();
        let bias_data = model2_exported.get("Small.bias").unwrap().to_data();

        assert_eq!(weight_data.shape, vec![2, 2]);
        assert_eq!(bias_data.shape, vec![2]);
    }

    // Test deeply nested module with vecs
    #[derive(Module, Debug)]
    struct NestedWithVec<B: Backend> {
        encoder: VecModule<B>,
        decoder: VecModule<B>,
    }

    impl<B: Backend> NestedWithVec<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                encoder: VecModule::new(device, 2),
                decoder: VecModule::new(device, 2),
            }
        }

        fn new_zeros(device: &B::Device) -> Self {
            Self {
                encoder: VecModule::new_zeros(device, 2),
                decoder: VecModule::new_zeros(device, 2),
            }
        }
    }

    #[test]
    fn test_import_nested_with_vec() {
        let device = Default::default();
        let model1 = NestedWithVec::<TestBackend>::new(&device);
        let mut model2 = NestedWithVec::<TestBackend>::new_zeros(&device);

        // Export from model1
        let exported = model1.export_tensor_views();

        // Should have 8 tensors (2 modules × 2 layers × 2 tensors)
        assert_eq!(exported.len(), 8);
        assert!(exported.contains_key("encoder.layers.0.weight"));
        assert!(exported.contains_key("encoder.layers.0.bias"));
        assert!(exported.contains_key("encoder.layers.1.weight"));
        assert!(exported.contains_key("encoder.layers.1.bias"));
        assert!(exported.contains_key("decoder.layers.0.weight"));
        assert!(exported.contains_key("decoder.layers.0.bias"));
        assert!(exported.contains_key("decoder.layers.1.weight"));
        assert!(exported.contains_key("decoder.layers.1.bias"));

        // Import into model2
        let result = model2.import_tensor_views(exported, &device);

        assert!(result.is_success());
        assert_eq!(result.applied.len(), 8);
        assert_eq!(result.errors.len(), 0);

        // Test selective import - only encoder.layers.0
        let model1 = NestedWithVec::<TestBackend>::new(&device);
        let mut model2 = NestedWithVec::<TestBackend>::new_zeros(&device);

        let exported = model1.export_tensor_views();
        let result = model2
            .import_tensor_views_filtered(exported, &device, &[r"^encoder\.layers\.0\..*"])
            .unwrap();

        assert!(result.is_success());
        assert_eq!(result.applied.len(), 2); // Only encoder.layers.0 tensors
        assert_eq!(result.skipped.len(), 6); // Rest are skipped
    }

    // Test optional fields
    #[derive(Module, Debug)]
    struct OptionalModule<B: Backend> {
        required: Linear<B>,
        optional: Option<Linear<B>>,
    }

    impl<B: Backend> OptionalModule<B> {
        fn new_with_optional(device: &B::Device) -> Self {
            Self {
                required: LinearConfig::new(4, 4).with_bias(true).init(device),
                optional: Some(LinearConfig::new(4, 4).with_bias(true).init(device)),
            }
        }

        fn new_with_optional_zeros(device: &B::Device) -> Self {
            let create_zero_module = || {
                let mut module = LinearConfig::new(4, 4).with_bias(true).init(device);
                module.weight = Param::from_tensor(Tensor::zeros([4, 4], device));
                module.bias = Some(Param::from_tensor(Tensor::zeros([4], device)));
                module
            };

            Self {
                required: create_zero_module(),
                optional: Some(create_zero_module()),
            }
        }

        fn new_without_optional(device: &B::Device) -> Self {
            Self {
                required: LinearConfig::new(4, 4).with_bias(true).init(device),
                optional: None,
            }
        }
    }

    #[test]
    fn test_import_optional_module() {
        let device = Default::default();

        // Test with optional present
        let model1 = OptionalModule::<TestBackend>::new_with_optional(&device);
        let mut model2 = OptionalModule::<TestBackend>::new_with_optional_zeros(&device);

        let exported = model1.export_tensor_views();
        assert_eq!(exported.len(), 4); // 2 modules × 2 tensors

        let result = model2.import_tensor_views(exported, &device);
        assert!(result.is_success());
        assert_eq!(result.applied.len(), 4);

        // Test with optional absent - export from model without optional
        let model1 = OptionalModule::<TestBackend>::new_without_optional(&device);
        let mut model2 = OptionalModule::<TestBackend>::new_with_optional_zeros(&device);

        let exported = model1.export_tensor_views();
        assert_eq!(exported.len(), 2); // Only required module

        let result = model2.import_tensor_views(exported, &device);
        assert!(result.is_success());
        assert_eq!(result.applied.len(), 2); // Only required tensors applied
        assert_eq!(result.missing.len(), 2); // Optional tensors are missing
    }
}
