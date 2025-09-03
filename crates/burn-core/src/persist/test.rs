#[cfg(all(test, feature = "std"))]
mod tests {
    use super::super::*;
    use crate as burn;
    use crate::{
        TestBackend,
        module::{Module, Param},
        nn::{self, Linear, LinearConfig},
    };
    use burn_tensor::{Tensor, backend::Backend};
    use hashbrown::HashMap;

    #[derive(Module, Debug)]
    struct TestModule<B: Backend> {
        encoder: TestSubModule<B>,
        decoder: TestSubModule<B>,
    }

    #[derive(Module, Debug)]
    struct TestSubModule<B: Backend> {
        weight: Param<Tensor<B, 2>>,
        bias: Param<Tensor<B, 1>>,
    }

    impl<B: Backend> TestModule<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                encoder: TestSubModule::new(device),
                decoder: TestSubModule::new(device),
            }
        }
    }

    impl<B: Backend> TestSubModule<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                weight: Param::from_data([[1.0, 2.0], [3.0, 4.0]], device),
                bias: Param::from_data([5.0, 6.0], device),
            }
        }
    }

    #[test]
    fn test_collect() {
        let device = Default::default();
        let module = TestModule::<TestBackend>::new(&device);

        let views = module.collect();

        assert_eq!(views.len(), 4);
        assert!(views.contains_key("encoder.weight"));
        assert!(views.contains_key("encoder.bias"));
        assert!(views.contains_key("decoder.weight"));
        assert!(views.contains_key("decoder.bias"));
    }

    #[test]
    #[cfg(target_has_atomic = "ptr")]
    fn test_collect_filtered() {
        let device = Default::default();
        let module = TestModule::<TestBackend>::new(&device);

        let filter = PathFilter::new().with_regex(r"^encoder\..*");
        let views = module.collect_with_filter(filter);

        assert_eq!(views.len(), 2);
        assert!(views.contains_key("encoder.weight"));
        assert!(views.contains_key("encoder.bias"));
        assert!(!views.contains_key("decoder.weight"));
        assert!(!views.contains_key("decoder.bias"));
    }

    #[test]
    #[cfg(target_has_atomic = "ptr")]
    fn test_collect_filtered_multiple_patterns() {
        let device = Default::default();
        let module = TestModule::<TestBackend>::new(&device);

        // Collect tensors matching either pattern (OR union)
        let filter = PathFilter::new()
            .with_regex(r"^encoder\.weight$") // Only encoder.weight
            .with_regex(r".*\.bias$"); // Any .bias tensor
        let views = module.collect_with_filter(filter);

        assert_eq!(views.len(), 3);
        assert!(views.contains_key("encoder.weight")); // matches first pattern
        assert!(views.contains_key("encoder.bias")); // matches second pattern
        assert!(views.contains_key("decoder.bias")); // matches second pattern
        assert!(!views.contains_key("decoder.weight")); // matches neither
    }

    #[test]
    fn test_with_linear_module() {
        let device = Default::default();
        let linear = LinearConfig::new(10, 20)
            .with_bias(true)
            .init::<TestBackend>(&device);

        let views = linear.collect();
        assert!(views.contains_key("weight"));
        assert!(views.contains_key("bias"));

        // Verify the views contain correct data
        let weight_view = views.get("weight").unwrap();
        let weight_data = weight_view.to_data();
        assert_eq!(weight_data.shape, vec![10, 20]);

        let bias_view = views.get("bias").unwrap();
        let bias_data = bias_view.to_data();
        assert_eq!(bias_data.shape, vec![20]);
    }

    // Deep nesting test (5 levels deep)
    #[derive(Module, Debug)]
    struct DeepNestedModule<B: Backend> {
        level1: Level1<B>,
    }

    #[derive(Module, Debug)]
    struct Level1<B: Backend> {
        level2_a: Level2<B>,
        level2_b: Level2<B>,
    }

    #[derive(Module, Debug)]
    struct Level2<B: Backend> {
        level3: Level3<B>,
    }

    #[derive(Module, Debug)]
    struct Level3<B: Backend> {
        level4_main: Level4<B>,
        level4_aux: Level4<B>,
    }

    #[derive(Module, Debug)]
    struct Level4<B: Backend> {
        conv: Param<Tensor<B, 4>>,
        norm: Param<Tensor<B, 1>>,
    }

    impl<B: Backend> DeepNestedModule<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                level1: Level1::new(device),
            }
        }
    }

    impl<B: Backend> Level1<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                level2_a: Level2::new(device),
                level2_b: Level2::new(device),
            }
        }
    }

    impl<B: Backend> Level2<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                level3: Level3::new(device),
            }
        }
    }

    impl<B: Backend> Level3<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                level4_main: Level4::new(device),
                level4_aux: Level4::new(device),
            }
        }
    }

    impl<B: Backend> Level4<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                conv: Param::from_data([[[[1.0]]]], device),
                norm: Param::from_data([0.5], device),
            }
        }
    }

    #[test]
    fn test_deep_nested_collect() {
        let device = Default::default();
        let model = DeepNestedModule::<TestBackend>::new(&device);

        // Test collecting all tensors from deeply nested structure
        let all_views = model.collect();

        // Should have 8 tensors total (2 tensors × 2 level4 modules × 2 level2 branches)
        assert_eq!(all_views.len(), 8);

        // Verify deep paths exist
        assert!(all_views.contains_key("level1.level2_a.level3.level4_main.conv"));
        assert!(all_views.contains_key("level1.level2_a.level3.level4_main.norm"));
        assert!(all_views.contains_key("level1.level2_a.level3.level4_aux.conv"));
        assert!(all_views.contains_key("level1.level2_a.level3.level4_aux.norm"));
        assert!(all_views.contains_key("level1.level2_b.level3.level4_main.conv"));
        assert!(all_views.contains_key("level1.level2_b.level3.level4_main.norm"));
        assert!(all_views.contains_key("level1.level2_b.level3.level4_aux.conv"));
        assert!(all_views.contains_key("level1.level2_b.level3.level4_aux.norm"));
    }

    #[test]
    #[cfg(target_has_atomic = "ptr")]
    fn test_deep_nested_filtered_collect() {
        let device = Default::default();
        let model = DeepNestedModule::<TestBackend>::new(&device);

        // Filter only level2_a branch
        let filter = PathFilter::new().with_regex(r"^level1\.level2_a\..*");
        let level2_a_views = model.collect_with_filter(filter);
        assert_eq!(level2_a_views.len(), 4);

        // Filter only main modules at any depth
        let filter = PathFilter::new().with_regex(r".*\.level4_main\..*");
        let main_views = model.collect_with_filter(filter);
        assert_eq!(main_views.len(), 4);

        // Filter only conv tensors
        let filter = PathFilter::new().with_regex(r".*\.conv$");
        let conv_views = model.collect_with_filter(filter);
        assert_eq!(conv_views.len(), 4);

        // Complex multi-pattern filter
        let filter = PathFilter::new()
            .with_regex(r"^level1\.level2_a\..*\.norm$") // All norms in level2_a
            .with_regex(r"^level1\.level2_b\..*\.conv$"); // All convs in level2_b
        let complex_views = model.collect_with_filter(filter);
        assert_eq!(complex_views.len(), 4);
        assert!(complex_views.contains_key("level1.level2_a.level3.level4_main.norm"));
        assert!(complex_views.contains_key("level1.level2_a.level3.level4_aux.norm"));
        assert!(complex_views.contains_key("level1.level2_b.level3.level4_main.conv"));
        assert!(complex_views.contains_key("level1.level2_b.level3.level4_aux.conv"));
    }

    // Custom layer for testing various module configurations
    #[derive(Module, Debug)]
    struct CustomLayer<B: Backend> {
        weight: Param<Tensor<B, 2>>,
        scale: Param<Tensor<B, 1>>,
    }

    impl<B: Backend> CustomLayer<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                weight: Param::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device),
                scale: Param::from_data([0.1, 0.2, 0.3], device),
            }
        }
    }

    #[test]
    fn test_custom_layer_collect() {
        let device = Default::default();
        let custom = CustomLayer::<TestBackend>::new(&device);

        let views = custom.collect();

        assert_eq!(views.len(), 2);
        assert!(views.contains_key("weight"));
        assert!(views.contains_key("scale"));

        // Verify tensor shapes
        let weight_data = views.get("weight").unwrap().to_data();
        assert_eq!(weight_data.shape, vec![2, 3]);

        let scale_data = views.get("scale").unwrap().to_data();
        assert_eq!(scale_data.shape, vec![3]);
    }

    #[test]
    fn test_composite_module_collect() {
        let device = Default::default();

        // Create a composite module with multiple custom layers
        #[derive(Module, Debug)]
        struct CompositeModule<B: Backend> {
            layer1: CustomLayer<B>,
            layer2: CustomLayer<B>,
        }

        impl<B: Backend> CompositeModule<B> {
            fn new(device: &B::Device) -> Self {
                Self {
                    layer1: CustomLayer::new(device),
                    layer2: CustomLayer::new(device),
                }
            }
        }

        let composite = CompositeModule::<TestBackend>::new(&device);
        let views = composite.collect();

        assert_eq!(views.len(), 4);
        assert!(views.contains_key("layer1.weight"));
        assert!(views.contains_key("layer1.scale"));
        assert!(views.contains_key("layer2.weight"));
        assert!(views.contains_key("layer2.scale"));

        // Test filtering
        #[cfg(target_has_atomic = "ptr")]
        {
            let filter = PathFilter::new().with_regex(r".*\.weight$");
            let weight_only = composite.collect_with_filter(filter);
            assert_eq!(weight_only.len(), 2);
            assert!(weight_only.contains_key("layer1.weight"));
            assert!(weight_only.contains_key("layer2.weight"));
        }
    }

    // Test module with Option fields
    #[derive(Module, Debug)]
    struct OptionalFieldModule<B: Backend> {
        required: Param<Tensor<B, 2>>,
        optional: Option<Param<Tensor<B, 1>>>,
    }

    impl<B: Backend> OptionalFieldModule<B> {
        fn new_with_optional(device: &B::Device) -> Self {
            Self {
                required: Param::from_data([[1.0, 2.0], [3.0, 4.0]], device),
                optional: Some(Param::from_data([5.0, 6.0], device)),
            }
        }

        fn new_without_optional(device: &B::Device) -> Self {
            Self {
                required: Param::from_data([[1.0, 2.0], [3.0, 4.0]], device),
                optional: None,
            }
        }
    }

    #[test]
    fn test_optional_field_module_with_value() {
        let device = Default::default();
        let module = OptionalFieldModule::<TestBackend>::new_with_optional(&device);

        let views = module.collect();

        assert_eq!(views.len(), 2);
        assert!(views.contains_key("required"));
        assert!(views.contains_key("optional"));
    }

    #[test]
    fn test_optional_field_module_without_value() {
        let device = Default::default();
        let module = OptionalFieldModule::<TestBackend>::new_without_optional(&device);

        let views = module.collect();

        assert_eq!(views.len(), 1);
        assert!(views.contains_key("required"));
        assert!(!views.contains_key("optional"));
    }

    // Test Vec of modules
    #[derive(Module, Debug)]
    struct VecModule<B: Backend> {
        layers: Vec<CustomLayer<B>>,
    }

    impl<B: Backend> VecModule<B> {
        fn new(device: &B::Device, num_layers: usize) -> Self {
            Self {
                layers: (0..num_layers).map(|_| CustomLayer::new(device)).collect(),
            }
        }
    }

    #[test]
    fn test_vec_module_collect() {
        let device = Default::default();
        let module = VecModule::<TestBackend>::new(&device, 3);

        let views = module.collect();

        // With the fix, all Vec items should now be properly indexed and visited
        assert_eq!(views.len(), 6); // 3 layers × 2 tensors each = 6 tensors

        // Check that all indexed paths exist
        assert!(views.contains_key("layers.0.weight"));
        assert!(views.contains_key("layers.0.scale"));
        assert!(views.contains_key("layers.1.weight"));
        assert!(views.contains_key("layers.1.scale"));
        assert!(views.contains_key("layers.2.weight"));
        assert!(views.contains_key("layers.2.scale"));

        // Verify the data from all tensors
        for i in 0..3 {
            let weight_path = format!("layers.{}.weight", i);
            let scale_path = format!("layers.{}.scale", i);

            let weight_data = views.get(&weight_path).unwrap().to_data();
            assert_eq!(weight_data.shape, vec![2, 3]);

            let scale_data = views.get(&scale_path).unwrap().to_data();
            assert_eq!(scale_data.shape, vec![3]);
        }

        // Test filtering for specific layer
        #[cfg(target_has_atomic = "ptr")]
        {
            let filter = PathFilter::new().with_regex(r"^layers\.1\..*");
            let layer1_only = module.collect_with_filter(filter);
            assert_eq!(layer1_only.len(), 2);
            assert!(layer1_only.contains_key("layers.1.weight"));
            assert!(layer1_only.contains_key("layers.1.scale"));
        }

        // Test filtering for all weights
        #[cfg(target_has_atomic = "ptr")]
        {
            let filter = PathFilter::new().with_regex(r".*\.weight$");
            let weights_only = module.collect_with_filter(filter);
            assert_eq!(weights_only.len(), 3);
            assert!(weights_only.contains_key("layers.0.weight"));
            assert!(weights_only.contains_key("layers.1.weight"));
            assert!(weights_only.contains_key("layers.2.weight"));
        }
    }

    // Test array of modules
    #[derive(Module, Debug)]
    struct ArrayModule<B: Backend> {
        layers: [CustomLayer<B>; 3],
    }

    impl<B: Backend> ArrayModule<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                layers: [
                    CustomLayer::new(device),
                    CustomLayer::new(device),
                    CustomLayer::new(device),
                ],
            }
        }
    }

    #[test]
    fn test_array_module_collect() {
        let device = Default::default();
        let module = ArrayModule::<TestBackend>::new(&device);

        let views = module.collect();

        // All array items should be properly indexed
        assert_eq!(views.len(), 6); // 3 layers × 2 tensors each = 6 tensors

        // Check indexed paths
        for i in 0..3 {
            assert!(views.contains_key(&format!("layers.{}.weight", i)));
            assert!(views.contains_key(&format!("layers.{}.scale", i)));
        }

        // Test filtering for specific index
        #[cfg(target_has_atomic = "ptr")]
        {
            let filter = PathFilter::new().with_regex(r"^layers\.2\..*");
            let layer2_only = module.collect_with_filter(filter);
            assert_eq!(layer2_only.len(), 2);
            assert!(layer2_only.contains_key("layers.2.weight"));
            assert!(layer2_only.contains_key("layers.2.scale"));
        }
    }

    #[test]
    fn test_collect_with_predicate() {
        let device = Default::default();
        let module = TestModule::<TestBackend>::new(&device);

        // Test with simple predicate - only encoder tensors
        fn starts_with_encoder(path: &str, _container_path: &str) -> bool {
            path.starts_with("encoder.")
        }
        let filter = PathFilter::new().with_predicate(starts_with_encoder);
        let encoder_only = module.collect_with_filter(filter);
        assert_eq!(encoder_only.len(), 2);
        assert!(encoder_only.contains_key("encoder.weight"));
        assert!(encoder_only.contains_key("encoder.bias"));

        // Test with exact match predicate
        fn specific_names(path: &str, _container_path: &str) -> bool {
            path == "encoder.weight" || path == "decoder.bias"
        }
        let filter = PathFilter::new().with_predicate(specific_names);
        let specific = module.collect_with_filter(filter);
        assert_eq!(specific.len(), 2);
        assert!(specific.contains_key("encoder.weight"));
        assert!(specific.contains_key("decoder.bias"));

        // Test with complex logic predicate
        fn complex_filter(path: &str, _container_path: &str) -> bool {
            let parts: Vec<&str> = path.split('.').collect();
            // All weights OR all encoder tensors
            (parts.len() == 2 && parts[1] == "weight") || parts[0] == "encoder"
        }
        let filter = PathFilter::new().with_predicate(complex_filter);
        let complex = module.collect_with_filter(filter);
        assert_eq!(complex.len(), 3);
        assert!(complex.contains_key("encoder.weight"));
        assert!(complex.contains_key("encoder.bias"));
        assert!(complex.contains_key("decoder.weight"));
    }

    #[test]
    fn test_predicate_with_deep_module() {
        let device = Default::default();
        let model = DeepNestedModule::<TestBackend>::new(&device);

        // Filter with predicate - only level2_a tensors
        fn contains_level2_a(path: &str, _container_path: &str) -> bool {
            path.contains("level2_a")
        }
        let filter = PathFilter::new().with_predicate(contains_level2_a);
        let level2_a_only = model.collect_with_filter(filter);
        assert_eq!(level2_a_only.len(), 4);

        // Filter by depth - only tensors at exactly 5 levels deep
        fn exactly_5_deep(path: &str, _container_path: &str) -> bool {
            path.split('.').count() == 5
        }
        let filter = PathFilter::new().with_predicate(exactly_5_deep);
        let deep_only = model.collect_with_filter(filter);
        assert_eq!(deep_only.len(), 8); // All conv and norm tensors are 5 levels deep

        // Complex predicate with multiple conditions
        fn main_conv_filter(path: &str, _container_path: &str) -> bool {
            let is_main = path.contains("level4_main");
            let is_conv = path.ends_with(".conv");
            is_main && is_conv
        }
        let filter = PathFilter::new().with_predicate(main_conv_filter);
        let complex = model.collect_with_filter(filter);
        assert_eq!(complex.len(), 2);
        assert!(complex.contains_key("level1.level2_a.level3.level4_main.conv"));
        assert!(complex.contains_key("level1.level2_b.level3.level4_main.conv"));
    }

    #[test]
    fn test_predicate_with_vec_module() {
        let device = Default::default();
        let module = VecModule::<TestBackend>::new(&device, 4);

        // Filter with predicate - even indices only
        fn even_indices(path: &str, _container_path: &str) -> bool {
            if let Some(captures) = path.split('.').nth(1) {
                if let Ok(index) = captures.parse::<usize>() {
                    return index % 2 == 0;
                }
            }
            false
        }
        let filter = PathFilter::new().with_predicate(even_indices);
        let even_only = module.collect_with_filter(filter);
        assert_eq!(even_only.len(), 4); // layers.0 and layers.2, each with 2 tensors

        // Filter for specific range of indices
        fn range_filter(path: &str, _container_path: &str) -> bool {
            if let Some(captures) = path.split('.').nth(1) {
                if let Ok(index) = captures.parse::<usize>() {
                    return index >= 1 && index <= 2;
                }
            }
            false
        }
        let filter = PathFilter::new().with_predicate(range_filter);
        let range = module.collect_with_filter(filter);
        assert_eq!(range.len(), 4); // layers.1 and layers.2, each with 2 tensors
    }

    #[test]
    fn test_predicate_with_exact_paths() {
        let device = Default::default();
        let module = TestModule::<TestBackend>::new(&device);

        // Use PathFilter with exact paths instead of closure
        let filter = PathFilter::new()
            .with_full_path("encoder.weight")
            .with_full_path("encoder.bias");

        let filtered = module.collect_with_filter(filter);

        assert_eq!(filtered.len(), 2);
        assert!(filtered.contains_key("encoder.weight"));
        assert!(filtered.contains_key("encoder.bias"));
    }

    // Test enum modules - they ARE supported by Burn
    // Note: Burn requires all enum variants to have the same field type
    #[derive(Module, Debug)]
    enum EnumModule<B: Backend> {
        LayerA(CustomLayer<B>),
        LayerB(CustomLayer<B>),
        LayerC(CustomLayer<B>),
    }

    #[test]
    fn test_enum_module_collect() {
        let device = Default::default();

        // Test variant A
        let module_a = EnumModule::<TestBackend>::LayerA(CustomLayer::new(&device));
        let views_a = module_a.collect();

        // Should have the variant name in the path
        assert_eq!(views_a.len(), 2);
        assert!(views_a.contains_key("LayerA.weight"));
        assert!(views_a.contains_key("LayerA.scale"));

        // Test variant B
        let module_b = EnumModule::<TestBackend>::LayerB(CustomLayer::new(&device));
        let views_b = module_b.collect();

        assert_eq!(views_b.len(), 2);
        assert!(views_b.contains_key("LayerB.weight"));
        assert!(views_b.contains_key("LayerB.scale"));

        // Test variant C
        let module_c = EnumModule::<TestBackend>::LayerC(CustomLayer::new(&device));
        let views_c = module_c.collect();

        assert_eq!(views_c.len(), 2);
        assert!(views_c.contains_key("LayerC.weight"));
        assert!(views_c.contains_key("LayerC.scale"));

        // Verify tensor data
        let weight_data = views_a.get("LayerA.weight").unwrap().to_data();
        assert_eq!(weight_data.shape, vec![2, 3]);

        // Test filtering on enum module
        #[cfg(target_has_atomic = "ptr")]
        {
            let filter = PathFilter::new().with_regex(r".*\.scale$");
            let scale_only = module_a.collect_with_filter(filter);
            assert_eq!(scale_only.len(), 1);
            assert!(scale_only.contains_key("LayerA.scale"));
        }
    }

    // Test enum module with built-in module types
    #[derive(Module, Debug)]
    enum NetworkVariant<B: Backend> {
        Small(nn::Linear<B>),
        Medium(nn::Linear<B>),
        Large(nn::Linear<B>),
    }

    #[test]
    fn test_enum_with_linear_modules() {
        let device = Default::default();

        // Create different sized linear layers
        let small_linear = LinearConfig::new(10, 20)
            .with_bias(true)
            .init::<TestBackend>(&device);
        let medium_linear = LinearConfig::new(20, 50)
            .with_bias(true)
            .init::<TestBackend>(&device);
        let large_linear = LinearConfig::new(50, 100)
            .with_bias(true)
            .init::<TestBackend>(&device);

        // Test small variant
        let small_net = NetworkVariant::Small(small_linear);
        let small_views = small_net.collect();

        assert_eq!(small_views.len(), 2);
        assert!(small_views.contains_key("Small.weight"));
        assert!(small_views.contains_key("Small.bias"));

        // Verify shapes
        let weight_data = small_views.get("Small.weight").unwrap().to_data();
        assert_eq!(weight_data.shape, vec![10, 20]);

        // Test medium variant
        let medium_net = NetworkVariant::Medium(medium_linear);
        let medium_views = medium_net.collect();

        assert_eq!(medium_views.len(), 2);
        assert!(medium_views.contains_key("Medium.weight"));
        assert!(medium_views.contains_key("Medium.bias"));

        let weight_data = medium_views.get("Medium.weight").unwrap().to_data();
        assert_eq!(weight_data.shape, vec![20, 50]);

        // Test large variant
        let large_net = NetworkVariant::Large(large_linear);
        let large_views = large_net.collect();

        assert_eq!(large_views.len(), 2);
        assert!(large_views.contains_key("Large.weight"));
        assert!(large_views.contains_key("Large.bias"));

        let weight_data = large_views.get("Large.weight").unwrap().to_data();
        assert_eq!(weight_data.shape, vec![50, 100]);
    }

    // Container type tracking tests

    // Test module with Linear layer
    #[derive(Module, Debug)]
    struct ModelWithLinear<B: Backend> {
        linear: Linear<B>,
    }

    impl<B: Backend> ModelWithLinear<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                linear: LinearConfig::new(10, 20).init(device),
            }
        }
    }

    #[test]
    fn test_linear_container_type() {
        let device = Default::default();
        let model = ModelWithLinear::<TestBackend>::new(&device);

        let views = model.collect();

        // Check that tensors inside Linear layers have "Linear" as their container type
        // because they are directly contained by the Linear module, not ModelWithLinear
        for (path, view) in views.iter() {
            if path == "linear.weight" || path == "linear.bias" {
                assert_eq!(
                    view.container_type(),
                    "Linear",
                    "Tensor '{}' should have container type 'Linear' (its immediate parent)",
                    path
                );
            }
        }
    }

    // Test module with Vec of modules
    #[derive(Module, Debug)]
    struct ModelWithVec<B: Backend> {
        layers: Vec<Linear<B>>,
    }

    impl<B: Backend> ModelWithVec<B> {
        fn new(device: &B::Device, num_layers: usize) -> Self {
            Self {
                layers: (0..num_layers)
                    .map(|_| LinearConfig::new(10, 10).init(device))
                    .collect(),
            }
        }
    }

    #[test]
    fn test_vec_container_type() {
        let device = Default::default();
        let model = ModelWithVec::<TestBackend>::new(&device, 3);

        let views = model.collect();

        // Check that modules in Vec have "Vec" as their immediate container type
        for (path, view) in views.iter() {
            if path.starts_with("layers.0")
                || path.starts_with("layers.1")
                || path.starts_with("layers.2")
            {
                // The immediate parent of layers.0, layers.1, etc. should be Vec
                // But the actual tensor (layers.0.weight) would have Linear as container
                if path.contains(".weight") || path.contains(".bias") {
                    assert_eq!(
                        view.container_type(),
                        "Linear",
                        "Tensor '{}' should have container type 'Linear'",
                        path
                    );
                }
            }
        }
    }

    // Test module with Array of modules
    #[derive(Module, Debug)]
    struct ModelWithArray<B: Backend> {
        layers: [Linear<B>; 2],
    }

    impl<B: Backend> ModelWithArray<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                layers: [
                    LinearConfig::new(10, 10).init(device),
                    LinearConfig::new(10, 10).init(device),
                ],
            }
        }
    }

    #[test]
    fn test_array_container_type() {
        let device = Default::default();
        let model = ModelWithArray::<TestBackend>::new(&device);

        let views = model.collect();

        // Check that modules in Array have proper container types
        for (path, view) in views.iter() {
            if path.starts_with("layers.0") || path.starts_with("layers.1") {
                if path.contains(".weight") || path.contains(".bias") {
                    assert_eq!(
                        view.container_type(),
                        "Linear",
                        "Tensor '{}' should have container type 'Linear'",
                        path
                    );
                }
            }
        }
    }

    // Test nested module structure
    #[derive(Module, Debug)]
    struct InnerModule<B: Backend> {
        linear: Linear<B>,
    }

    #[derive(Module, Debug)]
    struct OuterModule<B: Backend> {
        inner: InnerModule<B>,
        direct_linear: Linear<B>,
    }

    impl<B: Backend> InnerModule<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                linear: LinearConfig::new(5, 5).init(device),
            }
        }
    }

    impl<B: Backend> OuterModule<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                inner: InnerModule::new(device),
                direct_linear: LinearConfig::new(5, 5).init(device),
            }
        }
    }

    #[test]
    fn test_nested_container_types() {
        let device = Default::default();
        let model = OuterModule::<TestBackend>::new(&device);

        let views = model.collect();

        // Check nested container types
        for (path, view) in views.iter() {
            if path.starts_with("inner.linear") {
                assert_eq!(
                    view.container_type(),
                    "Linear",
                    "Nested linear at '{}' should have container type 'Linear'",
                    path
                );
            } else if path.starts_with("direct_linear") {
                assert_eq!(
                    view.container_type(),
                    "Linear",
                    "Direct linear at '{}' should have container type 'Linear'",
                    path
                );
            }
        }
    }

    // Test enum module
    #[derive(Module, Debug)]
    enum ModelEnum<B: Backend> {
        VariantA(Linear<B>),
        VariantB(Linear<B>),
    }

    #[test]
    fn test_enum_container_type() {
        let device = Default::default();
        let model = ModelEnum::VariantA(LinearConfig::new(10, 20).init::<TestBackend>(&device));

        let views = model.collect();

        // Check that enum variant modules have proper container types
        for (path, view) in views.iter() {
            if path.starts_with("VariantA") {
                assert_eq!(
                    view.container_type(),
                    "Linear",
                    "Enum variant tensor '{}' should have container type 'Linear'",
                    path
                );
            }
        }
    }

    // Test custom visitor that tracks container types
    struct ContainerTracker {
        containers: Vec<(String, String)>, // (name, container_type)
    }

    impl ContainerTracker {
        fn new() -> Self {
            Self {
                containers: Vec::new(),
            }
        }
    }

    impl<B: Backend> crate::module::ModuleVisitor<B> for ContainerTracker {
        fn enter_module(&mut self, name: &str, container_type: &str) {
            self.containers
                .push((name.to_string(), container_type.to_string()));
        }
    }

    #[test]
    fn test_visitor_receives_container_types() {
        let device = Default::default();
        let model = ModelWithLinear::<TestBackend>::new(&device);

        let mut tracker = ContainerTracker::new();
        model.visit(&mut tracker);

        // Verify that the visitor received container type information
        assert!(
            !tracker.containers.is_empty(),
            "Visitor should have received container information"
        );

        // The visitor should see "linear" field with "ModelWithLinear" as the container
        // And then nested fields inside Linear with "Linear" as the container
        let has_model_container = tracker
            .containers
            .iter()
            .any(|(name, container)| name == "linear" && container == "ModelWithLinear");
        assert!(
            has_model_container,
            "Should have ModelWithLinear container for linear field"
        );

        // There should also be entries for weight and bias with Linear as container
        let has_linear_weight = tracker
            .containers
            .iter()
            .any(|(name, container)| name == "weight" && container == "Linear");
        let has_linear_bias = tracker
            .containers
            .iter()
            .any(|(name, container)| name == "bias" && container == "Linear");

        assert!(
            has_linear_weight || has_linear_bias,
            "Should have Linear container for weight/bias fields inside Linear module"
        );
    }

    // Test mapper that uses container types
    struct ContainerMapper {
        processed: Vec<(String, String)>,
        path_stack: Vec<String>,
    }

    impl ContainerMapper {
        fn new() -> Self {
            Self {
                processed: Vec::new(),
                path_stack: Vec::new(),
            }
        }

        fn current_path(&self) -> String {
            self.path_stack.join(".")
        }
    }

    impl<B: Backend> crate::module::ModuleMapper<B> for ContainerMapper {
        fn enter_module(&mut self, name: &str, container_type: &str) {
            self.path_stack.push(name.to_string());
            self.processed
                .push((self.current_path(), container_type.to_string()));
        }

        fn exit_module(&mut self, _name: &str, _container_type: &str) {
            self.path_stack.pop();
        }
    }

    #[test]
    fn test_mapper_receives_container_types() {
        let device = Default::default();
        let model = ModelWithLinear::<TestBackend>::new(&device);

        let mut mapper = ContainerMapper::new();
        let _ = model.clone().map(&mut mapper);

        // Verify that the mapper received container type information
        assert!(
            !mapper.processed.is_empty(),
            "Mapper should have received container information"
        );

        // Check that we have the expected container type
        let has_model_container = mapper
            .processed
            .iter()
            .any(|(path, container)| path == "linear" && container == "ModelWithLinear");
        assert!(
            has_model_container,
            "Should have ModelWithLinear container for linear field"
        );
    }

    // Test complex nested structure with multiple container types
    #[derive(Module, Debug)]
    struct ComplexModel<B: Backend> {
        linear_layers: [Linear<B>; 2],
        vec_layers: Vec<Linear<B>>,
        single_linear: Linear<B>,
    }

    impl<B: Backend> ComplexModel<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                linear_layers: [
                    LinearConfig::new(100, 50).init(device),
                    LinearConfig::new(50, 10).init(device),
                ],
                vec_layers: vec![
                    LinearConfig::new(10, 10).init(device),
                    LinearConfig::new(10, 10).init(device),
                ],
                single_linear: LinearConfig::new(10, 1).init(device),
            }
        }
    }

    #[test]
    fn test_complex_model_container_types() {
        let device = Default::default();
        let model = ComplexModel::<TestBackend>::new(&device);

        let views = model.collect();

        // Verify different container types in the complex model
        let mut found_linear = false;

        for (_path, view) in views.iter() {
            match view.container_type().as_str() {
                "Linear" => found_linear = true,
                _ => {}
            }
        }

        assert!(found_linear, "Should have found Linear container type");

        // Print summary for debugging
        println!("Complex model has {} total tensor views", views.len());

        // Group by container type for analysis
        let mut by_container: HashMap<String, usize> = HashMap::new();
        for (_path, view) in views.iter() {
            *by_container.entry(view.container_type()).or_insert(0) += 1;
        }

        println!("Tensors by container type:");
        for (container, count) in by_container.iter() {
            println!("  {}: {} tensors", container, count);
        }
    }

    #[test]
    fn test_collect_with_container_filter() {
        let device = Default::default();
        let model = ComplexModel::<TestBackend>::new(&device);

        // Filter to only collect tensors from Linear modules
        let filter = PathFilter::new().with_predicate(|_path, container_path| {
            container_path.split('.').last() == Some("Linear")
        });

        let linear_views = model.collect_with_filter(filter);

        // All collected tensors should be from Linear modules
        for (_path, view) in linear_views.iter() {
            assert_eq!(
                view.container_type(),
                "Linear",
                "All tensors should be from Linear modules"
            );
        }

        // Should have collected all Linear tensors (10 total: 5 Linear modules × 2 tensors each)
        assert_eq!(
            linear_views.len(),
            10,
            "Should have 10 tensors from 5 Linear modules"
        );

        // Filter to only collect weights from Linear modules
        let weight_filter = PathFilter::new().with_predicate(|path, container_path| {
            container_path.split('.').last() == Some("Linear") && path.ends_with(".weight")
        });

        let weight_views = model.collect_with_filter(weight_filter);

        // Should have exactly 5 weight tensors
        assert_eq!(
            weight_views.len(),
            5,
            "Should have 5 weight tensors from 5 Linear modules"
        );

        for (path, view) in weight_views.iter() {
            assert!(path.ends_with(".weight"), "Should only have weight tensors");
            assert_eq!(
                view.container_type(),
                "Linear",
                "Should be from Linear modules"
            );
        }

        // Test using container_path() method
        for (_path, view) in linear_views.iter() {
            let container_path = view.container_path();
            // Container path should end with Linear (e.g., "ComplexModel.Linear" or "ComplexModel.Array.Linear")
            assert!(
                container_path.ends_with("Linear") || container_path.ends_with("Linear.Param"),
                "Container path {} should contain Linear",
                container_path
            );
        }

        // Filter using dot-notated container path
        let dot_filter = PathFilter::new().with_predicate(|_path, container_path| {
            // Match tensors that are in a Linear module directly under ComplexModel
            container_path.starts_with("ComplexModel.Linear") ||
            // Or in Linear modules inside arrays/vecs
            container_path.contains(".Linear")
        });

        let dot_filtered = model.collect_with_filter(dot_filter);
        assert_eq!(
            dot_filtered.len(),
            10,
            "Should match all Linear module tensors"
        );
    }
}
