use alloc::string::String;
use hashbrown::HashMap;

#[cfg(target_has_atomic = "ptr")]
use regex;

use super::TensorViewCollector;
use crate::module::Module;
use crate::persist::TensorView;
use crate::tensor::backend::Backend;

/// Extension trait for modules that provides tensor export functionality.
///
/// This trait provides convenient methods to export tensor views from any Burn module
/// without immediately copying the tensor data. The actual data copy only happens
/// when you call `to_data()` on a `TensorView`.
///
/// # Examples
///
/// ```ignore
/// use burn::module::export::ModuleExport;
///
/// // Export all tensors
/// let all_views = model.export_tensor_views();
/// for (path, view) in all_views.iter() {
///     println!("{}: {:?}", path, view.to_data().shape);
/// }
///
/// // Export only encoder tensors
/// let encoder_views = model.export_tensor_views_filtered(&[r"^encoder\..*"])?;
///
/// // Export tensors matching multiple patterns (OR union)
/// let views = model.export_tensor_views_filtered(&[
///     r"^encoder\..*",     // All encoder tensors
///     r".*\.bias$",        // OR any bias tensors
///     r"^attention\..*",   // OR attention tensors
/// ])?;
/// ```
pub trait ModuleExport<B: Backend>: Module<B> {
    /// Export tensor views for inspection without copying data.
    ///
    /// Returns a HashMap where keys are the full module paths (e.g., "encoder.layer1.weight")
    /// and values are `TensorView` objects that can lazily materialize the tensor data.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let views = model.export_tensor_views();
    /// println!("Total tensors: {}", views.len());
    ///
    /// // Materialize specific tensor data when needed
    /// if let Some(weight_view) = views.get("encoder.weight") {
    ///     let data = weight_view.to_data();
    ///     println!("Encoder weight shape: {:?}", data.shape);
    /// }
    /// ```
    fn export_tensor_views(&self) -> HashMap<String, TensorView> {
        let mut collector = TensorViewCollector::new();
        self.visit(&mut collector);
        collector.tensors
    }

    /// Export filtered tensor views matching any of the regex patterns.
    ///
    /// Multiple patterns work as an OR union - a tensor is collected if it matches ANY pattern.
    /// This allows flexible filtering strategies for complex module hierarchies.
    ///
    /// # Arguments
    ///
    /// * `patterns` - An iterable of regex patterns. Can be a slice, Vec, or any IntoIterator.
    ///
    /// # Returns
    ///
    /// * `Ok(HashMap)` - Map of matching tensor paths to their views
    /// * `Err(regex::Error)` - If any pattern is invalid regex
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Single pattern - export only encoder tensors
    /// let encoder_tensors = model.export_tensor_views_filtered(&[
    ///     r"^encoder\..*"
    /// ])?;
    ///
    /// // Multiple patterns (OR union) - export encoder OR decoder tensors
    /// let tensors = model.export_tensor_views_filtered(&[
    ///     r"^encoder\..*",
    ///     r"^decoder\..*",
    /// ])?;
    ///
    /// // Export all weights and biases
    /// let params = model.export_tensor_views_filtered(&[
    ///     r".*\.weight$",
    ///     r".*\.bias$",
    /// ])?;
    ///
    /// // Complex filtering - specific layers and tensor types
    /// let filtered = model.export_tensor_views_filtered(&[
    ///     r"^model\.layer[0-2]\..*",        // layers 0, 1, 2
    ///     r"^attention\..*\.(query|key)$",  // attention Q and K
    ///     r"^head\..*",                     // all head tensors
    /// ])?;
    ///
    /// // Using with Vec for dynamic patterns
    /// let mut patterns = vec![r"^encoder\..*"];
    /// if include_decoder {
    ///     patterns.push(r"^decoder\..*");
    /// }
    /// let tensors = model.export_tensor_views_filtered(patterns)?;
    /// ```
    #[cfg(target_has_atomic = "ptr")]
    fn export_tensor_views_filtered<I, S>(
        &self,
        patterns: I,
    ) -> Result<HashMap<String, TensorView>, regex::Error>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut collector = TensorViewCollector::with_filter(patterns)?;
        self.visit(&mut collector);
        Ok(collector.tensors)
    }

    /// Export tensor views filtered by a custom predicate function.
    ///
    /// This method allows you to provide a custom function to filter which tensors
    /// are collected. The function receives the tensor path (e.g., "encoder.layer1.weight")
    /// and should return `true` to include the tensor or `false` to exclude it.
    ///
    /// # Arguments
    ///
    /// * `predicate` - A function that takes a tensor path (&str) and returns bool
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Export only tensors with specific names
    /// let tensors = model.export_tensor_views_with_predicate(|path| {
    ///     path == "encoder.weight" || path == "decoder.bias"
    /// });
    ///
    /// // Export tensors based on custom logic
    /// let large_tensors = model.export_tensor_views_with_predicate(|path| {
    ///     // Only export tensors from layers 3, 4, and 5
    ///     if let Some(captures) = regex::Regex::new(r"layer(\d+)")
    ///         .unwrap()
    ///         .captures(path)
    ///     {
    ///         if let Some(layer_num) = captures.get(1) {
    ///             if let Ok(num) = layer_num.as_str().parse::<u32>() {
    ///                 return num >= 3 && num <= 5;
    ///             }
    ///         }
    ///     }
    ///     false
    /// });
    ///
    /// // Export tensors that don't contain certain keywords
    /// let filtered = model.export_tensor_views_with_predicate(|path| {
    ///     !path.contains("dropout") && !path.contains("auxiliary")
    /// });
    ///
    /// // Combine multiple conditions
    /// let specific_tensors = model.export_tensor_views_with_predicate(|path| {
    ///     let is_encoder = path.starts_with("encoder.");
    ///     let is_weight = path.ends_with(".weight");
    ///     let not_attention = !path.contains("attention");
    ///     
    ///     is_encoder && is_weight && not_attention
    /// });
    ///
    /// // Use with closure capturing external state
    /// let allowed_prefixes = vec!["encoder", "decoder", "head"];
    /// let tensors = model.export_tensor_views_with_predicate(|path| {
    ///     allowed_prefixes.iter().any(|prefix| path.starts_with(prefix))
    /// });
    /// ```
    fn export_tensor_views_with_predicate<F>(&self, predicate: F) -> HashMap<String, TensorView>
    where
        F: Fn(&str) -> bool + 'static,
    {
        let mut collector = TensorViewCollector::with_predicate(predicate);
        self.visit(&mut collector);
        collector.tensors
    }
}

// Blanket implementation for all modules recursively
impl<B: Backend, M: Module<B>> ModuleExport<B> for M {}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate as burn;
    use crate::{
        TestBackend,
        module::{Module, Param},
        nn::{self, LinearConfig},
    };
    use burn_tensor::Tensor;

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
    fn test_export_tensor_views() {
        let device = Default::default();
        let module = TestModule::<TestBackend>::new(&device);

        let views = module.export_tensor_views();

        assert_eq!(views.len(), 4);
        assert!(views.contains_key("encoder.weight"));
        assert!(views.contains_key("encoder.bias"));
        assert!(views.contains_key("decoder.weight"));
        assert!(views.contains_key("decoder.bias"));
    }

    #[test]
    fn test_export_tensor_views_filtered() {
        let device = Default::default();
        let module = TestModule::<TestBackend>::new(&device);

        let views = module
            .export_tensor_views_filtered(&[r"^encoder\..*"])
            .unwrap();

        assert_eq!(views.len(), 2);
        assert!(views.contains_key("encoder.weight"));
        assert!(views.contains_key("encoder.bias"));
        assert!(!views.contains_key("decoder.weight"));
        assert!(!views.contains_key("decoder.bias"));
    }

    #[test]
    fn test_export_tensor_views_filtered_multiple_patterns() {
        let device = Default::default();
        let module = TestModule::<TestBackend>::new(&device);

        // Export tensors matching either pattern (OR union)
        let views = module
            .export_tensor_views_filtered(&[
                r"^encoder\.weight$", // Only encoder.weight
                r".*\.bias$",         // Any .bias tensor
            ])
            .unwrap();

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

        let views = linear.export_tensor_views();
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
    fn test_deep_nested_export() {
        let device = Default::default();
        let model = DeepNestedModule::<TestBackend>::new(&device);

        // Test exporting all tensors from deeply nested structure
        let all_views = model.export_tensor_views();

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
    fn test_deep_nested_filtered_export() {
        let device = Default::default();
        let model = DeepNestedModule::<TestBackend>::new(&device);

        // Filter only level2_a branch
        let level2_a_views = model
            .export_tensor_views_filtered(&[r"^level1\.level2_a\..*"])
            .unwrap();
        assert_eq!(level2_a_views.len(), 4);

        // Filter only main modules at any depth
        let main_views = model
            .export_tensor_views_filtered(&[r".*\.level4_main\..*"])
            .unwrap();
        assert_eq!(main_views.len(), 4);

        // Filter only conv tensors
        let conv_views = model.export_tensor_views_filtered(&[r".*\.conv$"]).unwrap();
        assert_eq!(conv_views.len(), 4);

        // Complex multi-pattern filter
        let complex_views = model
            .export_tensor_views_filtered(&[
                r"^level1\.level2_a\..*\.norm$", // All norms in level2_a
                r"^level1\.level2_b\..*\.conv$", // All convs in level2_b
            ])
            .unwrap();
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
    fn test_custom_layer_export() {
        let device = Default::default();
        let custom = CustomLayer::<TestBackend>::new(&device);

        let views = custom.export_tensor_views();

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
    fn test_composite_module_export() {
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
        let views = composite.export_tensor_views();

        assert_eq!(views.len(), 4);
        assert!(views.contains_key("layer1.weight"));
        assert!(views.contains_key("layer1.scale"));
        assert!(views.contains_key("layer2.weight"));
        assert!(views.contains_key("layer2.scale"));

        // Test filtering
        let weight_only = composite
            .export_tensor_views_filtered(&[r".*\.weight$"])
            .unwrap();
        assert_eq!(weight_only.len(), 2);
        assert!(weight_only.contains_key("layer1.weight"));
        assert!(weight_only.contains_key("layer2.weight"));
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

        let views = module.export_tensor_views();

        assert_eq!(views.len(), 2);
        assert!(views.contains_key("required"));
        assert!(views.contains_key("optional"));
    }

    #[test]
    fn test_optional_field_module_without_value() {
        let device = Default::default();
        let module = OptionalFieldModule::<TestBackend>::new_without_optional(&device);

        let views = module.export_tensor_views();

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
    fn test_vec_module_export() {
        let device = Default::default();
        let module = VecModule::<TestBackend>::new(&device, 3);

        let views = module.export_tensor_views();

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
        let layer1_only = module
            .export_tensor_views_filtered(&[r"^layers\.1\..*"])
            .unwrap();
        assert_eq!(layer1_only.len(), 2);
        assert!(layer1_only.contains_key("layers.1.weight"));
        assert!(layer1_only.contains_key("layers.1.scale"));

        // Test filtering for all weights
        let weights_only = module
            .export_tensor_views_filtered(&[r".*\.weight$"])
            .unwrap();
        assert_eq!(weights_only.len(), 3);
        assert!(weights_only.contains_key("layers.0.weight"));
        assert!(weights_only.contains_key("layers.1.weight"));
        assert!(weights_only.contains_key("layers.2.weight"));
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
    fn test_array_module_export() {
        let device = Default::default();
        let module = ArrayModule::<TestBackend>::new(&device);

        let views = module.export_tensor_views();

        // All array items should be properly indexed
        assert_eq!(views.len(), 6); // 3 layers × 2 tensors each = 6 tensors

        // Check indexed paths
        for i in 0..3 {
            assert!(views.contains_key(&format!("layers.{}.weight", i)));
            assert!(views.contains_key(&format!("layers.{}.scale", i)));
        }

        // Test filtering for specific index
        let layer2_only = module
            .export_tensor_views_filtered(&[r"^layers\.2\..*"])
            .unwrap();
        assert_eq!(layer2_only.len(), 2);
        assert!(layer2_only.contains_key("layers.2.weight"));
        assert!(layer2_only.contains_key("layers.2.scale"));
    }

    #[test]
    fn test_export_with_predicate() {
        let device = Default::default();
        let module = TestModule::<TestBackend>::new(&device);

        // Test with simple predicate - only encoder tensors
        let encoder_only =
            module.export_tensor_views_with_predicate(|path| path.starts_with("encoder."));
        assert_eq!(encoder_only.len(), 2);
        assert!(encoder_only.contains_key("encoder.weight"));
        assert!(encoder_only.contains_key("encoder.bias"));

        // Test with specific names predicate
        let specific = module.export_tensor_views_with_predicate(|path| {
            path == "encoder.weight" || path == "decoder.bias"
        });
        assert_eq!(specific.len(), 2);
        assert!(specific.contains_key("encoder.weight"));
        assert!(specific.contains_key("decoder.bias"));

        // Test with complex logic
        let complex = module.export_tensor_views_with_predicate(|path| {
            let parts: Vec<&str> = path.split('.').collect();
            // Only collect if it's a weight OR if it's in the encoder
            (parts.len() == 2 && parts[1] == "weight") || parts[0] == "encoder"
        });
        assert_eq!(complex.len(), 3);
        assert!(complex.contains_key("encoder.weight"));
        assert!(complex.contains_key("encoder.bias"));
        assert!(complex.contains_key("decoder.weight"));
    }

    #[test]
    fn test_predicate_with_deep_module() {
        let device = Default::default();
        let model = DeepNestedModule::<TestBackend>::new(&device);

        // Filter using predicate - only level2_a branch
        let level2_a_only =
            model.export_tensor_views_with_predicate(|path| path.contains("level2_a"));
        assert_eq!(level2_a_only.len(), 4);

        // Filter by depth - only tensors at exactly 5 levels deep
        let deep_only =
            model.export_tensor_views_with_predicate(|path| path.split('.').count() == 5);
        assert_eq!(deep_only.len(), 8); // All conv and norm tensors are 5 levels deep

        // Complex predicate with multiple conditions
        let complex = model.export_tensor_views_with_predicate(|path| {
            let is_main = path.contains("level4_main");
            let is_conv = path.ends_with(".conv");
            is_main && is_conv
        });
        assert_eq!(complex.len(), 2);
        assert!(complex.contains_key("level1.level2_a.level3.level4_main.conv"));
        assert!(complex.contains_key("level1.level2_b.level3.level4_main.conv"));
    }

    #[test]
    fn test_predicate_with_vec_module() {
        let device = Default::default();
        let module = VecModule::<TestBackend>::new(&device, 4);

        // Filter using predicate - only even indices
        let even_only = module.export_tensor_views_with_predicate(|path| {
            if let Some(captures) = path.split('.').nth(1) {
                if let Ok(index) = captures.parse::<usize>() {
                    return index % 2 == 0;
                }
            }
            false
        });
        assert_eq!(even_only.len(), 4); // layers.0 and layers.2, each with 2 tensors

        // Filter for specific range of indices
        let range = module.export_tensor_views_with_predicate(|path| {
            if let Some(captures) = path.split('.').nth(1) {
                if let Ok(index) = captures.parse::<usize>() {
                    return index >= 1 && index <= 2;
                }
            }
            false
        });
        assert_eq!(range.len(), 4); // layers.1 and layers.2, each with 2 tensors
    }

    #[test]
    fn test_predicate_with_closure_capturing_state() {
        let device = Default::default();
        let module = TestModule::<TestBackend>::new(&device);

        // Use closure that captures external state
        let allowed_modules = vec!["encoder"];
        let allowed_types = vec!["weight", "bias"];

        let filtered = module.export_tensor_views_with_predicate(move |path| {
            let parts: Vec<&str> = path.split('.').collect();
            if parts.len() != 2 {
                return false;
            }
            allowed_modules.contains(&parts[0]) && allowed_types.contains(&parts[1])
        });

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
    fn test_enum_module_export() {
        let device = Default::default();

        // Test variant A
        let module_a = EnumModule::<TestBackend>::LayerA(CustomLayer::new(&device));
        let views_a = module_a.export_tensor_views();

        // Should have the variant name in the path
        assert_eq!(views_a.len(), 2);
        assert!(views_a.contains_key("LayerA.weight"));
        assert!(views_a.contains_key("LayerA.scale"));

        // Test variant B
        let module_b = EnumModule::<TestBackend>::LayerB(CustomLayer::new(&device));
        let views_b = module_b.export_tensor_views();

        assert_eq!(views_b.len(), 2);
        assert!(views_b.contains_key("LayerB.weight"));
        assert!(views_b.contains_key("LayerB.scale"));

        // Test variant C
        let module_c = EnumModule::<TestBackend>::LayerC(CustomLayer::new(&device));
        let views_c = module_c.export_tensor_views();

        assert_eq!(views_c.len(), 2);
        assert!(views_c.contains_key("LayerC.weight"));
        assert!(views_c.contains_key("LayerC.scale"));

        // Verify tensor data
        let weight_data = views_a.get("LayerA.weight").unwrap().to_data();
        assert_eq!(weight_data.shape, vec![2, 3]);

        // Test filtering on enum module
        let scale_only = module_a
            .export_tensor_views_filtered(&[r".*\.scale$"])
            .unwrap();
        assert_eq!(scale_only.len(), 1);
        assert!(scale_only.contains_key("LayerA.scale"));
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
        let small_views = small_net.export_tensor_views();

        assert_eq!(small_views.len(), 2);
        assert!(small_views.contains_key("Small.weight"));
        assert!(small_views.contains_key("Small.bias"));

        // Verify shapes
        let weight_data = small_views.get("Small.weight").unwrap().to_data();
        assert_eq!(weight_data.shape, vec![10, 20]);

        // Test medium variant
        let medium_net = NetworkVariant::Medium(medium_linear);
        let medium_views = medium_net.export_tensor_views();

        assert_eq!(medium_views.len(), 2);
        assert!(medium_views.contains_key("Medium.weight"));
        assert!(medium_views.contains_key("Medium.bias"));

        let weight_data = medium_views.get("Medium.weight").unwrap().to_data();
        assert_eq!(weight_data.shape, vec![20, 50]);

        // Test large variant
        let large_net = NetworkVariant::Large(large_linear);
        let large_views = large_net.export_tensor_views();

        assert_eq!(large_views.len(), 2);
        assert!(large_views.contains_key("Large.weight"));
        assert!(large_views.contains_key("Large.bias"));

        let weight_data = large_views.get("Large.weight").unwrap().to_data();
        assert_eq!(weight_data.shape, vec![50, 100]);
    }
}
