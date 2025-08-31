use hashbrown::HashMap;

use super::{TensorView, TensorViewCollector};
use crate::module::Module;
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
}

// Blanket implementation for all modules recursively
impl<B: Backend, M: Module<B>> ModuleExport<B> for M {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate as burn;
    use crate::{
        TestBackend,
        module::{Module, Param},
        nn::LinearConfig,
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
}
