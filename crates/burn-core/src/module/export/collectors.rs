use alloc::string::{String, ToString};
use alloc::vec::Vec;
use hashbrown::HashMap;

#[cfg(target_has_atomic = "ptr")]
use regex::Regex;

use burn_tensor::{Bool, Int, Tensor, backend::Backend};

use super::TensorView;
use crate::module::{ModuleVisitor, ParamId};

/// Collects tensor views from modules without copying data.
///
/// This collector traverses a module hierarchy and creates lightweight views
/// of tensors that can be materialized to `TensorData` on demand.
///
/// # Examples
///
/// ## Collect all tensors
/// ```ignore
/// let collector = TensorViewCollector::new();
/// module.visit(&mut collector);
/// let all_tensors = collector.tensors;
/// ```
///
/// ## Filter with single pattern
/// ```ignore
/// let collector = TensorViewCollector::with_filter(&[r"^encoder\..*"]).unwrap();
/// module.visit(&mut collector);
/// // Only collects tensors starting with "encoder."
/// ```
///
/// ## Filter with multiple patterns (OR union)
/// ```ignore
/// let collector = TensorViewCollector::with_filter(&[
///     r"^encoder\..*",  // Match all encoder tensors
///     r".*\.bias$",     // OR match any bias tensors
/// ]).unwrap();
/// module.visit(&mut collector);
/// // Collects tensors matching ANY of the patterns
/// ```
pub struct TensorViewCollector {
    /// Map of tensor paths to their views
    pub tensors: HashMap<String, TensorView>,
    path_stack: Vec<String>,
    #[cfg(target_has_atomic = "ptr")]
    filters: Vec<Regex>,
}

impl Default for TensorViewCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl TensorViewCollector {
    /// Create a new tensor view collector that collects all tensors.
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
            path_stack: Vec::new(),
            #[cfg(target_has_atomic = "ptr")]
            filters: Vec::new(),
        }
    }

    /// Create a new tensor view collector that only collects tensors matching any of the regex patterns.
    ///
    /// Multiple patterns work as an OR union - a tensor is collected if it matches ANY pattern.
    ///
    /// # Arguments
    ///
    /// * `patterns` - An iterable of regex patterns. Can be a slice, Vec, or any IntoIterator.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Single pattern
    /// let collector = TensorViewCollector::with_filter(&[r"^model\.layer1\..*"])?;
    ///
    /// // Multiple patterns (OR union)
    /// let collector = TensorViewCollector::with_filter(&[
    ///     r"^encoder\..*",     // All encoder tensors
    ///     r"^decoder\..*",     // All decoder tensors
    /// ])?;
    ///
    /// // Using Vec
    /// let patterns = vec![r".*\.weight$", r".*\.bias$"];
    /// let collector = TensorViewCollector::with_filter(patterns)?;
    ///
    /// // Complex patterns
    /// let collector = TensorViewCollector::with_filter(&[
    ///     r"^model\.layer[12]\..*",  // layer1 or layer2
    ///     r".*\.(weight|bias)$",     // any weight or bias
    ///     r"^attention\..*\.query",  // attention query tensors
    /// ])?;
    /// ```
    #[cfg(target_has_atomic = "ptr")]
    pub fn with_filter<I, S>(patterns: I) -> Result<Self, regex::Error>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut filters = Vec::new();
        for pattern in patterns {
            filters.push(Regex::new(pattern.as_ref())?);
        }

        Ok(Self {
            tensors: HashMap::new(),
            path_stack: Vec::new(),
            filters,
        })
    }

    fn current_path(&self) -> String {
        self.path_stack.join(".")
    }

    fn should_collect(&self, _path: &str) -> bool {
        #[cfg(target_has_atomic = "ptr")]
        {
            if self.filters.is_empty() {
                true
            } else {
                // OR union - collect if ANY filter matches
                self.filters.iter().any(|filter| filter.is_match(_path))
            }
        }
        #[cfg(not(target_has_atomic = "ptr"))]
        {
            // Without regex support, collect all tensors
            true
        }
    }
}

impl<B: Backend> ModuleVisitor<B> for TensorViewCollector {
    fn enter_module(&mut self, name: &str) {
        self.path_stack.push(name.to_string());
    }

    fn exit_module(&mut self, _name: &str) {
        self.path_stack.pop();
    }

    fn visit_float<const D: usize>(&mut self, _id: ParamId, tensor: &Tensor<B, D>) {
        let path = self.current_path();
        if !path.is_empty() && self.should_collect(&path) {
            self.tensors.insert(path, TensorView::from_float(tensor));
        }
    }

    fn visit_int<const D: usize>(&mut self, _id: ParamId, tensor: &Tensor<B, D, Int>) {
        let path = self.current_path();
        if !path.is_empty() && self.should_collect(&path) {
            self.tensors.insert(path, TensorView::from_int(tensor));
        }
    }

    fn visit_bool<const D: usize>(&mut self, _id: ParamId, tensor: &Tensor<B, D, Bool>) {
        let path = self.current_path();
        if !path.is_empty() && self.should_collect(&path) {
            self.tensors.insert(path, TensorView::from_bool(tensor));
        }
    }

    fn visit_float_with_path<const D: usize>(
        &mut self,
        path: &str,
        _id: ParamId,
        tensor: &Tensor<B, D>,
    ) {
        if self.should_collect(path) {
            self.tensors
                .insert(path.to_string(), TensorView::from_float(tensor));
        }
    }

    fn visit_int_with_path<const D: usize>(
        &mut self,
        path: &str,
        _id: ParamId,
        tensor: &Tensor<B, D, Int>,
    ) {
        if self.should_collect(path) {
            self.tensors
                .insert(path.to_string(), TensorView::from_int(tensor));
        }
    }

    fn visit_bool_with_path<const D: usize>(
        &mut self,
        path: &str,
        _id: ParamId,
        tensor: &Tensor<B, D, Bool>,
    ) {
        if self.should_collect(path) {
            self.tensors
                .insert(path.to_string(), TensorView::from_bool(tensor));
        }
    }
}

#[cfg(all(test, target_has_atomic = "ptr"))]
mod tests {
    use super::*;
    use crate as burn; // Required for the derive macro
    use crate::{
        TestBackend,
        module::{Module, Param},
        nn::LinearConfig,
    };
    use alloc::collections::BTreeMap;
    use alloc::string::String;

    #[test]
    fn test_tensor_view_collector() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);

        let mut collector = TensorViewCollector::new();
        let id = ParamId::new();

        // Collect a tensor
        collector.visit_float_with_path("model.weight", id, &tensor);

        assert_eq!(collector.tensors.len(), 1);
        assert!(collector.tensors.contains_key("model.weight"));

        // Verify the tensor can be converted to data
        let view = collector.tensors.get("model.weight").unwrap();
        let data = view.to_data();
        assert_eq!(data.shape, vec![2, 2]);
    }

    #[test]
    fn test_tensor_view_collector_with_filter() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);

        let mut collector = TensorViewCollector::with_filter(&[r"^encoder\..*"]).unwrap();
        let id = ParamId::new();

        // This should be collected
        collector.visit_float_with_path("encoder.weight", id, &tensor);
        // This should NOT be collected
        collector.visit_float_with_path("decoder.weight", id, &tensor);

        assert_eq!(collector.tensors.len(), 1);
        assert!(collector.tensors.contains_key("encoder.weight"));
        assert!(!collector.tensors.contains_key("decoder.weight"));
    }

    #[test]
    fn test_tensor_view_collector_with_multiple_filters() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);

        // Multiple patterns - collect if matches ANY (OR union)
        let mut collector = TensorViewCollector::with_filter(&[
            r"^encoder\..*", // Match encoder.*
            r".*\.bias$",    // Match *.bias
        ])
        .unwrap();
        let id = ParamId::new();

        // These should be collected
        collector.visit_float_with_path("encoder.weight", id, &tensor); // matches first pattern
        collector.visit_float_with_path("decoder.bias", id, &tensor); // matches second pattern
        collector.visit_float_with_path("encoder.bias", id, &tensor); // matches both patterns

        // This should NOT be collected
        collector.visit_float_with_path("decoder.weight", id, &tensor); // matches neither

        assert_eq!(collector.tensors.len(), 3);
        assert!(collector.tensors.contains_key("encoder.weight"));
        assert!(collector.tensors.contains_key("decoder.bias"));
        assert!(collector.tensors.contains_key("encoder.bias"));
        assert!(!collector.tensors.contains_key("decoder.weight"));
    }

    // Test visitor that collects tensor paths
    struct TensorPathCollector {
        pub paths: BTreeMap<String, (ParamId, Vec<usize>)>,
        path_stack: Vec<String>,
    }

    impl TensorPathCollector {
        fn new() -> Self {
            Self {
                paths: BTreeMap::new(),
                path_stack: Vec::new(),
            }
        }

        fn current_path(&self) -> String {
            self.path_stack.join(".")
        }
    }

    impl<B: Backend> ModuleVisitor<B> for TensorPathCollector {
        fn enter_module(&mut self, name: &str) {
            self.path_stack.push(name.to_string());
        }

        fn exit_module(&mut self, _name: &str) {
            self.path_stack.pop();
        }

        fn visit_float<const D: usize>(&mut self, id: ParamId, tensor: &Tensor<B, D>) {
            let path = self.current_path();
            if !path.is_empty() {
                self.paths.insert(path, (id, tensor.shape().dims.to_vec()));
            }
        }

        fn visit_int<const D: usize>(&mut self, id: ParamId, tensor: &Tensor<B, D, Int>) {
            let path = self.current_path();
            if !path.is_empty() {
                self.paths.insert(path, (id, tensor.shape().dims.to_vec()));
            }
        }

        fn visit_bool<const D: usize>(&mut self, id: ParamId, tensor: &Tensor<B, D, Bool>) {
            let path = self.current_path();
            if !path.is_empty() {
                self.paths.insert(path, (id, tensor.shape().dims.to_vec()));
            }
        }
    }

    // Simple nested module for testing
    #[derive(Module, Debug)]
    struct InnerModule<B: Backend> {
        weight: Param<Tensor<B, 2>>,
        bias: Param<Tensor<B, 1>>,
    }

    #[derive(Module, Debug)]
    struct OuterModule<B: Backend> {
        layer1: InnerModule<B>,
        layer2: InnerModule<B>,
    }

    impl<B: Backend> InnerModule<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                weight: Param::from_data([[1.0, 2.0], [3.0, 4.0]], device),
                bias: Param::from_data([5.0, 6.0], device),
            }
        }
    }

    impl<B: Backend> OuterModule<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                layer1: InnerModule::new(device),
                layer2: InnerModule::new(device),
            }
        }
    }

    #[test]
    fn test_nested_module_path_tracking() {
        let device = Default::default();
        let module = OuterModule::<TestBackend>::new(&device);

        let mut collector = TensorPathCollector::new();
        module.visit(&mut collector);

        let paths = collector.paths;

        // Verify we have the expected paths
        // Note: Param<Tensor> fields are themselves modules, so we get an extra level
        assert!(paths.contains_key("layer1.weight"), "Missing layer1.weight");
        assert!(paths.contains_key("layer1.bias"), "Missing layer1.bias");
        assert!(paths.contains_key("layer2.weight"), "Missing layer2.weight");
        assert!(paths.contains_key("layer2.bias"), "Missing layer2.bias");

        // Verify the shapes are correct
        assert_eq!(paths.get("layer1.weight").unwrap().1, vec![2, 2]);
        assert_eq!(paths.get("layer1.bias").unwrap().1, vec![2]);
        assert_eq!(paths.get("layer2.weight").unwrap().1, vec![2, 2]);
        assert_eq!(paths.get("layer2.bias").unwrap().1, vec![2]);
    }

    #[test]
    fn test_linear_module_paths() {
        let device = Default::default();
        let config = LinearConfig::new(10, 20).with_bias(true);
        let linear = config.init::<TestBackend>(&device);

        let mut collector = TensorPathCollector::new();
        linear.visit(&mut collector);

        let paths = collector.paths;

        // Linear module has weight and optional bias
        assert!(paths.contains_key("weight"));
        assert!(paths.contains_key("bias"));

        // Check dimensions
        assert_eq!(paths.get("weight").unwrap().1, vec![10, 20]);
        assert_eq!(paths.get("bias").unwrap().1, vec![20]);
    }

    // Deep nesting test structures (4+ levels)
    #[derive(Module, Debug)]
    struct Level4Module<B: Backend> {
        weight: Param<Tensor<B, 2>>,
        bias: Param<Tensor<B, 1>>,
    }

    #[derive(Module, Debug)]
    struct Level3Module<B: Backend> {
        layer: Level4Module<B>,
        extra: Level4Module<B>,
    }

    #[derive(Module, Debug)]
    struct Level2Module<B: Backend> {
        block1: Level3Module<B>,
        block2: Level3Module<B>,
    }

    #[derive(Module, Debug)]
    struct Level1Module<B: Backend> {
        encoder: Level2Module<B>,
        decoder: Level2Module<B>,
    }

    #[derive(Module, Debug)]
    struct DeepModel<B: Backend> {
        backbone: Level1Module<B>,
        head: Level4Module<B>,
    }

    impl<B: Backend> Level4Module<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                weight: Param::from_data([[1.0, 2.0], [3.0, 4.0]], device),
                bias: Param::from_data([5.0, 6.0], device),
            }
        }
    }

    impl<B: Backend> Level3Module<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                layer: Level4Module::new(device),
                extra: Level4Module::new(device),
            }
        }
    }

    impl<B: Backend> Level2Module<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                block1: Level3Module::new(device),
                block2: Level3Module::new(device),
            }
        }
    }

    impl<B: Backend> Level1Module<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                encoder: Level2Module::new(device),
                decoder: Level2Module::new(device),
            }
        }
    }

    impl<B: Backend> DeepModel<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                backbone: Level1Module::new(device),
                head: Level4Module::new(device),
            }
        }
    }

    #[test]
    fn test_deep_module_path_tracking() {
        let device = Default::default();
        let model = DeepModel::<TestBackend>::new(&device);

        let mut collector = TensorViewCollector::new();
        model.visit(&mut collector);

        let paths = collector.tensors;

        // Test 5-level deep paths
        assert!(paths.contains_key("backbone.encoder.block1.layer.weight"));
        assert!(paths.contains_key("backbone.encoder.block1.layer.bias"));
        assert!(paths.contains_key("backbone.encoder.block1.extra.weight"));
        assert!(paths.contains_key("backbone.encoder.block1.extra.bias"));

        assert!(paths.contains_key("backbone.encoder.block2.layer.weight"));
        assert!(paths.contains_key("backbone.encoder.block2.layer.bias"));
        assert!(paths.contains_key("backbone.encoder.block2.extra.weight"));
        assert!(paths.contains_key("backbone.encoder.block2.extra.bias"));

        assert!(paths.contains_key("backbone.decoder.block1.layer.weight"));
        assert!(paths.contains_key("backbone.decoder.block1.layer.bias"));
        assert!(paths.contains_key("backbone.decoder.block1.extra.weight"));
        assert!(paths.contains_key("backbone.decoder.block1.extra.bias"));

        assert!(paths.contains_key("backbone.decoder.block2.layer.weight"));
        assert!(paths.contains_key("backbone.decoder.block2.layer.bias"));
        assert!(paths.contains_key("backbone.decoder.block2.extra.weight"));
        assert!(paths.contains_key("backbone.decoder.block2.extra.bias"));

        // Test 2-level paths
        assert!(paths.contains_key("head.weight"));
        assert!(paths.contains_key("head.bias"));

        // Total should be 18 tensors (16 from backbone + 2 from head)
        assert_eq!(paths.len(), 18);

        // Verify data can be materialized
        let view = paths.get("backbone.encoder.block1.layer.weight").unwrap();
        let data = view.to_data();
        assert_eq!(data.shape, vec![2, 2]);
    }

    #[test]
    fn test_deep_module_filtered_export() {
        let device = Default::default();
        let model = DeepModel::<TestBackend>::new(&device);

        // Test filtering at different depths
        let encoder_only = TensorViewCollector::with_filter(&[r"^backbone\.encoder\..*"]).unwrap();
        let mut collector = encoder_only;
        model.visit(&mut collector);
        assert_eq!(collector.tensors.len(), 8); // Only encoder tensors

        // Test filtering specific blocks
        let block1_only = TensorViewCollector::with_filter(&[r".*\.block1\..*"]).unwrap();
        let mut collector = block1_only;
        model.visit(&mut collector);
        assert_eq!(collector.tensors.len(), 8); // block1 in both encoder and decoder

        // Test filtering by tensor type at any depth
        let weights_only = TensorViewCollector::with_filter(&[r".*\.weight$"]).unwrap();
        let mut collector = weights_only;
        model.visit(&mut collector);
        assert_eq!(collector.tensors.len(), 9); // All weight tensors

        // Test complex multi-pattern filtering
        let complex_filter = TensorViewCollector::with_filter(&[
            r"^backbone\.encoder\.block1\..*", // All encoder.block1 tensors
            r"^backbone\.decoder\..*\.bias$",  // All decoder biases
            r"^head\.weight$",                 // Head weight only
        ])
        .unwrap();
        let mut collector = complex_filter;
        model.visit(&mut collector);

        // Should have:
        // - 4 from encoder.block1 (2 weights + 2 biases)
        // - 4 decoder biases
        // - 1 head weight
        assert_eq!(collector.tensors.len(), 9);

        assert!(
            collector
                .tensors
                .contains_key("backbone.encoder.block1.layer.weight")
        );
        assert!(
            collector
                .tensors
                .contains_key("backbone.decoder.block1.layer.bias")
        );
        assert!(collector.tensors.contains_key("head.weight"));
        assert!(!collector.tensors.contains_key("head.bias")); // Not included
    }
}
