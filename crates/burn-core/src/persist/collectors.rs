use alloc::string::{String, ToString};
use alloc::vec::Vec;

use burn_tensor::{Bool, Int, Tensor, backend::Backend};

use crate::module::{ModuleVisitor, ParamId};
use crate::persist::{PathFilter, TensorView};

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
/// let collector = TensorViewCollector::with_filter(PathFilter::new().with_regex(r"^encoder\..*"));
/// module.visit(&mut collector);
/// // Only collects tensors starting with "encoder."
/// ```
///
/// ## Filter with multiple patterns (OR union)
/// ```ignore
/// let filter = PathFilter::new()
///     .with_regex(r"^encoder\..*")  // Match all encoder tensors
///     .with_regex(r".*\.bias$");    // OR match any bias tensors
/// let collector = TensorViewCollector::with_filter(filter);
/// module.visit(&mut collector);
/// // Collects tensors matching ANY of the patterns
/// ```
pub struct TensorViewCollector {
    /// Collection of tensor views
    pub tensors: Vec<TensorView>,
    path_stack: Vec<String>,
    container_stack: Vec<String>,
    filter: Option<PathFilter>,
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
            tensors: Vec::new(),
            path_stack: Vec::new(),
            container_stack: Vec::new(),
            filter: None,
        }
    }

    /// Create a new tensor view collector with a PathFilter.
    ///
    /// This provides the most flexible filtering using the PathFilter's capabilities.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use burn::persist::PathFilter;
    ///
    /// // Use PathFilter builder
    /// let filter = PathFilter::new()
    ///     .with_regex(r"^encoder\..*")
    ///     .with_full_path("decoder.weight");
    /// let collector = TensorViewCollector::with_filter(filter);
    /// ```
    pub fn with_filter(filter: PathFilter) -> Self {
        Self {
            tensors: Vec::new(),
            path_stack: Vec::new(),
            container_stack: Vec::new(),
            filter: Some(filter),
        }
    }

    fn current_path(&self) -> String {
        self.path_stack.join(".")
    }

    fn should_collect(&self, path: &str, container_path: &str) -> bool {
        // If filter is present, use it; otherwise collect all
        match &self.filter {
            None => true,
            Some(f) => f.matches_with_container_path(path, container_path),
        }
    }
}

impl<B: Backend> ModuleVisitor<B> for TensorViewCollector {
    fn enter_module(&mut self, name: &str, container_type: &str) {
        self.path_stack.push(name.to_string());
        self.container_stack.push(container_type.to_string());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path_stack.pop();
        self.container_stack.pop();
    }

    fn visit_float<const D: usize>(&mut self, id: ParamId, tensor: &Tensor<B, D>) {
        let path = self.current_path();
        let container_path = self.container_stack.join(".");
        if !path.is_empty() && self.should_collect(&path, &container_path) {
            self.tensors.push(TensorView::from_float(
                tensor,
                self.path_stack.clone(),
                self.container_stack.clone(),
                id,
            ));
        }
    }

    fn visit_int<const D: usize>(&mut self, id: ParamId, tensor: &Tensor<B, D, Int>) {
        let path = self.current_path();
        let container_path = self.container_stack.join(".");
        if !path.is_empty() && self.should_collect(&path, &container_path) {
            self.tensors.push(TensorView::from_int(
                tensor,
                self.path_stack.clone(),
                self.container_stack.clone(),
                id,
            ));
        }
    }

    fn visit_bool<const D: usize>(&mut self, id: ParamId, tensor: &Tensor<B, D, Bool>) {
        let path = self.current_path();
        let container_path = self.container_stack.join(".");
        if !path.is_empty() && self.should_collect(&path, &container_path) {
            self.tensors.push(TensorView::from_bool(
                tensor,
                self.path_stack.clone(),
                self.container_stack.clone(),
                id,
            ));
        }
    }

    fn visit_float_with_path<const D: usize>(
        &mut self,
        path: &str,
        id: ParamId,
        tensor: &Tensor<B, D>,
    ) {
        // For path-based visits, we use the current container stack for filtering
        let container_path = self.container_stack.join(".");
        if self.should_collect(path, &container_path) {
            // For path-based visits, we need to construct the path stack from the path string
            let path_parts: Vec<String> = path.split('.').map(|s| s.to_string()).collect();
            self.tensors.push(TensorView::from_float(
                tensor,
                path_parts,
                self.container_stack.clone(),
                id,
            ));
        }
    }

    fn visit_int_with_path<const D: usize>(
        &mut self,
        path: &str,
        id: ParamId,
        tensor: &Tensor<B, D, Int>,
    ) {
        let container_path = self.container_stack.join(".");
        if self.should_collect(path, &container_path) {
            let path_parts: Vec<String> = path.split('.').map(|s| s.to_string()).collect();
            self.tensors.push(TensorView::from_int(
                tensor,
                path_parts,
                self.container_stack.clone(),
                id,
            ));
        }
    }

    fn visit_bool_with_path<const D: usize>(
        &mut self,
        path: &str,
        id: ParamId,
        tensor: &Tensor<B, D, Bool>,
    ) {
        let container_path = self.container_stack.join(".");
        if self.should_collect(path, &container_path) {
            let path_parts: Vec<String> = path.split('.').map(|s| s.to_string()).collect();
            self.tensors.push(TensorView::from_bool(
                tensor,
                path_parts,
                self.container_stack.clone(),
                id,
            ));
        }
    }
}

#[cfg(all(test, feature = "std"))]
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
        assert_eq!(collector.tensors[0].full_path(), "model.weight");

        // Verify the tensor can be converted to data
        let view = &collector.tensors[0];
        let data = view.to_data();
        assert_eq!(data.shape, vec![2, 2]);
    }

    #[test]
    #[cfg(target_has_atomic = "ptr")]
    fn test_tensor_view_collector_with_filter() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);

        let filter = PathFilter::new().with_regex(r"^encoder\..*");
        let mut collector = TensorViewCollector::with_filter(filter);
        let id = ParamId::new();

        // This should be collected
        collector.visit_float_with_path("encoder.weight", id, &tensor);
        // This should NOT be collected
        collector.visit_float_with_path("decoder.weight", id, &tensor);

        assert_eq!(collector.tensors.len(), 1);
        assert_eq!(collector.tensors[0].full_path(), "encoder.weight");
    }

    #[test]
    #[cfg(target_has_atomic = "ptr")]
    fn test_tensor_view_collector_with_multiple_filters() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);

        // Multiple patterns - collect if matches ANY (OR union)
        let filter = PathFilter::new()
            .with_regex(r"^encoder\..*") // Match encoder.*
            .with_regex(r".*\.bias$"); // Match *.bias
        let mut collector = TensorViewCollector::with_filter(filter);
        let id = ParamId::new();

        // These should be collected
        collector.visit_float_with_path("encoder.weight", id, &tensor); // matches first pattern
        collector.visit_float_with_path("decoder.bias", id, &tensor); // matches second pattern
        collector.visit_float_with_path("encoder.bias", id, &tensor); // matches both patterns

        // This should NOT be collected
        collector.visit_float_with_path("decoder.weight", id, &tensor); // matches neither

        assert_eq!(collector.tensors.len(), 3);
        let paths: Vec<String> = collector.tensors.iter().map(|v| v.full_path()).collect();
        assert!(paths.contains(&"encoder.weight".to_string()));
        assert!(paths.contains(&"decoder.bias".to_string()));
        assert!(paths.contains(&"encoder.bias".to_string()));
        assert!(!paths.contains(&"decoder.weight".to_string()));
    }

    #[test]
    fn test_tensor_view_collector_with_predicate() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);

        // Use predicate function for filtering
        fn filter_fn(path: &str, _container_path: &str) -> bool {
            path.starts_with("encoder.") || path == "decoder.bias"
        }
        let filter = PathFilter::new().with_predicate(filter_fn);
        let mut collector = TensorViewCollector::with_filter(filter);
        let id = ParamId::new();

        // These should be collected
        collector.visit_float_with_path("encoder.weight", id, &tensor);
        collector.visit_float_with_path("encoder.bias", id, &tensor);
        collector.visit_float_with_path("decoder.bias", id, &tensor);

        // This should NOT be collected
        collector.visit_float_with_path("decoder.weight", id, &tensor);
        collector.visit_float_with_path("other.tensor", id, &tensor);

        assert_eq!(collector.tensors.len(), 3);
        let paths: Vec<String> = collector.tensors.iter().map(|v| v.full_path()).collect();
        assert!(paths.contains(&"encoder.weight".to_string()));
        assert!(paths.contains(&"encoder.bias".to_string()));
        assert!(paths.contains(&"decoder.bias".to_string()));
        assert!(!paths.contains(&"decoder.weight".to_string()));
        assert!(!paths.contains(&"other.tensor".to_string()));
    }

    #[test]
    fn test_tensor_view_collector_predicate_with_complex_logic() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);

        // Complex predicate with multiple conditions
        fn complex_filter(path: &str, _container_path: &str) -> bool {
            let parts: Vec<&str> = path.split('.').collect();
            if parts.len() != 3 {
                return false;
            }
            // Only collect if it's layer1 or layer2, and it's a weight tensor
            (parts[1] == "layer1" || parts[1] == "layer2") && parts[2] == "weight"
        }
        let filter = PathFilter::new().with_predicate(complex_filter);
        let mut collector = TensorViewCollector::with_filter(filter);
        let id = ParamId::new();

        // These should be collected
        collector.visit_float_with_path("model.layer1.weight", id, &tensor);
        collector.visit_float_with_path("model.layer2.weight", id, &tensor);

        // These should NOT be collected
        collector.visit_float_with_path("model.layer1.bias", id, &tensor);
        collector.visit_float_with_path("model.layer3.weight", id, &tensor);
        collector.visit_float_with_path("encoder.weight", id, &tensor); // wrong structure

        assert_eq!(collector.tensors.len(), 2);
        let paths: Vec<String> = collector.tensors.iter().map(|v| v.full_path()).collect();
        assert!(paths.contains(&"model.layer1.weight".to_string()));
        assert!(paths.contains(&"model.layer2.weight".to_string()));
        assert!(!paths.contains(&"model.layer1.bias".to_string()));
        assert!(!paths.contains(&"model.layer3.weight".to_string()));
        assert!(!paths.contains(&"encoder.weight".to_string()));
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
        fn enter_module(&mut self, name: &str, _container_type: &str) {
            self.path_stack.push(name.to_string());
        }

        fn exit_module(&mut self, _name: &str, _container_type: &str) {
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

        let views = collector.tensors;
        let paths: Vec<String> = views.iter().map(|v| v.full_path()).collect();

        // Test 5-level deep paths
        assert!(paths.contains(&"backbone.encoder.block1.layer.weight".to_string()));
        assert!(paths.contains(&"backbone.encoder.block1.layer.bias".to_string()));
        assert!(paths.contains(&"backbone.encoder.block1.extra.weight".to_string()));
        assert!(paths.contains(&"backbone.encoder.block1.extra.bias".to_string()));

        assert!(paths.contains(&"backbone.encoder.block2.layer.weight".to_string()));
        assert!(paths.contains(&"backbone.encoder.block2.layer.bias".to_string()));
        assert!(paths.contains(&"backbone.encoder.block2.extra.weight".to_string()));
        assert!(paths.contains(&"backbone.encoder.block2.extra.bias".to_string()));

        assert!(paths.contains(&"backbone.decoder.block1.layer.weight".to_string()));
        assert!(paths.contains(&"backbone.decoder.block1.layer.bias".to_string()));
        assert!(paths.contains(&"backbone.decoder.block1.extra.weight".to_string()));
        assert!(paths.contains(&"backbone.decoder.block1.extra.bias".to_string()));

        assert!(paths.contains(&"backbone.decoder.block2.layer.weight".to_string()));
        assert!(paths.contains(&"backbone.decoder.block2.layer.bias".to_string()));
        assert!(paths.contains(&"backbone.decoder.block2.extra.weight".to_string()));
        assert!(paths.contains(&"backbone.decoder.block2.extra.bias".to_string()));

        // Test 2-level paths
        assert!(paths.contains(&"head.weight".to_string()));
        assert!(paths.contains(&"head.bias".to_string()));

        // Total should be 18 tensors (16 from backbone + 2 from head)
        assert_eq!(views.len(), 18);

        // Verify data can be materialized
        let view = views
            .iter()
            .find(|v| v.full_path() == "backbone.encoder.block1.layer.weight")
            .unwrap();
        let data = view.to_data();
        assert_eq!(data.shape, vec![2, 2]);
    }

    #[test]
    fn test_deep_module_filtered_export() {
        let device = Default::default();
        let model = DeepModel::<TestBackend>::new(&device);

        // Test filtering at different depths
        #[cfg(target_has_atomic = "ptr")]
        {
            let filter = PathFilter::new().with_regex(r"^backbone\.encoder\..*");
            let mut collector = TensorViewCollector::with_filter(filter);
            model.visit(&mut collector);
            assert_eq!(collector.tensors.len(), 8); // Only encoder tensors
        }

        // Test filtering specific blocks
        #[cfg(target_has_atomic = "ptr")]
        {
            let filter = PathFilter::new().with_regex(r".*\.block1\..*");
            let mut collector = TensorViewCollector::with_filter(filter);
            model.visit(&mut collector);
            assert_eq!(collector.tensors.len(), 8); // block1 in both encoder and decoder
        }

        // Test filtering by tensor type at any depth
        #[cfg(target_has_atomic = "ptr")]
        {
            let filter = PathFilter::new().with_regex(r".*\.weight$");
            let mut collector = TensorViewCollector::with_filter(filter);
            model.visit(&mut collector);
            assert_eq!(collector.tensors.len(), 9); // All weight tensors
        }

        // Test complex multi-pattern filtering
        #[cfg(target_has_atomic = "ptr")]
        {
            let filter = PathFilter::new()
                .with_regex(r"^backbone\.encoder\.block1\..*") // All encoder.block1 tensors
                .with_regex(r"^backbone\.decoder\..*\.bias$") // All decoder biases
                .with_regex(r"^head\.weight$"); // Head weight only
            let mut collector = TensorViewCollector::with_filter(filter);
            model.visit(&mut collector);

            // Should have:
            // - 4 from encoder.block1 (2 weights + 2 biases)
            // - 4 decoder biases
            // - 1 head weight
            assert_eq!(collector.tensors.len(), 9);

            let paths: Vec<String> = collector.tensors.iter().map(|v| v.full_path()).collect();
            assert!(paths.contains(&"backbone.encoder.block1.layer.weight".to_string()));
            assert!(paths.contains(&"backbone.decoder.block1.layer.bias".to_string()));
            assert!(paths.contains(&"head.weight".to_string()));
            assert!(!paths.contains(&"head.bias".to_string())); // Not included
        }
    }
}
