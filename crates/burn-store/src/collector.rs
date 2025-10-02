use alloc::string::{String, ToString};
use alloc::vec::Vec;

use burn_tensor::{Bool, Int, Tensor, backend::Backend};

use crate::{PathFilter, TensorSnapshot};
use burn_core::module::{ModuleVisitor, ParamId};

/// Collects tensor views from modules without copying data.
///
/// This collector traverses a module hierarchy and creates lightweight views
/// of tensors that can be materialized to `TensorData` on demand.
///
/// # Examples
///
/// ## Collect all tensors
/// ```rust,ignore
/// let collector = Collector::new();
/// module.visit(&mut collector);
/// let all_tensors = collector.tensors;
/// ```
///
/// ## Filter with single pattern
/// ```rust,ignore
/// let collector = Collector::with_filter(PathFilter::new().with_regex(r"^encoder\..*"));
/// module.visit(&mut collector);
/// // Only collects tensors starting with "encoder."
/// ```
///
/// ## Filter with multiple patterns (OR union)
/// ```rust,ignore
/// let filter = PathFilter::new()
///     .with_regex(r"^encoder\..*")  // Match all encoder tensors
///     .with_regex(r".*\.bias$");    // OR match any bias tensors
/// let collector = Collector::with_filter(filter);
/// module.visit(&mut collector);
/// // Collects tensors matching ANY of the patterns
/// ```
pub struct Collector {
    /// Collection of tensor views
    pub tensors: Vec<TensorSnapshot>,
    path_stack: Vec<String>,
    container_stack: Vec<String>,
    filter: Option<PathFilter>,
}

impl Default for Collector {
    fn default() -> Self {
        Self::new()
    }
}

impl Collector {
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
    /// ```rust,ignore
    /// use burn_store::PathFilter;
    ///
    /// // Use PathFilter builder
    /// let filter = PathFilter::new()
    ///     .with_regex(r"^encoder\..*")
    ///     .with_full_path("decoder.weight");
    /// let collector = Collector::with_filter(filter);
    /// ```
    pub fn with_filter(filter: PathFilter) -> Self {
        Self {
            tensors: Vec::new(),
            path_stack: Vec::new(),
            container_stack: Vec::new(),
            filter: Some(filter),
        }
    }

    fn should_collect(&self, path: &[String], container_stack: &[String]) -> bool {
        // If filter is present, use it; otherwise collect all
        match &self.filter {
            None => true,
            Some(f) => f.matches_with_container_path(path, container_stack),
        }
    }
}

impl<B: Backend> ModuleVisitor<B> for Collector {
    fn enter_module(&mut self, name: &str, container_type: &str) {
        self.path_stack.push(name.to_string());
        self.container_stack.push(container_type.to_string());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path_stack.pop();
        self.container_stack.pop();
    }

    fn visit_float<const D: usize>(&mut self, id: ParamId, tensor: &Tensor<B, D>) {
        if !self.path_stack.is_empty()
            && self.should_collect(&self.path_stack, &self.container_stack)
        {
            self.tensors.push(TensorSnapshot::from_float(
                tensor,
                self.path_stack.clone(),
                self.container_stack.clone(),
                id,
            ));
        }
    }

    fn visit_int<const D: usize>(&mut self, id: ParamId, tensor: &Tensor<B, D, Int>) {
        if !self.path_stack.is_empty()
            && self.should_collect(&self.path_stack, &self.container_stack)
        {
            self.tensors.push(TensorSnapshot::from_int(
                tensor,
                self.path_stack.clone(),
                self.container_stack.clone(),
                id,
            ));
        }
    }

    fn visit_bool<const D: usize>(&mut self, id: ParamId, tensor: &Tensor<B, D, Bool>) {
        if !self.path_stack.is_empty()
            && self.should_collect(&self.path_stack, &self.container_stack)
        {
            self.tensors.push(TensorSnapshot::from_bool(
                tensor,
                self.path_stack.clone(),
                self.container_stack.clone(),
                id,
            ));
        }
    }

    fn visit_float_with_path<const D: usize>(
        &mut self,
        path: &[String],
        id: ParamId,
        tensor: &Tensor<B, D>,
    ) {
        // For path-based visits, we use the current container stack for filtering
        if self.should_collect(path, &self.container_stack) {
            self.tensors.push(TensorSnapshot::from_float(
                tensor,
                path.to_vec(),
                self.container_stack.clone(),
                id,
            ));
        }
    }

    fn visit_int_with_path<const D: usize>(
        &mut self,
        path: &[String],
        id: ParamId,
        tensor: &Tensor<B, D, Int>,
    ) {
        if self.should_collect(path, &self.container_stack) {
            self.tensors.push(TensorSnapshot::from_int(
                tensor,
                path.to_vec(),
                self.container_stack.clone(),
                id,
            ));
        }
    }

    fn visit_bool_with_path<const D: usize>(
        &mut self,
        path: &[String],
        id: ParamId,
        tensor: &Tensor<B, D, Bool>,
    ) {
        if self.should_collect(path, &self.container_stack) {
            self.tensors.push(TensorSnapshot::from_bool(
                tensor,
                path.to_vec(),
                self.container_stack.clone(),
                id,
            ));
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    type TestBackend = burn_ndarray::NdArray;
    use alloc::collections::BTreeMap;
    use alloc::string::String;
    use burn::nn::LinearConfig;
    use burn_core::module::{Module, Param};

    #[test]
    fn tensor_snapshot_collector() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);

        let mut collector = Collector::new();
        let id = ParamId::new();

        // Collect a tensor
        collector.visit_float_with_path(&["model".to_string(), "weight".to_string()], id, &tensor);

        assert_eq!(collector.tensors.len(), 1);
        assert_eq!(collector.tensors[0].full_path(), "model.weight");

        // Verify the tensor can be converted to data
        let view = &collector.tensors[0];
        let data = view.to_data();
        assert_eq!(data.shape, vec![2, 2]);
    }

    #[test]
    #[cfg(target_has_atomic = "ptr")]
    fn tensor_snapshot_collector_with_filter() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);

        let filter = PathFilter::new().with_regex(r"^encoder\..*");
        let mut collector = Collector::with_filter(filter);
        let id = ParamId::new();

        // This should be collected
        collector.visit_float_with_path(
            &["encoder".to_string(), "weight".to_string()],
            id,
            &tensor,
        );
        // This should NOT be collected
        collector.visit_float_with_path(
            &["decoder".to_string(), "weight".to_string()],
            id,
            &tensor,
        );

        assert_eq!(collector.tensors.len(), 1);
        assert_eq!(collector.tensors[0].full_path(), "encoder.weight");
    }

    #[test]
    #[cfg(target_has_atomic = "ptr")]
    fn tensor_snapshot_collector_with_multiple_filters() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);

        // Multiple patterns - collect if matches ANY (OR union)
        let filter = PathFilter::new()
            .with_regex(r"^encoder\..*") // Match encoder.*
            .with_regex(r".*\.bias$"); // Match *.bias
        let mut collector = Collector::with_filter(filter);
        let id = ParamId::new();

        // These should be collected
        collector.visit_float_with_path(
            &["encoder".to_string(), "weight".to_string()],
            id,
            &tensor,
        ); // matches first pattern
        collector.visit_float_with_path(&["decoder".to_string(), "bias".to_string()], id, &tensor); // matches second pattern
        collector.visit_float_with_path(&["encoder".to_string(), "bias".to_string()], id, &tensor); // matches both patterns

        // This should NOT be collected
        collector.visit_float_with_path(
            &["decoder".to_string(), "weight".to_string()],
            id,
            &tensor,
        ); // matches neither

        assert_eq!(collector.tensors.len(), 3);
        let paths: Vec<String> = collector.tensors.iter().map(|v| v.full_path()).collect();
        assert!(paths.contains(&"encoder.weight".to_string()));
        assert!(paths.contains(&"decoder.bias".to_string()));
        assert!(paths.contains(&"encoder.bias".to_string()));
        assert!(!paths.contains(&"decoder.weight".to_string()));
    }

    #[test]
    fn tensor_snapshot_collector_with_predicate() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);

        // Use predicate function for filtering
        fn filter_fn(path: &str, _container_path: &str) -> bool {
            path.starts_with("encoder.") || path == "decoder.bias"
        }
        let filter = PathFilter::new().with_predicate(filter_fn);
        let mut collector = Collector::with_filter(filter);
        let id = ParamId::new();

        // These should be collected
        collector.visit_float_with_path(
            &["encoder".to_string(), "weight".to_string()],
            id,
            &tensor,
        );
        collector.visit_float_with_path(&["encoder".to_string(), "bias".to_string()], id, &tensor);
        collector.visit_float_with_path(&["decoder".to_string(), "bias".to_string()], id, &tensor);

        // This should NOT be collected
        collector.visit_float_with_path(
            &["decoder".to_string(), "weight".to_string()],
            id,
            &tensor,
        );
        collector.visit_float_with_path(&["other".to_string(), "tensor".to_string()], id, &tensor);

        assert_eq!(collector.tensors.len(), 3);
        let paths: Vec<String> = collector.tensors.iter().map(|v| v.full_path()).collect();
        assert!(paths.contains(&"encoder.weight".to_string()));
        assert!(paths.contains(&"encoder.bias".to_string()));
        assert!(paths.contains(&"decoder.bias".to_string()));
        assert!(!paths.contains(&"decoder.weight".to_string()));
        assert!(!paths.contains(&"other.tensor".to_string()));
    }

    #[test]
    fn tensor_snapshot_collector_predicate_with_complex_logic() {
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
        let mut collector = Collector::with_filter(filter);
        let id = ParamId::new();

        // These should be collected
        collector.visit_float_with_path(
            &[
                "model".to_string(),
                "layer1".to_string(),
                "weight".to_string(),
            ],
            id,
            &tensor,
        );
        collector.visit_float_with_path(
            &[
                "model".to_string(),
                "layer2".to_string(),
                "weight".to_string(),
            ],
            id,
            &tensor,
        );

        // These should NOT be collected
        collector.visit_float_with_path(
            &[
                "model".to_string(),
                "layer1".to_string(),
                "bias".to_string(),
            ],
            id,
            &tensor,
        );
        collector.visit_float_with_path(
            &[
                "model".to_string(),
                "layer3".to_string(),
                "weight".to_string(),
            ],
            id,
            &tensor,
        );
        collector.visit_float_with_path(
            &["encoder".to_string(), "weight".to_string()],
            id,
            &tensor,
        ); // wrong structure

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
                self.paths.insert(path, (id, tensor.shape().to_vec()));
            }
        }

        fn visit_int<const D: usize>(&mut self, id: ParamId, tensor: &Tensor<B, D, Int>) {
            let path = self.current_path();
            if !path.is_empty() {
                self.paths.insert(path, (id, tensor.shape().to_vec()));
            }
        }

        fn visit_bool<const D: usize>(&mut self, id: ParamId, tensor: &Tensor<B, D, Bool>) {
            let path = self.current_path();
            if !path.is_empty() {
                self.paths.insert(path, (id, tensor.shape().to_vec()));
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
    fn nested_module_path_tracking() {
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
    fn linear_module_paths() {
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
    fn deep_module_path_tracking() {
        let device = Default::default();
        let model = DeepModel::<TestBackend>::new(&device);

        let mut collector = Collector::new();
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
    fn deep_module_filtered_export() {
        let device = Default::default();
        let model = DeepModel::<TestBackend>::new(&device);

        // Test filtering at different depths
        #[cfg(target_has_atomic = "ptr")]
        {
            let filter = PathFilter::new().with_regex(r"^backbone\.encoder\..*");
            let mut collector = Collector::with_filter(filter);
            model.visit(&mut collector);
            assert_eq!(collector.tensors.len(), 8); // Only encoder tensors
        }

        // Test filtering specific blocks
        #[cfg(target_has_atomic = "ptr")]
        {
            let filter = PathFilter::new().with_regex(r".*\.block1\..*");
            let mut collector = Collector::with_filter(filter);
            model.visit(&mut collector);
            assert_eq!(collector.tensors.len(), 8); // block1 in both encoder and decoder
        }

        // Test filtering by tensor type at any depth
        #[cfg(target_has_atomic = "ptr")]
        {
            let filter = PathFilter::new().with_regex(r".*\.weight$");
            let mut collector = Collector::with_filter(filter);
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
            let mut collector = Collector::with_filter(filter);
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

    use crate::traits::ModuleSnapshot;
    use burn::nn::Linear;
    use hashbrown::HashMap;

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
    fn optional_field_module_with_value() {
        let device = Default::default();
        let module = OptionalFieldModule::<TestBackend>::new_with_optional(&device);

        let views: HashMap<String, TensorSnapshot> = module
            .collect()
            .into_iter()
            .map(|v| (v.full_path(), v))
            .collect();

        assert_eq!(views.len(), 2);
        assert!(views.contains_key("required"));
        assert!(views.contains_key("optional"));
    }

    #[test]
    fn optional_field_module_without_value() {
        let device = Default::default();
        let module = OptionalFieldModule::<TestBackend>::new_without_optional(&device);

        let views: HashMap<String, TensorSnapshot> = module
            .collect()
            .into_iter()
            .map(|v| (v.full_path(), v))
            .collect();

        assert_eq!(views.len(), 1);
        assert!(views.contains_key("required"));
        assert!(!views.contains_key("optional"));
    }

    // Test Vec of modules
    #[derive(Module, Debug)]
    struct VecModule<B: Backend> {
        layers: Vec<Linear<B>>,
    }

    impl<B: Backend> VecModule<B> {
        fn new(device: &B::Device, num_layers: usize) -> Self {
            Self {
                layers: (0..num_layers)
                    .map(|_| LinearConfig::new(10, 10).init(device))
                    .collect(),
            }
        }
    }

    #[test]
    fn vec_module_collect() {
        let device = Default::default();
        let module = VecModule::<TestBackend>::new(&device, 3);

        let views: HashMap<String, TensorSnapshot> = module
            .collect()
            .into_iter()
            .map(|v| (v.full_path(), v))
            .collect();

        // With the fix, all Vec items should now be properly indexed and visited
        assert_eq!(views.len(), 6); // 3 layers × 2 tensors each = 6 tensors

        // Check that all indexed paths exist
        assert!(views.contains_key("layers.0.weight"));
        assert!(views.contains_key("layers.0.bias"));
        assert!(views.contains_key("layers.1.weight"));
        assert!(views.contains_key("layers.1.bias"));
        assert!(views.contains_key("layers.2.weight"));
        assert!(views.contains_key("layers.2.bias"));
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
                    LinearConfig::new(10, 10).init(device),
                    LinearConfig::new(10, 10).init(device),
                    LinearConfig::new(10, 10).init(device),
                ],
            }
        }
    }

    #[test]
    fn array_module_collect() {
        let device = Default::default();
        let module = ArrayModule::<TestBackend>::new(&device);

        let views: HashMap<String, TensorSnapshot> = module
            .collect()
            .into_iter()
            .map(|v| (v.full_path(), v))
            .collect();

        // All array items should be properly indexed
        assert_eq!(views.len(), 6); // 3 layers × 2 tensors each = 6 tensors

        // Check indexed paths
        for i in 0..3 {
            assert!(views.contains_key(&format!("layers.{}.weight", i)));
            assert!(views.contains_key(&format!("layers.{}.bias", i)));
        }
    }

    // Test enum modules
    #[derive(Module, Debug)]
    enum EnumModule<B: Backend> {
        LayerA(Linear<B>),
        LayerB(Linear<B>),
        LayerC(Linear<B>),
    }

    #[test]
    fn enum_module_collect() {
        let device = Default::default();

        // Test variant A
        let module_a = EnumModule::<TestBackend>::LayerA(LinearConfig::new(10, 20).init(&device));
        let views_a: HashMap<String, TensorSnapshot> = module_a
            .collect()
            .into_iter()
            .map(|v| (v.full_path(), v))
            .collect();

        // Should have the variant name in the path
        assert_eq!(views_a.len(), 2);
        assert!(views_a.contains_key("LayerA.weight"));
        assert!(views_a.contains_key("LayerA.bias"));

        // Test variant B
        let module_b = EnumModule::<TestBackend>::LayerB(LinearConfig::new(10, 20).init(&device));
        let views_b: HashMap<String, TensorSnapshot> = module_b
            .collect()
            .into_iter()
            .map(|v| (v.full_path(), v))
            .collect();

        assert_eq!(views_b.len(), 2);
        assert!(views_b.contains_key("LayerB.weight"));
        assert!(views_b.contains_key("LayerB.bias"));
    }

    // Container type tracking tests
    #[test]
    fn linear_container_type() {
        let device = Default::default();

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

        let model = ModelWithLinear::<TestBackend>::new(&device);

        let views: HashMap<String, TensorSnapshot> = model
            .collect()
            .into_iter()
            .map(|v| (v.full_path(), v))
            .collect();

        // Check that tensors inside Linear layers have "Linear" as their container type
        for (path, view) in views.iter() {
            if path == "linear.weight" || path == "linear.bias" {
                assert_eq!(
                    view.container_type(),
                    "Linear",
                    "Tensor '{}' should have container type 'Linear'",
                    path
                );
            }
        }
    }

    #[test]
    fn complex_model_container_types() {
        let device = Default::default();

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

        let model = ComplexModel::<TestBackend>::new(&device);

        let views: HashMap<String, TensorSnapshot> = model
            .collect()
            .into_iter()
            .map(|v| (v.full_path(), v))
            .collect();

        // Should have 10 tensors total
        assert_eq!(views.len(), 10);

        // Verify different container types
        for (_path, view) in views.iter() {
            assert_eq!(view.container_type(), "Linear");
        }
    }

    #[test]
    fn collect_with_container_filter() {
        let device = Default::default();

        #[derive(Module, Debug)]
        struct FilterTestModel<B: Backend> {
            layers: Vec<Linear<B>>,
        }

        impl<B: Backend> FilterTestModel<B> {
            fn new(device: &B::Device) -> Self {
                Self {
                    layers: vec![
                        LinearConfig::new(10, 10).init(device),
                        LinearConfig::new(10, 10).init(device),
                    ],
                }
            }
        }

        let model = FilterTestModel::<TestBackend>::new(&device);

        // Filter to only collect tensors from Linear modules
        let filter = PathFilter::new().with_predicate(|_path, container_path| {
            container_path.split('.').next_back() == Some("Linear")
        });

        let linear_views: Vec<TensorSnapshot> = model.collect_with_filter(filter);

        // All collected tensors should be from Linear modules
        for view in linear_views.iter() {
            assert_eq!(
                view.container_type(),
                "Linear",
                "All tensors should be from Linear modules"
            );
        }

        // Should have collected all Linear tensors
        assert_eq!(linear_views.len(), 4);
    }
}
