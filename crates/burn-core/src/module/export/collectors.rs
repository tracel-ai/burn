use alloc::string::{String, ToString};
use alloc::vec::Vec;
use hashbrown::HashMap;
use regex::Regex;

use burn_tensor::{Bool, Int, Tensor, backend::Backend};

use super::TensorView;
use crate::module::{ModuleVisitor, ParamId};

/// Collects tensor views without copying data
pub struct TensorViewCollector {
    /// Map of tensor paths to their views
    pub tensors: HashMap<String, TensorView>,
    path_stack: Vec<String>,
    filter: Option<Regex>,
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
            filter: None,
        }
    }

    /// Create a new tensor view collector that only collects tensors matching the regex pattern.
    pub fn with_filter(pattern: &str) -> Result<Self, regex::Error> {
        Ok(Self {
            tensors: HashMap::new(),
            path_stack: Vec::new(),
            filter: Some(Regex::new(pattern)?),
        })
    }

    fn current_path(&self) -> String {
        self.path_stack.join(".")
    }

    fn should_collect(&self, path: &str) -> bool {
        match &self.filter {
            Some(filter) => filter.is_match(path),
            None => true,
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

#[cfg(test)]
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

        let mut collector = TensorViewCollector::with_filter(r"^encoder\..*").unwrap();
        let id = ParamId::new();

        // This should be collected
        collector.visit_float_with_path("encoder.weight", id, &tensor);
        // This should NOT be collected
        collector.visit_float_with_path("decoder.weight", id, &tensor);

        assert_eq!(collector.tensors.len(), 1);
        assert!(collector.tensors.contains_key("encoder.weight"));
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
}
