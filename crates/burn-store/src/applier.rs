use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;
use hashbrown::{HashMap, HashSet};

use burn_tensor::{Bool, Int, Tensor, backend::Backend};

use crate::{PathFilter, TensorSnapshot};
use burn_core::module::{ModuleMapper, ParamId};

/// Error types for apply operations
#[derive(Debug)]
pub enum ApplyError {
    /// Shape mismatch between source and target tensor
    ShapeMismatch {
        /// Path of the tensor
        path: String,
        /// Expected shape
        expected: Vec<usize>,
        /// Found shape
        found: Vec<usize>,
    },
    /// Data type mismatch
    TypeMismatch {
        /// Path of the tensor
        path: String,
        /// Error message
        message: String,
    },
    /// Tensor path not found in target module
    PathNotFound {
        /// Path of the tensor
        path: String,
    },
    /// Invalid tensor data
    InvalidData {
        /// Path of the tensor
        path: String,
        /// Reason for invalidity
        reason: String,
    },
    /// Regex pattern error
    #[cfg(feature = "std")]
    RegexError(regex::Error),
    /// Generic error
    Other(String),
}

impl fmt::Display for ApplyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ApplyError::ShapeMismatch {
                path,
                expected,
                found,
            } => write!(
                f,
                "Shape mismatch for tensor '{}': expected {:?}, found {:?}",
                path, expected, found
            ),
            ApplyError::TypeMismatch { path, message } => {
                write!(f, "Type mismatch for tensor '{}': {}", path, message)
            }
            ApplyError::PathNotFound { path } => {
                write!(f, "Tensor path '{}' not found in module", path)
            }
            ApplyError::InvalidData { path, reason } => {
                write!(f, "Invalid data for tensor '{}': {}", path, reason)
            }
            #[cfg(feature = "std")]
            ApplyError::RegexError(e) => write!(f, "Regex error: {}", e),
            ApplyError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ApplyError {}

#[cfg(feature = "std")]
impl From<regex::Error> for ApplyError {
    fn from(err: regex::Error) -> Self {
        ApplyError::RegexError(err)
    }
}

/// Result of an apply operation
#[derive(Debug, Clone)]
pub struct ApplyResult {
    /// Successfully applied tensor paths
    pub applied: Vec<String>,
    /// Paths that were filtered out (not attempted)
    pub skipped: Vec<String>,
    /// Paths in module but not in sources
    pub missing: Vec<String>,
    /// Paths in sources but not found in module
    pub unused: Vec<String>,
    /// Errors encountered during apply
    pub errors: Vec<String>,
}

impl ApplyResult {
    /// Check if the apply was successful (no errors)
    pub fn is_success(&self) -> bool {
        self.errors.is_empty()
    }

    /// Get the total number of tensors processed
    pub fn total_processed(&self) -> usize {
        self.applied.len() + self.errors.len()
    }
}

impl fmt::Display for ApplyResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Apply Result:")?;
        writeln!(f, "  Applied: {} tensors", self.applied.len())?;
        if !self.skipped.is_empty() {
            writeln!(f, "  Skipped: {} tensors (filtered)", self.skipped.len())?;
        }
        if !self.missing.is_empty() {
            writeln!(f, "  Missing: {} tensors", self.missing.len())?;
        }
        if !self.unused.is_empty() {
            writeln!(f, "  Unused: {} tensors", self.unused.len())?;
        }
        if !self.errors.is_empty() {
            writeln!(f, "  Errors: {} tensors", self.errors.len())?;
            for error in &self.errors {
                writeln!(f, "    - {}", error)?;
            }
        }
        Ok(())
    }
}

/// Applies tensor views to a module using the ModuleMapper trait.
///
/// This applier traverses the module hierarchy and applies tensor data
/// from TensorSnapshots to the corresponding tensors in the module.
pub struct Applier<B: Backend> {
    /// Map of tensor paths to their views for O(1) lookup
    views: HashMap<String, TensorSnapshot>,
    /// Current path in the module hierarchy
    path_stack: Vec<String>,
    /// Current container type stack in the module hierarchy
    container_stack: Vec<String>,
    /// Path filter for selective apply
    filter: Option<PathFilter>,
    /// Successfully applied tensor paths (Vec for ordered output)
    applied: Vec<String>,
    /// Skipped tensor paths (HashSet for O(1) lookup in into_result)
    skipped: HashSet<String>,
    /// Errors encountered during application (Vec for ordered output)
    errors: Vec<String>,
    /// Track visited paths to find missing tensors (HashSet for O(1) lookup)
    visited_paths: HashSet<String>,
    /// Phantom data for backend type
    _backend: core::marker::PhantomData<B>,
}

impl<B: Backend> Applier<B> {
    /// Create a new tensor applier with all views
    pub fn new(views: Vec<TensorSnapshot>) -> Self {
        let views_map: HashMap<String, TensorSnapshot> = views
            .into_iter()
            .map(|view| (view.full_path(), view))
            .collect();

        Self {
            views: views_map,
            path_stack: Vec::new(),
            container_stack: Vec::new(),
            filter: None,
            applied: Vec::new(),
            skipped: HashSet::new(),
            errors: Vec::new(),
            visited_paths: HashSet::new(),
            _backend: core::marker::PhantomData,
        }
    }

    /// Create a new tensor applier with a PathFilter
    pub fn with_filter(views: Vec<TensorSnapshot>, filter: PathFilter) -> Self {
        let views_map: HashMap<String, TensorSnapshot> = views
            .into_iter()
            .map(|view| (view.full_path(), view))
            .collect();

        Self {
            views: views_map,
            path_stack: Vec::new(),
            container_stack: Vec::new(),
            filter: Some(filter),
            applied: Vec::new(),
            skipped: HashSet::new(),
            errors: Vec::new(),
            visited_paths: HashSet::new(),
            _backend: core::marker::PhantomData,
        }
    }

    /// Get the current path in the module hierarchy
    fn current_path(&self) -> String {
        self.path_stack.join(".")
    }

    /// Check if a tensor at the given path should be applied
    fn should_apply(&self, path: &[String], container_stack: &[String]) -> bool {
        // If filter is present, use it; otherwise apply all
        match &self.filter {
            None => true,
            Some(f) => f.matches_with_container_path(path, container_stack),
        }
    }

    /// Convert the applier into an ApplyResult
    pub fn into_result(self) -> ApplyResult {
        // Find unused tensors (in views but not visited)
        let unused: Vec<String> = self
            .views
            .keys()
            .filter(|path| !self.visited_paths.contains(*path) && !self.skipped.contains(*path))
            .cloned()
            .collect();

        // Find missing tensors (visited but not in views)
        let missing: Vec<String> = self
            .visited_paths
            .into_iter()
            .filter(|p| !self.views.contains_key(p) && !self.skipped.contains(p))
            .collect();

        // Convert skipped HashSet to Vec for the result
        let skipped: Vec<String> = self.skipped.into_iter().collect();

        ApplyResult {
            applied: self.applied,
            skipped,
            missing,
            unused,
            errors: self.errors,
        }
    }
}

// Implement ModuleMapper for applying the tensors
impl<B: Backend> ModuleMapper<B> for Applier<B> {
    fn enter_module(&mut self, name: &str, container_type: &str) {
        self.path_stack.push(name.to_string());
        self.container_stack.push(container_type.to_string());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path_stack.pop();
        self.container_stack.pop();
    }

    fn map_float<const D: usize>(&mut self, _id: ParamId, tensor: Tensor<B, D>) -> Tensor<B, D> {
        if self.path_stack.is_empty() {
            return tensor;
        }

        let path = self.current_path();
        self.visited_paths.insert(path.clone());

        if let Some(view) = self.views.get(&path) {
            if !self.should_apply(&self.path_stack, &self.container_stack) {
                self.skipped.insert(path);
                return tensor;
            }

            let data = view.to_data();

            // Validate shape
            let expected_shape = tensor.shape().dims;
            if data.shape != expected_shape {
                self.errors.push(format!(
                    "Shape mismatch for '{}': expected {:?}, found {:?}",
                    path, expected_shape, data.shape
                ));
                return tensor;
            }

            // Apply the tensor using the device from the existing tensor
            let device = tensor.device();
            let new_tensor = Tensor::from_data(data.convert::<B::FloatElem>(), &device);
            self.applied.push(path);
            new_tensor
        } else {
            tensor
        }
    }

    fn map_int<const D: usize>(
        &mut self,
        _id: ParamId,
        tensor: Tensor<B, D, Int>,
    ) -> Tensor<B, D, Int> {
        if self.path_stack.is_empty() {
            return tensor;
        }

        let path = self.current_path();
        self.visited_paths.insert(path.clone());

        if let Some(view) = self.views.get(&path) {
            if !self.should_apply(&self.path_stack, &self.container_stack) {
                self.skipped.insert(path);
                return tensor;
            }

            let data = view.to_data();

            // Validate shape
            let expected_shape = tensor.shape().dims;
            if data.shape != expected_shape {
                self.errors.push(format!(
                    "Shape mismatch for '{}': expected {:?}, found {:?}",
                    path, expected_shape, data.shape
                ));
                return tensor;
            }

            // Apply the tensor using the device from the existing tensor
            let device = tensor.device();
            let new_tensor = Tensor::from_data(data.convert::<B::IntElem>(), &device);
            self.applied.push(path);
            new_tensor
        } else {
            tensor
        }
    }

    fn map_bool<const D: usize>(
        &mut self,
        _id: ParamId,
        tensor: Tensor<B, D, Bool>,
    ) -> Tensor<B, D, Bool> {
        if self.path_stack.is_empty() {
            return tensor;
        }

        let path = self.current_path();
        self.visited_paths.insert(path.clone());

        if let Some(view) = self.views.get(&path) {
            if !self.should_apply(&self.path_stack, &self.container_stack) {
                self.skipped.insert(path);
                return tensor;
            }

            let data = view.to_data();

            // Validate shape
            let expected_shape = tensor.shape().dims;
            if data.shape != expected_shape {
                self.errors.push(format!(
                    "Shape mismatch for '{}': expected {:?}, found {:?}",
                    path, expected_shape, data.shape
                ));
                return tensor;
            }

            // Apply the tensor using the device from the existing tensor
            let device = tensor.device();
            let new_tensor = Tensor::from_data(data.convert::<bool>(), &device);
            self.applied.push(path);
            new_tensor
        } else {
            tensor
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_snapshot::TensorSnapshot;
    use burn_core::module::{Module, Param};
    use burn_tensor::Tensor;

    type TestBackend = burn_ndarray::NdArray;

    #[test]
    fn apply_error_display() {
        let err = ApplyError::ShapeMismatch {
            path: "model.weight".to_string(),
            expected: vec![2, 3],
            found: vec![3, 2],
        };
        let display = format!("{}", err);
        assert!(display.contains("Shape mismatch"));
        assert!(display.contains("model.weight"));
        assert!(display.contains("[2, 3]"));
        assert!(display.contains("[3, 2]"));

        let err = ApplyError::TypeMismatch {
            path: "model.bias".to_string(),
            message: "Expected Float, got Int".to_string(),
        };
        let display = format!("{}", err);
        assert!(display.contains("Type mismatch"));
        assert!(display.contains("model.bias"));
        assert!(display.contains("Expected Float, got Int"));

        let err = ApplyError::PathNotFound {
            path: "missing.tensor".to_string(),
        };
        let display = format!("{}", err);
        assert!(display.contains("not found"));
        assert!(display.contains("missing.tensor"));

        let err = ApplyError::InvalidData {
            path: "corrupted.tensor".to_string(),
            reason: "Data corruption detected".to_string(),
        };
        let display = format!("{}", err);
        assert!(display.contains("Invalid data"));
        assert!(display.contains("corrupted.tensor"));
        assert!(display.contains("Data corruption detected"));
    }

    #[test]
    fn apply_result_is_success() {
        let result = ApplyResult {
            applied: vec!["tensor1".to_string(), "tensor2".to_string()],
            skipped: vec![],
            missing: vec![],
            unused: vec![],
            errors: vec![],
        };
        assert!(result.is_success());
        assert_eq!(result.total_processed(), 2);

        let result_with_errors = ApplyResult {
            applied: vec!["tensor1".to_string()],
            skipped: vec![],
            missing: vec![],
            unused: vec![],
            errors: vec!["Error applying tensor2".to_string()],
        };
        assert!(!result_with_errors.is_success());
        assert_eq!(result_with_errors.total_processed(), 2);
    }

    #[test]
    fn apply_result_display() {
        let result = ApplyResult {
            applied: vec!["tensor1".to_string(), "tensor2".to_string()],
            skipped: vec!["filtered1".to_string()],
            missing: vec!["missing1".to_string()],
            unused: vec!["unused1".to_string()],
            errors: vec!["Error: shape mismatch".to_string()],
        };

        let display = format!("{}", result);
        assert!(display.contains("Applied: 2 tensors"));
        assert!(display.contains("Skipped: 1 tensors"));
        assert!(display.contains("Missing: 1 tensors"));
        assert!(display.contains("Unused: 1 tensors"));
        assert!(display.contains("Errors: 1 tensors"));
        assert!(display.contains("Error: shape mismatch"));
    }

    #[derive(Module, Debug)]
    struct SimpleModule<B: Backend> {
        weight: Param<Tensor<B, 2>>,
        bias: Param<Tensor<B, 1>>,
    }

    impl<B: Backend> SimpleModule<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                weight: Param::from_data([[1.0, 2.0], [3.0, 4.0]], device),
                bias: Param::from_data([0.1, 0.2], device),
            }
        }

        fn zeros(device: &B::Device) -> Self {
            Self {
                weight: Param::from_tensor(Tensor::zeros([2, 2], device)),
                bias: Param::from_tensor(Tensor::zeros([2], device)),
            }
        }
    }

    #[test]
    fn tensor_applier_basic() {
        let device = Default::default();

        // Create source module and extract views
        let source = SimpleModule::<TestBackend>::new(&device);
        let views = vec![
            TensorSnapshot::from_float(
                &source.weight.val(),
                vec!["weight".to_string()],
                vec!["SimpleModule".to_string()],
                ParamId::new(),
            ),
            TensorSnapshot::from_float(
                &source.bias.val(),
                vec!["bias".to_string()],
                vec!["SimpleModule".to_string()],
                ParamId::new(),
            ),
        ];

        // Create target module (zeros) and apply views
        let mut target = SimpleModule::<TestBackend>::zeros(&device);
        let mut applier = Applier::<TestBackend>::new(views);
        target = target.map(&mut applier);

        let result = applier.into_result();
        assert!(result.is_success());
        assert_eq!(result.applied.len(), 2);
        assert!(result.applied.contains(&"weight".to_string()));
        assert!(result.applied.contains(&"bias".to_string()));
        assert_eq!(result.errors.len(), 0);
        assert_eq!(result.missing.len(), 0);
        assert_eq!(result.unused.len(), 0);

        // Verify data was actually applied
        let weight_data = target.weight.val().to_data();
        assert_eq!(
            weight_data.to_vec::<f32>().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
        let bias_data = target.bias.val().to_data();
        assert_eq!(bias_data.to_vec::<f32>().unwrap(), vec![0.1, 0.2]);
    }

    #[test]
    #[cfg(target_has_atomic = "ptr")]
    fn tensor_applier_with_filter() {
        let device = Default::default();

        // Create source module and extract views
        let source = SimpleModule::<TestBackend>::new(&device);
        let views = vec![
            TensorSnapshot::from_float(
                &source.weight.val(),
                vec!["weight".to_string()],
                vec!["SimpleModule".to_string()],
                ParamId::new(),
            ),
            TensorSnapshot::from_float(
                &source.bias.val(),
                vec!["bias".to_string()],
                vec!["SimpleModule".to_string()],
                ParamId::new(),
            ),
        ];

        // Apply with filter that only accepts "weight"
        let filter = PathFilter::new().with_full_path("weight");
        let mut target = SimpleModule::<TestBackend>::zeros(&device);
        let mut applier = Applier::<TestBackend>::with_filter(views, filter);
        target = target.map(&mut applier);

        let result = applier.into_result();
        assert!(result.is_success());
        assert_eq!(result.applied.len(), 1);
        assert!(result.applied.contains(&"weight".to_string()));
        assert_eq!(result.skipped.len(), 1);
        assert!(result.skipped.contains(&"bias".to_string()));

        // Verify only weight was applied
        let weight_data = target.weight.val().to_data();
        assert_eq!(
            weight_data.to_vec::<f32>().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
        let bias_data = target.bias.val().to_data();
        assert_eq!(bias_data.to_vec::<f32>().unwrap(), vec![0.0, 0.0]); // Still zeros
    }

    #[test]
    fn tensor_applier_shape_mismatch() {
        let device = Default::default();

        // Create a view with wrong shape
        let wrong_tensor = Tensor::<TestBackend, 2>::from_data([[1.0, 2.0, 3.0]], &device);
        let views = vec![TensorSnapshot::from_float(
            &wrong_tensor,
            vec!["weight".to_string()],
            vec!["SimpleModule".to_string()],
            ParamId::new(),
        )];

        // Try to apply to module with different shape
        let target = SimpleModule::<TestBackend>::zeros(&device);
        let mut applier = Applier::<TestBackend>::new(views);
        let _ = target.map(&mut applier);

        let result = applier.into_result();
        assert!(!result.is_success());
        assert_eq!(result.applied.len(), 0);
        assert_eq!(result.errors.len(), 1);
        assert!(result.errors[0].contains("Shape mismatch"));
        assert!(result.errors[0].contains("weight"));
    }

    #[test]
    fn tensor_applier_missing_tensors() {
        let device = Default::default();

        // Create views with only partial tensors
        let source = SimpleModule::<TestBackend>::new(&device);
        let views = vec![
            TensorSnapshot::from_float(
                &source.weight.val(),
                vec!["weight".to_string()],
                vec!["SimpleModule".to_string()],
                ParamId::new(),
            ),
            // bias is missing
        ];

        // Apply to module that expects both tensors
        let target = SimpleModule::<TestBackend>::zeros(&device);
        let mut applier = Applier::<TestBackend>::new(views);
        let _ = target.map(&mut applier);

        let result = applier.into_result();
        assert!(result.is_success()); // No errors, just missing
        assert_eq!(result.applied.len(), 1);
        assert_eq!(result.missing.len(), 1);
        assert!(result.missing.contains(&"bias".to_string()));
    }

    #[test]
    fn tensor_applier_unused_tensors() {
        let device = Default::default();

        // Create views with extra tensors
        let source = SimpleModule::<TestBackend>::new(&device);
        let extra_tensor = Tensor::<TestBackend, 2>::from_data([[5.0, 6.0], [7.0, 8.0]], &device);
        let views = vec![
            TensorSnapshot::from_float(
                &source.weight.val(),
                vec!["weight".to_string()],
                vec!["SimpleModule".to_string()],
                ParamId::new(),
            ),
            TensorSnapshot::from_float(
                &source.bias.val(),
                vec!["bias".to_string()],
                vec!["SimpleModule".to_string()],
                ParamId::new(),
            ),
            TensorSnapshot::from_float(
                &extra_tensor,
                vec!["extra".to_string()],
                vec!["SimpleModule".to_string()],
                ParamId::new(),
            ),
        ];

        // Apply to module that doesn't have "extra"
        let target = SimpleModule::<TestBackend>::zeros(&device);
        let mut applier = Applier::<TestBackend>::new(views);
        let _ = target.map(&mut applier);

        let result = applier.into_result();
        assert!(result.is_success());
        assert_eq!(result.applied.len(), 2);
        assert_eq!(result.unused.len(), 1);
        assert!(result.unused.contains(&"extra".to_string()));
    }

    #[derive(Module, Debug)]
    struct NestedModule<B: Backend> {
        layer1: SimpleModule<B>,
        layer2: SimpleModule<B>,
    }

    impl<B: Backend> NestedModule<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                layer1: SimpleModule::new(device),
                layer2: SimpleModule::new(device),
            }
        }

        fn zeros(device: &B::Device) -> Self {
            Self {
                layer1: SimpleModule::zeros(device),
                layer2: SimpleModule::zeros(device),
            }
        }
    }

    #[test]
    fn tensor_applier_nested_modules() {
        let device = Default::default();

        // Create source with nested structure
        let source = NestedModule::<TestBackend>::new(&device);
        let views = vec![
            TensorSnapshot::from_float(
                &source.layer1.weight.val(),
                vec!["layer1".to_string(), "weight".to_string()],
                vec!["NestedModule".to_string(), "SimpleModule".to_string()],
                ParamId::new(),
            ),
            TensorSnapshot::from_float(
                &source.layer1.bias.val(),
                vec!["layer1".to_string(), "bias".to_string()],
                vec!["NestedModule".to_string(), "SimpleModule".to_string()],
                ParamId::new(),
            ),
            TensorSnapshot::from_float(
                &source.layer2.weight.val(),
                vec!["layer2".to_string(), "weight".to_string()],
                vec!["NestedModule".to_string(), "SimpleModule".to_string()],
                ParamId::new(),
            ),
            TensorSnapshot::from_float(
                &source.layer2.bias.val(),
                vec!["layer2".to_string(), "bias".to_string()],
                vec!["NestedModule".to_string(), "SimpleModule".to_string()],
                ParamId::new(),
            ),
        ];

        // Apply to nested target
        let target = NestedModule::<TestBackend>::zeros(&device);
        let mut applier = Applier::<TestBackend>::new(views);
        let _ = target.map(&mut applier);

        let result = applier.into_result();
        assert!(result.is_success());
        assert_eq!(result.applied.len(), 4);
        assert!(result.applied.contains(&"layer1.weight".to_string()));
        assert!(result.applied.contains(&"layer1.bias".to_string()));
        assert!(result.applied.contains(&"layer2.weight".to_string()));
        assert!(result.applied.contains(&"layer2.bias".to_string()));
    }

    #[test]
    #[cfg(target_has_atomic = "ptr")]
    fn tensor_applier_regex_filter() {
        let device = Default::default();

        let source = NestedModule::<TestBackend>::new(&device);
        let views = vec![
            TensorSnapshot::from_float(
                &source.layer1.weight.val(),
                vec!["layer1".to_string(), "weight".to_string()],
                vec!["NestedModule".to_string(), "SimpleModule".to_string()],
                ParamId::new(),
            ),
            TensorSnapshot::from_float(
                &source.layer1.bias.val(),
                vec!["layer1".to_string(), "bias".to_string()],
                vec!["NestedModule".to_string(), "SimpleModule".to_string()],
                ParamId::new(),
            ),
            TensorSnapshot::from_float(
                &source.layer2.weight.val(),
                vec!["layer2".to_string(), "weight".to_string()],
                vec!["NestedModule".to_string(), "SimpleModule".to_string()],
                ParamId::new(),
            ),
            TensorSnapshot::from_float(
                &source.layer2.bias.val(),
                vec!["layer2".to_string(), "bias".to_string()],
                vec!["NestedModule".to_string(), "SimpleModule".to_string()],
                ParamId::new(),
            ),
        ];

        // Filter to only apply layer1 tensors
        let filter = PathFilter::new().with_regex(r"^layer1\..*");
        let target = NestedModule::<TestBackend>::zeros(&device);
        let mut applier = Applier::<TestBackend>::with_filter(views, filter);

        let _ = target.map(&mut applier);

        let result = applier.into_result();
        assert!(result.is_success());
        assert_eq!(result.applied.len(), 2);
        assert!(result.applied.contains(&"layer1.weight".to_string()));
        assert!(result.applied.contains(&"layer1.bias".to_string()));
        assert_eq!(result.skipped.len(), 2);
        assert!(result.skipped.contains(&"layer2.weight".to_string()));
        assert!(result.skipped.contains(&"layer2.bias".to_string()));
    }

    #[test]
    #[cfg(feature = "std")]
    fn regex_error_conversion() {
        // Test that regex errors convert properly
        #[allow(clippy::invalid_regex)]
        let regex_err = regex::Regex::new("[invalid").unwrap_err();
        let apply_err: ApplyError = regex_err.into();
        match apply_err {
            ApplyError::RegexError(_) => (),
            _ => panic!("Expected RegexError variant"),
        }
    }
}
