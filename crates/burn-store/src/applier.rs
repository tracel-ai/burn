//! Applier that correctly applies tensor snapshots with adapter support

use alloc::boxed::Box;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use hashbrown::{HashMap, HashSet};

use burn_core::module::{ModuleMapper, Param};
use burn_tensor::{Bool, DType, Int, Shape, Tensor, backend::Backend};

use crate::{ModuleAdapter, PathFilter, TensorSnapshot};

/// Error types that can occur during tensor application
#[derive(Debug, Clone)]
pub enum ApplyError {
    /// Shape mismatch between expected and actual tensor
    ShapeMismatch {
        /// Path of the tensor
        path: String,
        /// Expected shape
        expected: Vec<usize>,
        /// Found shape
        found: Vec<usize>,
    },
    /// Data type mismatch between expected and actual tensor
    DTypeMismatch {
        /// Path of the tensor
        path: String,
        /// Expected data type
        expected: DType,
        /// Found data type
        found: DType,
    },
    /// Error from adapter transformation
    AdapterError {
        /// Path of the tensor
        path: String,
        /// Error message
        message: String,
    },
    /// Error loading tensor data
    LoadError {
        /// Path of the tensor
        path: String,
        /// Error message
        message: String,
    },
}

impl core::fmt::Display for ApplyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ShapeMismatch {
                path,
                expected,
                found,
            } => {
                write!(
                    f,
                    "Shape mismatch for '{}': expected {:?}, found {:?}",
                    path, expected, found
                )
            }
            Self::DTypeMismatch {
                path,
                expected,
                found,
            } => {
                write!(
                    f,
                    "DType mismatch for '{}': expected {:?}, found {:?}",
                    path, expected, found
                )
            }
            Self::AdapterError { path, message } => {
                write!(f, "Adapter error for '{}': {}", path, message)
            }
            Self::LoadError { path, message } => {
                write!(f, "Load error for '{}': {}", path, message)
            }
        }
    }
}

/// Result of applying tensor snapshots to a module
#[derive(Debug, Clone)]
pub struct ApplyResult {
    /// Successfully applied tensor paths
    pub applied: Vec<String>,
    /// Skipped tensor paths (due to filter)
    pub skipped: Vec<String>,
    /// Missing tensor paths (in module but not in snapshots)
    pub missing: Vec<String>,
    /// Unused tensor paths (in snapshots but not in module)
    pub unused: Vec<String>,
    /// Errors encountered during application
    pub errors: Vec<ApplyError>,
}

impl ApplyResult {
    /// Check if the apply operation was successful (no errors)
    /// Note: Missing tensors are not considered errors when allow_partial is true
    pub fn is_success(&self) -> bool {
        self.errors.is_empty()
    }
}

/// Applier that applies tensor snapshots to module parameters
/// with proper adapter support using container type information
pub struct Applier<B: Backend> {
    /// Map of tensor paths to their snapshots
    snapshots: HashMap<String, TensorSnapshot>,
    /// Current path in the module hierarchy
    path_stack: Vec<String>,
    /// Current container type stack in the module hierarchy
    container_stack: Vec<String>,
    /// Optional filter for selective application
    filter: Option<PathFilter>,
    /// Optional adapter to transform tensors based on container types
    adapter: Option<Box<dyn ModuleAdapter>>,
    /// Successfully applied tensor paths
    applied: Vec<String>,
    /// Skipped tensor paths
    skipped: HashSet<String>,
    /// Errors encountered during application
    errors: Vec<ApplyError>,
    /// Track visited paths to find missing tensors
    visited_paths: HashSet<String>,
    /// Phantom data for backend type
    _backend: core::marker::PhantomData<B>,
}

impl<B: Backend> Applier<B> {
    /// Create a new applier with snapshots, optional filter, and optional adapter
    ///
    /// # Arguments
    ///
    /// * `views` - A vector of TensorSnapshot objects to apply
    /// * `filter` - An optional [`PathFilter`] to determine which tensors to apply.
    ///   When `None`, all available tensors are applied.
    /// * `adapter` - Optional adapter to transform tensors based on container types
    pub fn new(
        views: Vec<TensorSnapshot>,
        filter: Option<PathFilter>,
        adapter: Option<Box<dyn ModuleAdapter>>,
    ) -> Self {
        let views_map: HashMap<String, TensorSnapshot> = views
            .into_iter()
            .map(|view| (view.full_path(), view))
            .collect();

        Self {
            snapshots: views_map,
            path_stack: Vec::new(),
            container_stack: Vec::new(),
            filter,
            adapter,
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

    /// Check if a tensor should be applied based on filter
    fn should_apply(&self) -> bool {
        match &self.filter {
            None => true,
            Some(f) => f.matches_with_container_path(&self.path_stack, &self.container_stack),
        }
    }

    /// Apply adapter to a snapshot using current container information
    fn adapt_snapshot(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
        if let Some(ref adapter) = self.adapter {
            // Create a snapshot with proper container information from module traversal
            let snapshot_with_context = TensorSnapshot::from_closure(
                snapshot.clone_data_fn(),
                snapshot.dtype,
                snapshot.shape.clone(),
                self.path_stack.clone(), // Use current path from traversal
                self.container_stack.clone(), // Use current container types!
                snapshot.tensor_id.unwrap_or_default(),
            );

            // Apply adapter with full context
            return adapter.adapt(&snapshot_with_context);
        }
        snapshot.clone()
    }

    /// Convert the applier into a result
    pub fn into_result(self) -> ApplyResult {
        let unused: Vec<String> = self
            .snapshots
            .keys()
            .filter(|path| !self.visited_paths.contains(*path) && !self.skipped.contains(*path))
            .cloned()
            .collect();

        let missing: Vec<String> = self
            .visited_paths
            .into_iter()
            .filter(|p| !self.snapshots.contains_key(p) && !self.skipped.contains(p))
            .collect();

        ApplyResult {
            applied: self.applied,
            skipped: self.skipped.into_iter().collect(),
            missing,
            unused,
            errors: self.errors,
        }
    }

    /// Apply a tensor snapshot with shape validation
    /// Returns None if snapshot not found, filtered, or validation fails
    fn apply_tensor<const D: usize, K>(
        &mut self,
        target_device: &B::Device,
        target_shape: Shape,
    ) -> Option<Tensor<B, D, K>>
    where
        K: burn_tensor::TensorKind<B>,
        K: burn_tensor::BasicOps<B>,
    {
        let path = self.current_path();
        self.visited_paths.insert(path.clone());

        // Check if we have a snapshot for this path
        let snapshot = match self.snapshots.get(&path) {
            Some(s) => s,
            None => {
                // No snapshot available - signal caller not to apply
                return None;
            }
        };

        // Check if we should apply based on filter
        if !self.should_apply() {
            self.skipped.insert(path.clone());
            return None;
        }

        // Apply adapter with current container context
        let adapted_snapshot = self.adapt_snapshot(snapshot);
        let data = match adapted_snapshot.to_data() {
            Ok(data) => data,
            Err(e) => {
                self.errors.push(ApplyError::LoadError {
                    path: path.clone(),
                    message: format!("Failed to load tensor data: {:?}", e),
                });
                return None; // Signal caller to fall back to initialization
            }
        };

        // Validate shape
        if data.shape != target_shape.dims {
            self.errors.push(ApplyError::ShapeMismatch {
                path: path.clone(),
                expected: target_shape.dims,
                found: data.shape.clone(),
            });
            return None; // Signal caller to fall back to initialization
        }

        self.applied.push(path);
        Some(Tensor::from_data(data, target_device))
    }
}

impl<B: Backend> ModuleMapper<B> for Applier<B> {
    fn enter_module(&mut self, name: &str, container_type: &str) {
        self.path_stack.push(name.to_string());
        self.container_stack.push(container_type.to_string());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path_stack.pop();
        self.container_stack.pop();
    }

    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let param_id = param.id;
        let target_device = param.lazy_device();
        let target_shape = param.lazy_shape();

        // Try to apply snapshot with shape validation
        match self.apply_tensor(&target_device, target_shape) {
            Some(tensor) => {
                // We have a tensor to apply - load it
                param.load(tensor, param_id)
            }
            None => {
                // No snapshot, filtered, or validation failed - return param unchanged
                param
            }
        }
    }

    fn map_int<const D: usize>(
        &mut self,
        param: Param<Tensor<B, D, Int>>,
    ) -> Param<Tensor<B, D, Int>> {
        let param_id = param.id;
        let target_device = param.lazy_device();
        let target_shape = param.lazy_shape();

        // Try to apply snapshot with shape validation
        match self.apply_tensor(&target_device, target_shape) {
            Some(tensor) => {
                // We have a tensor to apply - load it
                param.load(tensor, param_id)
            }
            None => {
                // No snapshot, filtered, or validation failed - return param unchanged
                param
            }
        }
    }

    fn map_bool<const D: usize>(
        &mut self,
        param: Param<Tensor<B, D, Bool>>,
    ) -> Param<Tensor<B, D, Bool>> {
        let param_id = param.id;
        let target_device = param.lazy_device();
        let target_shape = param.lazy_shape();

        // Try to apply snapshot with shape validation
        match self.apply_tensor(&target_device, target_shape) {
            Some(tensor) => {
                // We have a tensor to apply - load it
                param.load(tensor, param_id)
            }
            None => {
                // No snapshot, filtered, or validation failed - return param unchanged
                param
            }
        }
    }
}

#[cfg(all(test, feature = "std", target_has_atomic = "ptr"))]
mod tests {
    use super::*;
    use burn_core::module::{ModuleMapper, Param, ParamId};
    use burn_tensor::Tensor;

    type TestBackend = burn_ndarray::NdArray;

    #[test]
    fn root_level_parameters() {
        let device = Default::default();

        // Create root-level parameters (not inside any module)
        let weight = Param::<Tensor<TestBackend, 2>>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
        let bias = Param::<Tensor<TestBackend, 1>>::from_data([5.0, 6.0], &device);

        // Create snapshots with root-level paths (single-element path, no nested modules)
        let weight_snapshot = crate::TensorSnapshot::from_data(
            weight.val().to_data(),
            vec!["weight".to_string()], // root-level parameter name
            vec![],                     // no container
            ParamId::new(),
        );

        let bias_snapshot = crate::TensorSnapshot::from_data(
            bias.val().to_data(),
            vec!["bias".to_string()], // root-level parameter name
            vec![],                   // no container
            ParamId::new(),
        );

        // Create applier with root-level snapshots
        let mut applier =
            Applier::<TestBackend>::new(vec![weight_snapshot, bias_snapshot], None, None);

        // Create new params to load into
        let weight_target = Param::initialized(
            ParamId::new(),
            Tensor::<TestBackend, 2>::zeros([2, 2], &device),
        );
        let bias_target = Param::initialized(
            ParamId::new(),
            Tensor::<TestBackend, 1>::zeros([2], &device),
        );

        // Apply using the ModuleMapper interface - simulate module traversal
        // Enter "weight" path (as if we're visiting a field named "weight")
        applier.enter_module("weight", "");
        let weight_loaded = applier.map_float(weight_target);
        applier.exit_module("weight", "");

        // Enter "bias" path (as if we're visiting a field named "bias")
        applier.enter_module("bias", "");
        let bias_loaded = applier.map_float(bias_target);
        applier.exit_module("bias", "");

        // Verify values were loaded
        let weight_data = weight_loaded.val().to_data().to_vec::<f32>().unwrap();
        let bias_data = bias_loaded.val().to_data().to_vec::<f32>().unwrap();

        assert_eq!(weight_data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(bias_data, vec![5.0, 6.0]);

        // Verify applier result
        let result = applier.into_result();
        assert_eq!(result.applied.len(), 2);
        assert_eq!(result.errors.len(), 0);
    }
}
