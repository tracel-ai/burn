//! Applier that correctly applies tensor snapshots with adapter support

use alloc::boxed::Box;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use hashbrown::{HashMap, HashSet};

use burn_core::module::{ModuleMapper, Param, ParamId};
use burn_tensor::{Bool, DType, Int, Tensor, backend::Backend};

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

    /// Apply a tensor snapshot to the current tensor (generic over tensor kind)
    /// If tensor is None (uninitialized parameter), skip shape and dtype validation
    fn apply_tensor<const D: usize, K>(
        &mut self,
        tensor: Option<Tensor<B, D, K>>,
        device: &B::Device,
    ) -> Tensor<B, D, K>
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
                return match tensor {
                    Some(t) => t,
                    None => panic!("Cannot create uninitialized tensor without snapshot"),
                };
            }
        };

        // Check if we should apply based on filter
        if !self.should_apply() {
            self.skipped.insert(path);
            return match tensor {
                Some(t) => t,
                None => panic!("Cannot skip applying to uninitialized tensor"),
            };
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
                return match tensor {
                    Some(t) => t,
                    None => panic!("Cannot recover from load error for uninitialized tensor"),
                };
            }
        };

        // Validate shape and dtype only if tensor is initialized
        if let Some(ref t) = tensor {
            let expected_shape = t.shape().dims;
            if data.shape != expected_shape {
                self.errors.push(ApplyError::ShapeMismatch {
                    path: path.clone(),
                    expected: expected_shape,
                    found: data.shape.clone(),
                });
                return t.clone();
            }

            let expected_dtype = t.dtype();
            if data.dtype != expected_dtype {
                self.errors.push(ApplyError::DTypeMismatch {
                    path: path.clone(),
                    expected: expected_dtype,
                    found: data.dtype,
                });
                return t.clone();
            }
        }

        self.applied.push(path);
        Tensor::from_data(data, device)
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
        if self.path_stack.is_empty() {
            return param;
        }

        let param_id = param.id;

        let applied_tensor = if param.is_initialized() {
            self.apply_tensor(Some(param.val()), &param.device())
        } else {
            self.apply_tensor::<D, _>(None, &param.device())
        };

        param.load(applied_tensor, param_id)
    }

    fn map_int<const D: usize>(
        &mut self,
        param: Param<Tensor<B, D, Int>>,
    ) -> Param<Tensor<B, D, Int>> {
        if self.path_stack.is_empty() {
            return param;
        }

        let param_id = param.id;

        let applied_tensor = if param.is_initialized() {
            self.apply_tensor(Some(param.val()), &param.device())
        } else {
            self.apply_tensor::<D, _>(None, &param.device())
        };

        param.load(applied_tensor, param_id)
    }

    fn map_bool<const D: usize>(
        &mut self,
        param: Param<Tensor<B, D, Bool>>,
    ) -> Param<Tensor<B, D, Bool>> {
        if self.path_stack.is_empty() {
            return param;
        }

        let param_id = param.id;

        let applied_tensor = if param.is_initialized() {
            self.apply_tensor(Some(param.val()), &param.device())
        } else {
            self.apply_tensor::<D, _>(None, &param.device())
        };

        param.load(applied_tensor, param_id)
    }
}
