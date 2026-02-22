//! Applier that correctly applies tensor snapshots with adapter support

use alloc::boxed::Box;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use hashbrown::{HashMap, HashSet};

use burn_core::module::{ModuleMapper, Param};
use burn_tensor::{Bool, Int, Shape, Tensor, backend::Backend};

use crate::apply_result::{ApplyError, ApplyResult};
use crate::{ModuleAdapter, PathFilter, TensorSnapshot};

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
    /// Track visited paths with their container stacks (in dot notation) to find missing tensors
    visited_paths: HashMap<String, String>,
    /// Skip enum variant names when matching paths
    /// When true, "feature.BaseConv.weight" will also try to match "feature.weight"
    skip_enum_variants: bool,
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
    /// * `skip_enum_variants` - Skip enum variant names when matching paths
    pub fn new(
        views: Vec<TensorSnapshot>,
        filter: Option<PathFilter>,
        adapter: Option<Box<dyn ModuleAdapter>>,
        skip_enum_variants: bool,
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
            visited_paths: HashMap::new(),
            skip_enum_variants,
            _backend: core::marker::PhantomData,
        }
    }

    /// Get the current path in the module hierarchy
    fn current_path(&self) -> String {
        self.path_stack.join(".")
    }

    /// Get the current module type (last Struct/Enum in container stack)
    fn current_module_type(&self) -> Option<&str> {
        self.container_stack
            .iter()
            .rev()
            .find(|ct| ct.starts_with("Struct:") || ct.starts_with("Enum:"))
            .map(|s| s.as_str())
    }

    /// Check if a tensor should be applied based on filter
    fn should_apply(&self) -> bool {
        match &self.filter {
            None => true,
            Some(f) => f.matches_with_container_path(&self.path_stack, &self.container_stack),
        }
    }

    /// Convert the applier into a result
    pub fn into_result(self) -> ApplyResult {
        let mut unused: Vec<String> = self
            .snapshots
            .keys()
            .filter(|path| !self.visited_paths.contains_key(*path) && !self.skipped.contains(*path))
            .cloned()
            .collect();
        // Sort for stable output order
        unused.sort();

        // Create a set of successfully applied paths for efficient lookup
        let applied_set: HashSet<String> = self.applied.iter().cloned().collect();

        // Extract paths that have errors - these are not "missing", they were found but had issues
        let errored_paths: HashSet<String> = self
            .errors
            .iter()
            .map(|e| match e {
                ApplyError::ShapeMismatch { path, .. } => path.clone(),
                ApplyError::DTypeMismatch { path, .. } => path.clone(),
                ApplyError::AdapterError { path, .. } => path.clone(),
                ApplyError::LoadError { path, .. } => path.clone(),
            })
            .collect();

        // A path is missing if it was visited but not successfully applied, not skipped, and didn't have an error
        // Store both the path and its container stack (in dot notation)
        let mut missing: Vec<(String, String)> = self
            .visited_paths
            .into_iter()
            .filter(|(p, _)| {
                !applied_set.contains(p) && !self.skipped.contains(p) && !errored_paths.contains(p)
            })
            .collect();
        // Sort for stable output order (by path)
        missing.sort_by(|a, b| a.0.cmp(&b.0));

        // Convert skipped HashSet to sorted Vec for stable output
        let mut skipped: Vec<String> = self.skipped.into_iter().collect();
        skipped.sort();

        ApplyResult {
            applied: self.applied,
            skipped,
            missing,
            unused,
            errors: self.errors,
        }
    }

    /// Apply a tensor snapshot with shape validation and optional adapter transformation
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
        let container_stack_str = self.container_stack.join(".");
        self.visited_paths.insert(path.clone(), container_stack_str);

        // Try to get snapshot with original path first
        let mut snapshot = self.snapshots.get(&path).cloned();

        // If not found and we have an adapter, try alternative parameter names
        if snapshot.is_none()
            && let Some(ref adapter) = self.adapter
            && let Some(module_type) = self.current_module_type()
        {
            // Get alternative name based on current module type (user-defined module only)
            let param_name = self.path_stack.last()?;

            if let Some(alt_name) = adapter.get_alternative_param_name(param_name, module_type) {
                // Build alternative path with parameter name substitution
                let mut alt_path_stack = self.path_stack.clone();
                *alt_path_stack.last_mut().unwrap() = alt_name.clone();
                let alt_path = alt_path_stack.join(".");

                // Try to get snapshot with alternative name
                snapshot = self.snapshots.get(&alt_path).cloned();

                // Don't mark the alternative path as visited - only the original Burn path
                // should be tracked. The alternative path is just for lookup.
            }
        }

        let mut snapshot = snapshot?;

        // Apply adapter transformation using current container_stack context (for data transformation like transpose)
        if let Some(ref adapter) = self.adapter {
            // Create a temporary snapshot with current context for adaptation
            let snapshot_with_context = TensorSnapshot::from_closure(
                snapshot.clone_data_fn(),
                snapshot.dtype,
                snapshot.shape.clone(),
                self.path_stack.clone(),
                self.container_stack.clone(),
                snapshot.tensor_id.unwrap_or_default(),
            );

            // Transform using adapter (handles transpose)
            snapshot = adapter.adapt(&snapshot_with_context);
        }

        // Check if we should apply based on filter
        if !self.should_apply() {
            self.skipped.insert(path.clone());
            return None;
        }

        // Load tensor data
        let data = match snapshot.to_data() {
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
        if data.shape != target_shape {
            self.errors.push(ApplyError::ShapeMismatch {
                path: path.clone(),
                expected: target_shape.clone(),
                found: data.shape.clone(),
            });
            return None; // Signal caller to fall back to initialization
        }

        self.applied.push(path);
        Some(Tensor::from_data_dtype(data, target_device, snapshot.dtype))
    }
}

impl<B: Backend> ModuleMapper<B> for Applier<B> {
    fn enter_module(&mut self, name: &str, container_type: &str) {
        // Always track the container type for proper module type detection
        self.container_stack.push(container_type.to_string());

        // Only add to path if it's not an enum variant (when skip_enum_variants is enabled)
        // This ensures paths are built without enum variant names from the start
        if !self.skip_enum_variants || !container_type.starts_with("Enum:") {
            self.path_stack.push(name.to_string());
        }
    }

    fn exit_module(&mut self, _name: &str, container_type: &str) {
        self.container_stack.pop();

        // Only pop from path if we added it (not an enum variant when skip_enum_variants is enabled)
        if !self.skip_enum_variants || !container_type.starts_with("Enum:") {
            self.path_stack.pop();
        }
    }

    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let param_id = param.id;
        let target_device = param.lazy_device();
        let target_shape = param.lazy_shape();

        // Try to apply snapshot with shape validation
        match self.apply_tensor(&target_device, target_shape) {
            Some(tensor) => {
                // We have a tensor to apply - load it
                param.transform_for_load(tensor, param_id)
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
                param.transform_for_load(tensor, param_id)
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
                param.transform_for_load(tensor, param_id)
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
    use burn_tensor::{DType, Tensor, TensorData};

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
            Applier::<TestBackend>::new(vec![weight_snapshot, bias_snapshot], None, None, false);

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

    /// Test that the applier preserves dtype when loading tensor data.
    /// This is a regression test for the bug where F16 tensors were being
    /// loaded as F32 because `Tensor::from_data` was used instead of
    /// `Tensor::from_data_dtype`.
    #[test]
    fn dtype_preservation_f64() {
        // Use NdArray<f64> backend to properly test F64 dtype preservation
        type TestBackendF64 = burn_ndarray::NdArray<f64>;
        let device = Default::default();

        // Create TensorData with F64 dtype explicitly
        let f64_data = TensorData::new(vec![1.0f64, 2.0, 3.0, 4.0], [2, 2]);
        assert_eq!(f64_data.dtype, DType::F64, "Test setup: data should be F64");

        // Create a snapshot with F64 data
        let snapshot = crate::TensorSnapshot::from_data(
            f64_data.clone(),
            vec!["weight".to_string()],
            vec![],
            ParamId::new(),
        );
        assert_eq!(
            snapshot.dtype,
            DType::F64,
            "Snapshot should preserve F64 dtype"
        );

        // Create applier with the F64 snapshot
        let mut applier = Applier::<TestBackendF64>::new(vec![snapshot], None, None, false);

        // Create target parameter
        let target = Param::initialized(
            ParamId::new(),
            Tensor::<TestBackendF64, 2>::zeros([2, 2], &device),
        );

        // Apply the snapshot
        applier.enter_module("weight", "");
        let loaded = applier.map_float(target);
        applier.exit_module("weight", "");

        // Verify dtype is preserved - this would fail before the fix
        // because the data would be converted to the backend's default FloatElem
        assert_eq!(
            loaded.val().dtype(),
            DType::F64,
            "Loaded tensor should have F64 dtype"
        );

        // Verify data values are correct
        let loaded_data = loaded.val().to_data().to_vec::<f64>().unwrap();
        assert_eq!(loaded_data, vec![1.0, 2.0, 3.0, 4.0]);

        // Verify applier result
        let result = applier.into_result();
        assert_eq!(result.applied.len(), 1);
        assert_eq!(result.errors.len(), 0);
    }

    /// Test that F32 dtype is preserved when loading (verifies we didn't break F32 handling)
    #[test]
    fn dtype_preservation_f32() {
        let device = Default::default();

        // Create TensorData with F32 dtype
        let f32_data = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2]);
        assert_eq!(f32_data.dtype, DType::F32);

        // Create a snapshot with F32 data
        let snapshot = crate::TensorSnapshot::from_data(
            f32_data.clone(),
            vec!["weight".to_string()],
            vec![],
            ParamId::new(),
        );
        assert_eq!(snapshot.dtype, DType::F32);

        // Create applier with the F32 snapshot
        let mut applier = Applier::<TestBackend>::new(vec![snapshot], None, None, false);

        // Create target parameter
        let target = Param::initialized(
            ParamId::new(),
            Tensor::<TestBackend, 2>::zeros([2, 2], &device),
        );

        // Apply the snapshot
        applier.enter_module("weight", "");
        let loaded = applier.map_float(target);
        applier.exit_module("weight", "");

        // Verify dtype is F32
        assert_eq!(loaded.val().dtype(), DType::F32);

        // Verify data values
        let loaded_data = loaded.val().to_data().to_vec::<f32>().unwrap();
        assert_eq!(loaded_data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    /// Test that F16 dtype is correctly preserved in TensorSnapshot.
    ///
    /// Note: Full F16 tensor loading requires a backend that supports F16
    /// (e.g., CUDA, WebGPU). The NdArray backend does not support F16.
    /// This test verifies that the snapshot correctly preserves F16 dtype,
    /// which is the key part of the dtype preservation fix.
    #[test]
    fn dtype_preservation_f16_snapshot() {
        use half::f16;

        // Create TensorData with F16 dtype using the half crate
        let f16_values: Vec<f16> = vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
        ];
        let f16_data = TensorData::new(f16_values.clone(), [2, 2]);
        assert_eq!(
            f16_data.dtype,
            DType::F16,
            "TensorData should have F16 dtype"
        );

        // Create a snapshot with F16 data
        let snapshot = crate::TensorSnapshot::from_data(
            f16_data.clone(),
            vec!["weight".to_string()],
            vec![],
            ParamId::new(),
        );

        // Verify snapshot preserves F16 dtype
        assert_eq!(
            snapshot.dtype,
            DType::F16,
            "TensorSnapshot should preserve F16 dtype"
        );

        // Verify the data can be retrieved with correct dtype
        let retrieved_data = snapshot.to_data().expect("Should be able to retrieve data");
        assert_eq!(
            retrieved_data.dtype,
            DType::F16,
            "Retrieved data should have F16 dtype"
        );

        // Verify the actual values are preserved
        let retrieved_values: Vec<f16> = retrieved_data
            .to_vec()
            .expect("Should be able to convert to f16 vec");
        assert_eq!(
            retrieved_values, f16_values,
            "F16 values should be preserved"
        );

        // Note: To fully test F16 tensor creation, you would need a backend
        // that supports F16 (like CUDA or WebGPU). The applier fix ensures
        // that `Tensor::from_data_dtype(data, device, snapshot.dtype)` is
        // called with DType::F16, which will correctly create an F16 tensor
        // on backends that support it.
    }

    /// Test that BF16 dtype is correctly preserved in TensorSnapshot.
    #[test]
    fn dtype_preservation_bf16_snapshot() {
        use half::bf16;

        // Create TensorData with BF16 dtype
        let bf16_values: Vec<bf16> = vec![
            bf16::from_f32(1.0),
            bf16::from_f32(2.0),
            bf16::from_f32(3.0),
            bf16::from_f32(4.0),
        ];
        let bf16_data = TensorData::new(bf16_values.clone(), [2, 2]);
        assert_eq!(
            bf16_data.dtype,
            DType::BF16,
            "TensorData should have BF16 dtype"
        );

        // Create a snapshot with BF16 data
        let snapshot = crate::TensorSnapshot::from_data(
            bf16_data.clone(),
            vec!["weight".to_string()],
            vec![],
            ParamId::new(),
        );

        // Verify snapshot preserves BF16 dtype
        assert_eq!(
            snapshot.dtype,
            DType::BF16,
            "TensorSnapshot should preserve BF16 dtype"
        );

        // Verify the data can be retrieved with correct dtype
        let retrieved_data = snapshot.to_data().expect("Should be able to retrieve data");
        assert_eq!(
            retrieved_data.dtype,
            DType::BF16,
            "Retrieved data should have BF16 dtype"
        );

        // Verify the actual values are preserved
        let retrieved_values: Vec<bf16> = retrieved_data
            .to_vec()
            .expect("Should be able to convert to bf16 vec");
        assert_eq!(
            retrieved_values, bf16_values,
            "BF16 values should be preserved"
        );
    }
}
