use alloc::boxed::Box;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;
use hashbrown::HashMap;

#[cfg(target_has_atomic = "ptr")]
use regex::Regex;

use burn_tensor::{Bool, Int, Tensor, backend::Backend};

use crate::persist::TensorView;
use crate::module::{ModuleMapper, ParamId};

/// Type alias for the predicate function used to filter tensor paths.
type TensorPathPredicate = Box<dyn Fn(&str) -> bool>;

/// Error types for import operations
#[derive(Debug)]
pub enum ImportError {
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
    #[cfg(target_has_atomic = "ptr")]
    RegexError(regex::Error),
    /// Generic error
    Other(String),
}

impl fmt::Display for ImportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ImportError::ShapeMismatch {
                path,
                expected,
                found,
            } => write!(
                f,
                "Shape mismatch for tensor '{}': expected {:?}, found {:?}",
                path, expected, found
            ),
            ImportError::TypeMismatch { path, message } => {
                write!(f, "Type mismatch for tensor '{}': {}", path, message)
            }
            ImportError::PathNotFound { path } => {
                write!(f, "Tensor path '{}' not found in module", path)
            }
            ImportError::InvalidData { path, reason } => {
                write!(f, "Invalid data for tensor '{}': {}", path, reason)
            }
            #[cfg(target_has_atomic = "ptr")]
            ImportError::RegexError(e) => write!(f, "Regex error: {}", e),
            ImportError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ImportError {}

#[cfg(target_has_atomic = "ptr")]
impl From<regex::Error> for ImportError {
    fn from(err: regex::Error) -> Self {
        ImportError::RegexError(err)
    }
}

/// Result of an import operation
#[derive(Debug, Clone)]
pub struct ImportResult {
    /// Successfully applied tensor paths
    pub applied: Vec<String>,
    /// Paths that were filtered out (not attempted)
    pub skipped: Vec<String>,
    /// Paths in module but not in sources
    pub missing: Vec<String>,
    /// Paths in sources but not found in module
    pub unused: Vec<String>,
    /// Errors encountered during import
    pub errors: Vec<String>,
}

impl ImportResult {
    /// Check if the import was successful (no errors)
    pub fn is_success(&self) -> bool {
        self.errors.is_empty()
    }

    /// Get the total number of tensors processed
    pub fn total_processed(&self) -> usize {
        self.applied.len() + self.errors.len()
    }
}

impl fmt::Display for ImportResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Import Result:")?;
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
/// from TensorViews to the corresponding tensors in the module.
pub struct TensorApplier<B: Backend> {
    /// Map of tensor paths to their views
    views: HashMap<String, TensorView>,
    /// Current path in the module hierarchy
    path_stack: Vec<String>,
    /// Regex filters for selective import
    #[cfg(target_has_atomic = "ptr")]
    filters: Option<Vec<Regex>>,
    /// Custom predicate for filtering
    predicate: Option<TensorPathPredicate>,
    /// Successfully applied tensor paths
    applied: Vec<String>,
    /// Skipped tensor paths (due to filtering)
    skipped: Vec<String>,
    /// Errors encountered during application
    errors: Vec<String>,
    /// Track visited paths to find missing tensors
    visited_paths: Vec<String>,
    /// Phantom data for backend type
    _backend: core::marker::PhantomData<B>,
}

impl<B: Backend> TensorApplier<B> {
    /// Create a new tensor applier with all views
    pub fn new(views: HashMap<String, TensorView>) -> Self {
        Self {
            views,
            path_stack: Vec::new(),
            #[cfg(target_has_atomic = "ptr")]
            filters: None,
            predicate: None,
            applied: Vec::new(),
            skipped: Vec::new(),
            errors: Vec::new(),
            visited_paths: Vec::new(),
            _backend: core::marker::PhantomData,
        }
    }

    /// Create a new tensor applier with regex filtering
    #[cfg(target_has_atomic = "ptr")]
    pub fn with_filter<I, S>(
        views: HashMap<String, TensorView>,
        patterns: I,
    ) -> Result<Self, ImportError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut filters = Vec::new();
        for pattern in patterns {
            filters.push(Regex::new(pattern.as_ref())?);
        }

        Ok(Self {
            views,
            path_stack: Vec::new(),
            filters: Some(filters),
            predicate: None,
            applied: Vec::new(),
            skipped: Vec::new(),
            errors: Vec::new(),
            visited_paths: Vec::new(),
            _backend: core::marker::PhantomData,
        })
    }

    /// Create a new tensor applier with a custom predicate
    pub fn with_predicate<F>(views: HashMap<String, TensorView>, predicate: F) -> Self
    where
        F: Fn(&str) -> bool + 'static,
    {
        Self {
            views,
            path_stack: Vec::new(),
            #[cfg(target_has_atomic = "ptr")]
            filters: None,
            predicate: Some(Box::new(predicate)),
            applied: Vec::new(),
            skipped: Vec::new(),
            errors: Vec::new(),
            visited_paths: Vec::new(),
            _backend: core::marker::PhantomData,
        }
    }

    /// Get the current path in the module hierarchy
    fn current_path(&self) -> String {
        self.path_stack.join(".")
    }

    /// Check if a tensor at the given path should be imported
    fn should_import(&self, path: &str) -> bool {
        // First check the predicate if one is provided
        if let Some(ref predicate) = self.predicate {
            return predicate(path);
        }

        // Otherwise check regex filters
        #[cfg(target_has_atomic = "ptr")]
        {
            match &self.filters {
                None => true, // No filters means import all
                Some(filters) if filters.is_empty() => true,
                Some(filters) => {
                    // OR union - import if ANY filter matches
                    filters.iter().any(|filter| filter.is_match(path))
                }
            }
        }
        #[cfg(not(target_has_atomic = "ptr"))]
        {
            true // Without regex support, import all tensors
        }
    }

    /// Convert the applier into an ImportResult
    pub fn into_result(self) -> ImportResult {
        // Find unused tensors (in views but not visited)
        let unused: Vec<String> = self
            .views
            .keys()
            .filter(|k| !self.visited_paths.contains(k) && !self.skipped.contains(k))
            .cloned()
            .collect();

        // Find missing tensors (visited but not in views)
        let missing: Vec<String> = self
            .visited_paths
            .into_iter()
            .filter(|p| !self.views.contains_key(p) && !self.skipped.contains(p))
            .collect();

        ImportResult {
            applied: self.applied,
            skipped: self.skipped,
            missing,
            unused,
            errors: self.errors,
        }
    }
}

// Implement ModuleMapper for applying the tensors
impl<B: Backend> ModuleMapper<B> for TensorApplier<B> {
    fn enter_module(&mut self, name: &str) {
        self.path_stack.push(name.to_string());
    }

    fn exit_module(&mut self, _name: &str) {
        self.path_stack.pop();
    }

    fn map_float<const D: usize>(&mut self, _id: ParamId, tensor: Tensor<B, D>) -> Tensor<B, D> {
        let path = self.current_path();
        if path.is_empty() {
            return tensor;
        }

        self.visited_paths.push(path.clone());

        if let Some(view) = self.views.get(&path) {
            if !self.should_import(&path) {
                self.skipped.push(path);
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
        let path = self.current_path();
        if path.is_empty() {
            return tensor;
        }

        self.visited_paths.push(path.clone());

        if let Some(view) = self.views.get(&path) {
            if !self.should_import(&path) {
                self.skipped.push(path);
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
        let path = self.current_path();
        if path.is_empty() {
            return tensor;
        }

        self.visited_paths.push(path.clone());

        if let Some(view) = self.views.get(&path) {
            if !self.should_import(&path) {
                self.skipped.push(path);
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
