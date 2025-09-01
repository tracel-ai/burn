use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use burn_tensor::DType;
use core::fmt;
use hashbrown::HashMap;

use crate::module::export::TensorView;

/// Error types for tensor reading operations
#[derive(Debug)]
pub enum ReaderError {
    /// Tensor not found at the specified path
    TensorNotFound(String),
    /// IO error during reading
    Io(alloc::string::String),
    /// Invalid format or corrupted data
    InvalidFormat(String),
    /// Shape mismatch
    ShapeMismatch {
        /// Path of the tensor
        path: String,
        /// Expected shape
        expected: Vec<usize>,
        /// Found shape
        found: Vec<usize>,
    },
    /// Type mismatch
    TypeMismatch {
        /// Path of the tensor
        path: String,
        /// Expected data type
        expected: DType,
        /// Found data type
        found: DType,
    },
    /// Generic error with message
    Other(String),
}

impl fmt::Display for ReaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReaderError::TensorNotFound(path) => write!(f, "Tensor not found: {}", path),
            ReaderError::Io(msg) => write!(f, "IO error: {}", msg),
            ReaderError::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
            ReaderError::ShapeMismatch {
                path,
                expected,
                found,
            } => write!(
                f,
                "Shape mismatch for {}: expected {:?}, found {:?}",
                path, expected, found
            ),
            ReaderError::TypeMismatch {
                path,
                expected,
                found,
            } => write!(
                f,
                "Type mismatch for {}: expected {:?}, found {:?}",
                path, expected, found
            ),
            ReaderError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl core::error::Error for ReaderError {}

impl From<core::fmt::Error> for ReaderError {
    fn from(err: core::fmt::Error) -> Self {
        ReaderError::Other(format!("Format error: {}", err))
    }
}

/// Trait for lazy tensor loading from any source.
///
/// Implementations should:
/// 1. List available tensors without loading data
/// 2. Provide metadata without loading tensor data
/// 3. Create lazy TensorViews that only load data when needed
///
/// # Examples
///
/// ```ignore
/// struct MyReader {
///     // ... reader state
/// }
///
/// impl TensorReader for MyReader {
///     fn list_tensors(&mut self) -> Result<Vec<String>, ReaderError> {
///         // Return list of tensor paths without loading data
///     }
///
///     fn read_tensor_view(&mut self, path: &str) -> Result<TensorView, ReaderError> {
///         // Return a TensorView that lazily loads data
///     }
/// }
/// ```
pub trait TensorReader: Send {
    /// List available tensor paths without loading data
    fn list_tensors(&mut self) -> Result<Vec<String>, ReaderError>;

    /// Create a lazy TensorView for the given path.
    /// The actual tensor data should only be loaded when TensorView::to_data() is called.
    fn read_tensor_view(&mut self, path: &str) -> Result<TensorView, ReaderError>;

    /// Read all tensor views (still lazy - no data loading yet).
    /// Default implementation uses list_tensors and read_tensor_view.
    fn read_all_views(&mut self) -> Result<HashMap<String, TensorView>, ReaderError> {
        let paths = self.list_tensors()?;
        let mut views = HashMap::new();
        for path in paths {
            views.insert(path.clone(), self.read_tensor_view(&path)?);
        }
        Ok(views)
    }
}
