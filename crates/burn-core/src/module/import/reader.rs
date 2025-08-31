use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;
use hashbrown::HashMap;

use crate::module::export::TensorView;

/// Metadata about a tensor without loading its data
#[derive(Debug, Clone)]
pub struct TensorMetadata {
    /// Shape of the tensor
    pub shape: Vec<usize>,
    /// Data type of the tensor
    pub dtype: DType,
    /// Size in bytes (optional, for readers that know this upfront)
    pub size_bytes: Option<usize>,
}

/// Supported data types for tensors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    /// 32-bit floating point
    F32,
    /// 64-bit floating point
    F64,
    /// 16-bit floating point
    F16,
    /// Brain floating point
    BF16,
    /// 32-bit integer
    I32,
    /// 64-bit integer
    I64,
    /// 8-bit integer
    I8,
    /// 8-bit unsigned integer
    U8,
    /// Boolean
    Bool,
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::F32 => write!(f, "f32"),
            DType::F64 => write!(f, "f64"),
            DType::F16 => write!(f, "f16"),
            DType::BF16 => write!(f, "bf16"),
            DType::I32 => write!(f, "i32"),
            DType::I64 => write!(f, "i64"),
            DType::I8 => write!(f, "i8"),
            DType::U8 => write!(f, "u8"),
            DType::Bool => write!(f, "bool"),
        }
    }
}

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
                "Type mismatch for {}: expected {}, found {}",
                path, expected, found
            ),
            ReaderError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ReaderError {}

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

    /// Get metadata for a tensor without loading its data
    fn tensor_metadata(&mut self, path: &str) -> Result<TensorMetadata, ReaderError>;

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
