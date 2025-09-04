use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use burn_tensor::DType;
use core::fmt;
use hashbrown::HashMap;
use serde::{Deserialize, Serialize};

/// Error types for Safetensors operations
#[derive(Debug)]
pub enum SafetensorsError {
    /// IO errors
    Io(String),
    /// Invalid file format
    InvalidFormat(String),
    /// Tensor not found
    TensorNotFound(String),
    /// Validation failed
    ValidationFailed(String),
    /// Shape mismatch during import
    ShapeMismatch {
        tensor: String,
        expected: Vec<usize>,
        found: Vec<usize>,
    },
    /// Data type mismatch
    TypeMismatch {
        tensor: String,
        expected: DType,
        found: DType,
    },
    /// Unsupported operation
    Unsupported(String),
    /// Serialization/deserialization error
    SerdeError(String),
}

impl fmt::Display for SafetensorsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SafetensorsError::Io(msg) => write!(f, "IO error: {}", msg),
            SafetensorsError::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
            SafetensorsError::TensorNotFound(name) => write!(f, "Tensor not found: {}", name),
            SafetensorsError::ValidationFailed(msg) => write!(f, "Validation failed: {}", msg),
            SafetensorsError::ShapeMismatch {
                tensor,
                expected,
                found,
            } => write!(
                f,
                "Shape mismatch for {}: expected {:?}, found {:?}",
                tensor, expected, found
            ),
            SafetensorsError::TypeMismatch {
                tensor,
                expected,
                found,
            } => write!(
                f,
                "Type mismatch for {}: expected {:?}, found {:?}",
                tensor, expected, found
            ),
            SafetensorsError::Unsupported(msg) => write!(f, "Unsupported: {}", msg),
            SafetensorsError::SerdeError(msg) => write!(f, "Serialization error: {}", msg),
        }
    }
}

impl core::error::Error for SafetensorsError {}

impl From<core::fmt::Error> for SafetensorsError {
    fn from(err: core::fmt::Error) -> Self {
        SafetensorsError::SerdeError(format!("Format error: {}", err))
    }
}

/// Safetensors file header structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetensorsHeader {
    #[serde(flatten)]
    pub tensors: HashMap<String, TensorInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
}

impl SafetensorsHeader {
    /// Create a new empty header
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
            metadata: None,
        }
    }

    /// Add metadata to the header
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        if self.metadata.is_none() {
            self.metadata = Some(HashMap::new());
        }
        if let Some(ref mut metadata) = self.metadata {
            metadata.insert(key, value);
        }
        self
    }

    /// Add a tensor to the header
    pub fn add_tensor(&mut self, name: String, info: TensorInfo) {
        self.tensors.insert(name, info);
    }

    /// Serialize header to JSON bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>, SafetensorsError> {
        serde_json::to_vec(self)
            .map_err(|e| SafetensorsError::SerdeError(format!("Failed to serialize header: {}", e)))
    }

    /// Deserialize header from JSON bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SafetensorsError> {
        serde_json::from_slice(bytes).map_err(|e| {
            SafetensorsError::SerdeError(format!("Failed to deserialize header: {}", e))
        })
    }

    /// Calculate total data size needed for all tensors
    pub fn total_data_size(&self) -> usize {
        self.tensors.values().map(|info| info.byte_size()).sum()
    }
}

impl Default for SafetensorsHeader {
    fn default() -> Self {
        Self::new()
    }
}

/// Information about a tensor in the safetensors file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfo {
    /// Data type of the tensor
    pub dtype: SafetensorsDType,
    /// Shape of the tensor
    pub shape: Vec<usize>,
    /// Data offsets [start, end) in the file
    pub data_offsets: [usize; 2],
}

impl TensorInfo {
    /// Create new tensor info
    pub fn new(dtype: DType, shape: Vec<usize>, start: usize, end: usize) -> Self {
        Self {
            dtype: SafetensorsDType::from_burn_dtype(dtype),
            shape,
            data_offsets: [start, end],
        }
    }

    /// Get the byte size of this tensor
    pub fn byte_size(&self) -> usize {
        let num_elements: usize = self.shape.iter().product();
        num_elements * self.dtype.size()
    }

    /// Convert to Burn DType
    pub fn to_burn_dtype(&self) -> DType {
        self.dtype.to_burn_dtype()
    }

    /// Validate that the data offsets match the expected size
    pub fn validate_offsets(&self) -> Result<(), SafetensorsError> {
        let expected_size = self.byte_size();
        let actual_size = self.data_offsets[1] - self.data_offsets[0];

        if expected_size != actual_size {
            return Err(SafetensorsError::InvalidFormat(format!(
                "Tensor size mismatch: expected {} bytes, found {} bytes",
                expected_size, actual_size
            )));
        }

        Ok(())
    }
}

/// Safetensors data type representation
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum SafetensorsDType {
    /// Boolean type
    Bool,
    /// Unsigned 8-bit integer
    U8,
    /// Signed 8-bit integer
    I8,
    /// Signed 16-bit integer
    I16,
    /// Unsigned 16-bit integer
    U16,
    /// Signed 32-bit integer
    I32,
    /// Unsigned 32-bit integer
    U32,
    /// Signed 64-bit integer
    I64,
    /// Unsigned 64-bit integer
    U64,
    /// 16-bit float (half precision)
    F16,
    /// Brain 16-bit float
    BF16,
    /// 32-bit float (single precision)
    F32,
    /// 64-bit float (double precision)
    F64,
}

impl SafetensorsDType {
    /// Convert from Burn DType
    pub fn from_burn_dtype(dtype: DType) -> Self {
        match dtype {
            DType::Bool => Self::Bool,
            DType::U8 => Self::U8,
            DType::I8 => Self::I8,
            DType::I16 => Self::I16,
            DType::I32 => Self::I32,
            DType::U32 => Self::U32,
            DType::I64 => Self::I64,
            DType::U64 => Self::U64,
            DType::F16 => Self::F16,
            DType::BF16 => Self::BF16,
            DType::F32 => Self::F32,
            DType::F64 => Self::F64,
            _ => Self::F32, // Default to F32 for unknown types
        }
    }

    /// Convert to Burn DType
    pub fn to_burn_dtype(&self) -> DType {
        match self {
            Self::Bool => DType::Bool,
            Self::U8 => DType::U8,
            Self::I8 => DType::I8,
            Self::I16 => DType::I16,
            Self::U16 => DType::U32, // Burn doesn't have U16, use U32
            Self::I32 => DType::I32,
            Self::U32 => DType::U32,
            Self::I64 => DType::I64,
            Self::U64 => DType::U64,
            Self::F16 => DType::F16,
            Self::BF16 => DType::BF16,
            Self::F32 => DType::F32,
            Self::F64 => DType::F64,
        }
    }

    /// Get the size in bytes
    pub fn size(&self) -> usize {
        match self {
            Self::Bool | Self::U8 | Self::I8 => 1,
            Self::I16 | Self::U16 | Self::F16 | Self::BF16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 => 8,
        }
    }
}

/// Magic number for safetensors files
pub const SAFETENSORS_MAGIC: &[u8; 8] = b"<SAFETNS";

/// Helper to read/write the header size (8 bytes, little-endian)
pub fn read_header_size(bytes: &[u8]) -> Result<u64, SafetensorsError> {
    if bytes.len() < 8 {
        return Err(SafetensorsError::InvalidFormat(
            "Not enough bytes for header size".to_string(),
        ));
    }

    let mut size_bytes = [0u8; 8];
    size_bytes.copy_from_slice(&bytes[0..8]);
    Ok(u64::from_le_bytes(size_bytes))
}

/// Helper to write header size as 8 bytes (little-endian)
pub fn write_header_size(size: u64) -> [u8; 8] {
    size.to_le_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_info_byte_size() {
        let info = TensorInfo::new(DType::F32, vec![2, 3, 4], 0, 96);
        assert_eq!(info.byte_size(), 96); // 2*3*4*4 bytes

        let info = TensorInfo::new(DType::F64, vec![10, 10], 0, 800);
        assert_eq!(info.byte_size(), 800); // 10*10*8 bytes
    }

    #[test]
    fn test_header_serialization() {
        let mut header = SafetensorsHeader::new();
        header.add_tensor(
            "test".to_string(),
            TensorInfo::new(DType::F32, vec![2, 2], 0, 16),
        );

        let bytes = header.to_bytes().unwrap();
        let header2 = SafetensorsHeader::from_bytes(&bytes).unwrap();

        assert_eq!(header.tensors.len(), header2.tensors.len());
        assert!(header2.tensors.contains_key("test"));
    }

    #[test]
    fn test_dtype_conversion() {
        let burn_dtypes = vec![DType::Bool, DType::U8, DType::I32, DType::F32, DType::F64];

        for dtype in burn_dtypes {
            let safe_dtype = SafetensorsDType::from_burn_dtype(dtype);
            let converted = safe_dtype.to_burn_dtype();
            assert_eq!(dtype, converted);
        }
    }

    #[test]
    fn test_validate_offsets() {
        let info = TensorInfo::new(DType::F32, vec![2, 3], 0, 24);
        assert!(info.validate_offsets().is_ok());

        let bad_info = TensorInfo::new(DType::F32, vec![2, 3], 0, 20);
        assert!(bad_info.validate_offsets().is_err());
    }
}
