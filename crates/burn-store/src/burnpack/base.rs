//! Core types and constants for the Burnpack file format.
//!
//! See the [parent module](crate::burnpack) for the complete file format specification.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use burn_tensor::DType;
use byteorder::{ByteOrder, LittleEndian};
use serde::{Deserialize, Serialize};

/// Magic number identifying a Burnpack file: "BURN" in ASCII (0x4255524E)
/// When written to file in little-endian format, appears as "NRUB" bytes
pub const MAGIC_NUMBER: u32 = 0x4255524E;

/// Current format version
pub const FORMAT_VERSION: u16 = 0x0001;

/// Size of the magic number in bytes
pub const MAGIC_SIZE: usize = 4;

/// Size of the format version in bytes
pub const VERSION_SIZE: usize = 2;

/// Size of the metadata size field in bytes
pub const METADATA_SIZE_FIELD_SIZE: usize = 4;

/// Total header size (computed from components)
pub const HEADER_SIZE: usize = MAGIC_SIZE + VERSION_SIZE + METADATA_SIZE_FIELD_SIZE;

/// Alignment for tensor data in bytes.
///
/// All tensor data is aligned to 256-byte boundaries to enable efficient
/// memory-mapped (mmap) zero-copy loading. This alignment ensures:
/// - Proper pointer alignment for all tensor element types (f64 requires 8-byte alignment)
/// - Cache-line friendly access (most CPUs use 64-byte cache lines)
/// - GPU memory alignment (CUDA prefers 256-byte for coalesced access)
/// - Future-proofing for wider SIMD (AVX-512 = 64 bytes, future AVX-1024 = 128 bytes)
///
/// Industry alignment choices:
/// - 256-byte: GGUF, MLX, ncnn, MNN, TNN, vLLM-AWQ, Marlin (15+ formats)
/// - 64-byte: SafeTensors (minimum for AVX-512)
/// - 4096-byte: Core ML
///
/// 256-byte alignment has negligible overhead for typical tensor sizes while
/// providing maximum compatibility with current and future hardware.
pub const TENSOR_ALIGNMENT: u64 = 256;

/// Calculate the byte offset where the tensor data section starts.
///
/// The data section is padded to start at a 256-byte aligned position
/// so that all tensor offsets (which are relative to data section) result
/// in properly aligned absolute file positions for mmap zero-copy access.
///
/// This function must be used consistently by both writer and reader.
#[inline]
pub fn aligned_data_section_start(metadata_size: usize) -> usize {
    let unaligned_start = (HEADER_SIZE + metadata_size) as u64;
    unaligned_start.div_ceil(TENSOR_ALIGNMENT) as usize * TENSOR_ALIGNMENT as usize
}

// Security limits to prevent DoS attacks via resource exhaustion
// These can be adjusted based on your use case

/// Maximum allowed metadata size (100 MB)
/// Prevents memory exhaustion attacks via oversized metadata claims
pub const MAX_METADATA_SIZE: u32 = 100 * 1024 * 1024;

/// Maximum allowed tensor size per tensor
/// Prevents memory exhaustion attacks via oversized tensor claims
/// 32-bit platforms: 2 GB limit (to fit within usize range)
/// 64-bit platforms: 10 GB limit
#[cfg(target_pointer_width = "32")]
pub const MAX_TENSOR_SIZE: usize = 2 * 1024 * 1024 * 1024;
#[cfg(not(target_pointer_width = "32"))]
pub const MAX_TENSOR_SIZE: usize = 10 * 1024 * 1024 * 1024;

/// Maximum allowed number of tensors (100,000)
/// Prevents resource exhaustion via excessive tensor counts
pub const MAX_TENSOR_COUNT: usize = 100_000;

/// Maximum CBOR deserialization recursion depth (128 levels)
/// Prevents stack overflow attacks via deeply nested CBOR structures
pub const MAX_CBOR_RECURSION_DEPTH: usize = 128;

/// Maximum allowed file size (100 GB)
/// Prevents resource exhaustion from extremely large files
/// This limit applies to file-based loading (mmap and buffered)
#[cfg(feature = "std")]
pub const MAX_FILE_SIZE: u64 = 100 * 1024 * 1024 * 1024;

/// Byte range for magic number in header
pub const fn magic_range() -> core::ops::Range<usize> {
    let start = 0;
    let end = start + MAGIC_SIZE;
    start..end
}

/// Byte range for format version in header
pub const fn version_range() -> core::ops::Range<usize> {
    let start = MAGIC_SIZE;
    let end = start + VERSION_SIZE;
    start..end
}

/// Byte range for metadata size field in header
pub const fn metadata_size_range() -> core::ops::Range<usize> {
    let start = MAGIC_SIZE + VERSION_SIZE;
    let end = start + METADATA_SIZE_FIELD_SIZE;
    start..end
}

// Compile-time validation that ranges are correct
const _: () = assert!(MAGIC_SIZE + VERSION_SIZE + METADATA_SIZE_FIELD_SIZE == HEADER_SIZE);

/// Header structure for Burnpack files
#[derive(Debug, Clone, Copy)]
pub struct BurnpackHeader {
    /// Magic number (4 bytes): 0x4255524E ("BURN")
    pub magic: u32,
    /// Format version (2 bytes)
    pub version: u16,
    /// Size of CBOR metadata in bytes (4 bytes)
    pub metadata_size: u32,
}

impl BurnpackHeader {
    /// Create a new header with the given metadata size
    #[allow(dead_code)]
    pub fn new(metadata_size: u32) -> Self {
        Self {
            magic: MAGIC_NUMBER,
            version: FORMAT_VERSION,
            metadata_size,
        }
    }

    /// Serialize header into bytes
    pub fn into_bytes(self) -> [u8; HEADER_SIZE] {
        let mut bytes = [0u8; HEADER_SIZE];
        LittleEndian::write_u32(&mut bytes[magic_range()], self.magic);
        LittleEndian::write_u16(&mut bytes[version_range()], self.version);
        LittleEndian::write_u32(&mut bytes[metadata_size_range()], self.metadata_size);
        bytes
    }

    /// Deserialize header from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, BurnpackError> {
        if bytes.len() < HEADER_SIZE {
            return Err(BurnpackError::InvalidHeader);
        }

        let magic = LittleEndian::read_u32(&bytes[magic_range()]);
        if magic != MAGIC_NUMBER {
            return Err(BurnpackError::InvalidMagicNumber);
        }

        let version = LittleEndian::read_u16(&bytes[version_range()]);
        let metadata_size = LittleEndian::read_u32(&bytes[metadata_size_range()]);

        Ok(Self {
            magic,
            version,
            metadata_size,
        })
    }
}

/// Metadata structure serialized with CBOR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BurnpackMetadata {
    /// Tensor descriptors mapped by name for efficient lookup
    pub tensors: BTreeMap<String, TensorDescriptor>,
    /// Optional additional metadata
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, String>,
}

/// Individual tensor descriptor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorDescriptor {
    /// Data type of the tensor
    pub dtype: DType,
    /// Tensor shape dimensions
    pub shape: Vec<u64>,
    /// Byte offsets in data section (start, end)
    pub data_offsets: (u64, u64),
    /// Parameter ID for training state persistence matching.
    /// Generated automatically if not present during loading.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub param_id: Option<u64>,
}

/// Error types for Burnpack operations
#[derive(Debug)]
pub enum BurnpackError {
    InvalidHeader,
    InvalidMagicNumber,
    InvalidVersion,
    MetadataSerializationError(String),
    MetadataDeserializationError(String),
    IoError(String),
    TensorNotFound(String),
    TensorBytesSizeMismatch(String),
    ValidationError(String),
}

impl core::fmt::Display for BurnpackError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BurnpackError::InvalidHeader => write!(f, "Invalid header: insufficient bytes"),
            BurnpackError::InvalidMagicNumber => write!(f, "Invalid magic number"),
            BurnpackError::InvalidVersion => write!(f, "Unsupported version"),
            BurnpackError::MetadataSerializationError(e) => {
                write!(f, "Metadata serialization error: {}", e)
            }
            BurnpackError::MetadataDeserializationError(e) => {
                write!(f, "Metadata deserialization error: {}", e)
            }
            BurnpackError::IoError(e) => write!(f, "I/O error: {}", e),
            BurnpackError::TensorNotFound(name) => write!(f, "Tensor not found: {}", name),
            BurnpackError::TensorBytesSizeMismatch(e) => {
                write!(f, "Tensor bytes size mismatch: {}", e)
            }
            BurnpackError::ValidationError(e) => write!(f, "Validation error: {}", e),
        }
    }
}

impl core::error::Error for BurnpackError {}
