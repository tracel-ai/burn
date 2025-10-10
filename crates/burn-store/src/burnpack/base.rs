// Burnpack File Format Specification
// ===================================
//
// The Burnpack format is a binary file format designed specifically for Burn tensors.
//
// File Structure:
// ┌──────────────────────────────────┐
// │  Header (10 bytes)               │
// ├──────────────────────────────────┤
// │  - Magic number (4 bytes)        │  (0x4E525542) - Little Endian
// │  - Version (2 bytes)             │  Format version (0x0001)
// │  - Metadata size (4 bytes)       │  Size of CBOR metadata in bytes (u32)
// ├──────────────────────────────────┤
// │  Metadata (CBOR)                 │
// ├──────────────────────────────────┤
// │  - Tensor descriptors (BTreeMap) │  Sorted map of tensor metadata
// │    Key: tensor name (string)     │  Tensor identifier (e.g., "model.layer1.weight")
// │    Value: TensorDescriptor       │
// │      - dtype: enum               │  Data type (F32, F64, I32, I64, U32, U64, U8, Bool)
// │      - shape: Vec<u64>           │  Tensor dimensions
// │      - data_offsets: (u64, u64)  │  (start, end) byte offsets relative to data section
// │  - Additional metadata(BTreeMap) │  User-defined key-value pairs (string -> string)
// ├──────────────────────────────────┤
// │  Tensor Data Section             │
// ├──────────────────────────────────┤
// │  Raw tensor bytes                │  Contiguous tensor data (little-endian)
// │  (in order of offsets)           │  Each tensor's data at specified offsets
// └──────────────────────────────────┘
//
// Implementation Details:
// - Tensors are stored in a BTreeMap for O(log n) lookup by name and consistent ordering
// - Data offsets are relative to the start of the data section (offset 0)
// - All multi-byte values are stored in little-endian format
// - The reader supports multiple storage backends:
//   * Memory: Full file loaded into RAM
//   * Mmap: Memory-mapped file for efficient large file access
//   * FileBuffered: Direct file I/O with seeking for memory-constrained environments

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
    /// Data type
    pub dtype: DType,
    /// Tensor shape dimensions
    pub shape: Vec<u64>,
    /// Byte offsets in data section (start, end)
    pub data_offsets: (u64, u64),
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
