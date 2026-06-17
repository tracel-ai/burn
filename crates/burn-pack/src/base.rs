//! Core types and constants for the Burnpack file format.
//!
//! See the [parent module](crate::burnpack) for the complete file format specification.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use burn_std::DType;
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
    // Keep multiplication in u64 space to avoid overflow on 32-bit systems
    (unaligned_start.div_ceil(TENSOR_ALIGNMENT) * TENSOR_ALIGNMENT) as usize
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
pub struct Header {
    /// Magic number (4 bytes): 0x4255524E ("BURN")
    pub magic: u32,
    /// Format version (2 bytes)
    pub version: u16,
    /// Size of CBOR metadata in bytes (4 bytes)
    pub metadata_size: u32,
}

impl Header {
    /// Create a new header with the given metadata size
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
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        if bytes.len() < HEADER_SIZE {
            return Err(Error::InvalidHeader);
        }

        let magic = LittleEndian::read_u32(&bytes[magic_range()]);
        if magic != MAGIC_NUMBER {
            return Err(Error::InvalidMagicNumber);
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

/// A typed scalar value stored alongside tensors in a burnpack container.
///
/// Scalars are kept in the CBOR metadata section (not the tensor data section), so they carry
/// no alignment cost. The field is optional in the format: files written before scalar support
/// simply omit it, and readers default it to empty.
///
/// Convert to/from the primitive numeric and boolean types with [`From`] / [`TryFrom`]
/// (e.g. `Scalar::from(3usize)`, `u32::try_from(scalar)`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Scalar {
    /// A signed integer.
    Int(i64),
    /// An unsigned integer.
    UInt(u64),
    /// A floating-point number.
    Float(f64),
    /// A boolean.
    Bool(bool),
}

/// Error returned when a [`Scalar`] cannot be converted to a requested primitive type
/// (wrong variant or out of range).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalarConversionError;

impl core::fmt::Display for ScalarConversionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "scalar value does not fit the requested type")
    }
}

impl core::error::Error for ScalarConversionError {}

macro_rules! impl_scalar_int {
    ($($t:ty => $variant:ident),* $(,)?) => {
        $(
            impl From<$t> for Scalar {
                fn from(value: $t) -> Self {
                    Scalar::$variant(value as _)
                }
            }

            impl TryFrom<Scalar> for $t {
                type Error = ScalarConversionError;
                fn try_from(scalar: Scalar) -> Result<Self, Self::Error> {
                    match scalar {
                        Scalar::Int(v) => v.try_into().map_err(|_| ScalarConversionError),
                        Scalar::UInt(v) => v.try_into().map_err(|_| ScalarConversionError),
                        _ => Err(ScalarConversionError),
                    }
                }
            }
        )*
    };
}

impl_scalar_int!(
    i8 => Int, i16 => Int, i32 => Int, i64 => Int, isize => Int,
    u8 => UInt, u16 => UInt, u32 => UInt, u64 => UInt, usize => UInt,
);

impl From<f64> for Scalar {
    fn from(value: f64) -> Self {
        Scalar::Float(value)
    }
}

impl From<f32> for Scalar {
    fn from(value: f32) -> Self {
        Scalar::Float(value as f64)
    }
}

impl From<bool> for Scalar {
    fn from(value: bool) -> Self {
        Scalar::Bool(value)
    }
}

impl TryFrom<Scalar> for f64 {
    type Error = ScalarConversionError;
    fn try_from(scalar: Scalar) -> Result<Self, Self::Error> {
        match scalar {
            Scalar::Float(v) => Ok(v),
            Scalar::Int(v) => Ok(v as f64),
            Scalar::UInt(v) => Ok(v as f64),
            _ => Err(ScalarConversionError),
        }
    }
}

impl TryFrom<Scalar> for f32 {
    type Error = ScalarConversionError;
    fn try_from(scalar: Scalar) -> Result<Self, Self::Error> {
        // Mirror `f64`'s acceptance of integer variants; float reads may be lossy (documented).
        match scalar {
            Scalar::Float(v) => Ok(v as f32),
            Scalar::Int(v) => Ok(v as f32),
            Scalar::UInt(v) => Ok(v as f32),
            _ => Err(ScalarConversionError),
        }
    }
}

impl TryFrom<Scalar> for bool {
    type Error = ScalarConversionError;
    fn try_from(scalar: Scalar) -> Result<Self, Self::Error> {
        match scalar {
            Scalar::Bool(v) => Ok(v),
            _ => Err(ScalarConversionError),
        }
    }
}

/// Metadata structure serialized with CBOR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Metadata {
    /// Tensor descriptors mapped by name for efficient lookup
    pub tensors: BTreeMap<String, TensorDescriptor>,
    /// Optional additional metadata
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, String>,
    /// Optional typed scalars mapped by name.
    ///
    /// Defaulted on read for backward compatibility with files written before scalar support.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub scalars: BTreeMap<String, Scalar>,
}

/// Individual tensor descriptor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct TensorDescriptor {
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
pub enum Error {
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

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::InvalidHeader => write!(f, "Invalid header: insufficient bytes"),
            Error::InvalidMagicNumber => write!(f, "Invalid magic number"),
            Error::InvalidVersion => write!(f, "Unsupported version"),
            Error::MetadataSerializationError(e) => {
                write!(f, "Metadata serialization error: {}", e)
            }
            Error::MetadataDeserializationError(e) => {
                write!(f, "Metadata deserialization error: {}", e)
            }
            Error::IoError(e) => write!(f, "I/O error: {}", e),
            Error::TensorNotFound(name) => write!(f, "Tensor not found: {}", name),
            Error::TensorBytesSizeMismatch(e) => {
                write!(f, "Tensor bytes size mismatch: {}", e)
            }
            Error::ValidationError(e) => write!(f, "Validation error: {}", e),
        }
    }
}

impl core::error::Error for Error {}

#[cfg(test)]
mod scalar_tests {
    use super::*;

    #[test]
    fn int_round_trips_through_checked_conversion() {
        assert_eq!(i32::try_from(Scalar::from(-5i32)).unwrap(), -5);
        assert_eq!(u8::try_from(Scalar::from(200u8)).unwrap(), 200);
        assert_eq!(usize::try_from(Scalar::from(42usize)).unwrap(), 42);
    }

    #[test]
    fn out_of_range_int_conversion_is_rejected() {
        // u64 value beyond i32::MAX cannot become i32.
        let big = Scalar::from(5_000_000_000u64);
        assert!(i32::try_from(big).is_err());
        // Negative value cannot become u32.
        assert!(u32::try_from(Scalar::from(-1i32)).is_err());
        // 300 does not fit in u8.
        assert!(u8::try_from(Scalar::from(300u32)).is_err());
    }

    #[test]
    fn float_and_bool_variant_mismatches_are_rejected() {
        // An integer field must not read a stored float.
        assert!(i64::try_from(Scalar::Float(1.5)).is_err());
        // A bool field must not read a stored int.
        assert!(bool::try_from(Scalar::Int(1)).is_err());
        // A float field must not read a stored bool.
        assert!(f64::try_from(Scalar::Bool(true)).is_err());
    }

    #[test]
    fn float_accepts_int_variants_symmetrically() {
        assert_eq!(f64::try_from(Scalar::Int(3)).unwrap(), 3.0);
        assert_eq!(f32::try_from(Scalar::Int(3)).unwrap(), 3.0);
        assert_eq!(f64::try_from(Scalar::Float(2.5)).unwrap(), 2.5);
        assert_eq!(f32::try_from(Scalar::Float(2.5)).unwrap(), 2.5);
    }
}
