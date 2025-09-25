// Burnpack File Format Specification
// ===================================
//
// The Burnpack format is a binary file format designed specifically for Burn tensors,
// using MessagePack for metadata serialization instead of JSON (as in SafeTensors).
//
// File Structure:
// ┌─────────────────────────────────┐
// │  Header (10 bytes)              │
// ├─────────────────────────────────┤
// │  - Magic number (4 bytes)       │  "BURN" (0x4255524E)
// │  - Version (2 bytes)            │  Format version (e.g., 0x0001)
// │  - Metadata size (4 bytes)      │  Size of MessagePack metadata in bytes (u32)
// ├─────────────────────────────────┤
// │  Metadata (MessagePack)         │
// ├─────────────────────────────────┤
// │  - Tensor descriptors           │  Array of tensor metadata
// │    - name: string               │  Tensor identifier
// │    - dtype: string              │  Data type (f32, f64, i32, etc.)
// │    - shape: array<u64>          │  Tensor dimensions
// │    - data_offsets: [u64, u64]   │  [start, end] byte offsets in data section
// │  - Additional metadata (opt)    │  User-defined key-value pairs
// ├─────────────────────────────────┤
// │  Tensor Data Section            │
// ├─────────────────────────────────┤
// │  Raw tensor bytes               │  Contiguous tensor data (aligned)
// │  (in order of metadata)         │  Each tensor's data at specified offsets
// └─────────────────────────────────┘

use crate::{ApplyResult, ModuleSnapshot, ModuleSnapshoter, PathFilter, TensorSnapshot};
use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use burn_tensor::{DType, TensorData};
use byteorder::{ByteOrder, LittleEndian};
use serde::{Deserialize, Serialize};

#[cfg(feature = "std")]
use std::fs::File;
#[cfg(feature = "std")]
use std::io::{Read, Write};
#[cfg(feature = "std")]
use std::path::{Path, PathBuf};

/// Magic number identifying a Burnpack file: "BURN" in ASCII
pub const MAGIC_NUMBER: u32 = 0x4255524E;

/// Current format version
pub const FORMAT_VERSION: u16 = 0x0001;

/// Header structure for Burnpack files (10 bytes total)
#[derive(Debug, Clone, Copy)]
pub struct BurnpackHeader {
    /// Magic number (4 bytes): 0x4255524E ("BURN")
    pub magic: u32,
    /// Format version (2 bytes)
    pub version: u16,
    /// Size of MessagePack metadata in bytes (4 bytes) - supports up to 4GB of metadata
    pub metadata_size: u32,
}

impl BurnpackHeader {
    /// Create a new header with the given metadata size
    pub fn new(metadata_size: u32) -> Self {
        Self {
            magic: MAGIC_NUMBER,
            version: FORMAT_VERSION,
            metadata_size,
        }
    }

    /// Serialize header to bytes
    pub fn to_bytes(&self) -> [u8; 10] {
        let mut bytes = [0u8; 10];
        LittleEndian::write_u32(&mut bytes[0..4], self.magic);
        LittleEndian::write_u16(&mut bytes[4..6], self.version);
        LittleEndian::write_u32(&mut bytes[6..10], self.metadata_size);
        bytes
    }

    /// Deserialize header from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, BurnpackError> {
        if bytes.len() < 10 {
            return Err(BurnpackError::InvalidHeader);
        }

        let magic = LittleEndian::read_u32(&bytes[0..4]);
        if magic != MAGIC_NUMBER {
            return Err(BurnpackError::InvalidMagicNumber);
        }

        let version = LittleEndian::read_u16(&bytes[4..6]);
        let metadata_size = LittleEndian::read_u32(&bytes[6..10]);

        Ok(Self {
            magic,
            version,
            metadata_size,
        })
    }
}

/// Metadata structure serialized with MessagePack
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BurnpackMetadata {
    /// Tensor descriptors
    pub tensors: Vec<TensorDescriptor>,
    /// Optional additional metadata
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, String>,
}

/// Individual tensor descriptor
///
/// IMPORTANT: The DType field is serialized using serde's default enum serialization.
/// This means the serialization format depends on the enum variant indices.
/// If the DType enum is reordered or variants are added/removed in non-append fashion,
/// it may break compatibility with previously saved files.
///
/// To maintain compatibility:
/// - Only add new variants at the end of the DType enum
/// - Never remove or reorder existing variants
/// - Consider using a tagged enum serialization format in the future if needed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorDescriptor {
    /// Tensor name/path
    pub name: String,
    /// Data type
    pub dtype: DType,
    /// Tensor shape dimensions
    pub shape: Vec<u64>,
    /// Byte offsets in data section [start, end)
    pub data_offsets: [u64; 2],
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
        }
    }
}

/// Writer for creating Burnpack files
pub struct BurnpackWriter {
    tensors: Vec<TensorDescriptor>,
    tensor_data: Vec<u8>,
    metadata: BTreeMap<String, String>,
    current_offset: u64,
}

impl BurnpackWriter {
    /// Create a new writer
    pub fn new() -> Self {
        Self {
            tensors: Vec::new(),
            tensor_data: Vec::new(),
            metadata: BTreeMap::new(),
            current_offset: 0,
        }
    }

    /// Add a tensor from TensorSnapshot
    pub fn add_tensor_snapshot(&mut self, name: String, snapshot: &TensorSnapshot) {
        let data = snapshot.to_data();
        let bytes = data.bytes;

        let start = self.current_offset;
        let end = start + bytes.len() as u64;

        self.tensors.push(TensorDescriptor {
            name,
            dtype: snapshot.dtype,
            shape: snapshot.shape.iter().map(|&s| s as u64).collect(),
            data_offsets: [start, end],
        });

        self.tensor_data.extend_from_slice(&bytes);
        self.current_offset = end;
    }

    /// Add a tensor with raw data
    pub fn add_tensor(&mut self, name: String, dtype: DType, shape: Vec<u64>, data: &[u8]) {
        let start = self.current_offset;
        let end = start + data.len() as u64;

        self.tensors.push(TensorDescriptor {
            name,
            dtype,
            shape,
            data_offsets: [start, end],
        });

        self.tensor_data.extend_from_slice(data);
        self.current_offset = end;
    }

    /// Add metadata key-value pair
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Write to a byte buffer
    pub fn to_bytes(&self) -> Result<Vec<u8>, BurnpackError> {
        // Create metadata structure
        let metadata = BurnpackMetadata {
            tensors: self.tensors.clone(),
            metadata: self.metadata.clone(),
        };

        // Serialize metadata with MessagePack
        let metadata_bytes = rmp_serde::to_vec(&metadata)
            .map_err(|e| BurnpackError::MetadataSerializationError(e.to_string()))?;

        // Create header
        let header = BurnpackHeader::new(metadata_bytes.len() as u32);
        let header_bytes = header.to_bytes();

        // Combine all parts
        let mut result =
            Vec::with_capacity(header_bytes.len() + metadata_bytes.len() + self.tensor_data.len());
        result.extend_from_slice(&header_bytes);
        result.extend_from_slice(&metadata_bytes);
        result.extend_from_slice(&self.tensor_data);

        Ok(result)
    }

    /// Write to a file
    #[cfg(feature = "std")]
    pub fn write_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), BurnpackError> {
        let bytes = self.to_bytes()?;
        let mut file = File::create(path).map_err(|e| BurnpackError::IoError(e.to_string()))?;
        file.write_all(&bytes)
            .map_err(|e| BurnpackError::IoError(e.to_string()))?;
        Ok(())
    }
}

/// Reader for loading Burnpack files
pub struct BurnpackReader {
    metadata: BurnpackMetadata,
    data: Vec<u8>,
    data_offset: usize,
}

impl BurnpackReader {
    /// Read from a byte buffer
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, BurnpackError> {
        // Read header
        let header = BurnpackHeader::from_bytes(&bytes)?;

        // Verify version
        if header.version > FORMAT_VERSION {
            return Err(BurnpackError::InvalidVersion);
        }

        // Read metadata
        let metadata_start = 10;
        let metadata_end = metadata_start + header.metadata_size as usize;

        if bytes.len() < metadata_end {
            return Err(BurnpackError::InvalidHeader);
        }

        let metadata: BurnpackMetadata =
            rmp_serde::from_slice(&bytes[metadata_start..metadata_end])
                .map_err(|e| BurnpackError::MetadataDeserializationError(e.to_string()))?;

        Ok(Self {
            metadata,
            data: bytes,
            data_offset: metadata_end,
        })
    }

    /// Read from a file
    #[cfg(feature = "std")]
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, BurnpackError> {
        let mut file = File::open(path).map_err(|e| BurnpackError::IoError(e.to_string()))?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .map_err(|e| BurnpackError::IoError(e.to_string()))?;
        Self::from_bytes(bytes)
    }

    /// Get metadata
    pub fn metadata(&self) -> &BurnpackMetadata {
        &self.metadata
    }

    /// Get tensor data by name
    pub fn get_tensor_data(&self, name: &str) -> Result<&[u8], BurnpackError> {
        let descriptor = self
            .metadata
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| BurnpackError::TensorNotFound(name.to_string()))?;

        let start = self.data_offset + descriptor.data_offsets[0] as usize;
        let end = self.data_offset + descriptor.data_offsets[1] as usize;

        Ok(&self.data[start..end])
    }

    /// Get tensor as TensorSnapshot
    pub fn get_tensor_snapshot(&self, name: &str) -> Result<TensorSnapshot, BurnpackError> {
        let descriptor = self
            .metadata
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| BurnpackError::TensorNotFound(name.to_string()))?;

        let start = self.data_offset + descriptor.data_offsets[0] as usize;
        let end = self.data_offset + descriptor.data_offsets[1] as usize;
        let data_bytes = self.data[start..end].to_vec();

        // Convert shape
        let shape: Vec<usize> = descriptor.shape.iter().map(|&s| s as usize).collect();

        // Create TensorData with proper dtype
        let tensor_data = TensorData::from_bytes_vec(data_bytes, shape.clone(), descriptor.dtype);

        // Create TensorSnapshot from TensorData
        // We use a dummy ParamId since this is loaded from storage
        use burn_core::module::ParamId;
        Ok(TensorSnapshot::from_data(
            tensor_data,
            vec![name.to_string()], // path_stack with just the tensor name
            vec![],                 // empty container_stack
            ParamId::new(),         // new unique id
        ))
    }

    /// Get all tensor names
    pub fn tensor_names(&self) -> Vec<&str> {
        self.metadata
            .tensors
            .iter()
            .map(|t| t.name.as_str())
            .collect()
    }
}

/// BurnpackStore - A Burn-specific file format store using MessagePack for metadata
pub struct BurnpackStore {
    /// Store mode - either file path or bytes
    mode: StoreMode,
    /// Optional filter for selective loading/saving
    filter: Option<PathFilter>,
    /// Additional metadata
    metadata: BTreeMap<String, String>,
    /// Allow partial loading (ignore missing tensors)
    allow_partial: bool,
    /// Accumulated tensors for writing
    writer: Option<BurnpackWriter>,
    /// Reader for loading
    reader: Option<BurnpackReader>,
}

enum StoreMode {
    #[cfg(feature = "std")]
    File(PathBuf),
    Bytes(Option<Vec<u8>>),
}

impl BurnpackStore {
    /// Create a new store from a file path
    #[cfg(feature = "std")]
    pub fn from_file<P: AsRef<Path>>(path: P) -> Self {
        Self {
            mode: StoreMode::File(path.as_ref().to_path_buf()),
            filter: None,
            metadata: BTreeMap::new(),
            allow_partial: false,
            writer: None,
            reader: None,
        }
    }

    /// Create a new store from bytes (for reading) or empty (for writing)
    pub fn from_bytes(bytes: Option<Vec<u8>>) -> Self {
        Self {
            mode: StoreMode::Bytes(bytes),
            filter: None,
            metadata: BTreeMap::new(),
            allow_partial: false,
            writer: None,
            reader: None,
        }
    }
    /// Add metadata key-value pair
    pub fn metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Allow partial loading (ignore missing tensors)
    pub fn allow_partial(mut self, allow: bool) -> Self {
        self.allow_partial = allow;
        self
    }

    /// Set path filter for selective loading/saving
    pub fn with_filter(mut self, filter: PathFilter) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Add regex pattern to filter
    pub fn with_regex(mut self, pattern: &str) -> Self {
        let filter = self.filter.unwrap_or_default();
        self.filter = Some(filter.with_regex(pattern));
        self
    }

    /// Add exact path to filter
    pub fn with_full_path(mut self, path: impl Into<String>) -> Self {
        let filter = self.filter.unwrap_or_default();
        self.filter = Some(filter.with_full_path(path));
        self
    }

    /// Match all tensors (no filtering)
    pub fn match_all(mut self) -> Self {
        self.filter = Some(PathFilter::new().match_all());
        self
    }

    /// Get the bytes after writing (only valid for bytes mode after collecting)
    pub fn get_bytes(&self) -> Result<Vec<u8>, BurnpackError> {
        if let Some(writer) = &self.writer {
            return writer.to_bytes();
        }

        match &self.mode {
            StoreMode::Bytes(Some(bytes)) => Ok(bytes.clone()),
            _ => Err(BurnpackError::IoError("No bytes available".into())),
        }
    }
}

impl ModuleSnapshoter for BurnpackStore {
    type Error = BurnpackError;

    fn collect_from<B: burn_tensor::backend::Backend, M: ModuleSnapshot<B>>(
        &mut self,
        module: &M,
    ) -> Result<(), Self::Error> {
        // Collect snapshots from module
        let snapshots = module.collect(self.filter.clone(), None);

        // Initialize writer with metadata
        let mut writer = BurnpackWriter::new();
        for (key, value) in &self.metadata {
            writer.add_metadata(key.clone(), value.clone());
        }

        // Add each snapshot to writer
        for snapshot in snapshots {
            let path = snapshot.full_path();
            writer.add_tensor_snapshot(path, &snapshot);
        }

        // Store the writer for finalization
        self.writer = Some(writer);

        // Write to storage based on mode
        if let Some(writer) = &self.writer {
            match &self.mode {
                #[cfg(feature = "std")]
                StoreMode::File(path) => {
                    writer.write_to_file(path)?;
                }
                StoreMode::Bytes(_) => {
                    // For bytes mode, we keep the writer to generate bytes on demand
                }
            }
        }

        Ok(())
    }

    fn apply_to<B: burn_tensor::backend::Backend, M: ModuleSnapshot<B>>(
        &mut self,
        module: &mut M,
    ) -> Result<ApplyResult, Self::Error> {
        // Initialize reader if needed
        if self.reader.is_none() {
            let reader = match &self.mode {
                #[cfg(feature = "std")]
                StoreMode::File(file_path) => BurnpackReader::from_file(file_path)?,
                StoreMode::Bytes(Some(bytes)) => BurnpackReader::from_bytes(bytes.clone())?,
                _ => return Err(BurnpackError::IoError("No data to read from".into())),
            };
            self.reader = Some(reader);
        }

        let reader = self.reader.as_ref().unwrap();

        // Convert tensor data to snapshots
        let mut snapshots = Vec::new();
        for tensor_name in reader.tensor_names() {
            // Apply filter if present
            if let Some(filter) = &self.filter
                && !filter.matches(tensor_name)
            {
                continue;
            }

            // Get tensor snapshot
            match reader.get_tensor_snapshot(tensor_name) {
                Ok(mut snapshot) => {
                    // Set the path for the snapshot
                    snapshot.path_stack = Some(vec![tensor_name.to_string()]);
                    snapshots.push(snapshot);
                }
                Err(BurnpackError::TensorNotFound(_)) if self.allow_partial => {
                    // Skip missing tensors if allow_partial is true
                    continue;
                }
                Err(e) => return Err(e),
            }
        }

        // Apply snapshots to module
        let result = module.apply(snapshots, self.filter.clone(), None);

        Ok(result)
    }
}
