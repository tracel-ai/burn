// Burnpack File Format Specification
// ===================================
//
// The Burnpack format is a binary file format designed specifically for Burn tensors.
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
// │    - data_offsets: (u64, u64)   │  (start, end) byte offsets in data section
// │  - Additional metadata (opt)    │  User-defined key-value pairs
// ├─────────────────────────────────┤
// │  Tensor Data Section            │
// ├─────────────────────────────────┤
// │  Raw tensor bytes               │  Contiguous tensor data (aligned)
// │  (in order of metadata)         │  Each tensor's data at specified offsets
// └─────────────────────────────────┘

use crate::{ApplyResult, ModuleSnapshot, ModuleSnapshoter, PathFilter, TensorSnapshot};
use alloc::collections::BTreeMap;
use alloc::rc::Rc;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use burn_tensor::{DType, TensorData};
use byteorder::{ByteOrder, LittleEndian};
use serde::{Deserialize, Serialize};

#[cfg(feature = "std")]
use std::fs::File;
#[cfg(feature = "std")]
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
#[cfg(feature = "std")]
use std::path::{Path, PathBuf};

/// Magic number identifying a Burnpack file: "BURN" in ASCII
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

/// Header structure for Burnpack files (HEADER_SIZE bytes total)
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
    pub fn to_bytes(self) -> [u8; HEADER_SIZE] {
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

/// Metadata structure serialized with MessagePack
///
/// This is serialized using rmp_serde::to_vec for compact binary representation.
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
        }
    }
}

/// Writer for creating Burnpack files
///
/// This writer stores TensorSnapshots lazily and only materializes tensor data
/// when writing to file or bytes, allowing efficient handling of large models.
pub struct BurnpackWriter {
    /// Tensor snapshots with lazy data loading
    snapshots: Vec<(String, TensorSnapshot)>,
    /// Additional metadata
    metadata: BTreeMap<String, String>,
}

impl BurnpackWriter {
    /// Create a new writer
    pub fn new() -> Self {
        Self {
            snapshots: Vec::new(),
            metadata: BTreeMap::new(),
        }
    }

    /// Add a tensor from TensorSnapshot (stores lazily without materializing data)
    pub fn add_tensor_snapshot(&mut self, name: String, snapshot: &TensorSnapshot) {
        // Clone the snapshot (cheap - just clones the Rc to the closure)
        self.snapshots.push((name, snapshot.clone()));
    }

    /// Add a tensor with raw data
    #[allow(dead_code)]
    pub fn add_tensor(&mut self, name: String, dtype: DType, shape: Vec<u64>, data: &[u8]) {
        // Convert raw data to TensorSnapshot for consistency
        let shape_usize: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
        let tensor_data = TensorData::from_bytes_vec(data.to_vec(), shape_usize.clone(), dtype);

        use burn_core::module::ParamId;
        let snapshot =
            TensorSnapshot::from_data(tensor_data, vec![name.clone()], vec![], ParamId::new());

        self.snapshots.push((name, snapshot));
    }

    /// Add metadata key-value pair
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Write to a byte buffer
    pub fn to_bytes(&self) -> Result<Vec<u8>, BurnpackError> {
        // Build tensor descriptors and calculate offsets
        let mut tensors = Vec::new();
        let mut current_offset = 0u64;

        for (name, snapshot) in &self.snapshots {
            let data_len = snapshot.data_len() as u64;
            let start = current_offset;
            let end = start + data_len;

            tensors.push(TensorDescriptor {
                name: name.clone(),
                dtype: snapshot.dtype,
                shape: snapshot.shape.iter().map(|&s| s as u64).collect(),
                data_offsets: (start, end),
            });

            current_offset = end;
        }

        // Create metadata structure
        let metadata = BurnpackMetadata {
            tensors,
            metadata: self.metadata.clone(),
        };

        // Serialize metadata with MessagePack (unnamed for compactness)
        let metadata_bytes = rmp_serde::to_vec(&metadata)
            .map_err(|e| BurnpackError::MetadataSerializationError(e.to_string()))?;

        // Create header
        let header = BurnpackHeader::new(metadata_bytes.len() as u32);
        let header_bytes = header.to_bytes();

        // Calculate total size for pre-allocation
        let total_size = header_bytes.len() + metadata_bytes.len() + current_offset as usize;
        let mut result = Vec::with_capacity(total_size);

        // Write header and metadata
        result.extend_from_slice(&header_bytes);
        result.extend_from_slice(&metadata_bytes);

        // Materialize and write tensor data one at a time
        for (_, snapshot) in &self.snapshots {
            let data = snapshot.to_data();
            let data_len = data.bytes.len();
            if data.bytes.len() != data_len {
                return Err(BurnpackError::IoError(format!(
                    "Tensor data size mismatch: expected {} bytes but got {} bytes",
                    data_len,
                    data.bytes.len()
                )));
            }
            result.extend_from_slice(&data.bytes);
        }

        Ok(result)
    }

    /// Write to a file (streams tensors one at a time to minimize memory usage)
    #[cfg(feature = "std")]
    pub fn write_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), BurnpackError> {
        // Build tensor descriptors and calculate offsets
        let mut tensors = Vec::new();
        let mut current_offset = 0u64;

        for (name, snapshot) in &self.snapshots {
            let data_len = snapshot.data_len() as u64;
            let start = current_offset;
            let end = start + data_len;

            tensors.push(TensorDescriptor {
                name: name.clone(),
                dtype: snapshot.dtype,
                shape: snapshot.shape.iter().map(|&s| s as u64).collect(),
                data_offsets: (start, end),
            });

            current_offset = end;
        }

        // Create metadata structure
        let metadata = BurnpackMetadata {
            tensors,
            metadata: self.metadata.clone(),
        };

        // Serialize metadata with MessagePack (unnamed for compactness)
        let metadata_bytes = rmp_serde::to_vec(&metadata)
            .map_err(|e| BurnpackError::MetadataSerializationError(e.to_string()))?;

        // Create header
        let header = BurnpackHeader::new(metadata_bytes.len() as u32);
        let header_bytes = header.to_bytes();

        // Write directly to file, streaming tensors one at a time
        let mut file = File::create(path).map_err(|e| BurnpackError::IoError(e.to_string()))?;

        // Write header
        file.write_all(&header_bytes)
            .map_err(|e| BurnpackError::IoError(e.to_string()))?;

        // Write metadata
        file.write_all(&metadata_bytes)
            .map_err(|e| BurnpackError::IoError(e.to_string()))?;

        // Stream tensor data one at a time (only one tensor in memory at a time)
        for (_, snapshot) in &self.snapshots {
            let data = snapshot.to_data();
            let data_len = data.bytes.len();
            if data.bytes.len() != data_len {
                return Err(BurnpackError::IoError(format!(
                    "Tensor data size mismatch: expected {} bytes but got {} bytes",
                    data_len,
                    data.bytes.len()
                )));
            }
            file.write_all(&data.bytes)
                .map_err(|e| BurnpackError::IoError(e.to_string()))?;
        }

        Ok(())
    }
}

/// Storage backend for BurnpackReader
enum StorageBackend {
    /// Memory-based storage
    Memory(Rc<Vec<u8>>),
    /// Memory-mapped file storage (efficient for large files)
    #[cfg(all(feature = "std", feature = "memmap"))]
    Mmap(Rc<memmap2::Mmap>),
}

impl StorageBackend {
    /// Get data slice from the storage
    fn get_slice(&self, start: usize, end: usize) -> Vec<u8> {
        match self {
            StorageBackend::Memory(data) => data[start..end].to_vec(),
            #[cfg(all(feature = "std", feature = "memmap"))]
            StorageBackend::Mmap(mmap) => mmap[start..end].to_vec(),
        }
    }

    /// Get full data reference for raw access
    #[allow(dead_code)]
    fn as_bytes(&self) -> &[u8] {
        match self {
            StorageBackend::Memory(data) => data.as_ref(),
            #[cfg(all(feature = "std", feature = "memmap"))]
            StorageBackend::Mmap(mmap) => mmap.as_ref(),
        }
    }
}

/// Reader for loading Burnpack files
pub struct BurnpackReader {
    metadata: BurnpackMetadata,
    /// Storage backend (memory or mmap)
    storage: StorageBackend,
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
        let metadata_start = HEADER_SIZE;
        let metadata_end = metadata_start + header.metadata_size as usize;

        if bytes.len() < metadata_end {
            return Err(BurnpackError::InvalidHeader);
        }

        let metadata: BurnpackMetadata =
            rmp_serde::from_slice(&bytes[metadata_start..metadata_end])
                .map_err(|e| BurnpackError::MetadataDeserializationError(e.to_string()))?;

        Ok(Self {
            metadata,
            storage: StorageBackend::Memory(Rc::new(bytes)),
            data_offset: metadata_end,
        })
    }

    /// Read from a file with automatic strategy selection:
    /// - Uses memory mapping if available (most efficient for large files)
    /// - Falls back to buffered reading if memmap is not available
    #[cfg(feature = "std")]
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, BurnpackError> {
        #[cfg(feature = "memmap")]
        {
            Self::from_file_mmap(path)
        }
        #[cfg(not(feature = "memmap"))]
        {
            Self::from_file_buffered(path)
        }
    }

    /// Read from a file using memory mapping for efficiency
    #[cfg(all(feature = "std", feature = "memmap"))]
    pub fn from_file_mmap<P: AsRef<Path>>(path: P) -> Result<Self, BurnpackError> {
        use memmap2::MmapOptions;

        let file = File::open(path).map_err(|e| BurnpackError::IoError(e.to_string()))?;
        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .map_err(|e| BurnpackError::IoError(e.to_string()))?
        };

        // Read header from mmap
        let header = BurnpackHeader::from_bytes(&mmap)?;

        // Verify version
        if header.version > FORMAT_VERSION {
            return Err(BurnpackError::InvalidVersion);
        }

        // Read metadata from mmap
        let metadata_start = HEADER_SIZE;
        let metadata_end = metadata_start + header.metadata_size as usize;

        if mmap.len() < metadata_end {
            return Err(BurnpackError::InvalidHeader);
        }

        let metadata: BurnpackMetadata = rmp_serde::from_slice(&mmap[metadata_start..metadata_end])
            .map_err(|e| BurnpackError::MetadataDeserializationError(e.to_string()))?;

        Ok(Self {
            metadata,
            storage: StorageBackend::Mmap(Rc::new(mmap)),
            data_offset: metadata_end,
        })
    }

    /// Read from a file using buffered reading
    /// This is less efficient than memory mapping but works everywhere
    #[cfg(feature = "std")]
    #[allow(dead_code)]
    pub fn from_file_buffered<P: AsRef<Path>>(path: P) -> Result<Self, BurnpackError> {
        let file = File::open(&path).map_err(|e| BurnpackError::IoError(e.to_string()))?;
        let mut reader = BufReader::new(file);

        // Read header
        let mut header_bytes = [0u8; HEADER_SIZE];
        reader
            .read_exact(&mut header_bytes)
            .map_err(|e| BurnpackError::IoError(e.to_string()))?;

        let header = BurnpackHeader::from_bytes(&header_bytes)?;

        // Verify version
        if header.version > FORMAT_VERSION {
            return Err(BurnpackError::InvalidVersion);
        }

        // Read metadata
        let mut metadata_bytes = vec![0u8; header.metadata_size as usize];
        reader
            .read_exact(&mut metadata_bytes)
            .map_err(|e| BurnpackError::IoError(e.to_string()))?;

        let metadata: BurnpackMetadata = rmp_serde::from_slice(&metadata_bytes)
            .map_err(|e| BurnpackError::MetadataDeserializationError(e.to_string()))?;

        // For non-mmap, we still need to load the full file for tensor access
        // But we've at least parsed the header and metadata efficiently
        reader
            .seek(SeekFrom::Start(0))
            .map_err(|e| BurnpackError::IoError(e.to_string()))?;

        let mut bytes = Vec::new();
        reader
            .read_to_end(&mut bytes)
            .map_err(|e| BurnpackError::IoError(e.to_string()))?;

        Ok(Self {
            metadata,
            storage: StorageBackend::Memory(Rc::new(bytes)),
            data_offset: HEADER_SIZE + header.metadata_size as usize,
        })
    }

    /// Get metadata
    #[allow(dead_code)]
    pub fn metadata(&self) -> &BurnpackMetadata {
        &self.metadata
    }

    /// Get tensor data by name
    #[allow(dead_code)]
    pub fn get_tensor_data(&self, name: &str) -> Result<&[u8], BurnpackError> {
        let descriptor = self
            .metadata
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| BurnpackError::TensorNotFound(name.to_string()))?;

        let start = self.data_offset + descriptor.data_offsets.0 as usize;
        let end = self.data_offset + descriptor.data_offsets.1 as usize;

        Ok(&self.storage.as_bytes()[start..end])
    }

    /// Get tensor as TensorSnapshot with lazy loading
    pub fn get_tensor_snapshot(&self, name: &str) -> Result<TensorSnapshot, BurnpackError> {
        let descriptor = self
            .metadata
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| BurnpackError::TensorNotFound(name.to_string()))?;

        let start = self.data_offset + descriptor.data_offsets.0 as usize;
        let end = self.data_offset + descriptor.data_offsets.1 as usize;

        // Clone metadata for use in closure
        let shape: Vec<usize> = descriptor.shape.iter().map(|&s| s as usize).collect();
        let dtype = descriptor.dtype;

        // Clone storage reference for the closure
        let storage = match &self.storage {
            StorageBackend::Memory(data) => StorageBackend::Memory(data.clone()),
            #[cfg(all(feature = "std", feature = "memmap"))]
            StorageBackend::Mmap(mmap) => StorageBackend::Mmap(mmap.clone()),
        };

        // Clone shape for the closure
        let shape_for_closure = shape.clone();

        // Create lazy TensorSnapshot
        use burn_core::module::ParamId;
        Ok(TensorSnapshot::from_closure(
            Rc::new(move || {
                // This closure is only called when data is actually needed
                let data_bytes = storage.get_slice(start, end);
                TensorData::from_bytes_vec(data_bytes, shape_for_closure.clone(), dtype)
            }),
            dtype,
            shape,
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
            match &mut self.mode {
                #[cfg(feature = "std")]
                StoreMode::File(path) => {
                    writer.write_to_file(path)?;
                }
                StoreMode::Bytes(bytes) => {
                    // Generate and store the bytes
                    *bytes = Some(writer.to_bytes()?);
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
