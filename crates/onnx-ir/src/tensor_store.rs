//! Central tensor data storage with unique IDs for constants and initializers
//!
//! This module provides tensor data storage that enables zero-copy parsing
//! from memory-mapped ONNX files. Tensor data is stored as raw bytes references
//! and only converted to `TensorData` when accessed.
//!
//! Supports both embedded data (in the ONNX protobuf) and external data
//! (stored in separate files for models >2GB).

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::rc::Rc;

use burn_tensor::DType;

use crate::ir::{DataId, TensorData};

/// Source of tensor data - either embedded in protobuf or stored externally
#[derive(Debug, Clone)]
pub enum TensorDataSource {
    /// Embedded data: raw bytes already in memory (mmap'd or copied from protobuf)
    Embedded(bytes::Bytes),

    /// External data: load lazily from file when needed
    /// Used for large models (>2GB) that store weights in separate files
    External {
        /// Absolute path to external data file
        file_path: PathBuf,
        /// Byte offset within the file
        offset: u64,
        /// Number of bytes to read
        length: u64,
    },
}

/// A reference to tensor data that may be embedded or external.
///
/// This enables:
/// - Zero-copy parsing from mmap'd ONNX files (embedded data)
/// - Lazy loading of external tensor data (for models >2GB)
///
/// Data is only materialized when `to_tensor_data()` is called, at which point:
/// - Embedded: bytes are copied to aligned heap memory
/// - External: file is read (or mmap'd) and bytes are copied
#[derive(Debug, Clone)]
pub struct TensorDataRef {
    /// Shape of the tensor
    shape: Vec<usize>,
    /// Data type of elements
    dtype: DType,
    /// Where the actual bytes come from
    source: TensorDataSource,
}

impl TensorDataRef {
    /// Create new tensor data reference from embedded raw bytes
    pub fn new(raw_bytes: bytes::Bytes, shape: Vec<usize>, dtype: DType) -> Self {
        Self {
            shape,
            dtype,
            source: TensorDataSource::Embedded(raw_bytes),
        }
    }

    /// Create new tensor data reference from external file location
    pub fn new_external(
        file_path: PathBuf,
        offset: u64,
        length: u64,
        shape: Vec<usize>,
        dtype: DType,
    ) -> Self {
        Self {
            shape,
            dtype,
            source: TensorDataSource::External {
                file_path,
                offset,
                length,
            },
        }
    }

    /// Get the shape of the tensor
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the data type of the tensor
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get the tensor data, loading from source and copying to an aligned buffer.
    ///
    /// For embedded data: copies from mmap'd buffer to heap memory.
    /// For external data: reads from file (with optional mmap) and copies to heap.
    ///
    /// This copy is necessary for correctness: mmap'd buffers may not
    /// have proper alignment for multi-byte types (e.g., f32, i64), so copying to a
    /// heap-allocated Vec<u8> ensures alignment for safe typed access.
    pub fn to_tensor_data(&self) -> TensorData {
        let bytes = match &self.source {
            TensorDataSource::Embedded(raw_bytes) => raw_bytes.to_vec(),
            TensorDataSource::External {
                file_path,
                offset,
                length,
            } => Self::load_external_data(file_path, *offset, *length),
        };
        TensorData::from_bytes_vec(bytes, self.shape.clone(), self.dtype)
    }

    /// Load tensor data from an external file
    ///
    /// Uses mmap when the feature is enabled, otherwise falls back to standard file I/O.
    fn load_external_data(file_path: &PathBuf, offset: u64, length: u64) -> Vec<u8> {
        #[cfg(feature = "mmap")]
        {
            if let Ok(data) = Self::load_external_mmap(file_path, offset, length) {
                return data;
            }
            // Fall through to standard I/O on mmap failure
            log::debug!(
                "mmap failed for {}, falling back to standard I/O",
                file_path.display()
            );
        }

        // Standard file I/O fallback
        Self::load_external_read(file_path, offset, length)
    }

    /// Load external data using memory mapping (zero-copy until we need aligned buffer)
    #[cfg(feature = "mmap")]
    fn load_external_mmap(
        file_path: &PathBuf,
        offset: u64,
        length: u64,
    ) -> std::io::Result<Vec<u8>> {
        let file = File::open(file_path)?;

        // SAFETY: We're mapping a read-only file
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        let start = offset as usize;
        let end = start + length as usize;

        if end > mmap.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!(
                    "External data range {}..{} exceeds file size {}",
                    start,
                    end,
                    mmap.len()
                ),
            ));
        }

        Ok(mmap[start..end].to_vec())
    }

    /// Load external data using standard file I/O
    fn load_external_read(file_path: &PathBuf, offset: u64, length: u64) -> Vec<u8> {
        let mut file = File::open(file_path).unwrap_or_else(|e| {
            panic!(
                "Failed to open external data file '{}': {}",
                file_path.display(),
                e
            )
        });

        file.seek(SeekFrom::Start(offset)).unwrap_or_else(|e| {
            panic!(
                "Failed to seek to offset {} in '{}': {}",
                offset,
                file_path.display(),
                e
            )
        });

        let mut buffer = vec![0u8; length as usize];
        file.read_exact(&mut buffer).unwrap_or_else(|e| {
            panic!(
                "Failed to read {} bytes from '{}': {}",
                length,
                file_path.display(),
                e
            )
        });

        buffer
    }
}

/// Convert from TensorData to TensorDataRef
///
/// This is used for compatibility with existing code that produces TensorData,
/// such as the fallback paths in argument_from_initializer for scalars and empty tensors.
impl From<TensorData> for TensorDataRef {
    fn from(tensor_data: TensorData) -> Self {
        // Extract bytes from TensorData's internal storage
        // burn_tensor::Bytes implements Deref<[u8]>, so we can copy to bytes::Bytes
        let raw_bytes = bytes::Bytes::copy_from_slice(&tensor_data.bytes);
        Self {
            shape: tensor_data.shape,
            dtype: tensor_data.dtype,
            source: TensorDataSource::Embedded(raw_bytes),
        }
    }
}

/// Central storage for tensor data with unique ID assignment
///
/// Stores tensor data as references - raw bytes are kept as references to the
/// mmap'd buffer, and conversion to TensorData happens on access.
#[derive(Debug, Clone)]
pub struct TensorStore {
    /// Maps tensor IDs to their data references
    data: HashMap<DataId, TensorDataRef>,
    /// Next available tensor ID
    next_id: DataId,
}

impl TensorStore {
    /// Create a new empty tensor store
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
            next_id: 0,
        }
    }

    /// Store tensor data reference and return allocated ID
    pub fn store(&mut self, data: TensorDataRef) -> DataId {
        let id = self.next_id;
        self.next_id += 1;
        self.data.insert(id, data);
        id
    }

    /// Get tensor data by ID, converting from reference storage
    ///
    /// This triggers a copy from the mmap'd buffer to aligned heap memory.
    pub fn get(&self, id: DataId) -> Option<TensorData> {
        self.data.get(&id).map(|data_ref| data_ref.to_tensor_data())
    }

    /// Get the number of tensors in the store (for debugging)
    pub fn len(&self) -> usize {
        self.data.len()
    }
}

impl Default for TensorStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Immutable value store for Arguments
///
/// This combines:
/// - `tensor_store`: Immutable storage for tensor data (lookup by DataId)
/// - `constant_map`: Mapping from constant output names to their DataId
///
/// After graph construction, this allows Arguments to look up their values
/// without needing mutable access or RefCell.
#[derive(Debug, Clone)]
pub struct ValueStore {
    /// Immutable tensor data storage
    tensor_store: Rc<TensorStore>,
    /// Maps constant node output names to their data IDs
    /// e.g., "constant1_out1" -> 0
    constant_map: Rc<HashMap<String, DataId>>,
}

impl ValueStore {
    /// Create a new ValueStore from tensor store and constant map
    pub fn new(tensor_store: Rc<TensorStore>, constant_map: Rc<HashMap<String, DataId>>) -> Self {
        Self {
            tensor_store,
            constant_map,
        }
    }

    /// Get tensor data by ID, converting from lazy storage
    pub fn get_tensor_data(&self, id: DataId) -> Option<TensorData> {
        self.tensor_store.get(id)
    }

    /// Get data ID for a constant by its output name
    pub fn get_constant_data_id(&self, output_name: &str) -> Option<DataId> {
        self.constant_map.get(output_name).copied()
    }

    /// Get the number of entries in the constant map (for debugging)
    pub fn constant_map_len(&self) -> usize {
        self.constant_map.len()
    }

    /// Get the number of tensors in the store (for debugging)
    pub fn tensor_count(&self) -> usize {
        self.tensor_store.len()
    }
}
