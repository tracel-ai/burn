//! Central tensor data storage with unique IDs for constants and initializers
//!
//! This module provides tensor data storage that enables zero-copy parsing
//! from memory-mapped ONNX files. Tensor data is stored as raw bytes references
//! and only converted to `TensorData` when accessed.

use std::collections::HashMap;
use std::rc::Rc;

use burn_tensor::DType;

use crate::ir::{DataId, TensorData};

/// A reference to tensor data via cheaply cloneable `bytes::Bytes`.
///
/// This enables zero-copy parsing from mmap'd ONNX files - the raw bytes
/// reference the mmap'd buffer directly, and copying only happens when
/// the tensor data is actually accessed via `to_tensor_data()`.
#[derive(Debug, Clone)]
pub struct TensorDataRef {
    /// Raw bytes from protobuf (zero-copy slice of mmap'd buffer)
    raw_bytes: bytes::Bytes,
    /// Shape of the tensor
    shape: Vec<usize>,
    /// Data type of elements
    dtype: DType,
}

impl TensorDataRef {
    /// Create new tensor data reference from raw bytes
    pub fn new(raw_bytes: bytes::Bytes, shape: Vec<usize>, dtype: DType) -> Self {
        Self {
            raw_bytes,
            shape,
            dtype,
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

    /// Get the tensor data, copying bytes to an aligned buffer.
    ///
    /// This is the point where data is copied from the mmap'd buffer
    /// to heap memory. This copy is necessary for correctness: mmap'd buffers may not
    /// have proper alignment for multi-byte types (e.g., f32, i64), so copying to a
    /// heap-allocated Vec<u8> ensures alignment for safe typed access.
    pub fn to_tensor_data(&self) -> TensorData {
        TensorData::from_bytes_vec(self.raw_bytes.to_vec(), self.shape.clone(), self.dtype)
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
            raw_bytes,
            shape: tensor_data.shape,
            dtype: tensor_data.dtype,
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
}
