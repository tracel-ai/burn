//! Central tensor data storage with unique IDs for constants and initializers

use std::collections::HashMap;
use std::rc::Rc;

use crate::ir::{DataId, TensorData};

/// Central storage for tensor data with unique ID assignment
///
/// This is mutable during graph construction but becomes immutable
/// once wrapped in `Rc<TensorStore>` for sharing.
#[derive(Debug, Clone)]
pub struct TensorStore {
    /// Maps tensor IDs to their data
    data: HashMap<DataId, TensorData>,
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

    /// Store tensor data and return allocated ID
    pub fn store(&mut self, data: TensorData) -> DataId {
        let id = self.next_id;
        self.next_id += 1;
        self.data.insert(id, data);
        id
    }

    /// Get tensor data by ID
    pub fn get(&self, id: DataId) -> Option<&TensorData> {
        self.data.get(&id)
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

    /// Get tensor data by ID
    pub fn get_tensor_data(&self, id: DataId) -> Option<&TensorData> {
        self.tensor_store.get(id)
    }

    /// Get data ID for a constant by its output name
    pub fn get_constant_data_id(&self, output_name: &str) -> Option<DataId> {
        self.constant_map.get(output_name).copied()
    }
}
