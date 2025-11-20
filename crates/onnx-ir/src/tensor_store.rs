//! Central tensor data storage with unique IDs for constants and initializers

use std::collections::HashMap;

use crate::ir::{DataId, TensorData};

/// Central storage for tensor data with unique ID assignment
#[derive(Debug, Clone)]
pub(super) struct TensorStore {
    /// Maps tensor IDs to their data
    data: HashMap<DataId, TensorData>,
    /// Next available tensor ID
    next_id: DataId,
}

impl TensorStore {
    /// Create a new empty tensor store
    pub(super) fn new() -> Self {
        Self {
            data: HashMap::new(),
            next_id: 0,
        }
    }

    /// Store tensor data and return allocated ID
    pub(super) fn store(&mut self, data: TensorData) -> DataId {
        let id = self.next_id;
        self.next_id += 1;
        self.data.insert(id, data);
        id
    }

    /// Get tensor data by ID
    pub(super) fn get(&self, id: DataId) -> Option<&TensorData> {
        self.data.get(&id)
    }

    /// Get mutable tensor data by ID
    pub(super) fn get_mut(&mut self, id: DataId) -> Option<&mut TensorData> {
        self.data.get_mut(&id)
    }

    /// Get next available tensor ID
    #[allow(dead_code)]
    pub(super) fn next_id(&self) -> DataId {
        self.next_id
    }

    /// Clone the data map
    pub(super) fn clone_data(&self) -> HashMap<DataId, TensorData> {
        self.data.clone()
    }

    /// Restore from cloned data
    pub(super) fn restore_data(&mut self, data: HashMap<DataId, TensorData>, next_id: DataId) {
        self.data = data;
        self.next_id = next_id;
    }
}
