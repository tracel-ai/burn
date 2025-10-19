//! Central tensor data storage with unique IDs
//!
//! This module provides a centralized store for all tensor data (constants, initializers)
//! used during ONNX graph conversion. Each tensor is assigned a unique ID for reference.

use std::collections::HashMap;

use crate::ir::{TensorData, TensorId};

/// Central storage for tensor data with unique ID assignment
#[derive(Debug, Clone)]
pub(super) struct TensorStore {
    /// Maps tensor IDs to their data
    data: HashMap<TensorId, TensorData>,
    /// Next available tensor ID
    next_id: TensorId,
}

impl TensorStore {
    /// Create a new empty tensor store
    pub(super) fn new() -> Self {
        Self {
            data: HashMap::new(),
            next_id: 0,
        }
    }

    /// Allocate a new tensor ID and store the data
    ///
    /// Returns the allocated ID
    pub(super) fn store(&mut self, data: TensorData) -> TensorId {
        let id = self.next_id;
        self.next_id += 1;
        self.data.insert(id, data);
        id
    }

    /// Get tensor data by ID (immutable)
    pub(super) fn get(&self, id: TensorId) -> Option<&TensorData> {
        self.data.get(&id)
    }

    /// Get tensor data by ID (mutable)
    pub(super) fn get_mut(&mut self, id: TensorId) -> Option<&mut TensorData> {
        self.data.get_mut(&id)
    }

    /// Get the next available tensor ID (for internal use)
    #[allow(dead_code)]
    pub(super) fn next_id(&self) -> TensorId {
        self.next_id
    }

    /// Clone the underlying data map (for preserving during consume)
    pub(super) fn clone_data(&self) -> HashMap<TensorId, TensorData> {
        self.data.clone()
    }

    /// Restore tensor data from a cloned map
    pub(super) fn restore_data(&mut self, data: HashMap<TensorId, TensorData>, next_id: TensorId) {
        self.data = data;
        self.next_id = next_id;
    }
}
