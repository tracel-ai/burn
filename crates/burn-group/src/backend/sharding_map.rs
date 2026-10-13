use std::collections::HashMap;

use burn_ir::TensorId;

use crate::Sharding;

/// Where each tensor of a device group sits.
#[derive(Debug, Default)]
pub struct ShardingMap {
    shardings: HashMap<TensorId, Sharding>,
}

impl ShardingMap {
    /// # Panics
    ///
    /// When the tensor is not on the group.
    pub fn get(&self, id: &TensorId) -> Sharding {
        *self
            .shardings
            .get(id)
            .unwrap_or_else(|| panic!("{id} has no sharding on its device group"))
    }

    pub fn insert(&mut self, id: TensorId, sharding: Sharding) {
        self.shardings.insert(id, sharding);
    }

    pub fn remove(&mut self, id: &TensorId) -> Option<Sharding> {
        self.shardings.remove(id)
    }
}
