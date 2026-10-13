use std::collections::HashMap;

use burn_ir::TensorId;

use crate::GroupPlacement;

/// Where each tensor of a device group sits.
#[derive(Debug, Default)]
pub struct TensorPlacements {
    placements: HashMap<TensorId, GroupPlacement>,
}

impl TensorPlacements {
    /// # Panics
    ///
    /// When the tensor is not on the group.
    pub fn get(&self, id: &TensorId) -> GroupPlacement {
        *self
            .placements
            .get(id)
            .unwrap_or_else(|| panic!("{id} has no placement on its device group"))
    }

    pub fn insert(&mut self, id: TensorId, placement: GroupPlacement) {
        self.placements.insert(id, placement);
    }

    pub fn remove(&mut self, id: &TensorId) -> Option<GroupPlacement> {
        self.placements.remove(id)
    }
}
