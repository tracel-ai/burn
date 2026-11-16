use std::collections::HashMap;

use burn_ir::TensorId;

use crate::Sharding;

/// Where the shards of every tensor on the sharded compute device sit, by tensor id.
#[derive(Default)]
pub struct Shardings {
    by_id: HashMap<TensorId, Sharding>,
}

impl Shardings {
    /// # Panics
    ///
    /// When the tensor has no shards on the sharded compute device: it was never made there, or
    /// was dropped or consumed by an op before this use.
    pub fn get(&self, id: &TensorId) -> Sharding {
        *self.by_id.get(id).unwrap_or_else(|| {
            panic!(
                "Tensor {id:?} has no shards on the sharded compute device: it was never made \
                 there, or was dropped or consumed by an op before this use"
            )
        })
    }

    pub fn insert(&mut self, id: TensorId, sharding: Sharding) {
        self.by_id.insert(id, sharding);
    }

    /// Forgets the tensor, returning its sharding if it had one.
    pub fn remove(&mut self, id: &TensorId) -> Option<Sharding> {
        self.by_id.remove(id)
    }
}
