use super::OptimizationId;
use crate::graph::TensorOpsDescription;
use std::{
    collections::{hash_map::DefaultHasher, HashMap},
    hash::{Hash, Hasher},
};

#[derive(Default)]
pub(crate) struct Starters {
    starter_indices: HashMap<u64, Vec<(TensorOpsDescription, usize)>>,
    starters: Vec<Vec<OptimizationId>>,
}

impl Starters {
    pub(crate) fn get(&self, ops: &TensorOpsDescription) -> Vec<OptimizationId> {
        let key = self.graph_key(ops);
        let values = match self.starter_indices.get(&key) {
            Some(val) => val,
            None => return Vec::new(),
        };

        if values.is_empty() {
            return Vec::new();
        }

        let (_, index) = match values.iter().find(|value| &value.0 == ops) {
            Some(val) => val,
            None => return Vec::new(),
        };

        let val = match self.starters.get(*index) {
            Some(value) => value.clone(),
            None => Vec::new(),
        };

        val
    }

    pub(crate) fn insert(&mut self, ops: &TensorOpsDescription, new_id: OptimizationId) {
        let key = self.graph_key(ops);
        let values = match self.starter_indices.get_mut(&key) {
            Some(val) => val,
            None => {
                // New starter ops.
                let index = self.starters.len();
                self.starters.push(vec![new_id]);
                self.starter_indices.insert(key, vec![(ops.clone(), index)]);

                return;
            }
        };
        let (_, index) = match values.iter_mut().find(|value| &value.0 == ops) {
            Some(val) => val,
            None => {
                // New with hash collision.
                let index = self.starters.len();
                self.starters.push(vec![new_id]);
                values.push((ops.clone(), index));
                return;
            }
        };

        // New optimization for an existing starter.
        self.starters
            .get_mut(*index)
            .expect("Should exist")
            .push(new_id);
    }

    fn graph_key(&self, ops: &TensorOpsDescription) -> u64 {
        let mut hasher = DefaultHasher::new();
        ops.hash(&mut hasher);
        hasher.finish()
    }
}
