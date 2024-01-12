use super::Starters;
use crate::stream::TensorOpsDescription;
use serde::{Deserialize, Serialize};

#[derive(Default, Serialize, Deserialize)]
pub(crate) struct OptimizationStore<O> {
    pub(super) optimizations: Vec<OptimizationItem<O>>,
    pub(super) starters: Starters,
}

pub(crate) type OptimizationId = usize;

#[derive(Serialize, Deserialize)]
pub(crate) struct OptimizationItem<O> {
    pub(crate) stream: Vec<TensorOpsDescription>,
    pub(crate) end_conditions: Vec<TensorOpsDescription>,
    pub(crate) value: O,
}

impl<O> OptimizationStore<O> {
    pub fn new() -> Self {
        Self {
            optimizations: Vec::new(),
            starters: Starters::default(),
        }
    }

    pub fn find_starting_with(&self, ops: &TensorOpsDescription) -> Vec<OptimizationId> {
        self.starters.get(ops)
    }

    pub fn get_mut_unchecked(&mut self, id: OptimizationId) -> &mut OptimizationItem<O> {
        &mut self.optimizations[id]
    }

    pub fn get_unchecked(&self, id: OptimizationId) -> &OptimizationItem<O> {
        &self.optimizations[id]
    }

    pub fn add(&mut self, optimization: OptimizationItem<O>) -> OptimizationId {
        let new_id = self.optimizations.len();
        self.starters
            .insert(optimization.stream.first().unwrap(), new_id);
        self.optimizations.push(optimization);

        new_id
    }

    /// Add a new end condition for an optimization.
    pub fn add_end_condition(&mut self, id: OptimizationId, end_condition: TensorOpsDescription) {
        self.optimizations[id].end_conditions.push(end_condition)
    }
}
