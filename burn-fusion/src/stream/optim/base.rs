use super::starter::Starters;
use crate::stream::TensorOpsDescription;
use serde::{Deserialize, Serialize};

#[derive(Default)]
pub(crate) struct StreamOptimizations<O> {
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

impl<O> StreamOptimizations<O> {
    pub fn new() -> Self {
        Self {
            optimizations: Vec::new(),
            starters: Starters::default(),
        }
    }

    pub fn find_starting_with(&self, ops: &TensorOpsDescription) -> Vec<OptimizationId> {
        self.starters.get(ops)
    }

    pub fn get_optimization_mut_unckecked<'a>(&'a mut self, id: OptimizationId) -> &'a mut O {
        &mut self.optimizations[id].value
    }

    pub fn get_item_mut_unckecked<'a>(
        &'a mut self,
        id: OptimizationId,
    ) -> &'a mut OptimizationItem<O> {
        &mut self.optimizations[id]
    }

    pub fn get_item_unckecked<'a>(&'a self, id: OptimizationId) -> &'a OptimizationItem<O> {
        &self.optimizations[id]
    }

    pub fn add_new_optimization(&mut self, optimization: OptimizationItem<O>) -> OptimizationId {
        let new_id = self.optimizations.len();

        self.starters
            .insert(optimization.stream.first().unwrap(), new_id);
        self.optimizations.push(optimization);

        new_id
    }
}
/// Create an optimization.
pub(crate) trait OptimizationFactory<T> {
    /// Call only when a new optimization is found.
    fn create(&self) -> T;
}
