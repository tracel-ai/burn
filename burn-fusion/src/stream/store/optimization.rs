use super::{InsertQuery, OptimizationIndex, SearchQuery};
use crate::stream::TensorOpsDescription;
use serde::{Deserialize, Serialize};

#[derive(Default, Serialize, Deserialize)]
pub(crate) struct OptimizationStore<O> {
    pub(super) optimizations: Vec<OptimizationItem<O>>,
    pub(super) index: OptimizationIndex,
}

#[derive(Serialize, Deserialize, Clone)]
pub(crate) enum OptimizationKind<O> {
    CustomOptimization(O),
    ExecuteIndividualOps,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) enum StoppingCriteria {
    TensorOps(TensorOpsDescription),
    Always,
    OnSync,
}

pub(crate) type OptimizationId = usize;

#[derive(Serialize, Deserialize)]
pub(crate) struct OptimizationItem<O> {
    pub(crate) stream: Vec<TensorOpsDescription>,
    pub(crate) stopping_criteria: Vec<StoppingCriteria>,
    pub(crate) value: OptimizationKind<O>,
}

impl<O> OptimizationItem<O> {
    pub fn should_stop(&self, ops: &TensorOpsDescription) -> bool {
        for item in self.stopping_criteria.iter() {
            match item {
                StoppingCriteria::TensorOps(val) => {
                    if val == ops {
                        return true;
                    }
                }
                _ => {}
            }
        }

        false
    }
}

impl<O> OptimizationStore<O> {
    pub fn new() -> Self {
        Self {
            optimizations: Vec::new(),
            index: OptimizationIndex::default(),
        }
    }

    pub fn find(&self, query: SearchQuery<'_>) -> Vec<OptimizationId> {
        self.index.find(query)
    }

    pub fn add(&mut self, optimization: OptimizationItem<O>) -> OptimizationId {
        if optimization.stream.is_empty() {
            panic!("Can't add an empty optimization.");
        }

        let id = self.optimizations.len();
        println!(
            "Add new optimization {id} => {:?}",
            optimization.stopping_criteria
        );

        self.index.insert(InsertQuery::NewOptimization {
            stream: &optimization.stream,
            id,
        });

        self.optimizations.push(optimization);

        id
    }

    pub fn get_mut_unchecked(&mut self, id: OptimizationId) -> &mut OptimizationItem<O> {
        &mut self.optimizations[id]
    }

    pub fn get_unchecked(&self, id: OptimizationId) -> &OptimizationItem<O> {
        &self.optimizations[id]
    }

    /// Add a new end condition for an optimization.
    pub fn add_stopping_criteria(&mut self, id: OptimizationId, criteria: StoppingCriteria) {
        self.optimizations[id].stopping_criteria.push(criteria)
    }
}
