use super::{InsertQuery, OptimizationIndex, SearchQuery};
use crate::stream::TensorOpsDescription;
use serde::{Deserialize, Serialize};

/// The store that contains all explorations done on a device.
#[derive(Default, Serialize, Deserialize)]
pub(crate) struct ExplorationStore<O> {
    items: Vec<Exploration<O>>,
    index: OptimizationIndex,
}

/// How a stream should be executed.
#[derive(Serialize, Deserialize, Clone)]
pub(crate) enum ExecutionStrategy<O> {
    /// An optmization was found for this stream, and therefore should be executed.
    Optimization(O),
    /// No optimization was found for this stream, each operation should be executed individually.
    Operations,
}

/// The criterion exposing when to stop exploring on a stream.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) enum StopCriterion {
    OnOperation(TensorOpsDescription),
    OnSync,
    Always,
}

/// The unique identifier for an exploration that was executed.
pub(crate) type ExplorationId = usize;

/// The outcome of an exploration that can be stored.
#[derive(Serialize, Deserialize)]
pub(crate) struct Exploration<O> {
    /// The stream on which the exploration is related to.
    pub(crate) stream: Vec<TensorOpsDescription>,
    /// The criteria that signal when this stream is optimal to be executed.
    pub(crate) criteria: Vec<StopCriterion>,
    /// The strategy that should be used when executing this stream.
    pub(crate) execution: ExecutionStrategy<O>,
}

impl<O> Exploration<O> {
    pub fn should_stop(&self, ops: &TensorOpsDescription) -> bool {
        for item in self.criteria.iter() {
            match item {
                StopCriterion::OnOperation(val) => {
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

impl<O> ExplorationStore<O> {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            index: OptimizationIndex::default(),
        }
    }

    pub fn find(&self, query: SearchQuery<'_>) -> Vec<ExplorationId> {
        self.index.find(query)
    }

    pub fn add(&mut self, optimization: Exploration<O>) -> ExplorationId {
        if optimization.stream.is_empty() {
            panic!("Can't add an empty optimization.");
        }

        let id = self.items.len();
        println!("Add new optimization {id} => {:?}", optimization.criteria);

        self.index.insert(InsertQuery::NewOptimization {
            stream: &optimization.stream,
            id,
        });

        self.items.push(optimization);

        id
    }

    pub fn get_mut_unchecked(&mut self, id: ExplorationId) -> &mut Exploration<O> {
        &mut self.items[id]
    }

    pub fn get_unchecked(&self, id: ExplorationId) -> &Exploration<O> {
        &self.items[id]
    }

    /// Add a new end condition for an optimization.
    pub fn add_stop_criterion(&mut self, id: ExplorationId, criteria: StopCriterion) {
        self.items[id].criteria.push(criteria)
    }
}
