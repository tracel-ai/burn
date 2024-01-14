use super::{InsertQuery, OptimizationIndex, SearchQuery};
use crate::stream::OperationDescription;
use serde::{Deserialize, Serialize};

/// The store that contains all explorations done on a device.
#[derive(Default, Serialize, Deserialize)]
pub(crate) struct ExplorationStore<O> {
    items: Vec<Exploration<O>>,
    index: OptimizationIndex,
}

/// How a stream should be executed.
#[derive(PartialEq, Debug, Serialize, Deserialize, Clone)]
pub(crate) enum ExecutionStrategy<O> {
    /// An optmization was found for this stream, and therefore should be executed.
    Optimization(O),
    /// No optimization was found for this stream, each operation should be executed individually.
    Operations,
}

/// The criterion exposing when to stop exploring on a stream.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub(crate) enum StopCriterion {
    OnOperation(OperationDescription),
    OnSync,
    Always,
}

/// The unique identifier for an exploration that was executed.
pub(crate) type ExplorationId = usize;

/// The outcome of an exploration that can be stored.
#[derive(Serialize, Deserialize)]
pub(crate) struct Exploration<O> {
    /// The stream on which the exploration is related to.
    pub(crate) stream: Vec<OperationDescription>,
    /// The criteria that signal when this stream is optimal to be executed.
    pub(crate) criteria: Vec<StopCriterion>,
    /// The strategy that should be used when executing this stream.
    pub(crate) execution: ExecutionStrategy<O>,
}

impl<O> Exploration<O> {
    /// Whether exploration should be stop in an async mode.
    pub fn should_stop_async(&self, ops: &OperationDescription) -> bool {
        println!("Should stop async.");
        for item in self.criteria.iter() {
            match item {
                StopCriterion::OnOperation(val) => {
                    if val == ops {
                        return true;
                    }
                }
                StopCriterion::Always => return true,
                StopCriterion::OnSync => continue,
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

    pub fn add(&mut self, exploration: Exploration<O>) -> ExplorationId {
        if exploration.stream.is_empty() {
            panic!("Can't add an empty optimization.");
        }

        let id = self.items.len();

        self.index.insert(InsertQuery::NewOptimization {
            stream: &exploration.stream,
            id,
        });

        self.items.push(exploration);

        id
    }

    pub fn get_mut_unchecked(&mut self, id: ExplorationId) -> &mut Exploration<O> {
        &mut self.items[id]
    }

    pub fn get_unchecked(&self, id: ExplorationId) -> &Exploration<O> {
        &self.items[id]
    }

    /// Add a new end condition for an optimization.
    pub fn add_stop_criterion(&mut self, id: ExplorationId, criterion: StopCriterion) {
        let criteria = &mut self.items[id].criteria;

        if !criteria.contains(&criterion) {
            criteria.push(criterion);
        }
    }
}
