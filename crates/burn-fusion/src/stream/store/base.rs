use std::sync::Arc;

use crate::search::BlockOptimization;

use super::{ExecutionPlanIndex, InsertQuery, SearchQuery};
use burn_ir::OperationIr;
use serde::{Deserialize, Serialize};

/// The store that contains all explorations done on a device.
#[derive(Default)]
pub(crate) struct ExecutionPlanStore<O> {
    plans: Vec<ExecutionPlan<O>>,
    index: ExecutionPlanIndex,
}

/// How a list of operations should be executed.
#[derive(PartialEq, Debug, Clone)]
pub(crate) enum ExecutionStrategy<O> {
    /// An optimization was found, and therefore should be executed.
    Optimization { opt: O, ordering: Arc<Vec<usize>> },
    /// No optimization was found, each operation should be executed individually.
    Operations { ordering: Arc<Vec<usize>> },
    /// A composition of multiple execution strategies.
    Composed(Vec<Box<Self>>),
}

/// The trigger that indicates when to stop exploring.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub(crate) enum ExecutionTrigger {
    OnOperations(Vec<OperationIr>),
    OnSync,
    Always,
}

/// The unique identifier for an exploration that was executed.
pub(crate) type ExecutionPlanId = usize;

/// The outcome of an exploration that can be stored.
#[derive(Debug)]
pub(crate) struct ExecutionPlan<O> {
    /// The operations on which the exploration is related to.
    pub(crate) operations: Vec<OperationIr>,
    /// The criteria that signal when this plan should be executed. Only one trigger is necessary.
    pub(crate) triggers: Vec<ExecutionTrigger>,
    /// The optimization that should be used when executing this plan.
    pub(crate) optimization: BlockOptimization<O>,
}

impl<O: core::fmt::Debug> ExecutionPlanStore<O> {
    pub fn new() -> Self {
        Self {
            plans: Vec::new(),
            index: ExecutionPlanIndex::default(),
        }
    }

    pub fn find(&self, query: SearchQuery<'_>) -> Vec<ExecutionPlanId> {
        self.index.find(query)
    }

    pub fn add(&mut self, exploration: ExecutionPlan<O>) -> ExecutionPlanId {
        if exploration.operations.is_empty() {
            panic!("Can't add an empty optimization.");
        }

        let id = self.plans.len();

        self.index.insert(InsertQuery::NewPlan {
            operations: &exploration.operations,
            id,
        });

        self.plans.push(exploration);

        id
    }

    pub fn get_mut_unchecked(&mut self, id: ExecutionPlanId) -> &mut ExecutionPlan<O> {
        &mut self.plans[id]
    }

    pub fn get_unchecked(&self, id: ExecutionPlanId) -> &ExecutionPlan<O> {
        &self.plans[id]
    }

    /// Add a new end condition for an optimization.
    pub fn add_trigger(&mut self, id: ExecutionPlanId, trigger: ExecutionTrigger) {
        let criteria = &mut self.plans[id].triggers;

        if !criteria.contains(&trigger) {
            criteria.push(trigger);
        }
    }
}
