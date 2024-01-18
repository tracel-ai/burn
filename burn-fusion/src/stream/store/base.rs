use super::{ExecutionPlanIndex, InsertQuery, SearchQuery};
use crate::stream::OperationDescription;
use serde::{Deserialize, Serialize};

/// The store that contains all explorations done on a device.
#[derive(Default, Serialize, Deserialize)]
pub(crate) struct ExecutionPlanStore<O> {
    plans: Vec<ExecutionPlan<O>>,
    index: ExecutionPlanIndex,
}

/// How a list of operations should be executed.
#[derive(PartialEq, Debug, Serialize, Deserialize, Clone)]
pub(crate) enum ExecutionStrategy<O> {
    /// An optimization was found, and therefore should be executed.
    Optimization(O),
    /// No optimization was found, each operation should be executed individually.
    Operations,
}

/// The trigger that indicates when to stop exploring.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub(crate) enum ExecutionTrigger {
    OnOperations(Vec<OperationDescription>),
    OnSync,
    Always,
}

/// The unique identifier for an exploration that was executed.
pub(crate) type ExecutionPlanId = usize;

/// The outcome of an exploration that can be stored.
#[derive(Serialize, Deserialize)]
pub(crate) struct ExecutionPlan<O> {
    /// The operations on which the exploration is related to.
    pub(crate) operations: Vec<OperationDescription>,
    /// The criteria that signal when this plan should be executed. Only one trigger is necessary.
    pub(crate) triggers: Vec<ExecutionTrigger>,
    /// The strategy that should be used when executing this plan.
    pub(crate) strategy: ExecutionStrategy<O>,
}

impl<O> ExecutionPlan<O> {
    /// Whether exploration should be stop in an async mode.
    pub fn should_stop_async(&self, ops: &OperationDescription) -> bool {
        for item in self.triggers.iter() {
            match item {
                ExecutionTrigger::OnOperations(val) => {
                    if &val[0] == ops {
                        return true;
                    }
                }
                ExecutionTrigger::Always => return true,
                ExecutionTrigger::OnSync => continue,
            }
        }

        false
    }
}

impl<O> ExecutionPlanStore<O> {
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
    pub fn add_trigger(&mut self, id: ExecutionPlanId, criterion: ExecutionTrigger) {
        let criteria = &mut self.plans[id].triggers;

        if !criteria.contains(&criterion) {
            criteria.push(criterion);
        }
    }
}
