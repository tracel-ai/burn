use burn_ir::OperationIr;

use crate::stream::store::{ExecutionPlanId, ExecutionPlanStore, ExecutionTrigger};

/// Compare each operation in the list of operations provided by the [store](OperationsStore)
/// to verify if the newly added operations match the original list.
///
/// It is used by the [policy](crate::stream::execution::Policy) to check each candidate as well
/// as to verify if a list of operations is optimal to execute based on their triggers.
#[derive(Debug)]
pub(crate) struct OperationsValidator<ID> {
    /// The ID used to retrieve the operation list.
    pub(crate) id: ID,
    /// The current [state](MatchingState).
    pub(crate) state: ValidatorState,
}

/// The state of the validator.
#[derive(Debug)]
pub(crate) enum ValidatorState {
    /// A matching operation list has been found.
    Found { size: usize },
    /// No matching operation list has been found.
    Invalidated,
    /// Potentially going to find a matching operation list when more operations are added.
    Validating,
}

/// Provides a list of operations based on an Id.
pub(crate) trait OperationsStore {
    /// The type used for the identifier.
    type Id: Copy;

    /// retrieve the list of operations corresponding on the provided id.
    fn get(&self, id: Self::Id) -> &[OperationIr];
}

impl<ID> OperationsValidator<ID> {
    /// Create a new validator.
    pub(crate) fn new(id: ID) -> Self {
        Self {
            id,
            state: ValidatorState::Validating,
        }
    }

    /// Update the state of the validator based on the newly added operation.
    pub(crate) fn update<S>(&mut self, added: &OperationIr, added_position: usize, store: &S)
    where
        S: OperationsStore<Id = ID>,
        ID: PartialEq + Copy,
    {
        match &self.state {
            ValidatorState::Found { size: _ } => return,
            ValidatorState::Invalidated => return,
            ValidatorState::Validating => {}
        };

        let item = store.get(self.id);
        let operation_candidate = match item.get(added_position) {
            Some(val) => val,
            None => {
                self.state = ValidatorState::Invalidated;
                return;
            }
        };

        if operation_candidate != added {
            self.state = ValidatorState::Invalidated;
            return;
        }

        // Finished
        if item.len() == added_position + 1 {
            self.state = ValidatorState::Found { size: item.len() };
        }
    }
}

/// [Operations store](OperationsStore) used to retrieve the list of operations for a trigger.
#[derive(new)]
pub(crate) struct TriggerOperationsStore<'a, O> {
    id: ExecutionPlanId,
    store: &'a ExecutionPlanStore<O>,
}

/// Validates when operations match a trigger.
#[derive(Debug)]
pub(crate) enum TriggerValidator {
    OnOperations {
        matching: OperationsValidator<TriggerId>,
        progress: TriggerProgress,
    },
    Always,
    OnSync,
}

/// The progress made into the trigger validation process.
#[derive(Debug)]
pub(crate) enum TriggerProgress {
    /// When the validation hasn't started.
    NotInit,
    /// The number of operations that have been checked.
    NumChecked(usize),
}

/// An execution plan can have many triggers, so we use the position in the list to identify a
/// trigger.
pub(crate) type TriggerId = usize;

impl<O> OperationsStore for TriggerOperationsStore<'_, O> {
    type Id = TriggerId;

    fn get(&self, id: Self::Id) -> &[OperationIr] {
        match &self.store.get_unchecked(self.id).triggers[id] {
            ExecutionTrigger::OnOperations(operations) => operations,
            ExecutionTrigger::OnSync => &[],
            ExecutionTrigger::Always => &[],
        }
    }
}

/// [Operations store](OperationsStore) used to retrieve the list of operations for an
/// [execution plan](crate::stream::store::ExecutionPlan).
#[derive(new)]
pub(crate) struct ExecutionPlanOperationsStore<'a, O> {
    store: &'a ExecutionPlanStore<O>,
}

impl<O> OperationsStore for ExecutionPlanOperationsStore<'_, O> {
    type Id = ExecutionPlanId;

    fn get(&self, id: Self::Id) -> &[OperationIr] {
        &self.store.get_unchecked(id).operations
    }
}
