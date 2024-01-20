use crate::stream::{
    store::{ExecutionPlanId, ExecutionPlanStore, ExecutionTrigger},
    OperationDescription,
};

#[derive(Debug)]
pub(crate) struct OperationsMatching<ID> {
    pub(crate) id: ID,
    pub(crate) state: MatchingState,
}

pub(crate) trait OperationsStore<T: PartialEq> {
    type ID: Copy;

    fn get<'a>(&'a self, id: Self::ID) -> &'a [T];
}

#[derive(Debug)]
pub(crate) enum OnOperationProgress {
    NotInit,
    NumChecked(usize),
}

pub(crate) type TriggerIndex = usize;

#[derive(Debug)]
pub(crate) enum TriggerMatching {
    OnOperations {
        matching: OperationsMatching<TriggerIndex>,
        progress: OnOperationProgress,
    },
    Always,
    OnSync,
}

#[derive(new)]
pub(crate) struct OperationsTriggerStore<'a, O> {
    id: ExecutionPlanId,
    store: &'a ExecutionPlanStore<O>,
}

#[derive(new)]
pub(crate) struct MainOperationsStore<'a, O> {
    store: &'a ExecutionPlanStore<O>,
}

impl<ID> OperationsMatching<ID> {
    pub(crate) fn new(id: ID) -> Self {
        Self {
            id,
            state: MatchingState::Progressing,
        }
    }
}

impl<'b, O> OperationsStore<OperationDescription> for OperationsTriggerStore<'b, O> {
    type ID = TriggerIndex;

    fn get<'a>(&'a self, id: Self::ID) -> &'a [OperationDescription] {
        match &self.store.get_unchecked(self.id).triggers[id] {
            ExecutionTrigger::OnOperations(operations) => operations,
            ExecutionTrigger::OnSync => &[],
            ExecutionTrigger::Always => &[],
        }
    }
}

impl<'b, O> OperationsStore<OperationDescription> for MainOperationsStore<'b, O> {
    type ID = ExecutionPlanId;

    fn get<'a>(&'a self, id: Self::ID) -> &'a [OperationDescription] {
        &self.store.get_unchecked(id).operations
    }
}

#[derive(Debug)]
pub(crate) enum MatchingState {
    Found { size: usize },
    Invalidated,
    Progressing,
}

impl<ID> OperationsMatching<ID> {
    pub(crate) fn update<S, T>(&mut self, added: &T, store: &S, num_checked: usize)
    where
        S: OperationsStore<T, ID = ID>,
        ID: PartialEq + Copy,
        T: PartialEq,
    {
        match &self.state {
            MatchingState::Found { size: _ } => return,
            MatchingState::Invalidated => return,
            MatchingState::Progressing => {}
        };

        let item = store.get(self.id);
        let operation_candidate = match item.get(num_checked) {
            Some(val) => val,
            None => {
                self.state = MatchingState::Invalidated;
                return;
            }
        };

        if operation_candidate != added {
            self.state = MatchingState::Invalidated;
            return;
        }

        // Finished
        if item.len() == num_checked + 1 {
            self.state = MatchingState::Found { size: item.len() };
        }
    }
}
