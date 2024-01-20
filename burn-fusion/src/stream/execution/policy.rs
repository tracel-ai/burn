use super::matching::{
    MainOperationsStore, MatchingState, OnOperationProgress, OperationsTriggerStore,
    TriggerMatching,
};
use super::ExecutionMode;
use crate::stream::execution::matching::OperationsMatching;
use crate::stream::{
    store::{ExecutionPlanId, ExecutionPlanStore, ExecutionTrigger, SearchQuery},
    OperationDescription,
};
use std::marker::PhantomData;

/// The policy keeps track of all possible execution plans for the current operations.
///
/// # Details
///
/// We keep track of each new operation added and invalidate potential execution plans
/// when we see a different operation is added.
///
/// Therefore, the overhead is very minimal, since the time-complexity of checking for existing
/// execution plans scales with the number of concurrent potential plans for the current operations,
/// which isn't supposed to be big at any time.
pub(crate) struct Policy<O> {
    candidates: Vec<OperationsMatching<ExecutionPlanId>>,
    availables: Vec<(ExecutionPlanId, usize, Vec<TriggerMatching>)>,
    found: Option<(ExecutionPlanId, usize)>,
    num_operations: usize,
    _item_type: PhantomData<O>,
}

/// Action to be made depending on the stream.
#[derive(PartialEq, Eq, Debug)]
pub enum Action {
    /// Continue exploring using the [builder](crate::OptimizationBuilder).
    Explore,
    /// The current policy indicates that an explocation may be possible in the future, so the
    /// best action is to defer any execution.
    ///
    /// Sometimes, it can be a false positive and a new exploration should be built from scratch.
    /// Therefore it's important to keep the previous operations to rebuild the state if it
    /// happens.
    Defer,
    /// An exploration has been found, and the best action is to execute it!
    Execute(ExecutionPlanId),
}

impl<O> Policy<O> {
    /// Create a new policy.
    pub(crate) fn new() -> Self {
        Self {
            candidates: Vec::new(),
            availables: Vec::new(),
            found: None,
            num_operations: 0,
            _item_type: PhantomData,
        }
    }

    /// Returns the [action](Action) that should be taken given the state of the policy.
    pub fn action(
        &self,
        store: &ExecutionPlanStore<O>,
        operations: &[OperationDescription],
        mode: ExecutionMode,
    ) -> Action {
        if self.num_operations < operations.len() {
            panic!("Internal Error: Can't retrieve the policy action on a list of operations bigger than what is analyzed.");
        }

        if let Some((id, _length)) = self.found {
            return Action::Execute(id);
        }

        match mode {
            ExecutionMode::Lazy => self.action_lazy(operations),
            ExecutionMode::Sync => self.action_sync(operations, store),
        }
    }

    /// Update the policy state.
    pub fn update(&mut self, store: &ExecutionPlanStore<O>, operation: &OperationDescription) {
        if self.num_operations == 0 {
            self.candidates = store
                .find(SearchQuery::PlansStartingWith(operation))
                .into_iter()
                .map(|candidate| OperationsMatching::new(candidate))
                .collect();
        }

        self.update_candidates(store, operation);
        self.check_candidates(store);

        self.update_availables(store, operation);
        self.check_availables();
        self.num_operations += 1;
    }

    // Reset the state of the policy.
    pub fn reset(&mut self) {
        self.candidates.clear();
        self.availables.clear();

        self.num_operations = 0;
        self.found = None;
    }

    fn check_candidates(&mut self, store: &ExecutionPlanStore<O>) {
        let mut candidates_to_remove = Vec::new();

        for candidate in self.candidates.iter() {
            match candidate.state {
                MatchingState::Found { size } => {
                    let item = store.get_unchecked(candidate.id);
                    let mut triggers = Vec::with_capacity(item.triggers.len());

                    for (index, trigger) in item.triggers.iter().enumerate() {
                        triggers.push(match trigger {
                            ExecutionTrigger::OnOperations(_) => TriggerMatching::OnOperations {
                                matching: OperationsMatching::new(index),
                                progress: OnOperationProgress::NotInit,
                            },
                            ExecutionTrigger::OnSync => TriggerMatching::OnSync,
                            ExecutionTrigger::Always => TriggerMatching::Always,
                        });
                    }

                    self.availables.push((candidate.id, size, triggers));
                    candidates_to_remove.push(candidate.id);
                }
                MatchingState::Invalidated => {
                    candidates_to_remove.push(candidate.id);
                }
                MatchingState::Progressing => {}
            };
        }

        let mut updated_candidates = Vec::new();
        core::mem::swap(&mut updated_candidates, &mut self.candidates);

        self.candidates = updated_candidates
            .into_iter()
            .filter(|candidate| {
                candidates_to_remove
                    .iter()
                    .find(|id| *id == &candidate.id)
                    .is_none()
            })
            .collect();
    }

    fn check_availables(&mut self) {
        for (available, size, triggers) in self.availables.iter() {
            for trigger in triggers.iter() {
                match trigger {
                    TriggerMatching::OnOperations {
                        matching,
                        progress: _,
                    } => {
                        if let MatchingState::Found {
                            size: _size_of_trigger,
                        } = matching.state
                        {
                            self.found = Some((*available, *size));
                            return;
                        }
                    }
                    TriggerMatching::Always => {
                        self.found = Some((*available, *size));
                        return;
                    }
                    TriggerMatching::OnSync => {
                        // Does nothing during an update.
                    }
                }
            }
        }
    }

    fn update_candidates(
        &mut self,
        store: &ExecutionPlanStore<O>,
        operation: &OperationDescription,
    ) {
        let main_store = MainOperationsStore::new(store);

        self.candidates
            .iter_mut()
            .for_each(|candidate| candidate.update(operation, &main_store, self.num_operations));
    }

    fn update_availables(
        &mut self,
        store: &ExecutionPlanStore<O>,
        operation: &OperationDescription,
    ) {
        self.availables
            .iter_mut()
            .for_each(|(available, _, triggers)| {
                let store_trigger = OperationsTriggerStore::new(*available, store);

                triggers.iter_mut().for_each(|trigger| {
                    if let TriggerMatching::OnOperations { matching, progress } = trigger {
                        match progress {
                            OnOperationProgress::NotInit => {
                                *progress = OnOperationProgress::NumChecked(0);
                            }
                            OnOperationProgress::NumChecked(num_check) => {
                                matching.update(operation, &store_trigger, *num_check);
                                *num_check += 1;
                            }
                        }
                    }
                });
            });
    }

    fn action_lazy(&self, operations: &[OperationDescription]) -> Action {
        if !self.candidates.is_empty() {
            return Action::Defer;
        }

        for (_available, size, triggers) in self.availables.iter() {
            if *size == operations.len() {
                return Action::Defer;
            }

            for trigger in triggers {
                if let TriggerMatching::OnOperations {
                    matching,
                    progress: _,
                } = trigger
                {
                    if let MatchingState::Progressing = matching.state {
                        return Action::Defer;
                    }
                }
            }
        }

        Action::Explore
    }

    fn action_sync(
        &self,
        operations: &[OperationDescription],
        store: &ExecutionPlanStore<O>,
    ) -> Action {
        for (id, length, _triggers) in self.availables.iter() {
            if *length == operations.len() {
                return Action::Execute(*id);
            }
        }

        for candidate in self.candidates.iter() {
            let item = store.get_unchecked(candidate.id);

            if item.operations.len() == operations.len() {
                return Action::Execute(candidate.id);
            }
        }

        Action::Explore
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        stream::{
            store::{ExecutionPlan, ExecutionStrategy, ExecutionTrigger},
            FloatOperationDescription, UnaryOperationDescription,
        },
        TensorDescription, TensorId, TensorStatus,
    };
    use std::ops::Range;

    #[test]
    fn given_no_optimization_should_explore() {
        let store = ExecutionPlanStore::default();
        let mut policy = Policy::new();
        let stream = TestStream::new(3);

        stream.assert_updates(
            &store,
            &mut policy,
            AssertUpdatesOptions::OperationsIndex(0..3),
            Action::Explore,
        );
    }

    #[test]
    fn given_existing_optimizations_when_sync_should_execute_one_when_available() {
        let mut store = ExecutionPlanStore::default();
        let mut policy = Policy::new();
        let stream = TestStream::new(3);

        let id_1 = store.add(ExecutionPlan {
            operations: stream.operations[0..2].to_vec(),
            triggers: Vec::new(),
            strategy: ExecutionStrategy::Operations,
        });
        let _id_2 = store.add(ExecutionPlan {
            operations: stream.operations[0..3].to_vec(),
            triggers: Vec::new(),
            strategy: ExecutionStrategy::Operations,
        });

        stream.assert_updates(
            &store,
            &mut policy,
            AssertUpdatesOptions::OperationsIndex(0..2),
            Action::Defer,
        );

        let action = policy.action(&store, &stream.operations[0..2], ExecutionMode::Sync);
        assert_eq!(action, Action::Execute(id_1));
    }

    #[test]
    fn given_existing_plan_when_found_trigger_should_execute_plan() {
        let mut store = ExecutionPlanStore::default();
        let mut policy = Policy::new();

        let stream = TestStream::new(3);
        let id = store.add(ExecutionPlan {
            operations: stream.operations[0..2].to_vec(),
            triggers: stream.operations[2..3]
                .iter()
                .map(|desc| ExecutionTrigger::OnOperations(vec![desc.clone()]))
                .collect(),
            strategy: ExecutionStrategy::Operations,
        });

        stream.assert_updates(
            &store,
            &mut policy,
            AssertUpdatesOptions::OperationsIndex(0..2),
            Action::Defer,
        );
        stream.assert_updates(
            &store,
            &mut policy,
            AssertUpdatesOptions::OperationsIndex(2..3),
            Action::Execute(id),
        );
    }

    #[test]
    fn should_support_multiple_triggers() {
        let mut store = ExecutionPlanStore::default();
        let mut policy_1 = Policy::new();
        let mut policy_2 = Policy::new();

        let mut stream_1 = TestStream::new(2);
        let mut stream_2 = TestStream::new(2);

        // Create different end operation for each stream.
        let trigger_id_1 = 5;
        let trigger_id_2 = 6;
        stream_1.new_ops(trigger_id_1);
        stream_2.new_ops(trigger_id_2);

        let id = store.add(ExecutionPlan {
            operations: stream_1.operations[0..2].to_vec(),
            triggers: vec![
                ExecutionTrigger::OnOperations(vec![stream_1.operations[2].clone()]),
                ExecutionTrigger::OnOperations(vec![stream_2.operations[2].clone()]),
            ],
            strategy: ExecutionStrategy::Operations,
        });

        stream_1.assert_updates(
            &store,
            &mut policy_1,
            AssertUpdatesOptions::OperationsIndex(0..2),
            Action::Defer,
        );
        stream_2.assert_updates(
            &store,
            &mut policy_2,
            AssertUpdatesOptions::OperationsIndex(0..2),
            Action::Defer,
        );

        stream_1.assert_updates(
            &store,
            &mut policy_1,
            AssertUpdatesOptions::OperationsIndex(2..3), // First trigger.
            Action::Execute(id),
        );
        stream_2.assert_updates(
            &store,
            &mut policy_2,
            AssertUpdatesOptions::OperationsIndex(2..3), // Second trigger.
            Action::Execute(id),
        );
    }

    #[test]
    fn should_select_right_optimization() {
        let mut store = ExecutionPlanStore::default();
        let mut policy_1 = Policy::new();
        let mut policy_2 = Policy::new();

        let mut stream_1 = TestStream::new(2);
        let mut stream_2 = TestStream::new(2);

        // Create different streams after op 2.
        stream_1.new_ops(4);
        stream_1.new_ops(5);

        stream_2.new_ops(5);
        stream_2.new_ops(6);

        let optimization_stream_1 = store.add(ExecutionPlan {
            operations: stream_1.operations[0..3].to_vec(),
            triggers: stream_1.operations[3..4]
                .iter()
                .map(|desc| ExecutionTrigger::OnOperations(vec![desc.clone()]))
                .collect(),
            strategy: ExecutionStrategy::Operations,
        });
        let optimization_stream_2 = store.add(ExecutionPlan {
            operations: stream_2.operations[0..3].to_vec(),
            triggers: stream_2.operations[3..4]
                .iter()
                .map(|desc| ExecutionTrigger::OnOperations(vec![desc.clone()]))
                .collect(),
            strategy: ExecutionStrategy::Operations,
        });
        assert_ne!(optimization_stream_1, optimization_stream_2);

        stream_1.assert_updates(
            &store,
            &mut policy_1,
            AssertUpdatesOptions::OperationsIndex(0..3),
            Action::Defer,
        );
        stream_2.assert_updates(
            &store,
            &mut policy_2,
            AssertUpdatesOptions::OperationsIndex(0..3),
            Action::Defer,
        );

        stream_1.assert_updates(
            &store,
            &mut policy_1,
            AssertUpdatesOptions::OperationsIndex(3..4),
            Action::Execute(optimization_stream_1),
        );
        stream_2.assert_updates(
            &store,
            &mut policy_2,
            AssertUpdatesOptions::OperationsIndex(3..4),
            Action::Execute(optimization_stream_2),
        );
    }

    #[test]
    fn should_invalidate_wrong_optimizations() {
        let mut store = ExecutionPlanStore::default();
        let stream_1 = TestStream::new(4);
        let mut stream_2 = TestStream::new(2);
        stream_2.new_ops(6);
        stream_2.new_ops(7);

        store.add(ExecutionPlan {
            operations: stream_1.operations[0..3].to_vec(),
            triggers: stream_1.operations[3..4]
                .iter()
                .map(|desc| ExecutionTrigger::OnOperations(vec![desc.clone()]))
                .collect(),
            strategy: ExecutionStrategy::Operations,
        });

        let mut policy = Policy::new();
        // Same path as stream 1
        stream_2.assert_updates(
            &store,
            &mut policy,
            AssertUpdatesOptions::OperationsIndex(0..2),
            Action::Defer,
        );

        // But is different.
        stream_2.assert_updates(
            &store,
            &mut policy,
            AssertUpdatesOptions::OperationsIndex(2..4),
            Action::Explore,
        );
    }

    #[derive(Default, Debug)]
    struct TestStream {
        tensors: Vec<TensorDescription>,
        operations: Vec<OperationDescription>,
    }

    #[derive(Debug)]
    enum AssertUpdatesOptions {
        OperationsIndex(Range<usize>),
    }

    impl TestStream {
        /// Create a new test stream with `num_ops` operations registered.
        pub fn new(num_ops: usize) -> Self {
            let mut stream = Self::default();
            for id in 0..num_ops {
                stream.new_ops(id as u64 + 1);
            }

            stream
        }

        /// The first follow should only be cache miss.
        pub fn assert_updates(
            &self,
            optimizations: &ExecutionPlanStore<()>,
            policy: &mut Policy<()>,
            options: AssertUpdatesOptions,
            action: Action,
        ) {
            match options {
                AssertUpdatesOptions::OperationsIndex(range) => {
                    for i in range {
                        let stream = &self.operations[0..i];
                        let next_ops = &self.operations[i];
                        policy.update(optimizations, next_ops);
                        let result = policy.action(optimizations, stream, ExecutionMode::Lazy);

                        assert_eq!(result, action);
                    }
                }
            }
        }

        /// Add a simple operation to the stream.
        pub fn new_ops(&mut self, out_id: u64) {
            if self.tensors.is_empty() {
                // Root node.
                self.new_empty_node(0);
            }

            // Out node.
            self.new_empty_node(out_id);

            self.operations
                .push(OperationDescription::Float(FloatOperationDescription::Log(
                    self.unary_description(),
                )));
        }

        fn new_empty_node(&mut self, id: u64) {
            self.tensors.push(TensorDescription {
                id: TensorId::new(id),
                shape: vec![32, 32, 1],
                status: TensorStatus::NotInit,
            });
        }

        fn unary_description(&self) -> UnaryOperationDescription {
            let size = self.tensors.len();

            UnaryOperationDescription {
                input: self.tensors[size - 2].clone(),
                out: self.tensors[size - 1].clone(),
            }
        }
    }
}
