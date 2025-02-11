use burn_ir::OperationIr;

use super::validator::{
    ExecutionPlanOperationsStore, TriggerOperationsStore, TriggerProgress, TriggerValidator,
    ValidatorState,
};
use super::ExecutionMode;
use crate::stream::execution::validator::OperationsValidator;
use crate::stream::store::{ExecutionPlanId, ExecutionPlanStore, ExecutionTrigger, SearchQuery};
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
    /// List of potential execution plans that are compatible with current stream segment
    candidates: Vec<OperationsValidator<ExecutionPlanId>>,
    /// List of candidate execution plans that have been found; we can still keep searching
    /// to potentially find a better one.
    availables: Vec<AvailableItem>,
    /// The found execution plan that should be executed, along with the number of operations
    /// in the plan.
    found: Option<(ExecutionPlanId, usize)>,
    /// The number of operations that have been analyzed
    num_operations: usize,
    _item_type: PhantomData<O>,
}

#[derive(new)]
struct AvailableItem {
    id: ExecutionPlanId,
    size: usize,
    triggers: Vec<TriggerValidator>,
}

/// Action to be made depending on the stream.
#[derive(PartialEq, Eq, Debug)]
pub enum Action {
    /// Continue exploring using the [builder](crate::OptimizationBuilder).
    Explore,
    /// The current policy indicates that an exploration may be possible in the future, so the
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
        operations: &[OperationIr],
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
    pub fn update(&mut self, store: &ExecutionPlanStore<O>, operation: &OperationIr) {
        // reset the candidates to contain all execution plans starting with the operation.
        if self.num_operations == 0 {
            self.candidates = store
                .find(SearchQuery::PlansStartingWith(operation))
                .into_iter()
                .map(OperationsValidator::new)
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

    /// Check which candidates can be removed, and which one can go from
    /// 'candidate' to 'available'
    fn check_candidates(&mut self, store: &ExecutionPlanStore<O>) {
        let mut candidates_to_remove = Vec::new();

        for candidate in self.candidates.iter() {
            match candidate.state {
                ValidatorState::Found { size } => {
                    let item = store.get_unchecked(candidate.id);
                    let mut triggers = Vec::with_capacity(item.triggers.len());

                    for (index, trigger) in item.triggers.iter().enumerate() {
                        triggers.push(match trigger {
                            ExecutionTrigger::OnOperations(_) => TriggerValidator::OnOperations {
                                matching: OperationsValidator::new(index),
                                progress: TriggerProgress::NotInit,
                            },
                            ExecutionTrigger::OnSync => TriggerValidator::OnSync,
                            ExecutionTrigger::Always => TriggerValidator::Always,
                        });
                    }

                    self.availables
                        .push(AvailableItem::new(candidate.id, size, triggers));
                    candidates_to_remove.push(candidate.id);
                }
                ValidatorState::Invalidated => {
                    candidates_to_remove.push(candidate.id);
                }
                ValidatorState::Validating => {}
            };
        }

        let mut updated_candidates = Vec::new();
        core::mem::swap(&mut updated_candidates, &mut self.candidates);

        self.candidates = updated_candidates
            .into_iter()
            .filter(|candidate| !candidates_to_remove.iter().any(|id| id == &candidate.id))
            .collect();
    }

    fn check_availables(&mut self) {
        for available in self.availables.iter() {
            for trigger in available.triggers.iter() {
                match trigger {
                    TriggerValidator::OnOperations {
                        matching,
                        progress: _,
                    } => {
                        if let ValidatorState::Found {
                            size: _size_of_trigger,
                        } = matching.state
                        {
                            self.found = Some((available.id, available.size));
                            return;
                        }
                    }
                    TriggerValidator::Always => {
                        self.found = Some((available.id, available.size));
                        return;
                    }
                    TriggerValidator::OnSync => {
                        // Does nothing during an update.
                    }
                }
            }
        }
    }

    fn update_candidates(&mut self, store: &ExecutionPlanStore<O>, operation: &OperationIr) {
        let main_store = ExecutionPlanOperationsStore::new(store);

        self.candidates
            .iter_mut()
            .for_each(|candidate| candidate.update(operation, self.num_operations, &main_store));
    }

    fn update_availables(&mut self, store: &ExecutionPlanStore<O>, operation: &OperationIr) {
        self.availables.iter_mut().for_each(|available| {
            let store_trigger = TriggerOperationsStore::new(available.id, store);

            available.triggers.iter_mut().for_each(|trigger| {
                if let TriggerValidator::OnOperations { matching, progress } = trigger {
                    match progress {
                        TriggerProgress::NotInit => {
                            *progress = TriggerProgress::NumChecked(0);
                        }
                        TriggerProgress::NumChecked(num_check) => {
                            matching.update(operation, *num_check, &store_trigger);
                            *num_check += 1;
                        }
                    }
                }
            });
        });
    }

    fn action_lazy(&self, operations: &[OperationIr]) -> Action {
        if !self.candidates.is_empty() {
            return Action::Defer;
        }

        for available in self.availables.iter() {
            if available.size == operations.len() {
                return Action::Defer;
            }

            for trigger in available.triggers.iter() {
                if let TriggerValidator::OnOperations {
                    matching,
                    progress: _,
                } = trigger
                {
                    if let ValidatorState::Validating = matching.state {
                        return Action::Defer;
                    }
                }
            }
        }

        Action::Explore
    }

    fn action_sync(&self, operations: &[OperationIr], store: &ExecutionPlanStore<O>) -> Action {
        for available in self.availables.iter() {
            if available.size == operations.len() {
                return Action::Execute(available.id);
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
    use burn_ir::{FloatOperationIr, TensorId, TensorIr, TensorStatus, UnaryOpIr};
    use burn_tensor::DType;

    use super::*;
    use crate::stream::store::{ExecutionPlan, ExecutionStrategy, ExecutionTrigger};
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
        tensors: Vec<TensorIr>,
        operations: Vec<OperationIr>,
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

            self.operations.push(OperationIr::Float(
                DType::F32,
                FloatOperationIr::Log(self.unary_description()),
            ));
        }

        fn new_empty_node(&mut self, id: u64) {
            self.tensors.push(TensorIr {
                id: TensorId::new(id),
                shape: vec![32, 32, 1],
                status: TensorStatus::NotInit,
                dtype: DType::F32,
            });
        }

        fn unary_description(&self) -> UnaryOpIr {
            let size = self.tensors.len();

            UnaryOpIr {
                input: self.tensors[size - 2].clone(),
                out: self.tensors[size - 1].clone(),
            }
        }
    }
}
