use super::ExecutionMode;
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
    // The potential explocations that we could apply to the current stream, but their streams
    // still exceed the size of the current stream.
    candidates: Vec<ExecutionPlanId>,
    // Optimizations that we find during the `updates`, but none of their `trigger` matches the
    // current stream.
    availables: Vec<(ExecutionPlanId, usize)>,
    // Optimization that we find during the `updates` where one of its `triggers` matches the
    // current stream.
    found: Option<(ExecutionPlanId, usize)>,
    // The number of operations analyzed.
    num_operations: usize,
    _item_type: PhantomData<O>,
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
            ExecutionMode::Lazy => {
                if !self.candidates.is_empty() {
                    return Action::Defer;
                }

                Action::Explore
            }
            ExecutionMode::Sync => {
                // If an execution plan covers the _whole_ operation list, we return it, else we explore new
                // plans.
                for (id, length) in self.availables.iter() {
                    if *length == operations.len() {
                        return Action::Execute(*id);
                    }
                }

                for candidate in self.candidates.iter() {
                    let item = store.get_unchecked(*candidate);

                    // The candidate can actually be executed, since the stream is of the same
                    // size.
                    if item.operations.len() == operations.len() {
                        return Action::Execute(*candidate);
                    }
                }

                Action::Explore
            }
        }
    }

    /// Update the policy state.
    pub fn update(&mut self, store: &ExecutionPlanStore<O>, ops: &OperationDescription) {
        if self.num_operations == 0 {
            self.candidates = store.find(SearchQuery::PlansStartingWith(ops));
        } else {
            self.analyze_candidates(store, ops);
        }

        self.num_operations += 1;
    }

    // Reset the state of the policy.
    pub fn reset(&mut self) {
        self.candidates.clear();
        self.availables.clear();
        self.num_operations = 0;
        self.found = None;
    }

    fn analyze_candidates(
        &mut self,
        store: &ExecutionPlanStore<O>,
        operation: &OperationDescription,
    ) {
        // The index starts at zero.
        let mut invalidated_candidates = Vec::new();

        for id in self.candidates.iter() {
            let item = store.get_unchecked(*id);

            if item.operations.len() == self.num_operations + 1
                && item.operations.last().unwrap() == operation
                && item.triggers.contains(&ExecutionTrigger::Always)
            {
                self.found = Some((*id, item.operations.len()));
                break;
            }

            if item.operations.len() == self.num_operations {
                if item.should_stop_async(operation) {
                    self.found = Some((*id, item.operations.len()));
                    break;
                } else {
                    // The plan is available, but the current operation isn't an existing
                    // trigger for this plan, so we may find a better plan by
                    // still growing the operation list.
                    self.availables.push((*id, item.operations.len()));
                    invalidated_candidates.push(*id);
                    continue;
                }
            };

            let operation_candidate = match item.operations.get(self.num_operations) {
                Some(val) => val,
                None => {
                    // Operation list of different size, invalidated.
                    invalidated_candidates.push(*id);
                    continue;
                }
            };

            if operation_candidate != operation {
                // Operation list with different node at the current position, invalidated.
                invalidated_candidates.push(*id);
                continue;
            }
        }

        let mut updated_candidates = Vec::new();
        core::mem::swap(&mut updated_candidates, &mut self.candidates);

        self.candidates = updated_candidates
            .into_iter()
            .filter(|candidate| !invalidated_candidates.contains(candidate))
            .collect();
    }
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
    fn given_existing_optimization_when_sync_should_execute_optim() {
        let mut store = ExecutionPlanStore::default();
        let mut policy = Policy::new();
        let stream = TestStream::new(2);

        let id = store.add(ExecutionPlan {
            operations: stream.operations.clone(),
            triggers: Vec::new(),
            strategy: ExecutionStrategy::Operations,
        });

        stream.assert_updates(
            &store,
            &mut policy,
            AssertUpdatesOptions::OperationsIndex(0..2),
            Action::Defer,
        );

        let action = policy.action(&store, &stream.operations, ExecutionMode::Sync);
        assert_eq!(action, Action::Execute(id));
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
                .map(|desc| ExecutionTrigger::OnOperation(desc.clone()))
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
                ExecutionTrigger::OnOperation(stream_1.operations[2].clone()),
                ExecutionTrigger::OnOperation(stream_2.operations[2].clone()),
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
                .map(|desc| ExecutionTrigger::OnOperation(desc.clone()))
                .collect(),
            strategy: ExecutionStrategy::Operations,
        });
        let optimization_stream_2 = store.add(ExecutionPlan {
            operations: stream_2.operations[0..3].to_vec(),
            triggers: stream_2.operations[3..4]
                .iter()
                .map(|desc| ExecutionTrigger::OnOperation(desc.clone()))
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
                .map(|desc| ExecutionTrigger::OnOperation(desc.clone()))
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
