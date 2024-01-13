use super::ExecutionMode;
use crate::stream::{
    store::{OptimizationId, OptimizationStore, SearchQuery},
    TensorOpsDescription,
};
use std::marker::PhantomData;

/// The stream policy keeps track of all possible optimizations for the current stream.
///
/// # Details
///
/// We keep track of each new operation added to the stream and invalidate potential optimizations
/// when we see a different operation is added while keeping track of the current stream.
///
/// Therefore, the overhead is very minimal, since the time-complexity of checking for existing
/// optimizations scales with the number of concurrent potential optimizations for the current stream,
/// which isn't supposed to be big at any time.
pub(crate) struct Policy<O> {
    // The potential optimizations that we could apply to the current stream, but their streams
    // still exceed the size of the current stream.
    candidates: Vec<OptimizationId>,
    // Optimizations that we find during the `updates`, but none of their `end_conditions` matches the
    // current stream.
    availables: Vec<(OptimizationId, usize)>,
    // Optimization that we find during the `updates` where one of its `end_condition` matches the
    // current stream.
    found: Option<(OptimizationId, usize)>,
    // The size of the stream currently analyzed.
    stream_size: usize,
    _item_type: PhantomData<O>,
}

impl<O> Policy<O> {
    pub(crate) fn new() -> Self {
        Self {
            candidates: Vec::new(),
            availables: Vec::new(),
            found: None,
            stream_size: 0,
            _item_type: PhantomData,
        }
    }

    /// Returns the [action](Action) that should be taken given the state of the policy.
    pub fn action(
        &self,
        optimizations: &OptimizationStore<O>,
        stream: &[TensorOpsDescription],
        mode: ExecutionMode,
    ) -> Action {
        let num_minimum_analyzed = match mode {
            ExecutionMode::Lazy => self.stream_size - 1,
            ExecutionMode::Sync => self.stream_size,
        };

        if num_minimum_analyzed < stream.len() {
            panic!("Internal Error: Can't retrieve the policy action when the number of operations analyzed is lower than the stream itself.");
        }

        if let Some((id, _length)) = self.found {
            return Action::Execute(id);
        }

        match mode {
            ExecutionMode::Lazy => {
                if self.candidates.is_empty() {
                    // Even if there are optimizations available, we aren't sure if they are the best ones
                    // we can use. Exploring more optimizations might find a new `end_condition` or
                    // even find a better optimization.
                    return Action::Explore;
                }

                Action::Defer
            }
            ExecutionMode::Sync => {
                // If an optimization covers the _whole_ stream, we return it, else we explore new
                // optimizations.
                for (id, length) in self.availables.iter() {
                    if *length == stream.len() {
                        return Action::Execute(*id);
                    }
                }

                for candidate in self.candidates.iter() {
                    let item = optimizations.get_unchecked(*candidate);

                    // The candidate can actually be executed, since the stream is of the same
                    // size.
                    if item.stream.len() == stream.len() {
                        return Action::Execute(*candidate);
                    }
                }

                Action::Explore
            }
        }
    }

    /// Update the policy state.
    pub fn update(&mut self, store: &OptimizationStore<O>, ops: &TensorOpsDescription) {
        if self.stream_size == 0 {
            self.candidates = store.find(SearchQuery::OptimizationsStartingWith(ops));
        } else {
            self.analyze_candidates(store, ops, self.stream_size);
        }

        self.stream_size += 1;
    }

    // Reset the state of the policy.
    pub fn reset(&mut self) {
        self.candidates.clear();
        self.availables.clear();
        self.stream_size = 0;
        self.found = None;
    }

    fn analyze_candidates(
        &mut self,
        optimizations: &OptimizationStore<O>,
        next_ops: &TensorOpsDescription,
        stream_size: usize,
    ) {
        // The index starts at zero.
        let mut invalidated_candidates = Vec::new();

        for id in self.candidates.iter() {
            let item = optimizations.get_unchecked(*id);

            if item.stream.len() == stream_size {
                if item.end_conditions.contains(next_ops) {
                    self.found = Some((*id, item.stream.len()));
                    break;
                } else {
                    // The optimization is available, but the current operation isn't an existing
                    // end_condition for this optimization, so we may find a better optimization by
                    // still growing the stream.
                    self.availables.push((*id, item.stream.len()));
                    invalidated_candidates.push(*id);
                    continue;
                }
            };

            let next_ops_candidate = match item.stream.get(stream_size) {
                Some(val) => val,
                None => {
                    // Stream of different size, invalidated.
                    invalidated_candidates.push(*id);
                    continue;
                }
            };

            if next_ops_candidate != next_ops {
                // Stream with different node at the current position, invalidated.
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
    /// Continue exploring optimizations using the [builder](crate::OptimizationBuilder).
    Explore,
    /// The current policy indicates that an optimization may be possible in the future, so the
    /// best action is to defer any execution.
    ///
    /// Sometimes, it can be a false positive and a new optimization should be built from scratch.
    /// Therefore it's important to keep the previous operations to rebuild the state if it
    /// happens.
    Defer,
    /// An optimization has been found, and the best action is to execute it!
    Execute(OptimizationId),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        stream::{store::OptimizationItem, FloatOpsDescription, UnaryOpsDescription},
        TensorDescription, TensorId, TensorStatus,
    };
    use std::ops::Range;

    #[test]
    fn given_no_optimization_should_explore() {
        let store = OptimizationStore::default();
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
        let mut store = OptimizationStore::default();
        let mut policy = Policy::new();
        let stream = TestStream::new(2);

        let id = store.add(OptimizationItem {
            stream: stream.operations.clone(),
            end_conditions: Vec::new(),
            value: (),
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
    fn given_existing_optimization_when_found_end_condition_should_execute_optim() {
        let mut store = OptimizationStore::default();
        let mut policy = Policy::new();

        let stream = TestStream::new(3);
        let id = store.add(OptimizationItem {
            stream: stream.operations[0..2].to_vec(),
            end_conditions: stream.operations[2..3].to_vec(),
            value: (),
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
    fn should_support_multiple_end_conditions() {
        let mut store = OptimizationStore::default();
        let mut policy_1 = Policy::new();
        let mut policy_2 = Policy::new();

        let mut stream_1 = TestStream::new(2);
        let mut stream_2 = TestStream::new(2);

        // Create different end operation for each stream.
        let end_condition_id_1 = 5;
        let end_condition_id_2 = 5;
        stream_1.new_ops(end_condition_id_1);
        stream_2.new_ops(end_condition_id_2);

        let id = store.add(OptimizationItem {
            stream: stream_1.operations[0..2].to_vec(),
            end_conditions: vec![
                stream_1.operations[2].clone(),
                stream_2.operations[2].clone(),
            ],
            value: (),
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
            AssertUpdatesOptions::OperationsIndex(2..3), // First end condition.
            Action::Execute(id),
        );
        stream_2.assert_updates(
            &store,
            &mut policy_2,
            AssertUpdatesOptions::OperationsIndex(2..3), // Second end condition.
            Action::Execute(id),
        );
    }

    #[test]
    fn should_select_right_optimization() {
        let mut store = OptimizationStore::default();
        let mut policy_1 = Policy::new();
        let mut policy_2 = Policy::new();

        let mut stream_1 = TestStream::new(2);
        let mut stream_2 = TestStream::new(2);

        // Create different streams after op 2.
        stream_1.new_ops(4);
        stream_1.new_ops(5);

        stream_2.new_ops(5);
        stream_2.new_ops(6);

        let optimization_stream_1 = store.add(OptimizationItem {
            stream: stream_1.operations[0..3].to_vec(),
            end_conditions: stream_1.operations[3..4].to_vec(),
            value: (),
        });
        let optimization_stream_2 = store.add(OptimizationItem {
            stream: stream_2.operations[0..3].to_vec(),
            end_conditions: stream_2.operations[3..4].to_vec(),
            value: (),
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
        let mut store = OptimizationStore::default();
        let stream_1 = TestStream::new(4);
        let mut stream_2 = TestStream::new(2);
        stream_2.new_ops(6);
        stream_2.new_ops(7);

        store.add(OptimizationItem {
            stream: stream_1.operations[0..3].to_vec(),
            end_conditions: stream_1.operations[3..4].to_vec(),
            value: (),
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
        operations: Vec<TensorOpsDescription>,
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
            optimizations: &OptimizationStore<()>,
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
                .push(TensorOpsDescription::FloatOps(FloatOpsDescription::Log(
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

        fn unary_description(&self) -> UnaryOpsDescription {
            let size = self.tensors.len();

            UnaryOpsDescription {
                input: self.tensors[size - 2].clone(),
                out: self.tensors[size - 1].clone(),
            }
        }
    }
}
