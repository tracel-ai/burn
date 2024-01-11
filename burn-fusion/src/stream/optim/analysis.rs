use super::OptimizationId;
use crate::stream::{optim::StreamOptimizations, TensorOpsDescription};
use std::marker::PhantomData;

/// The stream analysis keeps track of all possible optimizations for the current stream.
///
/// # Details
///
/// We keep track of each new operation added to the stream and invalidate potential optimizations
/// when we see a different operation is added while keeping track of the current stream.
///
/// Therefore, the overhead is very minimal, since the time-complexity of checking for existing
/// optimizations scales with the number of concurrent potential optimizations for the current stream,
/// which isn't supposed to be big at any time.
pub(crate) struct StreamAnalysis<O> {
    // The potential optimizations that we could apply to the current stream, but their streams
    // still exceed the size of the current stream.
    candidates: Vec<OptimizationId>,
    // Optimizations that we find during the analysis, but none of their `end_conditions` matches the
    // current stream.
    availables: Vec<(OptimizationId, usize)>,
    // Optimization that we find during the analysis where one of its `end_condition` matches the
    // current stream.
    found: Option<(OptimizationId, usize)>,
    _item_type: PhantomData<O>,
}

impl<O> StreamAnalysis<O> {
    pub(crate) fn new() -> Self {
        Self {
            candidates: Vec::new(),
            availables: Vec::new(),
            found: None,
            _item_type: PhantomData,
        }
    }

    /// Update the analysis state and return the appropriate action.
    ///
    /// # Notes
    ///
    /// It is assumed that this function will be called for each new operation added to the graph (for
    /// each new operation). Only one stream can be analyzed at a time.
    pub fn update(
        &mut self,
        optimizations: &StreamOptimizations<O>,
        stream: &[TensorOpsDescription],
        mode: AnalysisMode,
    ) -> StreamAnalysisAction {
        if stream.is_empty() {
            return self.initialize_state(mode, optimizations);
        }

        if let Some((id, _length)) = self.found {
            return StreamAnalysisAction::ExecuteOptimization(id);
        }

        let (next_ops, length) = match mode {
            // Nest ops counts in the length of the stream.
            AnalysisMode::LazyExecution { next_ops } => (next_ops, stream.len() + 1),
            AnalysisMode::Sync => (
                stream
                    .last()
                    .expect("The stream should have at least 1 operation when sync"),
                stream.len(),
            ),
        };

        self.analyze_candidates(optimizations, next_ops, length);

        if let Some((id, _length)) = self.found {
            return StreamAnalysisAction::ExecuteOptimization(id);
        }

        match mode {
            AnalysisMode::LazyExecution { next_ops: _ } => {
                if self.candidates.is_empty() {
                    // Even if there are optimizations available, we aren't sure if they are the best ones
                    // we can use. Exploring more optimizations might find a new `end_condition` or
                    // even find a better optimization.
                    return StreamAnalysisAction::ExploreOptimization;
                }

                StreamAnalysisAction::WaitForOptimization
            }
            AnalysisMode::Sync => {
                // If an optimization covers the _whole_ stream, we return it, else we explore new
                // optimizations.
                for (id, length) in self.availables.iter() {
                    if *length == stream.len() {
                        return StreamAnalysisAction::ExecuteOptimization(*id);
                    }
                }

                for candidate in self.candidates.iter() {
                    let item = optimizations.get_unchecked(*candidate);

                    // The candidate can actually be executed, since the stream is of the same
                    // size.
                    if item.stream.len() == stream.len() {
                        return StreamAnalysisAction::ExecuteOptimization(*candidate);
                    }
                }

                StreamAnalysisAction::ExploreOptimization
            }
        }
    }

    /// Did the analysis find an optimization that can be applied for the current stream.
    ///
    /// # Important
    ///
    /// The optimization should optimize the _whole_ stream, not just part of it.
    pub fn found_optimal_optimization(
        &self,
        stream: &[TensorOpsDescription],
    ) -> Option<OptimizationId> {
        // If the optimization is for a stream of the same length as provided, we return it.

        if let Some((id, length)) = self.found {
            if length == stream.len() {
                return Some(id);
            }
        }

        for (id, length) in self.availables.iter() {
            if *length == stream.len() {
                return Some(*id);
            }
        }

        None
    }

    fn initialize_state(
        &mut self,
        mode: AnalysisMode,
        optimizations: &StreamOptimizations<O>,
    ) -> StreamAnalysisAction {
        self.reset();

        // When the graph is empty, we use the condition as the first operation to determine
        // the new possible opitmizations.
        let ops = match mode {
            AnalysisMode::LazyExecution { next_ops } => next_ops,
            AnalysisMode::Sync => return StreamAnalysisAction::ExploreOptimization, // Sync an empty graph doesn't make
                                                                                    // sense.
        };

        self.candidates = optimizations.find_starting_with(ops);

        if self.candidates.is_empty() {
            return StreamAnalysisAction::ExploreOptimization;
        }
        return StreamAnalysisAction::WaitForOptimization;
    }

    fn analyze_candidates(
        &mut self,
        optimizations: &StreamOptimizations<O>,
        next_ops: &TensorOpsDescription,
        stream_length: usize,
    ) {
        // The index starts at zero.
        let next_ops_index = stream_length - 1;
        let mut invalidated_candidates = Vec::new();

        for id in self.candidates.iter() {
            let item = optimizations.get_unchecked(*id);

            if item.stream.len() == next_ops_index {
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

            let next_ops_candidate = match item.stream.get(next_ops_index) {
                Some(val) => val,
                None => {
                    // Graph of different size, invalidated.
                    invalidated_candidates.push(*id);
                    continue;
                }
            };

            if next_ops_candidate != next_ops {
                // Graph with different node at the current position, invalidated.
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

    // Signal that a new path will begin.
    pub(crate) fn reset(&mut self) {
        self.candidates.clear();
        self.availables.clear();
        self.found = None;
    }
}

/// Action to be made depending on the graph.
#[derive(PartialEq, Eq, Debug)]
pub enum StreamAnalysisAction {
    /// Continue exploring optimizations using the [builder](crate::OptimizationBuilder).
    ExploreOptimization,
    /// The current graph indicates that an optimization may be possible in the future, so the
    /// best action is to wait for the optimization to become available.
    ///
    /// Sometimes, it can be a false positive and a new optimization should be built from scratch.
    /// Therefore it's important to keep the previous operations to rebuild the state if it
    /// happens.
    WaitForOptimization,
    /// An optimization has been found, and the best action is to execute it!
    ExecuteOptimization(OptimizationId),
}

/// When checking if an optimization is possible, a start or an end condition ensures that this optimization is
/// always optimal.
#[derive(Clone)]
pub enum AnalysisMode<'a> {
    /// The next operation that signals the start or end of the operation.
    LazyExecution { next_ops: &'a TensorOpsDescription },
    /// When sync, we should execute the optimization if found no matter what comes next.
    Sync,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        stream::{optim::OptimizationItem, FloatOpsDescription, UnaryOpsDescription},
        TensorDescription, TensorId, TensorStatus,
    };
    use std::ops::Range;

    #[test]
    fn given_no_optimization_should_explore() {
        let mut optimizations = StreamOptimizations::default();
        let mut analysis = StreamAnalysis::new();
        let stream = TestStream::new(3);

        stream.assert_updates(
            &mut optimizations,
            &mut analysis,
            AssertUpdatesOptions::OperationsIndex(0..3),
            StreamAnalysisAction::ExploreOptimization,
            false,
        );
    }

    #[test]
    fn given_existing_optimization_when_sync_should_execute_optim() {
        let mut optimizations = StreamOptimizations::default();
        let mut analysis = StreamAnalysis::new();

        let stream = TestStream::new(2);
        let id = optimizations.add(OptimizationItem {
            stream: stream.operations.clone(),
            end_conditions: Vec::new(),
            value: (),
        });

        stream.assert_updates(
            &mut optimizations,
            &mut analysis,
            AssertUpdatesOptions::OperationsIndex(0..1),
            StreamAnalysisAction::WaitForOptimization,
            false, // Async
        );

        stream.assert_updates(
            &mut optimizations,
            &mut analysis,
            AssertUpdatesOptions::OperationsIndex(1..2),
            StreamAnalysisAction::ExecuteOptimization(id),
            true, // Sync
        );
    }

    #[test]
    fn given_existing_optimization_when_found_end_condition_should_execute_optim() {
        let mut optimizations = StreamOptimizations::default();
        let mut analysis = StreamAnalysis::new();

        let stream = TestStream::new(3);
        let id = optimizations.add(OptimizationItem {
            stream: stream.operations[0..2].to_vec(),
            end_conditions: stream.operations[2..3].to_vec(),
            value: (),
        });

        stream.assert_updates(
            &mut optimizations,
            &mut analysis,
            AssertUpdatesOptions::OperationsIndex(0..2),
            StreamAnalysisAction::WaitForOptimization,
            false, // Async
        );
        stream.assert_updates(
            &mut optimizations,
            &mut analysis,
            AssertUpdatesOptions::OperationsIndex(2..3),
            StreamAnalysisAction::ExecuteOptimization(id),
            false, // Async
        );
    }

    #[test]
    fn should_support_multiple_end_conditions() {
        let mut optimizations = StreamOptimizations::default();
        let mut analysis1 = StreamAnalysis::new();
        let mut analysis2 = StreamAnalysis::new();

        let mut stream1 = TestStream::new(2);
        let mut stream2 = TestStream::new(2);

        // Create different end operation for each stream.
        let end_condition_id_1 = 5;
        let end_condition_id_2 = 5;
        stream1.new_ops(end_condition_id_1);
        stream2.new_ops(end_condition_id_2);

        let id = optimizations.add(OptimizationItem {
            stream: stream1.operations[0..2].to_vec(),
            end_conditions: vec![stream1.operations[2].clone(), stream2.operations[2].clone()],
            value: (),
        });

        stream1.assert_updates(
            &mut optimizations,
            &mut analysis1,
            AssertUpdatesOptions::OperationsIndex(0..2),
            StreamAnalysisAction::WaitForOptimization,
            false, // Async
        );
        stream2.assert_updates(
            &mut optimizations,
            &mut analysis2,
            AssertUpdatesOptions::OperationsIndex(0..2),
            StreamAnalysisAction::WaitForOptimization,
            false, // Async
        );

        stream1.assert_updates(
            &mut optimizations,
            &mut analysis1,
            AssertUpdatesOptions::OperationsIndex(2..3), // First end condition.
            StreamAnalysisAction::ExecuteOptimization(id),
            false, // Async
        );
        stream2.assert_updates(
            &mut optimizations,
            &mut analysis2,
            AssertUpdatesOptions::OperationsIndex(2..3), // Second end condition.
            StreamAnalysisAction::ExecuteOptimization(id),
            false, // Async
        );
    }

    #[test]
    fn should_select_right_optimization() {
        let mut optimizations = StreamOptimizations::default();
        let mut analysis1 = StreamAnalysis::new();
        let mut analysis2 = StreamAnalysis::new();

        let mut stream1 = TestStream::new(2);
        let mut stream2 = TestStream::new(2);

        // Create different streams after op 2.
        stream1.new_ops(4);
        stream1.new_ops(5);

        stream2.new_ops(5);
        stream2.new_ops(6);

        let optimization_stream1 = optimizations.add(OptimizationItem {
            stream: stream1.operations[0..3].to_vec(),
            end_conditions: stream1.operations[3..4].to_vec(),
            value: (),
        });
        let optimization_stream2 = optimizations.add(OptimizationItem {
            stream: stream2.operations[0..3].to_vec(),
            end_conditions: stream2.operations[3..4].to_vec(),
            value: (),
        });
        assert_ne!(optimization_stream1, optimization_stream2);

        stream1.assert_updates(
            &mut optimizations,
            &mut analysis1,
            AssertUpdatesOptions::OperationsIndex(0..3),
            StreamAnalysisAction::WaitForOptimization,
            false, // Async
        );
        stream2.assert_updates(
            &mut optimizations,
            &mut analysis2,
            AssertUpdatesOptions::OperationsIndex(0..3),
            StreamAnalysisAction::WaitForOptimization,
            false, // Async
        );

        stream1.assert_updates(
            &mut optimizations,
            &mut analysis1,
            AssertUpdatesOptions::OperationsIndex(3..4),
            StreamAnalysisAction::ExecuteOptimization(optimization_stream1),
            false, // Async
        );
        stream2.assert_updates(
            &mut optimizations,
            &mut analysis2,
            AssertUpdatesOptions::OperationsIndex(3..4),
            StreamAnalysisAction::ExecuteOptimization(optimization_stream2),
            false, // Async
        );
    }

    #[test]
    fn should_invalidate_wrong_optimizations() {
        let mut optimizations = StreamOptimizations::default();
        let stream1 = TestStream::new(4);
        let mut stream2 = TestStream::new(2);
        stream2.new_ops(6);
        stream2.new_ops(7);

        optimizations.add(OptimizationItem {
            stream: stream1.operations[0..3].to_vec(),
            end_conditions: stream1.operations[3..4].to_vec(),
            value: (),
        });

        let mut analysis = StreamAnalysis::new();
        // Same path as stream 1
        stream2.assert_updates(
            &mut optimizations,
            &mut analysis,
            AssertUpdatesOptions::OperationsIndex(0..2),
            StreamAnalysisAction::WaitForOptimization,
            false, // Async
        );

        // But is different.
        stream2.assert_updates(
            &mut optimizations,
            &mut analysis,
            AssertUpdatesOptions::OperationsIndex(2..4),
            StreamAnalysisAction::ExploreOptimization,
            false, // Async
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
        /// Create a new test graph with `num_ops` operations registered.
        pub fn new(num_ops: usize) -> Self {
            let mut graph = Self::default();
            for id in 0..num_ops {
                graph.new_ops(id as u64 + 1);
            }

            graph
        }

        /// The first follow should only be cache miss.
        pub fn assert_updates(
            &self,
            optimizations: &StreamOptimizations<()>,
            analysis: &mut StreamAnalysis<()>,
            options: AssertUpdatesOptions,
            update: StreamAnalysisAction,
            sync: bool,
        ) {
            match options {
                AssertUpdatesOptions::OperationsIndex(range) => {
                    let end = range.end;
                    for i in range {
                        let (mode, operations) = if sync && i == end - 1 {
                            (AnalysisMode::Sync, &self.operations[0..i + 1])
                        } else {
                            (
                                AnalysisMode::LazyExecution {
                                    next_ops: &self.operations[i],
                                },
                                &self.operations[0..i],
                            )
                        };
                        let result = analysis.update(optimizations, operations, mode);
                        assert_eq!(result, update);
                    }
                }
            }
        }

        /// Add a simple operation to the graph.
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
