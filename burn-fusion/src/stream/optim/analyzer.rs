use super::OptimizationId;
use crate::stream::{optim::StreamOptimizations, TensorOpsDescription};
use std::marker::PhantomData;

/// The stream optimizer works by keeping track of all possible optimizations for the current graph path.
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
    candidates: Vec<OptimizationId>,
    availables: Vec<(OptimizationId, usize)>,
    found: Option<OptimizationId>,
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

    /// Follow the current path on the provided graph with the start/end condition.
    ///
    /// # Notes
    ///
    /// It is assumed that this function will be called for each new edge added to the graph (for
    /// each new operation). Only one graph can be cached at a time.
    pub fn update(
        &mut self,
        optimizations: &StreamOptimizations<O>,
        stream: &[TensorOpsDescription],
        mode: AnalysisMode,
    ) -> StreamAnalysisUpdate {
        if stream.is_empty() {
            // When the graph is empty, we use the condition as the first operation to determine
            // the new possible opitmizations.
            let ops = match mode {
                AnalysisMode::LazyExecution { next_ops } => next_ops,
                AnalysisMode::Sync => return StreamAnalysisUpdate::ExploreOptimization, // Sync an empty graph doesn't make
                                                                                        // sense.
            };
            let candidates = optimizations.find_starting_with(ops);
            if candidates.is_empty() {
                return StreamAnalysisUpdate::ExploreOptimization;
            }
            self.candidates = candidates;
            return StreamAnalysisUpdate::WaitForOptimization;
        }

        if let Some(candidate) = self.found {
            return StreamAnalysisUpdate::ExecuteOptimization(candidate);
        }

        // Invalidate candidates.
        let mut invalidated_candidate = Vec::new();
        for id in self.candidates.iter() {
            let item = optimizations.get_unchecked(*id);
            let next_ops = stream.last().expect("Validated earlier");
            let next_ops_index = stream.len() - 1;
            let next_ops_candidate = match item.stream.get(next_ops_index) {
                Some(val) => val,
                None => {
                    // Graph of different size, invalidated.
                    invalidated_candidate.push(*id);
                    continue;
                }
            };

            if next_ops_candidate != next_ops {
                // Graph with different node at the current position, invalidated.
                invalidated_candidate.push(*id);
                continue;
            }

            // Is it optimal?
            if item.stream.len() == stream.len() {
                let ops = match mode {
                    AnalysisMode::LazyExecution { next_ops } => next_ops,
                    AnalysisMode::Sync => {
                        self.found = Some(*id);
                        break;
                    }
                };

                if item.end_conditions.contains(ops) {
                    self.found = Some(*id);
                    break;
                } else {
                    self.availables.push((*id, stream.len()));
                    invalidated_candidate.push(*id);
                }
            }
        }

        if let Some(id) = self.found {
            return StreamAnalysisUpdate::ExecuteOptimization(id);
        }

        let mut updated_candidates = Vec::new();
        core::mem::swap(&mut updated_candidates, &mut self.candidates);

        self.candidates = updated_candidates
            .into_iter()
            .filter(|candidate| !invalidated_candidate.contains(candidate))
            .collect();

        if self.candidates.is_empty() {
            StreamAnalysisUpdate::ExploreOptimization
        } else {
            StreamAnalysisUpdate::WaitForOptimization
        }
    }

    pub fn found_optimization(&self, stream: &[TensorOpsDescription]) -> Option<OptimizationId> {
        if let Some(id) = self.found {
            return Some(id);
        }

        let existing_optim = self
            .availables
            .iter()
            .find(|(_candidate, len)| *len == stream.len());

        if let Some((id, _)) = existing_optim {
            Some(*id)
        } else {
            None
        }
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
pub enum StreamAnalysisUpdate {
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
            StreamAnalysisUpdate::ExploreOptimization,
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
            StreamAnalysisUpdate::WaitForOptimization,
            false, // Async
        );

        stream.assert_updates(
            &mut optimizations,
            &mut analysis,
            AssertUpdatesOptions::OperationsIndex(1..2),
            StreamAnalysisUpdate::ExecuteOptimization(id),
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
            StreamAnalysisUpdate::WaitForOptimization,
            false, // Async
        );
        stream.assert_updates(
            &mut optimizations,
            &mut analysis,
            AssertUpdatesOptions::OperationsIndex(2..3),
            StreamAnalysisUpdate::ExecuteOptimization(id),
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
            StreamAnalysisUpdate::WaitForOptimization,
            false, // Async
        );
        stream2.assert_updates(
            &mut optimizations,
            &mut analysis2,
            AssertUpdatesOptions::OperationsIndex(0..2),
            StreamAnalysisUpdate::WaitForOptimization,
            false, // Async
        );

        stream1.assert_updates(
            &mut optimizations,
            &mut analysis1,
            AssertUpdatesOptions::OperationsIndex(2..3), // First end condition.
            StreamAnalysisUpdate::ExecuteOptimization(id),
            false, // Async
        );
        stream2.assert_updates(
            &mut optimizations,
            &mut analysis2,
            AssertUpdatesOptions::OperationsIndex(2..3), // Second end condition.
            StreamAnalysisUpdate::ExecuteOptimization(id),
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
            StreamAnalysisUpdate::WaitForOptimization,
            false, // Async
        );
        stream2.assert_updates(
            &mut optimizations,
            &mut analysis2,
            AssertUpdatesOptions::OperationsIndex(0..3),
            StreamAnalysisUpdate::WaitForOptimization,
            false, // Async
        );

        stream1.assert_updates(
            &mut optimizations,
            &mut analysis1,
            AssertUpdatesOptions::OperationsIndex(3..4),
            StreamAnalysisUpdate::ExecuteOptimization(optimization_stream1),
            false, // Async
        );
        stream2.assert_updates(
            &mut optimizations,
            &mut analysis2,
            AssertUpdatesOptions::OperationsIndex(3..4),
            StreamAnalysisUpdate::ExecuteOptimization(optimization_stream2),
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
            AssertUpdatesOptions::OperationsIndex(0..3),
            StreamAnalysisUpdate::WaitForOptimization,
            false, // Async
        );

        // But is different.
        stream2.assert_updates(
            &mut optimizations,
            &mut analysis,
            AssertUpdatesOptions::OperationsIndex(3..4),
            StreamAnalysisUpdate::ExploreOptimization,
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
            optimizations: &mut StreamOptimizations<()>,
            analysis: &mut StreamAnalysis<()>,
            options: AssertUpdatesOptions,
            update: StreamAnalysisUpdate,
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
