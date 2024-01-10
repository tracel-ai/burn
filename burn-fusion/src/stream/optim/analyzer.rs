use super::{OptimizationFactory, OptimizationId, OptimizationItem};
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
pub(crate) struct OptimizationAnalyzer<O> {
    candidates: Vec<OptimizationId>,
    availables: Vec<(OptimizationId, usize)>,
    found: Option<OptimizationId>,
    _item_type: PhantomData<O>,
}

impl<O> OptimizationAnalyzer<O> {
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
    pub fn new_operation_added(
        &mut self,
        cache: &mut StreamOptimizations<O>,
        stream: &[TensorOpsDescription],
        condition: Condition,
    ) -> OptimizationAnalysis {
        if stream.is_empty() {
            // When the graph is empty, we use the condition as the first operation to determine
            // the new possible opitmizations.
            let ops = match condition {
                Condition::NextOps(ops) => ops,
                Condition::Sync => return OptimizationAnalysis::NoneAvailable, // Sync an empty graph doesn't make
                                                                               // sense.
            };
            let candidates = cache.find_starting_with(ops);
            if candidates.is_empty() {
                return OptimizationAnalysis::NoneAvailable;
            }
            self.candidates = candidates;
            return OptimizationAnalysis::FutureAvailable;
        }

        if let Some(candidate) = self.found {
            return OptimizationAnalysis::Found(candidate);
        }

        // Invalidate candidates.
        let mut invalidated_candidate = Vec::new();
        for id in self.candidates.iter() {
            let item = cache.get_item_unckecked(*id);
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
                let ops = match condition {
                    Condition::NextOps(ops) => ops,
                    Condition::Sync => {
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
            return OptimizationAnalysis::Found(id);
        }

        let mut updated_candidates = Vec::new();
        core::mem::swap(&mut updated_candidates, &mut self.candidates);

        self.candidates = updated_candidates
            .into_iter()
            .filter(|candidate| !invalidated_candidate.contains(candidate))
            .collect();

        if self.candidates.is_empty() {
            OptimizationAnalysis::NoneAvailable
        } else {
            OptimizationAnalysis::FutureAvailable
        }
    }

    /// Signal the completion of a graph path that reached a new optimization.
    ///
    /// # Notes
    ///
    /// The optimization factory will only be called if the optimization is on a new graph.
    /// When the optimization already exists, but with a different end condition, a new end
    /// condition will be registered, but the old optimization will be used in following call. This
    /// is intended since we want to factory to be called only once per graph, but reused as much as
    /// possible.
    pub fn new_optimization_built<Factory: OptimizationFactory<O>>(
        &mut self,
        cache: &mut StreamOptimizations<O>,
        factory: &Factory,
        graph: Vec<TensorOpsDescription>,
        next_ops: Option<TensorOpsDescription>,
    ) -> OptimizationId {
        let existing_optim = self
            .availables
            .iter()
            .find(|(_candidate, len)| *len == graph.len());

        if let Some((id, _)) = existing_optim {
            let optimization = cache.get_item_mut_unckecked(*id);

            if let Some(ops) = next_ops {
                optimization.end_conditions.push(ops)
            };

            return *id;
        };

        let optimization = OptimizationItem {
            stream: graph,
            end_conditions: match next_ops {
                Some(val) => vec![val],
                None => Vec::new(),
            },
            value: factory.create(),
        };

        cache.add_new_optimization(optimization)
    }

    // Signal that a new path will begin.
    pub(crate) fn reset(&mut self) {
        self.candidates.clear();
        self.availables.clear();
        self.found = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        stream::{optim::OptimizationFactory, FloatOpsDescription, UnaryOpsDescription},
        TensorDescription, TensorId, TensorStatus,
    };

    #[test]
    fn should_cache_optimization_end_condition_forced() {
        // A graph with 3 ops.
        let stream = TestStream::new(2);
        let mut cache = StreamOptimizations::default();
        let mut path = OptimizationAnalyzer::new();

        // First following
        stream.follow_misses(&mut cache, &mut path);

        // Register the action.
        let optimization = path.new_optimization_built(
            &mut cache,
            &Optimization1,
            stream.edges[0..2].to_vec(),
            None,
        );

        assert_eq!(optimization, 1);

        // Second following on the same ops.
        path.reset();
        let result1 =
            path.new_operation_added(&mut cache, &[], Condition::NextOps(&stream.edges[0]));
        assert_eq!(result1, OptimizationAnalysis::FutureAvailable);

        let result2 = path.new_operation_added(
            &mut cache,
            &stream.edges[0..1],
            Condition::NextOps(&stream.edges[1]),
        );
        assert_eq!(result2, OptimizationAnalysis::FutureAvailable);

        let result3 = path.new_operation_added(&mut cache, &stream.edges[0..2], Condition::Sync);
        match result3 {
            OptimizationAnalysis::Found(ops) => assert_eq!(ops, 1),
            _ => panic!("Should have found the cached operation"),
        };
    }

    #[test]
    fn once_found_perfect_should_always_return_found() {
        let mut stream = TestStream::new(2);
        let mut cache = StreamOptimizations::default();
        let mut path = OptimizationAnalyzer::new();
        stream.follow_misses(&mut cache, &mut path);

        // Register the action.
        let _optimization = path.new_optimization_built(
            &mut cache,
            &Optimization1,
            stream.edges[0..1].to_vec(),
            Some(stream.edges[1].clone()),
        );

        path.reset();
        stream.new_ops();
        stream.new_ops();

        let result =
            path.new_operation_added(&mut cache, &[], Condition::NextOps(&stream.edges[0]));
        assert_eq!(result, OptimizationAnalysis::FutureAvailable);

        let result = path.new_operation_added(
            &mut cache,
            &stream.edges[0..1],
            Condition::NextOps(&stream.edges[1]),
        );
        match result {
            OptimizationAnalysis::Found(ops) => assert_eq!(ops, 1),
            _ => panic!("Should have found the cached operation"),
        }

        let result = path.new_operation_added(
            &mut cache,
            &stream.edges[0..2],
            Condition::NextOps(&stream.edges[2]),
        );
        match result {
            OptimizationAnalysis::Found(ops) => assert_eq!(ops, 1),
            _ => panic!("Should have found the cached operation"),
        }
    }

    #[test]
    fn should_cache_optimization_end_condition_next_ops() {
        // A graph with 4 ops.
        let stream = TestStream::new(3);
        let mut cache = StreamOptimizations::default();
        let mut path = OptimizationAnalyzer::new();

        // First following
        stream.follow_misses(&mut cache, &mut path);

        // Register the action.
        let optimization = path.new_optimization_built(
            &mut cache,
            &Optimization1,
            stream.edges[0..2].to_vec(),
            Some(stream.edges[2].clone()),
        );

        assert_eq!(optimization, 1);

        // Second following on the same ops.
        path.reset();
        let result1 =
            path.new_operation_added(&mut cache, &[], Condition::NextOps(&stream.edges[0]));
        assert_eq!(result1, OptimizationAnalysis::FutureAvailable);

        let result2 = path.new_operation_added(
            &mut cache,
            &stream.edges[0..1],
            Condition::NextOps(&stream.edges[1]),
        );
        assert_eq!(result2, OptimizationAnalysis::FutureAvailable);

        let result3 = path.new_operation_added(
            &mut cache,
            &stream.edges[0..2],
            Condition::NextOps(&stream.edges[2]),
        );
        match result3 {
            OptimizationAnalysis::Found(ops) => assert_eq!(ops, 1),
            _ => panic!("Should have found the cached operation"),
        };
    }

    #[test]
    fn should_support_many_different_end_conditions() {
        let mut cache = StreamOptimizations::default();
        let mut graph1 = TestStream::new(2);
        graph1.register_ops(|desc| TensorOpsDescription::FloatOps(FloatOpsDescription::Exp(desc)));

        let mut graph2 = TestStream::new(2);
        graph2.register_ops(|desc| TensorOpsDescription::FloatOps(FloatOpsDescription::Log(desc)));

        let mut path = OptimizationAnalyzer::<String>::new();
        let last_edge_index = graph1.edges.len() - 1;

        // Follow graph 1 with only misses.
        graph1.follow_misses(&mut cache, &mut path);
        let _ = path.new_optimization_built(
            &mut cache,
            &Optimization1,
            graph1.edges[0..last_edge_index].to_vec(),
            Some(graph1.edges[last_edge_index].clone()),
        );

        // Follow graph 2.
        let result =
            path.new_operation_added(&mut cache, &[], Condition::NextOps(&graph2.edges[0]));
        assert_eq!(result, OptimizationAnalysis::FutureAvailable);

        let result = path.new_operation_added(
            &mut cache,
            &graph2.edges[0..1],
            Condition::NextOps(&graph2.edges[1]),
        );
        assert_eq!(result, OptimizationAnalysis::FutureAvailable);

        let result = path.new_operation_added(
            &mut cache,
            &graph2.edges[0..2],
            Condition::NextOps(&graph2.edges[2]),
        );
        assert_eq!(result, OptimizationAnalysis::NoneAvailable);

        let optimization = path.new_optimization_built(
            &mut cache,
            &Optimization2,
            graph2.edges[0..last_edge_index].to_vec(),
            Some(graph2.edges[last_edge_index].clone()),
        );
        assert_eq!(
            optimization, 1,
            "Optimization 1 should still be returned, since same graph but not same end condition."
        );
    }

    #[test]
    fn should_support_multiple_concurrent_paths() {
        // Two different graphs with a different second ops, but the same last ops.
        let mut cache = StreamOptimizations::default();
        let mut graph1 = TestStream::new(1);
        graph1.register_ops(|desc| TensorOpsDescription::FloatOps(FloatOpsDescription::Exp(desc)));
        graph1.new_ops();

        let mut graph2 = TestStream::new(1);
        graph2.register_ops(|desc| TensorOpsDescription::FloatOps(FloatOpsDescription::Cos(desc)));
        graph2.new_ops();

        let mut path = OptimizationAnalyzer::<String>::new();

        // Follow graph 1 with only misses.
        graph1.follow_misses(&mut cache, &mut path);

        // Register the opitmization 1 for graph 1.
        let last_edge_index = graph1.edges.len() - 1;
        let _ = path.new_optimization_built(
            &mut cache,
            &Optimization1,
            graph1.edges[0..last_edge_index].to_vec(),
            Some(graph1.edges[last_edge_index].clone()),
        );

        // Follow graph 2 and register a new optimization.
        path.reset();

        let result =
            path.new_operation_added(&mut cache, &[], Condition::NextOps(&graph2.edges[0]));
        assert_eq!(result, OptimizationAnalysis::FutureAvailable);

        let result = path.new_operation_added(
            &mut cache,
            &graph2.edges[0..1],
            Condition::NextOps(&graph2.edges[1]),
        );
        assert_eq!(result, OptimizationAnalysis::FutureAvailable);

        let result = path.new_operation_added(
            &mut cache,
            &graph2.edges[0..2],
            Condition::NextOps(&graph2.edges[2]),
        );
        assert_eq!(
            result,
            OptimizationAnalysis::NoneAvailable,
            "Should invalidate the second operation"
        );

        // Register new optimization for path 2.
        let _ = path.new_optimization_built(
            &mut cache,
            &Optimization2,
            graph2.edges[0..last_edge_index].to_vec(),
            Some(graph2.edges[last_edge_index].clone()),
        );

        // Now let's validate that the cache works.

        // New path instance on graph 1.
        path.reset();

        let result =
            path.new_operation_added(&mut cache, &[], Condition::NextOps(&graph1.edges[0]));
        assert_eq!(result, OptimizationAnalysis::FutureAvailable);

        let result = path.new_operation_added(
            &mut cache,
            &graph1.edges[0..1],
            Condition::NextOps(&graph1.edges[1]),
        );
        assert_eq!(result, OptimizationAnalysis::FutureAvailable);

        let result = path.new_operation_added(
            &mut cache,
            &graph1.edges[0..2],
            Condition::NextOps(&graph1.edges[2]),
        );
        match result {
            OptimizationAnalysis::Found(ops) => assert_eq!(ops, 1),
            _ => panic!("Should have found the cached operation"),
        };

        // New path instance on graph 2.
        path.reset();

        let result =
            path.new_operation_added(&mut cache, &[], Condition::NextOps(&graph2.edges[0]));
        assert_eq!(result, OptimizationAnalysis::FutureAvailable);

        let result = path.new_operation_added(
            &mut cache,
            &graph2.edges[0..1],
            Condition::NextOps(&graph2.edges[1]),
        );
        assert_eq!(result, OptimizationAnalysis::FutureAvailable);

        let result = path.new_operation_added(
            &mut cache,
            &graph2.edges[0..2],
            Condition::NextOps(&graph2.edges[2]),
        );
        match result {
            OptimizationAnalysis::Found(ops) => assert_eq!(ops, 2),
            _ => panic!("Should have found the cached operation"),
        };
    }

    #[derive(Default, Debug)]
    struct TestStream {
        nodes: Vec<TensorDescription>,
        edges: Vec<TensorOpsDescription>,
    }

    impl TestStream {
        /// Create a new test graph with `num_ops` operations registered.
        pub fn new(num_ops: usize) -> Self {
            let mut graph = Self::default();
            for _ in 0..num_ops {
                graph.new_ops();
            }

            graph
        }

        /// The first follow should only be cache miss.
        pub fn follow_misses(
            &self,
            cache: &mut StreamOptimizations<String>,
            path: &mut OptimizationAnalyzer<String>,
        ) {
            for i in 0..self.edges.len() {
                let result = path.new_operation_added(
                    cache,
                    &self.edges[0..i],
                    Condition::NextOps(&self.edges[i]),
                );
                assert_eq!(result, OptimizationAnalysis::NoneAvailable);
            }
        }

        /// Register a unary operation in the graph.
        pub fn register_ops<F>(&mut self, func: F)
        where
            F: Fn(UnaryOpsDescription) -> TensorOpsDescription,
        {
            self.new_empty_node();
            let desc = self.unary_description();
            self.edges.push(func(desc));
        }

        /// Add a simple operation to the graph.
        pub fn new_ops(&mut self) {
            if self.nodes.is_empty() {
                // Root node.
                self.new_empty_node();
            }

            // Out node.
            self.new_empty_node();

            self.edges
                .push(TensorOpsDescription::FloatOps(FloatOpsDescription::Log(
                    self.unary_description(),
                )));
        }

        fn new_empty_node(&mut self) {
            self.nodes.push(TensorDescription {
                id: TensorId::new(self.nodes.len() as u64),
                shape: vec![32, 32, 1],
                status: TensorStatus::NotInit,
            });
        }

        fn unary_description(&self) -> UnaryOpsDescription {
            let size = self.nodes.len();

            UnaryOpsDescription {
                input: self.nodes[size - 2].clone(),
                out: self.nodes[size - 1].clone(),
            }
        }
    }

    struct Optimization1;
    struct Optimization2;

    impl OptimizationFactory<String> for Optimization1 {
        fn create(&self) -> String {
            "Optimization1".to_string()
        }
    }

    impl OptimizationFactory<String> for Optimization2 {
        fn create(&self) -> String {
            "Optimization2".to_string()
        }
    }
}

/// Action to be made depending on the graph.
#[derive(PartialEq, Eq, Debug)]
pub enum OptimizationAnalysis {
    /// Continue exploring optimizations using the [builder](crate::OptimizationBuilder).
    NoneAvailable,
    /// The current graph indicates that an optimization may be possible in the future, so the
    /// best action is to wait for the optimization to become available.
    ///
    /// Sometimes, it can be a false positive and a new optimization should be built from scratch.
    /// Therefore it's important to keep the previous operations to rebuild the state if it
    /// happens.
    FutureAvailable,
    /// An optimization has been found, and the best action is to execute it!
    Found(OptimizationId),
}

/// When checking if an optimization is possible, a start or an end condition ensures that this optimization is
/// always optimal.
#[derive(Clone)]
pub enum Condition<'a> {
    /// The next operation that signals the start or end of the operation.
    NextOps(&'a TensorOpsDescription),
    /// When sync, we should execute the optimization if found no matter what comes next.
    Sync,
}
