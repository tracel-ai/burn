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
                AnalysisMode::Lazy { next_ops } => next_ops,
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
                    AnalysisMode::Lazy { next_ops } => next_ops,
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
        &self,
        optimizations: &mut StreamOptimizations<O>,
        factory: &Factory,
        graph: Vec<TensorOpsDescription>,
        next_ops: Option<TensorOpsDescription>,
    ) -> OptimizationId {
        let existing_optim = self
            .availables
            .iter()
            .find(|(_candidate, len)| *len == graph.len());

        if let Some((id, _)) = existing_optim {
            if let Some(ops) = next_ops {
                optimizations.add_end_condition(*id, ops);
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

        optimizations.add(optimization)
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
    Lazy { next_ops: &'a TensorOpsDescription },
    /// When sync, we should execute the optimization if found no matter what comes next.
    Sync,
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::{
//         stream::{optim::OptimizationFactory, FloatOpsDescription, UnaryOpsDescription},
//         TensorDescription, TensorId, TensorStatus,
//     };
//
//     #[test]
//     fn should_cache_optimization_end_condition_forced() {
//         // A graph with 3 ops.
//         let stream = TestStream::new(2);
//         let mut cache = StreamOptimizations::default();
//         let mut path = StreamAnalysis::new();
//
//         // First following
//         stream.follow_misses(&mut cache, &mut path);
//
//         // Register the action.
//         let optimization = path.new_optimization_built(
//             &mut cache,
//             &Optimization1,
//             stream.edges[0..2].to_vec(),
//             None,
//         );
//
//         assert_eq!(optimization, 1);
//
//         // Second following on the same ops.
//         path.reset();
//         let result1 = path.update(&mut cache, &[], ExecutionMode::Lazy(&stream.edges[0]));
//         assert_eq!(result1, StreamAnalysisUpdate::WaitForOptimization);
//
//         let result2 = path.update(
//             &mut cache,
//             &stream.edges[0..1],
//             ExecutionMode::Lazy(&stream.edges[1]),
//         );
//         assert_eq!(result2, StreamAnalysisUpdate::WaitForOptimization);
//
//         let result3 = path.update(&mut cache, &stream.edges[0..2], ExecutionMode::Sync);
//         match result3 {
//             StreamAnalysisUpdate::ExecuteOptimization(ops) => assert_eq!(ops, 1),
//             _ => panic!("Should have found the cached operation"),
//         };
//     }
//
//     #[test]
//     fn once_found_perfect_should_always_return_found() {
//         let mut stream = TestStream::new(2);
//         let mut cache = StreamOptimizations::default();
//         let mut path = StreamAnalysis::new();
//         stream.follow_misses(&mut cache, &mut path);
//
//         // Register the action.
//         let _optimization = path.new_optimization_built(
//             &mut cache,
//             &Optimization1,
//             stream.edges[0..1].to_vec(),
//             Some(stream.edges[1].clone()),
//         );
//
//         path.reset();
//         stream.new_ops();
//         stream.new_ops();
//
//         let result = path.update(&mut cache, &[], ExecutionMode::Lazy(&stream.edges[0]));
//         assert_eq!(result, StreamAnalysisUpdate::WaitForOptimization);
//
//         let result = path.update(
//             &mut cache,
//             &stream.edges[0..1],
//             ExecutionMode::Lazy(&stream.edges[1]),
//         );
//         match result {
//             StreamAnalysisUpdate::ExecuteOptimization(ops) => assert_eq!(ops, 1),
//             _ => panic!("Should have found the cached operation"),
//         }
//
//         let result = path.update(
//             &mut cache,
//             &stream.edges[0..2],
//             ExecutionMode::Lazy(&stream.edges[2]),
//         );
//         match result {
//             StreamAnalysisUpdate::ExecuteOptimization(ops) => assert_eq!(ops, 1),
//             _ => panic!("Should have found the cached operation"),
//         }
//     }
//
//     #[test]
//     fn should_cache_optimization_end_condition_next_ops() {
//         // A graph with 4 ops.
//         let stream = TestStream::new(3);
//         let mut cache = StreamOptimizations::default();
//         let mut path = StreamAnalysis::new();
//
//         // First following
//         stream.follow_misses(&mut cache, &mut path);
//
//         // Register the action.
//         let optimization = path.new_optimization_built(
//             &mut cache,
//             &Optimization1,
//             stream.edges[0..2].to_vec(),
//             Some(stream.edges[2].clone()),
//         );
//
//         assert_eq!(optimization, 1);
//
//         // Second following on the same ops.
//         path.reset();
//         let result1 = path.update(&mut cache, &[], ExecutionMode::Lazy(&stream.edges[0]));
//         assert_eq!(result1, StreamAnalysisUpdate::WaitForOptimization);
//
//         let result2 = path.update(
//             &mut cache,
//             &stream.edges[0..1],
//             ExecutionMode::Lazy(&stream.edges[1]),
//         );
//         assert_eq!(result2, StreamAnalysisUpdate::WaitForOptimization);
//
//         let result3 = path.update(
//             &mut cache,
//             &stream.edges[0..2],
//             ExecutionMode::Lazy(&stream.edges[2]),
//         );
//         match result3 {
//             StreamAnalysisUpdate::ExecuteOptimization(ops) => assert_eq!(ops, 1),
//             _ => panic!("Should have found the cached operation"),
//         };
//     }
//
//     #[test]
//     fn should_support_many_different_end_conditions() {
//         let mut cache = StreamOptimizations::default();
//         let mut graph1 = TestStream::new(2);
//         graph1.register_ops(|desc| TensorOpsDescription::FloatOps(FloatOpsDescription::Exp(desc)));
//
//         let mut graph2 = TestStream::new(2);
//         graph2.register_ops(|desc| TensorOpsDescription::FloatOps(FloatOpsDescription::Log(desc)));
//
//         let mut path = StreamAnalysis::<String>::new();
//         let last_edge_index = graph1.edges.len() - 1;
//
//         // Follow graph 1 with only misses.
//         graph1.follow_misses(&mut cache, &mut path);
//         let _ = path.new_optimization_built(
//             &mut cache,
//             &Optimization1,
//             graph1.edges[0..last_edge_index].to_vec(),
//             Some(graph1.edges[last_edge_index].clone()),
//         );
//
//         // Follow graph 2.
//         let result = path.update(&mut cache, &[], ExecutionMode::Lazy(&graph2.edges[0]));
//         assert_eq!(result, StreamAnalysisUpdate::WaitForOptimization);
//
//         let result = path.update(
//             &mut cache,
//             &graph2.edges[0..1],
//             ExecutionMode::Lazy(&graph2.edges[1]),
//         );
//         assert_eq!(result, StreamAnalysisUpdate::WaitForOptimization);
//
//         let result = path.update(
//             &mut cache,
//             &graph2.edges[0..2],
//             ExecutionMode::Lazy(&graph2.edges[2]),
//         );
//         assert_eq!(result, StreamAnalysisUpdate::ExploreOptimization);
//
//         let optimization = path.new_optimization_built(
//             &mut cache,
//             &Optimization2,
//             graph2.edges[0..last_edge_index].to_vec(),
//             Some(graph2.edges[last_edge_index].clone()),
//         );
//         assert_eq!(
//             optimization, 1,
//             "Optimization 1 should still be returned, since same graph but not same end condition."
//         );
//     }
//
//     #[test]
//     fn should_support_multiple_concurrent_paths() {
//         // Two different graphs with a different second ops, but the same last ops.
//         let mut cache = StreamOptimizations::default();
//         let mut graph1 = TestStream::new(1);
//         graph1.register_ops(|desc| TensorOpsDescription::FloatOps(FloatOpsDescription::Exp(desc)));
//         graph1.new_ops();
//
//         let mut graph2 = TestStream::new(1);
//         graph2.register_ops(|desc| TensorOpsDescription::FloatOps(FloatOpsDescription::Cos(desc)));
//         graph2.new_ops();
//
//         let mut path = StreamAnalysis::<String>::new();
//
//         // Follow graph 1 with only misses.
//         graph1.follow_misses(&mut cache, &mut path);
//
//         // Register the opitmization 1 for graph 1.
//         let last_edge_index = graph1.edges.len() - 1;
//         let _ = path.new_optimization_built(
//             &mut cache,
//             &Optimization1,
//             graph1.edges[0..last_edge_index].to_vec(),
//             Some(graph1.edges[last_edge_index].clone()),
//         );
//
//         // Follow graph 2 and register a new optimization.
//         path.reset();
//
//         let result = path.update(&mut cache, &[], ExecutionMode::Lazy(&graph2.edges[0]));
//         assert_eq!(result, StreamAnalysisUpdate::WaitForOptimization);
//
//         let result = path.update(
//             &mut cache,
//             &graph2.edges[0..1],
//             ExecutionMode::Lazy(&graph2.edges[1]),
//         );
//         assert_eq!(result, StreamAnalysisUpdate::WaitForOptimization);
//
//         let result = path.update(
//             &mut cache,
//             &graph2.edges[0..2],
//             ExecutionMode::Lazy(&graph2.edges[2]),
//         );
//         assert_eq!(
//             result,
//             StreamAnalysisUpdate::ExploreOptimization,
//             "Should invalidate the second operation"
//         );
//
//         // Register new optimization for path 2.
//         let _ = path.new_optimization_built(
//             &mut cache,
//             &Optimization2,
//             graph2.edges[0..last_edge_index].to_vec(),
//             Some(graph2.edges[last_edge_index].clone()),
//         );
//
//         // Now let's validate that the cache works.
//
//         // New path instance on graph 1.
//         path.reset();
//
//         let result = path.update(&mut cache, &[], ExecutionMode::Lazy(&graph1.edges[0]));
//         assert_eq!(result, StreamAnalysisUpdate::WaitForOptimization);
//
//         let result = path.update(
//             &mut cache,
//             &graph1.edges[0..1],
//             ExecutionMode::Lazy(&graph1.edges[1]),
//         );
//         assert_eq!(result, StreamAnalysisUpdate::WaitForOptimization);
//
//         let result = path.update(
//             &mut cache,
//             &graph1.edges[0..2],
//             ExecutionMode::Lazy(&graph1.edges[2]),
//         );
//         match result {
//             StreamAnalysisUpdate::ExecuteOptimization(ops) => assert_eq!(ops, 1),
//             _ => panic!("Should have found the cached operation"),
//         };
//
//         // New path instance on graph 2.
//         path.reset();
//
//         let result = path.update(&mut cache, &[], ExecutionMode::Lazy(&graph2.edges[0]));
//         assert_eq!(result, StreamAnalysisUpdate::WaitForOptimization);
//
//         let result = path.update(
//             &mut cache,
//             &graph2.edges[0..1],
//             ExecutionMode::Lazy(&graph2.edges[1]),
//         );
//         assert_eq!(result, StreamAnalysisUpdate::WaitForOptimization);
//
//         let result = path.update(
//             &mut cache,
//             &graph2.edges[0..2],
//             ExecutionMode::Lazy(&graph2.edges[2]),
//         );
//         match result {
//             StreamAnalysisUpdate::ExecuteOptimization(ops) => assert_eq!(ops, 2),
//             _ => panic!("Should have found the cached operation"),
//         };
//     }
//
//     #[derive(Default, Debug)]
//     struct TestStream {
//         nodes: Vec<TensorDescription>,
//         edges: Vec<TensorOpsDescription>,
//     }
//
//     impl TestStream {
//         /// Create a new test graph with `num_ops` operations registered.
//         pub fn new(num_ops: usize) -> Self {
//             let mut graph = Self::default();
//             for _ in 0..num_ops {
//                 graph.new_ops();
//             }
//
//             graph
//         }
//
//         /// The first follow should only be cache miss.
//         pub fn follow_misses(
//             &self,
//             cache: &mut StreamOptimizations<String>,
//             path: &mut StreamAnalysis<String>,
//         ) {
//             for i in 0..self.edges.len() {
//                 let result = path.update(
//                     cache,
//                     &self.edges[0..i],
//                     ExecutionMode::Lazy(&self.edges[i]),
//                 );
//                 assert_eq!(result, StreamAnalysisUpdate::ExploreOptimization);
//             }
//         }
//
//         /// Register a unary operation in the graph.
//         pub fn register_ops<F>(&mut self, func: F)
//         where
//             F: Fn(UnaryOpsDescription) -> TensorOpsDescription,
//         {
//             self.new_empty_node();
//             let desc = self.unary_description();
//             self.edges.push(func(desc));
//         }
//
//         /// Add a simple operation to the graph.
//         pub fn new_ops(&mut self) {
//             if self.nodes.is_empty() {
//                 // Root node.
//                 self.new_empty_node();
//             }
//
//             // Out node.
//             self.new_empty_node();
//
//             self.edges
//                 .push(TensorOpsDescription::FloatOps(FloatOpsDescription::Log(
//                     self.unary_description(),
//                 )));
//         }
//
//         fn new_empty_node(&mut self) {
//             self.nodes.push(TensorDescription {
//                 id: TensorId::new(self.nodes.len() as u64),
//                 shape: vec![32, 32, 1],
//                 status: TensorStatus::NotInit,
//             });
//         }
//
//         fn unary_description(&self) -> UnaryOpsDescription {
//             let size = self.nodes.len();
//
//             UnaryOpsDescription {
//                 input: self.nodes[size - 2].clone(),
//                 out: self.nodes[size - 1].clone(),
//             }
//         }
//     }
//
//     struct Optimization1;
//     struct Optimization2;
//
//     impl OptimizationFactory<String> for Optimization1 {
//         fn create(&self) -> String {
//             "Optimization1".to_string()
//         }
//     }
//
//     impl OptimizationFactory<String> for Optimization2 {
//         fn create(&self) -> String {
//             "Optimization2".to_string()
//         }
//     }
// }
