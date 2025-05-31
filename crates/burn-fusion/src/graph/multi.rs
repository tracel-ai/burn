use burn_ir::OperationIr;

use crate::{NumOperations, OptimizationBuilder, stream::store::ExecutionStrategy};

use super::{Graph, Registration, compilation::Compilation, merging::merge_graphs};

pub struct MultiGraphs<O> {
    builders: Vec<Box<dyn OptimizationBuilder<O>>>,
    pub graphs: Vec<Graph<O>>,
    length: usize,
    merged_failed: bool,
}

impl<O: NumOperations> MultiGraphs<O> {
    pub fn new(builders: Vec<Box<dyn OptimizationBuilder<O>>>) -> Self {
        Self {
            builders,
            graphs: Vec::new(),
            length: 0,
            merged_failed: false,
        }
    }
    pub fn register(&mut self, operation: &OperationIr) {
        match self.merge_graphs(operation) {
            MergeGraphResult::Succeed | MergeGraphResult::NoNeed => {}
            MergeGraphResult::Fail => {
                self.merged_failed = true;
                return;
            }
        }

        let mut added_count = 0;

        for graph in self.graphs.iter_mut() {
            match graph.register(operation, false, self.length) {
                Registration::Accepted => {
                    added_count += 1;
                }
                Registration::NotPartOfTheGraph => {}
            }
        }
        if added_count == 0 {
            self.on_new_graph(operation);
        } else {
            assert_eq!(added_count, 1, "Can only add the operation to one graph.");
        }
        self.length += 1;
    }

    /// What we know here is that every graph is independent at that time and can be executed
    /// in any order.
    ///
    /// The contract is that the length of operations executed must include all operations.
    /// However, some executions might only be on a partial graph, which leaves some operations
    /// not included in any optimization.
    ///
    /// In this scenario, we have three options:
    ///   1. Try to merge those operations with other graphs, respecting order dependencies.
    ///   2. Execute those operations in eager mode without optimization.
    ///   3. Reduce the size of the optimization to only include operations that are optimized,
    ///      continuing the exploration to potentially fuse those operations in a following
    ///      pass.
    ///
    /// The best decision in order is:
    ///    1. If we can merge those operations with existing graphs, this is awesome.
    ///    3. We are optimistic and reduce the size of the current optimization to a minimum to
    ///       allow more exploration.
    ///    2. If we can't reduce the size of the exploration to at least one operation (no
    ///       optimization possible without holes), then we execute the minimum amount of
    ///       operations without optimization.
    pub fn compile(&self) -> (ExecutionStrategy<O>, usize) {
        Compilation::new(self.graphs.clone()).compile()
    }

    pub fn reset(&mut self) {
        self.builders.iter_mut().for_each(|b| b.reset());
        self.length = 0;
        self.graphs.clear();
        self.merged_failed = false;
    }

    pub fn still_optimizing(&self) -> bool {
        if self.graphs.is_empty() {
            return true;
        }
        if self.merged_failed {
            return false;
        }

        let mut num_stopped = 0;

        for graph in self.graphs.iter() {
            if !graph.still_optimizing() {
                num_stopped += 1
            }
        }

        num_stopped < self.graphs.len()
    }

    fn merge_graphs(&mut self, operation: &OperationIr) -> MergeGraphResult {
        let nodes = operation.nodes();
        let mut graph_merges = Vec::new();

        for (i, graph) in self.graphs.iter().enumerate() {
            if graph.should_include_nodes(&nodes) {
                graph_merges.push(i);
            }
        }

        if graph_merges.len() <= 1 {
            return MergeGraphResult::NoNeed;
        }

        let graphs_to_merge = self
            .graphs
            .iter()
            .enumerate()
            .filter_map(|(i, g)| match graph_merges.contains(&i) {
                true => Some(g),
                false => None,
            })
            .collect::<Vec<_>>();
        let merged = merge_graphs(&graphs_to_merge);

        if let Some(graph) = merged {
            let mut num_removed = 0;
            // If ordered it's fine
            for g in graph_merges {
                self.graphs.remove(g - num_removed);
                num_removed += 1;
            }
            self.graphs.push(graph);
            return MergeGraphResult::Succeed;
        }

        MergeGraphResult::Fail
    }

    fn on_new_graph(&mut self, operation: &OperationIr) {
        let mut graph = Graph::new(&self.builders);
        graph.register(operation, true, self.length);
        self.graphs.push(graph);
    }
}

enum MergeGraphResult {
    Succeed,
    NoNeed,
    Fail,
}
