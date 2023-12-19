use super::{CacheResult, Condition, Graph, OptimizationCache, TensorOpsDescription};
use crate::{
    FusionBackend, HandleContainer, Optimization, OptimizationBuilder, OptimizationStatus,
};

/// Execute an optimization following a greedy algorithm.
pub(crate) struct GraphExecution<B: FusionBackend> {
    optimization_cache: OptimizationCache<Box<dyn Optimization<B>>>,
    optimizations: Vec<Box<dyn OptimizationBuilder<B>>>,
    num_skipped: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum ExecutionMode {
    // Signal that we execute the graph after a new ops is added to the graph.
    NewOps,
    // Signal that we execute the graph because of a sync without any new ops added to the graph.
    Sync,
}

impl<B: FusionBackend> GraphExecution<B> {
    /// Create a new graph execution with the given optimization builders.
    pub fn new(optimizations: Vec<Box<dyn OptimizationBuilder<B>>>) -> Self {
        Self {
            optimization_cache: OptimizationCache::new(),
            optimizations,
            num_skipped: 0,
        }
    }

    /// Execute the graph with the provided mode.
    pub fn execute(
        &mut self,
        graph: &mut Graph<B>,
        handles: &mut HandleContainer<B>,
        mode: ExecutionMode,
    ) {
        loop {
            if graph.is_empty() {
                break;
            }

            match self.cache(graph, mode) {
                CacheResult::Miss => {
                    match self.build(graph, mode) {
                        BuildAction::ExecuteOptimization(ops) => {
                            graph.execute_optimization(handles, ops);
                            self.reset(graph);
                        }
                        BuildAction::ExecuteOperations => {
                            graph.execute_operations(handles);
                            self.reset(graph);
                        }
                        BuildAction::ContinueBuilding => {
                            if let ExecutionMode::Sync = mode {
                                panic!("Can't continue building when sync is called.")
                            }
                        }
                    };

                    if self.num_skipped == 0 {
                        break;
                    }
                }
                CacheResult::OnPath => {
                    self.num_skipped += 1;

                    match mode {
                        ExecutionMode::NewOps => break,
                        ExecutionMode::Sync => panic!("Can't wait while sync"),
                    };
                }
                CacheResult::Found(ops) => {
                    graph.execute_optimization(handles, ops.as_mut());
                    self.reset(graph);
                }
            };

            if let ExecutionMode::NewOps = mode {
                break;
            }
        }
    }

    fn build(&mut self, graph: &Graph<B>, mode: ExecutionMode) -> BuildAction<'_, B> {
        // When we are executing with the new ops mode, we need to register the last ops of the
        // graph even when there is no skipped operation.
        let offset = match mode {
            ExecutionMode::NewOps => 1,
            ExecutionMode::Sync => 0,
        };

        for i in (0..self.num_skipped + offset).rev() {
            let index = graph.relative.len() - 1 - i;
            let relative = &graph.relative[index];

            for ops in self.optimizations.iter_mut() {
                ops.register(relative);
            }
        }
        self.num_skipped = 0;

        // Can only be lazy when not sync.
        if let ExecutionMode::NewOps = mode {
            if still_optimizing(&self.optimizations) {
                return BuildAction::ContinueBuilding;
            }
        }

        match find_best_optimization_index(&mut self.optimizations) {
            Some(index) => {
                let (relative, next_ops) = Self::split_relative_graph_owned(graph, mode);
                let optimization = &self.optimizations[index];
                let ops = self
                    .optimization_cache
                    .complete(optimization, relative, next_ops);
                BuildAction::ExecuteOptimization(ops.as_mut())
            }
            None => {
                // TODO: Cache this result too.
                BuildAction::ExecuteOperations
            }
        }
    }

    fn reset(&mut self, graph: &Graph<B>) {
        for ops in self.optimizations.iter_mut() {
            ops.reset();
        }
        self.num_skipped = graph.relative.len();

        self.optimization_cache.reset();

        // Reset the policy state.
        for i in 0..self.num_skipped {
            let _ = self.optimization_cache.follow(
                &graph.relative[0..i],
                Condition::NextOps(&graph.relative[i]),
            );
        }
    }

    fn cache<'a>(
        &'a mut self,
        graph: &Graph<B>,
        mode: ExecutionMode,
    ) -> CacheResult<'a, Box<dyn Optimization<B>>> {
        let (graph, next_ops) = Self::split_relative_graph_ref(graph, mode);
        let end_condition = next_ops.map(Condition::NextOps).unwrap_or(Condition::Sync);
        let action = self.optimization_cache.follow(graph, end_condition);

        match mode {
            ExecutionMode::NewOps => action,
            ExecutionMode::Sync => match action {
                CacheResult::Miss => CacheResult::Miss,
                CacheResult::OnPath => CacheResult::Miss,
                CacheResult::Found(ops) => CacheResult::Found(ops),
            },
        }
    }

    fn split_relative_graph_owned(
        graph: &Graph<B>,
        mode: ExecutionMode,
    ) -> (Vec<TensorOpsDescription>, Option<TensorOpsDescription>) {
        match mode {
            ExecutionMode::NewOps => {
                let graph = graph.split_relative_graph();
                (graph.0.to_vec(), graph.1.cloned())
            }
            ExecutionMode::Sync => (graph.relative.clone(), None),
        }
    }

    fn split_relative_graph_ref(
        graph: &Graph<B>,
        mode: ExecutionMode,
    ) -> (&[TensorOpsDescription], Option<&TensorOpsDescription>) {
        match mode {
            ExecutionMode::NewOps => graph.split_relative_graph(),
            ExecutionMode::Sync => (graph.relative.as_slice(), None),
        }
    }
}

enum BuildAction<'a, B: FusionBackend> {
    ExecuteOptimization(&'a mut dyn Optimization<B>),
    ExecuteOperations,
    ContinueBuilding,
}

fn still_optimizing<B: FusionBackend>(optimizations: &[Box<dyn OptimizationBuilder<B>>]) -> bool {
    let mut num_stopped = 0;

    for optimization in optimizations.iter() {
        if let OptimizationStatus::Closed = optimization.status() {
            num_stopped += 1
        }
    }

    num_stopped < optimizations.len()
}

fn find_best_optimization_index<B: FusionBackend>(
    optimizations: &mut [Box<dyn OptimizationBuilder<B>>],
) -> Option<usize> {
    let mut best_index = None;
    let mut best_score = 0;

    for (i, optimization) in optimizations.iter().enumerate() {
        let properties = optimization.properties();

        if properties.ready && properties.score >= best_score {
            best_index = Some(i);
            best_score = properties.score;
        }
    }

    best_index
}
