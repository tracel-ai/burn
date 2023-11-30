use super::{CacheResult, EndCondition, Graph, Optimization, OptimizationCache};
use crate::{FusionBackend, FusionOps, FusionStatus, HandleContainer};

/// Execute an optimization following a greedy algorithm.
pub(crate) struct GraphExecution<B: FusionBackend> {
    optimization_path: OptimizationCache<Box<dyn FusionOps<B>>>,
    optimizations: Vec<Optimization<B>>,
    num_skipped: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum ExecutionMode {
    // Signal that we execute the graph after a new ops was added to the graph.
    NewOps,
    // Signal that we execute the graph because of a sync without any new ops added to the graph.
    Sync,
}

impl<B: FusionBackend> GraphExecution<B> {
    pub fn new(optimizations: Vec<Optimization<B>>) -> Self {
        Self {
            optimization_path: OptimizationCache::new(),
            optimizations,
            num_skipped: 0,
        }
    }

    pub fn maybe_execute(
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
                    let build_action = self.build(graph, mode);

                    match build_action {
                        BuildAction::ExecuteOptimization(ops) => {
                            graph.execute_ops(handles, ops);
                            self.reset(graph);
                        }
                        BuildAction::ExecuteOperations => {
                            graph.execute(handles);
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
                    graph.execute_ops(handles, ops);
                    self.reset(graph);
                }
            };

            if let ExecutionMode::NewOps = mode {
                break;
            }
        }
    }

    fn build(&mut self, graph: &mut Graph<B>, mode: ExecutionMode) -> BuildAction<'_, B> {
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

        if let ExecutionMode::NewOps = mode {
            if still_optimizing(&self.optimizations) {
                return BuildAction::ContinueBuilding;
            }
        }

        match find_best_optimization_index(&self.optimizations) {
            Some(index) => {
                let (relative, next_ops) = match mode {
                    ExecutionMode::NewOps => {
                        let graph = graph.lazy_format_relative();
                        (graph.0.to_vec(), graph.1.cloned())
                    }
                    ExecutionMode::Sync => (graph.relative.clone(), None),
                };

                let optimization = &self.optimizations[index];
                let ops = self
                    .optimization_path
                    .complete(&optimization.ops, relative, next_ops);
                BuildAction::ExecuteOptimization(ops)
            }
            None => BuildAction::ExecuteOperations,
        }
    }

    fn reset(&mut self, graph: &mut Graph<B>) {
        for ops in self.optimizations.iter_mut() {
            ops.reset();
        }
        self.num_skipped = graph.relative.len();

        self.optimization_path.reset();

        // Reset the policy state.
        for i in 0..self.num_skipped {
            let _ = self.optimization_path.follow(
                &graph.relative[0..i],
                EndCondition::NextOps(&graph.relative[i]),
            );
        }
    }

    fn cache<'a>(
        &'a mut self,
        graph: &mut Graph<B>,
        mode: ExecutionMode,
    ) -> CacheResult<'a, Box<dyn FusionOps<B>>> {
        let (graph, next_ops) = match mode {
            ExecutionMode::NewOps => graph.lazy_format_relative(),
            ExecutionMode::Sync => (graph.relative.as_slice(), None),
        };
        let end_condition = next_ops
            .map(|ops| EndCondition::NextOps(ops))
            .unwrap_or(EndCondition::Forced);

        let action = self.optimization_path.follow(graph, end_condition);

        match mode {
            ExecutionMode::NewOps => action,
            ExecutionMode::Sync => match action {
                CacheResult::Miss => CacheResult::Miss,
                CacheResult::OnPath => CacheResult::Miss,
                CacheResult::Found(ops) => CacheResult::Found(ops),
            },
        }
    }
}

enum BuildAction<'a, B: FusionBackend> {
    ExecuteOptimization(&'a Box<dyn FusionOps<B>>),
    ExecuteOperations,
    ContinueBuilding,
}

fn still_optimizing<B: FusionBackend>(optimizations: &[Optimization<B>]) -> bool {
    let mut num_stopped = 0;

    for optimization in optimizations.iter() {
        if let FusionStatus::Closed(_) = optimization.status {
            num_stopped += 1
        }
    }

    num_stopped < optimizations.len()
}

fn find_best_optimization_index<B: FusionBackend>(
    optimizations: &[Optimization<B>],
) -> Option<usize> {
    let mut best_index = None;
    let mut best_score = 0;

    for (i, optimization) in optimizations.iter().enumerate() {
        let properties = match optimization.status {
            FusionStatus::Closed(properties) => properties,
            FusionStatus::Open(properties) => properties,
        };

        if properties.ready && properties.score >= best_score {
            best_index = Some(i);
            best_score = properties.score;
        }
    }

    best_index
}
