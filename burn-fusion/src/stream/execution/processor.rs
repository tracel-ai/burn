use super::ExecutionMode;
use crate::stream::execution::{ExecutionAction, Policy};
use crate::stream::store::{OptimizationItem, OptimizationStore};
use crate::stream::{Stream, TensorOpsDescription};
use crate::{FusionBackend, HandleContainer, OptimizationBuilder, OptimizationStatus};

/// Execute an optimization following a greedy algorithm.
pub(crate) struct Processor<B: FusionBackend> {
    policy: Policy<B::Optimization>,
    builders: Vec<Box<dyn OptimizationBuilder<B>>>,
    num_skipped: usize,
}

impl<B: FusionBackend> Processor<B> {
    /// Create a new graph execution with the given optimization builders.
    pub fn new(optimizations: Vec<Box<dyn OptimizationBuilder<B>>>) -> Self {
        Self {
            policy: Policy::new(),
            builders: optimizations,
            num_skipped: 0,
        }
    }

    /// Execute the graph with the provided mode.
    pub fn process(
        &mut self,
        stream: &mut Stream<B>,
        optimizations: &mut OptimizationStore<B::Optimization>,
        handles: &mut HandleContainer<B>,
        mode: ExecutionMode,
    ) {
        loop {
            if stream.is_empty() {
                break;
            }

            match self.analysis_action(optimizations, stream, mode) {
                ExecutionAction::ExploreOptimization => {
                    match self.build(optimizations, stream, mode) {
                        BuildAction::ExecuteOptimization(ops) => {
                            stream.execute_optimization(handles, ops);
                            self.reset(optimizations, stream);
                        }
                        BuildAction::ExecuteOperations => {
                            stream.execute_operations(handles);
                            self.reset(optimizations, stream);
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
                ExecutionAction::WaitForOptimization => {
                    self.num_skipped += 1;

                    match mode {
                        ExecutionMode::Lazy => break,
                        ExecutionMode::Sync => panic!("Can't wait while sync"),
                    };
                }
                ExecutionAction::ExecuteOptimization(ops) => {
                    stream.execute_optimization(
                        handles,
                        &mut optimizations.get_mut_unchecked(ops).value,
                    );
                    self.reset(optimizations, stream);
                }
            };

            if let ExecutionMode::Lazy = mode {
                break;
            }
        }
    }

    fn build<'a>(
        &'a mut self,
        optimizations: &'a mut OptimizationStore<B::Optimization>,
        graph: &Stream<B>,
        mode: ExecutionMode,
    ) -> BuildAction<'_, B> {
        // When we are executing with the new ops mode, we need to register the last ops of the
        // graph even when there is no skipped operation.
        let offset = match mode {
            ExecutionMode::Lazy => 1,
            ExecutionMode::Sync => 0,
        };

        for i in (0..self.num_skipped + offset).rev() {
            let index = graph.relative.len() - 1 - i;
            let relative = &graph.relative[index];

            for builder in self.builders.iter_mut() {
                builder.register(relative);
            }
        }
        self.num_skipped = 0;

        // Can only be lazy when not sync.
        if let ExecutionMode::Lazy = mode {
            if still_optimizing(&self.builders) {
                return BuildAction::ContinueBuilding;
            }
        }

        match find_best_optimization_index(&mut self.builders) {
            Some(index) => {
                let (relative, next_ops) = Self::split_relative_graph_owned(graph, mode);
                let builder = &self.builders[index];

                let id = if let ExecutionAction::ExecuteOptimization(id) =
                    self.policy
                        .action(optimizations, &relative, ExecutionMode::Sync)
                {
                    if let Some(next_ops) = next_ops {
                        optimizations.add_end_condition(id, next_ops);
                    }
                    id
                } else {
                    optimizations.add(OptimizationItem {
                        stream: relative,
                        end_conditions: next_ops.map(|op| vec![op]).unwrap_or_default(),
                        value: builder.build(),
                    })
                };

                BuildAction::ExecuteOptimization(&mut optimizations.get_mut_unchecked(id).value)
            }
            None => {
                // TODO: Cache this result too.
                BuildAction::ExecuteOperations
            }
        }
    }

    fn reset(&mut self, cache: &mut OptimizationStore<B::Optimization>, graph: &Stream<B>) {
        for ops in self.builders.iter_mut() {
            ops.reset();
        }
        self.num_skipped = graph.relative.len();

        self.policy.reset();

        // Reset the policy state.
        for i in 0..self.num_skipped {
            let _ = self
                .policy
                .update(cache, &graph.relative[0..i], &graph.relative[i]);
        }
    }

    fn analysis_action<'a>(
        &'a mut self,
        cache: &'a mut OptimizationStore<B::Optimization>,
        stream: &Stream<B>,
        mode: ExecutionMode,
    ) -> ExecutionAction {
        let (stream, next_ops) = Self::split_relative_graph_ref(stream, mode);

        if let Some(next_ops) = next_ops {
            self.policy.update(cache, stream, next_ops)
        }

        self.policy.action(&cache, stream, mode)
    }

    fn split_relative_graph_owned(
        graph: &Stream<B>,
        mode: ExecutionMode,
    ) -> (Vec<TensorOpsDescription>, Option<TensorOpsDescription>) {
        match mode {
            ExecutionMode::Lazy => {
                let graph = graph.split_relative_graph();
                (graph.0.to_vec(), graph.1.cloned())
            }
            ExecutionMode::Sync => (graph.relative.clone(), None),
        }
    }

    fn split_relative_graph_ref(
        graph: &Stream<B>,
        mode: ExecutionMode,
    ) -> (&[TensorOpsDescription], Option<&TensorOpsDescription>) {
        match mode {
            ExecutionMode::Lazy => graph.split_relative_graph(),
            ExecutionMode::Sync => (graph.relative.as_slice(), None),
        }
    }
}

enum BuildAction<'a, B: FusionBackend> {
    ExecuteOptimization(&'a mut B::Optimization),
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
