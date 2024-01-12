use super::ExecutionMode;
use crate::{stream::Stream, FusionBackend, OptimizationBuilder, OptimizationStatus};

/// Explore and create new optimization.
pub struct Explorer<B: FusionBackend> {
    builders: Vec<Box<dyn OptimizationBuilder<B>>>,
    num_deferred: usize,
}

/// The result of an exploration.
///
/// Either a new optimization is found, or we just continue to explore further.
pub enum Exploration<'a, B: FusionBackend> {
    OptimizationFound(Option<&'a dyn OptimizationBuilder<B>>),
    Continue,
}

impl<B: FusionBackend> Explorer<B> {
    pub(crate) fn new(optimizations: Vec<Box<dyn OptimizationBuilder<B>>>) -> Self {
        Self {
            builders: optimizations,
            num_deferred: 0,
        }
    }

    pub(crate) fn defer(&mut self) {
        self.num_deferred += 1;
    }

    pub(crate) fn up_to_date(&self) -> bool {
        self.num_deferred == 0
    }

    pub(crate) fn explore<'a>(
        &'a mut self,
        graph: &Stream<B>,
        mode: ExecutionMode,
    ) -> Exploration<'a, B> {
        // When we are executing with the new ops mode, we need to register the last ops of the
        // graph even when there is no skipped operation.
        let offset = match mode {
            ExecutionMode::Lazy => 1,
            ExecutionMode::Sync => 0,
        };

        for i in (0..self.num_deferred + offset).rev() {
            let index = graph.relative.len() - 1 - i;
            let relative = &graph.relative[index];

            for builder in self.builders.iter_mut() {
                builder.register(relative);
            }
        }
        self.num_deferred = 0;

        // Can only be lazy when not sync.
        if let ExecutionMode::Lazy = mode {
            if still_optimizing(&self.builders) {
                return Exploration::Continue;
            }
        }

        match find_best_optimization_index(&mut self.builders) {
            Some(index) => Exploration::OptimizationFound(Some(self.builders[index].as_ref())),
            None => Exploration::OptimizationFound(None),
        }
    }

    pub(crate) fn reset(&mut self, stream: &Stream<B>) {
        for ops in self.builders.iter_mut() {
            ops.reset();
        }
        self.num_deferred = stream.relative.len();
    }
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
