use super::ExecutionMode;
use crate::{stream::OperationDescription, OptimizationBuilder, OptimizationStatus};

/// Explore and create new optimization.
pub struct Explorer<O> {
    builders: Vec<Box<dyn OptimizationBuilder<O>>>,
    num_deferred: usize,
    num_explored: usize,
}

/// The result of an exploration.
///
/// Either a new optimization is found, or we just continue to explore further.
pub enum ExplorerResult<'a, O> {
    Found(&'a dyn OptimizationBuilder<O>),
    NotFound { num_explored: usize },
    Continue,
}

impl<O> Explorer<O> {
    pub(crate) fn new(optimizations: Vec<Box<dyn OptimizationBuilder<O>>>) -> Self {
        Self {
            builders: optimizations,
            num_deferred: 0,
            num_explored: 0,
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
        stream: &[OperationDescription],
        mode: ExecutionMode,
    ) -> ExplorerResult<'a, O> {
        // When we are executing with the new ops mode, we need to register the last ops of the
        // stream even when there is no skipped operation.
        let offset = match mode {
            ExecutionMode::Lazy => 1,
            ExecutionMode::Sync => 0,
        };

        let mut is_still_optimizing = still_optimizing(&self.builders);

        for i in (0..self.num_deferred + offset).rev() {
            if !is_still_optimizing {
                break;
            }
            let index = stream.len() - 1 - i;
            let relative = &stream[index];

            for builder in self.builders.iter_mut() {
                builder.register(relative);
            }
            self.num_explored += 1;

            is_still_optimizing = still_optimizing(&self.builders);
        }
        self.num_deferred = 0;

        // Can only be lazy when not sync.
        if let ExecutionMode::Lazy = mode {
            if is_still_optimizing {
                return ExplorerResult::Continue;
            }
        }

        match find_best_optimization_index(&mut self.builders) {
            Some(index) => ExplorerResult::Found(self.builders[index].as_ref()),
            None => ExplorerResult::NotFound {
                num_explored: self.num_explored,
            },
        }
    }

    pub(crate) fn reset(&mut self, stream: &[OperationDescription]) {
        for ops in self.builders.iter_mut() {
            ops.reset();
        }
        self.num_explored = 0;
        self.num_deferred = stream.len();
    }
}

fn still_optimizing<O>(optimizations: &[Box<dyn OptimizationBuilder<O>>]) -> bool {
    let mut num_stopped = 0;

    for optimization in optimizations.iter() {
        if let OptimizationStatus::Closed = optimization.status() {
            num_stopped += 1
        }
    }

    num_stopped < optimizations.len()
}

fn find_best_optimization_index<O>(
    optimizations: &mut [Box<dyn OptimizationBuilder<O>>],
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

#[cfg(test)]
mod tests {}
