use super::ExecutionMode;
use crate::{stream::OperationDescription, OptimizationBuilder, OptimizationStatus};

/// Explore and create new optimization.
pub struct Explorer<O> {
    builders: Vec<Box<dyn OptimizationBuilder<O>>>,
    num_deferred: usize,
    num_explored: usize,
}

/// The result of an exploration done by the [explorer](Explorer).
pub enum Exploration<'a, O> {
    /// Found a new optimization.
    Found(&'a dyn OptimizationBuilder<O>),
    /// No optimization is found.
    NotFound { num_explored: usize },
    /// We should continue exploring before arriving at a conclusion.
    Continue,
}

impl<O> Explorer<O> {
    /// Create a new explorer.
    pub(crate) fn new(optimizations: Vec<Box<dyn OptimizationBuilder<O>>>) -> Self {
        Self {
            builders: optimizations,
            num_deferred: 0,
            num_explored: 0,
        }
    }

    /// Defer the exploration.
    pub(crate) fn defer(&mut self) {
        self.num_deferred += 1;
    }

    /// If the explorer is up to date.
    pub(crate) fn is_up_to_date(&self) -> bool {
        self.num_deferred == 0
    }

    /// Explore the provided operations.
    pub(crate) fn explore<'a>(
        &'a mut self,
        operations: &[OperationDescription],
        mode: ExecutionMode,
    ) -> Exploration<'a, O> {
        // When we are executing with the new operation mode, we need to register the last ops of the
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
            let index = operations.len() - 1 - i;
            let relative = &operations[index];

            for builder in self.builders.iter_mut() {
                builder.register(relative);
            }
            self.num_explored += 1;

            is_still_optimizing = still_optimizing(&self.builders);
        }
        self.num_deferred = 0;

        // Can only continue exploration when not sync.
        if let ExecutionMode::Lazy = mode {
            if is_still_optimizing {
                return Exploration::Continue;
            }
        }

        match find_best_optimization_index(&mut self.builders) {
            Some(index) => Exploration::Found(self.builders[index].as_ref()),
            None => Exploration::NotFound {
                num_explored: self.num_explored,
            },
        }
    }

    /// Reset the state of the explorer to the provided list of operations.
    pub(crate) fn reset(&mut self, operations: &[OperationDescription]) {
        for operation in self.builders.iter_mut() {
            operation.reset();
        }
        self.num_explored = 0;
        self.num_deferred = operations.len();
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
