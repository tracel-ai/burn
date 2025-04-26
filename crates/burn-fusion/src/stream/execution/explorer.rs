use burn_ir::OperationIr;

use super::ExecutionMode;
use crate::{OptimizationBuilder, OptimizationStatus};

/// Explore and create new optimization.
pub struct Explorer<O> {
    /// The optimization builders, one for each type of optimization that
    /// we want to explore.
    builders: Vec<Box<dyn OptimizationBuilder<O>>>,
    num_deferred: usize,
    num_explored: usize,
    is_still_optimizing: bool,
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
            is_still_optimizing: true,
        }
    }

    /// Indicate that a new operation is added.
    pub(crate) fn on_new_operation(&mut self) {
        self.num_deferred += 1;
    }

    /// If the explorer is up to date.
    pub(crate) fn is_up_to_date(&self) -> bool {
        self.num_deferred == 0
    }

    /// Explore the provided operations.
    pub(crate) fn explore<'a>(
        &'a mut self,
        operations: &[OperationIr],
        mode: ExecutionMode,
    ) -> Exploration<'a, O> {
        self.update(operations);

        // Can only continue exploration when not sync.
        if let ExecutionMode::Lazy = mode {
            if self.is_still_optimizing {
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
    pub(crate) fn reset(&mut self, operations: &[OperationIr]) {
        for operation in self.builders.iter_mut() {
            operation.reset();
        }
        self.num_explored = 0;
        self.num_deferred = operations.len();
        self.is_still_optimizing = true;
    }

    /// Register any operations that we had deferred
    fn update(&mut self, operations: &[OperationIr]) {
        for i in (0..self.num_deferred).rev() {
            if !self.is_still_optimizing {
                break;
            }
            let index = operations.len() - 1 - i;
            let relative = &operations[index];

            for builder in self.builders.iter_mut() {
                builder.register(relative);
            }
            self.num_explored += 1;

            self.is_still_optimizing = still_optimizing(&self.builders);
        }

        self.num_deferred = 0;
    }
}

/// Returns false if all optimization builders are closed, which means that no more
/// optimizations are possible.
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
