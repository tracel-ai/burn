use burn_ir::OperationIr;

use super::ExecutionMode;
use crate::{
    NumOperations, OptimizationBuilder, search::OptimizationSearch,
    stream::store::ExecutionStrategy,
};

/// Explore and create new optimization.
pub struct Explorer<O> {
    /// The optimization builders, one for each type of optimization that
    /// we want to explore.
    // builders: Vec<Box<dyn OptimizationBuilder<O>>>,
    search: OptimizationSearch<O>,
    num_deferred: usize,
    num_explored: usize,
    is_still_optimizing: bool,
}

/// The result of an exploration done by the [explorer](Explorer).
pub enum ExplorationAction<O> {
    /// Found a new optimization.
    Completed(Exploration<O>),
    /// We should continue exploring before arriving at a conclusion.
    Continue,
}

pub struct Exploration<O> {
    pub strategy: ExecutionStrategy<O>,
    pub num_optimized: usize,
}

impl<O: NumOperations> Explorer<O> {
    /// Create a new explorer.
    pub(crate) fn new(optimizations: Vec<Box<dyn OptimizationBuilder<O>>>) -> Self {
        Self {
            // builders: optimizations,
            search: OptimizationSearch::new(optimizations),
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
    pub(crate) fn explore(
        &mut self,
        operations: &[OperationIr],
        mode: ExecutionMode,
    ) -> ExplorationAction<O> {
        self.update(operations);

        // Can only continue exploration when not sync.
        if let ExecutionMode::Lazy = mode {
            if self.is_still_optimizing {
                return ExplorationAction::Continue;
            }
        }

        let found = self.search.execute();

        ExplorationAction::Completed(Exploration {
            strategy: found.strategy,
            num_optimized: found.num_operations,
        })
    }

    /// Reset the state of the explorer to the provided list of operations.
    pub(crate) fn reset(&mut self, operations: &[OperationIr]) {
        self.search.reset();
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

            self.search.register(relative);
            self.num_explored += 1;

            self.is_still_optimizing = self.search.still_optimizing();
        }

        self.num_deferred = 0;
    }
}
