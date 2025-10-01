use burn_ir::OperationIr;

use super::ExecutionMode;
use crate::{
    NumOperations, OptimizationBuilder,
    search::{BlockOptimization, StreamOptimizer},
};

/// Explore and create new optimization.
pub struct Explorer<O> {
    optimizer: StreamOptimizer<O>,
    num_deferred: usize,
    num_explored: usize,
    is_still_optimizing: bool,
}

/// The result of an exploration done by the [explorer](Explorer).
pub enum ExplorationAction<O> {
    /// Found a new optimization.
    Completed(BlockOptimization<O>),
    /// We should continue exploring before arriving at a conclusion.
    Continue,
}

impl<O: NumOperations> Explorer<O> {
    /// Create a new explorer.
    pub(crate) fn new(optimizations: Vec<Box<dyn OptimizationBuilder<O>>>) -> Self {
        Self {
            optimizer: StreamOptimizer::new(optimizations),
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
        if let ExecutionMode::Lazy = mode
            && self.is_still_optimizing
        {
            return ExplorationAction::Continue;
        }

        let optimization = self.optimizer.optimize(operations);

        ExplorationAction::Completed(optimization)
    }

    /// Reset the state of the explorer to the provided list of operations.
    pub(crate) fn reset(&mut self, operations: &[OperationIr]) {
        self.optimizer.reset();
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

            self.optimizer.register(relative);
            self.num_explored += 1;

            self.is_still_optimizing = self.optimizer.still_optimizing();
        }

        self.num_deferred = 0;
    }
}
