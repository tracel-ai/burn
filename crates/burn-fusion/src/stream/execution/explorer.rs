use burn_ir::OperationIr;
use burn_std::config::{fusion::FusionLogLevel, log_fusion};

use super::{ExecutionMode, op_kind};
use crate::{
    NumOperations, OperationFuser,
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
    pub(crate) fn new(optimizations: Vec<Box<dyn OperationFuser<O>>>) -> Self {
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
        let total_ops = operations.len();
        let deferred = self.num_deferred;
        let mode_dbg = match mode {
            ExecutionMode::Lazy => "lazy",
            ExecutionMode::Sync => "sync",
        };
        log_fusion(FusionLogLevel::Full, move || {
            format!("[explorer] explore ({mode_dbg}): {deferred} deferred of {total_ops} queued")
        });

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
        let start_explored = self.num_explored;
        for i in (0..self.num_deferred).rev() {
            if !self.is_still_optimizing {
                let remaining = i + 1;
                let seen = self.num_explored - start_explored;
                log_fusion(FusionLogLevel::Full, move || {
                    format!(
                        "[explorer] stopped optimizing after {seen} new op(s); {remaining} deferred op(s) left unprocessed"
                    )
                });
                break;
            }
            let index = operations.len() - 1 - i;
            let relative = &operations[index];

            self.optimizer.register(relative);
            self.num_explored += 1;

            let before = self.is_still_optimizing;
            self.is_still_optimizing = self.optimizer.still_optimizing();
            if before && !self.is_still_optimizing {
                let explored = self.num_explored;
                log_fusion(FusionLogLevel::Full, || {
                    format!(
                        "[explorer] still_optimizing → false after op {} (explored {explored} ops)",
                        op_kind(relative)
                    )
                });
            }
        }

        self.num_deferred = 0;
    }
}
