use std::sync::Arc;

use burn_ir::HandleContainer;

use crate::{FusionRuntime, NumOperations, Optimization, stream::Context};

use super::Operation;

/// Manage the execution of potentially multiple optimizations and operations out of order.
pub struct OrderedExecution<R: FusionRuntime> {
    operations: Vec<Box<dyn Operation<R>>>,
    num_executed: usize,
    ordering: Option<Arc<Vec<usize>>>,
}

impl<R: FusionRuntime> OrderedExecution<R> {
    /// Returns the operation that can be executed without impacting the state of the execution.
    ///
    /// This is useful to implement fallback for optimizations.
    #[allow(clippy::borrowed_box)]
    pub fn operation_within_optimization(&self, index: usize) -> &Box<dyn Operation<R>> {
        match &self.ordering {
            Some(val) => {
                let index = val[index];
                &self.operations[index]
            }
            None => panic!("No ordering provided"),
        }
    }

    pub(crate) fn new(operations: Vec<Box<dyn Operation<R>>>) -> Self {
        Self {
            operations,
            num_executed: 0,
            ordering: None,
        }
    }

    pub(crate) fn finish(mut self) -> (Vec<Box<dyn Operation<R>>>, usize) {
        self.operations.drain(0..self.num_executed);
        (self.operations, self.num_executed)
    }

    pub(crate) fn execute_optimization(
        &mut self,
        optimization: &mut R::Optimization,
        context: &mut Context<'_, R::FusionHandle>,
        ordering: Arc<Vec<usize>>,
    ) {
        self.ordering = Some(ordering);
        let num_drained = optimization.len();
        optimization.execute(context, self);
        self.num_executed += num_drained;
    }

    pub(crate) fn execute_operations(
        &mut self,
        handles: &mut HandleContainer<R::FusionHandle>,
        ordering: &[usize],
    ) {
        self.num_executed += ordering.len();

        println!("{ordering:?}");
        println!("Num Operations {:?}", self.operations.len());
        for id in ordering {
            let op = &self.operations[*id];
            op.execute(handles);
        }
    }
}
