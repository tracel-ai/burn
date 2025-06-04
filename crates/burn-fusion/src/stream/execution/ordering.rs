use burn_ir::HandleContainer;

use crate::{FusionRuntime, NumOperations, Optimization, stream::Context};

use super::Operation;

/// Manage the execution of potentially multiple optimizations and operations out of order.
pub struct OrderedExecution<R: FusionRuntime> {
    operations: Vec<Box<dyn Operation<R>>>,
    ordering: Vec<usize>,
    cursor: usize,
}

impl<R: FusionRuntime> OrderedExecution<R> {
    /// Returns the operation that can be executed without impacting the state of the execution.
    ///
    /// This is usefull to implement fallback for optimizations.
    pub fn operation_within_optimization(&self, index: usize) -> &Box<dyn Operation<R>> {
        let position = self.cursor + index;
        let index = self.ordering[position];
        &self.operations[index]
    }

    pub(crate) fn new(operations: Vec<Box<dyn Operation<R>>>, ordering: Vec<usize>) -> Self {
        Self {
            operations,
            ordering,
            cursor: 0,
        }
    }

    pub(crate) fn finish(mut self) -> (Vec<Box<dyn Operation<R>>>, usize) {
        println!("Executed {:?}", &self.ordering[0..self.cursor]);
        self.operations.drain(0..self.cursor);
        (self.operations, self.cursor)
    }

    pub(crate) fn execute_optimization(
        &mut self,
        optimization: &mut R::Optimization,
        context: &mut Context<'_, R::FusionHandle>,
    ) {
        println!("Execute optimization");
        let num_drained = optimization.len();
        optimization.execute(context, self);
        self.cursor += num_drained;
        println!("Executed optimization");
    }

    pub(crate) fn execute_optimization_with_fallbacks(
        &mut self,
        optimization: &mut R::Optimization,
        context: &mut Context<'_, R::FusionHandle>,
        fallbacks: &Vec<usize>,
    ) {
        println!("Execute optimization with fallbacks");
        println!("Ordering {:?}", self.ordering);
        println!("Cursor {:?}", self.cursor);
        println!("Fallbacks {:?}", fallbacks);
        assert_eq!(fallbacks.is_empty(), false, "No fallbacks");
        let num_drained = optimization.len() + fallbacks.len();

        optimization.execute(context, self);

        for f in fallbacks {
            let op = &self.operations[*f];
            op.execute(context.handles);
        }

        self.cursor += num_drained;
        println!("Executed optimization with fallbacks");
    }

    pub(crate) fn execute_operations(
        &mut self,
        handles: &mut HandleContainer<R::FusionHandle>,
        size: usize,
    ) {
        println!("Execute operations");
        for _ in 0..size {
            // let index = self.cursor;
            let index = self.ordering[self.cursor];
            println!("Execute op {index}");
            let op = &self.operations[index];
            op.execute(handles);
            self.cursor += 1;
        }
        println!("Executed operations");
    }
}
