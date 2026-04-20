use burn_ir::{HandleContainer, TensorStatus};

use crate::{
    FusionRuntime, UnfusedOp,
    search::BlockOptimization,
    stream::{
        Context, ContextGuard, OperationConverter, OrderedExecution, RelativeOps,
        store::{ExecutionPlanId, ExecutionPlanStore, ExecutionStrategy},
    },
};

use super::OperationQueue;

impl<R: FusionRuntime> OperationQueue<R> {
    /// Execute the queue partially following the execution strategy from the plan.
    pub(crate) fn execute(
        &mut self,
        id: ExecutionPlanId,
        handles: &mut HandleContainer<R::FusionHandle>,
        store: &mut ExecutionPlanStore<R::Optimization>,
    ) {
        let plan = store.get_mut_unchecked(id);
        self.execute_block_optimization(&mut plan.optimization, handles);
    }

    fn execute_block_optimization(
        &mut self,
        step: &mut BlockOptimization<R::Optimization>,
        handles: &mut HandleContainer<R::FusionHandle>,
    ) {
        let mut operations = Vec::new();
        core::mem::swap(&mut operations, &mut self.operations);

        let (operations, num_drained) =
            run_strategy(step, &mut self.converter, handles, operations);

        self.operations = operations;
        self.drain_queue(num_drained, handles);
    }

    /// Bookkeeping after executing `num_drained` operations from the queue.
    fn drain_queue(&mut self, num_drained: usize, handles: &mut HandleContainer<R::FusionHandle>) {
        self.global[0..num_drained]
            .iter()
            .flat_map(|desc| desc.nodes())
            .for_each(|tensor| {
                if tensor.status == TensorStatus::ReadWrite {
                    self.variables.remove(&tensor.id);
                };
                handles.free(tensor)
            });

        self.global.drain(0..num_drained);

        self.reset_relative();
    }

    fn reset_relative(&mut self) {
        self.relative.clear();
        self.converter.clear();

        for node in self.global.iter() {
            let relative = node.to_relative(&mut self.converter);
            self.relative.push(relative);
        }
    }
}

/// Drive one block's execution strategy.
///
/// Wraps the converter's per-block fields and the handle container into a single owned
/// [`Context`] via [`ContextGuard`] for the duration of this call, then threads `&mut Context`
/// through the recursive strategy walk. Operations-only strategies just grab
/// `&mut ctx.handles`; optimization strategies hand `&mut ctx` to the fused op.
fn run_strategy<R: FusionRuntime>(
    optimization: &mut BlockOptimization<R::Optimization>,
    converter: &mut OperationConverter,
    handles: &mut HandleContainer<R::FusionHandle>,
    operations: Vec<UnfusedOp<R>>,
) -> (Vec<UnfusedOp<R>>, usize) {
    let mut execution = OrderedExecution::new(operations);
    {
        let mut guard = ContextGuard::new(converter, handles);
        execute_strategy::<R>(&mut optimization.strategy, &mut guard, &mut execution);
    }
    execution.finish()
}

fn execute_strategy<R: FusionRuntime>(
    strategy: &mut ExecutionStrategy<R::Optimization>,
    context: &mut Context<R::FusionHandle>,
    execution: &mut OrderedExecution<R>,
) {
    match strategy {
        ExecutionStrategy::Optimization { ordering, opt } => {
            execution.execute_optimization(opt, context, ordering.clone());
        }
        ExecutionStrategy::Operations { ordering } => {
            execution.execute_operations(&mut context.handles, ordering);
        }
        ExecutionStrategy::Composed(items) => {
            for item in items.iter_mut() {
                execute_strategy::<R>(item, context, execution);
            }
        }
    }
}
