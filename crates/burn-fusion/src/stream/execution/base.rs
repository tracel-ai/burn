use burn_ir::HandleContainer;

use crate::{
    FusionRuntime, Optimization,
    stream::{
        OperationQueue, RelativeOps,
        store::{ExecutionPlanId, ExecutionPlanStore, ExecutionStrategy},
    },
};

/// The mode in which the execution is done.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ExecutionMode {
    Lazy,
    Sync,
}

/// General trait to abstract how a single operation is executed.
pub trait Operation<R: FusionRuntime>: Send + Sync {
    /// Execute the operation.
    fn execute(&self, handles: &mut HandleContainer<R::FusionHandle>);
}

impl<R: FusionRuntime> OperationQueue<R> {
    /// Execute the queue partially following the execution strategy from the plan.
    pub(crate) fn execute(
        &mut self,
        id: ExecutionPlanId,
        handles: &mut HandleContainer<R::FusionHandle>,
        store: &mut ExecutionPlanStore<R::Optimization>,
    ) {
        let plan = store.get_mut_unchecked(id);
        self.execute_strategy(&mut plan.strategy, handles);
    }

    fn execute_strategy(
        &mut self,
        strategy: &mut ExecutionStrategy<R::Optimization>,
        handles: &mut HandleContainer<R::FusionHandle>,
    ) {
        match strategy {
            ExecutionStrategy::Optimization(optimization) => {
                self.execute_optimization(handles, optimization)
            }
            ExecutionStrategy::Operations(size) => self.execute_operations(handles, *size),
            ExecutionStrategy::Composed(strategies) => {
                for s in strategies {
                    self.execute_strategy(s, handles);
                }
            }
        };
    }

    /// Execute the optimization (fused operations) and remove all the corresponding
    /// operations from the queue.
    fn execute_optimization(
        &mut self,
        handles: &mut HandleContainer<R::FusionHandle>,
        optimization: &mut R::Optimization,
    ) {
        let num_drained = optimization.len();

        let mut context = self.converter.context(handles);
        optimization.execute(&mut context, &self.operations);

        self.drain_queue(num_drained, handles);
        self.operations.drain(0..num_drained);
    }

    /// Execute all the operations in the [`OperationQueue`] sequentially
    /// without applying any optimization.
    fn execute_operations(&mut self, handles: &mut HandleContainer<R::FusionHandle>, size: usize) {
        for operation in self.operations.drain(0..size) {
            operation.execute(handles);
        }

        self.drain_queue(size, handles);
    }

    /// Bookkeeping after executing `num_drained` operations from the queue.
    fn drain_queue(&mut self, num_drained: usize, handles: &mut HandleContainer<R::FusionHandle>) {
        self.global[0..num_drained]
            .iter()
            .flat_map(|desc| desc.nodes())
            .for_each(|tensor| handles.free(tensor));

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
