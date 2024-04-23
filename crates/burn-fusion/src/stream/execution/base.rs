use burn_tensor::repr::HandleContainer;

use crate::{
    stream::{
        store::{ExecutionPlanId, ExecutionPlanStore, ExecutionStrategy},
        OperationQueue, RelativeOps,
    },
    FusionBackend, Optimization,
};

/// The mode in which the execution is done.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ExecutionMode {
    Lazy,
    Sync,
}

/// General trait to abstract how a single operation is executed.
pub trait Operation<B: FusionBackend>: Send + Sync {
    /// Execute the operation.
    fn execute(self: Box<Self>, handles: &mut HandleContainer<B>);
}

impl<B: FusionBackend> OperationQueue<B> {
    /// Execute the queue partially following the execution strategy from the plan.
    pub(crate) fn execute(
        &mut self,
        id: ExecutionPlanId,
        handles: &mut HandleContainer<B>,
        store: &mut ExecutionPlanStore<B::Optimization>,
    ) {
        match &mut store.get_mut_unchecked(id).strategy {
            ExecutionStrategy::Optimization(optimization) => {
                self.execute_optimization(handles, optimization)
            }
            ExecutionStrategy::Operations => self.execute_operations(handles),
        };
    }

    fn execute_optimization(
        &mut self,
        handles: &mut HandleContainer<B>,
        optimization: &mut B::Optimization,
    ) {
        let num_drained = optimization.len();

        let mut context = self.converter.context(handles);
        optimization.execute(&mut context);

        self.drain_stream(num_drained, handles);
        self.operations.drain(0..num_drained);
    }

    fn execute_operations(&mut self, handles: &mut HandleContainer<B>) {
        let num_drained = self.operations.len();

        for operation in self.operations.drain(0..num_drained) {
            operation.execute(handles);
        }

        self.drain_stream(num_drained, handles);
    }

    fn drain_stream(&mut self, num_drained: usize, handles: &mut HandleContainer<B>) {
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
