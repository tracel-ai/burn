use crate::{
    stream::{
        store::{ExecutionPlanId, ExecutionPlanStore, ExecutionStrategy},
        OperationQueue,
    },
    FusionBackend, HandleContainer, Optimization,
};

/// The mode in which the execution is done.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ExecutionMode {
    Lazy,
    Sync,
}

impl<B: FusionBackend> OperationQueue<B> {
    /// Execute the queue partially following the execution strategy from the plan.
    pub(crate) fn execute(
        &mut self,
        id: ExecutionPlanId,
        handles: &mut HandleContainer<B>,
        store: &mut ExecutionPlanStore<B::Optimization>,
    ) {
        println!("Execute.");
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
        println!("Execute optimization with {} operations", num_drained);

        let mut context = self.converter.context(handles);
        optimization.execute(&mut context);

        self.drain_stream(num_drained, handles);
        self.operations.drain(0..num_drained);
    }

    fn execute_operations(&mut self, handles: &mut HandleContainer<B>) {
        let num_drained = self.operations.len();
        println!("Execute {} operations", num_drained);
        if num_drained == 0 {
            panic!("What the hell")
        }


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

        handles.free_orphans(
            &self
                .global
                .iter()
                .flat_map(|desc| desc.nodes())
                .map(|tensor| &tensor.id)
                .collect::<Vec<_>>(),
        );
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
