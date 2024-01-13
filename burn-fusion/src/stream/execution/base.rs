use crate::{
    stream::{
        store::{OptimizationId, OptimizationKind, OptimizationStore},
        Stream,
    },
    FusionBackend, HandleContainer, Optimization,
};

#[derive(Clone, Copy, Debug)]
pub(crate) enum ExecutionMode {
    Lazy,
    Sync,
}

impl<B: FusionBackend> Stream<B> {
    /// Execute the stream.
    ///
    /// If an [optimization id](OptimizationId) is provided, use it to execute the stream partially, otherwise
    /// execute each [operation](crate::stream::Ops).
    pub(crate) fn execute(
        &mut self,
        id: OptimizationId,
        handles: &mut HandleContainer<B>,
        store: &mut OptimizationStore<B::Optimization>,
    ) {
        match &mut store.get_mut_unchecked(id).value {
            OptimizationKind::CustomOptimization(optimization) => {
                self.execute_optimization(handles, optimization)
            }
            OptimizationKind::ExecuteIndividualOps => self.execute_operations(handles),
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
        self.ops.drain(0..num_drained);
    }

    fn execute_operations(&mut self, handles: &mut HandleContainer<B>) {
        let num_drained = self.ops.len();

        for ops in self.ops.drain(0..num_drained) {
            ops.execute(handles);
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
