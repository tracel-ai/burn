use crate::{
    graph::{Graph, GraphExecution, Optimization, TensorOps},
    FusedBackend, FusionProperties, FusionStatus, HandleContainer, TensorId,
};
use burn_tensor::ops::FloatElem;
use std::sync::Arc;

pub struct FusionServer<B, G>
where
    B: FusedBackend,
    G: GraphExecution<B>,
{
    optimizations: Vec<Optimization<B>>,
    graph: Graph<B>,
    handles: HandleContainer<B>,
    execution: G,
    pub device: B::HandleDevice,
}

/// Trait name graph execution strategy.
impl<B, G> FusionServer<B, G>
where
    B: FusedBackend,
    G: GraphExecution<B>,
{
    pub fn new(device: B::HandleDevice) -> Self {
        let optimizations = B::operations()
            .into_iter()
            .map(|ops| Optimization::new(ops, FusionStatus::Open(FusionProperties::default())))
            .collect();

        Self {
            optimizations,
            graph: Graph::new(),
            handles: HandleContainer::new(device.clone()),
            execution: G::default(),
            device,
        }
    }

    pub fn register(&mut self, ops: TensorOps<B>) {
        let ops = Arc::new(ops);
        self.graph.add(ops.clone());

        self.optimizations
            .iter_mut()
            .for_each(|optimization| optimization.register(&ops));

        self.execution.maybe_execute(
            &mut self.graph,
            &mut self.handles,
            &mut self.optimizations,
            false,
        );
    }

    pub fn sync(&mut self) {
        self.execution.maybe_execute(
            &mut self.graph,
            &mut self.handles,
            &mut self.optimizations,
            true,
        );
    }

    pub fn create_empty_handle(&mut self) -> Arc<TensorId> {
        self.handles.create_emtpy()
    }

    pub fn create_float_handle(&mut self, values: Vec<FloatElem<B>>) -> Arc<TensorId> {
        self.handles.create_float(values)
    }

    pub fn read_float<const D: usize>(
        &mut self,
        tensor: crate::TensorDescription,
    ) -> burn_tensor::Reader<burn_tensor::Data<FloatElem<B>, D>> {
        // Make sure all registered operations are executed.
        // The underlying backend can still be async.
        self.sync();

        let tensor = self.handles.get_float_tensor(&tensor);
        B::into_data(tensor)
    }
}
