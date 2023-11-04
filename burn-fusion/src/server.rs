use crate::{
    graph::{
        FusedBackend, FusionProperties, FusionStatus, Graph, GraphExecution, Optimization,
        TensorOps,
    },
    HandleContainer, TensorId,
};
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
            handles: HandleContainer::new(device),
            execution: G::default(),
        }
    }

    pub fn register(&mut self, ops: TensorOps<B::FloatElem, B::IntElem>) {
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

    pub fn create(&mut self, shape: Vec<usize>) -> (B::HandleDevice, Arc<TensorId>) {
        self.handles.not_initialized(shape)
    }
}
