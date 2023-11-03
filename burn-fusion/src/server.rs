use crate::graph::{FusedBackend, Graph, GraphExecution, HandleContainer, Optimization, TensorOps};
use std::rc::Rc;

pub struct FusionServer<B, G>
where
    B: FusedBackend,
    G: GraphExecution<B>,
{
    optimizations: Vec<Optimization<B>>,
    graph: Graph<B>,
    handles: HandleContainer<B::Handle>,
    execution: G,
}

/// Trait name graph execution strategy.
impl<B, G> FusionServer<B, G>
where
    B: FusedBackend,
    G: GraphExecution<B>,
{
    pub fn register(&mut self, ops: TensorOps<B::FloatElem, B::IntElem>) {
        let ops = Rc::new(ops);
        self.graph.add(ops.clone());

        self.optimizations
            .iter_mut()
            .for_each(|optimization| optimization.register(&ops));

        self.execution
            .maybe_execute(&mut self.graph, &mut self.handles, &mut self.optimizations);
    }
}
