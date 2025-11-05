use crate::{
    NodeId,
    checkpoint::strategy::{CheckpointStrategy, NoCheckpointing},
    collections::HashMap,
    grads::Gradients,
    runtime::AutodiffClient,
    tensor::AutodiffTensor,
};
use alloc::{format, string::String, sync::Arc};
use burn_tensor::{
    backend::{AutodiffBackend, Backend},
    ops::{BoolTensor, IntTensor, QuantizedTensor},
};
use core::marker::PhantomData;

/// Enable auto-differentiation on a backend.
///
/// This works as a backend decorator, extending the functionality of any backend with
/// backpropagation.
#[derive(Clone, Copy, Debug, Default)]
pub struct Autodiff<B, C = NoCheckpointing> {
    _b: PhantomData<B>,
    _checkpoint_strategy: PhantomData<C>,
}

impl<B: Backend, C: CheckpointStrategy> Backend for Autodiff<B, C> {
    type Device = B::Device;

    type FloatTensorPrimitive = AutodiffTensor<B>;
    type FloatElem = B::FloatElem;

    type IntTensorPrimitive = B::IntTensorPrimitive;
    type IntElem = B::IntElem;

    type BoolTensorPrimitive = B::BoolTensorPrimitive;
    type BoolElem = B::BoolElem;

    type QuantizedTensorPrimitive = B::QuantizedTensorPrimitive;

    fn ad_enabled() -> bool {
        true
    }

    fn name(device: &Self::Device) -> String {
        format!("autodiff<{}>", B::name(device))
    }

    fn seed(device: &B::Device, seed: u64) {
        B::seed(device, seed)
    }

    fn sync(device: &B::Device) {
        B::sync(device)
    }

    fn memory_persistent_allocations<Output, Input, Func: Fn(Input) -> Output>(
        device: &Self::Device,
        input: Input,
        func: Func,
    ) -> Output {
        B::memory_persistent_allocations(device, input, func)
    }

    fn memory_cleanup(device: &Self::Device) {
        B::memory_cleanup(device)
    }
}

impl<B: Backend, C: CheckpointStrategy> AutodiffBackend for Autodiff<B, C> {
    type InnerBackend = B;
    type Gradients = Gradients;

    fn backward(tensor: AutodiffTensor<B>) -> Gradients {
        let client = tensor.node.client.clone();

        AutodiffClient::backward::<B>(&client, tensor)
    }

    fn grad(tensor: &AutodiffTensor<B>, grads: &Gradients) -> Option<B::FloatTensorPrimitive> {
        grads.get::<B>(tensor)
    }

    fn grad_remove(
        tensor: &AutodiffTensor<B>,
        grads: &mut Gradients,
    ) -> Option<B::FloatTensorPrimitive> {
        grads.remove::<B>(tensor)
    }
    fn inner(tensor: AutodiffTensor<B>) -> B::FloatTensorPrimitive {
        tensor.primitive
    }

    fn from_inner(tensor: B::FloatTensorPrimitive) -> AutodiffTensor<B> {
        AutodiffTensor::new(tensor)
    }

    fn grad_replace(
        tensor: &AutodiffTensor<B>,
        grads: &mut Self::Gradients,
        grad: B::FloatTensorPrimitive,
    ) {
        grads.remove::<B>(tensor);
        grads.register::<B>(tensor.node.id, grad);
    }

    fn int_inner(tensor: IntTensor<Self>) -> IntTensor<Self::InnerBackend> {
        tensor
    }

    fn bool_inner(tensor: BoolTensor<Self>) -> BoolTensor<Self::InnerBackend> {
        tensor
    }

    fn int_from_inner(tensor: IntTensor<Self::InnerBackend>) -> IntTensor<Self> {
        tensor
    }

    fn bool_from_inner(tensor: BoolTensor<Self::InnerBackend>) -> BoolTensor<Self> {
        tensor
    }

    fn q_inner(tensor: QuantizedTensor<Self>) -> QuantizedTensor<Self::InnerBackend> {
        tensor
    }

    fn q_from_inner(tensor: QuantizedTensor<Self::InnerBackend>) -> QuantizedTensor<Self> {
        tensor
    }

    fn graph_cleanup() {
        let graphs_to_visit = {
            let graph_locator = crate::runtime::graph::STATE.lock();
            let graph_locator = graph_locator.as_ref().unwrap();
            let mut graphs_to_visit = HashMap::new();
            for (_node_id, graph) in &graph_locator.graphs {
                graphs_to_visit
                    .entry(graph.origin)
                    .or_insert_with(|| Arc::clone(graph));
            }
            graphs_to_visit
        };

        use crate::runtime::NodeCleaner;
        let mut cleaner = crate::runtime::graph::GraphCleaner::init();
        cleaner.cleanup_orphaned_entries();
        for (_graph_origin, graph) in graphs_to_visit {
            let mut state = graph.state.lock().unwrap();
            let server = &mut state.server;
            server
                .memory_management
                .free_unavailable_nodes(|node_id: &NodeId| {
                    server.steps.remove(node_id);
                    server.actions_builder.remove(node_id);
                    cleaner.clean(node_id);
                });
        }
    }
}
