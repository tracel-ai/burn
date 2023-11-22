use crate::{
    graph::{Graph, GraphExecution, Ops, Optimization, TensorOpsDescription},
    FusionBackend, FusionProperties, FusionStatus, HandleContainer, TensorId,
};
use burn_tensor::ops::{FloatElem, IntElem};
use std::sync::Arc;

pub struct FusionServer<B, G>
where
    B: FusionBackend,
    G: GraphExecution<B>,
{
    optimizations: Vec<Optimization<B>>,
    graph: Graph<B>,
    pub(crate) handles: HandleContainer<B>,
    execution: G,
    pub device: B::FusionDevice,
}

impl<B, G> FusionServer<B, G>
where
    B: FusionBackend,
    G: GraphExecution<B>,
{
    pub fn new(device: B::FusionDevice) -> Self {
        let optimizations = B::operations(&device.clone().into())
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

    pub fn register(&mut self, desc: TensorOpsDescription, op: Box<dyn Ops<B>>) {
        let ops = Arc::new(desc);
        self.graph.add(ops.clone(), op);

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

    pub fn drain_graph(&mut self) {
        if self.graph.is_empty() {
            return;
        }

        self.execution.maybe_execute(
            &mut self.graph,
            &mut self.handles,
            &mut self.optimizations,
            true,
        );
    }

    pub fn create_empty_handle(&mut self) -> Arc<TensorId> {
        self.handles.create_tensor_uninit()
    }

    pub fn read_float<const D: usize>(
        &mut self,
        tensor: crate::TensorDescription,
    ) -> burn_tensor::Reader<burn_tensor::Data<FloatElem<B>, D>> {
        // Make sure all registered operations are executed.
        // The underlying backend can still be async.
        self.drain_graph();

        let tensor = self.handles.get_float_tensor(&tensor);
        B::into_data(tensor)
    }

    pub fn read_int<const D: usize>(
        &mut self,
        tensor: crate::TensorDescription,
    ) -> burn_tensor::Reader<burn_tensor::Data<IntElem<B>, D>> {
        // Make sure all registered operations are executed.
        // The underlying backend can still be async.
        self.drain_graph();

        let tensor = self.handles.get_int_tensor(&tensor);
        B::int_into_data(tensor)
    }

    pub fn read_bool<const D: usize>(
        &mut self,
        tensor: crate::TensorDescription,
    ) -> burn_tensor::Reader<burn_tensor::Data<bool, D>> {
        // Make sure all registered operations are executed.
        // The underlying backend can still be async.
        self.drain_graph();

        let tensor = self.handles.get_bool_tensor(&tensor);
        B::bool_into_data(tensor)
    }

    pub fn change_server_float<const D: usize>(
        &mut self,
        tensor: &crate::TensorDescription,
        device: &B::Device,
        server_device: &mut Self,
    ) -> Arc<TensorId> {
        let tensor = self.handles.get_float_tensor::<D>(tensor);
        let tensor = B::to_device(tensor, device);
        let id = server_device.create_empty_handle();

        server_device
            .handles
            .register_float_tensor(&id, tensor.clone());

        id
    }
    pub fn change_server_int<const D: usize>(
        &mut self,
        tensor: &crate::TensorDescription,
        device: &B::Device,
        server_device: &mut Self,
    ) -> Arc<TensorId> {
        let tensor = self.handles.get_int_tensor::<D>(tensor);
        let tensor = B::int_to_device(tensor, device);
        let id = server_device.create_empty_handle();

        server_device
            .handles
            .register_int_tensor(&id, tensor.clone());

        id
    }
    pub fn change_server_bool<const D: usize>(
        &mut self,
        tensor: &crate::TensorDescription,
        device: &B::Device,
        server_device: &mut Self,
    ) -> Arc<TensorId> {
        let tensor = self.handles.get_bool_tensor::<D>(tensor);
        let tensor = B::bool_to_device(tensor, device);
        let id = server_device.create_empty_handle();

        server_device
            .handles
            .register_bool_tensor(&id, tensor.clone());

        id
    }

    pub fn drop_tensor_handle(&mut self, id: TensorId) {
        self.handles.handles_orphan.push(id);
    }
}
