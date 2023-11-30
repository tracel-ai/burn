use crate::{
    graph::{
        execution::{ExecutionMode, GraphExecution},
        Graph, Ops, TensorOpsDescription,
    },
    FusionBackend, HandleContainer, TensorId,
};
use burn_tensor::ops::{FloatElem, IntElem};
use std::sync::Arc;

pub struct FusionServer<B>
where
    B: FusionBackend,
{
    execution: GraphExecution<B>,
    graph: Graph<B>,
    pub(crate) handles: HandleContainer<B>,
    pub device: B::FusionDevice,
    pub num_skipped: usize,
}

impl<B> FusionServer<B>
where
    B: FusionBackend,
{
    pub fn new(device: B::FusionDevice) -> Self {
        Self {
            execution: GraphExecution::new(B::optimizations(&device.clone().into())),
            graph: Graph::new(),
            handles: HandleContainer::new(device.clone()),
            num_skipped: 0,
            device,
        }
    }

    pub fn register(&mut self, ops_desc: TensorOpsDescription, ops: Box<dyn Ops<B>>) {
        self.graph.add(ops_desc, ops);
        self.execution
            .execute(&mut self.graph, &mut self.handles, ExecutionMode::NewOps);
    }

    pub fn drain_graph(&mut self) {
        // Check if we can execute.
        self.execution
            .execute(&mut self.graph, &mut self.handles, ExecutionMode::Sync);
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
