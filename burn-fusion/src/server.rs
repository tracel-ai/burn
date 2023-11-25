use crate::{
    graph::{
        Action, EndCondition, Graph, GraphExecution, Ops, Optimization, Policy,
        TensorOpsDescription,
    },
    FusionBackend, FusionOps, FusionProperties, FusionStatus, HandleContainer, TensorId,
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
    policy: Policy<Box<dyn FusionOps<B>>>,
    pub device: B::FusionDevice,
    pub num_skipped: usize,
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
            policy: Policy::new(),
            num_skipped: 0,
            device,
        }
    }

    pub fn register(&mut self, ops_desc: TensorOpsDescription, ops: Box<dyn Ops<B>>) {
        let next_ops = self.graph.to_relative(&ops_desc);

        let action = self.policy.action(
            &mut self.graph.key,
            &self.graph.relative,
            EndCondition::NextOps(&next_ops),
        );

        match action {
            Action::Build => {
                if self.num_skipped > 0 {
                    // We register operations that were skipped.

                    let graph_size = self.graph.global.len();
                    let start = graph_size - self.num_skipped;
                    let end = graph_size;

                    for i in start..end {
                        let desc_skipped = self.graph.relative.get(i).unwrap();

                        self.optimizations
                            .iter_mut()
                            .for_each(|optimization| optimization.register(&desc_skipped));
                    }

                    // All updated.
                    self.num_skipped = 0;
                }

                // Now we can register the current operation.
                self.graph.add(ops_desc, next_ops, ops);
                let last = self.graph.relative.last().unwrap();

                self.optimizations
                    .iter_mut()
                    .for_each(|optimization| optimization.register(last));

                // Check if we can execute.
                self.execution.maybe_execute(
                    &mut self.graph,
                    &mut self.handles,
                    &mut self.optimizations,
                    &mut self.policy,
                    false,
                );
            }
            Action::Wait => {
                // Skip one more.
                self.num_skipped += 1;
                self.graph.add(ops_desc, next_ops, ops);
            }
            Action::Execute(exe) => {
                self.num_skipped = self.graph.len() - exe.len();
                self.graph
                    .execute_ops(&mut self.handles, &mut self.optimizations, exe);

                self.graph.add(ops_desc, next_ops, ops);
            }
        };
    }

    pub fn drain_graph(&mut self) {
        if self.graph.is_empty() && self.num_skipped == 0 {
            return;
        }

        let action = self.policy.action(
            &mut self.graph.key,
            &self.graph.relative,
            EndCondition::Forced,
        );

        match action {
            Action::Execute(exe) => {
                self.num_skipped = self.graph.len() - exe.len();
                self.graph
                    .execute_ops(&mut self.handles, &mut self.optimizations, exe);
                // TODO: Fix update path key.
            }
            _ => {
                if self.num_skipped > 0 {
                    let start = self.graph.global.len() - self.num_skipped;
                    for i in start..self.graph.relative.len() {
                        let desc = self.graph.relative.get(i).unwrap();
                        self.optimizations
                            .iter_mut()
                            .for_each(|optimization| optimization.register(&desc));
                    }
                    // All updated.
                    self.num_skipped = 0;
                }

                self.execution.maybe_execute(
                    &mut self.graph,
                    &mut self.handles,
                    &mut self.optimizations,
                    &mut self.policy,
                    true,
                );
            }
        };
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
