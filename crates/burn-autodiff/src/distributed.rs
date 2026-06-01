use std::marker::PhantomData;

use crate::{collections::HashMap, grads::DistributedRegistration};
use burn_backend::{
    Backend, TensorPrimitive,
    distributed::{DistributedBackend, DistributedParams, TensorRef},
};
use burn_std::container::TensorContainer;

use crate::{NodeId, grads::GradID};

/// Submits sync operations on gradient registrations.
pub(crate) struct DistributedGradientRegistration<B: Backend> {
    n_required_map: HashMap<NodeId, usize>,
    sharded_parameters_map: HashMap<NodeId, DistributedParams>,
    _b: PhantomData<B>,
}

impl<B: DistributedBackend> DistributedGradientRegistration<B> {
    /// Creates a new registration and immediately registers the distributed parameters
    /// with the sync server so it can coordinate gradient reductions across devices.
    pub(crate) fn new(
        n_required_map: HashMap<NodeId, usize>,
        sharded_parameters_map: HashMap<NodeId, DistributedParams>,
        device: B::Device,
    ) -> Self {
        // For DDP, we register the distributed parameters of the tensors' nodes used in the graph and the number of times they
        // appear as nodes to know when to launch gradients reducing.
        println!("New distributed registration: {sharded_parameters_map:?}");
        println!("New n_required: {n_required_map:?}");
        if !sharded_parameters_map.is_empty() {
            B::register_sync_parameters(
                &device,
                sharded_parameters_map.values().cloned().collect(),
            );
        }

        Self {
            n_required_map,
            sharded_parameters_map,
            _b: PhantomData,
        }
    }
}

impl<B: DistributedBackend> DistributedRegistration for DistributedGradientRegistration<B> {
    fn on_register(&mut self, id: &NodeId, container: &mut TensorContainer<GradID>) {
        println!("On register node_id: {id}");
        if let Some(sharded_params) = self.sharded_parameters_map.get(id) {
            println!("On register {sharded_params:?}");
            let n_required = self.n_required_map.get_mut(id).unwrap();
            *n_required -= 1;

            if *n_required == 0 {
                let tensor_ref = container
                    .get_mut_ref::<TensorPrimitive<B>>(&id.value)
                    .unwrap();
                let tensor_ref = TensorRef(tensor_ref.get_mut_ref());
                B::submit_gradient_sync(tensor_ref, sharded_params.clone());
            }
        }
    }
}
