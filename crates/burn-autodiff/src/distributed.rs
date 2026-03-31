use std::marker::PhantomData;

use crate::collections::HashMap;
use burn_backend::{
    distributed::{DistributedBackend, DistributedParams, TensorRef},
    tensor::TensorContainer,
};

use crate::{NodeId, grads::GradID};

/// Trait for registering distributed gradients.
pub trait DistributedRegistration {
    /// Performs distributed registration operations on the tensor with the corresponding [`NodeId`].
    fn on_register(&mut self, node_id: &NodeId, container: &mut TensorContainer<GradID>);
}

/// Submits sync operations on gradient registrations.
#[derive(new)]
pub struct DistributedGradientRegistration<B: DistributedBackend> {
    n_required_map: HashMap<NodeId, usize>,
    sharded_parameters_map: HashMap<NodeId, DistributedParams>,
    _backend: PhantomData<B>,
}

impl<B: DistributedBackend> DistributedRegistration for DistributedGradientRegistration<B> {
    fn on_register(&mut self, id: &NodeId, container: &mut TensorContainer<GradID>) {
        if let Some(sharded_params) = self.sharded_parameters_map.get(id) {
            let n_required = self.n_required_map.get_mut(id).unwrap();
            *n_required -= 1;

            if *n_required == 0 {
                let tensor_ref = container.get_mut_ref::<B>(&id.value).unwrap();
                let tensor_ref = TensorRef(tensor_ref.get_mut_ref());
                B::submit_gradient_sync(tensor_ref, sharded_params.clone());
            }
        }
    }
}
