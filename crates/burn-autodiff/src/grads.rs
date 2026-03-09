use std::{collections::HashMap, sync::Arc};

use burn_backend::{
    Backend, ShardedParams, TensorMetadata, TensorPrimitive,
    ops::TensorRef,
    tensor::{FloatTensor, TensorContainer},
};

use crate::{
    NodeId,
    graph::{NodeRef, Requirement},
    tensor::AutodiffTensor,
};

/// Gradient identifier.
pub type GradID = u64;

/// Used to launch all_reduce operations on gradients registration.
pub struct GradientSyncRegistration {
    n_required_map: HashMap<NodeId, usize>,
    sharded_parameters_map: HashMap<NodeId, ShardedParams>,
}

impl GradientSyncRegistration {
    pub(crate) fn new(
        n_required_map: HashMap<NodeId, usize>,
        sharded_parameters_map: HashMap<NodeId, ShardedParams>,
    ) -> Self {
        println!("GradientSyncRegistration n_req : {:?}", n_required_map);
        println!(
            "GradientSyncRegistration sharded_params : {:?}",
            sharded_parameters_map
        );
        Self {
            n_required_map,
            sharded_parameters_map,
        }
    }

    pub(crate) fn on_register<B: Backend>(&mut self, id: NodeId, tensor: TensorRef<B>) {
        println!("Registering {id}");
        if let Some(sharded_params) = self.sharded_parameters_map.get(&id) {
            let param_id = sharded_params
                .param_id
                .expect("Sharded param should have a parameter ID."); // TODO: Remove option.
            println!("and param id {:?}", param_id);
            let n_required = self.n_required_map.get_mut(&id).unwrap();
            *n_required -= 1;

            if *n_required == 0 {
                println!("launch for node {id}, param {:?}", param_id);
                B::all_reduce_inplace(tensor, sharded_params.clone());
            }
        }
    }
}

/// Gradients container used during the backward pass.
pub struct Gradients {
    container: TensorContainer<GradID>,
    gradient_sync_registration: Option<GradientSyncRegistration>,
}

impl Gradients {
    /// Creates a new gradients container.
    pub fn new<B: Backend>(
        root_node: NodeRef,
        root_tensor: FloatTensor<B>,
        gradient_sync_registration: Option<GradientSyncRegistration>,
    ) -> Self {
        let mut gradients = Self {
            container: TensorContainer::new(),
            gradient_sync_registration,
        };
        gradients.register::<B>(
            root_node.id,
            B::float_ones(
                root_tensor.shape(),
                &B::float_device(&root_tensor),
                root_tensor.dtype().into(),
            ),
        );
        gradients
    }

    /// Consumes the gradients for a given tensor.
    ///
    /// Each tensor should be consumed exactly 1 time if its gradients are only required during the
    /// backward pass, otherwise, it may be consume multiple times.
    pub fn consume<B: Backend>(&mut self, node: &NodeRef) -> FloatTensor<B> {
        match node.requirement {
            Requirement::Grad => self
                .container
                .get::<B>(&node.id.value)
                .map(|tensor| tensor.tensor())
                .expect("Can't consume the gradients before they are registered at least once."),
            Requirement::GradInBackward => self
                .container
                .remove::<B>(&node.id.value)
                .map(|tensor| tensor.tensor())
                .expect("Can't consume the gradients before they are registered at least once."),
            Requirement::None => panic!("Trying to consume the gradients for an untracked tensor"),
        }
    }

    /// Removes a grad tensor from the container.
    pub fn remove<B: Backend>(&mut self, tensor: &AutodiffTensor<B>) -> Option<FloatTensor<B>> {
        self.container
            .remove::<B>(&tensor.node.id.value)
            .map(|tensor| tensor.tensor())
    }

    /// Gets a grad tensor from the container.
    pub fn get<B: Backend>(&self, tensor: &AutodiffTensor<B>) -> Option<FloatTensor<B>> {
        self.container
            .get::<B>(&tensor.node.id.value)
            .map(|tensor| tensor.tensor())
    }

    /// Register a grad tensor in the container.
    ///
    /// If the tensor already exists, add both tensors together before saving the result.
    ///
    /// If the registered tensor is sharded across multiple device, performs an all_reduce operation on the gradient.
    pub fn register<B: Backend>(&mut self, node_id: NodeId, value: FloatTensor<B>) {
        let out = if let Some(tensor_old) = self.container.remove::<B>(&node_id.value) {
            B::float_add(value, tensor_old.tensor())
        } else {
            value
        };

        self.container
            .register::<B>(node_id.value, TensorPrimitive::Float(out));
        if let Some(sync_registration) = self.gradient_sync_registration.as_mut() {
            let tensor_ref = self.container.get_mut_ref::<B>(&node_id.value).unwrap();
            let tensor_ref = TensorRef(Arc::new(tensor_ref.get_mut_ref()));
            sync_registration.on_register::<B>(node_id, tensor_ref);
        }

        // if let Some(sync_client) = get_gradient_sync_client::<B>(device) {
        // let tensor_ref = self.container.get_mut_ref::<B>(&node_id.value).unwrap();
        // let tensor_ref = TensorRef(Arc::new(tensor_ref.get_mut_ref()));

        // let id = B::comm_device(&tensor_ref).to_id();
        // println!("Device ID grad : {}", id);
        // sync_client.on_register(node_id, tensor_ref);

        // TODO: do the n_required stuff and then call the normal all_reduce_inplace
        // B::all_reduce_inplace(tensor, peer_id, all_ids, op);
        // };
    }
}
