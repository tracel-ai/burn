use crate::collections::HashMap;

use burn_backend::{
    Backend, ShardedParams, TensorMetadata, TensorPrimitive,
    tensor::{FloatTensor, TensorContainer},
};
use burn_collective::all_reduce;

use crate::{
    NodeId,
    graph::{NodeRef, Requirement},
    tensor::AutodiffTensor,
};

#[derive(new, Debug)]
pub(crate) struct ShardedRegistration {
    sharded_params: ShardedParams,
    n_required: usize,
}

impl ShardedRegistration {
    pub(crate) fn on_register<B: Backend>(&mut self, tensor: FloatTensor<B>) -> FloatTensor<B> {
        self.n_required -= 1;
        match self.n_required {
            0 => all_reduce::<B>(self.sharded_params.peer_id, tensor, self.sharded_params.op)
                .expect("error all reduce in register"),
            _ => tensor,
        }
    }
}

/// Gradient identifier.
pub type GradID = u64;

/// Gradients container used during the backward pass.
pub struct Gradients {
    container: TensorContainer<GradID>,
    sharded_registration: HashMap<NodeId, ShardedRegistration>,
}

impl Gradients {
    /// Creates a new gradients container.
    pub fn new<B: Backend>(
        root_node: NodeRef,
        root_tensor: FloatTensor<B>,
        n_required_map: HashMap<NodeId, usize>,
        sharded_params_map: HashMap<NodeId, ShardedParams>,
    ) -> Self {
        let mut gradients = Self {
            container: TensorContainer::new(),
            sharded_registration: Self::sharded_registration_from_maps(
                n_required_map,
                sharded_params_map,
            ),
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

    fn sharded_registration_from_maps(
        n_required_map: HashMap<NodeId, usize>,
        sharded_params: HashMap<NodeId, ShardedParams>,
    ) -> HashMap<NodeId, ShardedRegistration> {
        let mut sharded_registration = HashMap::default();
        sharded_params.iter().for_each(|(k, v)| {
            sharded_registration.insert(
                *k,
                ShardedRegistration::new(v.clone(), *n_required_map.get(k).unwrap_or(&1)),
            );
        });
        sharded_registration
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
    /// If the tensor is sharded across multiple device, perform an all_reduce operation.
    pub fn register<B: Backend>(&mut self, node_id: NodeId, value: FloatTensor<B>) {
        let mut out = if let Some(tensor_old) = self.container.remove::<B>(&node_id.value) {
            B::float_add(value, tensor_old.tensor())
        } else {
            value
        };

        if let Some(registration) = self.sharded_registration.get_mut(&node_id) {
            out = registration.on_register::<B>(out);
        }

        self.container
            .register::<B>(node_id.value, TensorPrimitive::Float(out));
    }
}
