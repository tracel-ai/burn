use burn_tensor::{backend::Backend, container::TensorContainer};

use crate::{
    graph::{NodeRef, Requirement},
    tensor::AutodiffTensor,
    NodeID,
};

/// Gradient identifier.
pub type GradID = u64;

/// Gradients container used during the backward pass.
pub struct Gradients {
    container: TensorContainer<GradID>,
}

type TensorPrimitive<B> = <B as Backend>::FloatTensorPrimitive;

impl Gradients {
    /// Creates a new gradients container.
    pub fn new<B: Backend>(root_node: NodeRef, root_tensor: TensorPrimitive<B>) -> Self {
        let mut gradients = Self {
            container: TensorContainer::new(),
        };
        gradients.register::<B>(
            root_node.id,
            B::float_ones(B::float_shape(&root_tensor), &B::float_device(&root_tensor)),
        );
        gradients
    }

    /// Consumes the gradients for a given tensor.
    ///
    /// Each tensor should be consumed exactly 1 time if its gradients are only required during the
    /// backward pass, otherwise, it may be consume multiple times.
    pub fn consume<B: Backend>(&mut self, node: &NodeRef) -> TensorPrimitive<B> {
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
    pub fn remove<B: Backend>(&mut self, tensor: &AutodiffTensor<B>) -> Option<TensorPrimitive<B>> {
        self.container
            .remove::<B>(&tensor.node.id.value)
            .map(|tensor| tensor.tensor())
    }

    /// Gets a grad tensor from the container.
    pub fn get<B: Backend>(&self, tensor: &AutodiffTensor<B>) -> Option<TensorPrimitive<B>> {
        self.container
            .get::<B>(&tensor.node.id.value)
            .map(|tensor| tensor.tensor())
    }

    /// Register a grad tensor in the container.
    ///
    /// If the tensor already exists, add both tensors together before saving the result.
    pub fn register<B: Backend>(&mut self, node_id: NodeID, value: TensorPrimitive<B>) {
        if let Some(tensor_old) = self.container.remove::<B>(&node_id.value) {
            self.container.register::<B>(
                node_id.value,
                burn_tensor::TensorPrimitive::Float(B::float_add(value, tensor_old.tensor())),
            );
        } else {
            self.container
                .register::<B>(node_id.value, burn_tensor::TensorPrimitive::Float(value));
        }
    }
}
