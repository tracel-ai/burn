use burn_tensor::{
    container::TensorContainer, ops::FloatTensor, DynPrimBackend, Tensor,
};

use crate::{
    graph::{NodeRef, Requirement},
    tensor::AutodiffTensor,
};

/// Gradient identifier.
pub type GradId = u64;

/// Gradients container used during the backward pass.
#[derive(Debug)]
pub struct Gradients<P> {
    container: TensorContainer<GradId, P>,
}

impl<P> Gradients<P> {
    /// Creates a new gradient container.
    pub fn new<B: DynPrimBackend<P>, const D: usize>(
        root_node: NodeRef,
        root_tensor: FloatTensor<B, D>,
    ) -> Self {
        let mut gradients = Self {
            container: TensorContainer::new(),
        };
        gradients.register::<B, D>(
            root_node,
            B::float_ones(B::float_shape(&root_tensor), &B::float_device(&root_tensor)),
        );
        gradients
    }

    /// Removes a grad tensor from the container.
    pub fn remove<B: DynPrimBackend<P>, const D: usize>(
        &mut self,
        tensor: &AutodiffTensor<B, D>,
    ) -> Option<FloatTensor<B, D>> {
        self.container
            .remove::<B, D>(&tensor.node.id.value)
            .map(|tensor| tensor.into_primitive())
    }

    /// Register a grad tensor in the container.
    ///
    /// If the tensor already exists, add both tensors together before saving the result.
    pub fn register<B: DynPrimBackend<P>, const D: usize>(
        &mut self,
        node: NodeRef,
        value: FloatTensor<B, D>,
    ) {
        if let Some(tensor_old) = self.container.remove::<B, D>(&node.id.value) {
            self.container
                .register(node.id.value, Tensor::from_primitive(value).add(tensor_old));
        } else {
            self.container
                .register::<B, D>(node.id.value, Tensor::from_primitive(value));
        }
    }
}

impl<P: Clone> Gradients<P> {
    /// Gets a grad tensor from the container.
    pub fn get<B: DynPrimBackend<P>, const D: usize>(
        &self,
        tensor: &AutodiffTensor<B, D>,
    ) -> Option<FloatTensor<B, D>> {
        self.container
            .get::<B, D>(&tensor.node.id.value)
            .map(|tensor| tensor.into_primitive())
    }

    /// Consumes the gradients for a given tensor.
    ///
    /// Each tensor should be consumed exactly 1 time if its gradients are only required during the
    /// backward pass, otherwise, it may be consumed multiple times.
    pub fn consume<B: DynPrimBackend<P>, const D: usize>(
        &mut self,
        node: &NodeRef,
    ) -> FloatTensor<B, D> {
        match node.requirement {
            Requirement::Grad => self
                .container
                .get::<B, D>(&node.id.value)
                .map(|tensor| tensor.into_primitive())
                .expect("Can't consume the gradients before they are registered at least once."),
            Requirement::GradInBackward => self
                .container
                .remove::<B, D>(&node.id.value)
                .map(|tensor| tensor.into_primitive())
                .expect("Can't consume the gradients before they are registered at least once."),
            Requirement::None => panic!("Trying to consume the gradients for an untracked tensor"),
        }
    }
}
