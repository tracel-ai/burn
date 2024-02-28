use burn_tensor::{backend::Backend, container::TensorContainer, ops::FloatTensor, Tensor};

use crate::{
    graph::{NodeRef, Requirement},
    tensor::AutodiffTensor,
};

/// Gradient identifier.
pub type GradId = u64;

/// Gradients container used during the backward pass.
pub struct Gradients<B: Backend> {
    container: TensorContainer<GradId, B>,
}

impl<B: Backend> Gradients<B> {
    /// Creates a new gradients container.
    pub fn new<const D: usize>(root_node: NodeRef, root_tensor: FloatTensor<B, D>) -> Self {
        let mut gradients = Self {
            container: TensorContainer::new(),
        };
        gradients.register::<B, D>(
            root_node,
            B::float_ones(B::float_shape(&root_tensor), &B::float_device(&root_tensor)),
        );
        gradients
    }

    /// Consumes the gradients for a given tensor.
    ///
    /// Each tensor should be consumed exactly 1 time if its gradients are only required during the
    /// backward pass, otherwise, it may be consume multiple times.
    pub fn consume<BOut: Backend<DynTensorPrimitive = B::DynTensorPrimitive>, const D: usize>(
        &mut self,
        node: &NodeRef,
    ) -> FloatTensor<BOut, D> {
        match node.requirement {
            Requirement::Grad => self
                .container
                .get::<BOut, D>(&node.id.value)
                .map(|tensor| tensor.into_primitive())
                .expect("Can't consume the gradients before they are registered at least once."),
            Requirement::GradInBackward => self
                .container
                .remove::<BOut, D>(&node.id.value)
                .map(|tensor| tensor.into_primitive())
                .expect("Can't consume the gradients before they are registered at least once."),
            Requirement::None => panic!("Trying to consume the gradients for an untracked tensor"),
        }
    }

    /// Removes a grad tensor from the container.
    pub fn remove<BOut: Backend<DynTensorPrimitive = B::DynTensorPrimitive>, const D: usize>(
        &mut self,
        tensor: &AutodiffTensor<BOut, D>,
    ) -> Option<FloatTensor<BOut, D>> {
        self.container
            .remove::<BOut, D>(&tensor.node.id.value)
            .map(|tensor| tensor.into_primitive())
    }

    /// Gets a grad tensor from the container.
    pub fn get<BOut: Backend<DynTensorPrimitive = B::DynTensorPrimitive>, const D: usize>(
        &self,
        tensor: &AutodiffTensor<BOut, D>,
    ) -> Option<FloatTensor<BOut, D>> {
        self.container
            .get::<BOut, D>(&tensor.node.id.value)
            .map(|tensor| tensor.into_primitive())
    }

    /// Register a grad tensor in the container.
    ///
    /// If the tensor already exists, add both tensors together before saving the result.
    pub fn register<BOut: Backend<DynTensorPrimitive = B::DynTensorPrimitive>, const D: usize>(
        &mut self,
        node: NodeRef,
        value: FloatTensor<BOut, D>,
    ) {
        if let Some(tensor_old) = self.container.remove::<BOut, D>(&node.id.value) {
            self.container
                .register(node.id.value, Tensor::from_primitive(value).add(tensor_old));
        } else {
            self.container
                .register::<BOut, D>(node.id.value, Tensor::from_primitive(value));
        }
    }
}
