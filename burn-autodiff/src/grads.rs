use burn_tensor::{backend::Backend, container::TensorContainer, Tensor};

use crate::{
    graph::{NodeRef, Requirement},
    tensor::ADTensor,
};

/// Gradient identifier.
pub type GradID = String;

/// Gradients container used during the backward pass.
pub struct Gradients {
    container: TensorContainer<GradID>,
}

type TensorPrimitive<B, const D: usize> = <B as Backend>::TensorPrimitive<D>;

impl Gradients {
    /// Creates a new gradients container.
    pub fn new<B: Backend, const D: usize>(
        root_node: NodeRef,
        root_tensor: TensorPrimitive<B, D>,
    ) -> Self {
        let mut gradients = Self {
            container: TensorContainer::new(),
        };
        gradients.register::<B, D>(
            root_node,
            B::ones(B::shape(&root_tensor), &B::device(&root_tensor)),
        );
        gradients
    }

    /// Consumes the gradients for a given tensor.
    ///
    /// Each tensor should be consumed exactly 1 time if its gradients are only required during the
    /// backward pass, otherwise, it may be consume multiple times.
    pub fn consume<B: Backend, const D: usize>(&mut self, node: &NodeRef) -> TensorPrimitive<B, D> {
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

    /// Removes a grad tensor from the container.
    pub fn remove<B: Backend, const D: usize>(
        &mut self,
        tensor: &ADTensor<B, D>,
    ) -> Option<TensorPrimitive<B, D>> {
        self.container
            .remove::<B, D>(&tensor.node.id.value)
            .map(|tensor| tensor.into_primitive())
    }

    /// Gets a grad tensor from the container.
    pub fn get<B: Backend, const D: usize>(
        &self,
        tensor: &ADTensor<B, D>,
    ) -> Option<TensorPrimitive<B, D>> {
        self.container
            .get::<B, D>(&tensor.node.id.value)
            .map(|tensor| tensor.into_primitive())
    }

    /// Register a grad tensor in the container.
    ///
    /// If the tensor already exists, add both tensors together before saving the result.
    pub fn register<B: Backend, const D: usize>(
        &mut self,
        node: NodeRef,
        value: TensorPrimitive<B, D>,
    ) {
        if let Some(tensor_old) = self.container.remove::<B, D>(&node.id.value) {
            self.container.register(
                node.id.value.clone(),
                Tensor::from_primitive(value).add(tensor_old),
            );
        } else {
            self.container
                .register::<B, D>(node.id.value.clone(), Tensor::from_primitive(value));
        }
    }
}
