use burn_tensor::{backend::Backend, container::TensorContainer, Tensor};

use crate::{
    graph::{NodeRef, Requirement},
    tensor::{ADTensor, BackwardTensor},
};

pub type GradID = String;

/// Gradients container used during the backward pass.
#[derive(Default)]
pub struct Gradients<B: Backend> {
    container: TensorContainer<B, GradID>,
}

impl<B: Backend> Gradients<B> {
    /// Create a new container.
    pub fn new() -> Self {
        Self::default()
    }
    /// Consume the gradient for a given tensor.
    ///
    /// Each tensor should be consumed exactly 1 time if its gradient is only required during the
    /// backward pass.
    pub fn consume<const D: usize>(
        &mut self,
        tensor: &BackwardTensor<B, D>,
    ) -> B::TensorPrimitive<D> {
        match tensor.node.requirement {
            Requirement::Grad => self
                .container
                .get(&tensor.node.id.value)
                .map(|tensor| tensor.into_primitive())
                .unwrap(),
            Requirement::GradInBackward => self
                .container
                .remove(&tensor.node.id.value)
                .map(|tensor| tensor.into_primitive())
                .unwrap(),
            Requirement::None => panic!("Trying to consume the gradients for an untracked tensor"),
        }
    }

    pub fn get<const D: usize>(&self, tensor: &ADTensor<B, D>) -> Option<B::TensorPrimitive<D>> {
        self.container
            .get(&tensor.node.id.value)
            .map(|tensor| tensor.into_primitive())
    }

    pub fn update<const D: usize>(&mut self, node: NodeRef, value: B::TensorPrimitive<D>) {
        if let Some(tensor_old) = self.container.remove(&node.id.value) {
            self.container.register(
                node.id.value.clone(),
                Tensor::from_primitive(value).add(tensor_old),
            );
        } else {
            self.container
                .register(node.id.value.clone(), Tensor::from_primitive(value));
        }
    }
}
