pub use burn_backend::tensor::BasicAutodiffOps;

use crate::{Tensor, TensorPrimitive, backend::AutodiffBackend};

impl<const D: usize, B: AutodiffBackend> Tensor<B, D> {
    /// Backward pass of the tensor.
    pub fn backward(&self) -> B::Gradients {
        B::backward(self.primitive.clone().tensor())
    }

    /// Get the gradients of a tensor if it exist.
    ///
    /// Returns a new reference to the same tensor. Therefore the same grad tensor can
    /// be accessed multiple times. If you only need to get the gradients one time,
    /// consider using [grad_remove](Tensor::grad_remove) for better performance.
    pub fn grad(&self, grads: &B::Gradients) -> Option<Tensor<B::InnerBackend, D>> {
        match &self.primitive {
            TensorPrimitive::Float(tensor) => B::grad(tensor, grads)
                .map(TensorPrimitive::Float)
                .map(Tensor::new),
            TensorPrimitive::QFloat(_tensor) => B::grad(&self.primitive.clone().tensor(), grads)
                .map(TensorPrimitive::Float)
                .map(Tensor::new),
        }
    }

    /// Remove the grad tensor from the [grads](AutodiffBackend::Gradients) struct returning the result.
    pub fn grad_remove(&self, grads: &mut B::Gradients) -> Option<Tensor<B::InnerBackend, D>> {
        match &self.primitive {
            TensorPrimitive::Float(tensor) => B::grad_remove(tensor, grads)
                .map(TensorPrimitive::Float)
                .map(Tensor::new),
            TensorPrimitive::QFloat(_tensor) => {
                B::grad_remove(&self.primitive.clone().tensor(), grads)
                    .map(TensorPrimitive::Float)
                    .map(Tensor::new)
            }
        }
    }

    /// Replace the grad tensor from the [grads](AutodiffBackend::Gradients) struct with the provided
    /// gradient.
    pub fn grad_replace(&self, grads: &mut B::Gradients, grad: Tensor<B::InnerBackend, D>) {
        match &self.primitive {
            TensorPrimitive::Float(tensor) => {
                B::grad_replace(tensor, grads, grad.primitive.tensor())
            }
            TensorPrimitive::QFloat(_tensor) => B::grad_replace(
                &self.primitive.clone().tensor(),
                grads,
                grad.primitive.tensor(),
            ),
        }
    }
}

impl<const D: usize, B: AutodiffBackend, K: BasicAutodiffOps<B>> Tensor<B, D, K> {
    /// Returns the inner tensor without the autodiff information.
    pub fn inner(self) -> Tensor<B::InnerBackend, D, K::InnerKind> {
        Tensor::new(K::inner(self.primitive))
    }

    /// Convert a tensor to the autodiff backend.
    ///
    /// # Arguments
    ///
    /// * `inner` - The tensor to convert.
    ///
    /// # Returns
    ///
    /// The tensor converted to the autodiff backend.
    pub fn from_inner(inner: Tensor<B::InnerBackend, D, K::InnerKind>) -> Self {
        Self::new(K::from_inner(inner.primitive))
    }
}
