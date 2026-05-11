pub use burn_backend::tensor::BasicAutodiffOps;

use crate::{Tensor, kind::Autodiff};

#[cfg(feature = "autodiff")]
use crate::TensorPrimitive;
#[cfg(feature = "autodiff")]
use burn_backend::AutodiffBackend;
#[cfg(feature = "autodiff")]
use burn_dispatch::Dispatch;

/// Gradients container used during the backward pass.
#[cfg(feature = "autodiff")]
pub struct Gradients {
    // Encapsulate the inner type to avoid leaking internals into the top-level API.
    pub(crate) inner: <Dispatch as AutodiffBackend>::Gradients,
}

#[cfg(feature = "autodiff")]
impl Gradients {
    fn new(inner: <Dispatch as AutodiffBackend>::Gradients) -> Self {
        Self { inner }
    }
}

#[cfg(feature = "autodiff")]
impl<const D: usize> Tensor<D> {
    /// Backward pass of the tensor.
    pub fn backward(&self) -> Gradients {
        Gradients::new(Dispatch::backward(self.primitive.clone().tensor()))
    }

    /// Get the gradients of a tensor if it exist.
    ///
    /// Returns a new reference to the same tensor. Therefore the same grad tensor can
    /// be accessed multiple times. If you only need to get the gradients one time,
    /// consider using [grad_remove](Tensor::grad_remove) for better performance.
    pub fn grad(&self, grads: &Gradients) -> Option<Tensor<D>> {
        match &self.primitive {
            TensorPrimitive::Float(tensor) => Dispatch::grad(tensor, &grads.inner)
                .map(TensorPrimitive::Float)
                .map(Tensor::new),
            TensorPrimitive::QFloat(_tensor) => {
                Dispatch::grad(&self.primitive.clone().tensor(), &grads.inner)
                    .map(TensorPrimitive::Float)
                    .map(Tensor::new)
            }
        }
    }

    /// Remove the grad tensor from the [grads](AutodiffBackend::Gradients) struct returning the result.
    pub fn grad_remove(&self, grads: &mut Gradients) -> Option<Tensor<D>> {
        match &self.primitive {
            TensorPrimitive::Float(tensor) => Dispatch::grad_remove(tensor, &mut grads.inner)
                .map(TensorPrimitive::Float)
                .map(Tensor::new),
            TensorPrimitive::QFloat(_tensor) => {
                Dispatch::grad_remove(&self.primitive.clone().tensor(), &mut grads.inner)
                    .map(TensorPrimitive::Float)
                    .map(Tensor::new)
            }
        }
    }

    /// Replace the grad tensor from the [grads](AutodiffBackend::Gradients) struct with the provided
    /// gradient.
    pub fn grad_replace(&self, grads: &mut Gradients, grad: Tensor<D>) {
        match &self.primitive {
            TensorPrimitive::Float(tensor) => {
                Dispatch::grad_replace(tensor, &mut grads.inner, grad.primitive.tensor())
            }
            TensorPrimitive::QFloat(_tensor) => Dispatch::grad_replace(
                &self.primitive.clone().tensor(),
                &mut grads.inner,
                grad.primitive.tensor(),
            ),
        }
    }
}

impl<const D: usize, K: Autodiff<InnerKind = K>> Tensor<D, K> {
    /// Returns the inner tensor without the autodiff information.
    pub fn inner(self) -> Tensor<D, K> {
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
    pub fn from_inner(inner: Tensor<D, K>) -> Self {
        Self::new(K::from_inner(inner.primitive))
    }
}

// TODO: a lot of the `tensor.inner` / `Tensor::from_inner(...)` are actually scoped to perform some operations
// so it might be cleaner and easier to manage the device etc. if we provide a method to scope the autodiff?
