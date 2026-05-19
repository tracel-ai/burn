use crate::{Tensor, kind::Autodiff};

#[cfg(feature = "autodiff")]
use crate::ops::BridgeTensor;
#[cfg(feature = "autodiff")]
use burn_backend::AutodiffBackend;
#[cfg(feature = "autodiff")]
use burn_dispatch::Dispatch;
#[cfg(feature = "autodiff")]
use core::mem::MaybeUninit;

#[cfg(feature = "autodiff")]
type AutodiffGradients = <Dispatch as AutodiffBackend>::Gradients;
#[cfg(feature = "autodiff")]
type GradientsInner = MaybeUninit<AutodiffGradients>;

/// Storage for [`Gradients`]. Holds the raw bytes of the dispatch-level
/// gradients container.
///
/// Intentionally has no type-level alignment marker (e.g.
/// `[GradientsInner; 0]`), since that would re-introduce a `burn_dispatch`
/// dependency in the type itself and undermine the compile-time goal of this
/// obfuscation. Alignment must therefore be handled at access sites
/// (TODO: bring back proper alignment without leaking the type).
#[cfg(feature = "autodiff")]
struct GradientsBlob {
    bytes: [u8; size_of::<GradientsInner>()],
}

/// Gradients container used during the backward pass.
#[cfg(feature = "autodiff")]
pub struct Gradients {
    blob: GradientsBlob,
}

#[cfg(feature = "autodiff")]
impl Drop for Gradients {
    fn drop(&mut self) {
        unsafe {
            let inner: &mut GradientsInner =
                &mut *(self.blob.bytes.as_mut_ptr() as *mut GradientsInner);
            inner.assume_init_drop();
        }
    }
}

#[cfg(feature = "autodiff")]
impl Gradients {
    /// Crate-internal constructor wrapping the dispatch-level gradients.
    pub(crate) fn from_inner(inner: AutodiffGradients) -> Self {
        let mut blob = GradientsBlob {
            bytes: [0u8; size_of::<GradientsInner>()],
        };
        unsafe {
            let dst = blob.bytes.as_mut_ptr() as *mut GradientsInner;
            dst.write(MaybeUninit::new(inner));
        }
        Self { blob }
    }

    /// Crate-internal borrow of the underlying gradients container.
    pub(crate) fn as_inner(&self) -> &AutodiffGradients {
        unsafe {
            let inner: &GradientsInner = &*(self.blob.bytes.as_ptr() as *const GradientsInner);
            inner.assume_init_ref()
        }
    }

    /// Crate-internal mutable borrow of the underlying gradients container.
    pub(crate) fn as_inner_mut(&mut self) -> &mut AutodiffGradients {
        unsafe {
            let inner: &mut GradientsInner =
                &mut *(self.blob.bytes.as_mut_ptr() as *mut GradientsInner);
            inner.assume_init_mut()
        }
    }
}

#[cfg(feature = "autodiff")]
impl<const D: usize> Tensor<D> {
    /// Backward pass of the tensor.
    pub fn backward(&self) -> Gradients {
        backward_impl(&self.primitive)
    }

    /// Get the gradients of a tensor if it exist.
    ///
    /// Returns a new reference to the same tensor. Therefore the same grad tensor can
    /// be accessed multiple times. If you only need to get the gradients one time,
    /// consider using [grad_remove](Tensor::grad_remove) for better performance.
    pub fn grad(&self, grads: &Gradients) -> Option<Tensor<D>> {
        grad_impl(&self.primitive, grads).map(Tensor::new)
    }

    /// Remove the grad tensor from the [grads](AutodiffBackend::Gradients) struct returning the result.
    pub fn grad_remove(&self, grads: &mut Gradients) -> Option<Tensor<D>> {
        grad_remove_impl(&self.primitive, grads).map(Tensor::new)
    }

    /// Replace the grad tensor from the [grads](AutodiffBackend::Gradients) struct with the provided
    /// gradient.
    pub fn grad_replace(&self, grads: &mut Gradients, grad: Tensor<D>) {
        grad_replace_impl(&self.primitive, grads, grad.primitive)
    }
}

#[cfg(feature = "autodiff")]
fn backward_impl(p: &BridgeTensor) -> Gradients {
    Gradients::from_inner(Dispatch::backward(p.clone().into_float()))
}

#[cfg(feature = "autodiff")]
fn grad_impl(p: &BridgeTensor, grads: &Gradients) -> Option<BridgeTensor> {
    Dispatch::grad(p.as_float(), grads.as_inner()).map(BridgeTensor::float)
}

#[cfg(feature = "autodiff")]
fn grad_remove_impl(p: &BridgeTensor, grads: &mut Gradients) -> Option<BridgeTensor> {
    Dispatch::grad_remove(p.as_float(), grads.as_inner_mut()).map(BridgeTensor::float)
}

#[cfg(feature = "autodiff")]
fn grad_replace_impl(p: &BridgeTensor, grads: &mut Gradients, grad: BridgeTensor) {
    Dispatch::grad_replace(p.as_float(), grads.as_inner_mut(), grad.into_float())
}

impl<const D: usize, K: Autodiff> Tensor<D, K> {
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
