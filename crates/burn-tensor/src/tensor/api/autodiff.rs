use crate::{Tensor, kind::Autodiff};

#[cfg(feature = "autodiff")]
use crate::ops::{BridgeKind, BridgeTensor};
#[cfg(feature = "autodiff")]
use burn_backend::AutodiffBackend;
#[cfg(feature = "autodiff")]
use burn_dispatch::Dispatch;
#[cfg(feature = "autodiff")]
use burn_dispatch::{DispatchAutodiffContext, GradientCheckpointingStrategy};

#[cfg(feature = "autodiff")]
type AutodiffGradients = <Dispatch as AutodiffBackend>::Gradients;

// Aligned, type-erased storage for `AutodiffGradients`. See `crate::macros`
// for why this indirection exists.
#[cfg(feature = "autodiff")]
burn_std::obfuscate!(
    type: AutodiffGradients,
    module: gradients_opaque,
    derives: [Send]
);

/// Gradients container used during the backward pass.
#[cfg(feature = "autodiff")]
pub struct Gradients {
    blob: gradients_opaque::Opaque,
}

#[cfg(feature = "autodiff")]
impl Gradients {
    /// Crate-internal constructor wrapping the dispatch-level gradients.
    pub(crate) fn from_inner(inner: AutodiffGradients) -> Self {
        Self {
            blob: gradients_opaque::Opaque::new(inner),
        }
    }

    /// Crate-internal borrow of the underlying gradients container.
    pub(crate) fn as_inner(&self) -> &AutodiffGradients {
        self.blob.as_ref()
    }

    /// Crate-internal mutable borrow of the underlying gradients container.
    pub(crate) fn as_inner_mut(&mut self) -> &mut AutodiffGradients {
        self.blob.as_mut()
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
    // A non-float tensor — a packed base included — records no tape, so there
    // is no gradient to look up.
    Dispatch::grad(p.try_as_float()?, grads.as_inner()).map(BridgeTensor::float)
}

#[cfg(feature = "autodiff")]
fn grad_remove_impl(p: &BridgeTensor, grads: &mut Gradients) -> Option<BridgeTensor> {
    Dispatch::grad_remove(p.try_as_float()?, grads.as_inner_mut()).map(BridgeTensor::float)
}

#[cfg(feature = "autodiff")]
fn grad_replace_impl(p: &BridgeTensor, grads: &mut Gradients, grad: BridgeTensor) {
    Dispatch::grad_replace(p.as_float(), grads.as_inner_mut(), grad.into_float())
}

impl<const D: usize, K: Autodiff> Tensor<D, K> {
    /// Returns this tensor without its autodiff association.
    ///
    /// If the tensor uses autodiff, this moves it to the inner backend and drops any graph
    /// reference it carries. If autodiff is already disabled, the tensor is returned unchanged.
    /// This operation is idempotent.
    ///
    /// This is equivalent to [`without_autodiff`](Tensor::without_autodiff). `inner` reflects the
    /// underlying backend-decorator model, while `without_autodiff` describes the operation in
    /// terms of the high-level tensor API.
    pub fn inner(self) -> Tensor<D, K> {
        self.without_autodiff()
    }

    /// Returns this tensor without its autodiff association.
    ///
    /// If the tensor uses autodiff, this moves it to the inner backend and drops any graph
    /// reference it carries. If autodiff is already disabled, the tensor is returned unchanged.
    /// This operation is idempotent.
    ///
    /// Unlike [`detach`](Tensor::detach), which severs the tensor from its current graph but keeps
    /// its autodiff association, the returned tensor pays no autodiff dispatch for subsequent
    /// operations involving only tensors without autodiff. When combined with a tensor that still
    /// uses autodiff, it is treated as a constant for that operation.
    pub fn without_autodiff(self) -> Self {
        if self.device().is_autodiff() {
            Tensor::new(K::inner(self.primitive))
        } else {
            self
        }
    }

    /// Returns this tensor with autodiff enabled.
    ///
    /// If autodiff is disabled, this associates the tensor with the autodiff backend using the
    /// default gradient-checkpointing strategy. If autodiff is already enabled, the tensor and its
    /// current strategy are returned unchanged. This operation is idempotent.
    ///
    /// Enabling autodiff does not make a floating-point tensor require gradients. Use
    /// [`require_grad`](Tensor::require_grad) when its gradient should be retained during the
    /// backward pass.
    pub fn autodiff(self) -> Self {
        if self.device().is_autodiff() {
            self
        } else {
            Self::new(K::from_inner(self.primitive))
        }
    }

    /// Returns the provided tensor with autodiff enabled.
    ///
    /// If the tensor does not yet use autodiff, this associates it with the autodiff backend using
    /// the default gradient-checkpointing strategy. If autodiff is already enabled, the tensor and
    /// its current strategy are returned unchanged. This operation is idempotent.
    ///
    /// This is equivalent to [`autodiff`](Tensor::autodiff). `from_inner` reflects the underlying
    /// backend-decorator model, while `autodiff` describes the operation in terms of the high-level
    /// tensor API.
    ///
    /// Enabling autodiff does not make a floating-point tensor require gradients. Use
    /// [`require_grad`](Tensor::require_grad) when its gradient should be retained during the
    /// backward pass.
    pub fn from_inner(inner: Tensor<D, K>) -> Self {
        inner.autodiff()
    }

    /// Sets the autodiff checkpointing strategy carried by this tensor.
    ///
    /// The strategy is normally derived from the device the tensor was created on (see
    /// [`Device::gradient_checkpointing`](crate::Device::gradient_checkpointing)); this
    /// method overrides it for a single tensor. A tensor carrying a strategy is treated
    /// as tracked by autodiff, so this also marks an inner-backend tensor for tracking.
    ///
    /// # Panics
    ///
    /// Operations combining tensors that carry different strategies panic; make sure all
    /// operands share the same one.
    #[cfg(feature = "autodiff")]
    pub fn with_gradient_checkpointing_strategy(
        self,
        strategy: GradientCheckpointingStrategy,
    ) -> Self {
        let primitive = match self.primitive.as_parts().1.autodiff {
            DispatchAutodiffContext::Disabled => K::from_inner(self.primitive),
            DispatchAutodiffContext::Enabled(_) => self.primitive,
        };
        let (kind, mut tensor) = primitive.into_parts();
        tensor.autodiff = DispatchAutodiffContext::Enabled(strategy);
        Self::new(match kind {
            BridgeKind::Bool => BridgeTensor::bool(tensor),
            BridgeKind::Int => BridgeTensor::int(tensor),
            BridgeKind::Float => BridgeTensor::float(tensor),
            BridgeKind::QFloat => BridgeTensor::qfloat(tensor),
        })
    }
}
