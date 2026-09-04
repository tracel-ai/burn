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
    /// Computes gradients by backpropagating from this tensor.
    ///
    /// The tensor must participate in an autodiff graph. Backward consumes the shared graph tape,
    /// even though this method borrows the tensor, so calling it again through this tensor or one
    /// of its clones may panic.
    ///
    /// # Panics
    ///
    /// Panics if autodiff is disabled, the tensor doesn't participate in a recorded graph, or the
    /// graph tape has already been consumed.
    pub fn backward(&self) -> Gradients {
        backward_impl(&self.primitive)
    }

    /// Returns this tensor's retained gradient, if present.
    ///
    /// The returned gradient doesn't have an autodiff association. This returns `None` when the
    /// tensor is outside autodiff, doesn't retain gradients, or isn't present in `grads`.
    /// Repeated calls return handles to the same gradient. If the gradient is only needed once,
    /// prefer [`grad_remove`](Tensor::grad_remove), which can enable in-place optimizations.
    pub fn grad(&self, grads: &Gradients) -> Option<Tensor<D>> {
        grad_impl(&self.primitive, grads).map(Tensor::new)
    }

    /// Removes and returns this tensor's retained gradient, if present.
    ///
    /// The returned gradient doesn't have an autodiff association. This returns `None` when the
    /// tensor is outside autodiff, doesn't retain gradients, or isn't present in `grads`.
    pub fn grad_remove(&self, grads: &mut Gradients) -> Option<Tensor<D>> {
        grad_remove_impl(&self.primitive, grads).map(Tensor::new)
    }

    /// Replaces this tensor's entry in `grads` with `grad`.
    ///
    /// # Panics
    ///
    /// Panics if this tensor isn't associated with autodiff, if `grad` has an autodiff association,
    /// or if the tensors use incompatible backends.
    pub fn grad_replace(&self, grads: &mut Gradients, grad: Tensor<D>) {
        grad_replace_impl(&self.primitive, grads, grad.primitive)
    }

    /// Returns whether this tensor participates in a recorded autodiff graph.
    ///
    /// A tensor can have autodiff enabled without being tracked, such as a constant that doesn't
    /// require gradients. [`is_autodiff`](Tensor::is_autodiff) reports whether autodiff is enabled
    /// for the tensor, while [`is_require_grad`](Tensor::is_require_grad) reports whether its
    /// gradient is retained after backward.
    pub fn is_tracked(&self) -> bool {
        is_tracked_impl(&self.primitive)
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
    let tensor = p.try_as_float()?;
    if tensor.autodiff == DispatchAutodiffContext::Disabled {
        return None;
    }
    Dispatch::grad(tensor, grads.as_inner()).map(BridgeTensor::float)
}

#[cfg(feature = "autodiff")]
fn grad_remove_impl(p: &BridgeTensor, grads: &mut Gradients) -> Option<BridgeTensor> {
    let tensor = p.try_as_float()?;
    if tensor.autodiff == DispatchAutodiffContext::Disabled {
        return None;
    }
    Dispatch::grad_remove(tensor, grads.as_inner_mut()).map(BridgeTensor::float)
}

#[cfg(feature = "autodiff")]
fn grad_replace_impl(p: &BridgeTensor, grads: &mut Gradients, grad: BridgeTensor) {
    Dispatch::grad_replace(p.as_float(), grads.as_inner_mut(), grad.into_float())
}

#[cfg(feature = "autodiff")]
fn is_tracked_impl(p: &BridgeTensor) -> bool {
    p.try_as_float().is_some_and(Dispatch::is_tracked)
}

impl<const D: usize, K: Autodiff> Tensor<D, K> {
    /// Returns whether autodiff is enabled for this tensor.
    ///
    /// This doesn't indicate whether the tensor participates in a recorded graph or retains its
    /// gradient. Inspect those properties with [`is_tracked`](Tensor::is_tracked) and
    /// [`is_require_grad`](Tensor::is_require_grad), respectively.
    pub fn is_autodiff(&self) -> bool {
        self.device().is_autodiff()
    }

    /// Returns this tensor without its autodiff association.
    ///
    /// If the tensor uses autodiff, this moves it to the inner backend and drops any graph
    /// reference it carries. If autodiff is already disabled, the tensor is returned unchanged.
    /// This operation is idempotent.
    ///
    /// This is equivalent to [`without_autodiff`](Tensor::without_autodiff). `inner` reflects the
    /// underlying backend-decorator model, while `without_autodiff` describes the operation in
    /// terms of the high-level tensor API.
    #[must_use]
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
    #[must_use]
    pub fn without_autodiff(self) -> Self {
        if self.is_autodiff() {
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
    #[must_use]
    pub fn autodiff(self) -> Self {
        if self.is_autodiff() {
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
    #[must_use]
    pub fn from_inner(inner: Tensor<D, K>) -> Self {
        inner.autodiff()
    }

    /// Returns this tensor's gradient-checkpointing strategy when autodiff is enabled.
    #[cfg(feature = "autodiff")]
    pub fn gradient_checkpointing_strategy(&self) -> Option<GradientCheckpointingStrategy> {
        match self.primitive.as_parts().1.autodiff {
            DispatchAutodiffContext::Disabled => None,
            DispatchAutodiffContext::Enabled(strategy) => Some(strategy),
        }
    }

    /// Sets the autodiff checkpointing strategy carried by this tensor.
    ///
    /// The strategy is normally derived from the device the tensor was created on (see
    /// [`Device::gradient_checkpointing`](crate::Device::gradient_checkpointing)); this
    /// method overrides it for a single tensor. Enable autodiff first with
    /// [`autodiff`](Tensor::autodiff) when needed.
    ///
    /// # Panics
    ///
    /// Panics if autodiff isn't enabled. Operations combining tensors that carry different
    /// strategies also panic; make sure all operands share the same one.
    #[cfg(feature = "autodiff")]
    #[must_use]
    pub fn with_gradient_checkpointing_strategy(
        self,
        strategy: GradientCheckpointingStrategy,
    ) -> Self {
        assert!(
            self.is_autodiff(),
            "Tensor::with_gradient_checkpointing_strategy requires autodiff; call Tensor::autodiff first"
        );
        let primitive = self.primitive;
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
