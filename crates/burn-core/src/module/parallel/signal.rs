use alloc::vec::Vec;

use burn_tensor::{Bool, Device, FloatDType, Gradients, Int, Tensor, kind::Autodiff};

pub use burn_derive::AutodiffSignal;

/// A signal passed between the layers of a split model, which pipeline training runs backward
/// through one stage at a time.
///
/// `#[derive(AutodiffSignal)]` implements it for a struct by calling it on each field; a field
/// that is neither a tensor nor a signal takes `#[autodiff_signal(skip)]` and is cloned through.
pub trait AutodiffSignal: Sized {
    /// `detach().require_grad()` on each float tensor, under the gradient-checkpointing strategy
    /// of `device`, where the stage receiving it starts a graph of its own.
    fn detach_require_grad(self, device: &Device) -> Self;

    /// The gradient of each float tensor, in the same shape: what the stage sends back.
    fn grad(&self, grads: &Gradients) -> Self;

    /// `(tensor * grad).sum()` added up over its float tensors, to run the backward of the stage
    /// that produced `self`; `None` when it holds no float tensor.
    fn grad_product(self, grad: Self) -> Option<Tensor<1>>;
}

impl<const D: usize> AutodiffSignal for Tensor<D> {
    /// A tensor made without autodiff, such as an attention bias, keeps it off when moved onto an
    /// autodiff device, so it is turned on here.
    fn detach_require_grad(self, device: &Device) -> Self {
        with_strategy_of(self.detach().autodiff(), device).require_grad()
    }

    /// Zeros when the stage never read the tensor, so nothing flows back through it. They carry no
    /// autodiff, as a gradient does, so they combine with the sender's tensors whatever its
    /// strategy.
    fn grad(&self, grads: &Gradients) -> Self {
        Tensor::grad(self, grads).unwrap_or_else(|| self.zeros_like().without_autodiff())
    }

    fn grad_product(self, grad: Self) -> Option<Tensor<1>> {
        Some((self * grad).sum())
    }
}

impl<const D: usize> AutodiffSignal for Tensor<D, Bool> {
    fn detach_require_grad(self, device: &Device) -> Self {
        with_strategy_of(self, device)
    }

    fn grad(&self, _grads: &Gradients) -> Self {
        self.clone()
    }

    fn grad_product(self, _grad: Self) -> Option<Tensor<1>> {
        None
    }
}

impl<const D: usize> AutodiffSignal for Tensor<D, Int> {
    fn detach_require_grad(self, device: &Device) -> Self {
        with_strategy_of(self, device)
    }

    fn grad(&self, _grads: &Gradients) -> Self {
        self.clone()
    }

    fn grad_product(self, _grad: Self) -> Option<Tensor<1>> {
        None
    }
}

impl<T: AutodiffSignal> AutodiffSignal for Option<T> {
    fn detach_require_grad(self, device: &Device) -> Self {
        self.map(|value| value.detach_require_grad(device))
    }

    fn grad(&self, grads: &Gradients) -> Self {
        self.as_ref().map(|value| value.grad(grads))
    }

    fn grad_product(self, grad: Self) -> Option<Tensor<1>> {
        self.zip(grad)
            .and_then(|(value, grad)| value.grad_product(grad))
    }
}

impl<T: AutodiffSignal> AutodiffSignal for Vec<T> {
    fn detach_require_grad(self, device: &Device) -> Self {
        self.into_iter()
            .map(|value| value.detach_require_grad(device))
            .collect()
    }

    fn grad(&self, grads: &Gradients) -> Self {
        self.iter().map(|value| value.grad(grads)).collect()
    }

    fn grad_product(self, grad: Self) -> Option<Tensor<1>> {
        sum_products(
            self.into_iter()
                .zip(grad)
                .map(|(value, grad)| value.grad_product(grad)),
        )
    }
}

/// `tensor` under `device`'s gradient-checkpointing strategy when it uses autodiff. Tensors of two
/// strategies cannot be combined, and `to_device` keeps the sender's.
fn with_strategy_of<const D: usize, K: Autodiff>(
    tensor: Tensor<D, K>,
    device: &Device,
) -> Tensor<D, K> {
    match device.gradient_checkpointing_strategy() {
        Some(strategy) if tensor.is_autodiff() => {
            tensor.with_gradient_checkpointing_strategy(strategy)
        }
        _ => tensor,
    }
}

/// The sum of the products that exist, or `None` when none does: what a derived signal returns.
///
/// Each is cast to f32 first, so fields of different float types add up. The sum only seeds the
/// backward, which runs from a gradient of one whatever its type.
#[doc(hidden)]
pub fn sum_products(products: impl IntoIterator<Item = Option<Tensor<1>>>) -> Option<Tensor<1>> {
    products
        .into_iter()
        .flatten()
        .map(|product| product.cast(FloatDType::F32))
        .reduce(|sum, product| sum + product)
}
