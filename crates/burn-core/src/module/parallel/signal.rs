use alloc::vec::Vec;

use burn_tensor::{Bool, Gradients, Int, Tensor};

pub use burn_derive::BackwardSignal;

/// A signal passed between the layers of a split model, which pipeline training runs backward
/// through one stage at a time.
///
/// `#[derive(BackwardSignal)]` implements it for a struct by calling it on each field; a field
/// that is neither a tensor nor a signal takes `#[backward_signal(skip)]` and is cloned through.
pub trait BackwardSignal: Sized {
    /// `detach().require_grad()` on each float tensor: the stage receiving it starts a graph of
    /// its own.
    fn detach_require_grad(self) -> Self;

    /// The gradient of each float tensor, in the same shape: what the stage sends back.
    fn grad(&self, grads: &Gradients) -> Self;

    /// `(tensor * grad).sum()` added up over its float tensors, to run the backward of the stage
    /// that produced `self`; `None` when it holds no float tensor.
    fn grad_product(self, grad: Self) -> Option<Tensor<1>>;
}

impl<const D: usize> BackwardSignal for Tensor<D> {
    fn detach_require_grad(self) -> Self {
        self.detach().require_grad()
    }

    /// Zeros when the stage never read the tensor, so nothing flows back through it.
    fn grad(&self, grads: &Gradients) -> Self {
        Tensor::grad(self, grads).unwrap_or_else(|| self.zeros_like())
    }

    fn grad_product(self, grad: Self) -> Option<Tensor<1>> {
        Some((self * grad).sum())
    }
}

impl<const D: usize> BackwardSignal for Tensor<D, Bool> {
    fn detach_require_grad(self) -> Self {
        self
    }

    fn grad(&self, _grads: &Gradients) -> Self {
        self.clone()
    }

    fn grad_product(self, _grad: Self) -> Option<Tensor<1>> {
        None
    }
}

impl<const D: usize> BackwardSignal for Tensor<D, Int> {
    fn detach_require_grad(self) -> Self {
        self
    }

    fn grad(&self, _grads: &Gradients) -> Self {
        self.clone()
    }

    fn grad_product(self, _grad: Self) -> Option<Tensor<1>> {
        None
    }
}

impl<T: BackwardSignal> BackwardSignal for Option<T> {
    fn detach_require_grad(self) -> Self {
        self.map(T::detach_require_grad)
    }

    fn grad(&self, grads: &Gradients) -> Self {
        self.as_ref().map(|value| value.grad(grads))
    }

    fn grad_product(self, grad: Self) -> Option<Tensor<1>> {
        self.zip(grad)
            .and_then(|(value, grad)| value.grad_product(grad))
    }
}

impl<T: BackwardSignal> BackwardSignal for Vec<T> {
    fn detach_require_grad(self) -> Self {
        self.into_iter().map(T::detach_require_grad).collect()
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

/// The sum of the products that exist, or `None` when none does: what a derived signal returns.
#[doc(hidden)]
pub fn sum_products(products: impl IntoIterator<Item = Option<Tensor<1>>>) -> Option<Tensor<1>> {
    products
        .into_iter()
        .flatten()
        .reduce(|sum, product| sum + product)
}
