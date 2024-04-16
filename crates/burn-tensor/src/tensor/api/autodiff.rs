use crate::{backend::AutodiffBackend, BasicOps, Bool, Float, Int, Tensor, TensorKind};

impl<const D: usize, B: AutodiffBackend> Tensor<B, D> {
    /// Backward pass of the tensor.
    pub fn backward(&self) -> B::Gradients {
        B::backward::<D>(self.primitive.clone())
    }

    /// Get the gradients of a tensor if it exist.
    ///
    /// Returns a new reference to the same tensor. Therefore the same grad tensor can
    /// be accessed multiple times. If you only need to get the gradients one time,
    /// consider using [grad_remove](Tensor::grad_remove) for better performance.
    pub fn grad(&self, grads: &B::Gradients) -> Option<Tensor<B::InnerBackend, D>> {
        B::grad(&self.primitive, grads).map(Tensor::new)
    }

    /// Remove the grad tensor from the [grads](AutodiffBackend::Gradients) struct returning the result.
    pub fn grad_remove(&self, grads: &mut B::Gradients) -> Option<Tensor<B::InnerBackend, D>> {
        B::grad_remove(&self.primitive, grads).map(Tensor::new)
    }

    /// Replace the grad tensor from the [grads](AutodiffBackend::Gradients) struct with the provided
    /// gradient.
    pub fn grad_replace(&self, grads: &mut B::Gradients, grad: Tensor<B::InnerBackend, D>) {
        B::grad_replace(&self.primitive, grads, grad.primitive);
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

impl<B: AutodiffBackend> BasicAutodiffOps<B> for Float {
    type InnerKind = Float;

    fn inner<const D: usize>(
        tensor: <Self as TensorKind<B>>::Primitive<D>,
    ) -> <Self::InnerKind as TensorKind<<B as AutodiffBackend>::InnerBackend>>::Primitive<D> {
        B::inner(tensor)
    }

    fn from_inner<const D: usize>(
        inner: <Self::InnerKind as TensorKind<<B as AutodiffBackend>::InnerBackend>>::Primitive<D>,
    ) -> <Self as TensorKind<B>>::Primitive<D> {
        B::from_inner(inner)
    }
}

impl<B: AutodiffBackend> BasicAutodiffOps<B> for Int {
    type InnerKind = Int;

    fn inner<const D: usize>(
        tensor: <Self as TensorKind<B>>::Primitive<D>,
    ) -> <Self::InnerKind as TensorKind<<B as AutodiffBackend>::InnerBackend>>::Primitive<D> {
        B::int_inner(tensor)
    }

    fn from_inner<const D: usize>(
        inner: <Self::InnerKind as TensorKind<<B as AutodiffBackend>::InnerBackend>>::Primitive<D>,
    ) -> <Self as TensorKind<B>>::Primitive<D> {
        B::int_from_inner(inner)
    }
}

impl<B: AutodiffBackend> BasicAutodiffOps<B> for Bool {
    type InnerKind = Bool;

    fn inner<const D: usize>(
        tensor: <Self as TensorKind<B>>::Primitive<D>,
    ) -> <Self::InnerKind as TensorKind<<B as AutodiffBackend>::InnerBackend>>::Primitive<D> {
        B::bool_inner(tensor)
    }

    fn from_inner<const D: usize>(
        inner: <Self::InnerKind as TensorKind<<B as AutodiffBackend>::InnerBackend>>::Primitive<D>,
    ) -> <Self as TensorKind<B>>::Primitive<D> {
        B::bool_from_inner(inner)
    }
}

/// Trait that list all operations that can be applied on all tensors on an autodiff backend.
///
/// # Warnings
///
/// This is an internal trait, use the public API provided by [tensor struct](Tensor).
pub trait BasicAutodiffOps<B: AutodiffBackend>: BasicOps<B> + BasicOps<B::InnerBackend> {
    /// Inner primitive tensor.
    type InnerKind: BasicOps<B::InnerBackend>;

    /// Returns the inner tensor without the autodiff information.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// Users should prefer the [Tensor::inner](Tensor::inner) function,
    /// which is more high-level and designed for public use.
    fn inner<const D: usize>(
        tensor: <Self as TensorKind<B>>::Primitive<D>,
    ) -> <Self::InnerKind as TensorKind<B::InnerBackend>>::Primitive<D>;

    /// Convert a tensor to the autodiff backend.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// Users should prefer the [Tensor::from_inner](Tensor::from_inner) function,
    /// which is more high-level and designed for public use.
    fn from_inner<const D: usize>(
        inner: <Self::InnerKind as TensorKind<B::InnerBackend>>::Primitive<D>,
    ) -> <Self as TensorKind<B>>::Primitive<D>;
}
