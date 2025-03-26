use crate::{
    BasicOps, Bool, Float, Int, Tensor, TensorKind, TensorPrimitive, backend::AutodiffBackend,
};

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

impl<B: AutodiffBackend> BasicAutodiffOps<B> for Float {
    type InnerKind = Float;

    fn inner(
        tensor: <Self as TensorKind<B>>::Primitive,
    ) -> <Self::InnerKind as TensorKind<<B as AutodiffBackend>::InnerBackend>>::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => TensorPrimitive::Float(B::inner(tensor)),
            TensorPrimitive::QFloat(tensor) => TensorPrimitive::QFloat(B::q_inner(tensor)),
        }
    }

    fn from_inner(
        inner: <Self::InnerKind as TensorKind<<B as AutodiffBackend>::InnerBackend>>::Primitive,
    ) -> <Self as TensorKind<B>>::Primitive {
        match inner {
            TensorPrimitive::Float(tensor) => TensorPrimitive::Float(B::from_inner(tensor)),
            TensorPrimitive::QFloat(tensor) => TensorPrimitive::QFloat(B::q_from_inner(tensor)),
        }
    }
}

impl<B: AutodiffBackend> BasicAutodiffOps<B> for Int {
    type InnerKind = Int;

    fn inner(
        tensor: <Self as TensorKind<B>>::Primitive,
    ) -> <Self::InnerKind as TensorKind<<B as AutodiffBackend>::InnerBackend>>::Primitive {
        B::int_inner(tensor)
    }

    fn from_inner(
        inner: <Self::InnerKind as TensorKind<<B as AutodiffBackend>::InnerBackend>>::Primitive,
    ) -> <Self as TensorKind<B>>::Primitive {
        B::int_from_inner(inner)
    }
}

impl<B: AutodiffBackend> BasicAutodiffOps<B> for Bool {
    type InnerKind = Bool;

    fn inner(
        tensor: <Self as TensorKind<B>>::Primitive,
    ) -> <Self::InnerKind as TensorKind<<B as AutodiffBackend>::InnerBackend>>::Primitive {
        B::bool_inner(tensor)
    }

    fn from_inner(
        inner: <Self::InnerKind as TensorKind<<B as AutodiffBackend>::InnerBackend>>::Primitive,
    ) -> <Self as TensorKind<B>>::Primitive {
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
    fn inner(
        tensor: <Self as TensorKind<B>>::Primitive,
    ) -> <Self::InnerKind as TensorKind<B::InnerBackend>>::Primitive;

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
    fn from_inner(
        inner: <Self::InnerKind as TensorKind<B::InnerBackend>>::Primitive,
    ) -> <Self as TensorKind<B>>::Primitive;
}
