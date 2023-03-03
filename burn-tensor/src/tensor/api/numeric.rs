use crate::{backend::Backend, ops::TensorOps, Float, Int, TensorKind, TensorNew};

impl<B, const D: usize, K> TensorNew<B, D, K>
where
    B: Backend,
    K: Numeric<B>,
{
    /// Applies element wise addition operation.
    ///
    /// `y = x2 + x1`
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, other: Self) -> Self {
        Self::new(K::add(self.primitive, other.primitive))
    }
}

/// Trait that list all operations that can be applied on all numerical tensors.
///
/// # Warnings
///
/// This is an internal trait, use the public API provided by [tensor struct](TensorNew).
pub trait Numeric<B: Backend>: TensorKind<B> {
    fn add<const D: usize>(lhs: Self::Primitive<D>, rhs: Self::Primitive<D>) -> Self::Primitive<D>;
}

impl<B: Backend> Numeric<B> for Int {
    fn add<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> <Int as TensorKind<B>>::Primitive<D> {
        B::IntegerBackend::add(lhs, rhs)
    }
}

impl<B: Backend> Numeric<B> for Float {
    fn add<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> <Float as TensorKind<B>>::Primitive<D> {
        B::add(lhs, rhs)
    }
}

impl<B, const D: usize, K> core::ops::Add<TensorNew<B, D, K>> for TensorNew<B, D, K>
where
    B: Backend,
    K: Numeric<B>,
{
    type Output = Self;

    fn add(self, rhs: TensorNew<B, D, K>) -> Self::Output {
        Self::add(self, rhs)
    }
}
