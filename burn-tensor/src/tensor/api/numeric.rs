use crate::{backend::Backend, ElementConversion, Float, Int, Shape, Tensor, TensorKind};

impl<B, const D: usize, K> Tensor<B, D, K>
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

    /// Applies element wise addition operation with a scalar.
    ///
    /// `y = x + s`
    pub fn add_scalar<E: ElementConversion>(self, other: E) -> Self {
        Self::new(K::add_scalar(self.primitive, other))
    }

    /// Applies element wise substraction operation.
    ///
    /// `y = x2 - x1`
    #[allow(clippy::should_implement_trait)]
    pub fn sub(self, other: Self) -> Self {
        Self::new(K::sub(self.primitive, other.primitive))
    }

    /// Applies element wise substraction operation with a scalar.
    ///
    /// `y = x - s`
    pub fn sub_scalar<E: ElementConversion>(self, other: E) -> Self {
        Self::new(K::sub_scalar(self.primitive, other))
    }

    /// Applies element wise division operation.
    ///
    /// `y = x2 / x1`
    #[allow(clippy::should_implement_trait)]
    pub fn div(self, other: Self) -> Self {
        Self::new(K::div(self.primitive, other.primitive))
    }

    /// Applies element wise division operation with a scalar.
    ///
    /// `y = x / s`
    pub fn div_scalar<E: ElementConversion>(self, other: E) -> Self {
        Self::new(K::div_scalar(self.primitive, other))
    }
    ///
    /// Applies element wise multiplication operation.
    ///
    /// `y = x2 * x1`
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, other: Self) -> Self {
        Self::new(K::mul(self.primitive, other.primitive))
    }

    /// Applies element wise multiplication operation with a scalar.
    ///
    /// `y = x * s`
    pub fn mul_scalar<E: ElementConversion>(self, other: E) -> Self {
        Self::new(K::mul_scalar(self.primitive, other))
    }

    /// Switch sign of each element in the tensor.
    ///
    /// `y = -x`
    #[allow(clippy::should_implement_trait)]
    pub fn neg(self) -> Self {
        Self::new(K::neg(self.primitive))
    }

    /// Create a tensor of the given shape where each element is zero.
    pub fn zeros<S: Into<Shape<D>>>(shape: S) -> Self {
        Self::zeros_device(shape, &B::Device::default())
    }

    /// Create a tensor of the given shape where each element is zero.
    pub fn zeros_device<S: Into<Shape<D>>>(shape: S, device: &B::Device) -> Self {
        Self::new(K::zeros(shape.into(), device))
    }

    /// Create a tensor of the given shape where each element is one.
    pub fn ones<S: Into<Shape<D>>>(shape: S) -> Self {
        Self::ones_device(shape, &B::Device::default())
    }

    /// Create a tensor of the given shape where each element is zero.
    pub fn ones_device<S: Into<Shape<D>>>(shape: S, device: &B::Device) -> Self {
        Self::new(K::ones(shape.into(), device))
    }

    /// Aggregate all elements in the tensor with the mean operation.
    pub fn mean(self) -> Tensor<B, 1, K> {
        Tensor::new(K::mean(self.primitive))
    }

    /// Aggregate all elements in the tensor with the sum operation.
    pub fn sum(self) -> Tensor<B, 1, K> {
        Tensor::new(K::sum(self.primitive))
    }

    /// Aggregate all elements along the given *dimension* or *axis* in the tensor with the mean operation.
    pub fn mean_dim(self, dim: usize) -> Self {
        Self::new(K::mean_dim(self.primitive, dim))
    }

    /// Aggregate all elements along the given *dimension* or *axis* in the tensor with the sum operation.
    pub fn sum_dim(self, dim: usize) -> Self {
        Self::new(K::sum_dim(self.primitive, dim))
    }
}

/// Trait that list all operations that can be applied on all numerical tensors.
///
/// # Warnings
///
/// This is an internal trait, use the public API provided by [tensor struct](Tensor).
pub trait Numeric<B: Backend>: TensorKind<B> {
    fn add<const D: usize>(lhs: Self::Primitive<D>, rhs: Self::Primitive<D>) -> Self::Primitive<D>;
    fn add_scalar<const D: usize, E: ElementConversion>(
        lhs: Self::Primitive<D>,
        rhs: E,
    ) -> Self::Primitive<D>;
    fn sub<const D: usize>(lhs: Self::Primitive<D>, rhs: Self::Primitive<D>) -> Self::Primitive<D>;
    fn sub_scalar<const D: usize, E: ElementConversion>(
        lhs: Self::Primitive<D>,
        rhs: E,
    ) -> Self::Primitive<D>;
    fn div<const D: usize>(lhs: Self::Primitive<D>, rhs: Self::Primitive<D>) -> Self::Primitive<D>;
    fn div_scalar<const D: usize, E: ElementConversion>(
        lhs: Self::Primitive<D>,
        rhs: E,
    ) -> Self::Primitive<D>;
    fn mul<const D: usize>(lhs: Self::Primitive<D>, rhs: Self::Primitive<D>) -> Self::Primitive<D>;
    fn mul_scalar<const D: usize, E: ElementConversion>(
        lhs: Self::Primitive<D>,
        rhs: E,
    ) -> Self::Primitive<D>;
    fn neg<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<D>;
    fn zeros<const D: usize>(shape: Shape<D>, device: &B::Device) -> Self::Primitive<D>;
    fn ones<const D: usize>(shape: Shape<D>, device: &B::Device) -> Self::Primitive<D>;
    fn sum<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<1>;
    fn sum_dim<const D: usize>(tensor: Self::Primitive<D>, dim: usize) -> Self::Primitive<D>;
    fn mean<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<1>;
    fn mean_dim<const D: usize>(tensor: Self::Primitive<D>, dim: usize) -> Self::Primitive<D>;
}

impl<B: Backend> Numeric<B> for Int {
    fn add<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> <Int as TensorKind<B>>::Primitive<D> {
        B::int_add(lhs, rhs)
    }
    fn add_scalar<const D: usize, E: ElementConversion>(
        lhs: Self::Primitive<D>,
        rhs: E,
    ) -> Self::Primitive<D> {
        B::int_add_scalar(lhs, rhs.elem())
    }
    fn sub<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> <Int as TensorKind<B>>::Primitive<D> {
        B::int_sub(lhs, rhs)
    }
    fn sub_scalar<const D: usize, E: ElementConversion>(
        lhs: Self::Primitive<D>,
        rhs: E,
    ) -> Self::Primitive<D> {
        B::int_sub_scalar(lhs, rhs.elem())
    }
    fn div<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> <Int as TensorKind<B>>::Primitive<D> {
        B::int_div(lhs, rhs)
    }
    fn div_scalar<const D: usize, E: ElementConversion>(
        lhs: Self::Primitive<D>,
        rhs: E,
    ) -> Self::Primitive<D> {
        B::int_div_scalar(lhs, rhs.elem())
    }
    fn mul<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> <Int as TensorKind<B>>::Primitive<D> {
        B::int_mul(lhs, rhs)
    }
    fn mul_scalar<const D: usize, E: ElementConversion>(
        lhs: Self::Primitive<D>,
        rhs: E,
    ) -> Self::Primitive<D> {
        B::int_mul_scalar(lhs, rhs.elem())
    }
    fn neg<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<D> {
        B::int_neg(tensor)
    }
    fn zeros<const D: usize>(shape: Shape<D>, device: &B::Device) -> Self::Primitive<D> {
        B::int_zeros(shape, device)
    }
    fn ones<const D: usize>(shape: Shape<D>, device: &B::Device) -> Self::Primitive<D> {
        B::int_ones(shape, device)
    }
    fn sum<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<1> {
        B::int_sum(tensor)
    }
    fn sum_dim<const D: usize>(tensor: Self::Primitive<D>, dim: usize) -> Self::Primitive<D> {
        B::int_sum_dim(tensor, dim)
    }
    fn mean<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<1> {
        B::int_mean(tensor)
    }
    fn mean_dim<const D: usize>(tensor: Self::Primitive<D>, dim: usize) -> Self::Primitive<D> {
        B::int_mean_dim(tensor, dim)
    }
}

impl<B: Backend> Numeric<B> for Float {
    fn add<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> <Float as TensorKind<B>>::Primitive<D> {
        B::add(lhs, rhs)
    }
    fn add_scalar<const D: usize, E: ElementConversion>(
        lhs: Self::Primitive<D>,
        rhs: E,
    ) -> Self::Primitive<D> {
        B::add_scalar(lhs, rhs.elem())
    }
    fn sub<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> <Float as TensorKind<B>>::Primitive<D> {
        B::sub(lhs, rhs)
    }
    fn sub_scalar<const D: usize, E: ElementConversion>(
        lhs: Self::Primitive<D>,
        rhs: E,
    ) -> Self::Primitive<D> {
        B::sub_scalar(lhs, rhs.elem())
    }
    fn div<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> <Float as TensorKind<B>>::Primitive<D> {
        B::div(lhs, rhs)
    }
    fn div_scalar<const D: usize, E: ElementConversion>(
        lhs: Self::Primitive<D>,
        rhs: E,
    ) -> Self::Primitive<D> {
        B::div_scalar(lhs, rhs.elem())
    }
    fn mul<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> <Float as TensorKind<B>>::Primitive<D> {
        B::mul(lhs, rhs)
    }
    fn mul_scalar<const D: usize, E: ElementConversion>(
        lhs: Self::Primitive<D>,
        rhs: E,
    ) -> Self::Primitive<D> {
        B::mul_scalar(lhs, rhs.elem())
    }
    fn neg<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<D> {
        B::neg(tensor)
    }
    fn zeros<const D: usize>(shape: Shape<D>, device: &B::Device) -> Self::Primitive<D> {
        B::zeros(shape, device)
    }
    fn ones<const D: usize>(shape: Shape<D>, device: &B::Device) -> Self::Primitive<D> {
        B::ones(shape, device)
    }
    fn sum<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<1> {
        B::sum(tensor)
    }
    fn sum_dim<const D: usize>(tensor: Self::Primitive<D>, dim: usize) -> Self::Primitive<D> {
        B::sum_dim(tensor, dim)
    }
    fn mean<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<1> {
        B::mean(tensor)
    }
    fn mean_dim<const D: usize>(tensor: Self::Primitive<D>, dim: usize) -> Self::Primitive<D> {
        B::mean_dim(tensor, dim)
    }
}

impl<B, const D: usize, K> core::ops::Add<Self> for Tensor<B, D, K>
where
    B: Backend,
    K: Numeric<B>,
{
    type Output = Self;

    fn add(self, rhs: Tensor<B, D, K>) -> Self {
        Self::add(self, rhs)
    }
}

impl<E, const D: usize, B, K> core::ops::Add<E> for Tensor<B, D, K>
where
    E: ElementConversion,
    B: Backend,
    K: Numeric<B>,
{
    type Output = Self;

    fn add(self, other: E) -> Self {
        Tensor::add_scalar(self, other)
    }
}

impl<B, const D: usize, K> core::ops::Sub<Tensor<B, D, K>> for Tensor<B, D, K>
where
    B: Backend,
    K: Numeric<B>,
{
    type Output = Self;

    fn sub(self, rhs: Tensor<B, D, K>) -> Self {
        Tensor::sub(self, rhs)
    }
}

impl<E, const D: usize, B, K> core::ops::Sub<E> for Tensor<B, D, K>
where
    E: ElementConversion,
    B: Backend,
    K: Numeric<B>,
{
    type Output = Self;

    fn sub(self, other: E) -> Self {
        Tensor::sub_scalar(self, other)
    }
}

impl<B, const D: usize, K> core::ops::Div<Tensor<B, D, K>> for Tensor<B, D, K>
where
    B: Backend,
    K: Numeric<B>,
{
    type Output = Self;

    fn div(self, rhs: Tensor<B, D, K>) -> Self {
        Tensor::div(self, rhs)
    }
}

impl<E, const D: usize, B, K> core::ops::Div<E> for Tensor<B, D, K>
where
    E: ElementConversion,
    B: Backend,
    K: Numeric<B>,
{
    type Output = Self;

    fn div(self, other: E) -> Self {
        Tensor::div_scalar(self, other)
    }
}

impl<B, const D: usize, K> core::ops::Mul<Tensor<B, D, K>> for Tensor<B, D, K>
where
    B: Backend,
    K: Numeric<B>,
{
    type Output = Self;

    fn mul(self, rhs: Tensor<B, D, K>) -> Self {
        Tensor::mul(self, rhs)
    }
}

impl<E, const D: usize, B, K> core::ops::Mul<E> for Tensor<B, D, K>
where
    E: ElementConversion,
    B: Backend,
    K: Numeric<B>,
{
    type Output = Self;

    fn mul(self, other: E) -> Self {
        Tensor::mul_scalar(self, other)
    }
}

impl<B, const D: usize, K> core::ops::Neg for Tensor<B, D, K>
where
    B: Backend,
    K: Numeric<B>,
{
    type Output = Self;

    fn neg(self) -> Self {
        Tensor::neg(self)
    }
}
