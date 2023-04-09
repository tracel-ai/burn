use crate::{
    backend::Backend, check, check::TensorCheck, BasicOps, Bool, Element, ElementConversion, Float,
    Int, Shape, Tensor, TensorKind,
};

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
        check!(TensorCheck::binary_ops_ew("Add", &self, &other));
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
        check!(TensorCheck::binary_ops_ew("Sub", &self, &other));
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
        check!(TensorCheck::binary_ops_ew("Div", &self, &other));
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
        check!(TensorCheck::binary_ops_ew("Mul", &self, &other));
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
        check!(TensorCheck::aggregate_dim::<D>("Mean", dim));
        Self::new(K::mean_dim(self.primitive, dim))
    }

    /// Aggregate all elements along the given *dimension* or *axis* in the tensor with the sum operation.
    pub fn sum_dim(self, dim: usize) -> Self {
        check!(TensorCheck::aggregate_dim::<D>("Sum", dim));
        Self::new(K::sum_dim(self.primitive, dim))
    }

    /// Applies element wise greater comparison and returns a boolean tensor.
    ///
    /// # Panics
    ///
    /// If the two tensors don't have the same shape.
    pub fn greater(self, other: Self) -> Tensor<B, D, Bool> {
        check!(TensorCheck::binary_ops_ew("Greater", &self, &other));
        K::greater(self.primitive, other.primitive)
    }

    /// Applies element wise greater-equal comparison and returns a boolean tensor.
    ///
    /// # Panics
    ///
    /// If the two tensors don't have the same shape.
    pub fn greater_equal(self, other: Self) -> Tensor<B, D, Bool> {
        check!(TensorCheck::binary_ops_ew("Greater_equal", &self, &other));
        K::greater_equal(self.primitive, other.primitive)
    }

    /// Applies element wise lower comparison and returns a boolean tensor.
    ///
    /// # Panics
    ///
    /// If the two tensors don't have the same shape.
    pub fn lower(self, other: Self) -> Tensor<B, D, Bool> {
        check!(TensorCheck::binary_ops_ew("Lower", &self, &other));
        K::lower(self.primitive, other.primitive)
    }

    /// Applies element wise lower-equal comparison and returns a boolean tensor.
    ///
    /// # Panics
    ///
    /// If the two tensors don't have the same shape.
    pub fn lower_equal(self, other: Self) -> Tensor<B, D, Bool> {
        check!(TensorCheck::binary_ops_ew("Lower_equal", &self, &other));
        K::lower_equal(self.primitive, other.primitive)
    }

    /// Applies element wise greater comparison and returns a boolean tensor.
    pub fn greater_elem<E: ElementConversion>(self, other: E) -> Tensor<B, D, Bool> {
        K::greater_elem(self.primitive, other.elem())
    }

    /// Applies element wise greater-equal comparison and returns a boolean tensor.
    pub fn greater_equal_elem<E: ElementConversion>(self, other: E) -> Tensor<B, D, Bool> {
        K::greater_equal_elem(self.primitive, other.elem())
    }

    /// Applies element wise lower comparison and returns a boolean tensor.
    pub fn lower_elem<E: ElementConversion>(self, other: E) -> Tensor<B, D, Bool> {
        K::lower_elem(self.primitive, other.elem())
    }

    /// Applies element wise lower-equal comparison and returns a boolean tensor.
    pub fn lower_equal_elem<E: ElementConversion>(self, other: E) -> Tensor<B, D, Bool> {
        K::lower_equal_elem(self.primitive, other.elem())
    }

    /// Select the tensor elements corresponding to the given indexes.
    ///
    /// # Notes
    ///
    /// The index tensor shoud have the same shape as the original tensor except for the last
    /// dimension.
    pub fn index_select(self, indexes: Tensor<B, D, Int>) -> Self {
        Self::new(K::index_select(self.primitive, indexes))
    }

    /// Assign the selected elements corresponding to the given indexes from the value tensor
    /// to the original tensor using sum reduction.
    ///
    /// # Notes
    ///
    /// The index tensor shoud have the same shape as the original tensor except for the last
    /// dimension. The value and index tensors should have the same shape.
    pub fn index_select_assign(self, indexes: Tensor<B, D, Int>, values: Self) -> Self {
        Self::new(K::index_select_assign(
            self.primitive,
            indexes,
            values.primitive,
        ))
    }

    /// Select the tensor elements along the given dimension corresponding to the given indexes.
    pub fn index_select_dim(self, dim: usize, indexes: Tensor<B, 1, Int>) -> Self {
        Self::new(K::index_select_dim(self.primitive, dim, indexes))
    }

    /// Assign the selected elements along the given dimension corresponding to the given indexes
    /// from the value tensor to the original tensor using sum reduction.
    pub fn index_select_dim_assign<const D2: usize>(
        self,
        dim: usize,
        indexes: Tensor<B, 1, Int>,
        values: Tensor<B, D2, K>,
    ) -> Self {
        Self::new(K::index_select_dim_assign(
            self.primitive,
            dim,
            indexes,
            values.primitive,
        ))
    }
}

/// Trait that list all operations that can be applied on all numerical tensors.
///
/// # Warnings
///
/// This is an internal trait, use the public API provided by [tensor struct](Tensor).
pub trait Numeric<B: Backend>: BasicOps<B> {
    type NumElem: Element;

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
    fn greater<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, Bool>;
    fn greater_elem<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::NumElem,
    ) -> Tensor<B, D, Bool>;
    fn greater_equal<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, Bool>;
    fn greater_equal_elem<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::NumElem,
    ) -> Tensor<B, D, Bool>;
    fn lower<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, Bool>;
    fn lower_elem<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::NumElem,
    ) -> Tensor<B, D, Bool>;
    fn lower_equal<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, Bool>;
    fn lower_equal_elem<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::NumElem,
    ) -> Tensor<B, D, Bool>;
    fn index_select<const D: usize>(
        tensor: Self::Primitive<D>,
        indexes: Tensor<B, D, Int>,
    ) -> Self::Primitive<D>;
    fn index_select_assign<const D: usize>(
        tensor: Self::Primitive<D>,
        indexes: Tensor<B, D, Int>,
        values: Self::Primitive<D>,
    ) -> Self::Primitive<D>;
    fn index_select_dim<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
        indexes: Tensor<B, 1, Int>,
    ) -> Self::Primitive<D>;
    fn index_select_dim_assign<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        dim: usize,
        indexes: Tensor<B, 1, Int>,
        values: Self::Primitive<D2>,
    ) -> Self::Primitive<D1>;
}

impl<B: Backend> Numeric<B> for Int {
    type NumElem = B::IntElem;

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

    fn greater<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, Bool> {
        Tensor::new(B::int_greater(lhs, rhs))
    }

    fn greater_elem<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::NumElem,
    ) -> Tensor<B, D, Bool> {
        Tensor::new(B::int_greater_elem(lhs, rhs))
    }

    fn greater_equal<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, Bool> {
        Tensor::new(B::int_greater_equal(lhs, rhs))
    }

    fn greater_equal_elem<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::NumElem,
    ) -> Tensor<B, D, Bool> {
        Tensor::new(B::int_greater_equal_elem(lhs, rhs))
    }

    fn lower<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, Bool> {
        Tensor::new(B::int_lower(lhs, rhs))
    }

    fn lower_elem<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::NumElem,
    ) -> Tensor<B, D, Bool> {
        Tensor::new(B::int_lower_elem(lhs, rhs))
    }

    fn lower_equal<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, Bool> {
        Tensor::new(B::int_lower_equal(lhs, rhs))
    }

    fn lower_equal_elem<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::NumElem,
    ) -> Tensor<B, D, Bool> {
        Tensor::new(B::int_lower_equal_elem(lhs, rhs))
    }

    fn index_select_dim<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
        indexes: Tensor<B, 1, Int>,
    ) -> Self::Primitive<D> {
        B::int_index_select_dim(tensor, dim, indexes.primitive)
    }

    fn index_select_dim_assign<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        dim: usize,
        indexes: Tensor<B, 1, Int>,
        values: Self::Primitive<D2>,
    ) -> Self::Primitive<D1> {
        B::int_index_select_dim_assign(tensor, dim, indexes.primitive, values)
    }
    fn index_select<const D: usize>(
        tensor: Self::Primitive<D>,
        indexes: Tensor<B, D, Int>,
    ) -> Self::Primitive<D> {
        B::int_index_select(tensor, indexes.primitive)
    }

    fn index_select_assign<const D: usize>(
        tensor: Self::Primitive<D>,
        indexes: Tensor<B, D, Int>,
        values: Self::Primitive<D>,
    ) -> Self::Primitive<D> {
        B::int_index_select_assign(tensor, indexes.primitive, values)
    }
}

impl<B: Backend> Numeric<B> for Float {
    type NumElem = B::FloatElem;

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

    fn greater<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, Bool> {
        Tensor::new(B::greater(lhs, rhs))
    }

    fn greater_elem<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::NumElem,
    ) -> Tensor<B, D, Bool> {
        Tensor::new(B::greater_elem(lhs, rhs))
    }

    fn greater_equal<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, Bool> {
        Tensor::new(B::greater_equal(lhs, rhs))
    }

    fn greater_equal_elem<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::NumElem,
    ) -> Tensor<B, D, Bool> {
        Tensor::new(B::greater_equal_elem(lhs, rhs))
    }

    fn lower<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, Bool> {
        Tensor::new(B::lower(lhs, rhs))
    }

    fn lower_elem<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::NumElem,
    ) -> Tensor<B, D, Bool> {
        Tensor::new(B::lower_elem(lhs, rhs))
    }

    fn lower_equal<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, Bool> {
        Tensor::new(B::lower_equal(lhs, rhs))
    }

    fn lower_equal_elem<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::NumElem,
    ) -> Tensor<B, D, Bool> {
        Tensor::new(B::lower_equal_elem(lhs, rhs))
    }

    fn index_select_dim<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
        indexes: Tensor<B, 1, Int>,
    ) -> Self::Primitive<D> {
        B::index_select_dim(tensor, dim, indexes.primitive)
    }

    fn index_select_dim_assign<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        dim: usize,
        indexes: Tensor<B, 1, Int>,
        values: Self::Primitive<D2>,
    ) -> Self::Primitive<D1> {
        B::index_select_dim_assign(tensor, dim, indexes.primitive, values)
    }

    fn index_select<const D: usize>(
        tensor: Self::Primitive<D>,
        indexes: Tensor<B, D, Int>,
    ) -> Self::Primitive<D> {
        B::index_select(tensor, indexes.primitive)
    }

    fn index_select_assign<const D: usize>(
        tensor: Self::Primitive<D>,
        indexes: Tensor<B, D, Int>,
        values: Self::Primitive<D>,
    ) -> Self::Primitive<D> {
        B::index_select_assign(tensor, indexes.primitive, values)
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
