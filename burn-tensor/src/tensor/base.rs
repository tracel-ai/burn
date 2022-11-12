use crate::graph::grad::Gradients;
use crate::tensor::backend::ADBackend;
use crate::tensor::backend::Backend;
use crate::tensor::ops::activation::*;
use crate::tensor::ops::*;
use crate::tensor::stats;
use crate::tensor::ElementConversion;
use crate::tensor::{Data, Distribution, Shape};
use crate::BoolTensor;
use std::convert::TryInto;
use std::ops::Range;

/// A tensor or a *n-dimensional* array.
#[derive(Debug, Clone)]
pub struct Tensor<B: Backend, const D: usize> {
    pub(crate) value: B::TensorPrimitive<D>,
}

impl<B> Tensor<B, 1>
where
    B: Backend,
{
    /// Returns a new integer tensor on the default device which values are generated from the given range.
    pub fn arange(range: Range<usize>) -> Tensor<B::IntegerBackend, 1> {
        Tensor::new(B::arange(range, B::Device::default()))
    }
    /// Returns a new integer tensor on the specified device which values are generated from the given range.
    pub fn arange_device(range: Range<usize>, device: B::Device) -> Tensor<B::IntegerBackend, 1> {
        Tensor::new(B::arange(range, device))
    }
}

impl<const D: usize, B> Tensor<B, D>
where
    B: Backend,
{
    pub(crate) fn new(tensor: B::TensorPrimitive<D>) -> Self {
        Self { value: tensor }
    }

    /// Reshape the tensor to have the given shape.
    ///
    /// # Panics
    ///
    /// If the tensor can not be reshape to the given shape.
    pub fn reshape<const D2: usize, S: Into<Shape<D2>>>(&self, shape: S) -> Tensor<B, D2> {
        Tensor::new(B::reshape(&self.value, shape.into()))
    }

    /// Returns a new tensor on the given device.
    pub fn to_device(&self, device: B::Device) -> Self {
        Self::new(B::to_device(&self.value, device))
    }

    /// Returns the device of the current tensor.
    pub fn device(&self) -> B::Device {
        B::device(&self.value)
    }

    /// Applies element wise exponential operation.
    ///
    /// `y = e^x`
    pub fn exp(&self) -> Self {
        Self::new(B::exp(&self.value))
    }

    /// Applies element wise natural log operation *ln*.
    ///
    /// `y = log(x)`
    pub fn log(&self) -> Self {
        Self::new(B::log(&self.value))
    }

    /// Applies the [error function](https://en.wikipedia.org/wiki/Error_function) element wise.
    ///
    /// `y = erf(x)`
    pub fn erf(&self) -> Self {
        Self::new(self.value.erf())
    }

    /// Applies element wise power operation.
    ///
    /// `y = x^a`
    pub fn powf(&self, value: f32) -> Self {
        Self::new(B::powf(&self.value, value))
    }

    /// Returns the shape of the current tensor.
    pub fn shape(&self) -> &Shape<D> {
        B::shape(&self.value)
    }

    /// Returns the dimensions of the current tensor.
    ///
    /// Equivalent to `tensor.shape().dims`.
    pub fn dims(&self) -> &[usize; D] {
        &B::shape(&self.value).dims
    }

    /// Returns the data of the current tensor.
    pub fn into_data(self) -> Data<B::Elem, D> {
        B::into_data(self.value)
    }

    /// Returns the data of the current tensor without taking ownership.
    pub fn to_data(&self) -> Data<B::Elem, D> {
        B::to_data(&self.value)
    }

    /// Create a tensor from the given data.
    pub fn from_data(data: Data<B::Elem, D>) -> Self {
        let tensor = B::from_data(data, B::Device::default());
        Tensor::new(tensor)
    }

    /// Create a tensor from the given data on the given device.
    pub fn from_data_device(data: Data<B::Elem, D>, device: B::Device) -> Self {
        let tensor = B::from_data(data, device);
        Tensor::new(tensor)
    }

    /// Returns a new tensor with the same shape and device as the current tensor filled with zeros.
    pub fn zeros_like(&self) -> Self {
        Tensor::new(B::zeros(*self.shape(), self.device()))
    }

    /// Returns a new tensor with the same shape and device as the current tensor filled with ones.
    pub fn ones_like(&self) -> Self {
        Tensor::new(B::ones(*self.shape(), self.device()))
    }

    /// Returns a new tensor with the same shape and device as the current tensor filled random
    /// values sampled from the given distribution.
    pub fn random_like(&self, distribution: Distribution<B::Elem>) -> Self {
        Tensor::new(B::random(*self.shape(), distribution, self.device()))
    }

    /// Create a one hot tensor.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///     let one_hot = Tensor::<B, 1>::one_hot(2, 10);
    ///     println!("{}", one_hot.to_data());
    ///     // [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    /// }
    /// ```
    pub fn one_hot(index: usize, num_classes: usize) -> Self {
        let mut dims = [1; D];
        dims[D - 1] = num_classes;
        let shape = Shape::new(dims);
        let tensor = Tensor::zeros(shape);
        let ranges: Vec<_> = shape.dims.iter().map(|dim| 0..*dim).collect();
        let mut ranges: [std::ops::Range<usize>; D] = ranges.try_into().unwrap();
        ranges[D - 1] = index..index + 1;

        tensor.index_assign(ranges, &Tensor::ones(Shape::new([1; D])))
    }

    /// Applies element wise addition operation.
    ///
    /// `y = x2 + x1`
    pub fn add(&self, other: &Self) -> Self {
        Self::new(B::add(&self.value, &other.value))
    }

    /// Applies element wise addition operation with a scalar.
    ///
    /// `y = x + s`
    pub fn add_scalar<E: ElementConversion>(&self, other: E) -> Self {
        Self::new(B::add_scalar(&self.value, &other.to_elem()))
    }

    /// Applies element wise substraction operation.
    ///
    /// `y = x2 - x1`
    pub fn sub(&self, other: &Self) -> Self {
        Self::new(B::sub(&self.value, &other.value))
    }

    /// Applies element wise substraction operation with a scalar.
    ///
    /// `y = x - s`
    pub fn sub_scalar<E: ElementConversion>(&self, other: E) -> Self {
        Self::new(B::sub_scalar(&self.value, &other.to_elem()))
    }

    /// Applies the transpose operation.
    ///
    /// On matrix and higher dimension tensor, it swap the last two dimensions.
    ///
    /// # Panics
    ///
    /// If the tensor is of 1 dimension or less.
    pub fn transpose(&self) -> Self {
        Self::new(B::transpose(&self.value))
    }

    /// Swap two dimensions.
    ///
    /// # Panics
    ///
    /// If the dimensions exceed the shape of than the tensor.
    pub fn swap_dims(&self, dim1: usize, dim2: usize) -> Self {
        Self::new(B::swap_dims(&self.value, dim1, dim2))
    }

    /// Applies the matrix multiplication operation.
    ///
    /// `C = AB`
    ///
    /// # Panics
    ///
    /// If the two tensors dont' have a compatible shape.
    pub fn matmul(&self, other: &Self) -> Self {
        Self::new(B::matmul(&self.value, &other.value))
    }

    /// Switch sign of each element in the tensor.
    ///
    /// `y = -x`
    pub fn neg(&self) -> Self {
        Self::new(B::neg(&self.value))
    }

    /// Applies element wise multiplication operation.
    ///
    /// `y = x2 * x1`
    pub fn mul(&self, other: &Self) -> Self {
        Self::new(B::mul(&self.value, &other.value))
    }

    /// Applies element wise multiplication operation with scalar.
    ///
    /// `y = x2 * x1`
    pub fn mul_scalar<E: ElementConversion>(&self, other: E) -> Self {
        Self::new(B::mul_scalar(&self.value, &other.to_elem()))
    }

    /// Applies element wise division operation.
    ///
    /// `y = x2 / x1`
    pub fn div(&self, other: &Self) -> Self {
        Self::new(B::div(&self.value, &other.value))
    }

    /// Applies element wise division operation with scalar.
    ///
    /// `y = x2 / x1`
    pub fn div_scalar<E: ElementConversion>(&self, other: E) -> Self {
        Self::new(B::div_scalar(&self.value, &other.to_elem()))
    }

    /// Aggregate all elements in the tensor with the mean operation.
    pub fn mean(&self) -> Tensor<B, 1> {
        Tensor::new(B::mean(&self.value))
    }

    /// Aggregate all elements in the tensor with the sum operation.
    pub fn sum(&self) -> Tensor<B, 1> {
        Tensor::new(B::sum(&self.value))
    }

    /// Aggregate all elements along the given *dimension* or *axis* in the tensor with the mean operation.
    pub fn mean_dim(&self, dim: usize) -> Self {
        Self::new(B::mean_dim(&self.value, dim))
    }

    /// Aggregate all elements along the given *dimension* or *axis* in the tensor with the sum operation.
    pub fn sum_dim(&self, dim: usize) -> Self {
        Self::new(B::sum_dim(&self.value, dim))
    }

    /// Calculate the variance along the given dimension.
    pub fn var(&self, dim: usize) -> Self {
        stats::var(self, dim)
    }

    /// Calculate the variance along the given dimension without applying the Bessel’s correction.
    pub fn var_bias(&self, dim: usize) -> Self {
        stats::var_bias(self, dim)
    }

    /// Calculate the variance along the given dimension and also returns the mean.
    pub fn var_mean(&self, dim: usize) -> (Self, Self) {
        let mean = self.mean_dim(dim);
        let var = stats::var_with_mean(self, &mean, dim);
        (var, mean)
    }

    /// Calculate the variance along the given dimension without applying the Bessel’s correction and also returns the mean.
    pub fn var_mean_bias(&self, dim: usize) -> (Self, Self) {
        let mean = self.mean_dim(dim);
        let var = stats::var_with_mean_bias(self, &mean, dim);
        (var, mean)
    }

    /// Applies element wise equal comparison and returns a boolean tensor.
    ///
    /// # Panics
    ///
    /// If the two tensors don't have the same shape.
    pub fn equal(&self, other: &Self) -> BoolTensor<B, D> {
        BoolTensor::new(B::equal(&self.value, &other.value))
    }

    /// Applies element wise greater comparison and returns a boolean tensor.
    ///
    /// # Panics
    ///
    /// If the two tensors don't have the same shape.
    pub fn greater(&self, other: &Self) -> BoolTensor<B, D> {
        BoolTensor::new(B::greater(&self.value, &other.value))
    }

    /// Applies element wise greater-equal comparison and returns a boolean tensor.
    ///
    /// # Panics
    ///
    /// If the two tensors don't have the same shape.
    pub fn greater_equal(&self, other: &Self) -> BoolTensor<B, D> {
        BoolTensor::new(B::greater_equal(&self.value, &other.value))
    }

    /// Applies element wise lower comparison and returns a boolean tensor.
    ///
    /// # Panics
    ///
    /// If the two tensors don't have the same shape.
    pub fn lower(&self, other: &Self) -> BoolTensor<B, D> {
        BoolTensor::new(B::lower(&self.value, &other.value))
    }

    /// Applies element wise lower-equal comparison and returns a boolean tensor.
    ///
    /// # Panics
    ///
    /// If the two tensors don't have the same shape.
    pub fn lower_equal(&self, other: &Self) -> BoolTensor<B, D> {
        BoolTensor::new(B::lower_equal(&self.value, &other.value))
    }

    /// Applies element wise equal comparison and returns a boolean tensor.
    pub fn equal_scalar<E: ElementConversion>(&self, other: E) -> BoolTensor<B, D> {
        BoolTensor::new(B::equal_scalar(&self.value, &other.to_elem()))
    }

    /// Applies element wise greater comparison and returns a boolean tensor.
    pub fn greater_scalar<E: ElementConversion>(&self, other: E) -> BoolTensor<B, D> {
        BoolTensor::new(B::greater_scalar(&self.value, &other.to_elem()))
    }

    /// Applies element wise greater-equal comparison and returns a boolean tensor.
    pub fn greater_equal_scalar<E: ElementConversion>(&self, other: E) -> BoolTensor<B, D> {
        BoolTensor::new(B::greater_equal_scalar(&self.value, &other.to_elem()))
    }

    /// Applies element wise lower comparison and returns a boolean tensor.
    pub fn lower_scalar<E: ElementConversion>(&self, other: E) -> BoolTensor<B, D> {
        BoolTensor::new(B::lower_scalar(&self.value, &other.to_elem()))
    }

    /// Applies element wise lower-equal comparison and returns a boolean tensor.
    pub fn lower_equal_scalar<E: ElementConversion>(&self, other: E) -> BoolTensor<B, D> {
        BoolTensor::new(B::lower_equal_scalar(&self.value, &other.to_elem()))
    }

    /// Create a random tensor of the given shape where each element is sampled from the given
    /// distribution.
    pub fn random<S: Into<Shape<D>>>(shape: S, distribution: Distribution<B::Elem>) -> Self {
        let tensor = B::random(shape.into(), distribution, B::Device::default());
        Self::new(tensor)
    }

    /// Create a tensor of the given shape where each element is zero.
    pub fn zeros<S: Into<Shape<D>>>(shape: S) -> Self {
        let tensor = B::zeros(shape.into(), B::Device::default());
        Self::new(tensor)
    }

    /// Create a tensor of the given shape where each element is zero.
    pub fn zeros_device<S: Into<Shape<D>>>(shape: S, device: B::Device) -> Self {
        let tensor = B::zeros(shape.into(), device);
        Self::new(tensor)
    }

    /// Create a tensor of the given shape where each element is one.
    pub fn ones<S: Into<Shape<D>>>(shape: S) -> Self {
        let tensor = B::ones(shape.into(), B::Device::default());
        Self::new(tensor)
    }

    /// Returns a tensor containing the elements selected from the given ranges.
    ///
    /// # Panics
    ///
    /// If a range exceeds the number of elements on a dimension.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///     let tensor = Tensor::<B, 3>::ones(Shape::new([2, 3, 3]));
    ///     let tensor_indexed = tensor.index([0..1, 0..3, 1..2]);
    ///     println!("{:?}", tensor_indexed.shape());
    ///     // Shape { dims: [1, 3, 2] }
    /// }
    /// ```
    pub fn index<const D2: usize>(&self, indexes: [std::ops::Range<usize>; D2]) -> Self {
        Self::new(B::index(&self.value, indexes))
    }

    /// Returns a copy of the current tensor with the selected elements changed to the new ones at
    /// the selected indexes.
    ///
    /// # Panics
    ///
    /// - If a range exceeds the number of elements on a dimension.
    /// - If the given values don't match the given ranges.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///     let tensor = Tensor::<B, 3>::ones(Shape::new([2, 3, 3]));
    ///     let values = Tensor::<B, 3>::zeros(Shape::new([1, 1, 1]));
    ///     let tensor_indexed = tensor.index_assign([0..1, 0..1, 0..1], &values);
    ///     println!("{:?}", tensor_indexed.shape());
    ///     // Shape { dims: [2, 3, 3] }
    /// }
    /// ```
    pub fn index_assign<const D2: usize>(
        &self,
        indexes: [std::ops::Range<usize>; D2],
        values: &Self,
    ) -> Self {
        Self::new(B::index_assign(&self.value, indexes, &values.value))
    }

    /// Fill each element with the given value based on the given mask.
    pub fn mask_fill<E: ElementConversion>(&self, mask: &BoolTensor<B, D>, value: E) -> Self {
        Self::new(B::mask_fill(&self.value, &mask.value, value.to_elem()))
    }

    /// Returns a tensor with full precision based on the selected backend.
    pub fn to_full_precision(&self) -> Tensor<B::FullPrecisionBackend, D> {
        Tensor::new(B::to_full_precision(&self.value))
    }

    /// Returns a tensor on the selected backend from a full precision tensor.
    pub fn from_full_precision(tensor: Tensor<B::FullPrecisionBackend, D>) -> Self {
        Self::new(B::from_full_precision(&tensor.value))
    }

    /// Applies the argmax function along the given dimension and returns an integer tensor.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///     let tensor = Tensor::<B, 3>::ones(Shape::new([2, 3, 3]));
    ///     let tensor = tensor.argmax(1);
    ///     println!("{:?}", tensor.shape());
    ///     // Shape { dims: [2, 1, 3] }
    /// }
    /// ```
    pub fn argmax(&self, dim: usize) -> Tensor<B::IntegerBackend, D> {
        Tensor::new(B::argmax(&self.value, dim))
    }

    /// Applies the argmin function along the given dimension and returns an integer tensor.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///     let tensor = Tensor::<B, 3>::ones(Shape::new([2, 3, 3]));
    ///     let tensor = tensor.argmin(1);
    ///     println!("{:?}", tensor.shape());
    ///     // Shape { dims: [2, 1, 3] }
    /// }
    /// ```
    pub fn argmin(&self, dim: usize) -> Tensor<B::IntegerBackend, D> {
        Tensor::new(B::argmin(&self.value, dim))
    }

    /// Concatenates all tensors into a new one along the given dimension.
    ///
    /// # Panics
    ///
    /// If all tensors don't have the same shape.
    pub fn cat(tensors: Vec<Self>, dim: usize) -> Self {
        let tensors: Vec<B::TensorPrimitive<D>> = tensors.into_iter().map(|a| a.value).collect();
        let tensors: Vec<&B::TensorPrimitive<D>> = tensors.iter().collect();
        let value = B::TensorPrimitive::cat(tensors, dim);

        Self::new(value)
    }

    /// Detach the current tensor from the autodiff graph.
    /// This function does nothing when autodiff is not enabled.
    /// This can be used in batchers or elsewere to ensure that previous operations are not
    /// considered in the autodiff graph.
    pub fn detach(self) -> Self {
        Self::new(B::detach(&self.value))
    }

    /// Unsqueeze the current tensor. Create new dimensions to fit the given size.
    ///
    /// # Panics
    ///
    /// If the output size is higher than the current tensor.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///     let tensor = Tensor::<B, 2>::ones(Shape::new([3, 3]));
    ///     let tensor = tensor.unsqueeze::<4>();
    ///     println!("{:?}", tensor.shape());
    ///     // Shape { dims: [1, 1, 3, 3] }
    /// }
    /// ```
    pub fn unsqueeze<const D2: usize>(&self) -> Tensor<B, D2> {
        if D2 < D {
            panic!(
                "Can't unsqueeze smaller tensor, got dim {}, expected > {}",
                D2, D
            )
        }

        let mut dims = [1; D2];
        let num_ones = D2 - D;
        let shape = self.shape();

        dims[num_ones..(D + num_ones)].copy_from_slice(&shape.dims[..D]);

        let shape = Shape::new(dims);
        self.reshape(shape)
    }

    /// Repeat the tensor along the given dimension.
    ///
    /// # Panics
    ///
    /// If the selected dimension more than one item.
    pub fn repeat(&self, dim: usize, times: usize) -> Self {
        Self::new(B::repeat(&self.value, dim, times))
    }

    pub(crate) fn relu(&self) -> Self {
        Self::new(self.value.relu())
    }
}

impl<const D: usize, B> std::ops::Add<Self> for Tensor<B, D>
where
    B: Backend,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Tensor::add(&self, &other)
    }
}

impl<E, const D: usize, B> std::ops::Add<E> for Tensor<B, D>
where
    E: ElementConversion,
    B: Backend,
{
    type Output = Self;

    fn add(self, other: E) -> Self {
        Tensor::add_scalar(&self, other)
    }
}

impl<const D: usize, B> std::ops::Sub<Self> for Tensor<B, D>
where
    B: Backend,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Tensor::sub(&self, &other)
    }
}

impl<E, const D: usize, B> std::ops::Sub<E> for Tensor<B, D>
where
    E: ElementConversion,
    B: Backend,
{
    type Output = Self;

    fn sub(self, other: E) -> Self {
        Tensor::sub_scalar(&self, other)
    }
}

impl<const D: usize, B> std::ops::Mul<Self> for Tensor<B, D>
where
    B: Backend,
{
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Tensor::mul(&self, &other)
    }
}

impl<E, const D: usize, B> std::ops::Mul<E> for Tensor<B, D>
where
    E: ElementConversion,
    B: Backend,
{
    type Output = Self;

    fn mul(self, other: E) -> Self {
        Tensor::mul_scalar(&self, other)
    }
}

impl<const D: usize, B> std::ops::Div<Self> for Tensor<B, D>
where
    B: Backend,
{
    type Output = Self;

    fn div(self, other: Self) -> Self {
        Tensor::div(&self, &other)
    }
}

impl<E, const D: usize, B> std::ops::Div<E> for Tensor<B, D>
where
    E: ElementConversion,
    B: Backend,
{
    type Output = Self;

    fn div(self, other: E) -> Self {
        Tensor::div_scalar(&self, other)
    }
}

impl<const D: usize, B: ADBackend> Tensor<B, D> {
    pub fn backward(&self) -> Gradients {
        B::backward::<D>(&self.value)
    }

    pub fn grad(&self, grads: &Gradients) -> Option<Tensor<B::InnerBackend, D>> {
        B::grad(&self.value, grads).map(Tensor::new)
    }

    pub fn inner(&self) -> Tensor<B::InnerBackend, D> {
        Tensor::new(B::inner(&self.value))
    }

    pub fn update(&mut self, other_inner: Tensor<B::InnerBackend, D>) {
        self.value = B::from_inner(other_inner.value);
    }

    pub fn from_inner(inner: Tensor<B::InnerBackend, D>) -> Self {
        Self::new(B::from_inner(inner.value))
    }
}
