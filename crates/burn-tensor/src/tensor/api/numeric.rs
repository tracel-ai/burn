use alloc::vec::Vec;

use crate::alloc::borrow::ToOwned;

use crate::TensorPrimitive;
use crate::{
    backend::Backend,
    check,
    check::TensorCheck,
    ops::{Device, IntTensor},
    BasicOps, Bool, Distribution, Element, ElementConversion, Float, Int, Shape, Tensor,
    TensorKind,
};

impl<B, const D: usize, K> Tensor<B, D, K>
where
    B: Backend,
    K: Numeric<B>,
    K::Elem: Element,
{
    /// Applies element wise addition operation.
    ///
    /// `y = x2 + x1`
    ///
    /// # Arguments
    ///
    /// * `other` - The tensor to add.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor1 = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let tensor2 = Tensor::<B, 2>::from_data([[2.0, 3.0, 4.0], [1.0, 2.0, 3.0]], &device);
    ///    let tensor = tensor1 + tensor2;
    ///    println!("{tensor}");
    ///    // [[3.0, 1.0, 7.0], [6.0, 11.0, 9.0]]
    /// }
    /// ```
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, other: Self) -> Self {
        check!(TensorCheck::binary_ops_ew("Add", &self, &other));
        Self::new(K::add(self.primitive, other.primitive))
    }

    /// Applies element wise addition operation with a scalar.
    ///
    /// `y = x + s`
    ///
    /// # Arguments
    ///
    /// * `other` - The scalar to add, element wise.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///   let device = B::Device::default();
    ///   let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///   let scalar = 2.0;
    ///   let tensor = tensor + scalar;
    ///   println!("{tensor}");
    ///   // [[3.0, 0.0, 5.0], [7.0, 11.0, 8.0]]
    /// }
    /// ```
    pub fn add_scalar<E: ElementConversion>(self, other: E) -> Self {
        Self::new(K::add_scalar::<E>(self.primitive, other))
    }

    /// Applies element wise subtraction operation.
    ///
    /// `y = x2 - x1`
    ///
    /// # Arguments
    ///
    /// * `other` - The tensor to subtract.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///   let device = B::Device::default();
    ///   let tensor1 = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///   let tensor2 = Tensor::<B, 2>::from_data([[2.0, 3.0, 4.0], [1.0, 2.0, 3.0]], &device);
    ///   let tensor = tensor1 - tensor2;
    ///   println!("{tensor}");
    ///   // [[-1.0, -5.0, -1.0], [4.0, 7.0, 3.0]]
    /// }
    /// ```
    #[allow(clippy::should_implement_trait)]
    pub fn sub(self, other: Self) -> Self {
        check!(TensorCheck::binary_ops_ew("Sub", &self, &other));
        Self::new(K::sub(self.primitive, other.primitive))
    }

    /// Applies element wise subtraction operation with a scalar.
    ///
    /// `y = x - s`
    ///
    /// # Arguments
    ///
    /// * `other` - The scalar to subtract, element wise.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let scalar = 2.0;
    ///    let tensor = tensor - scalar;
    ///    println!("{tensor}");
    ///    // [[-1.0, -4.0, 1.0], [3.0, 7.0, 4.0]]
    /// }
    /// ```
    pub fn sub_scalar<E: ElementConversion>(self, other: E) -> Self {
        Self::new(K::sub_scalar::<E>(self.primitive, other))
    }

    /// Applies element wise division operation.
    ///
    /// `y = x2 / x1`
    ///
    /// # Arguments
    ///
    /// * `other` - The tensor to divide.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor1 = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let tensor2 = Tensor::<B, 2>::from_data([[2.0, 3.0, 4.0], [1.0, 2.0, 3.0]], &device);
    ///    let tensor = tensor1 / tensor2;
    ///    println!("{tensor}");
    ///    // [[0.5, -0.6666667, 0.75], [5.0, 4.5, 2.0]]
    /// }
    /// ```
    #[allow(clippy::should_implement_trait)]
    pub fn div(self, other: Self) -> Self {
        check!(TensorCheck::binary_ops_ew("Div", &self, &other));
        Self::new(K::div(self.primitive, other.primitive))
    }

    /// Applies element wise division operation with a scalar.
    ///
    /// `y = x / s`
    ///
    /// # Arguments
    ///
    /// * `other` - The scalar to divide, element wise.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let scalar = 2.0;
    ///    let tensor = tensor / scalar;
    ///    println!("{tensor}");
    ///    // [[0.5, -1.0, 1.5], [2.5, 4.5, 3.0]]
    /// }
    /// ```
    pub fn div_scalar<E: ElementConversion>(self, other: E) -> Self {
        Self::new(K::div_scalar::<E>(self.primitive, other))
    }

    /// Applies element wise the remainder operation with a scalar.
    ///
    /// `y = x2 % x1`
    pub fn remainder(self, other: Self) -> Self {
        Self::new(K::remainder(self.primitive, other.primitive))
    }

    /// Applies element wise the remainder operation with a scalar.
    ///
    /// `y = x2 % x1`
    ///
    /// # Arguments
    ///
    /// * `other` - The scalar to divide, element wise.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor1 = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let scalar = 2.0;
    ///    let tensor = tensor1 % scalar;
    ///    println!("{tensor}");
    ///    // [[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    /// }
    /// ```
    pub fn remainder_scalar<E: ElementConversion>(self, other: E) -> Self {
        Self::new(K::remainder_scalar::<E>(self.primitive, other))
    }

    /// Applies element wise multiplication operation.
    ///
    /// `y = x2 * x1`
    ///
    /// # Arguments
    ///
    /// * `other` - The tensor to multiply.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor1 = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let tensor2 = Tensor::<B, 2>::from_data([[2.0, 3.0, 4.0], [1.0, 2.0, 3.0]], &device);
    ///    let tensor = tensor1 * tensor2;
    ///    println!("{tensor}");
    ///    // [[2.0, -6.0, 12.0], [5.0, 18.0, 18.0]]
    /// }
    /// ```
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, other: Self) -> Self {
        check!(TensorCheck::binary_ops_ew("Mul", &self, &other));
        Self::new(K::mul(self.primitive, other.primitive))
    }

    /// Applies element wise multiplication operation with a scalar.
    ///
    /// `y = x * s`
    ///
    /// # Arguments
    ///
    /// * `other` - The scalar to multiply, element wise.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let scalar = 2.0;
    ///    let tensor = tensor * scalar;
    ///    println!("{tensor}");
    ///    // [[2.0, -4.0, 6.0], [10.0, 18.0, 12.0]]
    /// }
    /// ```
    pub fn mul_scalar<E: ElementConversion>(self, other: E) -> Self {
        Self::new(K::mul_scalar::<E>(self.primitive, other))
    }

    /// Switch sign of each element in the tensor.
    ///
    /// `y = -x`
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let tensor = -tensor;
    ///    println!("{tensor}");
    ///    // [[-1.0, 2.0, -3.0], [-5.0, -9.0, -6.0]]
    /// }
    /// ```
    #[allow(clippy::should_implement_trait)]
    pub fn neg(self) -> Self {
        Self::new(K::neg(self.primitive))
    }

    /// Returns the signs of the elements of the input tensor.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let tensor = tensor.sign();
    ///    println!("{tensor}");
    ///    // [[1.0, -1.0, 1.0], [1.0, 1.0, 1.0]]
    /// }
    /// ```
    pub fn sign(self) -> Self {
        Self::new(K::sign(self.primitive))
    }

    /// Create a tensor of the given shape where each element is zero.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::zeros(Shape::new([2, 3]), &device);
    ///    println!("{tensor}");
    ///    // [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    /// }
    /// ```
    pub fn zeros<S: Into<Shape>>(shape: S, device: &B::Device) -> Self {
        let shape = shape.into();
        check!(TensorCheck::creation_ops::<D>("Zeros", &shape.dims));
        Self::new(K::zeros(shape, device))
    }

    /// Returns a new tensor with the same shape and device as the current tensor filled with zeros.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///   let device = B::Device::default();
    ///   let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///   let tensor = tensor.zeros_like();
    ///   println!("{tensor}");
    ///   // [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    /// }
    /// ```
    pub fn zeros_like(&self) -> Self {
        Self::zeros(self.shape(), &self.device())
    }

    /// Create a tensor of the given shape where each element is one.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///   let device = B::Device::default();
    ///   let tensor = Tensor::<B, 2>::ones(Shape::new([2, 3]), &device);
    ///   println!("{tensor}");
    ///   // [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    /// }
    /// ```
    pub fn ones<S: Into<Shape>>(shape: S, device: &B::Device) -> Self {
        let shape = shape.into();
        check!(TensorCheck::creation_ops::<D>("Ones", &shape.dims));
        Self::new(K::ones(shape, device))
    }

    /// Returns a new tensor with the same shape and device as the current tensor filled with ones.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let tensor = tensor.ones_like();
    ///    println!("{tensor}");
    ///    // [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    /// }
    /// ```
    pub fn ones_like(&self) -> Self {
        Self::ones(self.shape(), &self.device())
    }

    /// Create a tensor of the given shape where each element is equal to the provided value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///   let device = B::Device::default();
    ///   let tensor = Tensor::<B, 2>::full(Shape::new([2, 3]), 5.0, &device);
    ///   println!("{tensor}");
    ///   // [[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]]
    /// }
    /// ```
    pub fn full<S: Into<Shape>, E: ElementConversion>(
        shape: S,
        fill_value: E,
        device: &B::Device,
    ) -> Self {
        let shape = shape.into();
        check!(TensorCheck::creation_ops::<D>("Full", &shape.dims));
        Self::new(K::full(shape, fill_value, device))
    }

    ///Returns a new tensor with the same shape and device as the current tensor filled with the provided value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let tensor = tensor.full_like(5.0);
    ///    println!("{tensor}");
    ///    // [[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]]
    /// }
    /// ```
    pub fn full_like<E: ElementConversion>(&self, fill_value: E) -> Self {
        Self::full(self.shape(), fill_value, &self.device())
    }

    /// Aggregate all elements in the tensor with the mean operation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let tensor = tensor.mean();
    ///    println!("{tensor}");
    ///    // [3.6666667]
    /// }
    /// ```
    pub fn mean(self) -> Tensor<B, 1, K> {
        Tensor::new(K::mean(self.primitive))
    }

    /// Aggregate all elements in the tensor with the sum operation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///   let device = B::Device::default();
    ///   let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///   let tensor = tensor.sum();
    ///   println!("{tensor}");
    ///   // [22.0]
    /// }
    /// ```
    pub fn sum(self) -> Tensor<B, 1, K> {
        Tensor::new(K::sum(self.primitive))
    }

    /// Aggregate all elements along the given *dimension* or *axis*
    /// in the tensor with the mean operation.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension or axis along which to aggregate the elements.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///   let device = B::Device::default();
    ///   let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///   let tensor = tensor.clone().mean_dim(0);
    ///   println!("{tensor}");
    ///   // [[3.0, 3.5, 4.5]]
    ///   let tensor = tensor.clone().mean_dim(1);
    ///   println!("{tensor}");
    ///   // [[0.6666667], [6.6666665]]
    /// }
    /// ```
    pub fn mean_dim(self, dim: usize) -> Self {
        check!(TensorCheck::aggregate_dim::<D>("Mean", dim));
        Self::new(K::mean_dim(self.primitive, dim))
    }

    /// Aggregate all elements along the given *dimension* or *axis*
    /// in the tensor with the sum operation.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension or axis along which to aggregate the elements.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let tensor = tensor.clone().sum_dim(0);
    ///    println!("{tensor}");
    ///    let tensor = tensor.clone().sum_dim(1);
    ///    // [[6.0, 7.0, 9.0]]
    ///    println!("{tensor}");
    ///    // [[2.0], [20.0]]
    /// }
    /// ```
    pub fn sum_dim(self, dim: usize) -> Self {
        check!(TensorCheck::aggregate_dim::<D>("Sum", dim));
        Self::new(K::sum_dim(self.primitive, dim))
    }

    /// Aggregate all elements along the given *dimension* or *axis*
    /// in the tensor with the product operation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let tensor = tensor.prod();
    ///    println!("{tensor}");
    ///    // [-1620.0]
    /// }
    /// ```
    pub fn prod(self) -> Tensor<B, 1, K> {
        Tensor::new(K::prod(self.primitive))
    }

    /// Aggregate all elements along the given *dimension* or *axis*
    /// in the tensor with the product operation.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension or axis along which to aggregate the elements.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let tensor = tensor.clone().prod_dim(0);
    ///    println!("{tensor}");
    ///    // [[5.0, -18.0, 18.0]]
    ///    let tensor = tensor.clone().prod_dim(1);
    ///    println!("{tensor}");
    ///    // [[-6.0], [270.0]]
    /// }
    /// ```
    pub fn prod_dim(self, dim: usize) -> Self {
        check!(TensorCheck::aggregate_dim::<D>("Prod", dim));
        Self::new(K::prod_dim(self.primitive, dim))
    }

    /// Applies element wise equal comparison and returns a boolean tensor.
    ///
    /// # Arguments
    ///
    /// * `other` - The element to compare.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let tensor = tensor.equal_elem(3.0);
    ///    println!("{tensor}");
    ///    // [[false, false, true], [false, false, false]]
    /// }
    /// ```
    pub fn equal_elem<E: Element>(self, other: E) -> Tensor<B, D, Bool> {
        Tensor::new(K::equal_elem(self.primitive, other.elem()))
    }

    /// Applies element wise non-equality comparison and returns a boolean tensor.
    ///
    /// # Arguments
    ///
    /// * `other` - The element to compare.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let tensor = tensor.not_equal_elem(3.0);
    ///    println!("{tensor}");
    ///    // [[true, true, false], [true, true, true]]
    /// }
    /// ```
    pub fn not_equal_elem<E: Element>(self, other: E) -> Tensor<B, D, Bool> {
        Tensor::new(K::not_equal_elem(self.primitive, other.elem()))
    }

    /// Applies element wise greater comparison and returns a boolean tensor.
    ///
    /// # Panics
    ///
    /// If the two tensors don't have the same shape.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///   let device = B::Device::default();
    ///   let tensor1 = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///   let tensor2 = Tensor::<B, 2>::from_data([[2.0, 3.0, 4.0], [1.0, 2.0, 3.0]], &device);
    ///   let tensor = tensor1.greater(tensor2);
    ///   println!("{tensor}");
    ///   // [[false, false, false], [true, true, true]]
    /// }
    /// ```
    pub fn greater(self, other: Self) -> Tensor<B, D, Bool> {
        check!(TensorCheck::binary_ops_ew("Greater", &self, &other));
        Tensor::new(K::greater(self.primitive, other.primitive))
    }

    /// Applies element wise greater-equal comparison and returns a boolean tensor.
    ///
    /// # Panics
    ///
    /// If the two tensors don't have the same shape.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor1 = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let tensor2 = Tensor::<B, 2>::from_data([[2.0, 3.0, 4.0], [1.0, 2.0, 3.0]], &device);
    ///    let tensor = tensor1.greater_equal(tensor2);
    ///    println!("{tensor}");
    ///    // [[false, false, false], [true, true, true]]
    /// }
    /// ```
    pub fn greater_equal(self, other: Self) -> Tensor<B, D, Bool> {
        check!(TensorCheck::binary_ops_ew("Greater_equal", &self, &other));
        Tensor::new(K::greater_equal(self.primitive, other.primitive))
    }

    /// Applies element wise lower comparison and returns a boolean tensor.
    ///
    /// # Panics
    ///
    /// If the two tensors don't have the same shape.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor1 = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let tensor2 = Tensor::<B, 2>::from_data([[2.0, 3.0, 4.0], [1.0, 2.0, 3.0]], &device);
    ///    let tensor = tensor1.lower(tensor2);
    ///    println!("{tensor}");
    ///    // [[true, true, true], [false, false, false]]
    /// }
    /// ```
    pub fn lower(self, other: Self) -> Tensor<B, D, Bool> {
        check!(TensorCheck::binary_ops_ew("Lower", &self, &other));
        Tensor::new(K::lower(self.primitive, other.primitive))
    }

    /// Applies element wise lower-equal comparison and returns a boolean tensor.
    ///
    /// # Panics
    ///
    /// If the two tensors don't have the same shape.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor1 = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let tensor2 = Tensor::<B, 2>::from_data([[2.0, 3.0, 4.0], [1.0, 2.0, 3.0]], &device);
    ///    let tensor = tensor1.lower_equal(tensor2);
    ///    println!("{tensor}");
    ///    // [[true, true, true], [false, false, false]]
    /// }
    /// ```
    pub fn lower_equal(self, other: Self) -> Tensor<B, D, Bool> {
        check!(TensorCheck::binary_ops_ew("Lower_equal", &self, &other));
        Tensor::new(K::lower_equal(self.primitive, other.primitive))
    }

    /// Applies element wise greater comparison and returns a boolean tensor.
    ///
    /// # Arguments
    ///
    /// * `other` - The element to compare.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let tensor = tensor.greater_elem(3.0);
    ///    println!("{tensor}");
    ///    // [[false, false, true], [true, true, true]]
    /// }
    /// ```
    pub fn greater_elem<E: ElementConversion>(self, other: E) -> Tensor<B, D, Bool> {
        Tensor::new(K::greater_elem(self.primitive, other.elem()))
    }

    /// Applies element wise greater-equal comparison and returns a boolean tensor.
    ///
    /// # Arguments
    ///
    /// * `other` - The element to compare.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let tensor = tensor.greater_equal_elem(3.0);
    ///    println!("{tensor}");
    ///    // [[false, false, true], [true, true, true]]
    /// }
    /// ```
    pub fn greater_equal_elem<E: ElementConversion>(self, other: E) -> Tensor<B, D, Bool> {
        Tensor::new(K::greater_equal_elem(self.primitive, other.elem()))
    }

    /// Applies element wise lower comparison and returns a boolean tensor.
    ///
    /// # Arguments
    ///
    /// * `other` - The element to compare.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///     let device = B::Device::default();
    ///     let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///     let tensor = tensor.lower_elem(3.0);
    ///     println!("{tensor}");
    ///     // [[true, true, false], [false, false, false]]
    /// }
    /// ```
    pub fn lower_elem<E: ElementConversion>(self, other: E) -> Tensor<B, D, Bool> {
        Tensor::new(K::lower_elem(self.primitive, other.elem()))
    }

    /// Applies element wise lower-equal comparison and returns a boolean tensor.
    ///
    /// # Arguments
    ///
    /// * `other` - The element to compare.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let tensor = tensor.lower_equal_elem(3.0);
    ///    println!("{tensor}");
    ///    // [[true, true, true], [false, false, false]]
    /// }
    /// ```
    pub fn lower_equal_elem<E: ElementConversion>(self, other: E) -> Tensor<B, D, Bool> {
        Tensor::new(K::lower_equal_elem(self.primitive, other.elem()))
    }

    /// Update the given tensor with the value tensor where the mask is true.
    ///
    /// This is similar to [mask_fill](Tensor::mask_fill), however the value is a tensor instead of
    /// a scalar.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape, Bool};
    ///
    /// fn example<B: Backend>() {
    ///   let device = B::Device::default();
    ///   let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///   let mask = Tensor::<B, 2, Bool>::from_data([[true, false, true], [false, true, false]], &device);
    ///   let value = Tensor::<B, 2>::from_data([[2.0, 3.0, 4.0], [1.0, 2.0, 3.0]], &device);
    ///   let tensor = tensor.mask_where(mask, value);
    ///   println!("{tensor}");
    ///   // [[2.0, -2.0, 4.0], [5.0, 2.0, 6.0]]
    /// }
    /// ```
    pub fn mask_where(self, mask: Tensor<B, D, Bool>, value: Self) -> Self {
        Self::new(K::mask_where(
            self.primitive,
            mask.primitive,
            value.primitive,
        ))
    }

    /// Update the given tensor with the value where the mask is true.
    ///
    /// This is similar to [mask_where](Tensor::mask_where), however the value is a scalar instead of
    /// a tensor.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape, Bool};
    ///
    /// fn example<B: Backend>() {
    ///   let device = B::Device::default();
    ///   let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///   let mask = Tensor::<B, 2, Bool>::from_data([[true, false, true], [false, true, false]], &device);
    ///   let tensor = tensor.mask_fill(mask, 3.0);
    ///   println!("{tensor}");
    ///   // [[3.0, -2.0, 3.0], [5.0, 3.0, 6.0]]
    /// }
    /// ```
    pub fn mask_fill<E: ElementConversion>(self, mask: Tensor<B, D, Bool>, value: E) -> Self {
        Self::new(K::mask_fill(self.primitive, mask.primitive, value.elem()))
    }

    /// Gather tensor elements corresponding to the given indices from the specified dim.
    ///
    /// Example using a 3D tensor:
    ///
    /// `output[i, j, k] = input[indices[i, j, k], j, k]; // dim = 0`
    /// `output[i, j, k] = input[i, indices[i, j, k], k]; // dim = 1`
    /// `output[i, j, k] = input[i, j, indices[i, j, k]]; // dim = 2`
    ///
    /// # Notes
    ///
    /// The index tensor should have the same shape as the original tensor except for the dim
    /// specified.
    ///
    /// # Warning
    /// Not all backends have runtime bound checks for the indices, so make sure the they are valid.
    /// Otherwise, out of bounds indices could lead to unexpected results instead of panicking.
    pub fn gather(self, dim: usize, indices: Tensor<B, D, Int>) -> Self {
        check!(TensorCheck::gather::<D>(
            dim,
            &self.shape(),
            &indices.shape()
        ));

        Self::new(K::gather(dim, self.primitive, indices.primitive))
    }

    /// Assign the gathered elements corresponding to the given indices along the specified dimension
    /// from the value tensor to the original tensor using sum reduction.
    ///
    /// Example using a 3D tensor:
    ///
    /// `input[indices[i, j, k], j, k] += values[i, j, k]; // dim = 0`
    /// `input[i, indices[i, j, k], k] += values[i, j, k]; // dim = 1`
    /// `input[i, j, indices[i, j, k]] += values[i, j, k]; // dim = 2`
    ///
    /// # Notes
    ///
    /// The index tensor should have the same shape as the original tensor except for the specified
    /// dimension. The value and index tensors should have the same shape.
    ///
    /// Other references to the input tensor will not be modified by this operation.
    ///
    /// # Warning
    /// Not all backends have runtime bound checks for the indices, so make sure the they are valid.
    /// Otherwise, out of bounds indices could lead to unexpected results instead of panicking.
    pub fn scatter(self, dim: usize, indices: Tensor<B, D, Int>, values: Self) -> Self {
        check!(TensorCheck::scatter::<D>(
            dim,
            &self.shape(),
            &indices.shape(),
            &values.shape()
        ));

        Self::new(K::scatter(
            dim,
            self.primitive,
            indices.primitive,
            values.primitive,
        ))
    }

    /// Select the tensor elements along the given dimension corresponding to the given indices.
    ///
    /// Example using a 3D tensor:
    ///
    /// `output[i, j, k] = input[indices[i], j, k]; // dim = 0`
    /// `output[i, j, k] = input[i, indices[j], k]; // dim = 1`
    /// `output[i, j, k] = input[i, j, indices[k]]; // dim = 2`
    ///
    /// # Warning
    /// Not all backends have runtime bound checks for the indices, so make sure the they are valid.
    /// Otherwise, out of bounds indices could lead to unexpected results instead of panicking.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape, Int};
    ///
    /// fn example<B: Backend>() {
    ///   let device = B::Device::default();
    ///   let tensor = Tensor::<B, 3>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///   let indices = Tensor::<B, 1, Int>::from_data([0], &device);
    ///   let tensor = tensor.select(0, indices);
    ///   println!("{tensor}");
    ///   //  [[1.0, -2.0, 3.0]]
    /// }
    /// ```
    pub fn select(self, dim: usize, indices: Tensor<B, 1, Int>) -> Self {
        check!(TensorCheck::select::<D>(dim));
        Self::new(K::select(self.primitive, dim, indices))
    }

    /// Assign the selected elements along the given dimension corresponding to the given indices
    /// from the value tensor to the original tensor using sum reduction.
    ///
    /// Example using a 3D tensor:
    ///
    /// `input[indices[i], j, k] += values[i, j, k]; // dim = 0`
    /// `input[i, indices[j], k] += values[i, j, k]; // dim = 1`
    /// `input[i, j, indices[k]] += values[i, j, k]; // dim = 2`
    ///
    /// # Warning
    /// Not all backends have runtime bound checks for the indices, so make sure the they are valid.
    /// Otherwise, out of bounds indices could lead to unexpected results instead of panicking.
    pub fn select_assign(
        self,
        dim: usize,
        indices: Tensor<B, 1, Int>,
        values: Tensor<B, D, K>,
    ) -> Self {
        check!(TensorCheck::select_assign::<D>(dim));

        Self::new(K::select_assign(
            self.primitive,
            dim,
            indices,
            values.primitive,
        ))
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
    ///     let device = B::Device::default();
    ///     let tensor = Tensor::<B, 3>::ones(Shape::new([2, 3, 3]), &device);
    ///     let tensor = tensor.argmax(1);
    ///     println!("{:?}", tensor.shape());
    ///     // Shape { dims: [2, 1, 3] }
    /// }
    /// ```
    pub fn argmax(self, dim: usize) -> Tensor<B, D, Int> {
        Tensor::new(K::argmax(self.primitive, dim))
    }

    /// Find the maximum value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///   let device = B::Device::default();
    ///   let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///   let tensor = tensor.max();
    ///   println!("{tensor}");
    ///   // [9.0]
    /// }
    /// ```
    pub fn max(self) -> Tensor<B, 1, K> {
        Tensor::new(K::max(self.primitive))
    }

    /// Find the maximum value along the given dimension.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///   let device = B::Device::default();
    ///   let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///   let tensor = tensor.max_dim(0);
    ///   println!("{tensor}");
    ///   // [[5.0, 9.0, 6.0]]
    /// }
    /// ```
    pub fn max_dim(self, dim: usize) -> Tensor<B, D, K> {
        check!(TensorCheck::aggregate_dim::<D>("Max", dim));

        Tensor::new(K::max_dim(self.primitive, dim))
    }

    /// Find the maximum value along the given dimension.
    ///
    /// Also returns the indices.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let (tensor, index) = tensor.max_dim_with_indices(0);
    ///    // [[5.0, 9.0, 6.0]]
    ///    println!("{tensor}");
    ///    // [[1, 1, 1]]
    ///    println!("{index}");
    /// }
    /// ```
    pub fn max_dim_with_indices(self, dim: usize) -> (Tensor<B, D, K>, Tensor<B, D, Int>) {
        check!(TensorCheck::aggregate_dim::<D>("Max", dim));

        let (tensor, index) = K::max_dim_with_indices(self.primitive, dim);

        let tensor = Tensor::new(tensor);
        let index = Tensor::new(index);

        (tensor, index)
    }

    /// Finds the maximum pair wise values with another Tensor
    ///
    /// # Arguments
    ///
    /// * `other` - Other tensor to find maximum elements with
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensors containing the maximum value found
    /// in the input tensors.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor1 = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let tensor2 = Tensor::<B, 2>::from_data([[2.0, 3.0, 4.0], [1.0, 2.0, 3.0]], &device);
    ///    let tensor = tensor1.max_pair(tensor2);
    ///    println!("{tensor}");
    ///    // [[2.0, 3.0, 4.0], [5.0, 9.0, 6.0]]
    /// }
    /// ```
    pub fn max_pair(self, other: Self) -> Self {
        let mask = self.clone().lower(other.clone());
        self.mask_where(mask, other)
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
    ///     let device = Default::default();
    ///     let tensor = Tensor::<B, 3>::ones(Shape::new([2, 3, 3]), &device);
    ///     let tensor = tensor.argmin(1);
    ///     println!("{:?}", tensor.shape());
    ///     // Shape { dims: [2, 1, 3] }
    /// }
    /// ```
    pub fn argmin(self, dim: usize) -> Tensor<B, D, Int> {
        Tensor::new(K::argmin(self.primitive, dim))
    }

    /// Find the minimum value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let tensor = tensor.min();
    ///    println!("{tensor}");
    ///    // [-2.0]
    /// }
    /// ```
    pub fn min(self) -> Tensor<B, 1, K> {
        Tensor::new(K::min(self.primitive))
    }

    /// Find the minimum value along the given dimension.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let tensor = tensor.min_dim(0);
    ///    println!("{tensor}");
    ///    // [[1.0, -2.0, 3.0]]
    /// }
    /// ```
    pub fn min_dim(self, dim: usize) -> Tensor<B, D, K> {
        check!(TensorCheck::aggregate_dim::<D>("Min", dim));
        Tensor::new(K::min_dim(self.primitive, dim))
    }

    /// Find the minimum value along the given dimension.
    ///
    /// Also returns the indices.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::from_data([[7.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let (tensor, index) = tensor.min_dim_with_indices(0);
    ///    println!("{tensor}");
    ///    // [[1.0, -2.0, 3.0]]
    ///    println!("{}", index);
    ///    // [[1, 0, 0]]
    /// }
    /// ```
    pub fn min_dim_with_indices(self, dim: usize) -> (Tensor<B, D, K>, Tensor<B, D, Int>) {
        check!(TensorCheck::aggregate_dim::<D>("Min", dim));

        let (tensor, index) = K::min_dim_with_indices(self.primitive, dim);

        let tensor = Tensor::new(tensor);
        let index = Tensor::new(index);

        (tensor, index)
    }

    /// Finds the minimum pair wise values with another Tensor
    ///
    /// # Arguments
    ///
    /// * `other` - Other tensor to find minimum elements with
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensors containing the minimum value found
    /// between each element of the two source tensors.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor1 = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let tensor2 = Tensor::<B, 2>::from_data([[2.0, 3.0, 4.0], [1.0, 2.0, 3.0]], &device);
    ///    let tensor = tensor1.min_pair(tensor2);
    ///    println!("{tensor}");
    ///    // [[1.0, -2.0, 3.0], [1.0, 2.0, 3.0]]
    /// }
    pub fn min_pair(self, other: Self) -> Self {
        let mask = other.clone().lower(self.clone());
        self.mask_where(mask, other)
    }

    /// Clamp the tensor between the given min and max values.
    ///
    /// # Arguments
    ///
    /// * `min` - The minimum value.
    /// * `max` - The maximum value.
    ///
    /// # Returns
    ///
    /// A new tensor with the values clamped between the given min and max values.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Int, Tensor};
    ///
    /// fn example<B: Backend>() {
    ///   let device = Default::default();
    ///   let tensor = Tensor::<B, 2, Int>::from_ints(
    ///    [
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    ///     [7, 8, 9]
    ///    ],
    ///    &device);
    ///    let tensor = tensor.clamp(2, 6);
    ///    println!("{tensor}");
    ///    // [[2, 2, 3], [4, 5, 6], [6, 6, 6]]
    /// }
    /// ```
    pub fn clamp<E: ElementConversion>(self, min: E, max: E) -> Self {
        Self::new(K::clamp(self.primitive, min.elem(), max.elem()))
    }

    /// Clamps a tensor under a minimum value.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to clamp.
    /// * `min` - The minimum value.
    ///
    /// # Returns
    ///
    /// A new tensor with the values clamped under the given min value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Int, Tensor};
    ///
    /// fn example<B: Backend>() {
    ///    let device = Default::default();
    ///    let tensor = Tensor::<B, 2, Int>::from_ints(
    ///    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    ///    &device);
    ///    let tensor = tensor.clamp_min(4);
    ///    println!("{tensor}");
    ///    // [[4, 4, 4], [4, 5, 6], [7, 8, 9]]
    /// }
    /// ```
    pub fn clamp_min<E: ElementConversion>(self, min: E) -> Self {
        Self::new(K::clamp_min(self.primitive, min.elem()))
    }

    /// Clamps a tensor over a maximum value.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to clamp.
    /// * `max` - The maximum value.
    ///
    /// # Returns
    ///
    /// A new tensor with the values clamped over the given max value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Int, Tensor};
    ///
    /// fn example<B: Backend>() {
    ///    let device = Default::default();
    ///    let tensor = Tensor::<B, 2, Int>::from_ints(
    ///    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    ///    &device);
    ///    let tensor = tensor.clamp_max(5);
    ///    println!("{tensor}");
    ///    // [[1, 2, 3], [4, 5, 5], [5, 5, 5]]
    /// }
    /// ```
    pub fn clamp_max<E: ElementConversion>(self, max: E) -> Self {
        Self::new(K::clamp_max(self.primitive, max.elem()))
    }

    /// Apply element wise absolute value operation
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Int, Tensor};
    ///
    /// fn example<B: Backend>() {
    ///   let device = Default::default();
    ///   let tensor = Tensor::<B, 2, Int>::from_ints([[1, -2, 3], [4, -5, 6], [7, -8, 9]], &device);
    ///   let tensor = tensor.abs();
    ///   println!("{tensor}");
    ///   // [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    /// }
    /// ```
    pub fn abs(self) -> Self {
        Self::new(K::abs(self.primitive))
    }

    /// Returns the upper triangular part of a matrix (2-D tensor) or batch of matrices input,
    /// the other elements of the result tensor out are set to 0.
    ///
    /// # Example
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Int, Tensor};
    ///
    /// fn example<B: Backend>() {
    ///    let device = Default::default();
    ///    let tensor = Tensor::<B, 2, Int>::from_ints(
    ///        [
    ///          [1, 2, 3],
    ///          [4, 5, 6],
    ///          [7, 8, 9]
    ///        ],
    ///        &device
    ///    );
    ///    let tensor = tensor.triu(1);
    ///    println!("{tensor}");
    ///    // [
    ///    //   [0, 2, 3],
    ///    //   [0, 0, 6],
    ///    //   [0, 0, 0]
    ///    // ]
    /// }
    /// ```
    pub fn triu(self, diagonal: i64) -> Self {
        check!(TensorCheck::tri::<{ D }>());

        // last two dimensions
        let shape = &self.shape().dims[D - 2..].to_owned();

        let mask = Tensor::<B, 2, Bool>::triu_mask(shape, diagonal, &self.device()).unsqueeze();
        self.mask_fill(mask, 0)
    }

    /// Returns the lower triangular part of a matrix (2-D tensor) or batch of matrices input,
    /// the other elements of the result tensor out are set to 0.
    ///
    /// # Example
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Int, Tensor};
    ///
    /// fn example<B: Backend>() {
    ///    let device = Default::default();
    ///    let tensor = Tensor::<B, 2, Int>::from_ints(
    ///        [
    ///          [1, 2, 3],
    ///          [4, 5, 6],
    ///          [7, 8, 9]
    ///        ],
    ///        &device
    ///    );
    ///
    ///    let tensor = tensor.tril(-1);
    ///    println!("{tensor}");
    ///    // [
    ///    //   [0, 0, 0],
    ///    //   [4, 0, 0],
    ///    //   [7, 8, 0]
    ///    // ]
    /// }
    /// ```
    pub fn tril(self, diagonal: i64) -> Self {
        check!(TensorCheck::tri::<{ D }>());

        // last two dimensions
        let shape = &self.shape().dims[D - 2..].to_owned();
        let mask = Tensor::<B, 2, Bool>::tril_mask(shape, diagonal, &self.device()).unsqueeze();

        self.mask_fill(mask, 0)
    }

    /// Applies element wise power operation with a float Tensor
    ///
    /// # Arguments
    ///
    /// * `other` - The tensor to apply the power operation with.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor1 = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let tensor2 = Tensor::<B, 2>::from_data([[2.0, 3.0, 4.0], [1.0, 2.0, 3.0]], &device);
    ///    let tensor = tensor1.powf(tensor2);
    ///    println!("{tensor}");
    ///    // [[1.0, 8.0, 81.0], [5.0, 81.0, 216.0]]
    /// }
    /// ```
    pub fn powf(self, other: Self) -> Self {
        Self::new(K::powf(self.primitive, other.primitive))
    }

    /// Applies element wise power operation with a float scalar
    ///
    /// # Arguments
    ///
    /// * `other` - The scalar to apply the power operation with.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let tensor = tensor.powf_scalar(2.0);
    ///    println!("{tensor}");
    ///    // [[1.0, 4.0, 9.0], [25.0, 81.0, 36.0]]
    /// }
    /// ```
    pub fn powf_scalar<E: ElementConversion>(self, other: E) -> Self {
        Self::new(K::powf_scalar::<E>(self.primitive, other))
    }

    /// Applies element wise power operation with a integer Tensor
    ///
    /// # Arguments
    ///
    /// * `other` - The tensor to apply the power operation with.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape, Int};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor1 = Tensor::<B, 2, Int>::from_ints([[1, -2, 3], [5, 9, 6]], &device);
    ///    let tensor2 = Tensor::<B, 2, Int>::from_ints([[2, 3, 4], [1, 2, 3]], &device);
    ///    let tensor = tensor1.powi(tensor2);
    ///    println!("{tensor}");
    ///    // [[1, -8, 81], [5, 81, 216]]
    /// }
    /// ```
    pub fn powi(self, other: Self) -> Self {
        Self::new(K::powi(self.primitive, other.primitive))
    }

    /// Applies element wise power operation with a integer scalar
    ///
    /// # Arguments
    ///
    /// * `other` - The scalar to apply the power operation with.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape, Int};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2, Int>::from_ints([[1, -2, 3], [5, 9, 6]], &device);
    ///    let tensor = tensor.powi_scalar(2);
    ///    println!("{tensor}");
    ///    // [[1, 4, 9], [25, 81, 36]]
    /// }
    /// ```
    pub fn powi_scalar<E: ElementConversion>(self, other: E) -> Self {
        Self::new(K::powi_scalar::<E>(self.primitive, other))
    }

    /// Checks element wise if the tensor is close to another tensor.
    ///
    /// The tolerance is defined by the following equation:
    ///
    /// ```text
    /// abs(a - b) <= (atol + rtol * abs(b))
    ///
    /// where `a` is the first tensor, `b` is the second tensor, `rtol` is the relative tolerance,
    /// and `atol` is the absolute tolerance.
    /// ```
    ///
    /// # Arguments
    ///
    /// * `other` - The tensor to compare with.
    /// * `rtol` - Optional relative tolerance. Default is 1e-5.
    /// * `atol` - Optional absolute tolerance. Default is 1e-8.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the same shape as the input tensors.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor1 = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let tensor2 = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let tensor = tensor1.is_close(tensor2, None, None);
    ///    println!("{tensor}");
    ///    // [[true, true, true], [true, true, true]]
    /// }
    /// ```
    pub fn is_close(self, other: Self, rtol: Option<f64>, atol: Option<f64>) -> Tensor<B, D, Bool> {
        let rtol = rtol.unwrap_or(1e-5);
        let atol = atol.unwrap_or(1e-8);

        Tensor::new(K::lower_equal(
            K::abs(K::sub(self.primitive, other.primitive.clone())),
            K::add_scalar(K::mul_scalar(K::abs(other.primitive), rtol), atol),
        ))
    }

    /// Checks if all elements are close to another tensor.
    ///
    /// The tolerance is defined by the following equation:
    ///
    /// ```text
    ///
    /// abs(a - b) <= (atol + rtol * abs(b))
    ///
    /// where `a` is the first tensor, `b` is the second tensor, `rtol` is the relative tolerance,
    /// and `atol` is the absolute tolerance.
    ///
    /// ```
    ///
    /// # Arguments
    ///
    /// * `other` - The tensor to compare with.
    /// * `rtol` - Optional relative tolerance. Default is 1e-5.
    /// * `atol` - Optional absolute tolerance. Default is 1e-8.
    ///
    /// # Returns
    ///
    /// A boolean scalar.
    ///
    /// # Remarks
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor1 = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let tensor2 = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let result = tensor1.all_close(tensor2, None, None);
    ///    println!("{}", result);
    ///    // true
    /// }
    /// ```
    pub fn all_close(self, other: Self, rtol: Option<f64>, atol: Option<f64>) -> bool {
        self.is_close(other, rtol, atol).all().into_scalar()
    }

    /// Converts the tensor to a boolean tensor by checking if the elements are non-zero.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the same shape as the input tensor.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///   let device = B::Device::default();
    ///   let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [0.0, 9.0, 6.0]], &device);
    ///   let tensor = tensor.bool();
    ///   println!("{tensor}");
    ///   // [
    ///   //   [true, true, true],
    ///   //   [false, true, true]
    ///   // ]
    /// }
    pub fn bool(self) -> Tensor<B, D, Bool> {
        Tensor::new(K::not_equal_elem(self.primitive, 0.elem()))
    }

    /// Create a random tensor of the given shape on the given device where each element is
    /// sampled from the given distribution.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `distribution` - The distribution to sample from.
    /// * `device` - The device to create the tensor on.
    ///
    /// # Returns
    ///
    /// A new tensor with the given shape and elements sampled from the given distribution.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape, Distribution};
    ///
    /// fn example<B: Backend>() {
    ///   let device = B::Device::default();
    ///   let distribution = Distribution::Uniform(0.0, 1.0); // Any random value between 0.0 and 1.0
    ///   let tensor = Tensor::<B, 2>::random(Shape::new([2, 3]), distribution, &device);
    ///   println!("{tensor}");
    ///   // [
    ///   //   [0.08347523, 0.70498955, 0.60332155],
    ///   //   [0.08173251, 0.18028641, 0.97942924]
    ///   // ]
    /// }
    /// ```
    pub fn random<S: Into<Shape>>(
        shape: S,
        distribution: Distribution,
        device: &B::Device,
    ) -> Self {
        Self::new(K::random(shape.into(), distribution, device))
    }

    /// Sort the elements by value in ascending order along a given dimension.
    ///
    /// This sort is unstable (i.e., may reorder equal elements).
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension to sort along.
    ///
    /// # Returns
    ///
    /// A new tensor with the elements sorted in ascending order along the given dimension.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///   let device = B::Device::default();
    ///   let tensor = Tensor::<B, 2>::from_data([[12.0, -2.0, 3.0], [5.0, 3.0, 6.0]], &device);
    ///   let tensor = tensor.sort(0);
    ///   println!("{tensor}");
    ///   // [[5.0, -2.0, 3.0], [12.0, 3.0, 6.0]]
    ///   let tensor = tensor.sort(1);
    ///   println!("{tensor}");
    ///   // [[-2.0, 3.0, 12.0], [3.0, 5.0, 6.0]]
    /// }
    /// ```
    pub fn sort(self, dim: usize) -> Tensor<B, D, K> {
        check!(TensorCheck::sort_dim::<D>("Sort", dim));
        Tensor::new(K::sort(self.primitive, dim, /*descending*/ false))
    }

    /// Sort the elements by value in descending order along a given dimension.
    ///
    /// This sort is unstable (i.e., may reorder equal elements).
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension to sort along.
    ///
    /// # Returns
    ///
    /// A new tensor with the elements sorted in descending order along the given dimension.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::from_data([[12.0, -2.0, 3.0], [5.0, 3.0, 6.0]], &device);
    ///    let tensor = tensor.sort_descending(0);
    ///    println!("{tensor}");
    ///    // [[12.0, 3.0, 6.0], [5.0, -2.0, 3.0]]
    ///    let tensor = tensor.sort_descending(1);
    ///    println!("{tensor}");
    ///    // [[12.0, 3.0, -2.0], [6.0, 5.0, 3.0]]
    /// }
    /// ```
    pub fn sort_descending(self, dim: usize) -> Tensor<B, D, K> {
        check!(TensorCheck::sort_dim::<D>("Sort", dim));
        Tensor::new(K::sort(self.primitive, dim, /*descending*/ true))
    }

    /// Sort the elements by value in ascending order along a given dimension.
    /// Also returns the indices.
    ///
    /// This sort is unstable (i.e., may reorder equal elements).
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension to sort along.
    ///
    /// # Returns
    ///
    /// A tuple containing the sorted tensor and the indices tensor.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///   let device = B::Device::default();
    ///   let tensor = Tensor::<B, 2>::from_data([[12.0, -2.0, 3.0], [5.0, 3.0, 6.0]], &device);
    ///   let (tensor, indices) = tensor.sort_with_indices(0);
    ///   println!("{tensor}");
    ///   // [[5.0, -2.0, 3.0], [12.0, 3.0, 6.0]]
    ///   println!("{}", indices);
    ///   // [[1, 0, 0], [0, 1, 1]]
    /// }
    /// ```
    pub fn sort_with_indices(self, dim: usize) -> (Tensor<B, D, K>, Tensor<B, D, Int>) {
        check!(TensorCheck::sort_dim::<D>("Sort_with_indices", dim));
        let (values, indices) =
            K::sort_with_indices(self.primitive, dim, /*descending*/ false);
        (Tensor::new(values), Tensor::new(indices))
    }

    /// Sort the elements by value in descending order along a given dimension.
    /// Also returns the indices.
    ///
    /// This sort is unstable (i.e., may reorder equal elements).
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension to sort along.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::from_data([[12.0, -2.0, 3.0], [5.0, 3.0, 6.0]], &device);
    ///    let (tensor, indices) = tensor.sort_descending_with_indices(0);
    ///    println!("{tensor}");
    ///    // [[12.0, 3.0, 6.0], [5.0, -2.0, 3.0]]
    ///    println!("{}", indices);
    ///    // [[0, 1, 1], [1, 0, 0]]
    /// }
    /// ```
    pub fn sort_descending_with_indices(self, dim: usize) -> (Tensor<B, D, K>, Tensor<B, D, Int>) {
        check!(TensorCheck::sort_dim::<D>("Sort_with_indices", dim));
        let (values, indices) = K::sort_with_indices(self.primitive, dim, /*descending*/ true);
        (Tensor::new(values), Tensor::new(indices))
    }

    /// Returns the indices that sort the elements by value in ascending order along a given dimension.
    ///
    /// This sort is unstable (i.e., may reorder equal elements).
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension to sort along.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::from_data([[12.0, -2.0, 3.0], [5.0, 3.0, 6.0]], &device);
    ///    let tensor = tensor.argsort(0);
    ///    println!("{tensor}");
    ///    // [[1, 0, 0], [0, 1, 1]]
    /// }
    /// ```
    pub fn argsort(self, dim: usize) -> Tensor<B, D, Int> {
        check!(TensorCheck::sort_dim::<D>("Argsort", dim));
        Tensor::new(K::argsort(self.primitive, dim, /*descending*/ false))
    }

    /// Returns the indices that sort the elements by value in descending order along a given dimension.
    ///
    /// This sort is unstable (i.e., may reorder equal elements).
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension to sort along.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::from_data([[12.0, -2.0, 3.0], [5.0, 3.0, 6.0]], &device);
    ///    let tensor = tensor.argsort_descending(0);
    ///    println!("{tensor}");
    ///    // [[0, 1, 1], [1, 0, 0]]
    ///    let tensor = tensor.argsort_descending(1);
    ///    println!("{tensor}");
    ///    // [[0, 2, 1], [2, 0, 1]]
    /// }
    /// ```
    pub fn argsort_descending(self, dim: usize) -> Tensor<B, D, Int> {
        check!(TensorCheck::sort_dim::<D>("Argsort", dim));
        Tensor::new(K::argsort(self.primitive, dim, /*descending*/ true))
    }

    /// Returns the `k` largest elements of the given input tensor along a given dimension.
    ///
    /// # Arguments
    ///
    /// * `k` - The number of elements to return.
    ///
    /// # Returns
    ///
    /// A new tensor with the `k` largest elements along the given dimension.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///   let device = B::Device::default();
    ///   let tensor = Tensor::<B, 2>::from_data([[12.0, -2.0, 3.0], [5.0, 3.0, 6.0]], &device);
    ///   let tensor = tensor.topk(2, 0);
    ///   println!("{tensor}");
    ///   // [[12.0, 3.0, 6.0], [5.0, -2.0, 3.0]]
    ///   let tensor = tensor.topk(1, 1);
    ///   println!("{tensor}");   
    ///   // [[12.0], [6.0]]
    /// }
    /// ```
    pub fn topk(self, k: usize, dim: usize) -> Tensor<B, D, K> {
        let k_indices = Tensor::arange(0..k as i64, &self.device());
        self.sort_descending(dim).select(dim, k_indices)
    }

    /// Returns the `k` largest elements of the given input tensor along a given dimension.
    /// Also returns the indices.
    ///
    /// # Arguments
    ///
    /// * `k` - The number of elements to return.
    /// * `dim` - The dimension to sort along.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::from_data([[12.0, -2.0, 3.0], [5.0, 3.0, 6.0]], &device);
    ///    let (tensor, indices) = tensor.topk_with_indices(2, 0);
    ///    println!("{tensor}");
    ///    // [[12.0, 3.0, 6.0], [5.0, -2.0, 3.0]]
    ///    println!("{}", indices);
    ///    // [[0, 1, 1], [1, 0, 0]]
    ///    let (tensor, indices) = tensor.topk_with_indices(1, 1);
    ///    println!("{tensor}");
    ///    // [[12.0], [6.0]]
    ///    println!("{indices}");
    ///    // [[0], [2]]
    /// }
    /// ```
    pub fn topk_with_indices(self, k: usize, dim: usize) -> (Tensor<B, D, K>, Tensor<B, D, Int>) {
        let k_indices = Tensor::arange(0..k as i64, &self.device());
        let (values, indices) = self.sort_descending_with_indices(dim);
        (
            values.select(dim, k_indices.clone()),
            indices.select(dim, k_indices),
        )
    }

    /// Pad the tensor of rank two or higher with the given value on the last two dimensions.
    ///
    /// # Arguments
    ///
    /// * `padding` - A tuple of four integers representing the padding on the left, right, top, and bottom.
    /// * `value` - The value to pad the tensor with.
    ///
    /// # Returns
    ///
    /// A new tensor with the given padding.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend<FloatElem: From<f32>>>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::from_data([[12.0, -2.0, 3.0], [5.0, 3.0, 6.0]], &device);
    ///    let tensor = tensor.pad((1, 1, 1, 1), 0.0);
    ///    println!("{tensor}");
    ///    // [
    ///    //   [0.0, 0.0, 0.0, 0.0, 0.0],
    ///    //   [0.0, 12.0, -2.0, 3.0, 0.0],
    ///    //   [0.0, 5.0, 3.0, 6.0, 0.0],
    ///    //   [0.0, 0.0, 0.0, 0.0, 0.0]
    ///    // ]
    /// }
    /// ```
    pub fn pad<E: ElementConversion>(
        self,
        padding: (usize, usize, usize, usize),
        value: E,
    ) -> Tensor<B, D, K> {
        let (left, right, top, bottom) = padding;

        let mut padded_dims: [usize; D] = self.dims();

        // Update the last two dimensions with padding
        padded_dims[D - 2] += top + bottom;
        padded_dims[D - 1] += left + right;

        // Create the ranges for the padded tensor
        let ranges: [core::ops::Range<usize>; D] = padded_dims
            .iter()
            .enumerate()
            .map(|(i, &dim)| {
                if i == D - 2 {
                    top..dim - bottom
                } else if i == D - 1 {
                    left..dim - right
                } else {
                    0..dim
                }
            })
            .collect::<Vec<core::ops::Range<usize>>>()
            .try_into()
            .unwrap();

        // Create the padded tensor
        let padded_tensor = Tensor::full(padded_dims, value, &self.device());

        // Assign the original tensor data to the appropriate slice of the padded tensor
        padded_tensor.slice_assign(ranges, self)
    }
    /// Create a one hot tensor.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>(){
    ///     let device = Default::default();
    ///     let indices: Tensor<B, 1> = Tensor::from_floats([0.0, 1.0, 2.0, 3.0], &device);
    ///     let one_hot: Tensor<B, 2> = indices.one_hot(4);
    ///     println!("{}", one_hot.to_data());
    ///     // [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    /// }
    /// ```
    pub fn one_hot<const D2: usize>(self, num_classes: usize) -> Tensor<B, D2, K> {
        check!(TensorCheck::one_hot_tensor(self.clone(), num_classes));
        self.one_hot_fill(num_classes, 1.0, 0.0, -1)
    }

    /// Create a one-hot encoded tensor with configurable `num_classes`, `on_value`, `off_value`, and `axis` including high-ranked tensors.
    ///
    /// # Arguments
    ///
    /// * `num_classes`: The number of classes for the one-hot encoding, which defines the size of the one-hot dimension.
    /// * `on_value`: The value to assign for active positions (corresponding to indices).
    /// * `off_value`: The value to assign for inactive positions.
    /// * `axis`: The axis along which the one-hot dimension is added. Supports negative indexing.
    ///
    /// # Returns
    ///
    /// A tensor with one additional dimension for the one-hot encoding, where active positions are filled with `on_value` and others with `off_value`.
    ///
    /// # Example
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Float};
    /// fn example<B: Backend<FloatElem: From<f32>>>() {
    ///     let device = B::Device::default();
    ///     let indices: Tensor<B, 2, Float> = Tensor::from_floats([[0., 2.], [1., -1.]], &device);
    ///     // One-hot encoding
    ///     let tensor:Tensor<B, 3, Float> = indices.one_hot_fill(3, 5.0.into(), 0.0.into(), -1);
    ///     println!("{tensor}");
    ///     // [[[5.0, 0.0, 0.0],
    ///     // [0.0, 0.0, 5.0]],
    ///     // [[0.0, 5.0, 0.0],
    ///     // [0.0, 0.0, 5.0]]]
    /// }
    /// ```
    pub fn one_hot_fill<const D2: usize>(
        self,
        num_classes: usize,
        on_value: f32,
        off_value: f32,
        axis: i64,
    ) -> Tensor<B, D2, K> {
        check!(TensorCheck::one_hot_tensor_rank::<D, D2>());
        // Initialize shape from the current tensor dimensions and prepare for modification
        let mut shape = self.shape().dims::<D>().to_vec();
        let device = self.device();
        let rank = self.dims().len();

        // Adjust negative axis to a positive index
        let axis = if axis < 0 {
            axis + rank as i64 + 1
        } else {
            axis
        };

        // Ensure axis is within valid range
        if axis < 0 || axis > rank as i64 {
            panic!("Axis out of range. Accepted range is [-r-1, r] where r = rank(indices).");
        }
        // Convert the input tensor to integer indices
        let indices: Tensor<B, D, Int> =
            Tensor::from_data(self.to_data().convert::<i64>(), &device);
        // Insert the new dimension for the one-hot representation
        shape.insert(axis as usize, num_classes);
        // Adjust indices to valid range and handle invalid indices
        let adjusted_indices = indices
            .clone()
            .mask_fill(self.clone().lower_elem(0), num_classes as i64) // Handle negative indices
            .add(indices.clone().mask_fill(self.clone().greater_elem(0), 0)); // Handle positive indices
                                                                              // Unsqueeze the indices tensor along the specified axis
        let indices_unsqueezed: Tensor<B, D2, Int> = adjusted_indices.unsqueeze_dim(axis as usize);

        // Initialize the output tensor with the off_value
        let output = Tensor::full(shape.clone(), off_value, &device);

        // Prepare scatter tensor for on_value and off_value adjustments
        let scatter_on_values = Tensor::full(indices_unsqueezed.shape(), on_value, &device)
            - Tensor::full(indices_unsqueezed.shape(), off_value, &self.device());

        // Scatter on_value at the appropriate indices to create the one-hot representation
        output.scatter(axis as usize, indices_unsqueezed, scatter_on_values)
    }

    /// Returns a new tensor with boolean elements indicating whether each element of the input is NaN.
    ///
    /// # Returns
    ///
    /// A boolean tensor where `true` indicates NaN and `false` indicates a non-NaN value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Bool, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::from_data([[1.0, f64::NAN, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let tensor = tensor.is_nan();
    ///    println!("{tensor}");
    ///    // [[false, true, false], [false, false, false]]
    /// }
    /// ```
    pub fn is_nan(&self) -> Tensor<B, D, Bool> {
        // Check if the input tensor is NaN by comparing it to itself
        // NaN is the only value that is not equal to itself
        Tensor::new(K::not_equal(self.primitive.clone(), self.primitive.clone()))
    }

    /// Checks if the tensor contains any NaN values.
    ///
    /// # Returns
    ///
    /// A boolean tensor with a single element indicating whether the tensor contains any NaN values.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Bool, Shape};
    ///
    /// fn example<B: Backend>() {
    ///   let device = B::Device::default();
    ///   let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [f64::NAN, 9.0, 6.0]], &device);
    ///   let tensor = tensor.contains_nan();
    ///   println!("{tensor}");
    ///   // [true]
    ///   let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///   let tensor = tensor.contains_nan();
    ///   println!("{tensor}");
    ///   // [false]
    /// }
    /// ```
    pub fn contains_nan(&self) -> Tensor<B, 1, Bool> {
        // Summing the tensor will result in NaN if the tensor contains any NaN values
        // This is faster than checking each element individually
        // because it rolls up the NaN values into a single value
        let sum = K::sum(self.primitive.clone());

        // Check if the sum is NaN by comparing it to itself
        Tensor::new(K::not_equal(sum.clone(), sum))
    }
}

impl<B, K> Tensor<B, 2, K>
where
    B: Backend,
    K: Numeric<B>,
    K::Elem: Element,
{
    /// Creates a new 2D tensor with ones on the diagonal and zeros elsewhere.
    ///
    /// # Arguments
    ///
    /// * `size` - The size of the square matrix.
    pub fn eye(size: usize, device: &B::Device) -> Self {
        let indices = Tensor::<B, 1, Int>::arange(0..size as i64, device).unsqueeze::<2>();
        let ones = K::ones([1, size].into(), device);
        let zeros = K::zeros([size, size].into(), device);

        Self::new(K::scatter(0, zeros, indices.primitive, ones))
    }
}

/// Trait that list all operations that can be applied on all numerical tensors.
///
/// # Warnings
///
/// This is an internal trait, use the public API provided by [tensor struct](Tensor).
pub trait Numeric<B: Backend>: BasicOps<B>
where
    Self::Elem: Element,
{
    /// Adds two tensors together.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The sum of the two tensors.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For adding tensors, users should prefer the [Tensor::add](Tensor::add) function,
    /// which is more high-level and designed for public use.
    fn add(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive;

    /// Adds a scalar to a tensor element-wise.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// The sum of the tensor and the scalar.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For adding a scalar to a tensor, users should prefer the [Tensor::add_scalar](Tensor::add_scalar) function,
    /// which is more high-level and designed for public use.
    fn add_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive;

    /// Subtracts two tensors.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The difference of the two tensors.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For subtracting tensors, users should prefer the [Tensor::sub](Tensor::sub) function,
    /// which is more high-level and designed for public use.
    fn sub(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive;

    /// Subtracts a scalar from a tensor element-wise.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// The difference of the tensor and the scalar.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For subtracting a scalar from a tensor, users should prefer the [Tensor::sub_scalar](Tensor::sub_scalar) function,
    /// which is more high-level and designed for public use.
    fn sub_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive;

    /// Divides two tensors.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The quotient of the two tensors.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For dividing tensors, users should prefer the [Tensor::div](Tensor::div) function,
    /// which is more high-level and designed for public use.
    fn div(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive;

    /// Divides a tensor by a scalar element-wise.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// The quotient of the tensor and the scalar.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For dividing a tensor by a scalar, users should prefer the [Tensor::div_scalar](Tensor::div_scalar) function,
    /// which is more high-level and designed for public use.
    fn div_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive;

    /// Computes the modulo element-wise. The result is the *signed* remainder of the division and its absolute value is
    /// less than that of the divisor.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The dividend.
    /// * `rhs` - The divisor.
    ///
    /// # Returns
    ///
    /// The modulo of the input tensor with the divisor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For performing the modulo operation, users should prefer the [Tensor::remainder](Tensor::remainder) function,
    /// which is more high-level and designed for public use.
    fn remainder(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive;

    /// Computes the modulo element-wise. The result is the *signed* remainder of the division and its absolute value is
    /// less than that of the divisor.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The dividend.
    /// * `rhs` - The divisor.
    ///
    /// # Returns
    ///
    /// The modulo of the input tensor with the divisor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For performing the modulo operation, users should prefer the [Tensor::remainder_scalar](Tensor::remainder_scalar) function,
    /// which is more high-level and designed for public use.
    fn remainder_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive;

    /// Multiplies two tensors.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The product of the two tensors.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For multiplying tensors, users should prefer the [Tensor::mul](Tensor::mul) function,
    /// which is more high-level and designed for public use.
    fn mul(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive;

    /// Multiplies a tensor by a scalar element-wise.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// The product of the tensor and the scalar.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For multiplying a tensor by a scalar, users should prefer the [Tensor::mul_scalar](Tensor::mul_scalar) function,
    /// which is more high-level and designed for public use.
    fn mul_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive;

    /// Negates a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to negate.
    ///
    /// # Returns
    ///
    /// The negated tensor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For negating a tensor, users should prefer the [Tensor::neg](Tensor::neg) function,
    /// which is more high-level and designed for public use.
    fn neg(tensor: Self::Primitive) -> Self::Primitive;

    /// Returns the signs of the elements of a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The signs of the elements of the tensor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For getting the signs of the elements of a tensor, users should prefer the [Tensor::sign](Tensor::sign) function,
    /// which is more high-level and designed for public use.
    fn sign(tensor: Self::Primitive) -> Self::Primitive;

    /// Creates a tensor filled with zeros.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `device` - The device on which the tensor will be allocated.
    ///
    /// # Returns
    ///
    /// The tensor filled with zeros.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For creating a tensor filled with zeros, users should prefer the [Tensor::zeros](Tensor::zeros) function,
    /// which is more high-level and designed for public use.
    fn zeros(shape: Shape, device: &B::Device) -> Self::Primitive;

    /// Creates a tensor filled with ones.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `device` - The device on which the tensor will be allocated.
    ///
    /// # Returns
    ///
    /// The tensor filled with ones.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For creating a tensor filled with ones, users should prefer the [Tensor::ones](Tensor::ones) function,
    /// which is more high-level and designed for public use.
    fn ones(shape: Shape, device: &B::Device) -> Self::Primitive;

    /// Creates a tensor filled with elements equal to the given value.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `fill_value` - The value with which to fill the tensor
    /// * `device` - The device on which the tensor will be allocated.
    ///
    /// # Returns
    ///
    /// The tensor filled with elements equal to the given value
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For creating a tensor filled with a specific value, users should prefer the [Tensor::full](Tensor::full) function,
    /// which is more high-level and designed for public use.
    fn full<E: ElementConversion>(
        shape: Shape,
        fill_value: E,
        device: &B::Device,
    ) -> Self::Primitive;

    /// Sums all the elements of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to sum.
    ///
    /// # Returns
    ///
    /// The sum of all the elements of the tensor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For summing all the elements of a tensor, users should prefer the [Tensor::sum](Tensor::sum) function,
    /// which is more high-level and designed for public use.
    fn sum(tensor: Self::Primitive) -> Self::Primitive;

    /// Sums all the elements of the tensor along a dimension.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to sum.
    /// * `dim` - The dimension along which to sum.
    ///
    /// # Returns
    ///
    /// The sum of all the elements of the tensor along the specified dimension.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For summing all the elements of a tensor along a dimension, users should prefer the [Tensor::sum_dim](Tensor::sum_dim) function,
    /// which is more high-level and designed for public use.
    fn sum_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive;

    /// Computes the product of all the elements of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to compute the product of.
    ///
    /// # Returns
    ///
    /// The product of all the elements of the tensor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For computing the product of all the elements of a tensor, users should prefer the
    /// [Tensor::prod](Tensor::prod) function,
    /// which is more high-level and designed for public use.
    fn prod(tensor: Self::Primitive) -> Self::Primitive;

    /// Computes the product of all the elements of the tensor along a dimension.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to compute the product of.
    /// * `dim` - The dimension along which to compute the product.
    ///
    /// # Returns
    ///
    /// The product of all the elements of the tensor along the specified dimension.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For computing the product of all the elements of a tensor along a dimension, users should
    /// prefer the [Tensor::prod_dim](Tensor::prod_dim) function,
    /// which is more high-level and designed for public use.
    ///
    ///
    fn prod_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive;

    /// Computes the mean of all the elements of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to compute the mean of.
    ///
    /// # Returns
    ///
    /// The mean of all the elements of the tensor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For computing the mean of all the elements of a tensor, users should prefer the [Tensor::mean](Tensor::mean) function,
    /// which is more high-level and designed for public use.
    fn mean(tensor: Self::Primitive) -> Self::Primitive;

    /// Computes the mean of all the elements of the tensor along a dimension.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to compute the mean of.
    /// * `dim` - The dimension along which to compute the mean.
    ///
    /// # Returns
    ///
    /// The mean of all the elements of the tensor along the specified dimension.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For computing the mean of all the elements of a tensor along a dimension, users should prefer
    /// the [Tensor::mean_dim](Tensor::mean_dim) function, which is more high-level and designed for public use.
    fn mean_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive;

    /// Element-wise equality between two tensors.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the same shape as the input tensors, where each element is true if the
    /// corresponding elements of the input tensors are equal, and false otherwise.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For element-wise equality between two tensors, users should prefer the [Tensor::equal_elem](Tensor::equal_elem)
    /// function, which is more high-level and designed for public use.
    fn equal_elem(lhs: Self::Primitive, rhs: Self::Elem) -> B::BoolTensorPrimitive;

    /// Element-wise non-equality between two tensors.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the same shape as the input tensors, where each element is true if the
    /// corresponding elements of the input tensors are equal, and false otherwise.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For element-wise non-equality between two tensors, users should prefer the [Tensor::not_equal_elem](Tensor::not_equal_elem)
    /// function, which is more high-level and designed for public use.
    fn not_equal_elem(lhs: Self::Primitive, rhs: Self::Elem) -> B::BoolTensorPrimitive;

    /// Element-wise greater than comparison between two tensors.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the same shape as the input tensors, where each element is true if the
    /// corresponding element of the left hand side tensor is greater than the corresponding element
    /// of the right hand side tensor, and false otherwise.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For element-wise greater than comparison between two tensors, users should prefer the [Tensor::greater](Tensor::greater) function,
    /// which is more high-level and designed for public use.
    fn greater(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive;

    /// Element-wise greater than comparison between a tensor and a scalar.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the same shape as the input tensor, where each element is true if the
    /// corresponding element of the left hand side tensor is greater than the right hand side
    /// scalar, and false otherwise.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For element-wise greater than comparison between a tensor and a scalar, users should prefer
    /// the [Tensor::greater_elem](Tensor::greater_elem) function, which is more high-level and designed for public use.
    fn greater_elem(lhs: Self::Primitive, rhs: Self::Elem) -> B::BoolTensorPrimitive;

    /// Element-wise greater than or equal comparison between two tensors.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the same shape as the input tensors, where each element is true if the
    /// corresponding element of the left hand side tensor is greater than or equal to the
    /// corresponding element of the right hand side tensor, and false otherwise.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For element-wise greater than or equal comparison between two tensors, users should prefer
    /// the [Tensor::greater_equal](Tensor::greater_equal) function, which is more high-level and designed for public use.
    fn greater_equal(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive;

    /// Element-wise greater than or equal comparison between a tensor and a scalar.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the same shape as the input tensor, where each element is true if the
    /// corresponding element of the left hand side tensor is greater than or equal to the right
    /// hand side scalar, and false otherwise.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For element-wise greater than or equal comparison between a tensor and a scalar, users should prefer
    /// the [Tensor::greater_equal_elem](Tensor::greater_equal_elem) function, which is more high-level and designed for public use.
    fn greater_equal_elem(lhs: Self::Primitive, rhs: Self::Elem) -> B::BoolTensorPrimitive;

    /// Element-wise less than comparison between two tensors.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the same shape as the input tensors, where each element is true if the
    /// corresponding element of the left hand side tensor is less than the corresponding element of
    /// the right hand side tensor, and false otherwise.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For element-wise less than comparison between two tensors, users should prefer the [Tensor::lower](Tensor::lower) function,
    /// which is more high-level and designed for public use.
    fn lower(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive;

    /// Element-wise less than comparison between a tensor and a scalar.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the same shape as the input tensor, where each element is true if the
    /// corresponding element of the left hand side tensor is less than the right hand side scalar,
    /// and false otherwise.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For element-wise less than comparison between a tensor and a scalar, users should prefer
    /// the [Tensor::lower_elem](Tensor::lower_elem) function, which is more high-level and designed for public use.
    fn lower_elem(lhs: Self::Primitive, rhs: Self::Elem) -> B::BoolTensorPrimitive;

    /// Element-wise less than or equal comparison between two tensors.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the same shape as the input tensors, where each element is true if the
    /// corresponding element of the left hand side tensor is less than or equal to the corresponding
    /// element of the right hand side tensor, and false otherwise.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For element-wise less than or equal comparison between two tensors, users should prefer
    /// the [Tensor::lower_equal](Tensor::lower_equal) function, which is more high-level and designed for public use.
    fn lower_equal(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive;

    /// Element-wise less than or equal comparison between a tensor and a scalar.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the same shape as the input tensor, where each element is true if the
    /// corresponding element of the left hand side tensor is less than or equal to the right hand
    /// side scalar, and false otherwise.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For element-wise less than or equal comparison between a tensor and a scalar, users should prefer
    /// the [Tensor::lower_equal_elem](Tensor::lower_equal_elem) function, which is more high-level and designed for public use.
    fn lower_equal_elem(lhs: Self::Primitive, rhs: Self::Elem) -> B::BoolTensorPrimitive;

    /// Selects elements from a tensor based on a boolean mask.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to select elements from if the corresponding element of the mask is true.
    /// * `mask` - The boolean mask to use for selecting elements.
    /// * `source` - The tensor to select elements from when the corresponding element of the mask is false.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensors, where each element is taken from the
    /// corresponding element of the left hand side tensor if the corresponding element of the mask
    /// is true, and from the corresponding element of the right hand side tensor otherwise.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For selecting elements from a tensor based on a boolean mask, users should prefer the
    /// [Tensor::mask_where](Tensor::mask_where) function, which is more high-level and designed for public use.
    fn mask_where(
        tensor: Self::Primitive,
        mask: B::BoolTensorPrimitive,
        source: Self::Primitive,
    ) -> Self::Primitive;

    /// Fills elements of a tensor based on a boolean mask.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor where will be overwritten with the value
    ///              when the corresponding element of the mask is true.
    /// * `mask` - The boolean mask to use for filling elements.
    /// * `value` - The value to fill elements with when the corresponding element of the mask is true.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensors, where each element is taken from the
    /// corresponding element unmodified if the corresponding element of the mask is false, and
    /// filled with the value otherwise.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For filling elements of a tensor based on a boolean mask, users should prefer the
    /// [Tensor::mask_fill](Tensor::mask_fill) function, which is more high-level and designed for public use.
    fn mask_fill(
        tensor: Self::Primitive,
        mask: B::BoolTensorPrimitive,
        value: Self::Elem,
    ) -> Self::Primitive;

    /// Gathers elements from a tensor along an axis.
    ///
    /// # Arguments
    ///
    /// * `dim` - The axis along which to gather elements.
    /// * `tensor` - The tensor to gather elements from.
    /// * `indices` - The indices of the elements to gather.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensor, where each element is taken from the
    /// corresponding element of the input tensor at the corresponding index along the specified axis.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For gathering elements from a tensor along an axis, users should prefer the
    /// [Tensor::gather](Tensor::gather) function, which is more high-level and designed for public use.
    fn gather(
        dim: usize,
        tensor: Self::Primitive,
        indices: B::IntTensorPrimitive,
    ) -> Self::Primitive;

    /// Scatters elements into a tensor along an axis.
    ///
    /// # Arguments
    ///
    /// * `dim` - The axis along which to scatter elements.
    /// * `tensor` - The tensor to scatter elements into.
    /// * `indices` - The indices of the elements to scatter.
    /// * `values` - The values to scatter into the tensor.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensor, where each element is taken from the
    /// corresponding element of the input tensor at the corresponding index along the specified axis,
    /// except for the elements at the specified indices, which are taken from the corresponding
    /// element of the values tensor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For scattering elements into a tensor along an axis, users should prefer the [Tensor::scatter](Tensor::scatter) function,
    /// which is more high-level and designed for public use.
    fn scatter(
        dim: usize,
        tensor: Self::Primitive,
        indices: B::IntTensorPrimitive,
        values: Self::Primitive,
    ) -> Self::Primitive;

    /// Select tensor elements along the given dimension corresponding for the given indices.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to select elements from.
    /// * `dim` - The axis along which to select elements.
    /// * `indices` - The indices of the elements to select.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensor, where each element is taken from the
    /// corresponding element of the input tensor at the corresponding index along the specified axis.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For selecting elements from a tensor along an axis, users should prefer the
    /// [Tensor::select](Tensor::select) function, which is more high-level and designed for public use.
    fn select(tensor: Self::Primitive, dim: usize, indices: Tensor<B, 1, Int>) -> Self::Primitive;

    /// Assign the selected elements along the given dimension corresponding to the given indices
    /// from the value tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to assign elements to.
    /// * `dim` - The axis along which to assign elements.
    /// * `indices` - The indices of the elements to assign.
    /// * `values` - The values to assign to the tensor.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensor, where each element is taken from the
    /// corresponding element of the input tensor at the corresponding index along the specified axis,
    /// except for the elements at the specified indices, which are taken from the corresponding
    /// element of the values tensor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For assigning elements to a tensor along an axis, users should prefer the
    /// [Tensor::select_assign](Tensor::select_assign) function, which is more high-level and designed for public use.
    fn select_assign(
        tensor: Self::Primitive,
        dim: usize,
        indices: Tensor<B, 1, Int>,
        values: Self::Primitive,
    ) -> Self::Primitive;

    /// Gets the indices of the maximum elements of a tensor along an axis.
    ///
    /// # Arguments
    ///
    /// * `dim` - The axis along which to get the indices of the maximum elements.
    /// * `tensor` - The tensor to get the indices of the maximum elements from.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensor, where each element is the index of the
    /// maximum element of the input tensor at the corresponding index along the specified axis.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For getting the indices of the maximum elements of a tensor along an axis, users should prefer the
    /// [Tensor::argmax](Tensor::argmax) function, which is more high-level and designed for public use.
    fn argmax(tensor: Self::Primitive, dim: usize) -> B::IntTensorPrimitive;

    /// Gets the indices of the minimum elements of a tensor along an axis.
    ///
    /// # Arguments
    ///
    /// * `dim` - The axis along which to get the indices of the minimum elements.
    /// * `tensor` - The tensor to get the indices of the minimum elements from.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensor, where each element is the index of the
    /// minimum element of the input tensor at the corresponding index along the specified axis.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For getting the indices of the minimum elements of a tensor along an axis, users should prefer the
    /// [Tensor::argmin](Tensor::argmin) function, which is more high-level and designed for public use.
    fn argmin(tensor: Self::Primitive, dim: usize) -> B::IntTensorPrimitive;

    /// Gets the maximum elements of a tensor along an axis.
    ///
    /// # Arguments
    ///
    /// * `dim` - The axis along which to get the maximum elements.
    ///
    /// # Returns
    ///
    /// A single-element tensor containing the maximum element of the input tensor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For getting the maximum elements of a tensor along an axis, users should prefer the
    /// [Tensor::max](Tensor::max) function, which is more high-level and designed for public use.
    fn max(tensor: Self::Primitive) -> Self::Primitive;

    /// Gets the maximum elements of a tensor along an axis.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the maximum elements from.
    /// * `dim` - The axis along which to get the maximum elements.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensor, where each element is the maximum element
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For getting the maximum elements of a tensor along an axis, users should prefer the
    /// [Tensor::max_dim](Tensor::max_dim) function, which is more high-level and designed for public use.
    fn max_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive;

    /// Gets the maximum elements of a tensor along an axis.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the maximum elements from.
    /// * `dim` - The axis along which to get the maximum elements.
    ///
    /// # Returns
    ///
    /// A tuple containing the maximum element of the input tensor, and a tensor with the same shape
    /// as the input tensor, where each element is the index of the maximum element of the input tensor
    /// at the corresponding index along the specified axis.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For getting the maximum elements of a tensor along an axis, users should prefer the
    /// [Tensor::max_dim_with_indices](Tensor::max_dim_with_indices) function, which is more high-level and designed for public use.
    fn max_dim_with_indices(
        tensor: Self::Primitive,
        dim: usize,
    ) -> (Self::Primitive, B::IntTensorPrimitive);

    /// Gets the minimum elements of a tensor along an axis.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the minimum elements from.
    ///
    /// # Returns
    ///
    /// A single-element tensor containing the minimum element of the input tensor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For getting the minimum elements of a tensor along an axis, users should prefer the
    /// [Tensor::min](Tensor::min) function, which is more high-level and designed for public use.
    fn min(tensor: Self::Primitive) -> Self::Primitive;

    /// Gets the minimum elements of a tensor along an axis.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the minimum elements from.
    /// * `dim` - The axis along which to get the minimum elements.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensor, where each element is the minimum element
    /// of the input tensor at the corresponding index along the specified axis.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For getting the minimum elements of a tensor along an axis, users should prefer the
    /// [Tensor::min_dim](Tensor::min_dim) function, which is more high-level and designed for public use.
    fn min_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive;

    /// Gets the minimum elements and indices of a tensor along an axis.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the minimum elements from.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensor and corresponding indices, where
    /// each element is the minimum element of the input tensor at the corresponding index
    /// along the specified axis.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For getting the minimum elements of a tensor along an axis, users should prefer the
    /// [Tensor::min_dim_with_indices](Tensor::min_dim_with_indices) function, which is more high-level and designed for public use.
    fn min_dim_with_indices(
        tensor: Self::Primitive,
        dim: usize,
    ) -> (Self::Primitive, B::IntTensorPrimitive);

    /// Clamp the tensor between the given min and max values.
    ///
    /// # Arguments
    ///
    /// * `min` - The minimum value.
    /// * `max` - The maximum value.
    ///
    /// # Returns
    ///
    /// A new tensor with the values clamped between the given min and max values.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users.
    ///
    /// For clamping a tensor between the given min and max values, users should prefer the
    /// [Tensor::clamp](Tensor::clamp) function, which is more high-level and designed for public use.
    fn clamp(tensor: Self::Primitive, min: Self::Elem, max: Self::Elem) -> Self::Primitive;

    /// Clamps a tensor under a minimum value.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to clamp.
    /// * `min` - The minimum value.
    ///
    /// # Returns
    ///
    /// A new tensor with the values clamped under the given min value.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users.
    ///
    /// For clamping a tensor under a minimum value, users should prefer the
    /// [Tensor::clamp_min](Tensor::clamp_min) function, which is more high-level and designed for public use.
    fn clamp_min(tensor: Self::Primitive, min: Self::Elem) -> Self::Primitive;

    /// Clamps a tensor over a maximum value.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to clamp.
    /// * `max` - The maximum value.
    ///
    /// # Returns
    ///
    /// A new tensor with the values clamped over the given max value.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users.
    ///
    /// For clamping a tensor over a maximum value, users should prefer the
    /// [Tensor::clamp_max](Tensor::clamp_max) function, which is more high-level and designed for public use.
    fn clamp_max(tensor: Self::Primitive, max: Self::Elem) -> Self::Primitive;

    /// Calculate absolute value on all elements of a tensor
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to apply abs to.
    ///
    /// # Returns
    ///
    /// A tensor with absolute values.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For calculating abs of the elements of a tensor, users should prefer the [Tensor::abs](Tensor::abs) function,
    /// which is more high-level and designed for public use.
    fn abs(tensor: Self::Primitive) -> Self::Primitive;

    /// Element-wise power of a tensor to a float tensor
    ///
    /// # Arguments
    /// * `tensor` - The tensor to apply power to.
    /// * `power` - The power to apply to the tensor.
    fn powf(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive;

    /// Element-wise power of a tensor
    ///
    /// # Arguments
    /// * `tensor` - The tensor to apply power to.
    /// * `power` - The power to apply to the tensor.
    fn powi(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive;

    /// Element-wise power of a tensor to a scalar float
    ///
    /// # Arguments
    /// * `tensor` - The tensor to apply power to.
    /// * `power` - The power to apply to the tensor.
    fn powf_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive;

    /// Element-wise power of a tensor to a scalar int
    ///
    /// # Arguments
    /// * `tensor` - The tensor to apply power to.
    /// * `power` - The power to apply to the tensor.
    fn powi_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive;

    /// Create a random tensor.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the output tensor.
    /// * `distribution` - The distribution used to sample.
    /// * `device` - The device to use.
    ///
    /// # Returns
    ///
    /// A new tensor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// Users should prefer the [Tensor::random](Tensor::random) function,
    /// which is more high-level and designed for public use.
    fn random(shape: Shape, distribution: Distribution, device: &B::Device) -> Self::Primitive;

    /// Sort the elements of the input `tensor` by value along a given dimension.
    ///
    /// This sort is unstable (i.e., may reorder equal elements).
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    /// * `dim` - The axis along which to sort.
    /// * `descending` - The sorting order.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensor, where the elements are sorted by value.
    ///
    /// # Remarks
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// Users should prefer the [Tensor::sort](Tensor::sort) function,
    /// which is more high-level and designed for public use.
    fn sort(tensor: Self::Primitive, dim: usize, descending: bool) -> Self::Primitive;

    /// Sort the elements of the input `tensor` by value along a given dimension.
    ///
    /// This sort is unstable (i.e., may reorder equal elements).
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    /// * `dim` - The axis along which to sort.
    /// * `descending` - The sorting order.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensor and corresponding indices, where
    /// the elements are sorted by value and the indices map back to the original input tensor.
    ///
    /// # Remarks
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For sorting the elements of a tensor, users should prefer the
    /// [Tensor::sort_with_indices](Tensor::sort_with_indices) function, which is more high-level
    /// and designed for public use.
    fn sort_with_indices(
        tensor: Self::Primitive,
        dim: usize,
        descending: bool,
    ) -> (Self::Primitive, <Int as TensorKind<B>>::Primitive);

    /// Returns the indices that sort the elements of the input `tensor` by value along a given dimension.
    ///
    /// This sort is unstable (i.e., may reorder equal elements).
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    /// * `dim` - The axis along which to sort.
    /// * `descending` - The sorting order.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensor the indices map back to the original input tensor.
    ///
    /// # Remarks
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// Users should prefer the [Tensor::argsort](Tensor::argsort) function,
    /// which is more high-level and designed for public use.
    fn argsort(
        tensor: Self::Primitive,
        dim: usize,
        descending: bool,
    ) -> <Int as TensorKind<B>>::Primitive;
}

impl<B: Backend> Numeric<B> for Int {
    fn add(lhs: Self::Primitive, rhs: Self::Primitive) -> <Int as TensorKind<B>>::Primitive {
        B::int_add(lhs, rhs)
    }
    fn add_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive {
        B::int_add_scalar(lhs, rhs.elem())
    }
    fn sub(lhs: Self::Primitive, rhs: Self::Primitive) -> <Int as TensorKind<B>>::Primitive {
        B::int_sub(lhs, rhs)
    }
    fn sub_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive {
        B::int_sub_scalar(lhs, rhs.elem())
    }
    fn div(lhs: Self::Primitive, rhs: Self::Primitive) -> <Int as TensorKind<B>>::Primitive {
        B::int_div(lhs, rhs)
    }
    fn div_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive {
        B::int_div_scalar(lhs, rhs.elem())
    }
    fn remainder(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        B::int_remainder(lhs, rhs)
    }
    fn remainder_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive {
        B::int_remainder_scalar(lhs, rhs.elem())
    }
    fn mul(lhs: Self::Primitive, rhs: Self::Primitive) -> <Int as TensorKind<B>>::Primitive {
        B::int_mul(lhs, rhs)
    }
    fn mul_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive {
        B::int_mul_scalar(lhs, rhs.elem())
    }
    fn neg(tensor: Self::Primitive) -> Self::Primitive {
        B::int_neg(tensor)
    }
    fn zeros(shape: Shape, device: &B::Device) -> Self::Primitive {
        B::int_zeros(shape, device)
    }
    fn ones(shape: Shape, device: &B::Device) -> Self::Primitive {
        B::int_ones(shape, device)
    }
    fn full<E: ElementConversion>(
        shape: Shape,
        fill_value: E,
        device: &B::Device,
    ) -> Self::Primitive {
        B::int_full(shape, fill_value.elem(), device)
    }

    fn sum(tensor: Self::Primitive) -> Self::Primitive {
        B::int_sum(tensor)
    }

    fn sum_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        B::int_sum_dim(tensor, dim)
    }

    fn prod(tensor: Self::Primitive) -> Self::Primitive {
        B::int_prod(tensor)
    }

    fn prod_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        B::int_prod_dim(tensor, dim)
    }

    fn mean(tensor: Self::Primitive) -> Self::Primitive {
        B::int_mean(tensor)
    }
    fn mean_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        B::int_mean_dim(tensor, dim)
    }

    fn equal_elem(lhs: Self::Primitive, rhs: Self::Elem) -> B::BoolTensorPrimitive {
        B::int_equal_elem(lhs, rhs)
    }
    fn not_equal_elem(lhs: Self::Primitive, rhs: Self::Elem) -> B::BoolTensorPrimitive {
        B::int_not_equal_elem(lhs, rhs)
    }
    fn greater(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive {
        B::int_greater(lhs, rhs)
    }

    fn greater_elem(lhs: Self::Primitive, rhs: Self::Elem) -> B::BoolTensorPrimitive {
        B::int_greater_elem(lhs, rhs)
    }

    fn greater_equal(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive {
        B::int_greater_equal(lhs, rhs)
    }

    fn greater_equal_elem(lhs: Self::Primitive, rhs: Self::Elem) -> B::BoolTensorPrimitive {
        B::int_greater_equal_elem(lhs, rhs)
    }

    fn lower(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive {
        B::int_lower(lhs, rhs)
    }

    fn lower_elem(lhs: Self::Primitive, rhs: Self::Elem) -> B::BoolTensorPrimitive {
        B::int_lower_elem(lhs, rhs)
    }

    fn lower_equal(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive {
        B::int_lower_equal(lhs, rhs)
    }

    fn lower_equal_elem(lhs: Self::Primitive, rhs: Self::Elem) -> B::BoolTensorPrimitive {
        B::int_lower_equal_elem(lhs, rhs)
    }

    fn mask_where(
        tensor: Self::Primitive,
        mask: B::BoolTensorPrimitive,
        source: Self::Primitive,
    ) -> Self::Primitive {
        B::int_mask_where(tensor, mask, source)
    }

    fn mask_fill(
        tensor: Self::Primitive,
        mask: B::BoolTensorPrimitive,
        value: Self::Elem,
    ) -> Self::Primitive {
        B::int_mask_fill(tensor, mask, value)
    }

    fn select(tensor: Self::Primitive, dim: usize, indices: Tensor<B, 1, Int>) -> Self::Primitive {
        B::int_select(tensor, dim, indices.primitive)
    }

    fn select_assign(
        tensor: Self::Primitive,
        dim: usize,
        indices: Tensor<B, 1, Int>,
        values: Self::Primitive,
    ) -> Self::Primitive {
        B::int_select_assign(tensor, dim, indices.primitive, values)
    }
    fn gather(
        dim: usize,
        tensor: Self::Primitive,
        indices: B::IntTensorPrimitive,
    ) -> Self::Primitive {
        B::int_gather(dim, tensor, indices)
    }

    fn scatter(
        dim: usize,
        tensor: Self::Primitive,
        indices: B::IntTensorPrimitive,
        values: Self::Primitive,
    ) -> Self::Primitive {
        B::int_scatter(dim, tensor, indices, values)
    }

    fn argmax(tensor: Self::Primitive, dim: usize) -> IntTensor<B> {
        B::int_argmax(tensor, dim)
    }

    fn argmin(tensor: Self::Primitive, dim: usize) -> IntTensor<B> {
        B::int_argmin(tensor, dim)
    }

    fn max(tensor: Self::Primitive) -> Self::Primitive {
        B::int_max(tensor)
    }

    fn max_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        B::int_max_dim(tensor, dim)
    }

    fn max_dim_with_indices(
        tensor: Self::Primitive,
        dim: usize,
    ) -> (Self::Primitive, IntTensor<B>) {
        B::int_max_dim_with_indices(tensor, dim)
    }

    fn min(tensor: Self::Primitive) -> Self::Primitive {
        B::int_min(tensor)
    }

    fn min_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        B::int_min_dim(tensor, dim)
    }

    fn min_dim_with_indices(
        tensor: Self::Primitive,
        dim: usize,
    ) -> (Self::Primitive, IntTensor<B>) {
        B::int_min_dim_with_indices(tensor, dim)
    }

    fn clamp(tensor: Self::Primitive, min: B::IntElem, max: B::IntElem) -> Self::Primitive {
        B::int_clamp(tensor, min, max)
    }

    fn clamp_min(tensor: Self::Primitive, min: B::IntElem) -> Self::Primitive {
        B::int_clamp_min(tensor, min)
    }

    fn clamp_max(tensor: Self::Primitive, max: B::IntElem) -> Self::Primitive {
        B::int_clamp_max(tensor, max)
    }

    fn abs(tensor: Self::Primitive) -> Self::Primitive {
        B::int_abs(tensor)
    }

    fn powf(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        B::int_powf(lhs, B::int_into_float(rhs))
    }

    fn powf_scalar<E: ElementConversion>(
        lhs: Self::Primitive,
        rhs: E,
    ) -> <Int as TensorKind<B>>::Primitive {
        B::int_powf_scalar(lhs, rhs.elem())
    }

    fn powi(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        B::int_powi(lhs, rhs)
    }

    fn powi_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive {
        B::int_powf_scalar(lhs, rhs.elem())
    }

    fn random(shape: Shape, distribution: Distribution, device: &Device<B>) -> Self::Primitive {
        B::int_random(shape, distribution, device)
    }

    fn sign(tensor: Self::Primitive) -> Self::Primitive {
        B::int_sign(tensor)
    }

    fn sort(tensor: Self::Primitive, dim: usize, descending: bool) -> Self::Primitive {
        B::int_sort(tensor, dim, descending)
    }

    fn sort_with_indices(
        tensor: Self::Primitive,
        dim: usize,
        descending: bool,
    ) -> (Self::Primitive, <Int as TensorKind<B>>::Primitive) {
        B::int_sort_with_indices(tensor, dim, descending)
    }

    fn argsort(
        tensor: Self::Primitive,
        dim: usize,
        descending: bool,
    ) -> <Int as TensorKind<B>>::Primitive {
        B::int_argsort(tensor, dim, descending)
    }
}

impl<B: Backend> Numeric<B> for Float {
    fn add(lhs: Self::Primitive, rhs: Self::Primitive) -> <Float as TensorKind<B>>::Primitive {
        match (lhs, rhs) {
            (TensorPrimitive::Float(lhs), TensorPrimitive::Float(rhs)) => {
                TensorPrimitive::Float(B::float_add(lhs, rhs))
            }
            (TensorPrimitive::QFloat(lhs), TensorPrimitive::QFloat(rhs)) => {
                TensorPrimitive::QFloat(B::q_add(lhs, rhs))
            }
            _ => panic!("Primitive type mismatch for lhs and rhs"),
        }
    }
    fn add_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive {
        match lhs {
            TensorPrimitive::Float(lhs) => {
                TensorPrimitive::Float(B::float_add_scalar(lhs, rhs.elem()))
            }
            TensorPrimitive::QFloat(lhs) => {
                TensorPrimitive::QFloat(B::q_add_scalar(lhs, rhs.elem()))
            }
        }
    }
    fn sub(lhs: Self::Primitive, rhs: Self::Primitive) -> <Float as TensorKind<B>>::Primitive {
        match (lhs, rhs) {
            (TensorPrimitive::Float(lhs), TensorPrimitive::Float(rhs)) => {
                TensorPrimitive::Float(B::float_sub(lhs, rhs))
            }
            (TensorPrimitive::QFloat(lhs), TensorPrimitive::QFloat(rhs)) => {
                TensorPrimitive::QFloat(B::q_sub(lhs, rhs))
            }
            _ => panic!("Primitive type mismatch for lhs and rhs"),
        }
    }
    fn sub_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive {
        match lhs {
            TensorPrimitive::Float(lhs) => {
                TensorPrimitive::Float(B::float_sub_scalar(lhs, rhs.elem()))
            }
            TensorPrimitive::QFloat(lhs) => {
                TensorPrimitive::QFloat(B::q_sub_scalar(lhs, rhs.elem()))
            }
        }
    }
    fn div(lhs: Self::Primitive, rhs: Self::Primitive) -> <Float as TensorKind<B>>::Primitive {
        match (lhs, rhs) {
            (TensorPrimitive::Float(lhs), TensorPrimitive::Float(rhs)) => {
                TensorPrimitive::Float(B::float_div(lhs, rhs))
            }
            (TensorPrimitive::QFloat(lhs), TensorPrimitive::QFloat(rhs)) => {
                TensorPrimitive::QFloat(B::q_div(lhs, rhs))
            }
            _ => panic!("Primitive type mismatch for lhs and rhs"),
        }
    }
    fn div_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive {
        match lhs {
            TensorPrimitive::Float(lhs) => {
                TensorPrimitive::Float(B::float_div_scalar(lhs, rhs.elem()))
            }
            TensorPrimitive::QFloat(lhs) => {
                TensorPrimitive::QFloat(B::q_div_scalar(lhs, rhs.elem()))
            }
        }
    }
    fn remainder(
        lhs: Self::Primitive,
        rhs: Self::Primitive,
    ) -> <Float as TensorKind<B>>::Primitive {
        match (lhs, rhs) {
            (TensorPrimitive::Float(lhs), TensorPrimitive::Float(rhs)) => {
                TensorPrimitive::Float(B::float_remainder(lhs, rhs))
            }
            (TensorPrimitive::QFloat(lhs), TensorPrimitive::QFloat(rhs)) => {
                TensorPrimitive::QFloat(B::q_remainder(lhs, rhs))
            }
            _ => panic!("Primitive type mismatch for lhs and rhs"),
        }
    }
    fn remainder_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive {
        match lhs {
            TensorPrimitive::Float(lhs) => {
                TensorPrimitive::Float(B::float_remainder_scalar(lhs, rhs.elem()))
            }
            TensorPrimitive::QFloat(lhs) => {
                TensorPrimitive::QFloat(B::q_remainder_scalar(lhs, rhs.elem()))
            }
        }
    }
    fn mul(lhs: Self::Primitive, rhs: Self::Primitive) -> <Float as TensorKind<B>>::Primitive {
        match (lhs, rhs) {
            (TensorPrimitive::Float(lhs), TensorPrimitive::Float(rhs)) => {
                TensorPrimitive::Float(B::float_mul(lhs, rhs))
            }
            (TensorPrimitive::QFloat(lhs), TensorPrimitive::QFloat(rhs)) => {
                TensorPrimitive::QFloat(B::q_mul(lhs, rhs))
            }
            _ => panic!("Primitive type mismatch for lhs and rhs"),
        }
    }
    fn mul_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive {
        match lhs {
            TensorPrimitive::Float(lhs) => {
                TensorPrimitive::Float(B::float_mul_scalar(lhs, rhs.elem()))
            }
            TensorPrimitive::QFloat(lhs) => {
                TensorPrimitive::QFloat(B::q_mul_scalar(lhs, rhs.elem()))
            }
        }
    }
    fn neg(tensor: Self::Primitive) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => TensorPrimitive::Float(B::float_neg(tensor)),
            TensorPrimitive::QFloat(tensor) => TensorPrimitive::QFloat(B::q_neg(tensor)),
        }
    }
    fn zeros(shape: Shape, device: &B::Device) -> Self::Primitive {
        TensorPrimitive::Float(B::float_zeros(shape, device))
    }
    fn ones(shape: Shape, device: &B::Device) -> Self::Primitive {
        TensorPrimitive::Float(B::float_ones(shape, device))
    }

    fn full<E: ElementConversion>(
        shape: Shape,
        fill_value: E,
        device: &B::Device,
    ) -> Self::Primitive {
        TensorPrimitive::Float(B::float_full(shape, fill_value.elem(), device))
    }

    fn sum(tensor: Self::Primitive) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => TensorPrimitive::Float(B::float_sum(tensor)),
            TensorPrimitive::QFloat(tensor) => TensorPrimitive::QFloat(B::q_sum(tensor)),
        }
    }

    fn sum_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => TensorPrimitive::Float(B::float_sum_dim(tensor, dim)),
            TensorPrimitive::QFloat(tensor) => TensorPrimitive::QFloat(B::q_sum_dim(tensor, dim)),
        }
    }

    fn prod(tensor: Self::Primitive) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => TensorPrimitive::Float(B::float_prod(tensor)),
            TensorPrimitive::QFloat(tensor) => TensorPrimitive::QFloat(B::q_prod(tensor)),
        }
    }

    fn prod_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(B::float_prod_dim(tensor, dim))
            }
            TensorPrimitive::QFloat(tensor) => TensorPrimitive::QFloat(B::q_prod_dim(tensor, dim)),
        }
    }

    fn mean(tensor: Self::Primitive) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => TensorPrimitive::Float(B::float_mean(tensor)),
            TensorPrimitive::QFloat(tensor) => TensorPrimitive::QFloat(B::q_mean(tensor)),
        }
    }

    fn mean_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(B::float_mean_dim(tensor, dim))
            }
            TensorPrimitive::QFloat(tensor) => TensorPrimitive::QFloat(B::q_mean_dim(tensor, dim)),
        }
    }

    fn equal_elem(lhs: Self::Primitive, rhs: Self::Elem) -> B::BoolTensorPrimitive {
        B::float_equal_elem(lhs.tensor(), rhs)
    }
    fn not_equal_elem(lhs: Self::Primitive, rhs: Self::Elem) -> B::BoolTensorPrimitive {
        B::float_not_equal_elem(lhs.tensor(), rhs)
    }
    fn greater(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive {
        B::float_greater(lhs.tensor(), rhs.tensor())
    }

    fn greater_elem(lhs: Self::Primitive, rhs: Self::Elem) -> B::BoolTensorPrimitive {
        B::float_greater_elem(lhs.tensor(), rhs)
    }

    fn greater_equal(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive {
        B::float_greater_equal(lhs.tensor(), rhs.tensor())
    }

    fn greater_equal_elem(lhs: Self::Primitive, rhs: Self::Elem) -> B::BoolTensorPrimitive {
        B::float_greater_equal_elem(lhs.tensor(), rhs)
    }

    fn lower(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive {
        B::float_lower(lhs.tensor(), rhs.tensor())
    }

    fn lower_elem(lhs: Self::Primitive, rhs: Self::Elem) -> B::BoolTensorPrimitive {
        B::float_lower_elem(lhs.tensor(), rhs)
    }

    fn lower_equal(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive {
        B::float_lower_equal(lhs.tensor(), rhs.tensor())
    }

    fn lower_equal_elem(lhs: Self::Primitive, rhs: Self::Elem) -> B::BoolTensorPrimitive {
        B::float_lower_equal_elem(lhs.tensor(), rhs)
    }

    fn mask_where(
        tensor: Self::Primitive,
        mask: B::BoolTensorPrimitive,
        source: Self::Primitive,
    ) -> Self::Primitive {
        match (tensor, source) {
            (TensorPrimitive::Float(tensor), TensorPrimitive::Float(source)) => {
                TensorPrimitive::Float(B::float_mask_where(tensor, mask, source))
            }
            (TensorPrimitive::QFloat(tensor), TensorPrimitive::QFloat(source)) => {
                TensorPrimitive::QFloat(B::q_mask_where(tensor, mask, source))
            }
            _ => panic!("Primitive type mismatch for tensor and source"),
        }
    }

    fn mask_fill(
        tensor: Self::Primitive,
        mask: B::BoolTensorPrimitive,
        value: Self::Elem,
    ) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(B::float_mask_fill(tensor, mask, value))
            }
            TensorPrimitive::QFloat(tensor) => {
                TensorPrimitive::QFloat(B::q_mask_fill(tensor, mask, value))
            }
        }
    }

    fn select(tensor: Self::Primitive, dim: usize, indices: Tensor<B, 1, Int>) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(B::float_select(tensor, dim, indices.primitive))
            }
            TensorPrimitive::QFloat(tensor) => {
                TensorPrimitive::QFloat(B::q_select(tensor, dim, indices.primitive))
            }
        }
    }

    fn select_assign(
        tensor: Self::Primitive,
        dim: usize,
        indices: Tensor<B, 1, Int>,
        values: Self::Primitive,
    ) -> Self::Primitive {
        match (tensor, values) {
            (TensorPrimitive::Float(tensor), TensorPrimitive::Float(values)) => {
                TensorPrimitive::Float(B::float_select_assign(
                    tensor,
                    dim,
                    indices.primitive,
                    values,
                ))
            }
            (TensorPrimitive::QFloat(tensor), TensorPrimitive::QFloat(values)) => {
                TensorPrimitive::QFloat(B::q_select_assign(tensor, dim, indices.primitive, values))
            }
            _ => panic!("Primitive type mismatch for tensor and values"),
        }
    }

    fn gather(
        dim: usize,
        tensor: Self::Primitive,
        indices: B::IntTensorPrimitive,
    ) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(B::float_gather(dim, tensor, indices))
            }
            TensorPrimitive::QFloat(tensor) => {
                TensorPrimitive::QFloat(B::q_gather(dim, tensor, indices))
            }
        }
    }

    fn scatter(
        dim: usize,
        tensor: Self::Primitive,
        indices: B::IntTensorPrimitive,
        values: Self::Primitive,
    ) -> Self::Primitive {
        match (tensor, values) {
            (TensorPrimitive::Float(tensor), TensorPrimitive::Float(values)) => {
                TensorPrimitive::Float(B::float_scatter(dim, tensor, indices, values))
            }
            (TensorPrimitive::QFloat(tensor), TensorPrimitive::QFloat(values)) => {
                TensorPrimitive::QFloat(B::q_scatter(dim, tensor, indices, values))
            }
            _ => panic!("Primitive type mismatch for tensor and values"),
        }
    }

    fn argmax(tensor: Self::Primitive, dim: usize) -> IntTensor<B> {
        match tensor {
            TensorPrimitive::Float(tensor) => B::float_argmax(tensor, dim),
            TensorPrimitive::QFloat(tensor) => B::q_argmax(tensor, dim),
        }
    }

    fn argmin(tensor: Self::Primitive, dim: usize) -> IntTensor<B> {
        match tensor {
            TensorPrimitive::Float(tensor) => B::float_argmin(tensor, dim),
            TensorPrimitive::QFloat(tensor) => B::q_argmin(tensor, dim),
        }
    }

    fn max(tensor: Self::Primitive) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => TensorPrimitive::Float(B::float_max(tensor)),
            TensorPrimitive::QFloat(tensor) => TensorPrimitive::QFloat(B::q_max(tensor)),
        }
    }

    fn max_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => TensorPrimitive::Float(B::float_max_dim(tensor, dim)),
            TensorPrimitive::QFloat(tensor) => TensorPrimitive::QFloat(B::q_max_dim(tensor, dim)),
        }
    }

    fn max_dim_with_indices(
        tensor: Self::Primitive,
        dim: usize,
    ) -> (Self::Primitive, IntTensor<B>) {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                let (values, indices) = B::float_max_dim_with_indices(tensor, dim);
                (TensorPrimitive::Float(values), indices)
            }
            TensorPrimitive::QFloat(tensor) => {
                let (values, indices) = B::q_max_dim_with_indices(tensor, dim);
                (TensorPrimitive::QFloat(values), indices)
            }
        }
    }

    fn min(tensor: Self::Primitive) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => TensorPrimitive::Float(B::float_min(tensor)),
            TensorPrimitive::QFloat(tensor) => TensorPrimitive::QFloat(B::q_min(tensor)),
        }
    }

    fn min_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => TensorPrimitive::Float(B::float_min_dim(tensor, dim)),
            TensorPrimitive::QFloat(tensor) => TensorPrimitive::QFloat(B::q_min_dim(tensor, dim)),
        }
    }

    fn min_dim_with_indices(
        tensor: Self::Primitive,
        dim: usize,
    ) -> (Self::Primitive, IntTensor<B>) {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                let (values, indices) = B::float_min_dim_with_indices(tensor, dim);
                (TensorPrimitive::Float(values), indices)
            }
            TensorPrimitive::QFloat(tensor) => {
                let (values, indices) = B::q_min_dim_with_indices(tensor, dim);
                (TensorPrimitive::QFloat(values), indices)
            }
        }
    }

    fn clamp(tensor: Self::Primitive, min: B::FloatElem, max: B::FloatElem) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(B::float_clamp(tensor, min, max))
            }
            TensorPrimitive::QFloat(tensor) => {
                TensorPrimitive::QFloat(B::q_clamp(tensor, min, max))
            }
        }
    }

    fn clamp_min(tensor: Self::Primitive, min: B::FloatElem) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(B::float_clamp_min(tensor, min))
            }
            TensorPrimitive::QFloat(tensor) => TensorPrimitive::QFloat(B::q_clamp_min(tensor, min)),
        }
    }

    fn clamp_max(tensor: Self::Primitive, max: B::FloatElem) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(B::float_clamp_max(tensor, max))
            }
            TensorPrimitive::QFloat(tensor) => TensorPrimitive::QFloat(B::q_clamp_max(tensor, max)),
        }
    }

    fn abs(tensor: Self::Primitive) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => TensorPrimitive::Float(B::float_abs(tensor)),
            TensorPrimitive::QFloat(tensor) => TensorPrimitive::QFloat(B::q_abs(tensor)),
        }
    }

    fn powf(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        match (lhs, rhs) {
            (TensorPrimitive::Float(lhs), TensorPrimitive::Float(rhs)) => {
                TensorPrimitive::Float(B::float_powf(lhs, rhs))
            }
            (TensorPrimitive::QFloat(lhs), TensorPrimitive::QFloat(rhs)) => {
                TensorPrimitive::QFloat(B::q_powf(lhs, rhs))
            }
            _ => panic!("Primitive type mismatch for lhs and rhs"),
        }
    }

    fn powf_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive {
        match lhs {
            TensorPrimitive::Float(lhs) => {
                TensorPrimitive::Float(B::float_powf_scalar(lhs, rhs.elem()))
            }
            TensorPrimitive::QFloat(lhs) => {
                TensorPrimitive::QFloat(B::q_powf_scalar(lhs, rhs.elem()))
            }
        }
    }

    fn powi(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        match (lhs, rhs) {
            (TensorPrimitive::Float(lhs), TensorPrimitive::Float(rhs)) => {
                TensorPrimitive::Float(B::float_powf(lhs, rhs))
            }
            (TensorPrimitive::QFloat(lhs), TensorPrimitive::QFloat(rhs)) => {
                TensorPrimitive::QFloat(B::q_powf(lhs, rhs))
            }
            _ => panic!("Primitive type mismatch for lhs and rhs"),
        }
    }

    fn powi_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive {
        match lhs {
            TensorPrimitive::Float(lhs) => {
                TensorPrimitive::Float(B::float_powf_scalar(lhs, rhs.elem()))
            }
            TensorPrimitive::QFloat(lhs) => {
                TensorPrimitive::QFloat(B::q_powf_scalar(lhs, rhs.elem()))
            }
        }
    }

    fn random(shape: Shape, distribution: Distribution, device: &Device<B>) -> Self::Primitive {
        TensorPrimitive::Float(B::float_random(shape, distribution, device))
    }

    fn sign(tensor: Self::Primitive) -> Self::Primitive {
        TensorPrimitive::Float(B::float_sign(tensor.tensor()))
    }

    fn sort(tensor: Self::Primitive, dim: usize, descending: bool) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(B::float_sort(tensor, dim, descending))
            }
            TensorPrimitive::QFloat(tensor) => {
                TensorPrimitive::QFloat(B::q_sort(tensor, dim, descending))
            }
        }
    }

    fn sort_with_indices(
        tensor: Self::Primitive,
        dim: usize,
        descending: bool,
    ) -> (Self::Primitive, <Int as TensorKind<B>>::Primitive) {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                let (values, indices) = B::float_sort_with_indices(tensor, dim, descending);
                (TensorPrimitive::Float(values), indices)
            }
            TensorPrimitive::QFloat(tensor) => {
                let (values, indices) = B::q_sort_with_indices(tensor, dim, descending);
                (TensorPrimitive::QFloat(values), indices)
            }
        }
    }

    fn argsort(
        tensor: Self::Primitive,
        dim: usize,
        descending: bool,
    ) -> <Int as TensorKind<B>>::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => B::float_argsort(tensor, dim, descending),
            TensorPrimitive::QFloat(tensor) => B::q_argsort(tensor, dim, descending),
        }
    }
}

impl<B, const D: usize, K> core::ops::Add<Self> for Tensor<B, D, K>
where
    B: Backend,
    K: Numeric<B>,
    K::Elem: Element,
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
    K::Elem: Element,
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
    K::Elem: Element,
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
    K::Elem: Element,
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
    K::Elem: Element,
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
    K::Elem: Element,
{
    type Output = Self;

    fn div(self, other: E) -> Self {
        Tensor::div_scalar(self, other)
    }
}

impl<const D: usize, B, K> core::ops::Rem<Tensor<B, D, K>> for Tensor<B, D, K>
where
    B: Backend,
    K: Numeric<B>,
    K::Elem: Element,
{
    type Output = Self;
    fn rem(self, rhs: Tensor<B, D, K>) -> Self::Output {
        Tensor::remainder(self, rhs)
    }
}

impl<E, const D: usize, B, K> core::ops::Rem<E> for Tensor<B, D, K>
where
    E: ElementConversion,
    B: Backend,
    K: Numeric<B>,
    K::Elem: Element,
{
    type Output = Self;

    fn rem(self, other: E) -> Self {
        Tensor::remainder_scalar(self, other)
    }
}

impl<B, const D: usize, K> core::ops::Mul<Tensor<B, D, K>> for Tensor<B, D, K>
where
    B: Backend,
    K: Numeric<B>,
    K::Elem: Element,
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
    K::Elem: Element,
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
    K::Elem: Element,
{
    type Output = Self;

    fn neg(self) -> Self {
        Tensor::neg(self)
    }
}
