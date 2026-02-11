use burn_backend::Scalar;
pub use burn_backend::tensor::Numeric;

use crate::alloc::borrow::ToOwned;
use alloc::vec::Vec;

use crate::IndexingUpdateOp;
use crate::{
    AsIndex, Bool, Distribution, Element, ElementConversion, Int, Shape, Tensor, backend::Backend,
    check, check::TensorCheck,
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
        let other = Scalar::new(other, &self.dtype());
        Self::new(K::add_scalar(self.primitive, other))
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
        let other = Scalar::new(other, &self.dtype());
        Self::new(K::sub_scalar(self.primitive, other))
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
        let other = Scalar::new(other, &self.dtype());
        Self::new(K::div_scalar(self.primitive, other))
    }

    /// Applies element wise the remainder operation with a scalar.
    ///
    /// `y = x2 % x1`
    pub fn remainder(self, other: Self) -> Self {
        Self::new(K::remainder(self.primitive, other.primitive))
    }

    /// Applies element wise the remainder operation with a scalar.
    ///
    /// `y = x % s`
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
        let other = Scalar::new(other, &self.dtype());
        Self::new(K::remainder_scalar(self.primitive, other))
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
        let other = Scalar::new(other, &self.dtype());
        Self::new(K::mul_scalar(self.primitive, other))
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
    /// * `dim` - The dimension or axis along which to aggregate the elements;
    ///   supports negative indexing.
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
    pub fn mean_dim<I: AsIndex>(self, dim: I) -> Self {
        let dim = dim.expect_dim_index(D);
        check!(TensorCheck::aggregate_dim::<D>("Mean", dim));
        Self::new(K::mean_dim(self.primitive, dim))
    }

    /// Aggregate all elements along the given *axes*
    /// in the tensor with the mean operation.
    ///
    /// # Arguments
    ///
    /// * `dims` - the dimensions to aggregate; supports negative indexing.
    ///
    /// # Returns
    ///
    /// The returned tensor will have the same rank,
    /// but the aggregated dimensions will have size 1.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::from_data([[2.0, 4.0], [6.0, -4.0]], &device);
    ///    let tensor = tensor.clone().mean_dims(&[0, 1]);
    ///    println!("{tensor}");
    ///    // [[2.0]]
    /// }
    /// ```
    pub fn mean_dims<I: AsIndex>(self, dims: &[I]) -> Self {
        dims.iter().fold(self, |tensor, &dim| tensor.mean_dim(dim))
    }

    /// Aggregate all elements along the given *dimension* or *axis*
    /// in the tensor with the sum operation.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension or axis along which to aggregate the elements;
    ///   supports negative indexing.
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
    ///    // [[6.0, 7.0, 9.0]]
    ///    let tensor = tensor.clone().sum_dim(1);
    ///    println!("{tensor}");
    ///    // [[2.0], [20.0]]
    /// }
    /// ```
    pub fn sum_dim<I: AsIndex>(self, dim: I) -> Self {
        let dim = dim.expect_dim_index(D);
        check!(TensorCheck::aggregate_dim::<D>("Sum", dim));
        Self::new(K::sum_dim(self.primitive, dim))
    }

    /// Aggregate all elements along the given *axes*
    /// in the tensor with the sum operation.
    ///
    /// # Arguments
    ///
    /// * `dims` - the dimensions to aggregate; supports negative indexing.
    ///
    /// # Returns
    ///
    /// The returned tensor will have the same rank,
    /// but the aggregated dimensions will have size 1.
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
    ///    let tensor = tensor.clone().sum_dims(&[0, 1]);
    ///    println!("{tensor}");
    ///    // [[27]]
    /// }
    /// ```
    pub fn sum_dims<I: AsIndex>(self, dims: &[I]) -> Self {
        dims.iter().fold(self, |tensor, &dim| tensor.sum_dim(dim))
    }

    /// Aggregate and squeeze along the given dimensions.
    ///
    /// This is equivalent to ``tensor.sum_dims(dims).squeeze_dims(dims)``
    ///
    /// # Arguments
    ///
    /// * `dims` - the dimensions to aggregate; supports negative indexing.
    ///
    /// # Returns
    ///
    /// The returned tensor will have the same rank,
    /// but the aggregated dimensions will have size 1.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///     let device = B::Device::default();
    ///     let tensor = Tensor::<B, 3>::from_data([
    ///         [[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]],
    ///         [[9.0, 2.0, 5.0], [5.0, 7.0, 7.0]],
    ///     ], &device);
    ///     let tensor = tensor.clone().sum_dims_squeeze::<1, _>(&[0, 1]);
    ///     println!("{tensor}");
    ///     // [20.0, 16.0, 21.0]
    /// }
    /// ```
    pub fn sum_dims_squeeze<const D2: usize, I: AsIndex>(self, dims: &[I]) -> Tensor<B, D2, K> {
        // TODO: remove idims when squeeze_dims uses AsIndex.
        let idims = dims
            .iter()
            .map(|&dim| (dim.expect_dim_index(D)) as isize)
            .collect::<Vec<_>>();
        self.sum_dims(dims).squeeze_dims::<D2>(&idims)
    }

    /// Aggregate all elements in the tensor with the product operation.
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
    /// * `dim` - The dimension or axis along which to aggregate the elements,
    ///   supports negative indexing.
    ///
    /// # Returns
    ///
    /// The returned tensor will have the same rank,
    /// but the aggregated dimension will have size 1.
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
    pub fn prod_dim<I: AsIndex>(self, dim: I) -> Self {
        let dim = dim.expect_dim_index(D);
        check!(TensorCheck::aggregate_dim::<D>("Prod", dim));
        Self::new(K::prod_dim(self.primitive, dim))
    }

    /// Aggregate all elements along the given *axes*
    /// in the tensor with the prod operation.
    ///
    /// # Arguments
    ///
    /// * `dims` - the dimensions to aggregate, supports negative indexing.
    ///
    /// # Returns
    ///
    /// The returned tensor will have the same rank,
    /// but the aggregated dimensions will have size 1.
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
    ///    let tensor = tensor.clone().sum_dims(&[0, 1]);
    ///    println!("{tensor}");
    ///    // [[-1620.0]]
    /// }
    /// ```
    pub fn prod_dims<I: AsIndex>(self, dims: &[I]) -> Self {
        dims.iter().fold(self, |tensor, &dim| tensor.prod_dim(dim))
    }

    /// Computes the cumulative sum of elements along the given *dimension* or *axis*.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension or axis along which to compute the cumulative sum.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
    ///    let result = tensor.clone().cumsum(0);
    ///    println!("{result}");
    ///    // [[1.0, 2.0, 3.0], [5.0, 7.0, 9.0]]
    ///    let result = tensor.cumsum(1);
    ///    println!("{result}");
    ///    // [[1.0, 3.0, 6.0], [4.0, 9.0, 15.0]]
    /// }
    /// ```
    pub fn cumsum(self, dim: usize) -> Self {
        check!(TensorCheck::aggregate_dim::<D>("CumSum", dim));
        Self::new(K::cumsum(self.primitive, dim))
    }

    /// Computes the cumulative product of elements along the given *dimension* or *axis*.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension or axis along which to compute the cumulative product.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
    ///    let result = tensor.clone().cumprod(0);
    ///    println!("{result}");
    ///    // [[1.0, 2.0, 3.0], [4.0, 10.0, 18.0]]
    ///    let result = tensor.cumprod(1);
    ///    println!("{result}");
    ///    // [[1.0, 2.0, 6.0], [4.0, 20.0, 120.0]]
    /// }
    /// ```
    pub fn cumprod(self, dim: usize) -> Self {
        check!(TensorCheck::aggregate_dim::<D>("CumProd", dim));
        Self::new(K::cumprod(self.primitive, dim))
    }

    /// Apply element wise absolute value operation.
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
    ///
    /// # Notes
    ///
    /// For signed integer dtypes, this operation uses two's-complement wraparound semantics, similar to
    /// `x.wrapping_abs()`. For example, `abs(i64::MIN) == i64::MIN`.
    pub fn abs(self) -> Self {
        Self::new(K::abs(self.primitive))
    }

    /// Returns the upper triangular part of a matrix (2-D tensor) or batch of matrices input,
    /// the other elements of the result tensor out are set to 0.
    ///
    /// See also [`triu_mask`](Tensor::triu_mask).
    ///
    /// # Arguments
    ///
    /// * `diagonal` - The offset from the diagonal, where 0 means the diagonal, and positive values shift
    ///   towards the upper triangle.
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
    /// See also [`tril_mask`](Tensor::tril_mask).
    ///
    /// # Arguments
    ///
    /// * `diagonal` - The offset from the diagonal, where 0 means the diagonal, and positive values shift
    ///   towards the upper triangle.
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
        let other = Scalar::new(other, &self.dtype());
        Self::new(K::powf_scalar(self.primitive, other))
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
    ///
    ///    // [[1, 4, 9], [25, 81, 36]]
    ///    let tensor = Tensor::<B, 2>::from_data([[1.5, -2., 3.], [5., 9., 6.]], &device);
    ///    let tensor = tensor.powi_scalar(2);
    ///    println!("{tensor}");
    ///    // [[2.25, 4., 9.], [25., 81., 36.]]
    /// }
    /// ```
    pub fn powi_scalar<E: ElementConversion>(self, other: E) -> Self {
        let other = Scalar::new(other, &self.dtype());
        Self::new(K::powi_scalar(self.primitive, other))
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
        self.not_equal_elem(0)
    }

    /// Create a random tensor of the given shape on the given device where each element is
    /// sampled from the given distribution.
    ///
    /// See also [`random_like`](Tensor::random_like).
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

    /// Applies the matrix multiplication operation.
    ///
    /// ```math
    /// C = AB
    /// ```
    ///
    /// Shapes of the form `[..., B, 1, K] @ [..., 1, K, N]` are reinterpreted as
    /// `[..., 1, B, K] @ [..., 1, K, N]`, turning a batched vec-mat into a general
    /// matmul, which is often faster.
    pub fn matmul(self, other: Self) -> Self {
        check!(TensorCheck::matmul(&self, &other));

        if D >= 3 {
            let batch_index = D - 3;
            let vector_index = D - 2;
            let lhs_dims = &self.shape()[batch_index..D];
            let rhs_dims = &other.shape()[batch_index..D];

            if let ([_, 1, k1], [1, k2, _]) = (lhs_dims, rhs_dims)
                && k1 == k2
            {
                return Tensor::new(K::matmul(
                    self.swap_dims(batch_index, vector_index).primitive,
                    other.primitive,
                ))
                .swap_dims(batch_index, vector_index);
            }
        }

        Tensor::new(K::matmul(self.primitive, other.primitive))
    }
}

impl<B, K> Tensor<B, 1, K>
where
    B: Backend,
    K: Numeric<B>,
    K::Elem: Element,
{
    /// Calculates the dot product with another tensor.
    ///
    /// `y = x2.dot(x1)`
    ///
    /// # Arguments
    ///
    /// * `other` - The tensor to compute dot product with.
    ///
    /// # Notes
    ///
    /// Both tensors must have the same number of elements.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor1 = Tensor::<B, 1>::from_data([1.0, 2.0], &device);
    ///    let tensor2 = Tensor::<B, 1>::from_data([-2.0, 3.0], &device);
    ///    let tensor = tensor1.dot(tensor2);
    ///    println!("{tensor}");
    ///    // [4]
    /// }
    /// ```
    pub fn dot(self, other: Self) -> Self {
        self.mul(other).sum()
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
        let ones = Self::ones([1, size], device);
        let zeros = Self::zeros([size, size], device);

        zeros.scatter(0, indices, ones, IndexingUpdateOp::Add)
    }
}

// Tensor + tensor
impl<B: Backend, const D: usize, K: Numeric<B>> core::ops::Add<Self> for Tensor<B, D, K>
where
    K::Elem: Element,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::add(self, rhs)
    }
}

// Tensor + scalar
impl<E: ElementConversion, const D: usize, B: Backend, K: Numeric<B>> core::ops::Add<E>
    for Tensor<B, D, K>
where
    K::Elem: Element,
{
    type Output = Self;

    fn add(self, other: E) -> Self::Output {
        Tensor::add_scalar(self, other)
    }
}

// Scalar + tensor
macro_rules! impl_tensor_scalar_add {
    ($($t:ty),*) => {
        $(
            impl<const D: usize, B: Backend, K: Numeric<B>> core::ops::Add<Tensor<B, D, K>> for $t
            where
                K::Elem: Element,
            {
                type Output = Tensor<B, D, K>;

                fn add(self, tensor: Tensor<B, D, K>) -> Self::Output {
                    Tensor::add_scalar(tensor, self)
                }
            }
        )*
    }
}
impl_tensor_scalar_add!(f32, f64, i32, i64, u32, u64);

// Tensor - tensor
impl<B: Backend, const D: usize, K: Numeric<B>> core::ops::Sub<Self> for Tensor<B, D, K>
where
    K::Elem: Element,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Tensor::sub(self, rhs)
    }
}

// Tensor - scalar
impl<E: ElementConversion, const D: usize, B: Backend, K: Numeric<B>> core::ops::Sub<E>
    for Tensor<B, D, K>
where
    K::Elem: Element,
{
    type Output = Self;

    fn sub(self, other: E) -> Self::Output {
        Tensor::sub_scalar(self, other)
    }
}

// Scalar - tensor
macro_rules! impl_tensor_scalar_sub {
    ($($t:ty),*) => {
        $(
            impl<const D: usize, B: Backend, K: Numeric<B>> core::ops::Sub<Tensor<B, D, K>> for $t
            where
                K::Elem: Element,
            {
                type Output = Tensor<B, D, K>;

                fn sub(self, tensor: Tensor<B, D, K>) -> Self::Output {
                    Tensor::add_scalar(Tensor::neg(tensor), self)
                }
            }
        )*
    }
}
impl_tensor_scalar_sub!(f32, f64, i32, i64, u32, u64);

// Tensor / tensor
impl<B: Backend, const D: usize, K: Numeric<B>> core::ops::Div<Self> for Tensor<B, D, K>
where
    K::Elem: Element,
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Tensor::div(self, rhs)
    }
}

// Tensor / scalar
impl<E: ElementConversion, const D: usize, B: Backend, K: Numeric<B>> core::ops::Div<E>
    for Tensor<B, D, K>
where
    K::Elem: Element,
{
    type Output = Self;

    fn div(self, other: E) -> Self::Output {
        Tensor::div_scalar(self, other)
    }
}

// Scalar / tensor (float only)
macro_rules! impl_tensor_scalar_div {
    ($($t:ty),*) => {
        $(
            impl<const D: usize, B: Backend> core::ops::Div<Tensor<B, D>> for $t
            {
                type Output = Tensor<B, D>;

                fn div(self, tensor: Tensor<B, D>) -> Self::Output {
                    tensor.recip().mul_scalar(self)
                }
            }
        )*
    }
}

impl_tensor_scalar_div!(f32, f64);

// Tensor % tensor.
impl<const D: usize, B: Backend, K: Numeric<B>> core::ops::Rem<Self> for Tensor<B, D, K>
where
    K::Elem: Element,
{
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        Tensor::remainder(self, rhs)
    }
}

// Tensor % scalar.
impl<E: ElementConversion, const D: usize, B: Backend, K: Numeric<B>> core::ops::Rem<E>
    for Tensor<B, D, K>
where
    K::Elem: Element,
{
    type Output = Self;

    fn rem(self, other: E) -> Self::Output {
        Tensor::remainder_scalar(self, other)
    }
}

// Tensor * tensor.
impl<B: Backend, const D: usize, K: Numeric<B>> core::ops::Mul<Self> for Tensor<B, D, K>
where
    K::Elem: Element,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Tensor::mul(self, rhs)
    }
}

// Tensor * scalar.
impl<E: ElementConversion, const D: usize, B: Backend, K: Numeric<B>> core::ops::Mul<E>
    for Tensor<B, D, K>
where
    K::Elem: Element,
{
    type Output = Self;

    fn mul(self, other: E) -> Self::Output {
        Tensor::mul_scalar(self, other)
    }
}

macro_rules! impl_tensor_scalar_mul {
    ($($t:ty),*) => {
        $(
            impl<const D: usize, B: Backend, K: Numeric<B>> core::ops::Mul<Tensor<B, D, K>> for $t
            where
                K::Elem: Element,
            {
                type Output = Tensor<B, D, K>;

                fn mul(self, other: Tensor<B, D, K>) -> Self::Output {
                    Tensor::mul_scalar(other, self)
                }
            }
        )*
    }
}

impl_tensor_scalar_mul!(f32, f64, i32, i64, u32, u64);

impl<B, const D: usize, K> core::ops::Neg for Tensor<B, D, K>
where
    B: Backend,
    K: Numeric<B>,
    K::Elem: Element,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        Tensor::neg(self)
    }
}
