use alloc::vec::Vec;
use core::convert::TryInto;

use crate::check;
use crate::check::TensorCheck;
use crate::tensor::backend::Backend;
use crate::tensor::stats;
use crate::tensor::{Data, Distribution, Shape};
use crate::Int;
use crate::Tensor;

#[cfg(all(not(feature = "wasm-sync"), target_family = "wasm"))]
use crate::{argsort, sort, sort_with_indices, Float};

impl<const D: usize, B> Tensor<B, D>
where
    B: Backend,
{
    /// Executes an operation on the tensor and modifies its value.
    ///
    /// # Notes
    ///
    /// This won't necessary reuse the same tensor data/buffer, but it should if there is
    /// no other reference pointing to the same tensor.
    ///
    /// Wrapping operations with inplace is not an optimization, it's mainly there if you
    /// want to mutate a tensor by using owned operations. A plausible usage would be to
    /// update the weights of a mutable model reference.
    pub fn inplace<F: FnOnce(Self) -> Self>(&mut self, func: F) {
        let mut tensor_owned = Tensor::empty([0; D], &self.device());
        core::mem::swap(&mut tensor_owned, self);

        let mut tensor_new = func(tensor_owned);
        core::mem::swap(&mut tensor_new, self);
    }

    /// Applies element wise exponential operation.
    ///
    /// `y = e^x`
    pub fn exp(self) -> Self {
        Self::new(B::float_exp(self.primitive))
    }

    /// Applies element wise natural log operation *ln*.
    ///
    /// `y = log(x)`
    pub fn log(self) -> Self {
        Self::new(B::float_log(self.primitive))
    }

    /// Applies the natural logarithm of one plus the input tensor, element-wise.
    ///
    /// `y = log(x+1)`
    pub fn log1p(self) -> Self {
        Self::new(B::float_log1p(self.primitive))
    }

    /// Applies the [error function](https://en.wikipedia.org/wiki/Error_function) element wise.
    ///
    /// `y = erf(x)`
    pub fn erf(self) -> Self {
        Self::new(B::float_erf(self.primitive))
    }

    /// Applies element wise reciprocal operation.
    pub fn recip(self) -> Self {
        Self::new(B::float_recip(self.primitive))
    }

    /// Applies element wise root square operation.
    pub fn sqrt(self) -> Self {
        Self::new(B::float_sqrt(self.primitive))
    }

    /// Applies element wise cosine operation.
    pub fn cos(self) -> Self {
        Self::new(B::float_cos(self.primitive))
    }

    /// Applies element wise sine operation.
    pub fn sin(self) -> Self {
        Self::new(B::float_sin(self.primitive))
    }

    /// Applies element wise hyperbolic tangent operation.
    pub fn tanh(self) -> Self {
        Self::new(B::float_tanh(self.primitive))
    }

    /// Create a tensor from floats (f32) on a given device.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///     let device = B::Device::default();
    ///     let _ = Tensor::<B, 1>::from_floats([1.0, 2.0], &device);
    ///     let _ = Tensor::<B, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
    /// }
    /// ```
    pub fn from_floats<A: Into<Data<f32, D>>>(floats: A, device: &B::Device) -> Self {
        Self::from_data(floats.into().convert(), device)
    }

    /// Returns a new tensor with the same shape and device as the current tensor and the data
    /// casted to Integer.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///     let float_tensor = Tensor::<B, 1>::from_floats([1.0, 2.0], &device);
    ///     let int_tensor = float_tensor.int();
    /// }
    /// ```
    pub fn int(self) -> Tensor<B, D, Int> {
        Tensor::new(B::float_into_int(self.primitive))
    }

    /// Returns a new tensor with the same shape and device as the current tensor filled with zeros.
    pub fn zeros_like(&self) -> Self {
        Tensor::new(B::float_zeros(self.shape(), &self.device()))
    }

    /// Returns a new tensor with the same shape and device as the current tensor filled with ones.
    pub fn ones_like(&self) -> Self {
        Tensor::new(B::float_ones(self.shape(), &self.device()))
    }

    /// Returns a new tensor with the same shape and device as the current tensor filled random
    /// values sampled from the given distribution.
    pub fn random_like(&self, distribution: Distribution) -> Self {
        Tensor::new(B::float_random(self.shape(), distribution, &self.device()))
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
    ///     let device = Default::default();
    ///     let one_hot = Tensor::<B, 1>::one_hot(2, 10, &device);
    ///     println!("{}", one_hot.to_data());
    ///     // [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    /// }
    /// ```
    pub fn one_hot(index: usize, num_classes: usize, device: &B::Device) -> Self {
        check!(TensorCheck::one_hot(index, num_classes));

        let mut dims = [1; D];
        dims[D - 1] = num_classes;
        let shape = Shape::new(dims);
        let ranges: Vec<_> = shape.dims.iter().map(|dim| 0..*dim).collect();
        let tensor = Tensor::zeros(shape, device);
        let mut ranges: [core::ops::Range<usize>; D] = ranges.try_into().unwrap();
        ranges[D - 1] = index..index + 1;

        tensor.slice_assign(ranges, Tensor::ones(Shape::new([1; D]), device))
    }

    /// Applies the matrix multiplication operation.
    ///
    /// `C = AB`
    ///
    /// # Panics
    ///
    /// If the two tensors dont' have a compatible shape.
    pub fn matmul(self, other: Self) -> Self {
        check!(TensorCheck::matmul(&self, &other));
        Self::new(B::float_matmul(self.primitive, other.primitive))
    }

    /// Calculate the variance along the given dimension.
    pub fn var(self, dim: usize) -> Self {
        stats::var(self, dim)
    }

    /// Calculate the variance along the given dimension without applying the Bessel’s correction.
    pub fn var_bias(self, dim: usize) -> Self {
        stats::var_bias(self, dim)
    }

    /// Calculate the variance along the given dimension and also returns the mean.
    pub fn var_mean(self, dim: usize) -> (Self, Self) {
        let mean = self.clone().mean_dim(dim);
        let var = stats::var_with_mean(self, mean.clone(), dim);
        (var, mean)
    }

    /// Calculate the variance along the given dimension without applying the Bessel’s correction and also returns the mean.
    pub fn var_mean_bias(self, dim: usize) -> (Self, Self) {
        let mean = self.clone().mean_dim(dim);
        let var = stats::var_with_mean_bias(self, mean.clone(), dim);
        (var, mean)
    }

    /// Returns a tensor with full precision based on the selected backend.
    pub fn to_full_precision(&self) -> Tensor<B::FullPrecisionBackend, D> {
        Tensor::new(B::float_to_full_precision(&self.primitive))
    }

    /// Returns a tensor on the selected backend from a full precision tensor.
    pub fn from_full_precision(tensor: Tensor<B::FullPrecisionBackend, D>) -> Self {
        Self::new(B::float_from_full_precision(tensor.primitive))
    }

    /// Detach the current tensor from the autodiff graph.
    ///
    /// This function does nothing when autodiff is not enabled.
    /// This can be used in batchers or elsewhere to ensure that previous operations are not
    /// considered in the autodiff graph.
    pub fn detach(self) -> Self {
        Self::new(B::float_detach(self.primitive))
    }

    /// Mark the tensor to keep gradients during the backward pass.
    ///
    /// This function does nothing when autodiff is not enabled.
    pub fn require_grad(self) -> Self {
        self.set_require_grad(true)
    }

    /// Returns true if the tensor requires gradients during the backward pass.
    pub fn is_require_grad(&self) -> bool {
        B::float_is_require_grad(&self.primitive)
    }

    /// Mark the tensor as tracked or untracked depending on the require grad argument.
    /// When tracked, the gradients will be available after the backward pass.
    ///
    /// This function does nothing when autodiff is not enabled.
    pub fn set_require_grad(self, require_grad: bool) -> Self {
        Self::new(B::float_set_require_grad(self.primitive, require_grad))
    }

    /// Applies the relu function to the tensor.
    pub(crate) fn relu(self) -> Self {
        Self::new(B::relu(self.primitive))
    }

    /// Calculate covaraince matrix between different entries alongside a given dimension.
    ///
    /// # Arguments
    ///
    /// * `size` - The size of the square matrix.
    /// * `correction_factor` - Is usually 1 for samples and 0 for population.
    pub fn cov(self, dim: usize, correction_factor: usize) -> Tensor<B, D> {
        let n = self.dims()[dim];
        let centered = (self.clone() - self.mean_dim(dim)).swap_dims(dim, 0);
        centered
            .clone()
            .transpose()
            .matmul(centered)
            .div_scalar(n as f32 - correction_factor as f32)
    }

    /// Sort the elements by value in ascending order along a given dimension.
    ///
    /// This sort is unstable (i.e., may reorder equal elements).
    #[cfg(all(not(feature = "wasm-sync"), target_family = "wasm"))]
    pub async fn sort(self, dim: usize) -> Tensor<B, D> {
        Tensor::new(sort::<B, D, Float>(self.primitive, dim, /*descending*/ false).await)
    }

    /// Sort the elements by value in descending order along a given dimension.
    ///
    /// This sort is unstable (i.e., may reorder equal elements).
    #[cfg(all(not(feature = "wasm-sync"), target_family = "wasm"))]
    pub async fn sort_descending(self, dim: usize) -> Tensor<B, D> {
        Tensor::new(sort::<B, D, Float>(self.primitive, dim, /*descending*/ true).await)
    }

    /// Sort the elements by value in ascending order along a given dimension.
    /// Also returns the indices.
    ///
    /// This sort is unstable (i.e., may reorder equal elements).
    #[cfg(all(not(feature = "wasm-sync"), target_family = "wasm"))]
    pub async fn sort_with_indices(self, dim: usize) -> (Tensor<B, D>, Tensor<B, D, Int>) {
        check!(TensorCheck::sort_dim::<D>("Sort_with_indices", dim));
        let (values, indices) =
            sort_with_indices::<B, D, Float>(self.primitive, dim, /*descending*/ false).await;
        (Tensor::new(values), Tensor::new(indices))
    }

    /// Sort the elements by value in descending order along a given dimension.
    /// Also returns the indices.
    ///
    /// This sort is unstable (i.e., may reorder equal elements).
    #[cfg(all(not(feature = "wasm-sync"), target_family = "wasm"))]
    pub async fn sort_descending_with_indices(
        self,
        dim: usize,
    ) -> (Tensor<B, D>, Tensor<B, D, Int>) {
        check!(TensorCheck::sort_dim::<D>("Sort_with_indices", dim));
        let (values, indices) =
            sort_with_indices::<B, D, Float>(self.primitive, dim, /*descending*/ true).await;
        (Tensor::new(values), Tensor::new(indices))
    }

    /// Returns the indices that sort the elements by value in ascending order along a given dimension.
    ///
    /// This sort is unstable (i.e., may reorder equal elements).
    #[cfg(all(not(feature = "wasm-sync"), target_family = "wasm"))]
    pub async fn argsort(self, dim: usize) -> Tensor<B, D, Int> {
        check!(TensorCheck::sort_dim::<D>("Argsort", dim));
        Tensor::new(argsort::<B, D, Float>(self.primitive, dim, /*descending*/ false).await)
    }

    /// Returns the indices that sort the elements by value in descending order along a given dimension.
    ///
    /// This sort is unstable (i.e., may reorder equal elements).
    #[cfg(all(not(feature = "wasm-sync"), target_family = "wasm"))]
    pub async fn argsort_descending(self, dim: usize) -> Tensor<B, D, Int> {
        check!(TensorCheck::sort_dim::<D>("Argsort", dim));
        Tensor::new(argsort::<B, D, Float>(self.primitive, dim, /*descending*/ true).await)
    }

    /// Returns the `k` largest elements of the given input tensor along a given dimension.
    #[cfg(all(not(feature = "wasm-sync"), target_family = "wasm"))]
    pub async fn topk(self, k: usize, dim: usize) -> Tensor<B, D> {
        let k_indices = Tensor::arange(0..k as i64, &self.device());
        self.sort_descending(dim).await.select(dim, k_indices)
    }

    /// Returns the `k` largest elements of the given input tensor along a given dimension.
    /// Also returns the indices.
    #[cfg(all(not(feature = "wasm-sync"), target_family = "wasm"))]
    pub async fn topk_with_indices(
        self,
        k: usize,
        dim: usize,
    ) -> (Tensor<B, D>, Tensor<B, D, Int>) {
        let k_indices = Tensor::arange(0..k as i64, &self.device());
        let (values, indices) = self.sort_descending_with_indices(dim).await;
        (
            values.select(dim, k_indices.clone()),
            indices.select(dim, k_indices),
        )
    }
}
