use alloc::vec::Vec;
use core::convert::TryInto;

use crate::backend::ADBackend;
use crate::check;
use crate::check::TensorCheck;
use crate::tensor::backend::Backend;
use crate::tensor::stats;
use crate::tensor::{Data, Distribution, Shape};
use crate::Int;
use crate::Tensor;

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
        let mut tensor_owned = Tensor::empty([0; D]);
        core::mem::swap(&mut tensor_owned, self);

        let mut tensor_new = func(tensor_owned);
        core::mem::swap(&mut tensor_new, self);
    }

    /// Applies element wise exponential operation.
    ///
    /// `y = e^x`
    pub fn exp(self) -> Self {
        Self::new(B::exp(self.primitive))
    }

    /// Applies element wise natural log operation *ln*.
    ///
    /// `y = log(x)`
    pub fn log(self) -> Self {
        Self::new(B::log(self.primitive))
    }

    /// Applies the natural logarithm of one plus the input tensor, element-wise.
    ///
    /// `y = log(x+1)`
    pub fn log1p(self) -> Self {
        Self::new(B::log1p(self.primitive))
    }

    /// Applies the [error function](https://en.wikipedia.org/wiki/Error_function) element wise.
    ///
    /// `y = erf(x)`
    pub fn erf(self) -> Self {
        Self::new(B::erf(self.primitive))
    }

    /// Applies element wise power operation.
    ///
    /// `y = x^a`
    pub fn powf(self, value: f32) -> Self {
        Self::new(B::powf(self.primitive, value))
    }

    /// Applies element wise root square operation.
    pub fn sqrt(self) -> Self {
        Self::new(B::sqrt(self.primitive))
    }

    /// Applies element wise cosine operation.
    pub fn cos(self) -> Self {
        Self::new(B::cos(self.primitive))
    }

    /// Applies element wise sine operation.
    pub fn sin(self) -> Self {
        Self::new(B::sin(self.primitive))
    }

    /// Applies element wise hyperbolic tangent operation.
    pub fn tanh(self) -> Self {
        Self::new(B::tanh(self.primitive))
    }

    /// Create a tensor from floats (f32).
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///     let _ = Tensor::<B, 1>::from_floats([1.0, 2.0]);
    ///     let _ = Tensor::<B, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]]);
    /// }
    /// ```
    pub fn from_floats<A: Into<Data<f32, D>>>(floats: A) -> Self {
        Self::from_data(floats.into().convert())
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
    ///     let float_tensor = Tensor::<B, 1>::from_floats([1.0, 2.0]);
    ///     let int_tensor = float_tensor.int();
    /// }
    /// ```
    pub fn int(self) -> Tensor<B, D, Int> {
        Tensor::new(B::into_int(self.primitive))
    }

    /// Returns a new tensor with the same shape and device as the current tensor filled with zeros.
    pub fn zeros_like(&self) -> Self {
        Tensor::new(B::zeros(self.shape(), &self.device()))
    }

    /// Returns a new tensor with the same shape and device as the current tensor filled with ones.
    pub fn ones_like(&self) -> Self {
        Tensor::new(B::ones(self.shape(), &self.device()))
    }

    /// Returns a new tensor with the same shape and device as the current tensor filled random
    /// values sampled from the given distribution.
    pub fn random_like(&self, distribution: Distribution<B::FloatElem>) -> Self {
        Tensor::new(B::random(self.shape(), distribution, &self.device()))
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
        let ranges: Vec<_> = shape.dims.iter().map(|dim| 0..*dim).collect();
        let tensor = Tensor::zeros(shape);
        let mut ranges: [core::ops::Range<usize>; D] = ranges.try_into().unwrap();
        ranges[D - 1] = index..index + 1;

        tensor.slice_assign(ranges, Tensor::ones(Shape::new([1; D])))
    }

    /// Applies the transpose operation.
    ///
    /// On matrix and higher dimension tensor, it swap the last two dimensions.
    ///
    /// # Panics
    ///
    /// If the tensor is of 1 dimension or less.
    pub fn transpose(self) -> Self {
        Self::new(B::transpose(self.primitive))
    }

    /// Swap two dimensions.
    ///
    /// # Panics
    ///
    /// If the dimensions exceed the shape of than the tensor.
    pub fn swap_dims(self, dim1: usize, dim2: usize) -> Self {
        check!(TensorCheck::swap_dims::<D>(dim1, dim2));
        Self::new(B::swap_dims(self.primitive, dim1, dim2))
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
        Self::new(B::matmul(self.primitive, other.primitive))
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

    /// Create a random tensor of the given shape where each element is sampled from the given
    /// distribution.
    pub fn random<S: Into<Shape<D>>>(shape: S, distribution: Distribution<B::FloatElem>) -> Self {
        let tensor = B::random(shape.into(), distribution, &B::Device::default());
        Self::new(tensor)
    }

    /// Create a random tensor of the given shape on the given device where each element is
    /// sampled from the given distribution.
    pub fn random_device<S: Into<Shape<D>>>(
        shape: S,
        distribution: Distribution<B::FloatElem>,
        device: &B::Device,
    ) -> Self {
        let tensor = B::random(shape.into(), distribution, device);
        Self::new(tensor)
    }
    /// Returns a tensor with full precision based on the selected backend.
    pub fn to_full_precision(&self) -> Tensor<B::FullPrecisionBackend, D> {
        Tensor::new(B::to_full_precision(&self.primitive))
    }

    /// Returns a tensor on the selected backend from a full precision tensor.
    pub fn from_full_precision(tensor: Tensor<B::FullPrecisionBackend, D>) -> Self {
        Self::new(B::from_full_precision(tensor.primitive))
    }

    /// Detach the current tensor from the autodiff graph.
    /// This function does nothing when autodiff is not enabled.
    /// This can be used in batchers or elsewhere to ensure that previous operations are not
    /// considered in the autodiff graph.
    pub fn detach(self) -> Self {
        Self::new(B::detach(self.primitive))
    }

    /// Mark the tensor to keep gradients during the backward pass.
    /// This function does nothing when autodiff is not enabled.
    pub fn require_grad(self) -> Self {
        self.set_require_grad(true)
    }

    /// Returns true if the tensor requires gradients during the backward pass.
    pub fn is_require_grad(&self) -> bool {
        B::is_require_grad(&self.primitive)
    }

    /// Mark the tensor as tracked or untracked depending on the require grad argument.
    /// When tracked, the gradients will be available after the backward pass.
    ///
    /// This function does nothing when autodiff is not enabled.
    pub fn set_require_grad(self, require_grad: bool) -> Self {
        Self::new(B::set_require_grad(self.primitive, require_grad))
    }

    /// Applies the relu function to the tensor.
    pub(crate) fn relu(self) -> Self {
        Self::new(B::relu(self.primitive))
    }
}

impl<const D: usize, B: ADBackend> Tensor<B, D> {
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

    /// Remove the grad tensor from the [grads](ADBackend::Gradients) struct returning the result.
    pub fn grad_remove(&self, grads: &mut B::Gradients) -> Option<Tensor<B::InnerBackend, D>> {
        B::grad_remove(&self.primitive, grads).map(Tensor::new)
    }

    /// Replace the grad tensor from the [grads](ADBackend::Gradients) struct with the provided
    /// gradient.
    pub fn grad_replace(&self, grads: &mut B::Gradients, grad: Tensor<B::InnerBackend, D>) {
        B::grad_replace(&self.primitive, grads, grad.primitive);
    }

    /// Returns the inner tensor without the autodiff information.
    pub fn inner(self) -> Tensor<B::InnerBackend, D> {
        Tensor::new(B::inner(self.primitive))
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
    pub fn from_inner(inner: Tensor<B::InnerBackend, D>) -> Self {
        Self::new(B::from_inner(inner.primitive))
    }
}
