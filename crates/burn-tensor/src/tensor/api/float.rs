use crate::AsIndex;
use crate::FloatDType;
use crate::Tensor;
use crate::cast::ToElement;
use crate::check;
use crate::check::TensorCheck;
use crate::indexing::canonicalize_dim;
use crate::ops::InterpolateMode;
use crate::quantization::{QuantScheme, QuantizationParameters};
use crate::tensor::backend::Backend;
use crate::tensor::stats;
use crate::tensor::{Distribution, TensorData};
use crate::{Int, TensorPrimitive};

use super::Bool;

/// Default RTOL value for `is_close` and `all_close`.
pub const DEFAULT_RTOL: f64 = 1e-5;

/// Default ATOL value for `is_close` and `all_close`.
pub const DEFAULT_ATOL: f64 = 1e-8;

impl<const D: usize, B> Tensor<B, D>
where
    B: Backend,
{
    /// Applies element wise exponential operation.
    ///
    #[cfg_attr(doc, doc = "$y_i = e^{x_i}$")]
    #[cfg_attr(not(doc), doc = "`y = e^x`")]
    pub fn exp(self) -> Self {
        Self::new(TensorPrimitive::Float(B::float_exp(
            self.primitive.tensor(),
        )))
    }

    /// Applies element wise natural log operation *ln*.
    ///
    #[cfg_attr(doc, doc = r#"$y_i = \log_e\(x_i\)$"#)]
    #[cfg_attr(not(doc), doc = "`y_i = log(x_i)`")]
    pub fn log(self) -> Self {
        Self::new(TensorPrimitive::Float(B::float_log(
            self.primitive.tensor(),
        )))
    }

    /// Applies the natural logarithm of one plus the input tensor, element-wise.
    ///
    #[cfg_attr(doc, doc = r#"$y_i = \log_e\(x_i + 1\)$"#)]
    #[cfg_attr(not(doc), doc = "`y_i = log(x_i + 1)`")]
    pub fn log1p(self) -> Self {
        Self::new(TensorPrimitive::Float(B::float_log1p(
            self.primitive.tensor(),
        )))
    }

    /// Applies the [error function](https://en.wikipedia.org/wiki/Error_function) element wise.
    ///
    #[cfg_attr(
        doc,
        doc = r#"
$y_i = \text{erf}\(x_i\)$

The error function is defined as:

$$\text{erf}\(x\) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} dt$$
"#
    )]
    #[cfg_attr(not(doc), doc = "`y_i = erf(x_i)`")]
    pub fn erf(self) -> Self {
        Self::new(TensorPrimitive::Float(B::float_erf(
            self.primitive.tensor(),
        )))
    }

    /// Applies [reciprocal operation](https://en.wikipedia.org/wiki/Multiplicative_inverse)
    /// (or multiplicative inverse) element wise.
    ///
    #[cfg_attr(doc, doc = r#"$y_i = \frac{1}{x_i}$"#)]
    #[cfg_attr(not(doc), doc = "`y_i = 1/x_i`")]
    pub fn recip(self) -> Self {
        Self::new(TensorPrimitive::Float(B::float_recip(
            self.primitive.tensor(),
        )))
    }

    /// Applies element wise square operation.
    ///
    #[cfg_attr(doc, doc = r#"$y_i = x_i * x_i$"#)]
    #[cfg_attr(not(doc), doc = "`y_i = x_i * x_i`")]
    pub fn square(self) -> Self {
        self.powi_scalar(2)
    }

    /// Applies element wise root square operation.
    ///
    #[cfg_attr(doc, doc = r#"$y_i = \sqrt{x_i}$"#)]
    #[cfg_attr(not(doc), doc = "`y_i = sqrt(x_i)`")]
    pub fn sqrt(self) -> Self {
        Self::new(TensorPrimitive::Float(B::float_sqrt(
            self.primitive.tensor(),
        )))
    }

    /// Applies element wise cosine operation.
    ///
    #[cfg_attr(doc, doc = r#"$y_i = \cos\(x_i\)$"#)]
    #[cfg_attr(not(doc), doc = "`y_i = cos(x_i)`")]
    pub fn cos(self) -> Self {
        Self::new(TensorPrimitive::Float(B::float_cos(
            self.primitive.tensor(),
        )))
    }

    /// Applies element wise sine operation.
    ///
    #[cfg_attr(doc, doc = r#"$y_i = \sin\(x_i\)$"#)]
    #[cfg_attr(not(doc), doc = "`y_i = sin(x_i)`")]
    pub fn sin(self) -> Self {
        Self::new(TensorPrimitive::Float(B::float_sin(
            self.primitive.tensor(),
        )))
    }

    /// Applies element wise tangent operation.
    ///
    #[cfg_attr(doc, doc = r#"$y_i = \tan\(x_i\)$"#)]
    #[cfg_attr(not(doc), doc = "`y_i = tan(x_i)`")]
    pub fn tan(self) -> Self {
        Self::new(TensorPrimitive::Float(B::float_tan(
            self.primitive.tensor(),
        )))
    }

    /// Applies element wise hyperbolic cosine operation.
    ///
    #[cfg_attr(doc, doc = r#"$y_i = \cosh\(x_i\)$"#)]
    #[cfg_attr(not(doc), doc = "`y_i = cosh(x_i)`")]
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///
    ///     let tensor = Tensor::<B, 1>::from_data([0.0, -1.0, 2.0], &device);
    ///     println!("{}", tensor.cosh()); // [1.0, 1.5430, 3.7621]
    /// }
    /// ```
    pub fn cosh(self) -> Self {
        Self::new(TensorPrimitive::Float(B::float_cosh(
            self.primitive.tensor(),
        )))
    }

    /// Applies element wise hyperbolic sine operation.
    ///
    #[cfg_attr(doc, doc = r#"$y_i = \sinh\(x_i\)$"#)]
    #[cfg_attr(not(doc), doc = "`y_i = sinh(x_i)`")]
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///
    ///     let tensor = Tensor::<B, 1>::from_data([0.0, -1.0, 2.0], &device);
    ///     println!("{}", tensor.sinh()); // [0.0, -1.1752, 3.6269]
    /// }
    /// ```
    pub fn sinh(self) -> Self {
        Self::new(TensorPrimitive::Float(B::float_sinh(
            self.primitive.tensor(),
        )))
    }

    /// Applies element wise hyperbolic tangent operation.
    ///
    #[cfg_attr(doc, doc = r#"$y_i = \tanh\(x_i\)$"#)]
    #[cfg_attr(not(doc), doc = "`y_i = tanh(x_i)`")]
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///
    ///     let tensor = Tensor::<B, 1>::from_data([0.0, -1.0, 2.0], &device);
    ///     println!("{}", tensor.tanh()); // [0.0, -0.7616, 0.9640]
    /// }
    /// ```
    pub fn tanh(self) -> Self {
        Self::new(TensorPrimitive::Float(B::float_tanh(
            self.primitive.tensor(),
        )))
    }

    /// Applies element wise round operation.
    ///
    /// This function implements the [round half to even](https://en.wikipedia.org/wiki/Rounding#Rounding_half_to_even)
    /// strategy, with halfway cases rounded to the nearest even integer value.
    pub fn round(self) -> Self {
        Self::new(TensorPrimitive::Float(B::float_round(
            self.primitive.tensor(),
        )))
    }

    /// Applies element wise floor operation.
    pub fn floor(self) -> Self {
        Self::new(TensorPrimitive::Float(B::float_floor(
            self.primitive.tensor(),
        )))
    }

    /// Applies element wise ceil operation.
    pub fn ceil(self) -> Self {
        Self::new(TensorPrimitive::Float(B::float_ceil(
            self.primitive.tensor(),
        )))
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
    pub fn from_floats<A: Into<TensorData>>(floats: A, device: &B::Device) -> Self {
        Self::from_data(floats.into().convert::<f32>(), device)
    }

    /// Returns a new tensor with the same shape and device as the current tensor and the data
    /// cast to Integer.
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
        Tensor::new(B::float_into_int(self.primitive.tensor()))
    }

    /// Returns a new tensor with the same shape, dtype, and device as the current tensor filled random
    /// values sampled from the given distribution.
    pub fn random_like(&self, distribution: Distribution) -> Self {
        Self::new(TensorPrimitive::Float(B::float_random(
            self.shape(),
            distribution,
            &self.device(),
        )))
        .cast(self.dtype())
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

    /// Converts a tensor to the specified floating point data type.
    ///
    /// This is always a no-op when casting to the current dtype.
    ///
    /// # Warning
    /// Most backends don't have automatic type promotion at this time, so make sure that all tensors
    /// have the same floating point precision data type for operations multiple input tensors (e.g., binary ops).
    pub fn cast<F: Into<FloatDType>>(self, dtype: F) -> Tensor<B, D> {
        let dtype = dtype.into();
        let self_type: FloatDType = self.dtype().into();
        if dtype == self_type {
            // no-op.
            return self;
        }

        Tensor::new(TensorPrimitive::Float(B::float_cast(
            self.primitive.tensor(),
            dtype,
        )))
    }

    /// Detach the current tensor from the autodiff graph.
    ///
    /// This function does nothing when autodiff is not enabled.
    /// This can be used in batchers or elsewhere to ensure that previous operations are not
    /// considered in the autodiff graph.
    pub fn detach(self) -> Self {
        Self::new(TensorPrimitive::Float(B::float_detach(
            self.primitive.tensor(),
        )))
    }

    /// Mark the tensor to keep gradients during the backward pass.
    ///
    /// This function does nothing when autodiff is not enabled.
    pub fn require_grad(self) -> Self {
        self.set_require_grad(true)
    }

    /// Returns true if the tensor requires gradients during the backward pass.
    pub fn is_require_grad(&self) -> bool {
        match &self.primitive {
            TensorPrimitive::Float(tensor) => B::float_is_require_grad(tensor),
            TensorPrimitive::QFloat(tensor) => B::q_is_require_grad(tensor),
        }
    }

    /// Mark the tensor as tracked or untracked depending on the require_grad argument.
    /// When tracked, the gradients will be available after the backward pass.
    ///
    /// This function does nothing when autodiff is not enabled.
    pub fn set_require_grad(self, require_grad: bool) -> Self {
        let primitive = match self.primitive {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(B::float_set_require_grad(tensor, require_grad))
            }
            TensorPrimitive::QFloat(tensor) => {
                TensorPrimitive::QFloat(B::q_set_require_grad(tensor, require_grad))
            }
        };
        Self::new(primitive)
    }

    /// Applies the relu function to the tensor.
    pub(crate) fn relu(self) -> Self {
        Self::new(TensorPrimitive::Float(B::relu(self.primitive.tensor())))
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

    /// Convert the tensor to a lower precision data type based on the quantization scheme.
    ///
    /// # Arguments
    ///
    /// * `scheme` - The quantization scheme.
    /// * `qparams` - The pre-computed quantization parameters.
    ///
    /// # Returns
    ///
    /// The quantized tensor.
    pub fn quantize(
        self,
        scheme: &QuantScheme,
        qparams: QuantizationParameters<B>,
    ) -> Tensor<B, D> {
        Tensor::new(TensorPrimitive::QFloat(B::quantize(
            self.primitive.tensor(),
            scheme,
            qparams.into(),
        )))
    }

    /// Dynamically convert the tensor to a lower precision data type based on the quantization scheme.
    ///
    /// # Arguments
    ///
    /// * `scheme` - The quantization scheme.
    ///
    /// # Returns
    ///
    /// The quantized tensor.
    ///
    /// # Notes
    /// This uses [min-max calibration](crate::quantization::Calibration::MinMax).
    pub fn quantize_dynamic(self, scheme: &QuantScheme) -> Tensor<B, D> {
        Tensor::new(TensorPrimitive::QFloat(B::quantize_dynamic(
            self.primitive.tensor(),
            scheme,
        )))
    }

    /// Convert the tensor back to a higher precision data type.
    ///
    /// If the tensor is not quantized, its value is simply returned.
    ///
    /// # Returns
    ///
    /// The dequantized tensor.
    pub fn dequantize(self) -> Tensor<B, D> {
        Tensor::new(TensorPrimitive::Float(self.primitive.tensor()))
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
    /// * `rtol` - Optional relative tolerance. Default is 1e-5; see `DEFAULT_RTOL`.
    /// * `atol` - Optional absolute tolerance. Default is 1e-8; see `DEFAULT_ATOL`.
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
        let rtol = rtol.unwrap_or(DEFAULT_RTOL);
        let atol = atol.unwrap_or(DEFAULT_ATOL);

        // check finite difference is close
        let is_close_finite_val = self
            .clone()
            .sub(other.clone())
            .abs()
            .lower_equal(other.clone().abs().mul_scalar(rtol).add_scalar(atol))
            .bool_and(self.clone().is_finite())
            .bool_and(other.clone().is_finite());

        // check if both are infinite and have same sign
        let inf_same_sign = self
            .clone()
            .is_finite()
            .bool_not()
            .bool_and(other.clone().is_finite().bool_not())
            .bool_and(self.equal(other));

        is_close_finite_val.bool_or(inf_same_sign)
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
    /// * `rtol` - Optional relative tolerance. Default is 1e-5; see `DEFAULT_RTOL`.
    /// * `atol` - Optional absolute tolerance. Default is 1e-8; see `DEFAULT_ATOL`.
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
        self.is_close(other, rtol, atol)
            .all()
            .into_scalar()
            .to_bool()
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
    pub fn is_nan(self) -> Tensor<B, D, Bool> {
        Tensor::new(B::float_is_nan(self.primitive.tensor()))
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
    pub fn contains_nan(self) -> Tensor<B, 1, Bool> {
        // Summing the tensor will result in NaN if the tensor contains any NaN values
        // This is faster than checking each element individually
        // because it rolls up the NaN values into a single value
        let sum = self.sum();

        sum.is_nan()
    }

    /// Returns a new tensor with boolean elements indicating whether each element of the input is infinite (either +INF or -INF).
    ///
    /// # Returns
    ///
    /// A boolean tensor where `true` indicates that the value is infinite
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Bool, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::from_data([[1.0, f64::INFINITY, 3.0], [f64::NAN, 9.0, 6.0]], &device);
    ///    let tensor = tensor.is_finite();
    ///    println!("{tensor}");
    ///    // [[false, true, false], [false, false, false]]
    /// }
    /// ```
    pub fn is_inf(self) -> Tensor<B, D, Bool> {
        Tensor::new(B::float_is_inf(self.primitive.tensor()))
    }

    /// Returns a new tensor with boolean elements indicating whether each element of the input is finite
    ///
    /// # Returns
    ///
    /// A boolean tensor where `true` indicates that the value is finite and `false` indicates
    /// either INF, -INF or NAN
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Bool, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::from_data([[1.0, f64::INFINITY, 3.0], [f64::NAN, 9.0, 6.0]], &device);
    ///    let tensor = tensor.is_finite();
    ///    println!("{tensor}");
    ///    // [[true, false, true], [false, true, true]]
    /// }
    /// ```
    pub fn is_finite(self) -> Tensor<B, D, Bool> {
        self.clone()
            .is_nan()
            .bool_not()
            .bool_and(self.is_inf().bool_not())
    }

    /// Samples tensor as a two-dimensional spatial grid of (possibly multi-channel) values,
    /// using the given locations in [-1, 1].
    ///
    /// Interpolation is bilinear.
    /// Padding is border: out of bounds locations will be clamped to the nearest border
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor being sampled from, shape (N, C, H_in, W_in)
    /// * `grid` - A tensor of locations, with shape (N, H_out, W_out, 2). Values are [-1, 1].
    ///   A [x = -1, y = -1] means top-left, and [x = 1, y = 1] means bottom-right
    /// * `method` - How to interpolate between samples
    ///
    /// # Returns
    ///
    /// A tensor with shape (N, C, H_out, W_out)
    pub fn grid_sample_2d(self, grid: Tensor<B, D>, method: InterpolateMode) -> Tensor<B, D> {
        Tensor::new(TensorPrimitive::Float(B::float_grid_sample_2d(
            self.primitive.tensor(),
            grid.primitive.tensor(),
            method,
        )))
    }

    /// Computes the cross product of `self` and another tensor along a given dimension.
    ///
    /// Both `self` and `other` **must have size 3** along the specified `dim`,
    /// because the cross product is only defined in three-dimensional space.
    ///
    /// # Arguments
    ///
    /// * `other` - The other tensor to take the cross product with.
    /// * `dim`   - The dimension along which to compute the cross product.
    ///
    /// # Returns
    ///
    /// A tensor containing the cross product of `self` and `other` along `dim`.
    pub fn cross<Dim: AsIndex>(self, other: Tensor<B, D>, dim: Dim) -> Tensor<B, D> {
        let dim = canonicalize_dim(dim, D, false);
        check!(TensorCheck::cross(&self, &other, dim));
        Tensor::new(TensorPrimitive::Float(B::float_cross(
            self.primitive.tensor(),
            other.primitive.tensor(),
            dim,
        )))
    }
}
