use crate::AsIndex;
use crate::Cast;
use crate::Tensor;
use crate::cast::ToElement;
use crate::check;
use crate::check::TensorCheck;
use crate::ops::GridSampleOptions;
use crate::quantization::{QuantScheme, QuantizationParameters};
use crate::tensor::backend::Backend;
use crate::tensor::stats;
use crate::tensor::{Distribution, TensorData};
use crate::{Bool, Float, Int, TensorPrimitive};
#[cfg(feature = "distributed")]
use burn_backend::AutodiffBackend;
use burn_backend::ElementConversion;
use burn_backend::Scalar;
use burn_backend::TensorMetadata;
#[cfg(feature = "distributed")]
use burn_backend::distributed::DistributedParamId;
use burn_backend::get_device_settings;
use burn_backend::tensor::FloatMathOps;
use burn_backend::tensor::quantization::QuantizationParametersPrimitive;
use core::f32;

/// Default RTOL value for `is_close` and `all_close`.
pub const DEFAULT_RTOL: f64 = 1e-5;

/// Default ATOL value for `is_close` and `all_close`.
pub const DEFAULT_ATOL: f64 = 1e-8;

impl<const D: usize, B> Tensor<B, D>
where
    B: Backend,
{
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

    /// Converts each of the elements of the input tensor from angles in degrees to radians.
    ///
    /// # Example
    /// ```ignore
    /// let tensor_in_radians = tensor.deg2rad();
    /// ```
    pub fn deg2rad(self) -> Self {
        self.mul_scalar(f32::consts::PI / 180.0)
    }

    /// Converts each of the elements of the input tensor from angles in radians to degrees.
    ///
    /// # Example
    /// ```ignore
    /// let tensor_in_degrees = tensor.rad2deg();
    /// ```
    pub fn rad2deg(self) -> Self {
        self.mul_scalar(180.0 / f32::consts::PI)
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
        let out_dtype = get_device_settings::<B>(&self.device()).int_dtype;
        Tensor::new(B::float_into_int(self.primitive.tensor(), out_dtype))
    }

    /// Returns a new tensor with the same shape, dtype, and device as the current tensor filled random
    /// values sampled from the given distribution.
    pub fn random_like(&self, distribution: Distribution) -> Self {
        Self::new(TensorPrimitive::Float(B::float_random(
            self.shape(),
            distribution,
            &self.device(),
            self.dtype().into(),
        )))
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

    /// Returns the median value along the specified dimension.
    ///
    /// The median is not unique for input tensors with an even number of elements
    /// in the reduced dimension. In this case, the lower of the two medians is returned,
    /// following PyTorch's behavior.
    ///
    /// # Note
    ///
    /// The current implementation performs a full sort along the specified dimension,
    /// which has O(nlog(n)) complexity. Additionally, most backends currently fall back
    /// to CPU for the sort operation, which may result in slower performance compared
    /// to native GPU operations.
    ///
    /// # Arguments
    ///
    /// - `dim` - The dimension along which to compute the median.
    ///
    /// # Returns
    ///
    /// - A tensor containing the median values along the specified dimension.
    ///
    /// # Example 1
    ///
    /// ```ignore
    /// // Assuming backend B
    /// let device = B::Device::default();
    /// let tensor = Tensor::<B, 2>::from_data(
    ///     [[1.0, 5.0, 3.0, 2.0], [8.0, 4.0, 6.0, 7.0]],
    ///     &device,
    /// );
    ///
    /// // Median along dimension 0:
    /// // sorted columns are [1.0, 8.0], [4.0, 5.0], [3.0, 6.0], [2.0, 7.0]
    /// let median = tensor.median(0);
    /// // Result: [[1.0, 4.0, 3.0, 2.0]]
    ///
    /// // Median along dimension 1:
    /// // sorted rows are [1.0, 2.0, 3.0, 5.0] and [4.0, 6.0, 7.0, 8.0]
    /// let median = tensor.median(1);
    /// // Result: [[2.0], [6.0]]
    /// ```
    ///
    /// # Example 2
    ///
    /// The median across all elements can be calculated as follows:
    ///
    /// ```ignore
    /// // D is the number of dimensions of the tensor
    /// let flattened_tensor: Tensor<B, 1> = tensor.flatten(0, D - 1);
    ///
    /// // Calculate median for dim 0 since the tensor has become 1 dimensional
    /// let median = flattened_tensor.median(0);
    /// // Result: [4.0]
    /// ```
    pub fn median(self, dim: usize) -> Self {
        // TODO: Allow backend specialization. Optimally, implement a median kernel for cubecl
        // instead of leveraging a full sort to get the median.
        stats::median(self, dim)
    }

    /// Returns the median value along the specified dimension and its index.
    ///
    /// The median is not unique for input tensors with an even number of elements
    /// in the reduced dimension. In this case, the lower of the two medians is returned,
    /// following PyTorch's behavior.
    ///
    /// # Note
    ///
    /// The current implementation performs a full sort along the specified dimension,
    /// which has O(nlog(n)) complexity. Additionally, most backends currently fall back
    /// to CPU for the sort operation, which may result in slower performance compared
    /// to native GPU operations.
    ///
    /// # Arguments
    ///
    /// - `dim` - The dimension along which to compute the median.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - A tensor with the median values.
    /// - A tensor with the indices of the median values in the original tensor.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Assuming backend B
    /// let device = B::Device::default();
    /// let tensor = Tensor::<B, 2>::from_data(
    ///     [[1.0, 5.0, 3.0, 2.0], [8.0, 4.0, 6.0, 7.0]],
    ///     &device,
    /// );
    ///
    /// // Median along dimension 1:
    /// // sorted rows are [1.0, 2.0, 3.0, 5.0] and [4.0, 6.0, 7.0, 8.0]
    /// let (values, indices) = tensor.median_with_indices(1);
    /// // values: [[2.0], [6.0]], indices: [[3], [2]] (position in the original tensor)
    /// ```
    pub fn median_with_indices(self, dim: usize) -> (Self, Tensor<B, D, Int>) {
        // TODO: Allow backend specialization. Optimally, implement a median kernel for cubecl
        // instead of leveraging a full sort to get the median.
        stats::median_with_indices(self, dim)
    }

    /// Converts a tensor to the specified data type.
    ///
    /// Supports both within-kind casting (e.g., `FloatDType::F64`) and cross-kind casting
    /// (e.g., `IntDType::I64` to produce an int tensor).
    ///
    /// This is a no-op when casting to the current dtype within the same kind.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, FloatDType, IntDType};
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///     let float_tensor = Tensor::<B, 1>::from_floats([1.0, 2.5], &device);
    ///
    ///     // Within-kind cast (float to float)
    ///     let f64_tensor = float_tensor.clone().cast(FloatDType::F64);
    ///
    ///     // Cross-kind cast (float to int)
    ///     let int_tensor = float_tensor.cast(IntDType::I64);
    /// }
    /// ```
    #[must_use]
    pub fn cast<T: Cast<B, Float>>(self, dtype: T) -> Tensor<B, D, T::OutputKind> {
        Tensor::new(T::cast(self.primitive, dtype))
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
            QuantizationParametersPrimitive {
                scales: qparams.scales.primitive.tensor(),
            },
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
        let out_dtype = get_device_settings::<B>(&self.device()).bool_dtype;
        Tensor::new(B::float_is_nan(self.primitive.tensor(), out_dtype))
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
        let out_dtype = get_device_settings::<B>(&self.device()).bool_dtype;
        Tensor::new(B::float_is_inf(self.primitive.tensor(), out_dtype))
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
    /// # Arguments
    ///
    /// * `grid` - A tensor of locations, with shape (N, H_out, W_out, 2). Values are [-1, 1].
    ///   A [x = -1, y = -1] means top-left, and [x = 1, y = 1] means bottom-right
    /// * `options` - Grid sampling options (mode, padding_mode, align_corners)
    ///
    /// # Returns
    ///
    /// A tensor with shape (N, C, H_out, W_out)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use burn_tensor::ops::{GridSampleOptions, GridSamplePaddingMode, InterpolateMode};
    ///
    /// // Default options (bilinear, zeros padding, align_corners=false)
    /// let output = tensor.grid_sample_2d(grid, GridSampleOptions::default());
    ///
    /// // Custom options
    /// let options = GridSampleOptions::new(InterpolateMode::Bilinear)
    ///     .with_padding_mode(GridSamplePaddingMode::Border)
    ///     .with_align_corners(true);
    /// let output = tensor.grid_sample_2d(grid, options);
    /// ```
    pub fn grid_sample_2d(
        self,
        grid: Tensor<B, D>,
        options: impl Into<GridSampleOptions>,
    ) -> Tensor<B, D> {
        Tensor::new(TensorPrimitive::Float(B::float_grid_sample_2d(
            self.primitive.tensor(),
            grid.primitive.tensor(),
            options.into(),
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
        let dim = dim.expect_dim_index(D);
        check!(TensorCheck::cross(&self, &other, dim));
        Tensor::new(TensorPrimitive::Float(B::float_cross(
            self.primitive.tensor(),
            other.primitive.tensor(),
            dim,
        )))
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
        let primitive = match (self.primitive, other.primitive) {
            (TensorPrimitive::Float(lhs), TensorPrimitive::Float(rhs)) => {
                TensorPrimitive::Float(B::float_powf(lhs, rhs))
            }
            (TensorPrimitive::QFloat(lhs), TensorPrimitive::QFloat(rhs)) => B::q_powf(lhs, rhs),
            (TensorPrimitive::QFloat(lhs), TensorPrimitive::Float(rhs)) => {
                let dtype = rhs.dtype();
                TensorPrimitive::Float(B::float_powf(B::dequantize(lhs, dtype.into()), rhs))
            }
            (TensorPrimitive::Float(lhs), TensorPrimitive::QFloat(rhs)) => {
                let dtype = lhs.dtype();
                TensorPrimitive::Float(B::float_powf(lhs, B::dequantize(rhs, dtype.into())))
            }
        };

        Tensor::new(primitive)
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
        let rhs = Scalar::new(other, &self.dtype());

        let primitive = match self.primitive {
            TensorPrimitive::Float(lhs) => TensorPrimitive::Float(B::float_powf_scalar(lhs, rhs)),
            TensorPrimitive::QFloat(lhs) => B::q_powf_scalar(lhs, rhs),
        };

        Tensor::new(primitive)
    }
}

impl<const D: usize, B: Backend> Tensor<B, D> {
    /// Draws samples from a categorical distribution defined by the last dimension
    /// of the input tensor.
    ///
    /// The last dimension is treated as a (possibly unnormalized) set of weights
    /// defining a categorical distribution over categories. All leading dimensions
    /// are treated as batch dimensions. The method returns integer indices of the
    /// sampled categories.
    ///
    /// # Arguments
    ///
    /// * `num_samples` - Number of samples to draw per distribution. Must be >= 1.
    ///
    /// # Panics
    ///
    /// Panics if `num_samples` is 0.
    ///
    /// # Note
    ///
    /// Distributions with all-zero weights produce undefined (NaN-based) sampling
    /// results. Callers should ensure each distribution has at least one positive
    /// weight.
    ///
    /// # Returns
    ///
    /// An integer tensor with the same shape as the input, except the last dimension
    /// is replaced by `num_samples`, containing sampled category indices in
    /// `[0, num_categories)`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///     let device = B::Device::default();
    ///     let probs = Tensor::<B, 2>::from_floats(
    ///         [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    ///         &device,
    ///     );
    ///     let samples = probs.categorical(4);
    ///     // First row always samples index 1, second row always samples index 2
    ///     println!("{samples}");
    /// }
    /// ```
    pub fn categorical(self, num_samples: usize) -> Tensor<B, D, Int> {
        assert!(num_samples > 0, "categorical: num_samples must be >= 1");

        let shape = self.shape();
        let num_categories = shape[D - 1];
        let batch_size = (shape.num_elements() / num_categories).max(1);
        let device = self.device();

        // Flatten leading dimensions into a single batch dimension: [batch, categories]
        let flat: Tensor<B, 2> = self.reshape([batch_size, num_categories]);

        // Normalize weights to probabilities
        let sum = flat.clone().sum_dim(1); // [batch, 1]
        let probs = flat / sum;

        // Cumulative sum along categories dimension
        let cumsum = probs.cumsum(1); // [batch, categories]

        // Uniform random values for each sample
        let uniform = Tensor::<B, 2>::random(
            [batch_size, num_samples],
            Distribution::Uniform(0.0, 1.0),
            &device,
        ); // [batch, num_samples]

        // Expand dimensions for broadcasting:
        //   cumsum: [batch, categories, 1]
        //   uniform: [batch, 1, num_samples]
        let cumsum_3d: Tensor<B, 3> = cumsum.unsqueeze_dim(2);
        let uniform_3d: Tensor<B, 3> = uniform.unsqueeze_dim(1);

        // Count categories where cumsum < uniform (inverse CDF)
        let mask: Tensor<B, 3, Bool> = cumsum_3d.lower(uniform_3d);
        let indices: Tensor<B, 2, Int> = mask.int().sum_dim(1).squeeze_dim::<2>(1);

        // Clamp to valid range to guard against floating-point imprecision in cumsum
        let indices = indices.clamp(0, num_categories as i64 - 1);

        // Reshape back to [...leading_dims, num_samples]
        let mut out_shape = shape;
        out_shape[D - 1] = num_samples;
        indices.reshape(out_shape)
    }
}

#[cfg(feature = "distributed")]
impl<const D: usize, B> Tensor<B, D>
where
    B: AutodiffBackend,
{
    /// Returns true if the tensor is marked as distributed.
    pub fn is_distributed(&self) -> bool {
        match &self.primitive {
            TensorPrimitive::Float(tensor) => B::is_distributed(tensor),
            TensorPrimitive::QFloat(_) => unimplemented!(),
        }
    }

    /// Mark the tensor as distributed.
    ///
    /// This function does nothing when autodiff or distributed is not enabled.
    pub fn set_distributed(self, param_id: DistributedParamId) -> Self {
        let primitive = match self.primitive {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(B::set_distributed_params(tensor, param_id))
            }
            TensorPrimitive::QFloat(_) => unimplemented!(),
        };
        Self::new(primitive)
    }
}

impl<B, const D: usize, K> Tensor<B, D, K>
where
    B: Backend,
    K: FloatMathOps<B>,
{
    /// Applies element wise square operation.
    ///
    #[cfg_attr(doc, doc = r#"$y_i = x_i * x_i$"#)]
    #[cfg_attr(not(doc), doc = "`y_i = x_i * x_i`")]
    pub fn square(self) -> Self {
        Self::new(K::square(self.primitive))
    }

    /// Applies element wise exponential operation.
    ///
    #[cfg_attr(doc, doc = r#"$y_i = e^{x_i}$"#)]
    #[cfg_attr(not(doc), doc = "`y = e^x`")]
    pub fn exp(self) -> Self {
        Self::new(K::exp(self.primitive))
    }

    /// Applies element wise natural logarithm of one plus the input tensor.
    ///
    #[cfg_attr(doc, doc = r#"$y_i = \log_e\(x_i + 1\)$"#)]
    #[cfg_attr(not(doc), doc = "`y_i = log1p(x_i)`")]
    pub fn log1p(self) -> Self {
        Self::new(K::log1p(self.primitive))
    }

    /// Applies element wise natural log operation *ln*.
    ///
    #[cfg_attr(doc, doc = r#"$y_i = \log_e\(x_i\)$"#)]
    #[cfg_attr(not(doc), doc = "`y_i = log(x_i)`")]
    pub fn log(self) -> Self {
        Self::new(K::log(self.primitive))
    }

    /// Applies element wise square root operation.
    ///
    pub fn sqrt(self) -> Self {
        Tensor::new(K::sqrt(self.primitive))
    }
    /// Applies element wise cosine operation.
    ///
    #[cfg_attr(doc, doc = r#"$y_i = \cos\(x_i\)$"#)]
    #[cfg_attr(not(doc), doc = "`y_i = cos(x_i)`")]
    pub fn cos(self) -> Self {
        Tensor::new(K::cos(self.primitive))
    }

    /// Applies element wise sine operation.
    ///
    #[cfg_attr(doc, doc = r#"$y_i = \sin\(x_i\)$"#)]
    #[cfg_attr(not(doc), doc = "`y_i = sin(x_i)`")]
    pub fn sin(self) -> Self {
        Tensor::new(K::sin(self.primitive))
    }

    /// Applies element wise tangent operation.
    ///
    #[cfg_attr(doc, doc = r#"$y_i = \tan\(x_i\)$"#)]
    #[cfg_attr(not(doc), doc = "`y_i = tan(x_i)`")]
    pub fn tan(self) -> Self {
        Tensor::new(K::tan(self.primitive))
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
        Tensor::new(K::cosh(self.primitive))
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
        Tensor::new(K::sinh(self.primitive))
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
        Tensor::new(K::tanh(self.primitive))
    }

    /// Applies element wise inverse cosine operation.
    ///
    #[cfg_attr(doc, doc = r#"$y_i = \acos\(x_i\)$"#)]
    #[cfg_attr(not(doc), doc = "`y_i = acos(x_i)`")]
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
    ///     let tensor = Tensor::<B, 1>::from_data([0.0, -1.0, 1.0], &device);
    ///     println!("{}", tensor.acos()); // [1.5708, 3.1416, 0.0]
    /// }
    /// ```
    pub fn acos(self) -> Self {
        Tensor::new(K::acos(self.primitive))
    }

    /// Applies element wise inverse hyperbolic cosine operation.
    ///
    #[cfg_attr(doc, doc = r#"$y_i = \acosh\(x_i\)$"#)]
    #[cfg_attr(not(doc), doc = "`y_i = acosh(x_i)`")]
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
    ///     let tensor = Tensor::<B, 1>::from_data([1.0, 2.0, 3.0], &device);
    ///     println!("{}", tensor.acosh()); // [0.0000, 1.3170, 1.7627]
    /// }
    /// ```
    pub fn acosh(self) -> Self {
        Tensor::new(K::acosh(self.primitive))
    }

    /// Applies element wise inverse sine operation.
    ///
    #[cfg_attr(doc, doc = r#"$y_i = \asin\(x_i\)$"#)]
    #[cfg_attr(not(doc), doc = "`y_i = asin(x_i)`")]
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
    ///     let tensor = Tensor::<B, 1>::from_data([0.0, -1.0, 1.0], &device);
    ///     println!("{}", tensor.asin()); // [ 0.0000, -1.5708,  1.5708]
    /// }
    /// ```
    pub fn asin(self) -> Self {
        Tensor::new(K::asin(self.primitive))
    }

    /// Applies element wise inverse hyperbolic sine operation.
    ///
    #[cfg_attr(doc, doc = r#"$y_i = \asinh\(x_i\)$"#)]
    #[cfg_attr(not(doc), doc = "`y_i = asinh(x_i)`")]
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
    ///     let tensor = Tensor::<B, 1>::from_data([0.0, -1.0, 1.0], &device);
    ///     println!("{}", tensor.asinh()); // [ 0.0000, -0.8814,  0.8814]
    /// }
    /// ```
    pub fn asinh(self) -> Self {
        Tensor::new(K::asinh(self.primitive))
    }

    /// Applies element wise inverse tangent operation.
    ///
    #[cfg_attr(doc, doc = r#"$y_i = \atan\(x_i\)$"#)]
    #[cfg_attr(not(doc), doc = "`y_i = atan(x_i)`")]
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
    ///     println!("{}", tensor.atan()); // [ 0.0, -0.7854,  1.1071]
    /// }
    /// ```
    pub fn atan(self) -> Self {
        Tensor::new(K::atan(self.primitive))
    }

    /// Applies element wise inverse hyperbolic tangent operation.
    ///
    #[cfg_attr(doc, doc = r#"$y_i = \atanh\(x_i\)$"#)]
    #[cfg_attr(not(doc), doc = "`y_i = atanh(x_i)`")]
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
    ///     let tensor = Tensor::<B, 1>::from_data([0.0, -0.5, 0.5], &device);
    ///     println!("{}", tensor.atanh()); // [ 0.0, -0.5493,  0.5493]
    /// }
    /// ```
    pub fn atanh(self) -> Self {
        Tensor::new(K::atanh(self.primitive))
    }

    /// Applies element wise inverse tangent operation using the signs of arguments to determine the correct quadrant.
    ///
    #[cfg_attr(doc, doc = r#"$z_i = \atan2\(y_i, x_i\)$"#)]
    #[cfg_attr(not(doc), doc = "`z_i = atan2(y_i, x_i)`")]
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
    ///     let lhs = Tensor::<B, 1>::from_data([-2.0, 2.0, -2.0], &device);
    ///     let rhs = Tensor::<B, 1>::from_data([1.0, -1.0, -1.0], &device);
    ///     println!("{}", lhs.atan2(rhs)); // [-1.1071,  2.0344, -2.0344]
    /// }
    /// ```
    pub fn atan2(self, other: Self) -> Self {
        Tensor::new(K::atan2(self.primitive, other.primitive))
    }
}
