use alloc::vec::Vec;
use burn_backend::{
    Backend, DType, Scalar, Shape, TensorData, TensorMetadata, get_or_init_device_settings,
    ops::{AttentionModuleOptions, GridSampleOptions},
    tensor::{BoolTensor, FloatTensor, IntTensor},
};
use burn_std::{FloatDType, IndexingUpdateOp, PadMode, Slice};

/// A zero-cost wrapper around a backend float tensor primitive.
///
/// `FuncTensor` exists so functional transformations can be written with ordinary tensor
/// syntax (`a * b + c`, `x.matmul(w)`, `x.relu()`) while the transformations themselves remain
/// generic over `B: Backend` and compose decorators such as `Autodiff<B>`, `ForwardDiff<B>`
/// and `Vmap<B>`.
///
/// The rank `D` is tracked in the type for shape checks and ergonomic APIs. It does not change
/// the runtime representation: the wrapper is exactly one backend primitive.
#[derive(Clone, Debug)]
pub struct FuncTensor<B: Backend, const D: usize = 1> {
    primitive: FloatTensor<B>,
}

impl<B: Backend, const D: usize> FuncTensor<B, D> {
    /// Wraps an existing backend float primitive.
    pub fn from_primitive(primitive: FloatTensor<B>) -> Self {
        Self { primitive }
    }

    /// Unwraps the backend float primitive.
    pub fn into_primitive(self) -> FloatTensor<B> {
        self.primitive
    }

    /// Borrows the backend float primitive.
    pub fn primitive(&self) -> &FloatTensor<B> {
        &self.primitive
    }

    /// Creates a tensor from data on the given device.
    ///
    /// # Panics
    ///
    /// Panics if the data rank does not match `D`.
    pub fn from_data<Data: Into<TensorData>>(data: Data, device: &B::Device) -> Self {
        let data = data.into();
        let shape = data.shape();
        assert_eq!(
            shape.num_dims(),
            D,
            "FuncTensor<_, {D}> requires rank-{D} data, got shape {shape:?}"
        );
        Self::from_primitive(B::float_from_data(data, device))
    }

    /// Reads the tensor data asynchronously.
    pub async fn into_data(self) -> TensorData {
        B::float_into_data(self.primitive)
            .await
            .expect("Failed to read FuncTensor data")
    }

    /// Reads the tensor data synchronously.
    ///
    /// # Panics
    ///
    /// Panics when the backend cannot complete the read synchronously.
    #[cfg(feature = "std")]
    pub fn into_data_sync(self) -> TensorData {
        burn_backend::try_read_sync(B::float_into_data(self.primitive))
            .expect("Failed to synchronously read FuncTensor data")
            .expect("Failed to read FuncTensor data")
    }

    /// Creates a tensor filled with zeros using the device's default float dtype.
    pub fn zeros<Sh: Into<Shape>>(shape: Sh, device: &B::Device) -> Self {
        let shape = shape.into();
        assert_rank::<D>(&shape, "FuncTensor::zeros");
        let dtype = get_or_init_device_settings::<B>(device).float_dtype;
        Self::from_primitive(B::float_zeros(shape, device, dtype))
    }

    /// Creates a tensor filled with ones using the device's default float dtype.
    pub fn ones<Sh: Into<Shape>>(shape: Sh, device: &B::Device) -> Self {
        let shape = shape.into();
        assert_rank::<D>(&shape, "FuncTensor::ones");
        let dtype = get_or_init_device_settings::<B>(device).float_dtype;
        Self::from_primitive(B::float_ones(shape, device, dtype))
    }

    /// Creates a tensor filled with `value` using the device's default float dtype.
    pub fn full<Sh: Into<Shape>>(shape: Sh, value: Scalar, device: &B::Device) -> Self {
        let shape = shape.into();
        assert_rank::<D>(&shape, "FuncTensor::full");
        let dtype = get_or_init_device_settings::<B>(device).float_dtype;
        Self::from_primitive(B::float_full(shape, value, device, dtype))
    }

    /// Creates a tensor filled with zeros using an explicit float dtype.
    pub fn zeros_dtype<Sh: Into<Shape>>(shape: Sh, device: &B::Device, dtype: FloatDType) -> Self {
        let shape = shape.into();
        assert_rank::<D>(&shape, "FuncTensor::zeros_dtype");
        Self::from_primitive(B::float_zeros(shape, device, dtype))
    }

    /// Creates a tensor filled with ones using an explicit float dtype.
    pub fn ones_dtype<Sh: Into<Shape>>(shape: Sh, device: &B::Device, dtype: FloatDType) -> Self {
        let shape = shape.into();
        assert_rank::<D>(&shape, "FuncTensor::ones_dtype");
        Self::from_primitive(B::float_ones(shape, device, dtype))
    }

    /// Returns the tensor shape.
    pub fn shape(&self) -> Shape {
        self.primitive.shape()
    }

    /// Returns the dimensions as an array.
    ///
    /// # Panics
    ///
    /// Panics if the runtime rank does not match `D`.
    pub fn dims(&self) -> [usize; D] {
        let shape = self.primitive.shape();
        assert_rank::<D>(&shape, "FuncTensor::dims");
        shape.dims()
    }

    /// Returns the tensor device.
    pub fn device(&self) -> B::Device {
        self.primitive.device()
    }

    /// Returns the tensor dtype.
    pub fn dtype(&self) -> DType {
        self.primitive.dtype()
    }

    /// Returns the number of elements.
    pub fn num_elements(&self) -> usize {
        self.primitive.shape().num_elements()
    }

    /// Detaches the tensor from any autodiff graph.
    pub fn detach(self) -> Self {
        Self::from_primitive(B::float_detach(self.primitive))
    }

    /// Sets whether gradients are required for this tensor.
    pub fn set_require_grad(self, require_grad: bool) -> Self {
        Self::from_primitive(B::float_set_require_grad(self.primitive, require_grad))
    }

    /// Returns whether gradients are required for this tensor.
    pub fn is_require_grad(&self) -> bool {
        B::float_is_require_grad(&self.primitive)
    }

    /// Matrix multiplication.
    pub fn matmul(self, other: Self) -> Self {
        Self::from_primitive(B::float_matmul(self.primitive, other.primitive))
    }

    /// Element-wise addition with broadcasting across potentially different ranks.
    pub fn add_broadcast<const D2: usize>(self, other: FuncTensor<B, D2>) -> Self {
        Self::from_primitive(B::float_add(self.primitive, other.primitive))
    }

    /// Matrix multiplication with broadcasting across potentially different ranks.
    pub fn matmul_broadcast<const D2: usize>(self, other: FuncTensor<B, D2>) -> Self {
        Self::from_primitive(B::float_matmul(self.primitive, other.primitive))
    }

    /// Transposes the last two dimensions.
    pub fn transpose(self) -> Self {
        Self::from_primitive(B::float_transpose(self.primitive))
    }

    /// Swaps two dimensions.
    pub fn swap_dims(self, dim1: usize, dim2: usize) -> Self {
        Self::from_primitive(B::float_swap_dims(self.primitive, dim1, dim2))
    }

    /// Permutes the dimensions.
    pub fn permute(self, axes: &[usize]) -> Self {
        Self::from_primitive(B::float_permute(self.primitive, axes))
    }

    /// Flips the elements along the given axes.
    pub fn flip(self, axes: &[usize]) -> Self {
        Self::from_primitive(B::float_flip(self.primitive, axes))
    }

    /// Reshapes the tensor.
    pub fn reshape<const D2: usize, Sh: Into<Shape>>(self, shape: Sh) -> FuncTensor<B, D2> {
        let shape = shape.into();
        assert_rank::<D2>(&shape, "FuncTensor::reshape");
        FuncTensor::from_primitive(B::float_reshape(self.primitive, shape))
    }

    /// Broadcasts the tensor to the given shape.
    pub fn expand<const D2: usize, Sh: Into<Shape>>(self, shape: Sh) -> FuncTensor<B, D2> {
        let shape = shape.into();
        assert_rank::<D2>(&shape, "FuncTensor::expand");
        FuncTensor::from_primitive(B::float_expand(self.primitive, shape))
    }

    /// Slices the tensor.
    pub fn slice(self, slices: &[Slice]) -> Self {
        Self::from_primitive(B::float_slice(self.primitive, slices))
    }

    /// Assigns values into a slice.
    ///
    /// The value may have a different rank than the tensor, as long as it matches the slice's
    /// output shape.
    pub fn slice_assign<const D2: usize>(self, slices: &[Slice], value: FuncTensor<B, D2>) -> Self {
        Self::from_primitive(B::float_slice_assign(
            self.primitive,
            slices,
            value.primitive,
        ))
    }

    /// Concatenates tensors along a dimension.
    pub fn cat(tensors: Vec<Self>, dim: usize) -> Self {
        assert!(
            !tensors.is_empty(),
            "FuncTensor::cat requires at least one tensor"
        );
        Self::from_primitive(B::float_cat(
            tensors.into_iter().map(|t| t.primitive).collect(),
            dim,
        ))
    }

    /// Pads the tensor.
    pub fn pad(self, padding: &[(usize, usize)], mode: PadMode) -> Self {
        Self::from_primitive(B::float_pad(self.primitive, padding, mode))
    }

    /// Gathers elements along a dimension.
    pub fn gather(self, dim: usize, indices: IntTensor<B>) -> Self {
        Self::from_primitive(B::float_gather(dim, self.primitive, indices))
    }

    /// Scatters values along a dimension.
    pub fn scatter(
        self,
        dim: usize,
        indices: IntTensor<B>,
        value: Self,
        update: IndexingUpdateOp,
    ) -> Self {
        Self::from_primitive(B::float_scatter(
            dim,
            self.primitive,
            indices,
            value.primitive,
            update,
        ))
    }

    /// Selects elements along a dimension.
    pub fn select(self, dim: usize, indices: IntTensor<B>) -> Self {
        Self::from_primitive(B::float_select(self.primitive, dim, indices))
    }

    /// Assigns selected elements along a dimension.
    pub fn select_assign(
        self,
        dim: usize,
        indices: IntTensor<B>,
        value: Self,
        update: IndexingUpdateOp,
    ) -> Self {
        Self::from_primitive(B::float_select_assign(
            self.primitive,
            dim,
            indices,
            value.primitive,
            update,
        ))
    }

    /// Repeats the tensor along a dimension.
    pub fn repeat_dim(self, dim: usize, times: usize) -> Self {
        Self::from_primitive(B::float_repeat_dim(self.primitive, dim, times))
    }

    /// Updates elements where the mask is true.
    pub fn mask_where(self, mask: BoolTensor<B>, value: Self) -> Self {
        Self::from_primitive(B::float_mask_where(self.primitive, mask, value.primitive))
    }

    /// Fills elements where the mask is true.
    pub fn mask_fill(self, mask: BoolTensor<B>, value: Scalar) -> Self {
        Self::from_primitive(B::float_mask_fill(self.primitive, mask, value))
    }

    /// Sums all elements.
    pub fn sum(self) -> FuncTensor<B, 1> {
        FuncTensor::from_primitive(B::float_sum(self.primitive))
    }

    /// Sums elements along a dimension.
    pub fn sum_dim(self, dim: usize) -> Self {
        Self::from_primitive(B::float_sum_dim(self.primitive, dim))
    }

    /// Sums elements along several dimensions.
    pub fn sum_dims(self, dims: &[usize]) -> Self {
        Self::from_primitive(B::float_sum_dims(self.primitive, dims))
    }

    /// Computes the mean of all elements.
    pub fn mean(self) -> FuncTensor<B, 1> {
        FuncTensor::from_primitive(B::float_mean(self.primitive))
    }

    /// Computes the mean along a dimension.
    pub fn mean_dim(self, dim: usize) -> Self {
        Self::from_primitive(B::float_mean_dim(self.primitive, dim))
    }

    /// Computes the product of all elements.
    pub fn prod(self) -> FuncTensor<B, 1> {
        FuncTensor::from_primitive(B::float_prod(self.primitive))
    }

    /// Computes the product along a dimension.
    pub fn prod_dim(self, dim: usize) -> Self {
        Self::from_primitive(B::float_prod_dim(self.primitive, dim))
    }

    /// Returns the maximum element.
    pub fn max(self) -> FuncTensor<B, 1> {
        FuncTensor::from_primitive(B::float_max(self.primitive))
    }

    /// Returns the maximum along a dimension.
    pub fn max_dim(self, dim: usize) -> Self {
        Self::from_primitive(B::float_max_dim(self.primitive, dim))
    }

    /// Returns the minimum element.
    pub fn min(self) -> FuncTensor<B, 1> {
        FuncTensor::from_primitive(B::float_min(self.primitive))
    }

    /// Returns the minimum along a dimension.
    pub fn min_dim(self, dim: usize) -> Self {
        Self::from_primitive(B::float_min_dim(self.primitive, dim))
    }

    /// Computes the cumulative sum along a dimension.
    pub fn cumsum(self, dim: usize) -> Self {
        Self::from_primitive(B::float_cumsum(self.primitive, dim))
    }

    /// Computes the cumulative product along a dimension.
    pub fn cumprod(self, dim: usize) -> Self {
        Self::from_primitive(B::float_cumprod(self.primitive, dim))
    }

    /// Computes the cumulative minimum along a dimension.
    pub fn cummin(self, dim: usize) -> Self {
        Self::from_primitive(B::float_cummin(self.primitive, dim))
    }

    /// Computes the cumulative maximum along a dimension.
    pub fn cummax(self, dim: usize) -> Self {
        Self::from_primitive(B::float_cummax(self.primitive, dim))
    }

    /// Element-wise exponential.
    pub fn exp(self) -> Self {
        Self::from_primitive(B::float_exp(self.primitive))
    }

    /// Element-wise natural logarithm.
    pub fn log(self) -> Self {
        Self::from_primitive(B::float_log(self.primitive))
    }

    /// Element-wise `log(1 + x)`.
    pub fn log1p(self) -> Self {
        Self::from_primitive(B::float_log1p(self.primitive))
    }

    /// Element-wise square root.
    pub fn sqrt(self) -> Self {
        Self::from_primitive(B::float_sqrt(self.primitive))
    }

    /// Element-wise absolute value.
    pub fn abs(self) -> Self {
        Self::from_primitive(B::float_abs(self.primitive))
    }

    /// Element-wise sine.
    pub fn sin(self) -> Self {
        Self::from_primitive(B::float_sin(self.primitive))
    }

    /// Element-wise cosine.
    pub fn cos(self) -> Self {
        Self::from_primitive(B::float_cos(self.primitive))
    }

    /// Element-wise tangent.
    pub fn tan(self) -> Self {
        Self::from_primitive(B::float_tan(self.primitive))
    }

    /// Element-wise hyperbolic sine.
    pub fn sinh(self) -> Self {
        Self::from_primitive(B::float_sinh(self.primitive))
    }

    /// Element-wise hyperbolic cosine.
    pub fn cosh(self) -> Self {
        Self::from_primitive(B::float_cosh(self.primitive))
    }

    /// Element-wise hyperbolic tangent.
    pub fn tanh(self) -> Self {
        Self::from_primitive(B::float_tanh(self.primitive))
    }

    /// Element-wise error function.
    pub fn erf(self) -> Self {
        Self::from_primitive(B::float_erf(self.primitive))
    }

    /// Element-wise reciprocal.
    pub fn recip(self) -> Self {
        Self::from_primitive(B::float_recip(self.primitive))
    }

    /// Element-wise power with another tensor.
    pub fn powf(self, other: Self) -> Self {
        Self::from_primitive(B::float_powf(self.primitive, other.primitive))
    }

    /// Element-wise power with a scalar.
    pub fn powf_scalar(self, value: Scalar) -> Self {
        Self::from_primitive(B::float_powf_scalar(self.primitive, value))
    }

    /// Element-wise integer power with a scalar.
    pub fn powi_scalar(self, value: Scalar) -> Self {
        Self::from_primitive(B::float_powi_scalar(self.primitive, value))
    }

    /// Clamps between `min` and `max`.
    pub fn clamp(self, min: Scalar, max: Scalar) -> Self {
        Self::from_primitive(B::float_clamp(self.primitive, min, max))
    }

    /// Clamps below `min`.
    pub fn clamp_min(self, min: Scalar) -> Self {
        Self::from_primitive(B::float_clamp_min(self.primitive, min))
    }

    /// Clamps above `max`.
    pub fn clamp_max(self, max: Scalar) -> Self {
        Self::from_primitive(B::float_clamp_max(self.primitive, max))
    }

    /// Applies the ReLU activation.
    pub fn relu(self) -> Self {
        Self::from_primitive(B::relu(self.primitive))
    }

    /// Applies the LeakyReLU activation.
    pub fn leaky_relu(self, negative_slope: Scalar) -> Self {
        Self::from_primitive(B::leaky_relu(self.primitive, negative_slope))
    }

    /// Applies the GELU activation.
    pub fn gelu(self) -> Self {
        Self::from_primitive(B::gelu(self.primitive))
    }

    /// Applies the sigmoid activation.
    pub fn sigmoid(self) -> Self {
        Self::from_primitive(B::sigmoid(self.primitive))
    }

    /// Applies the log-sigmoid activation.
    pub fn log_sigmoid(self) -> Self {
        Self::from_primitive(B::log_sigmoid(self.primitive))
    }

    /// Applies softmax along `dim`.
    pub fn softmax(self, dim: usize) -> Self {
        Self::from_primitive(B::softmax(self.primitive, dim))
    }

    /// Applies log-softmax along `dim`.
    pub fn log_softmax(self, dim: usize) -> Self {
        Self::from_primitive(B::log_softmax(self.primitive, dim))
    }

    /// Applies a linear transformation.
    ///
    /// The weight is rank 2 (`[d_input, d_output]`) and the bias is a rank-1 tensor of output
    /// features. `self` may have any leading batch dimensions.
    pub fn linear(self, weight: FuncTensor<B, 2>, bias: Option<FuncTensor<B, 1>>) -> Self {
        Self::from_primitive(B::linear(
            self.primitive,
            weight.primitive,
            bias.map(|b| b.primitive),
        ))
    }

    /// Scaled dot-product attention: `softmax(QK^T * scale) * V`.
    pub fn attention(self, key: Self, value: Self) -> Self {
        Self::from_primitive(B::attention(
            self.primitive,
            key.primitive,
            value.primitive,
            None,
            None,
            AttentionModuleOptions::default(),
        ))
    }

    /// 2D grid sample.
    pub fn grid_sample_2d(self, grid: FuncTensor<B, 4>, options: GridSampleOptions) -> Self {
        Self::from_primitive(B::float_grid_sample_2d(
            self.primitive,
            grid.primitive,
            options,
        ))
    }
}

impl<B: Backend> FuncTensor<B, 4> {
    /// 2D max pooling.
    pub fn max_pool2d(
        self,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> Self {
        Self::from_primitive(B::max_pool2d(
            self.primitive,
            kernel_size,
            stride,
            padding,
            dilation,
            false,
        ))
    }

    /// 2D convolution.
    pub fn conv2d(
        self,
        weight: Self,
        bias: Option<FuncTensor<B, 1>>,
        options: burn_backend::ops::ConvOptions<2>,
    ) -> Self {
        Self::from_primitive(B::conv2d(
            self.primitive,
            weight.primitive,
            bias.map(|b| b.primitive),
            options,
        ))
    }

    /// 2D average pooling.
    pub fn avg_pool2d(
        self,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> Self {
        Self::from_primitive(B::avg_pool2d(
            self.primitive,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            false,
        ))
    }
}

impl<B: Backend> FuncTensor<B, 5> {
    /// 2D max pooling on rank-5 (e.g. batched under vmap).
    pub fn max_pool2d(
        self,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> Self {
        Self::from_primitive(B::max_pool2d(
            self.primitive,
            kernel_size,
            stride,
            padding,
            dilation,
            false,
        ))
    }

    /// 2D convolution on rank-5 (e.g. batched under vmap).
    pub fn conv2d<const DW: usize>(
        self,
        weight: FuncTensor<B, DW>,
        bias: Option<FuncTensor<B, 1>>,
        options: burn_backend::ops::ConvOptions<2>,
    ) -> Self {
        Self::from_primitive(B::conv2d(
            self.primitive,
            weight.primitive,
            bias.map(|b| b.primitive),
            options,
        ))
    }

    /// 2D average pooling on rank-5 (e.g. batched under vmap).
    pub fn avg_pool2d(
        self,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> Self {
        Self::from_primitive(B::avg_pool2d(
            self.primitive,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            false,
        ))
    }
}

impl<B: Backend, const D: usize> TensorMetadata for FuncTensor<B, D> {
    type Device = B::Device;

    fn dtype(&self) -> DType {
        self.primitive.dtype()
    }

    fn shape(&self) -> Shape {
        self.primitive.shape()
    }

    fn rank(&self) -> usize {
        self.primitive.rank()
    }

    fn device(&self) -> B::Device {
        self.primitive.device()
    }

    fn can_mut(&self) -> bool {
        self.primitive.can_mut()
    }
}

/// Conversions to and from the backend-erased `burn_tensor::Tensor<D>` API.
///
/// These are available only for dispatch-supported backends that implement
/// `DispatchKindConversion<B>`, such as `Flex` and `Autodiff<Flex>`.
#[cfg(feature = "bridge")]
impl<B: Backend, const D: usize> FuncTensor<B, D>
where
    burn_dispatch::DispatchTensor: burn_dispatch::DispatchKindConversion<B>,
{
    /// Converts a `burn_tensor::Tensor<D>` into a `FuncTensor` on `B`.
    ///
    /// # Panics
    ///
    /// Panics if the tensor does not currently live on backend `B`.
    pub fn from_tensor(tensor: &burn_tensor::Tensor<D>) -> Self {
        let primitive = tensor
            .clone()
            .try_into_primitive::<B>()
            .unwrap_or_else(|error| {
                panic!("FuncTensor::from_tensor could not extract a {error:?} primitive")
            });
        Self::from_primitive(primitive)
    }

    /// Converts this tensor into a `burn_tensor::Tensor<D>`.
    pub fn into_tensor(self) -> burn_tensor::Tensor<D> {
        burn_tensor::Tensor::<D>::from_primitive(self.primitive)
    }
}

fn assert_rank<const D: usize>(shape: &Shape, operation: &str) {
    assert_eq!(
        shape.num_dims(),
        D,
        "{operation} requires rank {D}, got shape {shape:?}"
    );
}

macro_rules! impl_binary_op {
    ($trait:ident, $method:ident, $backend_method:ident) => {
        impl<B: Backend, const D: usize> core::ops::$trait for FuncTensor<B, D> {
            type Output = Self;

            fn $method(self, rhs: Self) -> Self {
                Self::from_primitive(B::$backend_method(self.primitive, rhs.primitive))
            }
        }

        impl<B: Backend, const D: usize> core::ops::$trait<&FuncTensor<B, D>> for FuncTensor<B, D> {
            type Output = Self;

            fn $method(self, rhs: &Self) -> Self {
                Self::from_primitive(B::$backend_method(self.primitive, rhs.primitive.clone()))
            }
        }

        impl<B: Backend, const D: usize> core::ops::$trait<FuncTensor<B, D>> for &FuncTensor<B, D> {
            type Output = FuncTensor<B, D>;

            fn $method(self, rhs: FuncTensor<B, D>) -> FuncTensor<B, D> {
                FuncTensor::from_primitive(B::$backend_method(
                    self.primitive.clone(),
                    rhs.primitive,
                ))
            }
        }

        impl<B: Backend, const D: usize> core::ops::$trait<&FuncTensor<B, D>>
            for &FuncTensor<B, D>
        {
            type Output = FuncTensor<B, D>;

            fn $method(self, rhs: &FuncTensor<B, D>) -> FuncTensor<B, D> {
                FuncTensor::from_primitive(B::$backend_method(
                    self.primitive.clone(),
                    rhs.primitive.clone(),
                ))
            }
        }
    };
}

impl_binary_op!(Add, add, float_add);
impl_binary_op!(Sub, sub, float_sub);
impl_binary_op!(Mul, mul, float_mul);
impl_binary_op!(Div, div, float_div);

macro_rules! impl_scalar_op {
    ($trait:ident, $method:ident, $backend_method:ident) => {
        impl<B: Backend, const D: usize> core::ops::$trait<Scalar> for FuncTensor<B, D> {
            type Output = Self;

            fn $method(self, rhs: Scalar) -> Self {
                Self::from_primitive(B::$backend_method(self.primitive, rhs))
            }
        }

        impl<B: Backend, const D: usize> core::ops::$trait<Scalar> for &FuncTensor<B, D> {
            type Output = FuncTensor<B, D>;

            fn $method(self, rhs: Scalar) -> FuncTensor<B, D> {
                FuncTensor::from_primitive(B::$backend_method(self.primitive.clone(), rhs))
            }
        }

        impl<B: Backend, const D: usize> core::ops::$trait<f64> for FuncTensor<B, D> {
            type Output = Self;

            fn $method(self, rhs: f64) -> Self {
                self.$method(Scalar::from(rhs))
            }
        }

        impl<B: Backend, const D: usize> core::ops::$trait<f64> for &FuncTensor<B, D> {
            type Output = FuncTensor<B, D>;

            fn $method(self, rhs: f64) -> FuncTensor<B, D> {
                self.$method(Scalar::from(rhs))
            }
        }

        impl<B: Backend, const D: usize> core::ops::$trait<i32> for FuncTensor<B, D> {
            type Output = Self;

            fn $method(self, rhs: i32) -> Self {
                self.$method(Scalar::from(rhs))
            }
        }

        impl<B: Backend, const D: usize> core::ops::$trait<i32> for &FuncTensor<B, D> {
            type Output = FuncTensor<B, D>;

            fn $method(self, rhs: i32) -> FuncTensor<B, D> {
                self.$method(Scalar::from(rhs))
            }
        }

        impl<B: Backend, const D: usize> core::ops::$trait<i64> for FuncTensor<B, D> {
            type Output = Self;

            fn $method(self, rhs: i64) -> Self {
                self.$method(Scalar::from(rhs))
            }
        }

        impl<B: Backend, const D: usize> core::ops::$trait<i64> for &FuncTensor<B, D> {
            type Output = FuncTensor<B, D>;

            fn $method(self, rhs: i64) -> FuncTensor<B, D> {
                self.$method(Scalar::from(rhs))
            }
        }
    };
}

impl_scalar_op!(Add, add, float_add_scalar);
impl_scalar_op!(Sub, sub, float_sub_scalar);
impl_scalar_op!(Mul, mul, float_mul_scalar);
impl_scalar_op!(Div, div, float_div_scalar);

impl<B: Backend, const D: usize> core::ops::Neg for FuncTensor<B, D> {
    type Output = Self;

    fn neg(self) -> Self {
        Self::from_primitive(B::float_neg(self.primitive))
    }
}

impl<B: Backend, const D: usize> core::ops::Neg for &FuncTensor<B, D> {
    type Output = FuncTensor<B, D>;

    fn neg(self) -> FuncTensor<B, D> {
        FuncTensor::from_primitive(B::float_neg(self.primitive.clone()))
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;
    use burn_backend::TensorData;
    use burn_flex::{Flex, FlexDevice};

    use super::*;

    fn tensor(values: &[f32]) -> FuncTensor<Flex, 1> {
        FuncTensor::from_data(TensorData::from(values), &FlexDevice)
    }

    fn values(tensor: FuncTensor<Flex, 1>) -> Vec<f32> {
        tensor.into_data_sync().try_to_vec::<f32>().unwrap()
    }

    #[test]
    fn arithmetic_operators() {
        let x = tensor(&[2.0, 4.0]);
        let y = tensor(&[1.0, 2.0]);

        assert_eq!(values(x.clone() + y.clone()), vec![3.0, 6.0]);
        assert_eq!(values(x.clone() - y.clone()), vec![1.0, 2.0]);
        assert_eq!(values(x.clone() * y.clone()), vec![2.0, 8.0]);
        assert_eq!(values(x.clone() / y.clone()), vec![2.0, 2.0]);
        assert_eq!(values(-x.clone()), vec![-2.0, -4.0]);
        assert_eq!(values(x.clone() * 2.0f64), vec![4.0, 8.0]);
        assert_eq!(values(x.clone() + 1.0f64), vec![3.0, 5.0]);
    }

    #[test]
    fn unary_and_activation_methods() {
        let x = tensor(&[0.0, 1.0]);
        assert_eq!(values(x.clone().exp())[0], 1.0);
        assert!((values(x.clone().sigmoid())[0] - 0.5).abs() < 1e-6);
        assert_eq!(values(x.clone().relu()), vec![0.0, 1.0]);
        assert_eq!(values(tensor(&[-1.0, 2.0]).relu()), vec![0.0, 2.0]);
        assert_eq!(values(x.clone().abs()), vec![0.0, 1.0]);
        assert!((values(x.clone().tanh())[1] - 1.0f32.tanh()).abs() < 1e-6);
    }

    #[test]
    fn reductions_and_shape_methods() {
        let x = tensor(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(values(x.clone().sum()), vec![10.0]);
        assert_eq!(values(x.clone().mean()), vec![2.5]);
        assert_eq!(values(x.clone().max()), vec![4.0]);
        assert_eq!(values(x.clone().min()), vec![1.0]);
        assert_eq!(values(x.clone().cumsum(0)), vec![1.0, 3.0, 6.0, 10.0]);
        assert_eq!(x.dims(), [4]);

        let matrix = FuncTensor::<Flex, 2>::from_data(
            TensorData::from([[1.0f32, 2.0], [3.0, 4.0]]),
            &FlexDevice,
        );
        let transposed = matrix.clone().transpose();
        assert_eq!(
            transposed.into_data_sync().try_to_vec::<f32>().unwrap(),
            vec![1.0, 3.0, 2.0, 4.0]
        );
        let reshaped: FuncTensor<Flex, 1> = matrix.reshape([4]);
        assert_eq!(values(reshaped), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn matmul_and_linear() {
        let x = FuncTensor::<Flex, 2>::from_data(
            TensorData::from([[1.0f32, 2.0], [3.0, 4.0]]),
            &FlexDevice,
        );
        let weight = FuncTensor::<Flex, 2>::from_data(
            TensorData::from([[2.0f32, 0.0], [0.0, 2.0]]),
            &FlexDevice,
        );
        let bias = FuncTensor::<Flex, 1>::from_data(TensorData::from([1.0f32, -1.0]), &FlexDevice);

        let output = x.clone().linear(weight.clone(), Some(bias));
        assert_eq!(
            output.into_data_sync().try_to_vec::<f32>().unwrap(),
            vec![3.0, 3.0, 7.0, 7.0]
        );

        let matmul = x.matmul(weight);
        assert_eq!(
            matmul.into_data_sync().try_to_vec::<f32>().unwrap(),
            vec![2.0, 4.0, 6.0, 8.0]
        );
    }

    #[test]
    fn slicing_and_cat() {
        let x = tensor(&[1.0, 2.0, 3.0]);
        let sliced = x.clone().slice(&[burn_std::Slice::new(0, Some(2), 1)]);
        assert_eq!(values(sliced), vec![1.0, 2.0]);

        let value = tensor(&[9.0]);
        let assigned = x
            .clone()
            .slice_assign(&[burn_std::Slice::new(0, Some(1), 1)], value);
        assert_eq!(values(assigned), vec![9.0, 2.0, 3.0]);

        let cat = FuncTensor::cat(vec![x.clone(), x], 0);
        assert_eq!(values(cat), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[cfg(feature = "bridge")]
    #[test]
    fn tensor_bridge_roundtrip() {
        let tensor = burn_tensor::Tensor::<1>::from_floats([1.0f32, 2.0, 3.0], &Default::default());
        let func = FuncTensor::<Flex, 1>::from_tensor(&tensor);
        assert_eq!(values(func.clone()), vec![1.0, 2.0, 3.0]);

        let roundtrip = func.into_tensor();
        assert_eq!(
            roundtrip.into_data().try_to_vec::<f32>().unwrap(),
            vec![1.0, 2.0, 3.0]
        );
    }
}
