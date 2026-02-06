pub mod element;
/*
The base implementation for complex tensors, contains everything that would be in burn-tensor.
May get split into separate files at some point, but for now it's easier to keep all the base
definitions in one spot.
*/
use burn_tensor::{
    BasicOps, Bytes, DType, Device, Distribution, Element, ElementConversion, FloatDType,
    IndexingUpdateOp, Numeric, Scalar, Shape, Slice, TensorData, TensorKind, TensorMetadata,
    TransactionPrimitive,
    backend::{Backend, ExecutionError},
    ops::{FloatTensor, IntTensor},
};

/// The layout of the complex tensor. Used to define shared behavior only meant
/// to be used for a specific layout (such as butterfly operations).
pub trait ComplexLayout {}

/// Complex element type used by backend.
pub type ComplexElem<B> = <B as ComplexTensorBackend>::ComplexElem;

/// Complex tensor primitive type used by the backend.
pub type ComplexTensor<B> = <B as ComplexTensorBackend>::ComplexTensorPrimitive;

pub trait ComplexTensorBackend: Backend + ComplexTensorOps<Self> {
    /// The inner backend type.
    type InnerBackend: Backend<Device = Self::Device, FloatElem = Self::FloatElem>;

    /// Tensor primitive to be used for all complex operations.
    type ComplexTensorPrimitive: TensorMetadata + 'static;

    /// Complex element type.
    type ComplexElem: Element;

    /// The underlaying layout for the complex elements
    type Layout: ComplexLayout;

    /// Returns the real part of a complex tensor.
    fn real(tensor: ComplexTensor<Self>) -> FloatTensor<Self::InnerBackend>;
    /// Returns the imaginary part of a complex tensor.
    fn imag(tensor: ComplexTensor<Self>) -> FloatTensor<Self::InnerBackend>;

    fn to_complex(tensor: FloatTensor<Self::InnerBackend>) -> ComplexTensor<Self>;
    // can reuse float random
}
//Note: changing to adopt terminology used in fftw doc

/// Indicates that the underlying implementation has separate real and imaginary tensors.
pub struct SplitLayout;

/// Indicates that the underlying implementation uses a complex primitive type [float,float] like that found in the
/// num_complex trait.
pub struct InterleavedLayout;

pub struct InterleavedTensorData {
    /// The values of the tensor (as bytes).
    pub bytes: Bytes,

    /// The shape of the tensor.
    pub shape: Vec<usize>,

    /// The data type of the tensor.
    pub dtype: DType,
}

pub struct SplitTensorData {
    /// The real values of the tensor (as bytes).
    pub real_bytes: Bytes,

    /// The imaginary values of the tensor (as bytes).
    pub imag_bytes: Bytes,

    /// The shape of the tensor.
    pub shape: Vec<usize>,

    /// The data type of the tensor.
    pub dtype: DType,
}
// /// Indicates that the underlying implementation uses an interleaved layout for complex numbers.
// pub struct InterleavedLayout;

impl ComplexLayout for SplitLayout {}

impl ComplexLayout for InterleavedLayout {}
// impl ComplexLayout for InterleavedLayout {}

/// Operations on complex tensors.
pub trait ComplexTensorOps<B: ComplexTensorBackend> {
    type Layout: ComplexLayout;
    /// Creates a new complex tensor from the data structure.
    ///
    /// # Arguments
    ///
    /// * `data` - The data structure.
    /// * `device` - The device to create the tensor on.
    ///
    /// # Returns
    ///
    /// The tensor with the given data.
    fn complex_from_data(data: TensorData, device: &Device<B>) -> ComplexTensor<B>;

    /// Creates a new complex tensor with random values.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `distribution` - The distribution to sample from.
    /// * `device` - The device to create the tensor on.
    ///
    /// # Returns
    ///
    /// The tensor with the given shape and random values.
    fn complex_random(
        shape: Shape,
        distribution: Distribution,
        device: &Device<B>,
    ) -> ComplexTensor<B>;

    /// Creates a new complex tensor with zeros.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `device` - The device to create the tensor on.
    ///
    /// # Returns
    ///
    /// The tensor with the given shape and zeros.
    fn complex_zeros(shape: Shape, device: &Device<B>) -> ComplexTensor<B> {
        Self::complex_from_data(TensorData::zeros::<ComplexElem<B>, _>(shape), device)
    }

    /// Creates a new complex tensor with ones.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `device` - The device to create the tensor on.
    ///
    /// # Returns
    ///
    /// The tensor with the given shape and ones.
    fn complex_ones(shape: Shape, device: &Device<B>) -> ComplexTensor<B> {
        Self::complex_from_data(TensorData::ones::<ComplexElem<B>, _>(shape), device)
    }

    /// Creates a new complex tensor with the given shape and a single value.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `value` - The value to fill the tensor with.
    /// * `device` - The device to create the tensor on.
    ///
    /// # Returns
    ///
    /// The tensor with the given shape and value.
    fn complex_full(
        shape: Shape,
        fill_value: ComplexElem<B>,
        device: &Device<B>,
    ) -> ComplexTensor<B> {
        Self::complex_from_data(TensorData::full(shape, fill_value), device)
    }

    /// Gets the shape of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The shape of the tensor.
    fn complex_shape(tensor: &ComplexTensor<B>) -> Shape;

    /// Converts the tensor to a data structure.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The data structure with the tensor's data.
    fn complex_to_data(tensor: &ComplexTensor<B>) -> TensorData;

    /// Gets the device of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The device of the tensor.
    fn complex_device(tensor: &ComplexTensor<B>) -> Device<B>;

    /// Moves the tensor to the given device.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `device` - The device to move the tensor to.
    ///
    /// # Returns
    ///
    /// The tensor on the given device.
    fn complex_to_device(tensor: ComplexTensor<B>, device: &Device<B>) -> ComplexTensor<B>;

    /// Converts the tensor to a different element type.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The tensor with the new element type.
    fn complex_into_data(
        tensor: ComplexTensor<B>,
    ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send;

    /// Reshapes the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `shape` - The new shape.
    ///
    /// # Returns
    ///
    /// The tensor with the new shape.
    fn complex_reshape(tensor: ComplexTensor<B>, shape: Shape) -> ComplexTensor<B>;

    /// Transposes the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The transposed tensor.
    fn complex_transpose(tensor: ComplexTensor<B>) -> ComplexTensor<B>;

    /// Adds two tensors together.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The result of adding the two tensors together.
    fn complex_add(lhs: ComplexTensor<B>, rhs: ComplexTensor<B>) -> ComplexTensor<B>;

    /// Subtracts the second tensor from the first tensor.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The result of subtracting the second tensor from the first tensor.
    fn complex_sub(lhs: ComplexTensor<B>, rhs: ComplexTensor<B>) -> ComplexTensor<B>;

    /// Multiplies two complex tensors together using complex multiplication.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The result of multiplying the two tensors together.
    fn complex_mul(lhs: ComplexTensor<B>, rhs: ComplexTensor<B>) -> ComplexTensor<B>;

    /// Divides the first tensor by the second tensor using complex division.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The result of dividing the first tensor by the second tensor.
    fn complex_div(lhs: ComplexTensor<B>, rhs: ComplexTensor<B>) -> ComplexTensor<B>;

    /// Negates the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The negated tensor.
    fn complex_neg(tensor: ComplexTensor<B>) -> ComplexTensor<B>;

    /// Returns the complex conjugate of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The complex conjugate of the tensor.
    fn complex_conj(tensor: ComplexTensor<B>) -> ComplexTensor<B>;

    /// Returns the real part of the complex tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The complex tensor.
    ///
    /// # Returns
    ///
    /// A float tensor containing the real parts.
    fn complex_real(tensor: ComplexTensor<B>) -> B::FloatTensorPrimitive;

    /// Returns the imaginary part of the complex tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The complex tensor.
    ///
    /// # Returns
    ///
    /// A float tensor containing the imaginary parts.
    fn complex_imag(tensor: ComplexTensor<B>) -> B::FloatTensorPrimitive;

    /// Returns the magnitude (absolute value) of the complex tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The complex tensor.
    ///
    /// # Returns
    ///
    /// A float tensor containing the magnitudes.
    fn complex_abs(tensor: ComplexTensor<B>) -> B::FloatTensorPrimitive;

    /// Returns the phase (argument) of the complex tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The complex tensor.
    ///
    /// # Returns
    ///
    /// A float tensor containing the phases in radians.
    fn complex_arg(tensor: ComplexTensor<B>) -> B::FloatTensorPrimitive;

    /// Creates a complex tensor from real and imaginary parts.
    ///
    /// # Arguments
    ///
    /// * `real` - The real part tensor.
    /// * `imag` - The imaginary part tensor.
    ///
    /// # Returns
    ///
    /// A complex tensor constructed from the real and imaginary parts.
    fn complex_from_parts(
        real: B::FloatTensorPrimitive,
        imag: B::FloatTensorPrimitive,
    ) -> ComplexTensor<B>;

    /// Creates a complex tensor from magnitude and phase.
    ///
    /// # Arguments
    ///
    /// * `magnitude` - The magnitude tensor.
    /// * `phase` - The phase tensor (in radians).
    ///
    /// # Returns
    ///
    /// A complex tensor constructed from polar coordinates.
    fn complex_from_polar(
        magnitude: B::FloatTensorPrimitive,
        phase: B::FloatTensorPrimitive,
    ) -> ComplexTensor<B>;

    /// Complex exponential function.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The exponential of the tensor.
    fn complex_exp(tensor: ComplexTensor<B>) -> ComplexTensor<B>;

    /// Complex natural logarithm.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The natural logarithm of the tensor.
    fn complex_log(tensor: ComplexTensor<B>) -> ComplexTensor<B>;

    /// Complex power function.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The base tensor.
    /// * `exponent` - The exponent tensor.
    ///
    /// # Returns
    ///
    /// The result of raising the base to the exponent.
    fn complex_powc(lhs: ComplexTensor<B>, rhs: ComplexTensor<B>) -> ComplexTensor<B>;

    /// Complex square root.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The square root of the tensor.
    fn complex_sqrt(tensor: ComplexTensor<B>) -> ComplexTensor<B>;

    /// Complex sine function.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The sine of the tensor.
    fn complex_sin(tensor: ComplexTensor<B>) -> ComplexTensor<B>;

    /// Complex cosine function.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The cosine of the tensor.
    fn complex_cos(tensor: ComplexTensor<B>) -> ComplexTensor<B>;

    /// Complex tangent function.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The tangent of the tensor.
    fn complex_tan(tensor: ComplexTensor<B>) -> ComplexTensor<B>;

    /// Complex select function.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `dim` - The dimension to select.
    /// * `indices` - The indices to select.
    ///
    /// # Returns
    ///
    /// The selected tensor.
    fn select(
        tensor: ComplexTensor<B>,
        dim: usize,
        indices: <B as Backend>::IntTensorPrimitive,
    ) -> ComplexTensor<B>;

    /// Complex select assign function.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `dim` - The dimension to select.
    /// * `indices` - The indices to select.
    /// * `values` - The values to assign.
    ///
    /// # Returns
    ///
    /// The assigned tensor.
    fn select_assign(
        tensor: ComplexTensor<B>,
        dim: usize,
        indices: <B as Backend>::IntTensorPrimitive,
        values: ComplexTensor<B>,
    ) -> ComplexTensor<B>;

    /// Complex slice function.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `dim` - The dimension to slice.
    /// * `start` - The start index.
    /// * `end` - The end index.
    ///
    /// # Returns
    ///
    /// The sliced tensor.
    fn complex_slice(tensor: ComplexTensor<B>, slices: &[burn_tensor::Slice]) -> ComplexTensor<B>;

    /// Assign the selected elements corresponding for the given ranges to the given value.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to select from.
    /// * `ranges` - The ranges to select.
    /// * `value` - The value to assign.
    ///
    /// # Returns
    ///
    /// The tensor with the selected elements assigned to the given value.
    fn complex_slice_assign(
        tensor: ComplexTensor<B>,
        ranges: &[burn_tensor::Slice],
        value: ComplexTensor<B>,
    ) -> ComplexTensor<B>;

    /// Swaps two dimensions of a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to swap the dimensions of.
    /// * `dim1` - The first dimension to swap.
    /// * `dim2` - The second dimension to swap.
    ///
    /// # Returns
    ///
    /// The tensor with the dimensions swapped.
    fn complex_swap_dims(tensor: ComplexTensor<B>, dim1: usize, dim2: usize) -> ComplexTensor<B>;

    /// Repeat the tensor along the given dimension.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `dim` - The dimension to repeat.
    /// * `times` - The number of times to repeat the dimension.
    ///
    /// # Returns
    ///
    /// The tensor with the given dimension repeated.
    fn complex_repeat_dim(tensor: ComplexTensor<B>, dim: usize, times: usize) -> ComplexTensor<B>;

    /// Equal comparison of two tensors.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left-hand side tensor.
    /// * `rhs` - The right-hand side tensor.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the result of the comparison.
    fn complex_equal(lhs: ComplexTensor<B>, rhs: ComplexTensor<B>) -> B::BoolTensorPrimitive;

    /// Element-wise non-equality comparison.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left-hand side tensor.
    /// * `rhs` - The right-hand side tensor.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the result of the comparison.
    fn complex_not_equal(lhs: ComplexTensor<B>, rhs: ComplexTensor<B>) -> B::BoolTensorPrimitive;

    /// Concatenates tensors along a dimension.
    ///
    /// # Arguments
    ///
    /// * `tensors` - The tensors to concatenate.
    /// * `dim` - The dimension along which to concatenate.
    ///
    /// # Returns
    ///
    /// A tensor with the concatenated tensors along `dim`.
    fn complex_cat(tensors: Vec<ComplexTensor<B>>, dim: usize) -> ComplexTensor<B>;

    /// Tests if any element in the `tensor` evaluates to True.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to test.
    ///
    /// # Returns
    ///
    /// A boolean tensor with a single element, True if any element in the tensor is True, False otherwise.
    fn complex_any(tensor: ComplexTensor<B>) -> B::BoolTensorPrimitive;

    /// Tests if any element in the float `tensor` evaluates to True along a given dimension `dim`.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to test.
    /// * `dim` - The axis along which to test.
    ///
    /// # Returns
    ///
    /// A boolean tensor `Tensor<B, D, Bool>` with the same size as input `tensor`, except in the `dim` axis
    /// where the size is 1. The elem in the `dim` axis is True if any element along this dim in the
    /// input evaluates to True, False otherwise.
    fn complex_any_dim(tensor: ComplexTensor<B>, dim: usize) -> B::BoolTensorPrimitive;

    /// Tests if all elements in the float `tensor` evaluate to True.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to test.
    ///
    /// # Returns
    ///
    /// A boolean tensor `Tensor<B, 1, Bool>` with a single element, True if all elements in the input tensor
    /// evaluate to True, False otherwise.
    fn complex_all(tensor: ComplexTensor<B>) -> B::BoolTensorPrimitive;

    /// Tests if all elements in the float `tensor` evaluate to True along a given dimension `dim`.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to test.
    /// * `dim` - The axis along which to test.
    ///
    /// # Returns
    ///
    /// A boolean tensor `Tensor<B, D, Bool>` with the same size as input `tensor`, except in the `dim` axis
    /// where the size is 1. The elem in the `dim` axis is True if all elements along this dim in the input
    /// evaluates to True, False otherwise.
    fn complex_all_dim(tensor: ComplexTensor<B>, dim: usize) -> B::BoolTensorPrimitive;

    /// Permute axes.
    fn complex_permute(tensor: ComplexTensor<B>, axes: &[usize]) -> ComplexTensor<B>;

    /// Expand to broadcast shape.
    fn complex_expand(tensor: ComplexTensor<B>, shape: Shape) -> ComplexTensor<B>;

    /// Flip along given axes.
    fn complex_flip(tensor: ComplexTensor<B>, axes: &[usize]) -> ComplexTensor<B>;

    /// Unfold (im2col-like) along a dimension.
    fn complex_unfold(
        tensor: ComplexTensor<B>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> ComplexTensor<B>;

    /// Select tensor elements along the given dimension corresponding for the given indices.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to select from.
    /// * `dim` - The dimension to select from.
    /// * `indices` - The indices to select.
    ///
    /// # Returns
    ///
    /// The selected elements.
    fn complex_select(
        tensor: ComplexTensor<B>,
        dim: usize,
        indices: IntTensor<B>,
    ) -> ComplexTensor<B>;

    fn complex_sum(tensor: ComplexTensor<B>) -> ComplexTensor<B>;
    fn complex_sum_dim(tensor: ComplexTensor<B>, dim: usize) -> ComplexTensor<B>;
    fn complex_prod(tensor: ComplexTensor<B>) -> ComplexTensor<B>;
    fn complex_prod_dim(tensor: ComplexTensor<B>, dim: usize) -> ComplexTensor<B>;
    fn complex_mean(tensor: ComplexTensor<B>) -> ComplexTensor<B>;
    fn complex_mean_dim(tensor: ComplexTensor<B>, dim: usize) -> ComplexTensor<B>;

    fn complex_remainder(lhs: ComplexTensor<B>, rhs: ComplexTensor<B>) -> ComplexTensor<B>;
    fn complex_remainder_scalar(lhs: ComplexTensor<B>, rhs: B::ComplexElem) -> ComplexTensor<B>;

    fn complex_equal_elem(lhs: ComplexTensor<B>, rhs: B::ComplexElem) -> B::BoolTensorPrimitive;
    fn complex_not_equal_elem(lhs: ComplexTensor<B>, rhs: B::ComplexElem)
    -> B::BoolTensorPrimitive;

    fn complex_mask_where(
        tensor: ComplexTensor<B>,
        mask: B::BoolTensorPrimitive,
        source: ComplexTensor<B>,
    ) -> ComplexTensor<B>;
    fn complex_mask_fill(
        tensor: ComplexTensor<B>,
        mask: B::BoolTensorPrimitive,
        value: B::ComplexElem,
    ) -> ComplexTensor<B>;
    fn complex_gather(
        dim: usize,
        tensor: ComplexTensor<B>,
        indices: B::IntTensorPrimitive,
    ) -> ComplexTensor<B>;
    fn complex_scatter(
        dim: usize,
        tensor: ComplexTensor<B>,
        indices: B::IntTensorPrimitive,
        values: ComplexTensor<B>,
    ) -> ComplexTensor<B>;
    //todo: add doc strings
    fn complex_sign(tensor: ComplexTensor<B>) -> ComplexTensor<B>;

    fn complex_clamp(
        tensor: ComplexTensor<B>,
        min: B::ComplexElem,
        max: B::ComplexElem,
    ) -> ComplexTensor<B>;
    fn complex_clamp_min(tensor: ComplexTensor<B>, min: B::ComplexElem) -> ComplexTensor<B>;
    fn complex_clamp_max(tensor: ComplexTensor<B>, max: B::ComplexElem) -> ComplexTensor<B>;

    fn complex_powi(lhs: ComplexTensor<B>, rhs: ComplexTensor<B>) -> ComplexTensor<B>;
    fn complex_powi_scalar(lhs: ComplexTensor<B>, rhs: B::ComplexElem) -> ComplexTensor<B>;

    fn complex_matmul(lhs: ComplexTensor<B>, rhs: ComplexTensor<B>) -> ComplexTensor<B>;

    fn complex_cumsum(tensor: ComplexTensor<B>, dim: usize) -> ComplexTensor<B>;
    fn complex_cumprod(tensor: ComplexTensor<B>, dim: usize) -> ComplexTensor<B>;
}

/// A type-level representation of the kind of a complex tensor.
#[derive(Clone, Debug)]
pub struct Complex;

#[allow(unused_variables)]
impl<B: ComplexTensorBackend> BasicOps<B> for Complex {
    type Elem = B::ComplexElem;

    fn empty(shape: Shape, device: &B::Device, dtype: DType) -> Self::Primitive {
        // should I check then pass the dtype?
        B::complex_zeros(shape, device)
    }

    fn register_transaction(tr: &mut TransactionPrimitive<B>, tensor: Self::Primitive) {
        // Complex tensors don't support transactions yet
        // TODO: Implement complex tensor transaction support
    }

    fn reshape(tensor: Self::Primitive, shape: Shape) -> Self::Primitive {
        B::complex_reshape(tensor, shape)
    }

    fn transpose(tensor: Self::Primitive) -> Self::Primitive {
        B::complex_transpose(tensor)
    }

    fn swap_dims(tensor: Self::Primitive, dim1: usize, dim2: usize) -> Self::Primitive {
        B::complex_swap_dims(tensor, dim1, dim2)
    }

    fn slice(tensor: Self::Primitive, ranges: &[Slice]) -> Self::Primitive {
        //TensorPrimitive::Complex(B::complex_slice(tensor, ranges))
        B::complex_slice(tensor, ranges)
    }

    fn device(tensor: &Self::Primitive) -> Device<B> {
        B::complex_device(tensor)
    }

    fn to_device(tensor: Self::Primitive, device: &Device<B>) -> Self::Primitive {
        B::complex_to_device(tensor, device)
    }

    async fn into_data_async(tensor: Self::Primitive) -> Result<TensorData, ExecutionError> {
        B::complex_into_data(tensor).await
    }

    fn from_data(data: TensorData, device: &B::Device) -> Self::Primitive {
        B::complex_from_data(data.convert::<B::ComplexElem>(), device)
    }

    fn from_data_dtype(data: TensorData, device: &B::Device, dtype: DType) -> Self::Primitive {
        if !dtype.is_complex() {
            panic!("Expected complex dtype, got {dtype:?}")
        }
        B::complex_from_data(data.convert_dtype(dtype), device)
    }

    fn repeat_dim(tensor: Self::Primitive, dim: usize, times: usize) -> Self::Primitive {
        B::complex_repeat_dim(tensor, dim, times)
    }
    fn equal(lhs: Self::Primitive, rhs: Self::Primitive) -> <B as Backend>::BoolTensorPrimitive {
        B::complex_equal(lhs, rhs)
    }

    fn not_equal(
        lhs: Self::Primitive,
        rhs: Self::Primitive,
    ) -> <B as Backend>::BoolTensorPrimitive {
        B::complex_not_equal(lhs, rhs)
    }

    fn cat(tensors: Vec<Self::Primitive>, dim: usize) -> Self::Primitive {
        B::complex_cat(tensors, dim)
    }

    fn any(tensor: Self::Primitive) -> <B as Backend>::BoolTensorPrimitive {
        B::complex_any(tensor)
    }

    fn any_dim(tensor: Self::Primitive, dim: usize) -> <B as Backend>::BoolTensorPrimitive {
        B::complex_any_dim(tensor, dim)
    }

    fn all(tensor: Self::Primitive) -> <B as Backend>::BoolTensorPrimitive {
        B::complex_all(tensor)
    }

    fn all_dim(tensor: Self::Primitive, dim: usize) -> <B as Backend>::BoolTensorPrimitive {
        B::complex_all_dim(tensor, dim)
    }

    fn permute(tensor: Self::Primitive, axes: &[usize]) -> Self::Primitive {
        B::complex_permute(tensor, axes)
    }

    fn expand(tensor: Self::Primitive, shape: Shape) -> Self::Primitive {
        B::complex_expand(tensor, shape)
    }

    fn flip(tensor: Self::Primitive, axes: &[usize]) -> Self::Primitive {
        B::complex_flip(tensor, axes)
    }

    fn unfold(tensor: Self::Primitive, dim: usize, size: usize, step: usize) -> Self::Primitive {
        B::complex_unfold(tensor, dim, size, step)
    }

    fn slice_assign(
        tensor: Self::Primitive,
        ranges: &[Slice],
        value: Self::Primitive,
    ) -> Self::Primitive {
        B::complex_slice_assign(tensor, ranges, value)
    }

    fn select(
        tensor: Self::Primitive,
        dim: usize,
        indices: <B as Backend>::IntTensorPrimitive,
    ) -> Self::Primitive {
        // Uses your existing `select` name.
        B::select(tensor, dim, indices)
    }

    fn select_assign(
        tensor: Self::Primitive,
        dim: usize,
        indices: <B as Backend>::IntTensorPrimitive,
        values: Self::Primitive,
        update: IndexingUpdateOp,
    ) -> Self::Primitive {
        // // Uses your existing `select_assign` name.
        // B::select_assign(tensor, dim, indices, values)
        todo!()
    }

    fn zeros(shape: Shape, device: &<B as Backend>::Device, dtype: DType) -> Self::Primitive {
        match dtype {
            DType::Complex32 | DType::Complex64 => B::complex_zeros(shape, device),
            _ => panic!("Unsupported complex dtype"),
        }
    }

    fn ones(shape: Shape, device: &<B as Backend>::Device, dtype: DType) -> Self::Primitive {
        match dtype {
            DType::Complex32 | DType::Complex64 => B::complex_ones(shape, device),
            _ => panic!("Unsupported complex dtype"),
        }
    }

    fn mask_where(
        tensor: Self::Primitive,
        mask: <B as Backend>::BoolTensorPrimitive,
        source: Self::Primitive,
    ) -> Self::Primitive {
        todo!()
    }

    fn mask_fill(
        tensor: Self::Primitive,
        mask: <B as Backend>::BoolTensorPrimitive,
        value: burn_tensor::Scalar,
    ) -> Self::Primitive {
        todo!()
    }

    fn gather(dim: usize, tensor: Self::Primitive, indices: IntTensor<B>) -> Self::Primitive {
        todo!()
    }

    fn scatter(
        dim: usize,
        tensor: Self::Primitive,
        indices: IntTensor<B>,
        values: Self::Primitive,
        update: burn_tensor::IndexingUpdateOp,
    ) -> Self::Primitive {
        todo!()
    }

    fn equal_elem(
        lhs: Self::Primitive,
        rhs: burn_tensor::Scalar,
    ) -> <B as Backend>::BoolTensorPrimitive {
        todo!()
    }

    fn not_equal_elem(
        lhs: Self::Primitive,
        rhs: burn_tensor::Scalar,
    ) -> <B as Backend>::BoolTensorPrimitive {
        todo!()
    }

    fn full(
        shape: Shape,
        fill_value: burn_tensor::Scalar,
        device: &<B as Backend>::Device,
        dtype: DType,
    ) -> Self::Primitive {
        // Enforce complex dtype for clarity (mirrors from_data_dtype below).
        if !dtype.is_complex() {
            panic!("Expected complex dtype, got {dtype:?}");
        }
        // `elem()` should yield something convertible to `B::ComplexElem`.
        B::complex_full(shape, fill_value.elem(), device)
    }
}

#[allow(unused_variables)]
impl<B: ComplexTensorBackend> Numeric<B> for Complex
where
    B::ComplexElem: Element,
{
    fn add(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        B::complex_add(lhs, rhs)
    }

    fn sub(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        B::complex_sub(lhs, rhs)
    }

    fn sub_scalar(lhs: Self::Primitive, rhs: Scalar) -> Self::Primitive {
        // TODO: Implement complex_sub_scalar in ComplexTensorOps
        let device = B::complex_device(&lhs);
        let shape = B::complex_shape(&lhs);
        let scalar_complex: B::ComplexElem = rhs.elem();
        let scalar_tensor = B::complex_full(shape, scalar_complex, &device);
        B::complex_sub(lhs, scalar_tensor)
    }

    fn mul(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        B::complex_mul(lhs, rhs)
    }

    fn mul_scalar(lhs: Self::Primitive, rhs: Scalar) -> Self::Primitive {
        // TODO: Implement complex_mul_scalar in ComplexTensorOps
        let device = B::complex_device(&lhs);
        let shape = B::complex_shape(&lhs);
        let scalar_complex: B::ComplexElem = rhs.elem();
        let scalar_tensor = B::complex_full(shape, scalar_complex, &device);
        B::complex_mul(lhs, scalar_tensor)
    }

    fn div(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        B::complex_div(lhs, rhs)
    }

    fn div_scalar(lhs: Self::Primitive, rhs: Scalar) -> Self::Primitive {
        // TODO: Implement complex_div_scalar in ComplexTensorOps
        let device = B::complex_device(&lhs);
        let shape = B::complex_shape(&lhs);
        let scalar_complex: B::ComplexElem = rhs.elem();
        let scalar_tensor = B::complex_full(shape, scalar_complex, &device);
        B::complex_div(lhs, scalar_tensor)
    }

    fn abs(tensor: Self::Primitive) -> Self::Primitive {
        // For complex numbers, abs returns the magnitude as a complex number (real part = magnitude, imag = 0)
        let magnitude = B::complex_abs(tensor.clone());
        let zeros = B::float_zeros(
            B::complex_shape(&tensor),
            &B::complex_device(&tensor),
            match tensor.dtype() {
                DType::Complex32 => FloatDType::F32,
                DType::Complex64 => FloatDType::F64,
                _ => panic!("Unsupported complex dtype"),
            },
        );
        B::complex_from_parts(magnitude, zeros)
    }

    fn powf(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        B::complex_powc(lhs, rhs)
    }

    fn powf_scalar(lhs: Self::Primitive, rhs: Scalar) -> Self::Primitive {
        let device = B::complex_device(&lhs);
        let shape = B::complex_shape(&lhs);
        let scalar_complex: B::ComplexElem = rhs.elem();
        let scalar_tensor = B::complex_full(shape, scalar_complex, &device);
        B::complex_powc(lhs, scalar_tensor)
    }

    fn random(shape: Shape, distribution: Distribution, device: &Device<B>) -> Self::Primitive {
        B::complex_random(shape, distribution, device)
    }

    fn remainder(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        // not mathematically defined; mimic float backend remainder
        B::complex_remainder(lhs, rhs)
    }

    fn remainder_scalar(lhs: Self::Primitive, rhs: Scalar) -> Self::Primitive {
        B::complex_remainder_scalar(lhs, rhs.elem())
    }

    fn sum(tensor: Self::Primitive) -> Self::Primitive {
        B::complex_sum(tensor)
    }

    fn sum_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        B::complex_sum_dim(tensor, dim)
    }

    fn prod(tensor: Self::Primitive) -> Self::Primitive {
        B::complex_prod(tensor)
    }

    fn prod_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        B::complex_prod_dim(tensor, dim)
    }

    fn mean(tensor: Self::Primitive) -> Self::Primitive {
        B::complex_mean(tensor)
    }

    fn mean_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        B::complex_mean_dim(tensor, dim)
    }

    // fn equal_elem(lhs: Self::Primitive, rhs: Self::Elem) -> B::BoolTensorPrimitive {
    //     B::complex_equal_elem(lhs, rhs)
    // }

    // fn not_equal_elem(lhs: Self::Primitive, rhs: Self::Elem) -> B::BoolTensorPrimitive {
    //     B::complex_not_equal_elem(lhs, rhs)
    // }

    fn powi(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        B::complex_powi(lhs, rhs)
    }

    fn powi_scalar(lhs: Self::Primitive, rhs: Scalar) -> Self::Primitive {
        B::complex_powi_scalar(lhs, rhs.elem())
    }

    fn matmul(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        B::complex_matmul(lhs, rhs)
    }

    fn cumsum(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        B::complex_cumsum(tensor, dim)
    }

    fn cumprod(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        B::complex_cumprod(tensor, dim)
    }

    fn neg(tensor: Self::Primitive) -> Self::Primitive {
        B::complex_neg(tensor)
    }

    fn sign(tensor: Self::Primitive) -> Self::Primitive {
        todo!()
    }

    fn add_scalar(lhs: Self::Primitive, rhs: burn_tensor::Scalar) -> Self::Primitive {
        // TODO: Implement complex_add_scalar in ComplexTensorOps
        // For now, create a tensor with the scalar value and use add
        let device = B::complex_device(&lhs);
        let shape = B::complex_shape(&lhs);
        let scalar_complex: B::ComplexElem = rhs.elem();
        let scalar_tensor = B::complex_full(shape, scalar_complex, &device);
        B::complex_add(lhs, scalar_tensor)
    }
}

// // Complex-specific methods for Tensor<B, D, Complex>
// impl<B, const D: usize> Tensor<B, D, Complex>
// where
//     B: ComplexTensorBackend,
// {
//     /// Returns the complex conjugate of the tensor.
//     ///
//     /// For a complex number z = a + bi, the conjugate is z* = a - bi.
//     pub fn conj(self) -> Self {
//         Self::new(B::complex_conj(self.primitive))
//     }

//     /// Returns the real part of the complex tensor as a float tensor.
//     pub fn real(self) -> Tensor<B, D> {
//         Tensor::new(TensorPrimitive::Float(B::complex_real(self.primitive)))
//     }

//     /// Returns the imaginary part of the complex tensor as a float tensor.
//     pub fn imag(self) -> Tensor<B, D> {
//         Tensor::new(TensorPrimitive::Float(B::complex_imag(self.primitive)))
//     }

//     /// Returns the magnitude (absolute value) of the complex tensor as a float tensor.
//     pub fn magnitude(self) -> Tensor<B, D> {
//         Tensor::new(TensorPrimitive::Float(B::complex_abs(self.primitive)))
//     }

//     /// Returns the phase (argument) of the complex tensor as a float tensor.
//     pub fn phase(self) -> Tensor<B, D> {
//         Tensor::new(TensorPrimitive::Float(B::complex_arg(self.primitive)))
//     }

//     /// Creates a complex tensor from real and imaginary parts.
//     pub fn from_parts(real: Tensor<B, D>, imag: Tensor<B, D>) -> Self {
//         Self::new(B::complex_from_parts(
//             real.primitive.tensor(),
//             imag.primitive.tensor(),
//         ))
//     }

//     /// Creates a complex tensor from magnitude and phase (polar coordinates).
//     pub fn from_polar(magnitude: Tensor<B, D>, phase: Tensor<B, D>) -> Self {
//         Self::new(B::complex_from_polar(
//             magnitude.primitive.tensor(),
//             phase.primitive.tensor(),
//         ))
//     }

//     /// Complex exponential function.
//     pub fn exp(self) -> Self {
//         Self::new(B::complex_exp(self.primitive))
//     }

//     /// Complex natural logarithm.
//     pub fn log(self) -> Self {
//         Self::new(B::complex_log(self.primitive))
//     }

//     /// Complex power function.
//     pub fn powc(self, rhs: Self) -> Self {
//         Self::new(B::complex_powc(self.primitive, rhs.primitive))
//     }

//     /// Complex square root.
//     pub fn sqrt(self) -> Self {
//         Self::new(B::complex_sqrt(self.primitive))
//     }

//     /// Complex sine function.
//     pub fn sin(self) -> Self {
//         Self::new(B::complex_sin(self.primitive))
//     }

//     /// Complex cosine function.
//     pub fn cos(self) -> Self {
//         Self::new(B::complex_cos(self.primitive))
//     }

//     /// Complex tangent function.
//     pub fn tan(self) -> Self {
//         Self::new(B::complex_tan(self.primitive))
//     }
// }

impl<B: ComplexTensorBackend> TensorKind<B> for Complex {
    type Primitive = B::ComplexTensorPrimitive;
    fn name() -> &'static str {
        "Complex"
    }
}
