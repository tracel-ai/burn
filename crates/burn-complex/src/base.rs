use burn_std::ComplexDType;
/*
The base implementation for complex tensors, contains everything that would be in burn-tensor.
May get split into separate files at some point, but for now it's easier to keep all the base
definitions in one spot.
*/
use burn_tensor::{
    Bytes, ComplexElement, DType, Device, Distribution, FloatDType, IndexingUpdateOp, Scalar,
    Shape, TensorData, TensorMetadata,
    backend::{Backend, BackendTypes, ExecutionError},
    ops::{FloatTensor, IntTensorOps},
};
use serde::{Deserialize, Serialize};

pub trait CBT: BackendTypes {
    type ComplexTensorPrimitive: TensorMetadata + 'static;
}

/// The layout of the complex tensor. Used to define shared behavior only meant
/// to be used for a specific layout (such as butterfly operations).
pub trait Layout {
    // /// The complex Tensor primitive type for this layout. For interleaved, this will be
    // /// a tensor of Complex\<E\>,for split this will be a tuple tensor Complex\<FloatTensorPrimitive\<E\>, FloatTensorPrimitive\<E\>\>.
    // type ComplexTensorPrimitive: TensorMetadata + 'static;
}

/// Complex tensor primitive type used by the backend.
pub type ComplexTensor<B> = <B as CBT>::ComplexTensorPrimitive;

pub trait ComplexTensorBackend: ComplexTensorOps<Self> + Sized + CBT {
    /// The inner backend type.
    ///
    /// Must share all primitive types and device with `Self` so that operations
    /// can delegate directly without any type-level conversion.
    type InnerBackend: Backend<
            Device = Self::Device,
            FloatTensorPrimitive = Self::FloatTensorPrimitive,
            FloatElem = Self::FloatElem,
            IntTensorPrimitive = Self::IntTensorPrimitive,
            IntElem = Self::IntElem,
            BoolTensorPrimitive = Self::BoolTensorPrimitive,
            BoolElem = Self::BoolElem,
        >;

    ///// Tensor primitive to be used for all complex operations.
    //type ComplexTensorPrimitive: TensorMetadata + 'static;

    /// a complex element in interleaved layout
    type ComplexScalar: ComplexElement;

    /// The underlying layout for the complex elements
    type Layout: Layout + DefaultComplexOps<Self>;

    /// Creates a complex tensor from real-valued data, padding the imaginary part with zeros.
    ///
    /// Each element `x` in `data` becomes `x + 0i`.
    ///
    /// # Arguments
    ///
    /// * `data` - The real-valued data. Must contain scalar float elements.
    /// * `device` - The device to create the tensor on.
    ///
    /// # Returns
    ///
    /// A complex tensor with the given real values and a zero imaginary part.
    fn complex_from_real_data(data: TensorData, device: &Self::Device) -> ComplexTensor<Self>;

    /// Creates a complex tensor from imaginary-valued data, padding the real part with zeros.
    ///
    /// Each element `y` in `data` becomes `0 + yi`.
    ///
    /// # Arguments
    ///
    /// * `data` - The imaginary-valued data. Must contain scalar float elements.
    /// * `device` - The device to create the tensor on.
    ///
    /// # Returns
    ///
    /// A complex tensor with a zero real part and the given imaginary values.
    fn complex_from_imag_data(data: TensorData, device: &Self::Device) -> ComplexTensor<Self>;

    /// Creates a complex tensor from interleaved complex data.
    ///
    /// The `data` buffer must already be in interleaved layout, i.e. elements are ordered as
    /// `[re₀, im₀, re₁, im₁, …]`. No conversion is performed — the buffer is used directly.
    ///
    /// # Arguments
    ///
    /// * `data` - Interleaved complex data with alternating real and imaginary values.
    /// * `device` - The device to create the tensor on.
    ///
    /// # Returns
    ///
    /// A complex tensor backed by the interleaved buffer.
    fn complex_from_interleaved_data(
        data: TensorData,
        device: &Self::Device,
    ) -> ComplexTensor<Self>;

    /// Creates a complex tensor from two separate real and imaginary data buffers.
    ///
    /// Both buffers must have the same shape and element type. They are combined into a single
    /// complex tensor, pairing each `real_data[i]` with `imag_data[i]` as `re + im·i`.
    ///
    /// # Arguments
    ///
    /// * `real_data` - The real parts.
    /// * `imag_data` - The imaginary parts. Must match the shape and dtype of `real_data`.
    /// * `device` - The device to create the tensor on.
    ///
    /// # Returns
    ///
    /// A complex tensor constructed from the paired real and imaginary buffers.
    fn complex_from_parts_data(
        real_data: TensorData,
        imag_data: TensorData,
        device: &Self::Device,
    ) -> ComplexTensor<Self>;
}

//Note: changing to adopt terminology used in fftw doc

/// Indicates that the underlying implementation has separate real and imaginary tensors.
pub struct SplitLayout {
    //_marker: core::marker::PhantomData<T>,
}

/// Indicates that the underlying implementation uses a complex primitive type \[float,float\] like that found in the
/// num_complex trait.
pub struct InterleavedLayout {
    //_marker: core::marker::PhantomData<E>,
}

/// Data structure for tensors.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitTensorData {
    /// The real values of the tensor (as bytes).
    pub real_bytes: Bytes,

    /// The imaginary values of the tensor (as bytes).
    pub imag_bytes: Bytes,

    #[serde(with = "shape_inner")]
    /// The shape of the tensor.
    pub shape: Shape,

    /// The data type of the tensor.
    pub dtype: DType,
}

mod shape_inner {
    use burn_std::SmallVec;

    use super::*;

    pub fn serialize<S: serde::Serializer>(
        shape: &Shape,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        shape.as_slice().serialize(serializer)
    }

    pub fn deserialize<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Shape, D::Error> {
        let dims = SmallVec::<[usize; _]>::deserialize(deserializer)?;
        Ok(Shape::new_raw(dims))
    }
}

// impl<T: TensorMetadata + 'static> Layout for InterleavedLayout<T> {
//     type ComplexTensorPrimitive = T;
// }

impl Layout for InterleavedLayout {}

// The evolution of Laziness
pub trait DefaultComplexOps<B: ComplexTensorBackend> {
    type OutTensorData;
    fn ones(shape: Shape, device: &Device<B>, dtype: ComplexDType) -> ComplexTensor<B>;
    fn zeros(shape: Shape, device: &Device<B>, dtype: ComplexDType) -> ComplexTensor<B>;
    fn full(shape: Shape, fill_value: B::ComplexScalar, device: &Device<B>) -> ComplexTensor<B>;
    fn complex_into_data(
        tensor: ComplexTensor<B>,
    ) -> impl Future<Output = Result<Self::OutTensorData, ExecutionError>> + Send;
}

impl<B> DefaultComplexOps<B> for InterleavedLayout
where
    B: ComplexTensorBackend<Layout = InterleavedLayout>,
{
    type OutTensorData = TensorData;

    fn ones(shape: Shape, device: &Device<B>, dtype: ComplexDType) -> ComplexTensor<B> {
        B::complex_from_interleaved_data(TensorData::ones::<B::ComplexScalar, _>(shape), device)
    }

    fn zeros(shape: Shape, device: &Device<B>, dtype: ComplexDType) -> ComplexTensor<B> {
        B::complex_from_interleaved_data(TensorData::zeros::<B::ComplexScalar, _>(shape), device)
    }

    fn full(shape: Shape, fill_value: B::ComplexScalar, device: &Device<B>) -> ComplexTensor<B> {
        B::complex_from_interleaved_data(TensorData::full(shape, fill_value), device)
    }

    async fn complex_into_data(
        tensor: ComplexTensor<B>,
    ) -> Result<Self::OutTensorData, ExecutionError> {
        B::complex_into_interleaved_data(tensor).await
    }
}

type OutTensorData<B> =
    <<B as ComplexTensorBackend>::Layout as DefaultComplexOps<B>>::OutTensorData;
/// Operations on complex tensors.
pub trait ComplexTensorOps<B: ComplexTensorBackend> {
    /// Converts the tensor's real component to a data structure.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The data structure with the tensor's data.
    fn complex_into_real_data(
        tensor: ComplexTensor<B>,
    ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send;

    /// Converts the tensor's imaginary component to a data structure.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The data structure with the tensor's imaginary data.
    fn complex_into_imag_data(
        tensor: ComplexTensor<B>,
    ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send;

    /// Converts the tensor to interleaved complex data, where real and imaginary parts alternate.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The data structure with the tensor's data in interleaved format.
    fn complex_into_interleaved_data(
        tensor: ComplexTensor<B>,
    ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send;

    /// Converts the tensor to split complex data, returning real and imaginary parts separately.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// A tuple of data structures containing the real and imaginary parts of the tensor's data.
    fn complex_into_split_data(
        tensor: ComplexTensor<B>,
    ) -> impl Future<Output = Result<SplitTensorData, ExecutionError>> + Send;

    /// Converts a real float tensor to a complex tensor with zero imaginary part.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The float tensor.
    ///
    /// # Returns
    ///
    /// A complex tensor with the same values as `tensor` and a zero imaginary part.
    fn to_complex(tensor: FloatTensor<B>) -> ComplexTensor<B>;

    // was going to add a norm function here, but float tensor ops doesn't have a hypot function
    // easy enough to add, but a bit out of scope for this PR

    /// Returns the squared norm (squared magnitude) of each element of the complex tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The complex tensor.
    ///
    /// # Returns
    ///
    /// A float tensor with the squared norm (i.e., `re² + im²`) of each element.
    fn complex_squared_norm(tensor: ComplexTensor<B>) -> FloatTensor<B>;

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
        dtype: FloatDType,
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
    fn complex_zeros(shape: Shape, device: &Device<B>, dtype: ComplexDType) -> ComplexTensor<B> {
        B::Layout::zeros(shape, device, dtype)
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
    fn complex_ones(shape: Shape, device: &Device<B>, dtype: ComplexDType) -> ComplexTensor<B> {
        B::Layout::ones(shape, device, dtype)
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
        fill_value: B::ComplexScalar,
        device: &Device<B>,
    ) -> ComplexTensor<B> {
        B::Layout::full(shape, fill_value, device)
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
    fn complex_shape(tensor: &ComplexTensor<B>) -> Shape {
        tensor.shape()
    }

    /// Gets the device of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The device of the tensor.
    fn complex_device(tensor: &ComplexTensor<B>) -> B::Device;

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
    fn complex_to_device(tensor: ComplexTensor<B>, device: &B::Device) -> ComplexTensor<B>;

    /// Converts the tensor to a data structure.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The data structure with the tensor's data.
    fn complex_into_data(
        tensor: ComplexTensor<B>,
    ) -> impl Future<Output = Result<OutTensorData<B>, ExecutionError>> + Send {
        B::Layout::complex_into_data(tensor)
    }

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
    fn conj(tensor: ComplexTensor<B>) -> ComplexTensor<B>;

    /// Returns the real part of the complex tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The complex tensor.
    ///
    /// # Returns
    ///
    /// A float tensor containing the real parts.
    fn real(tensor: ComplexTensor<B>) -> FloatTensor<B>;

    /// Returns the imaginary part of the complex tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The complex tensor.
    ///
    /// # Returns
    ///
    /// A float tensor containing the imaginary parts.
    fn imag(tensor: ComplexTensor<B>) -> FloatTensor<B>;

    /// Returns the magnitude (absolute value) of the complex tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The complex tensor.
    ///
    /// # Returns
    ///
    /// A float tensor containing the magnitudes.
    fn abs(tensor: ComplexTensor<B>) -> FloatTensor<B>;

    /// Returns the phase (argument) of the complex tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The complex tensor.
    ///
    /// # Returns
    ///
    /// A float tensor containing the phases in radians.
    fn complex_arg(tensor: ComplexTensor<B>) -> FloatTensor<B>;

    #[inline]
    /// Returns the phase (argument) of the complex tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The complex tensor.
    ///
    /// # Returns
    ///
    /// A float tensor containing the phases in radians.
    fn phase(tensor: ComplexTensor<B>) -> FloatTensor<B> {
        Self::complex_arg(tensor)
    }

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
    fn complex_from_parts(real: TensorData, imag: TensorData) -> ComplexTensor<B>;

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
    fn complex_from_polar(magnitude: FloatTensor<B>, phase: FloatTensor<B>) -> ComplexTensor<B>;

    // formula: e^(a + bi) = e^a (cos(b) + i*sin(b)) = from_polar(e^a, b)
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
    fn complex_select(
        tensor: ComplexTensor<B>,
        dim: usize,
        indices: B::IntTensorPrimitive,
    ) -> ComplexTensor<B>;

    /// Select tensor elements corresponding to the given slices.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to select from.
    /// * `slices` - The slices specifying ranges and steps for each dimension.
    ///
    /// # Returns
    ///
    /// The selected elements in a new tensor.
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

    fn complex_scatter_nd(
        tensor: ComplexTensor<B>,
        indices: B::IntTensorPrimitive,
        value: ComplexTensor<B>,
        reduction: IndexingUpdateOp,
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
    fn complex_equal(
        lhs: ComplexTensor<B>,
        rhs: ComplexTensor<B>,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive;

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
    fn complex_not_equal(
        lhs: ComplexTensor<B>,
        rhs: ComplexTensor<B>,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive;

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
    fn complex_cat(tensors: alloc::vec::Vec<ComplexTensor<B>>, dim: usize) -> ComplexTensor<B>;

    /// Tests if any element in the `tensor` evaluates to True.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to test.
    ///
    /// # Returns
    ///
    /// A boolean tensor with a single element, True if any element in the tensor is True, False otherwise.
    fn complex_any(
        tensor: ComplexTensor<B>,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive;

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
    fn complex_any_dim(
        tensor: ComplexTensor<B>,
        dim: usize,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive;

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
    fn complex_all(
        tensor: ComplexTensor<B>,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive;

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
    fn complex_all_dim(
        tensor: ComplexTensor<B>,
        dim: usize,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive;

    /// Permutes the dimensions of a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to permute the dimensions of.
    /// * `axes` - The new order of the dimensions.
    ///
    /// # Returns
    ///
    /// The tensor with the dimensions permuted.
    fn complex_permute(tensor: ComplexTensor<B>, axes: &[usize]) -> ComplexTensor<B>;

    /// Broadcasts the complex `tensor` to the given `shape`.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to broadcast.
    /// * `shape` - The target shape.
    ///
    /// # Returns
    ///
    /// The tensor broadcast to the given shape.
    fn complex_expand(tensor: ComplexTensor<B>, shape: Shape) -> ComplexTensor<B>;

    /// Reverse the order of elements in a tensor along the given axes.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to reverse.
    /// * `axes` - The axes to reverse.
    ///
    /// # Returns
    ///
    /// The tensor with the elements reversed.
    fn complex_flip(tensor: ComplexTensor<B>, axes: &[usize]) -> ComplexTensor<B>;

    /// Unfold windows along a dimension.
    ///
    /// Returns a view of the tensor with all complete windows of size `size` in dimension `dim`;
    /// where windows are advanced by `step` at each index.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    /// * `dim` - The selected dimension.
    /// * `size` - The size of each unfolded window.
    /// * `step` - The step between each window.
    ///
    /// # Returns
    ///
    /// A tensor view with an additional trailing dimension of size `size`.
    fn complex_unfold(
        tensor: ComplexTensor<B>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> ComplexTensor<B>;

    /// Assign the selected elements along the given dimension corresponding for the given indices
    /// to the given value using sum reduction.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to assign to.
    /// * `dim` - The dimension to assign along.
    /// * `indices` - The indices to assign to.
    /// * `values` - The values to assign.
    ///
    /// # Returns
    ///
    /// The tensor with the selected elements assigned to the given values.
    fn complex_select_add(
        tensor: ComplexTensor<B>,
        dim: usize,
        indices: B::IntTensorPrimitive,
        values: ComplexTensor<B>,
    ) -> ComplexTensor<B>;

    /// Sum of all elements in a complex tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to sum.
    ///
    /// # Returns
    ///
    /// A scalar complex tensor with the sum of all elements in `tensor`.
    fn complex_sum(tensor: ComplexTensor<B>) -> ComplexTensor<B>;

    /// Sum of all elements in a complex tensor along a dimension.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to sum.
    /// * `dim` - The dimension along which to sum.
    ///
    /// # Returns
    ///
    /// A tensor with the sum of all elements in `tensor` along `dim`.
    fn complex_sum_dim(tensor: ComplexTensor<B>, dim: usize) -> ComplexTensor<B>;

    /// Product of all elements in a complex tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to take the product of.
    ///
    /// # Returns
    ///
    /// A scalar complex tensor with the product of all elements in `tensor`.
    fn complex_prod(tensor: ComplexTensor<B>) -> ComplexTensor<B>;

    /// Product of all elements in a complex tensor along a dimension.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to take the product of.
    /// * `dim` - The dimension along which to take the product.
    ///
    /// # Returns
    ///
    /// A tensor with the product of all elements in `tensor` along `dim`.
    fn complex_prod_dim(tensor: ComplexTensor<B>, dim: usize) -> ComplexTensor<B>;

    /// Mean of all elements in a complex tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to compute the mean of.
    ///
    /// # Returns
    ///
    /// A scalar complex tensor with the mean of all elements in `tensor`.
    fn complex_mean(tensor: ComplexTensor<B>) -> ComplexTensor<B>;

    /// Mean of all elements in a complex tensor along a dimension.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to compute the mean of.
    /// * `dim` - The dimension along which to compute the mean.
    ///
    /// # Returns
    ///
    /// A tensor with the mean of all elements in `tensor` along `dim`.
    fn complex_mean_dim(tensor: ComplexTensor<B>, dim: usize) -> ComplexTensor<B>;

    /// Computes the remainder of division between two complex tensors element-wise.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left-hand side tensor.
    /// * `rhs` - The right-hand side tensor.
    ///
    /// # Returns
    ///
    /// The element-wise remainder when dividing `lhs` by `rhs`.
    fn complex_remainder(lhs: ComplexTensor<B>, rhs: ComplexTensor<B>) -> ComplexTensor<B>;

    /// Computes the remainder of division between a complex tensor and a scalar element-wise.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left-hand side tensor.
    /// * `rhs` - The right-hand side complex scalar.
    ///
    /// # Returns
    ///
    /// The element-wise remainder when dividing `lhs` by `rhs`.
    fn complex_remainder_scalar(lhs: ComplexTensor<B>, rhs: B::ComplexScalar) -> ComplexTensor<B>;

    /// Equal comparison of a complex tensor and a scalar.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left-hand side tensor.
    /// * `rhs` - The right-hand side complex scalar.
    /// * `out_dtype` - The output tensor dtype.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the result of the comparison.
    fn complex_equal_elem(
        lhs: ComplexTensor<B>,
        rhs: B::ComplexScalar,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive;

    /// Element-wise non-equality comparison of a complex tensor and a scalar.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left-hand side tensor.
    /// * `rhs` - The right-hand side complex scalar.
    /// * `out_dtype` - The output tensor dtype.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the result of the comparison.
    fn complex_not_equal_elem(
        lhs: ComplexTensor<B>,
        rhs: B::ComplexScalar,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive;

    /// Update the given tensor with the source tensor where the mask is true.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to select from.
    /// * `mask` - The boolean mask to select with.
    /// * `source` - The source tensor to assign to the selected elements.
    ///
    /// # Returns
    ///
    /// The tensor with the selected elements assigned to the corresponding values in `source`.
    fn complex_mask_where(
        tensor: ComplexTensor<B>,
        mask: B::BoolTensorPrimitive,
        source: ComplexTensor<B>,
    ) -> ComplexTensor<B>;

    /// Update the given tensor with the scalar value where the mask is true.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to select from.
    /// * `mask` - The boolean mask to select with.
    /// * `value` - The complex scalar value to assign to the selected elements.
    ///
    /// # Returns
    ///
    /// The tensor with the selected elements assigned to `value`.
    fn complex_mask_fill(
        tensor: ComplexTensor<B>,
        mask: B::BoolTensorPrimitive,
        value: B::ComplexScalar,
    ) -> ComplexTensor<B>;

    /// Gather elements from a complex tensor.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension to gather from.
    /// * `tensor` - The tensor to gather from.
    /// * `indices` - The indices to gather.
    ///
    /// # Returns
    ///
    /// The gathered elements.
    fn complex_gather(
        dim: usize,
        tensor: ComplexTensor<B>,
        indices: B::IntTensorPrimitive,
    ) -> ComplexTensor<B>;

    /// Scatter elements into a complex tensor using sum reduction.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension to scatter into.
    /// * `tensor` - The tensor to scatter into.
    /// * `indices` - The indices to scatter into.
    /// * `values` - The values to scatter.
    ///
    /// # Returns
    ///
    /// The tensor with the scattered elements.
    fn complex_scatter_add(
        dim: usize,
        tensor: ComplexTensor<B>,
        indices: B::IntTensorPrimitive,
        values: ComplexTensor<B>,
    ) -> ComplexTensor<B>;

    /// Returns the sign of each complex element as a unit complex number.
    ///
    /// Unlike the float sign which returns -1, 0, or 1, the complex sign returns a complex number
    /// on the unit circle (i.e., `z / |z|`), retaining information about the angle. For zero
    /// elements, the result is zero.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to extract the signs from.
    ///
    /// # Returns
    ///
    /// A complex tensor with the same shape as `tensor` containing the unit-circle signs.
    fn complex_sign(tensor: ComplexTensor<B>) -> ComplexTensor<B>;

    /// Element-wise complex power with a complex scalar.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The base tensor.
    /// * `rhs` - The complex scalar exponent.
    ///
    /// # Returns
    ///
    /// The elements of `lhs` raised to the power of `rhs`.
    fn complex_powc_scalar(lhs: ComplexTensor<B>, rhs: B::ComplexScalar) -> ComplexTensor<B>;

    /// Element-wise complex power with a float tensor.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The base complex tensor.
    /// * `rhs` - The float exponent tensor.
    ///
    /// # Returns
    ///
    /// The elements of `lhs` raised to the power of the corresponding elements of `rhs`.
    fn complex_powf(lhs: ComplexTensor<B>, rhs: FloatTensor<B>) -> ComplexTensor<B>;

    /// Element-wise complex power with an integer tensor.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The base complex tensor.
    /// * `rhs` - The integer exponent tensor.
    ///
    /// # Returns
    ///
    /// The elements of `lhs` raised to the power of the corresponding elements of `rhs`.
    fn complex_powi(lhs: ComplexTensor<B>, rhs: B::IntTensorPrimitive) -> ComplexTensor<B>
    where
        B::InnerBackend: IntTensorOps<B::InnerBackend>,
        // make the equality explicit at the use site
        <B::InnerBackend as BackendTypes>::IntTensorPrimitive: From<B::IntTensorPrimitive>,
    {
        let dtype = crate::utils::complex_to_real_dtype(lhs.dtype());

        Self::complex_powf(
            lhs,
            <B::InnerBackend as IntTensorOps<B::InnerBackend>>::int_into_float(
                rhs,
                FloatDType::from(dtype),
            ),
        )
    }

    /// Element-wise complex power with a float scalar.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The base complex tensor.
    /// * `rhs` - The float scalar exponent.
    ///
    /// # Returns
    ///
    /// The elements of `lhs` raised to the power of `rhs`.
    fn complex_powf_scalar(lhs: ComplexTensor<B>, rhs: Scalar) -> ComplexTensor<B>;

    /// Element-wise complex power with an integer scalar.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The base complex tensor.
    /// * `rhs` - The integer scalar exponent.
    ///
    /// # Returns
    ///
    /// The elements of `lhs` raised to the power of `rhs`.
    fn complex_powi_scalar(lhs: ComplexTensor<B>, rhs: Scalar) -> ComplexTensor<B> {
        Self::complex_powf_scalar(lhs, rhs)
    }

    /// Multiplies two complex tensors together using matrix multiplication.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left-hand side tensor.
    /// * `rhs` - The right-hand side tensor.
    ///
    /// # Returns
    ///
    /// The result of multiplying the two tensors together using matrix multiplication.
    fn complex_matmul(lhs: ComplexTensor<B>, rhs: ComplexTensor<B>) -> ComplexTensor<B>;

    /// Computes the cumulative sum of elements along a dimension.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to compute the cumulative sum of.
    /// * `dim` - The dimension along which to compute the cumulative sum.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape where each element is the cumulative sum
    /// of all elements up to and including that position along the dimension.
    fn complex_cumsum(tensor: ComplexTensor<B>, dim: usize) -> ComplexTensor<B>;

    /// Computes the cumulative product of elements along a dimension.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to compute the cumulative product of.
    /// * `dim` - The dimension along which to compute the cumulative product.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape where each element is the cumulative product
    /// of all elements up to and including that position along the dimension.
    fn complex_cumprod(tensor: ComplexTensor<B>, dim: usize) -> ComplexTensor<B>;
}
