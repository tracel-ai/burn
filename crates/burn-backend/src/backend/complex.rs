use burn_std::{ComplexDType, ExecutionError, Scalar, Shape, TensorData};

use crate::{Backend, BackendTypes, TypedDevice, ops::ComplexTensorOps, tensor::Device};

/// Describes the memory layout used by complex tensor primitives.
///
/// This marker trait is used to select layout-specific behavior through trait bounds,
/// for example when implementing optimized kernels that rely on a particular memory
/// representation.
///
/// Current layouts include interleaved complex values (real/imaginary pairs stored
/// together), but additional layouts may be added in the future.
pub trait Layout {
    // /// The complex Tensor primitive type for this layout. For interleaved, this will be
    // /// a tensor of Complex\<E\>,for split this will be a tuple tensor Complex\<FloatTensorPrimitive\<E\>, FloatTensorPrimitive\<E\>\>.
    //type ComplexTensorPrimitive: TensorMetadata + 'static;
}

/// Indicates that the underlying implementation uses a complex primitive type \[float,float\] like that found in the
/// num_complex trait.
pub struct InterleavedLayout {
    //_marker: core::marker::PhantomData<E>,
}

impl Layout for InterleavedLayout {}

/// Complex tensor primitive type used by the backend.
pub type ComplexTensor<B> = <B as BackendTypes>::ComplexTensorPrimitive;

/// Backend for complex tensor operations. This trait can act either as an extension or a wrapper
/// around a standard `Backend` implementation. For backends that don't yet natively support complex
/// operations, a default implementation exists for any backend that sets the complex primitive to
/// UnimplmentedTensorPrimitive.
pub trait ComplexTensorBackend:
    ComplexTensorOps<Self> + Sized + BackendTypes + TypedDevice<Self>
{
    /// The inner backend type.
    ///
    /// Must share all primitive types and device with `Self` so that operations
    /// can delegate directly without any type-level conversion.
    type InnerBackend: Backend<
            Device = Self::Device,
            FloatTensorPrimitive = Self::FloatTensorPrimitive,
            IntTensorPrimitive = Self::IntTensorPrimitive,
            BoolTensorPrimitive = Self::BoolTensorPrimitive,
        >;

    ///// Tensor primitive to be used for all complex operations.
    //type ComplexTensorPrimitive: TensorMetadata + 'static;

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

/// Default construction and host-transfer operations for complex tensors.
///
/// This trait centralizes basic complex tensor behaviors that can vary by layout
/// and backend implementation, while still exposing a consistent API to higher
/// layers.
///
/// Implementors are expected to respect the provided complex dtype and allocate
/// tensors on the requested device.
pub trait DefaultComplexOps<B: ComplexTensorBackend> {
    /// Backend-specific data type returned when materializing complex tensor data on host.
    type OutTensorData;

    /// Creates a complex tensor filled with one-valued complex numbers.
    ///
    /// Every element is initialized to `1 + 0i` using the requested complex dtype.
    ///
    /// # Arguments
    ///
    /// * `shape` - Output tensor shape.
    /// * `device` - Device where the tensor is allocated.
    /// * `dtype` - Complex element dtype to use.
    fn ones(shape: Shape, device: &Device<B>, dtype: ComplexDType) -> ComplexTensor<B>;

    /// Creates a complex tensor filled with zeros.
    ///
    /// Every element is initialized to `0 + 0i` using the requested complex dtype.
    ///
    /// # Arguments
    ///
    /// * `shape` - Output tensor shape.
    /// * `device` - Device where the tensor is allocated.
    /// * `dtype` - Complex element dtype to use.
    fn zeros(shape: Shape, device: &Device<B>, dtype: ComplexDType) -> ComplexTensor<B>;

    /// Creates a complex tensor filled with a scalar value.
    ///
    /// Real scalars are promoted to complex values with zero imaginary component.
    /// Implementations may also accept complex scalars depending on backend support.
    ///
    /// # Arguments
    ///
    /// * `shape` - Output tensor shape.
    /// * `fill_value` - Scalar value used to initialize all elements.
    /// * `device` - Device where the tensor is allocated.
    /// * `dtype` - Complex element dtype to use.
    fn full(
        shape: Shape,
        fill_value: Scalar,
        device: &Device<B>,
        dtype: ComplexDType,
    ) -> ComplexTensor<B>;

    /// Asynchronously materializes a complex tensor into host-accessible data.
    ///
    /// This operation may involve device-to-host transfer and synchronization.
    ///
    /// # Errors
    ///
    /// Returns an [`ExecutionError`] if the transfer or synchronization fails.
    fn complex_into_data(
        tensor: ComplexTensor<B>,
    ) -> impl Future<Output = Result<Self::OutTensorData, ExecutionError>> + Send;
}
