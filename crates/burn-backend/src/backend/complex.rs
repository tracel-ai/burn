use burn_std::{ComplexDType, ComplexElement, ExecutionError, Scalar, Shape, TensorData};

use crate::{Backend, BackendTypes, TensorMetadata, ops::ComplexTensorOps, tensor::Device};

pub trait CBT: BackendTypes {
    /// a complex element in interleaved layout
    type ComplexScalar: ComplexElement;

    type ComplexTensorPrimitive: TensorMetadata + 'static;
}

/// The layout of the complex tensor. Used to define shared behavior only meant
/// to be used for a specific layout (such as butterfly operations).
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

pub trait ComplexTensorBackend: ComplexTensorOps<Self> + Sized + BackendTypes {
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

    // /// The underlying layout for the complex elements
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

// The evolution of Laziness
pub trait DefaultComplexOps<B: ComplexTensorBackend> {
    type OutTensorData;
    fn ones(shape: Shape, device: &Device<B>, dtype: ComplexDType) -> ComplexTensor<B>;
    fn zeros(shape: Shape, device: &Device<B>, dtype: ComplexDType) -> ComplexTensor<B>;
    fn full(shape: Shape, fill_value: Scalar, device: &Device<B>, dtype: ComplexDType) -> ComplexTensor<B>;
    fn complex_into_data(
        tensor: ComplexTensor<B>,
    ) -> impl Future<Output = Result<Self::OutTensorData, ExecutionError>> + Send;
}
