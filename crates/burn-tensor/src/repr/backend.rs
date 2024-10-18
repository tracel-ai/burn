use crate::{
    backend::Backend,
    ops::{BoolTensor, FloatTensor, IntTensor, QuantizedTensor},
    quantization::QuantizationScheme,
    Shape,
};

/// A tensor representation containing a reference to a tensor resource with a given shape.
#[derive(Clone)]
pub struct TensorHandle<H: Clone> {
    /// The type that can be used to point to a tensor of any kind.
    pub handle: H,
    /// The shape associated to the tensor.
    pub shape: Shape,
}

/// A simple struct to encapsulate a quantized tensor kind.
#[derive(Clone)]
pub struct QuantizedKind<T: Clone> {
    /// The quantized tensor.
    pub tensor: T,
    /// The scaling factor.
    pub scale: T,
    /// The zero-point offset.
    pub offset: Option<T>,
}

/// Backend extension trait that allows an existing [backend](Backend) to use the Burn tensor representation
/// for compilation purpose or other...
pub trait ReprBackend: Backend {
    /// The type that can be used to point to a tensor of any kind.
    type Handle: Sync + Send + Clone;

    /// Convert a [handle](ReprBackend::Handle) to a [float tensor](Backend::FloatTensorPrimitive).
    fn float_tensor(handle: TensorHandle<Self::Handle>) -> FloatTensor<Self>;
    /// Convert a [handle](ReprBackend::Handle) to an [int tensor](Backend::IntTensorPrimitive).
    fn int_tensor(handle: TensorHandle<Self::Handle>) -> IntTensor<Self>;
    /// Convert a [handle](ReprBackend::Handle) to a [bool tensor](Backend::BoolTensorPrimitive).
    fn bool_tensor(handle: TensorHandle<Self::Handle>) -> BoolTensor<Self>;
    /// Convert a [handle](ReprBackend::Handle) to a [quantized tensor](Backend::QuantizedTensorPrimitive).
    fn quantized_tensor(
        handle: QuantizedKind<TensorHandle<Self::Handle>>,
        scheme: QuantizationScheme,
    ) -> QuantizedTensor<Self>;

    /// Convert a [float tensor](Backend::FloatTensorPrimitive) to a [handle](ReprBackend::Handle).
    fn float_tensor_handle(tensor: FloatTensor<Self>) -> Self::Handle;
    /// Convert an [int tensor](Backend::IntTensorPrimitive) to a [handle](ReprBackend::Handle).
    fn int_tensor_handle(tensor: IntTensor<Self>) -> Self::Handle;
    /// Convert a [bool tensor](Backend::BoolTensorPrimitive) to a [handle](ReprBackend::Handle).
    fn bool_tensor_handle(tensor: BoolTensor<Self>) -> Self::Handle;
    /// Convert a [quantized tensor](Backend::QuantizedTensorPrimitive) to a [handle](ReprBackend::Handle).
    /// A quantized tensor has multiple handles for the tensor itself and the quantization parameters.
    fn quantized_tensor_handle(tensor: QuantizedTensor<Self>) -> QuantizedKind<Self::Handle>;
}

/// Handle which points to a backend tensor primitive kind.
#[derive(Clone, Debug)]
pub enum HandleKind<B: Backend> {
    /// Float tensor handle.
    Float(B::FloatTensorPrimitive),
    /// Int tensor handle.
    Int(B::IntTensorPrimitive),
    /// Bool tensor handle.
    Bool(B::BoolTensorPrimitive),
    /// Quantized tensor handle.
    Quantized(B::QuantizedTensorPrimitive),
    /// Empty handle (used as a dummy representation).
    Empty,
}

impl<B: Backend> HandleKind<B> {
    /// Returns the handle kind name.
    pub fn name(&self) -> &str {
        match self {
            HandleKind::Float(_) => "float",
            HandleKind::Int(_) => "int",
            HandleKind::Bool(_) => "bool",
            HandleKind::Quantized(_) => "quantized",
            HandleKind::Empty => unreachable!(), // should not happen
        }
    }
}
