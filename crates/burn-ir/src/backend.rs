use burn_backend::{
    Backend, Shape,
    tensor::{BoolTensor, FloatTensor, IntTensor, QuantizedTensor},
};

/// A tensor representation containing a reference to a tensor resource with a given shape.
#[derive(Clone)]
pub struct TensorHandle<H: Clone> {
    /// The type that can be used to point to a tensor of any kind.
    pub handle: H,
    /// The shape associated to the tensor.
    pub shape: Shape,
}

/// Backend extension trait that allows an existing [backend](Backend) to use the Burn tensor
/// intermediate representation for compilation purpose or other...
pub trait BackendIr: Backend {
    /// The type that can be used to point to a tensor of any kind.
    type Handle: Sync + Send + Clone;

    /// Convert a [handle](BackendIr::Handle) to a [float tensor](burn_backend::BackendTypes::FloatTensorPrimitive).
    fn float_tensor(handle: TensorHandle<Self::Handle>) -> FloatTensor<Self>;
    /// Convert a [handle](BackendIr::Handle) to an [int tensor](burn_backend::BackendTypes::IntTensorPrimitive).
    fn int_tensor(handle: TensorHandle<Self::Handle>) -> IntTensor<Self>;
    /// Convert a [handle](BackendIr::Handle) to a [bool tensor](burn_backend::BackendTypes::BoolTensorPrimitive).
    fn bool_tensor(handle: TensorHandle<Self::Handle>) -> BoolTensor<Self>;
    /// Convert a [handle](BackendIr::Handle) to a [quantized tensor](burn_backend::BackendTypes::QuantizedTensorPrimitive).
    fn quantized_tensor(handle: TensorHandle<Self::Handle>) -> QuantizedTensor<Self>;

    /// Convert a [float tensor](burn_backend::BackendTypes::FloatTensorPrimitive) to a [handle](BackendIr::Handle).
    fn float_tensor_handle(tensor: FloatTensor<Self>) -> Self::Handle;
    /// Convert an [int tensor](burn_backend::BackendTypes::IntTensorPrimitive) to a [handle](BackendIr::Handle).
    fn int_tensor_handle(tensor: IntTensor<Self>) -> Self::Handle;
    /// Convert a [bool tensor](burn_backend::BackendTypes::BoolTensorPrimitive) to a [handle](BackendIr::Handle).
    fn bool_tensor_handle(tensor: BoolTensor<Self>) -> Self::Handle;
    /// Convert a [quantized tensor](burn_backend::BackendTypes::QuantizedTensorPrimitive) to a [handle](BackendIr::Handle).
    fn quantized_tensor_handle(tensor: QuantizedTensor<Self>) -> Self::Handle;
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
}

impl<B: Backend> HandleKind<B> {
    /// Returns the handle kind name.
    pub fn name(&self) -> &str {
        match self {
            HandleKind::Float(_) => "float",
            HandleKind::Int(_) => "int",
            HandleKind::Bool(_) => "bool",
            HandleKind::Quantized(_) => "quantized",
        }
    }
}
