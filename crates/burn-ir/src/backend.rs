use burn_tensor::{
    Shape,
    backend::Backend,
    ops::{BoolTensor, FloatTensor, IntTensor, QuantizedTensor},
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

    /// Convert a [handle](BackendIr::Handle) to a [float tensor](Backend::FloatTensorPrimitive).
    fn float_tensor(handle: TensorHandle<Self::Handle>) -> FloatTensor<Self>;
    /// Convert a [handle](BackendIr::Handle) to an [int tensor](Backend::IntTensorPrimitive).
    fn int_tensor(handle: TensorHandle<Self::Handle>) -> IntTensor<Self>;
    /// Convert a [handle](BackendIr::Handle) to a [bool tensor](Backend::BoolTensorPrimitive).
    fn bool_tensor(handle: TensorHandle<Self::Handle>) -> BoolTensor<Self>;
    /// Convert a [handle](BackendIr::Handle) to a [quantized tensor](Backend::QuantizedTensorPrimitive).
    fn quantized_tensor(handle: TensorHandle<Self::Handle>) -> QuantizedTensor<Self>;
    // /// Convert a [handle](BackendIr::Handle) to a [complex tensor](Backend::ComplexTensorPrimitive).
    // fn complex_tensor(handle: TensorHandle<Self::Handle>) -> ComplexTensor<Self>;

    /// Convert a [float tensor](Backend::FloatTensorPrimitive) to a [handle](BackendIr::Handle).
    fn float_tensor_handle(tensor: FloatTensor<Self>) -> Self::Handle;
    /// Convert an [int tensor](Backend::IntTensorPrimitive) to a [handle](BackendIr::Handle).
    fn int_tensor_handle(tensor: IntTensor<Self>) -> Self::Handle;
    /// Convert a [bool tensor](Backend::BoolTensorPrimitive) to a [handle](BackendIr::Handle).
    fn bool_tensor_handle(tensor: BoolTensor<Self>) -> Self::Handle;
    /// Convert a [quantized tensor](Backend::QuantizedTensorPrimitive) to a [handle](BackendIr::Handle).
    fn quantized_tensor_handle(tensor: QuantizedTensor<Self>) -> Self::Handle;
    // /// Convert a [complex tensor](Backend::ComplexTensorPrimitive) to a [handle](BackendIr::Handle).
    // fn complex_tensor_handle(tensor: ComplexTensor<Self>) -> Self::Handle;
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
    // NOTE: will it pose a problem to not have the primitive as part of the core backend?
    // /// Complex tensor handle.
    // Complex(B::ComplexTensorPrimitive),
}

impl<B: Backend> HandleKind<B> {
    /// Returns the handle kind name.
    pub fn name(&self) -> &str {
        match self {
            HandleKind::Float(_) => "float",
            HandleKind::Int(_) => "int",
            HandleKind::Bool(_) => "bool",
            HandleKind::Quantized(_) => "quantized",
            //HandleKind::Complex(_) => "complex",
        }
    }
}
