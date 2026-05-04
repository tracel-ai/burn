use crate::backend::BackendTypes;

// We provide some type aliases to improve the readability of using associated types without
// having to use the disambiguation syntax.

/// Device type used by the backend.
pub type Device<B> = <B as BackendTypes>::Device;

/// Float element type used by backend.
pub type FloatElem<B> = <B as BackendTypes>::FloatElem;
/// Integer element type used by backend.
pub type IntElem<B> = <B as BackendTypes>::IntElem;
/// Boolean element type used by backend.
pub type BoolElem<B> = <B as BackendTypes>::BoolElem;

/// Float tensor primitive type used by the backend.
pub type FloatTensor<B> = <B as BackendTypes>::FloatTensorPrimitive;
/// Integer tensor primitive type used by the backend.
pub type IntTensor<B> = <B as BackendTypes>::IntTensorPrimitive;
/// Boolean tensor primitive type used by the backend.
pub type BoolTensor<B> = <B as BackendTypes>::BoolTensorPrimitive;
/// Quantized tensor primitive type used by the backend.
pub type QuantizedTensor<B> = <B as BackendTypes>::QuantizedTensorPrimitive;
