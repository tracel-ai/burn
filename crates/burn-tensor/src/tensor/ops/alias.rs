use crate::backend::Backend;

// We provide some type aliases to improve the readability of using associated types without
// having to use the disambiguation syntax.

/// Device type used by the backend.
pub type Device<B> = <B as Backend>::Device;

/// Float element type used by backend.
pub type FloatElem<B> = <B as Backend>::FloatElem;
/// Integer element type used by backend.
pub type IntElem<B> = <B as Backend>::IntElem;

/// Float tensor primitive type used by the backend.
pub type FloatTensor<B> = <B as Backend>::FloatTensorPrimitive;
/// Integer tensor primitive type used by the backend.
pub type IntTensor<B> = <B as Backend>::IntTensorPrimitive;
/// Boolean tensor primitive type used by the backend.
pub type BoolTensor<B> = <B as Backend>::BoolTensorPrimitive;
/// Quantized tensor primitive type used by the backend.
pub type QuantizedTensor<B> = <B as Backend>::QuantizedTensorPrimitive;
