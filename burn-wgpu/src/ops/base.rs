use burn_tensor::backend::Backend;

pub type FloatElem<B> = <B as Backend>::FloatElem;
pub type IntElem<B> = <B as Backend>::IntElem;
pub type Device<B> = <B as Backend>::Device;

pub type FloatTensor<B, const D: usize> = <B as Backend>::TensorPrimitive<D>;
pub type IntTensor<B, const D: usize> = <B as Backend>::IntTensorPrimitive<D>;
pub type BoolTensor<B, const D: usize> = <B as Backend>::BoolTensorPrimitive<D>;
