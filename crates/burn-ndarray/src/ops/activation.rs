use crate::{
    NdArray, NdArrayTensor, SharedArray,
    element::{FloatNdArrayElement, IntNdArrayElement, QuantElement},
    execute_with_numeric_dtype,
    ops::NdArrayMathOps,
};
use burn_backend::{ElementConversion, TensorMetadata, ops::ActivationOps, tensor::FloatTensor};

impl<E: FloatNdArrayElement, I: IntNdArrayElement, Q: QuantElement> ActivationOps<Self>
    for NdArray<E, I, Q>
where
    NdArrayTensor: From<SharedArray<E>>,
    NdArrayTensor: From<SharedArray<I>>,
{
    fn relu(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_numeric_dtype!(tensor, |array| NdArrayMathOps::clamp_min(array, 0.elem()))
    }
}
