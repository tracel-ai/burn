use crate::{
    FloatNdArrayElement, NdArray, NdArrayTensor, SharedArray,
    element::{IntNdArrayElement, QuantElement},
};
use burn_tensor::ops::TransactionOps;

impl<E: FloatNdArrayElement, I: IntNdArrayElement, Q: QuantElement> TransactionOps<Self>
    for NdArray<E, I, Q>
where
    NdArrayTensor: From<SharedArray<E>>,
    NdArrayTensor: From<SharedArray<I>>,
{
}
