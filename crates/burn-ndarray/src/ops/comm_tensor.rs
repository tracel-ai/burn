use burn_backend::ops::CommunicationTensorOps;

use crate::{
    FloatNdArrayElement, IntNdArrayElement, NdArray, NdArrayTensor, QuantElement, SharedArray,
};

impl<E: FloatNdArrayElement, I: IntNdArrayElement, Q: QuantElement> CommunicationTensorOps<Self>
    for NdArray<E, I, Q>
where
    NdArrayTensor: From<SharedArray<E>>,
    NdArrayTensor: From<SharedArray<I>>,
{
}
