use burn_backend::distributed::DistributedBackend;

use crate::{
    FloatNdArrayElement, IntNdArrayElement, NdArray, NdArrayTensor, QuantElement, SharedArray,
};

impl<E: FloatNdArrayElement, I: IntNdArrayElement, Q: QuantElement> DistributedBackend
    for NdArray<E, I, Q>
where
    NdArrayTensor: From<SharedArray<E>>,
    NdArrayTensor: From<SharedArray<I>>,
{
}
