use burn_backend::TypedDevice;

use crate::{FloatNdArrayElement, IntNdArrayElement, NdArray, NdArrayTensor, QuantElement, SharedArray};

impl<E: FloatNdArrayElement, I: IntNdArrayElement, Q: QuantElement> TypedDevice<Self>
    for NdArray<E, I, Q>
    where
    NdArrayTensor: From<SharedArray<E>>,
    NdArrayTensor: From<SharedArray<I>> {
        fn complex_device(_tensor: &burn_backend::ComplexTensor<Self>) -> <Self as burn_backend::BackendTypes>::Device {
            panic!("NdArray backend does not yet support interleaved complex tensors")
    }
    }