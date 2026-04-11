#[cfg(feature = "ndarray")]
mod ndarray {
    use crate::{BoolVisionOps, FloatVisionOps, IntVisionOps, QVisionOps, VisionBackend};
    use burn_ndarray::{
        FloatNdArrayElement, IntNdArrayElement, NdArray, NdArrayTensor, QuantElement, SharedArray,
    };

    impl<E: FloatNdArrayElement, I: IntNdArrayElement, Q: QuantElement> BoolVisionOps
        for NdArray<E, I, Q>
    where
        NdArrayTensor: From<SharedArray<E>>,
        NdArrayTensor: From<SharedArray<I>>,
    {
    }
    impl<E: FloatNdArrayElement, I: IntNdArrayElement, Q: QuantElement> IntVisionOps
        for NdArray<E, I, Q>
    where
        NdArrayTensor: From<SharedArray<E>>,
        NdArrayTensor: From<SharedArray<I>>,
    {
    }
    impl<E: FloatNdArrayElement, I: IntNdArrayElement, Q: QuantElement> FloatVisionOps
        for NdArray<E, I, Q>
    where
        NdArrayTensor: From<SharedArray<E>>,
        NdArrayTensor: From<SharedArray<I>>,
    {
    }
    impl<E: FloatNdArrayElement, I: IntNdArrayElement, Q: QuantElement> QVisionOps for NdArray<E, I, Q>
    where
        NdArrayTensor: From<SharedArray<E>>,
        NdArrayTensor: From<SharedArray<I>>,
    {
    }
    impl<E: FloatNdArrayElement, I: IntNdArrayElement, Q: QuantElement> VisionBackend
        for NdArray<E, I, Q>
    where
        NdArrayTensor: From<SharedArray<E>>,
        NdArrayTensor: From<SharedArray<I>>,
    {
    }
}

#[cfg(feature = "tch")]
mod tch {
    use crate::{BoolVisionOps, FloatVisionOps, IntVisionOps, QVisionOps, VisionBackend};
    use burn_tch::{LibTorch, TchElement};

    impl<E: TchElement> BoolVisionOps for LibTorch<E> {}
    impl<E: TchElement> IntVisionOps for LibTorch<E> {}
    impl<E: TchElement> FloatVisionOps for LibTorch<E> {}
    impl<E: TchElement> QVisionOps for LibTorch<E> {}
    impl<E: TchElement> VisionBackend for LibTorch<E> {}
}
