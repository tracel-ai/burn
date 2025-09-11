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

#[cfg(feature = "candle")]
mod candle {
    use crate::{BoolVisionOps, FloatVisionOps, IntVisionOps, QVisionOps, VisionBackend};
    use burn_candle::{Candle, FloatCandleElement, IntCandleElement};

    impl<F: FloatCandleElement, I: IntCandleElement> BoolVisionOps for Candle<F, I> {}
    impl<F: FloatCandleElement, I: IntCandleElement> IntVisionOps for Candle<F, I> {}
    impl<F: FloatCandleElement, I: IntCandleElement> FloatVisionOps for Candle<F, I> {}
    impl<F: FloatCandleElement, I: IntCandleElement> QVisionOps for Candle<F, I> {}
    impl<F: FloatCandleElement, I: IntCandleElement> VisionBackend for Candle<F, I> {}
}

#[cfg(feature = "tch")]
mod tch {
    use crate::{BoolVisionOps, FloatVisionOps, IntVisionOps, QVisionOps, VisionBackend};
    use burn_tch::{LibTorch, TchElement};

    impl<E: TchElement, Q: burn_tch::QuantElement> BoolVisionOps for LibTorch<E, Q> {}
    impl<E: TchElement, Q: burn_tch::QuantElement> IntVisionOps for LibTorch<E, Q> {}
    impl<E: TchElement, Q: burn_tch::QuantElement> FloatVisionOps for LibTorch<E, Q> {}
    impl<E: TchElement, Q: burn_tch::QuantElement> QVisionOps for LibTorch<E, Q> {}
    impl<E: TchElement, Q: burn_tch::QuantElement> VisionBackend for LibTorch<E, Q> {}
}
