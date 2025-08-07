use crate::{BoolVisionOps, FloatVisionOps, IntVisionOps, QVisionOps, Transform2D, VisionBackend};
use burn_ndarray::{FloatNdArrayElement, IntNdArrayElement, NdArray, QuantElement};
use burn_tensor::ops::FloatTensor;

impl<E: FloatNdArrayElement, I: IntNdArrayElement, Q: QuantElement> BoolVisionOps
    for NdArray<E, I, Q>
{
}
impl<E: FloatNdArrayElement, I: IntNdArrayElement, Q: QuantElement> IntVisionOps
    for NdArray<E, I, Q>
{
}
impl<E: FloatNdArrayElement, I: IntNdArrayElement, Q: QuantElement> FloatVisionOps
    for NdArray<E, I, Q>
{
    /// Rotates an input tensor around a point
    ///
    /// `input` - A tensor to treat as an image
    fn float_resample(
        input: FloatTensor<Self>,
        transform: Transform2D<Self>,
        default: f32,
    ) -> FloatTensor<Self> {
        todo!();
    }
}
impl<E: FloatNdArrayElement, I: IntNdArrayElement, Q: QuantElement> QVisionOps
    for NdArray<E, I, Q>
{
}
impl<E: FloatNdArrayElement, I: IntNdArrayElement, Q: QuantElement> VisionBackend
    for NdArray<E, I, Q>
{
}
