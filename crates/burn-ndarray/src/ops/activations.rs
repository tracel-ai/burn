use crate::{
    element::{FloatNdArrayElement, IntNdArrayElement, QuantElement},
    execute_with_float_dtype,
    ops::NdArrayMathOps,
    NdArray,
};
use burn_tensor::{
    ops::{ActivationOps, FloatTensor},
    ElementConversion,
};

impl<E: FloatNdArrayElement, I: IntNdArrayElement, Q: QuantElement> ActivationOps<Self>
    for NdArray<E, I, Q>
{
    fn relu(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, |tensor| NdArrayMathOps::clamp_min(tensor, 0.elem()))
    }
}
