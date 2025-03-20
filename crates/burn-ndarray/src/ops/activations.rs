use crate::{
    NdArray,
    element::{FloatNdArrayElement, IntNdArrayElement, QuantElement},
    execute_with_float_dtype,
    ops::NdArrayMathOps,
};
use burn_tensor::{
    ElementConversion,
    ops::{ActivationOps, FloatTensor},
};

impl<E: FloatNdArrayElement, I: IntNdArrayElement, Q: QuantElement> ActivationOps<Self>
    for NdArray<E, I, Q>
{
    fn relu(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, |tensor| NdArrayMathOps::clamp_min(tensor, 0.elem()))
    }
}
