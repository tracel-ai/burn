use crate::{
    element::{FloatNdArrayElement, IntNdArrayElement, QuantElement},
    execute_with_float_dtype,
    tensor::NdArrayTensor,
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
        execute_with_float_dtype!(tensor, |tensor: NdArrayTensor<_>| {
            let zero = 0.elem();
            let array = tensor
                .array
                .mapv_into(|elem| match elem < zero {
                    true => zero,
                    false => elem,
                })
                .into_shared();

            NdArrayTensor::new(array)
        })
    }
}
