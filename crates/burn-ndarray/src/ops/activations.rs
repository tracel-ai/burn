use crate::{
    NdArray,
    element::{FloatNdArrayElement, IntNdArrayElement, QuantElement},
    execute_with_float_dtype,
    tensor::NdArrayTensor,
};
use burn_tensor::{
    ElementConversion,
    ops::{ActivationOps, FloatTensor},
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
