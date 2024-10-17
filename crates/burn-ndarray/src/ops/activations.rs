use crate::{
    element::{FloatNdArrayElement, IntNdArrayElement, QuantElement},
    tensor::NdArrayTensor,
    NdArray,
};
use burn_tensor::{ops::ActivationOps, ElementConversion};

impl<E: FloatNdArrayElement, I: IntNdArrayElement, Q: QuantElement> ActivationOps<Self>
    for NdArray<E, I, Q>
{
    fn relu(tensor: NdArrayTensor<E>) -> NdArrayTensor<E> {
        let zero = 0.elem();
        let array = tensor
            .array
            .mapv_into(|elem| match elem < zero {
                true => zero,
                false => elem,
            })
            .into_shared();

        NdArrayTensor::new(array)
    }
}
