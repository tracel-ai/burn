use crate::{element::FloatNdArrayElement, tensor::NdArrayTensor, NdArray};
use burn_tensor::{ops::ActivationOps, ElementConversion};

impl<E: FloatNdArrayElement> ActivationOps<Self> for NdArray<E> {
    fn relu<const D: usize>(tensor: NdArrayTensor<E, D>) -> NdArrayTensor<E, D> {
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
