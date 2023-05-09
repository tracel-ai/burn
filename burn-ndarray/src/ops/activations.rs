use crate::{element::FloatNdArrayElement, tensor::NdArrayTensor, NdArrayBackend};
use burn_tensor::{ops::ActivationOps, ElementConversion};

impl<E: FloatNdArrayElement> ActivationOps<NdArrayBackend<E>> for NdArrayBackend<E> {
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
