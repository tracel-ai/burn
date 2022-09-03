use crate::{
    backend::ndarray::NdArrayBackend, backend::ndarray::NdArrayTensor, backend::Backend,
    ops::TensorOpsMask, NdArrayElement,
};

impl<E, const D: usize> TensorOpsMask<NdArrayBackend<E>, D> for NdArrayTensor<E, D>
where
    E: NdArrayElement,
{
    fn mask_fill(
        &self,
        mask: &<NdArrayBackend<E> as Backend>::BoolTensorPrimitive<D>,
        value: E,
    ) -> Self {
        let elem = E::default();
        let mask_mul = mask.array.mapv(|x| match x {
            true => E::zeros(&elem),
            false => E::ones(&elem),
        });
        let mask_add = mask.array.mapv(|x| match x {
            true => value,
            false => E::zeros(&elem),
        });
        let array = (self.array.clone() * mask_mul) + mask_add;

        Self {
            array,
            shape: self.shape,
        }
    }
}
