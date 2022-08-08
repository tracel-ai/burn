use crate::{
    back::Backend, backend::ndarray::NdArrayBackend, backend::ndarray::NdArrayTensor,
    ops::TensorOpsMask, Element,
};
use rand::distributions::Standard;

impl<E: Element, const D: usize> TensorOpsMask<NdArrayBackend<E>, D> for NdArrayTensor<E, D>
where
    E: Element,
    Standard: rand::distributions::Distribution<E>,
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
