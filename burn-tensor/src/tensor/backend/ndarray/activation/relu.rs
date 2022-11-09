use crate::ops::TensorOps;
use crate::{
    backend::ndarray::{NdArrayBackend, NdArrayTensor},
    ops::activation::ReLU,
    NdArrayElement,
};

impl<E, const D: usize> ReLU<E, D> for NdArrayTensor<E, D>
where
    E: NdArrayElement,
{
    fn relu(&self) -> Self {
        let zero = E::zeros(&E::default());
        let mask = NdArrayBackend::lower_equal_scalar(self, &zero);

        NdArrayBackend::mask_fill(self, &mask, zero)
    }
}
