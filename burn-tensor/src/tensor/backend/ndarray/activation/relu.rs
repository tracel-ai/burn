use crate::tensor::ops::{TensorOpsMapComparison, TensorOpsMask};
use crate::{
    activation::ReLU,
    backend::ndarray::{NdArrayBackend, NdArrayTensor},
    NdArrayElement,
};
use rand::distributions::Standard;

impl<E, const D: usize> ReLU<E, D> for NdArrayTensor<E, D>
where
    E: NdArrayElement,
    Standard: rand::distributions::Distribution<E>,
{
    fn relu(&self) -> Self {
        let zero = E::zeros(&E::default());
        let mask = TensorOpsMapComparison::<NdArrayBackend<E>, D>::lower_equal_scalar(self, &zero);
        self.mask_fill(&mask, zero)
    }
}
