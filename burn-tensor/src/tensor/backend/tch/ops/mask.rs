use crate::{
    back::Backend, backend::tch::TchBackend, backend::tch::TchTensor, ops::TensorOpsMask,
    TchElement,
};
use rand::distributions::Standard;

impl<E: TchElement, const D: usize> TensorOpsMask<TchBackend<E>, D> for TchTensor<E, D>
where
    Standard: rand::distributions::Distribution<E>,
{
    fn mask_fill(
        &self,
        mask: &<TchBackend<E> as Backend>::BoolTensorPrimitive<D>,
        value: E,
    ) -> Self {
        let value: f64 = value.into();
        let tensor = self.tensor.f_masked_fill(&mask.tensor, value).unwrap();

        Self {
            tensor,
            kind: self.kind.clone(),
            shape: self.shape,
        }
    }
}
