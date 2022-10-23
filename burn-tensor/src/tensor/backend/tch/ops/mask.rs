use crate::{
    backend::tch::TchBackend, backend::tch::TchTensor, backend::Backend, ops::TensorOpsMask,
    TchElement,
};

impl<E: TchElement, const D: usize> TensorOpsMask<TchBackend<E>, D> for TchTensor<E, D> {
    fn mask_fill(
        &self,
        mask: &<TchBackend<E> as Backend>::BoolTensorPrimitive<D>,
        value: E,
    ) -> Self {
        let value: f64 = value.to_elem();
        let tensor = self.tensor.f_masked_fill(&mask.tensor, value).unwrap();

        Self {
            tensor,
            kind: self.kind,
            shape: self.shape,
        }
    }
}
