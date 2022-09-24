use crate::{
    tensor::{backend::ndarray::NdArrayTensor, ops::*},
    ElementConversion, NdArrayElement,
};

impl<E, const D: usize> TensorOpsErf<E, D> for NdArrayTensor<E, D>
where
    E: NdArrayElement,
{
    fn erf(&self) -> Self {
        let array = self
            .array
            .mapv(|a| libm::erf(a.to_f64().unwrap()).to_elem())
            .into_shared();
        let shape = self.shape;

        Self { array, shape }
    }
}
