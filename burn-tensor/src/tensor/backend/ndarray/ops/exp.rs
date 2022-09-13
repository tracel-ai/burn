use crate::{
    tensor::{backend::ndarray::NdArrayTensor, ops::*},
    NdArrayElement,
};

impl<E, const D: usize> TensorOpsExp<E, D> for NdArrayTensor<E, D>
where
    E: NdArrayElement,
{
    fn exp(&self) -> Self {
        let array = self.array.mapv(|a| a.exp_elem()).into_shared();
        let shape = self.shape;

        Self { array, shape }
    }
}
