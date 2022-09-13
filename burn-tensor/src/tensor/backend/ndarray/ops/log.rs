use crate::{
    tensor::{backend::ndarray::NdArrayTensor, ops::*},
    NdArrayElement,
};

impl<E, const D: usize> TensorOpsLog<E, D> for NdArrayTensor<E, D>
where
    E: NdArrayElement,
{
    fn log(&self) -> Self {
        let array = self.array.mapv(|a| a.log_elem()).into_shared();
        let shape = self.shape;

        Self { array, shape }
    }
}
