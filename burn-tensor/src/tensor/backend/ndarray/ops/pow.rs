use crate::{
    tensor::{backend::ndarray::NdArrayTensor, ops::*},
    NdArrayElement,
};

impl<E, const D: usize> TensorOpsPow<E, D> for NdArrayTensor<E, D>
where
    E: NdArrayElement,
{
    fn powf(&self, value: f32) -> Self {
        let array = self.array.mapv(|a| a.pow_elem(value)).into_shared();
        let shape = self.shape.clone();

        Self { array, shape }
    }
}
