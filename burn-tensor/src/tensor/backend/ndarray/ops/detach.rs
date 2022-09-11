use crate::{
    tensor::{backend::ndarray::NdArrayTensor, ops::*},
    NdArrayElement,
};

impl<E, const D: usize> TensorOpsDetach<E, D> for NdArrayTensor<E, D>
where
    E: NdArrayElement,
{
    fn detach(self) -> Self {
        self
    }
}
