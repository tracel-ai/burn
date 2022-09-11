use crate::{
    tensor::{backend::tch::TchTensor, ops::*},
    TchElement,
};

impl<E, const D: usize> TensorOpsDetach<E, D> for TchTensor<E, D>
where
    E: TchElement,
{
    fn detach(self) -> Self {
        self
    }
}
