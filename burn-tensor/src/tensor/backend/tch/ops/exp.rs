use crate::{
    tensor::{backend::tch::TchTensor, ops::*},
    TchElement,
};

impl<E, const D: usize> TensorOpsExp<E, D> for TchTensor<E, D>
where
    E: TchElement,
{
    fn exp(&self) -> Self {
        let tensor = self.tensor.exp();
        let kind = self.kind.clone();
        let shape = self.shape;

        Self {
            tensor,
            shape,
            kind,
        }
    }
}
