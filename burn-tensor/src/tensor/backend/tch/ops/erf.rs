use crate::{
    tensor::{backend::tch::TchTensor, ops::*},
    TchElement,
};

impl<E, const D: usize> TensorOpsErf<E, D> for TchTensor<E, D>
where
    E: TchElement,
{
    fn erf(&self) -> Self {
        let tensor = self.tensor.erf();
        let kind = self.kind.clone();
        let shape = self.shape;

        Self {
            tensor,
            shape,
            kind,
        }
    }
}
