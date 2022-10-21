use crate::{
    tensor::{backend::tch::TchTensor, ops::*},
    TchElement,
};

impl<E, const D: usize> TensorOpsLog<E, D> for TchTensor<E, D>
where
    E: TchElement,
{
    fn log(&self) -> Self {
        let tensor = self.tensor.log();
        let kind = self.kind;
        let shape = self.shape;

        Self {
            tensor,
            shape,
            kind,
        }
    }
}
