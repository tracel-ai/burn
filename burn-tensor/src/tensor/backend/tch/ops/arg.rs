use crate::backend::tch::TchBackend;
use crate::tensor::TchElement;
use crate::tensor::{
    backend::tch::{TchKind, TchTensor},
    ops::*,
};

impl<E, const D: usize> TensorOpsArg<TchBackend<E>, D> for TchTensor<E, D>
where
    E: TchElement,
{
    fn argmax(&self, dim: usize) -> TchTensor<i64, D> {
        let tensor = self.tensor.argmax(dim as i64, true);
        let shape = self.shape.clone();

        TchTensor {
            tensor,
            kind: TchKind::<i64>::new(),
            shape,
        }
    }

    fn argmin(&self, dim: usize) -> TchTensor<i64, D> {
        let tensor = self.tensor.argmin(dim as i64, true);
        let shape = self.shape.clone();

        TchTensor {
            tensor,
            kind: TchKind::<i64>::new(),
            shape,
        }
    }
}
