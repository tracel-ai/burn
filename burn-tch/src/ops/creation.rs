use crate::{to_tensor, TchTensor};
use burn_tensor::ops::*;

impl<E, const D: usize> Zeros for TchTensor<E, D>
where
    E: tch::kind::Element + Default,
{
    fn zeros(&self) -> TchTensor<E, D> {
        to_tensor(self.tensor.zeros_like())
    }
}

impl<E, const D: usize> Ones for TchTensor<E, D>
where
    E: tch::kind::Element + Default,
{
    fn ones(&self) -> TchTensor<E, D> {
        to_tensor(self.tensor.ones_like())
    }
}
