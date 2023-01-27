use crate::TchTensor;
use burn_tensor::ops::*;

impl<P, const D: usize> Zeros for TchTensor<P, D>
where
    P: tch::kind::Element,
{
    fn zeros(&self) -> TchTensor<P, D> {
        let tensor = self.tensor.zeros_like();
        let kind = self.kind.clone();

        Self { kind, tensor }
    }
}

impl<P, const D: usize> Ones for TchTensor<P, D>
where
    P: tch::kind::Element,
{
    fn ones(&self) -> TchTensor<P, D> {
        let tensor = self.tensor.ones_like();
        let kind = self.kind.clone();

        Self { kind, tensor }
    }
}
