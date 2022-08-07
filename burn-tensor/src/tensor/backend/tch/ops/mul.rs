use crate::tensor::{backend::tch::TchTensor, ops::*};
use std::ops::Mul;

impl<P: tch::kind::Element + Into<f64>, const D: usize> TensorOpsMul<P, D> for TchTensor<P, D> {
    fn mul(&self, other: &Self) -> Self {
        let tensor = (&self.tensor) * &other.tensor;
        let shape = self.shape.clone();
        let kind = self.kind.clone();

        Self {
            tensor,
            shape,
            kind,
        }
    }
    fn mul_scalar(&self, other: &P) -> Self {
        let other: f64 = (other.clone()).into();
        let tensor = (&self.tensor).mul(other);
        let shape = self.shape.clone();
        let kind = self.kind.clone();

        Self {
            tensor,
            shape,
            kind,
        }
    }
}

impl<P: tch::kind::Element + Into<f64>, const D: usize> std::ops::Mul<P> for TchTensor<P, D> {
    type Output = TchTensor<P, D>;

    fn mul(self, rhs: P) -> Self::Output {
        TensorOpsMul::mul_scalar(&self, &rhs)
    }
}

impl<P: tch::kind::Element + Into<f64>, const D: usize> std::ops::Mul<TchTensor<P, D>>
    for TchTensor<P, D>
{
    type Output = TchTensor<P, D>;

    fn mul(self, rhs: Self) -> Self::Output {
        TensorOpsMul::mul(&self, &rhs)
    }
}
