use crate::{
    tensor::{backend::tch::TchTensor, ops::*, Shape},
    Element,
};
use std::ops::Div;

impl<P: Element + tch::kind::Element, const D: usize> TensorOpsDiv<P, D> for TchTensor<P, D> {
    fn div(&self, other: &Self) -> Self {
        let tensor = (&self.tensor).div(&other.tensor);
        let shape = self.shape.higher(&other.shape);
        let kind = self.kind;

        Self {
            tensor,
            shape,
            kind,
        }
    }
    fn div_scalar(&self, other: &P) -> Self {
        let other: f64 = (other.clone()).to_elem();
        let tensor = (&self.tensor).div(other);
        let shape = Shape::from(tensor.size());
        let kind = self.kind;

        Self {
            tensor,
            shape,
            kind,
        }
    }
}

impl<P: Element + tch::kind::Element, const D: usize> std::ops::Div<P> for TchTensor<P, D> {
    type Output = TchTensor<P, D>;

    fn div(self, rhs: P) -> Self::Output {
        TensorOpsDiv::div_scalar(&self, &rhs)
    }
}

impl<P: Element + tch::kind::Element, const D: usize> std::ops::Div<TchTensor<P, D>>
    for TchTensor<P, D>
{
    type Output = TchTensor<P, D>;

    fn div(self, rhs: Self) -> Self::Output {
        TensorOpsDiv::div(&self, &rhs)
    }
}
