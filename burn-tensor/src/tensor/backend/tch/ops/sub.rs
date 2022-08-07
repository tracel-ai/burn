use crate::tensor::{backend::tch::TchTensor, ops::*, Data};
use std::ops::Sub;

impl<P: tch::kind::Element + Default + Copy + std::fmt::Debug, const D: usize> TensorOpsSub<P, D>
    for TchTensor<P, D>
{
    fn sub(&self, other: &Self) -> Self {
        let tensor = (&self.tensor).sub(&other.tensor);
        let kind = self.kind.clone();
        let shape = self.shape.clone();

        Self {
            tensor,
            shape,
            kind,
        }
    }
    fn sub_scalar(&self, other: &P) -> Self {
        let elems: [P; D] = [*other; D];
        let data = Data::from(elems);
        let other = TchTensor::from_data(data, self.tensor.device());
        let tensor = (&self.tensor).sub(&other.tensor);
        let kind = self.kind.clone();
        let shape = self.shape.clone();

        Self {
            tensor,
            shape,
            kind,
        }
    }
}

impl<P: tch::kind::Element + Default + std::fmt::Debug + Copy, const D: usize> std::ops::Sub<Self>
    for TchTensor<P, D>
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        TensorOpsSub::sub(&self, &rhs)
    }
}

impl<P: tch::kind::Element + Default + std::fmt::Debug + Copy, const D: usize> std::ops::Sub<P>
    for TchTensor<P, D>
{
    type Output = Self;

    fn sub(self, rhs: P) -> Self::Output {
        TensorOpsSub::sub_scalar(&self, &rhs)
    }
}
