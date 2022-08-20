use crate::tensor::{backend::tch::TchTensor, ops::*, Data};
use std::ops::Add;

impl<P: tch::kind::Element + Default + Copy + std::fmt::Debug, const D: usize> TensorOpsAdd<P, D>
    for TchTensor<P, D>
{
    fn add(&self, other: &Self) -> Self {
        let tensor = (&self.tensor).add(&other.tensor);
        let kind = self.kind.clone();
        let shape = self.shape.higher(&other.shape);

        Self {
            tensor,
            shape,
            kind,
        }
    }
    fn add_scalar(&self, other: &P) -> Self {
        let elems: [P; D] = [*other; D];
        let data = Data::from(elems);
        let other = TchTensor::from_data(data, self.tensor.device());
        let tensor = (&self.tensor).add(&other.tensor);
        let kind = self.kind.clone();
        let shape = self.shape.clone();

        Self {
            tensor,
            shape,
            kind,
        }
    }
}

impl<P: tch::kind::Element + Default + std::fmt::Debug + Copy, const D: usize> std::ops::Add<Self>
    for TchTensor<P, D>
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        TensorOpsAdd::add(&self, &rhs)
    }
}

impl<P: tch::kind::Element + Default + std::fmt::Debug + Copy, const D: usize> std::ops::Add<P>
    for TchTensor<P, D>
{
    type Output = Self;

    fn add(self, rhs: P) -> Self::Output {
        TensorOpsAdd::add_scalar(&self, &rhs)
    }
}
