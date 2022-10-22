use crate::{
    tensor::{backend::tch::TchTensor, ops::*},
    TchElement,
};
use std::ops::Add;

impl<P: TchElement, const D: usize> TensorOpsAdd<P, D> for TchTensor<P, D> {
    fn add(&self, other: &Self) -> Self {
        let tensor = (&self.tensor).add(&other.tensor);
        let kind = self.kind;
        let shape = self.shape.higher(&other.shape);

        Self {
            tensor,
            shape,
            kind,
        }
    }
    fn add_scalar(&self, other: &P) -> Self {
        let other: f64 = (other.clone()).to_elem();
        let tensor = (&self.tensor).add(other).to_kind(self.kind.kind());
        let kind = self.kind;
        let shape = self.shape;

        Self {
            tensor,
            shape,
            kind,
        }
    }
}

impl<P: TchElement, const D: usize> std::ops::Add<Self> for TchTensor<P, D> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        TensorOpsAdd::add(&self, &rhs)
    }
}

impl<P: TchElement, const D: usize> std::ops::Add<P> for TchTensor<P, D> {
    type Output = Self;

    fn add(self, rhs: P) -> Self::Output {
        TensorOpsAdd::add_scalar(&self, &rhs)
    }
}
