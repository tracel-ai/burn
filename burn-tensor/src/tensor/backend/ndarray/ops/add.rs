use crate::tensor::{backend::ndarray::NdArrayTensor, ops::*};
use ndarray::{LinalgScalar, ScalarOperand};

impl<P, const D: usize> TensorOpsAdd<P, D> for NdArrayTensor<P, D>
where
    P: Clone + LinalgScalar + Default + std::fmt::Debug + ScalarOperand,
{
    fn add(&self, other: &Self) -> Self {
        let array = self.array.clone() + other.array.clone();
        let array = array.into_shared();
        let shape = self.shape.higher(other.shape());

        Self { array, shape }
    }
    fn add_scalar(&self, other: &P) -> Self {
        let array = self.array.clone() + *other;
        let shape = self.shape;

        Self { array, shape }
    }
}

impl<P, const D: usize> std::ops::Add<Self> for NdArrayTensor<P, D>
where
    P: Clone + LinalgScalar + Default + std::fmt::Debug + ScalarOperand,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        TensorOpsAdd::add(&self, &rhs)
    }
}

impl<P, const D: usize> std::ops::Add<P> for NdArrayTensor<P, D>
where
    P: Clone + LinalgScalar + Default + std::fmt::Debug + ScalarOperand,
{
    type Output = Self;

    fn add(self, rhs: P) -> Self::Output {
        TensorOpsAdd::add_scalar(&self, &rhs)
    }
}
