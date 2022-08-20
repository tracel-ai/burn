use crate::tensor::{backend::ndarray::NdArrayTensor, ops::*};
use ndarray::{LinalgScalar, ScalarOperand};

impl<P, const D: usize> TensorOpsSub<P, D> for NdArrayTensor<P, D>
where
    P: Clone + LinalgScalar + Default + std::fmt::Debug + ScalarOperand,
{
    fn sub(&self, other: &Self) -> Self {
        let array = self.array.clone() - other.array.clone();
        let array = array.into_shared();
        let shape = self.shape.higher(&other.shape);

        Self { array, shape }
    }
    fn sub_scalar(&self, other: &P) -> Self {
        let array = self.array.clone() - other.clone();
        let shape = self.shape.clone();

        Self { array, shape }
    }
}

impl<P, const D: usize> std::ops::Sub<Self> for NdArrayTensor<P, D>
where
    P: Clone + LinalgScalar + Default + std::fmt::Debug + ScalarOperand,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        TensorOpsSub::sub(&self, &rhs)
    }
}

impl<P, const D: usize> std::ops::Sub<P> for NdArrayTensor<P, D>
where
    P: Clone + LinalgScalar + Default + std::fmt::Debug + ScalarOperand,
{
    type Output = Self;

    fn sub(self, rhs: P) -> Self::Output {
        TensorOpsSub::sub_scalar(&self, &rhs)
    }
}
