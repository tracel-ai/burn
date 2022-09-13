use crate::tensor::{backend::ndarray::NdArrayTensor, ops::*};
use ndarray::{LinalgScalar, ScalarOperand};

impl<P, const D: usize> TensorOpsMul<P, D> for NdArrayTensor<P, D>
where
    P: Clone + LinalgScalar + Default + std::fmt::Debug + ScalarOperand,
{
    fn mul(&self, other: &Self) -> Self {
        let array = self.array.clone() * other.array.clone();
        let array = array.to_owned().into_shared();
        let shape = self.shape.higher(&other.shape);

        Self { array, shape }
    }
    fn mul_scalar(&self, other: &P) -> Self {
        let array = self.array.clone() * *other;
        let array = array.to_owned().into_shared();
        let shape = self.shape;

        Self { array, shape }
    }
}

impl<P, const D: usize> std::ops::Mul<Self> for NdArrayTensor<P, D>
where
    P: Clone + LinalgScalar + Default + std::fmt::Debug + ScalarOperand,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        TensorOpsMul::mul(&self, &rhs)
    }
}

impl<P, const D: usize> std::ops::Mul<P> for NdArrayTensor<P, D>
where
    P: Clone + LinalgScalar + Default + std::fmt::Debug + ScalarOperand,
{
    type Output = Self;

    fn mul(self, rhs: P) -> Self::Output {
        TensorOpsMul::mul_scalar(&self, &rhs)
    }
}
