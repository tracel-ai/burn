use crate::tensor::{backend::ndarray::NdArrayTensor, ops::*};
use ndarray::{LinalgScalar, ScalarOperand};

impl<P, const D: usize> TensorOpsDiv<P, D> for NdArrayTensor<P, D>
where
    P: Clone + LinalgScalar + Default + std::fmt::Debug + ScalarOperand,
{
    fn div(&self, other: &Self) -> Self {
        let array = self.array.clone() / other.array.clone();
        let array = array.to_owned().into_shared();
        let shape = self.shape.higher(&other.shape);

        Self { array, shape }
    }
    fn div_scalar(&self, other: &P) -> Self {
        let array = self.array.clone() / *other;
        let array = array.to_owned().into_shared();
        let shape = self.shape;

        Self { array, shape }
    }
}

impl<P, const D: usize> std::ops::Div<Self> for NdArrayTensor<P, D>
where
    P: Clone + LinalgScalar + Default + std::fmt::Debug + ScalarOperand,
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        TensorOpsDiv::div(&self, &rhs)
    }
}

impl<P, const D: usize> std::ops::Div<P> for NdArrayTensor<P, D>
where
    P: Clone + LinalgScalar + Default + std::fmt::Debug + ScalarOperand,
{
    type Output = Self;

    fn div(self, rhs: P) -> Self::Output {
        TensorOpsDiv::div_scalar(&self, &rhs)
    }
}
