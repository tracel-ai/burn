use crate::{backend::ndarray::NdArrayTensor, TensorOpsMul};
use ndarray::{Dim, Dimension, LinalgScalar, ScalarOperand};

impl<P, const D: usize> TensorOpsMul<P, D> for NdArrayTensor<P, D>
where
    P: Clone + LinalgScalar + Default + std::fmt::Debug + ScalarOperand,
    Dim<[usize; D]>: Dimension,
{
    fn mul(&self, other: &Self) -> Self {
        let array = self.array.clone() * other.array.clone();
        let array = array.to_owned().into_shared();
        let shape = self.shape.clone();

        Self { array, shape }
    }
    fn mul_scalar(&self, other: &P) -> Self {
        let array = self.array.clone() * other.clone();
        let array = array.to_owned().into_shared();
        let shape = self.shape.clone();

        Self { array, shape }
    }
}

impl<P, const D: usize> std::ops::Mul<Self> for NdArrayTensor<P, D>
where
    P: Clone + LinalgScalar + Default + std::fmt::Debug + ScalarOperand,
    Dim<[usize; D]>: Dimension,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        TensorOpsMul::mul(&self, &rhs)
    }
}

impl<P, const D: usize> std::ops::Mul<P> for NdArrayTensor<P, D>
where
    P: Clone + LinalgScalar + Default + std::fmt::Debug + ScalarOperand,
    Dim<[usize; D]>: Dimension,
{
    type Output = Self;

    fn mul(self, rhs: P) -> Self::Output {
        TensorOpsMul::mul_scalar(&self, &rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Data, TensorBase};

    #[test]
    fn should_support_mul_ops() {
        let data_1 = Data::<f64, 2>::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let data_2 = Data::<f64, 2>::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor_1 = NdArrayTensor::from_data(data_1);
        let tensor_2 = NdArrayTensor::from_data(data_2);

        let data_actual = (tensor_1 * tensor_2).into_data();

        let data_expected = Data::from([[0.0, 1.0, 4.0], [9.0, 16.0, 25.0]]);
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_mul_scalar_ops() {
        let data = Data::<f64, 2>::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let scalar = 2.0;
        let tensor = NdArrayTensor::from_data(data);

        let output = tensor * scalar;
        let data_actual = output.into_data();

        let data_expected = Data::from([[0.0, 2.0, 4.0], [6.0, 8.0, 10.0]]);
        assert_eq!(data_expected, data_actual);
    }
}
