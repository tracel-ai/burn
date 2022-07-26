use crate::{backend::ndarray::NdArrayTensor, TensorOpsAdd};
use ndarray::{Dim, Dimension, LinalgScalar, ScalarOperand};

impl<P, const D: usize> TensorOpsAdd<P, D> for NdArrayTensor<P, D>
where
    P: Clone + LinalgScalar + Default + std::fmt::Debug + ScalarOperand,
    Dim<[usize; D]>: Dimension,
{
    fn add(&self, other: &Self) -> Self {
        let array = self.array.clone() + other.array.clone();
        let array = array.into_shared();
        let shape = self.shape.clone();

        Self { array, shape }
    }
    fn add_scalar(&self, other: &P) -> Self {
        let array = self.array.clone() + other.clone();
        let shape = self.shape.clone();

        Self { array, shape }
    }
}

impl<P, const D: usize> std::ops::Add<Self> for NdArrayTensor<P, D>
where
    P: Clone + LinalgScalar + Default + std::fmt::Debug + ScalarOperand,
    Dim<[usize; D]>: Dimension,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        TensorOpsAdd::add(&self, &rhs)
    }
}

impl<P, const D: usize> std::ops::Add<P> for NdArrayTensor<P, D>
where
    P: Clone + LinalgScalar + Default + std::fmt::Debug + ScalarOperand,
    Dim<[usize; D]>: Dimension,
{
    type Output = Self;

    fn add(self, rhs: P) -> Self::Output {
        TensorOpsAdd::add_scalar(&self, &rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Data, TensorBase};

    #[test]
    fn should_support_add_ops() {
        let data_1 = Data::<f64, 2>::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let data_2 = Data::<f64, 2>::from([[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]);
        let data_expected = Data::from([[6.0, 8.0, 10.0], [12.0, 14.0, 16.0]]);
        let tensor_1 = NdArrayTensor::from(data_1);
        let tensor_2 = NdArrayTensor::from(data_2);

        let data_actual = (tensor_1 + tensor_2).into_data();

        assert_eq!(data_expected, data_actual);
    }
}
