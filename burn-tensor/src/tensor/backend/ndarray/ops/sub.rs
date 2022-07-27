use crate::tensor::{backend::ndarray::NdArrayTensor, ops::*};
use ndarray::{LinalgScalar, ScalarOperand};

impl<P, const D: usize> TensorOpsSub<P, D> for NdArrayTensor<P, D>
where
    P: Clone + LinalgScalar + Default + std::fmt::Debug + ScalarOperand,
{
    fn sub(&self, other: &Self) -> Self {
        let array = self.array.clone() - other.array.clone();
        let array = array.into_shared();
        let shape = self.shape.clone();

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Data, TensorBase};

    #[test]
    fn should_support_sub_ops() {
        let data_1 = Data::<f64, 2>::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let data_2 = Data::<f64, 2>::from([[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]);
        let data_expected = Data::from([[-6.0, -6.0, -6.0], [-6.0, -6.0, -6.0]]);
        let tensor_1 = NdArrayTensor::from_data(data_1);
        let tensor_2 = NdArrayTensor::from_data(data_2);

        let data_actual = (tensor_1 - tensor_2).into_data();

        assert_eq!(data_expected, data_actual);
    }
}
