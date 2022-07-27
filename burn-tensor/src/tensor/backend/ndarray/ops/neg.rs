use crate::tensor::{backend::ndarray::NdArrayTensor, ops::*};
use ndarray::{LinalgScalar, ScalarOperand};

impl<P, const D: usize> TensorOpsNeg<P, D> for NdArrayTensor<P, D>
where
    P: Clone + LinalgScalar + Default + std::fmt::Debug + ScalarOperand,
{
    fn neg(&self) -> Self {
        let minus_one = P::zero() - P::one();
        let array = self.array.clone() * minus_one;
        let array = array.into_shared();
        let shape = self.shape.clone();

        Self { array, shape }
    }
}

impl<P, const D: usize> std::ops::Neg for NdArrayTensor<P, D>
where
    P: Clone + LinalgScalar + Default + std::fmt::Debug + ScalarOperand,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        TensorOpsNeg::neg(&self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Data;

    #[test]
    fn should_support_neg_ops() {
        let data = Data::<f64, 2>::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = NdArrayTensor::from_data(data);

        let data_actual = tensor.neg().into_data();

        let data_expected = Data::from([[-0.0, -1.0, -2.0], [-3.0, -4.0, -5.0]]);
        assert_eq!(data_expected, data_actual);
    }
}
