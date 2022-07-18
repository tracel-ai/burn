use crate::{backend::tch::TchTensor, TensorOpsMul};
use std::ops::Mul;

impl<P: tch::kind::Element + Into<f64>, const D: usize> TensorOpsMul<P, D> for TchTensor<P, D> {
    fn mul(&self, other: &Self) -> Self {
        let tensor = (&self.tensor) * &other.tensor;
        let shape = self.shape.clone();
        let kind = self.kind.clone();

        Self {
            tensor,
            shape,
            kind,
        }
    }
    fn mul_scalar(&self, other: &P) -> Self {
        let other: f64 = (other.clone()).into();
        let tensor = (&self.tensor).mul(other);
        let shape = self.shape.clone();
        let kind = self.kind.clone();

        Self {
            tensor,
            shape,
            kind,
        }
    }
}

impl<P: tch::kind::Element + Into<f64>, const D: usize> std::ops::Mul<P> for TchTensor<P, D> {
    type Output = TchTensor<P, D>;

    fn mul(self, rhs: P) -> Self::Output {
        TensorOpsMul::mul_scalar(&self, &rhs)
    }
}

impl<P: tch::kind::Element + Into<f64>, const D: usize> std::ops::Mul<TchTensor<P, D>>
    for TchTensor<P, D>
{
    type Output = TchTensor<P, D>;

    fn mul(self, rhs: Self) -> Self::Output {
        TensorOpsMul::mul(&self, &rhs)
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
        let tensor_1 = TchTensor::from_data(data_1, tch::Device::Cpu);
        let tensor_2 = TchTensor::from_data(data_2, tch::Device::Cpu);

        let output = tensor_1 * tensor_2;
        let data_actual = output.into_data();

        let data_expected = Data::from([[0.0, 1.0, 4.0], [9.0, 16.0, 25.0]]);
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_mul_scalar_ops() {
        let data = Data::<f64, 2>::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let scalar = 2.0;
        let tensor = TchTensor::from_data(data, tch::Device::Cpu);

        let output = tensor * scalar;
        let data_actual = output.into_data();

        let data_expected = Data::from([[0.0, 2.0, 4.0], [6.0, 8.0, 10.0]]);
        assert_eq!(data_expected, data_actual);
    }
}
