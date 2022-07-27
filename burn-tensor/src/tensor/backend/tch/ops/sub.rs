use crate::tensor::{backend::tch::TchTensor, ops::*, Data};
use std::ops::Sub;

impl<P: tch::kind::Element + Default + Copy + std::fmt::Debug, const D: usize> TensorOpsSub<P, D>
    for TchTensor<P, D>
{
    fn sub(&self, other: &Self) -> Self {
        let tensor = (&self.tensor).sub(&other.tensor);
        let kind = self.kind.clone();
        let shape = self.shape.clone();

        Self {
            tensor,
            shape,
            kind,
        }
    }
    fn sub_scalar(&self, other: &P) -> Self {
        let elems: [P; D] = [*other; D];
        let data = Data::from(elems);
        let other = TchTensor::from_data(data, self.tensor.device());
        let tensor = (&self.tensor).sub(&other.tensor);
        let kind = self.kind.clone();
        let shape = self.shape.clone();

        Self {
            tensor,
            shape,
            kind,
        }
    }
}

impl<P: tch::kind::Element + Default + std::fmt::Debug + Copy, const D: usize> std::ops::Sub<Self>
    for TchTensor<P, D>
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        TensorOpsSub::sub(&self, &rhs)
    }
}

impl<P: tch::kind::Element + Default + std::fmt::Debug + Copy, const D: usize> std::ops::Sub<P>
    for TchTensor<P, D>
{
    type Output = Self;

    fn sub(self, rhs: P) -> Self::Output {
        TensorOpsSub::sub_scalar(&self, &rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorBase;

    #[test]
    fn should_support_sub_ops() {
        let data_1 = Data::<f64, 2>::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let data_2 = Data::<f64, 2>::from([[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]);
        let data_expected = Data::from([[-6.0, -6.0, -6.0], [-6.0, -6.0, -6.0]]);
        let tensor_1 = TchTensor::from_data(data_1, tch::Device::Cpu);
        let tensor_2 = TchTensor::from_data(data_2, tch::Device::Cpu);

        let data_actual = (tensor_1 - tensor_2).into_data();

        assert_eq!(data_expected, data_actual);
    }
}
