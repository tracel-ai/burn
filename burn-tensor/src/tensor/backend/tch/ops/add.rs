use crate::tensor::{backend::tch::TchTensor, ops::*, Data};
use std::ops::Add;

impl<P: tch::kind::Element + Default + Copy + std::fmt::Debug, const D: usize> TensorOpsAdd<P, D>
    for TchTensor<P, D>
{
    fn add(&self, other: &Self) -> Self {
        let tensor = (&self.tensor).add(&other.tensor);
        let kind = self.kind.clone();
        let shape = self.shape.clone();

        Self {
            tensor,
            shape,
            kind,
        }
    }
    fn add_scalar(&self, other: &P) -> Self {
        let elems: [P; D] = [*other; D];
        let data = Data::from(elems);
        let other = TchTensor::from_data(data, self.tensor.device());
        let tensor = (&self.tensor).add(&other.tensor);
        let kind = self.kind.clone();
        let shape = self.shape.clone();

        Self {
            tensor,
            shape,
            kind,
        }
    }
}

impl<P: tch::kind::Element + Default + std::fmt::Debug + Copy, const D: usize> std::ops::Add<Self>
    for TchTensor<P, D>
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        TensorOpsAdd::add(&self, &rhs)
    }
}

impl<P: tch::kind::Element + Default + std::fmt::Debug + Copy, const D: usize> std::ops::Add<P>
    for TchTensor<P, D>
{
    type Output = Self;

    fn add(self, rhs: P) -> Self::Output {
        TensorOpsAdd::add_scalar(&self, &rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_support_add_ops() {
        let data_1 = Data::<f64, 2>::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let data_2 = Data::<f64, 2>::from([[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]);
        let data_expected = Data::from([[6.0, 8.0, 10.0], [12.0, 14.0, 16.0]]);
        let tensor_1 = TchTensor::from_data(data_1, tch::Device::Cpu);
        let tensor_2 = TchTensor::from_data(data_2, tch::Device::Cpu);

        let data_actual = (tensor_1 + tensor_2).into_data();

        assert_eq!(data_expected, data_actual);
    }
}
