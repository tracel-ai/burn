use crate::tensor::{backend::tch::TchTensor, ops::*};

impl<P: tch::kind::Element + Default + Copy + std::fmt::Debug, const D: usize> TensorOpsNeg<P, D>
    for TchTensor<P, D>
{
    fn neg(&self) -> Self {
        let tensor = -(&self.tensor);
        let kind = self.kind;
        let shape = self.shape;

        Self {
            tensor,
            shape,
            kind,
        }
    }
}

impl<P: tch::kind::Element + Default + std::fmt::Debug + Copy, const D: usize> std::ops::Neg
    for TchTensor<P, D>
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
        let tensor = TchTensor::from_data(data, tch::Device::Cpu);

        let data_actual = tensor.neg().into_data();

        let data_expected = Data::from([[-0.0, -1.0, -2.0], [-3.0, -4.0, -5.0]]);
        assert_eq!(data_expected, data_actual);
    }
}
