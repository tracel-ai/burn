use crate::{backend::tch::TchTensor, Shape, TensorOpsMatmul};

impl<P: tch::kind::Element, const D: usize> TensorOpsMatmul<P, D> for TchTensor<P, D> {
    fn matmul(&self, other: &Self) -> Self {
        let tensor = self.tensor.matmul(&other.tensor);
        let kind = self.kind.clone();
        let shape = Shape::from(tensor.size());

        Self {
            kind,
            tensor,
            shape,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Data, TensorBase};

    #[test]
    fn should_support_matmul_2_dims() {
        let data_1 = Data::<f64, 2>::from([[4.0, 3.0], [8.0, 7.0]]);
        let data_2 = Data::<f64, 2>::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor_1 = TchTensor::from_data(data_1, tch::Device::Cpu);
        let tensor_2 = TchTensor::from_data(data_2, tch::Device::Cpu);

        let data_actual = tensor_1.matmul(&tensor_2).into_data();

        let data_expected = Data::from([[9.0, 16.0, 23.0], [21.0, 36.0, 51.0]]);
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_matmul_3_dims() {
        let data_1 = Data::<f64, 3>::from([[[4.0, 3.0], [8.0, 7.0]], [[4.0, 3.0], [8.0, 7.0]]]);
        let data_2 = Data::<f64, 3>::from([
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
        ]);
        let tensor_1 = TchTensor::from_data(data_1, tch::Device::Cpu);
        let tensor_2 = TchTensor::from_data(data_2, tch::Device::Cpu);

        let data_actual = tensor_1.matmul(&tensor_2).into_data();

        let data_expected = Data::from([
            [[9.0, 16.0, 23.0], [21.0, 36.0, 51.0]],
            [[9.0, 16.0, 23.0], [21.0, 36.0, 51.0]],
        ]);
        assert_eq!(data_expected, data_actual);
    }
}
