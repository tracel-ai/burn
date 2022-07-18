use crate::{backend::tch::TchTensor, TensorOpsIndex};
use std::ops::Range;

impl<
        P: tch::kind::Element + std::fmt::Debug + Copy + Default,
        const D1: usize,
        const D2: usize,
    > TensorOpsIndex<P, D1, D2> for TchTensor<P, D1>
{
    fn index(&self, indices: [Range<usize>; D2]) -> Self {
        let mut tensor = self.tensor.shallow_clone();

        for i in 0..D2 {
            let index = indices[i].clone();
            let start = index.start as i64;
            let length = (index.end - index.start) as i64;
            tensor = tensor.narrow(i as i64, start, length)
        }
        let shape = self.shape.index(indices);
        let kind = self.kind.clone();

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
    fn should_support_full_indexing_1d() {
        let data = Data::<f64, 1>::from([0.0, 1.0, 2.0]);
        let tensor = TchTensor::from_data(data.clone(), tch::Device::Cpu);

        let data_actual = tensor.index([0..3]).into_data();

        assert_eq!(data, data_actual);
    }

    #[test]
    fn should_support_partial_indexing_1d() {
        let data = Data::<f64, 1>::from([0.0, 1.0, 2.0]);
        let tensor = TchTensor::from_data(data.clone(), tch::Device::Cpu);

        let data_actual = tensor.index([1..3]).into_data();

        let data_expected = Data::from([1.0, 2.0]);
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_full_indexing_2d() {
        let data = Data::<f64, 2>::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = TchTensor::from_data(data.clone(), tch::Device::Cpu);

        let data_actual_1 = tensor.index([0..2]).into_data();
        let data_actual_2 = tensor.index([0..2, 0..3]).into_data();

        assert_eq!(data, data_actual_1);
        assert_eq!(data, data_actual_2);
    }

    #[test]
    fn should_support_partial_indexing_2d() {
        let data = Data::<f64, 2>::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = TchTensor::from_data(data.clone(), tch::Device::Cpu);

        let data_actual = tensor.index([0..2, 0..2]).into_data();

        let data_expected = Data::from([[0.0, 1.0], [3.0, 4.0]]);
        assert_eq!(data_expected, data_actual);
    }
}
