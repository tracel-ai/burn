use crate::{backend::arrayfire::ArrayfireTensor, TensorOpsIndex};
use arrayfire::{HasAfEnum, Seq};
use std::ops::Range;

impl<P: HasAfEnum + std::fmt::Debug + Copy + Default, const D1: usize, const D2: usize>
    TensorOpsIndex<P, D1, D2> for ArrayfireTensor<P, D1>
{
    fn index(&self, index: [Range<usize>; D2]) -> Self {
        self.set_backend_single_ops();
        let shape = self.shape.index(index.clone());

        let mut seqs = Vec::new();
        for i in 0..D2 {
            let range = index[i].clone();
            let start = range.start;
            let end = range.end - 1;
            seqs.push(Seq::new(start as f64, end as f64, 1.0));
        }

        for i in D2..D1 {
            let dim = self.shape.dims[i];
            let start = 0;
            let end = dim - 1;
            seqs.push(Seq::new(start as f64, end as f64, 1.0));
        }

        let array = arrayfire::index(&self.array, &seqs);
        let device = self.device;

        Self {
            array,
            shape,
            device,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::arrayfire::device::Device;
    use crate::{Data, TensorBase};

    #[test]
    fn should_support_full_indexing_1d() {
        let data = Data::from([0.0, 1.0, 2.0]);
        let tensor = ArrayfireTensor::<f64, 1>::from_data(data.clone(), Device::CPU);

        let data_actual = tensor.index([0..3]).into_data();

        assert_eq!(data, data_actual);
    }

    #[test]
    fn should_support_partial_indexing_1d() {
        let data = Data::from([0.0, 1.0, 2.0]);
        let tensor = ArrayfireTensor::<f64, 1>::from_data(data, Device::CPU);

        let data_actual = tensor.index([1..3]).into_data();

        let data_expected = Data::from([1.0, 2.0]);
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_full_indexing_2d() {
        let data = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = ArrayfireTensor::<f64, 2>::from_data(data.clone(), Device::CPU);

        let data_actual_1 = tensor.index([0..2]).into_data();
        let data_actual_2 = tensor.index([0..2, 0..3]).into_data();

        assert_eq!(data, data_actual_1);
        assert_eq!(data, data_actual_2);
    }

    #[test]
    fn should_support_partial_indexing_2d() {
        let data = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = ArrayfireTensor::<f64, 2>::from_data(data, Device::CPU);

        let data_actual = tensor.index([0..2, 0..2]).into_data();

        let data_expected = Data::from([[0.0, 1.0], [3.0, 4.0]]);
        assert_eq!(data_expected, data_actual);
    }
}
