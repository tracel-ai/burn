use crate::{backend::ndarray::NdArrayTensor, TensorOpsIndex};
use ndarray::SliceInfoElem;
use std::ops::Range;

impl<
        P: tch::kind::Element + std::fmt::Debug + Copy + Default,
        const D1: usize,
        const D2: usize,
    > TensorOpsIndex<P, D1, D2> for NdArrayTensor<P, D1>
{
    fn index(&self, indexes: [Range<usize>; D2]) -> Self {
        let slices = to_slice_args::<D1, D2>(indexes.clone());
        let array = self
            .array
            .clone()
            .slice_move(slices.as_slice())
            .into_shared();
        let shape = self.shape.index(indexes.clone());

        Self { array, shape }
    }

    fn index_assign(&self, indexes: [Range<usize>; D2], values: &Self) -> Self {
        let slices = to_slice_args::<D1, D2>(indexes.clone());
        let mut array = self.array.to_owned();
        array.slice_mut(slices.as_slice()).assign(&values.array);
        let array = array.into_owned().into_shared();

        let shape = self.shape.clone();

        Self { array, shape }
    }
}

fn to_slice_args<const D1: usize, const D2: usize>(
    indexes: [Range<usize>; D2],
) -> [SliceInfoElem; D1] {
    let mut slices = [SliceInfoElem::NewAxis; D1];
    for i in 0..D1 {
        if i >= D2 {
            slices[i] = SliceInfoElem::Slice {
                start: 0,
                end: None,
                step: 1,
            }
        } else {
            slices[i] = SliceInfoElem::Slice {
                start: indexes[i].start as isize,
                end: Some(indexes[i].end as isize),
                step: 1,
            }
        }
    }
    slices
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Data, TensorBase};

    #[test]
    fn should_support_full_indexing_1d() {
        let data = Data::<f64, 1>::from([0.0, 1.0, 2.0]);
        let tensor = NdArrayTensor::from_data(data.clone());

        let data_actual = tensor.index([0..3]).into_data();

        assert_eq!(data, data_actual);
    }

    #[test]
    fn should_support_partial_indexing_1d() {
        let data = Data::<f64, 1>::from([0.0, 1.0, 2.0]);
        let tensor = NdArrayTensor::from_data(data.clone());

        let data_actual = tensor.index([1..3]).into_data();

        let data_expected = Data::from([1.0, 2.0]);
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_full_indexing_2d() {
        let data = Data::<f64, 2>::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = NdArrayTensor::from_data(data.clone());

        let data_actual_1 = tensor.index([0..2]).into_data();
        let data_actual_2 = tensor.index([0..2, 0..3]).into_data();

        assert_eq!(data, data_actual_1);
        assert_eq!(data, data_actual_2);
    }

    #[test]
    fn should_support_partial_indexing_2d() {
        let data = Data::<f64, 2>::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = NdArrayTensor::from_data(data.clone());

        let data_actual = tensor.index([0..2, 0..2]).into_data();

        let data_expected = Data::from([[0.0, 1.0], [3.0, 4.0]]);
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_indexe_assign_1d() {
        let data = Data::<f64, 1>::from([0.0, 1.0, 2.0]);
        let data_assigned = Data::<f64, 1>::from([10.0, 5.0]);

        let tensor = NdArrayTensor::from_data(data.clone());
        let tensor_assigned = NdArrayTensor::from_data(data_assigned.clone());

        let data_actual = tensor.index_assign([0..2], &tensor_assigned).into_data();

        let data_expected = Data::<f64, 1>::from([10.0, 5.0, 2.0]);
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_indexe_assign_2d() {
        let data = Data::<f64, 2>::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let data_assigned = Data::<f64, 2>::from([[10.0, 5.0]]);

        let tensor = NdArrayTensor::from_data(data.clone());
        let tensor_assigned = NdArrayTensor::from_data(data_assigned.clone());

        let data_actual = tensor
            .index_assign([1..2, 0..2], &tensor_assigned)
            .into_data();

        let data_expected = Data::<f64, 2>::from([[0.0, 1.0, 2.0], [10.0, 5.0, 5.0]]);
        assert_eq!(data_expected, data_actual);
    }
}
