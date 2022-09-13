use crate::tensor::{backend::ndarray::NdArrayTensor, ops::*};
use ndarray::SliceInfoElem;
use std::ops::Range;

impl<P: std::fmt::Debug + Copy + Default, const D1: usize> TensorOpsIndex<P, D1>
    for NdArrayTensor<P, D1>
{
    fn index<const D2: usize>(&self, indexes: [Range<usize>; D2]) -> Self {
        let slices = to_slice_args::<D1, D2>(indexes.clone());
        let array = self
            .array
            .clone()
            .slice_move(slices.as_slice())
            .into_shared();
        let shape = self.shape.index(indexes);

        Self { array, shape }
    }

    fn index_assign<const D2: usize>(&self, indexes: [Range<usize>; D2], values: &Self) -> Self {
        let slices = to_slice_args::<D1, D2>(indexes);
        let mut array = self.array.to_owned();
        array.slice_mut(slices.as_slice()).assign(&values.array);
        let array = array.into_owned().into_shared();

        let shape = self.shape;

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
