use alloc::vec::Vec;
use burn_tensor::Data;
use core::{marker::PhantomData, ops::Range};

use burn_tensor::Shape;
use ndarray::Axis;
use ndarray::Dim;
use ndarray::IxDyn;
use ndarray::SliceInfoElem;

use crate::element::NdArrayElement;
use crate::ops::macros::{keepdim, mean_dim, sum_dim};
use crate::{reshape, tensor::NdArrayTensor};

pub struct NdArrayOps<E> {
    e: PhantomData<E>,
}

pub(crate) struct NdArrayMathOps<E> {
    e: PhantomData<E>,
}

impl<E> NdArrayOps<E>
where
    E: Copy,
{
    pub fn index<const D1: usize, const D2: usize>(
        tensor: NdArrayTensor<E, D1>,
        indexes: [Range<usize>; D2],
    ) -> NdArrayTensor<E, D1> {
        let slices = Self::to_slice_args::<D1, D2>(indexes);
        let array = tensor.array.slice_move(slices.as_slice()).into_shared();

        NdArrayTensor { array }
    }

    pub fn index_assign<const D1: usize, const D2: usize>(
        tensor: NdArrayTensor<E, D1>,
        indexes: [Range<usize>; D2],
        value: NdArrayTensor<E, D1>,
    ) -> NdArrayTensor<E, D1> {
        let slices = Self::to_slice_args::<D1, D2>(indexes);
        let mut array = tensor.array.into_owned();
        array.slice_mut(slices.as_slice()).assign(&value.array);
        let array = array.into_shared();

        NdArrayTensor { array }
    }

    pub fn reshape<const D1: usize, const D2: usize>(
        tensor: NdArrayTensor<E, D1>,
        shape: Shape<D2>,
    ) -> NdArrayTensor<E, D2> {
        reshape!(
            ty E,
            shape shape,
            array tensor.array,
            d D2
        )
    }

    pub fn cat<const D: usize>(
        tensors: Vec<NdArrayTensor<E, D>>,
        dim: usize,
    ) -> NdArrayTensor<E, D> {
        let arrays: Vec<ndarray::ArrayView<E, IxDyn>> =
            tensors.iter().map(|t| t.array.view()).collect();
        let array = ndarray::concatenate(Axis(dim), &arrays)
            .unwrap()
            .into_shared();

        NdArrayTensor { array }
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
}

impl<E> NdArrayMathOps<E>
where
    E: Copy + NdArrayElement,
{
    pub fn add<const D: usize>(
        lhs: NdArrayTensor<E, D>,
        rhs: NdArrayTensor<E, D>,
    ) -> NdArrayTensor<E, D> {
        let array = &lhs.array + &rhs.array;
        let array = array.into_shared();

        NdArrayTensor { array }
    }

    pub fn add_scalar<const D: usize>(lhs: NdArrayTensor<E, D>, rhs: E) -> NdArrayTensor<E, D> {
        let array = lhs.array + rhs;
        let array = array.into_shared();

        NdArrayTensor { array }
    }

    pub fn sub<const D: usize>(
        lhs: NdArrayTensor<E, D>,
        rhs: NdArrayTensor<E, D>,
    ) -> NdArrayTensor<E, D> {
        let array = lhs.array - rhs.array;
        let array = array.into_shared();

        NdArrayTensor { array }
    }

    pub fn sub_scalar<const D: usize>(lhs: NdArrayTensor<E, D>, rhs: E) -> NdArrayTensor<E, D> {
        let array = lhs.array - rhs;
        let array = array.into_shared();

        NdArrayTensor { array }
    }

    pub fn mul<const D: usize>(
        lhs: NdArrayTensor<E, D>,
        rhs: NdArrayTensor<E, D>,
    ) -> NdArrayTensor<E, D> {
        let array = lhs.array * rhs.array;
        let array = array.into_shared();

        NdArrayTensor { array }
    }

    pub fn mul_scalar<const D: usize>(lhs: NdArrayTensor<E, D>, rhs: E) -> NdArrayTensor<E, D> {
        let array = lhs.array * rhs;
        let array = array.into_shared();

        NdArrayTensor { array }
    }

    pub fn div<const D: usize>(
        lhs: NdArrayTensor<E, D>,
        rhs: NdArrayTensor<E, D>,
    ) -> NdArrayTensor<E, D> {
        let array = lhs.array / rhs.array;
        let array = array.into_shared();

        NdArrayTensor { array }
    }

    pub fn div_scalar<const D: usize>(lhs: NdArrayTensor<E, D>, rhs: E) -> NdArrayTensor<E, D> {
        let array = lhs.array / rhs;
        let array = array.into_shared();

        NdArrayTensor { array }
    }

    pub fn mean<const D: usize>(tensor: NdArrayTensor<E, D>) -> NdArrayTensor<E, 1> {
        let data = Data::from([tensor.array.mean().unwrap()]);
        NdArrayTensor::from_data(data)
    }

    pub fn sum<const D: usize>(tensor: NdArrayTensor<E, D>) -> NdArrayTensor<E, 1> {
        let data = Data::from([tensor.array.sum()]);
        NdArrayTensor::from_data(data)
    }

    pub fn mean_dim<const D: usize>(
        tensor: NdArrayTensor<E, D>,
        dim: usize,
    ) -> NdArrayTensor<E, D> {
        match D {
            1 => keepdim!(0, dim, tensor, mean),
            2 => keepdim!(1, dim, tensor, mean),
            3 => keepdim!(2, dim, tensor, mean),
            4 => keepdim!(3, dim, tensor, mean),
            5 => keepdim!(4, dim, tensor, mean),
            6 => keepdim!(5, dim, tensor, mean),
            _ => panic!("Dim not supported {D}"),
        }
    }

    pub fn sum_dim<const D: usize>(tensor: NdArrayTensor<E, D>, dim: usize) -> NdArrayTensor<E, D> {
        match D {
            1 => keepdim!(0, dim, tensor, sum),
            2 => keepdim!(1, dim, tensor, sum),
            3 => keepdim!(2, dim, tensor, sum),
            4 => keepdim!(3, dim, tensor, sum),
            5 => keepdim!(4, dim, tensor, sum),
            6 => keepdim!(5, dim, tensor, sum),
            _ => panic!("Dim not supported {D}"),
        }
    }

    pub fn index_select_dim<const D: usize>(
        tensor: NdArrayTensor<E, D>,
        dim: usize,
        indexes: NdArrayTensor<i64, 1>,
    ) -> NdArrayTensor<E, D> {
        let array = tensor.array.select(
            Axis(dim),
            &indexes
                .array
                .into_iter()
                .map(|i| i as usize)
                .collect::<Vec<_>>(),
        );

        NdArrayTensor::new(array.into_shared())
    }

    pub fn index_select_dim_assign<const D1: usize, const D2: usize>(
        tensor: NdArrayTensor<E, D1>,
        dim: usize,
        indexes: NdArrayTensor<i64, 1>,
        value: NdArrayTensor<E, D2>,
    ) -> NdArrayTensor<E, D1> {
        let mut output_array = tensor.array.into_owned();

        for (index_value, index) in indexes.array.into_iter().enumerate() {
            let mut view = output_array.index_axis_mut(Axis(dim), index as usize);
            let value = value.array.index_axis(Axis(0), index_value);

            view.zip_mut_with(&value, |a, b| *a = *a + *b);
        }

        NdArrayTensor::new(output_array.into_shared())
    }
}
