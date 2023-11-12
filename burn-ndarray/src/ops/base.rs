use alloc::vec::Vec;
use burn_tensor::Data;
use burn_tensor::ElementConversion;
use core::{marker::PhantomData, ops::Range};
use ndarray::s;
use ndarray::Array2;

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
    pub fn slice<const D1: usize, const D2: usize>(
        tensor: NdArrayTensor<E, D1>,
        ranges: [Range<usize>; D2],
    ) -> NdArrayTensor<E, D1> {
        let slices = Self::to_slice_args::<D1, D2>(ranges);
        let array = tensor.array.slice_move(slices.as_slice()).into_shared();

        NdArrayTensor { array }
    }

    pub fn slice_assign<const D1: usize, const D2: usize>(
        tensor: NdArrayTensor<E, D1>,
        ranges: [Range<usize>; D2],
        value: NdArrayTensor<E, D1>,
    ) -> NdArrayTensor<E, D1> {
        let slices = Self::to_slice_args::<D1, D2>(ranges);
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
        ranges: [Range<usize>; D2],
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
                    start: ranges[i].start as isize,
                    end: Some(ranges[i].end as isize),
                    step: 1,
                }
            }
        }
        slices
    }

    pub fn swap_dims<const D: usize>(
        tensor: NdArrayTensor<E, D>,
        dim1: usize,
        dim2: usize,
    ) -> NdArrayTensor<E, D> {
        let mut array = tensor.array;
        array.swap_axes(dim1, dim2);

        NdArrayTensor::new(array)
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

    pub fn recip<const D: usize>(tensor: NdArrayTensor<E, D>) -> NdArrayTensor<E, D> {
        let array = tensor.array.map(|x| 1.elem::<E>() / *x);
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

    pub fn gather<const D: usize>(
        dim: usize,
        mut tensor: NdArrayTensor<E, D>,
        mut indices: NdArrayTensor<i64, D>,
    ) -> NdArrayTensor<E, D> {
        if dim != D - 1 {
            tensor.array.swap_axes(D - 1, dim);
            indices.array.swap_axes(D - 1, dim);
        }
        let (shape_tensor, shape_indices) = (tensor.shape(), indices.shape());
        let (size_tensor, size_index) = (shape_tensor.dims[D - 1], shape_indices.dims[D - 1]);
        let batch_size = Self::gather_batch_size(&shape_tensor, &shape_indices);

        let indices = NdArrayOps::reshape(indices, Shape::new([batch_size, size_index])).array;
        let tensor = NdArrayOps::reshape(tensor, Shape::new([batch_size, size_tensor])).array;
        let mut output = Array2::zeros((batch_size, size_index));

        for b in 0..batch_size {
            let indices = indices.slice(s!(b, ..));

            for (i, index) in indices.iter().enumerate() {
                output[[b, i]] = tensor[[b, *index as usize]];
            }
        }

        let mut output = NdArrayOps::reshape(
            NdArrayTensor::<E, 2>::new(output.into_shared().into_dyn()),
            shape_indices,
        );

        if dim != D - 1 {
            output.array.swap_axes(D - 1, dim);
        }

        output
    }

    pub fn scatter<const D: usize>(
        dim: usize,
        mut tensor: NdArrayTensor<E, D>,
        mut indices: NdArrayTensor<i64, D>,
        mut value: NdArrayTensor<E, D>,
    ) -> NdArrayTensor<E, D> {
        if dim != D - 1 {
            tensor.array.swap_axes(D - 1, dim);
            indices.array.swap_axes(D - 1, dim);
            value.array.swap_axes(D - 1, dim);
        }

        let (shape_tensor, shape_indices, shape_value) =
            (tensor.shape(), indices.shape(), value.shape());
        let (size_tensor, size_index, size_value) = (
            shape_tensor.dims[D - 1],
            shape_indices.dims[D - 1],
            shape_value.dims[D - 1],
        );
        let batch_size = Self::gather_batch_size(&shape_tensor, &shape_indices);

        if shape_value != shape_indices {
            panic!("Invalid dimension: the shape of the index tensor should be the same as the value tensor: Index {:?} value {:?}", shape_indices.dims, shape_value.dims);
        }

        let indices = NdArrayOps::reshape(indices, Shape::new([batch_size, size_index])).array;
        let value = NdArrayOps::reshape(value, Shape::new([batch_size, size_value])).array;
        let mut tensor = NdArrayOps::reshape(tensor, Shape::new([batch_size, size_tensor])).array;

        for b in 0..batch_size {
            let indices = indices.slice(s!(b, ..));

            for (i, index) in indices.iter().enumerate() {
                let index = *index as usize;
                tensor[[b, index]] += value[[b, i]];
            }
        }

        let mut output = NdArrayOps::reshape(
            NdArrayTensor::<E, 2>::new(tensor.into_shared().into_dyn()),
            shape_tensor,
        );
        if dim != D - 1 {
            output.array.swap_axes(D - 1, dim);
        }
        output
    }

    pub fn mask_where<const D: usize>(
        tensor: NdArrayTensor<E, D>,
        mask: NdArrayTensor<bool, D>,
        source: NdArrayTensor<E, D>,
    ) -> NdArrayTensor<E, D> {
        let mask_mul_4tensor = mask.array.mapv(|x| match x {
            true => 0.elem(),
            false => 1.elem(),
        });
        let mask_mul_4source = mask.array.mapv(|x| match x {
            true => 1.elem(),
            false => 0.elem(),
        });
        let array = (tensor.array * mask_mul_4tensor) + (source.array * mask_mul_4source);

        NdArrayTensor::new(array)
    }

    pub fn mask_fill<const D: usize>(
        tensor: NdArrayTensor<E, D>,
        mask: NdArrayTensor<bool, D>,
        value: E,
    ) -> NdArrayTensor<E, D> {
        let mask_mul = mask.array.mapv(|x| match x {
            true => 0.elem(),
            false => 1.elem(),
        });
        let mask_add = mask.array.mapv(|x| match x {
            true => value,
            false => 0.elem(),
        });
        let array = (tensor.array * mask_mul) + mask_add;

        NdArrayTensor::new(array)
    }

    fn gather_batch_size<const D: usize>(
        shape_tensor: &Shape<D>,
        shape_indices: &Shape<D>,
    ) -> usize {
        let mut batch_size = 1;

        for i in 0..D - 1 {
            if shape_tensor.dims[i] != shape_indices.dims[i] {
                panic!("Unsupported dimension, only the last dimension can differ: Tensor {:?} Index {:?}", shape_tensor.dims, shape_indices.dims);
            }
            batch_size *= shape_indices.dims[i];
        }

        batch_size
    }

    pub fn select<const D: usize>(
        tensor: NdArrayTensor<E, D>,
        dim: usize,
        indices: NdArrayTensor<i64, 1>,
    ) -> NdArrayTensor<E, D> {
        let array = tensor.array.select(
            Axis(dim),
            &indices
                .array
                .into_iter()
                .map(|i| i as usize)
                .collect::<Vec<_>>(),
        );

        NdArrayTensor::new(array.into_shared())
    }

    pub fn select_assign<const D1: usize, const D2: usize>(
        tensor: NdArrayTensor<E, D1>,
        dim: usize,
        indices: NdArrayTensor<i64, 1>,
        value: NdArrayTensor<E, D2>,
    ) -> NdArrayTensor<E, D1> {
        let mut output_array = tensor.array.into_owned();

        for (index_value, index) in indices.array.into_iter().enumerate() {
            let mut view = output_array.index_axis_mut(Axis(dim), index as usize);
            let value = value.array.index_axis(Axis(dim), index_value);

            view.zip_mut_with(&value, |a, b| *a += *b);
        }

        NdArrayTensor::new(output_array.into_shared())
    }
    pub fn argmax<const D: usize>(
        tensor: NdArrayTensor<E, D>,
        dim: usize,
    ) -> NdArrayTensor<i64, D> {
        arg(tensor, dim, CmpType::Max)
    }

    pub fn argmin<const D: usize>(
        tensor: NdArrayTensor<E, D>,
        dim: usize,
    ) -> NdArrayTensor<i64, D> {
        arg(tensor, dim, CmpType::Min)
    }

    pub fn clamp_min<const D: usize>(
        mut tensor: NdArrayTensor<E, D>,
        min: E,
    ) -> NdArrayTensor<E, D> {
        tensor.array.mapv_inplace(|x| match x < min {
            true => min,
            false => x,
        });

        tensor
    }

    pub fn clamp_max<const D: usize>(
        mut tensor: NdArrayTensor<E, D>,
        max: E,
    ) -> NdArrayTensor<E, D> {
        tensor.array.mapv_inplace(|x| match x > max {
            true => max,
            false => x,
        });

        tensor
    }

    pub fn clamp<const D: usize>(
        mut tensor: NdArrayTensor<E, D>,
        min: E,
        max: E,
    ) -> NdArrayTensor<E, D> {
        tensor.array.mapv_inplace(|x| match x < min {
            true => min,
            false => match x > max {
                true => max,
                false => x,
            },
        });

        tensor
    }
}

enum CmpType {
    Min,
    Max,
}

fn arg<E: NdArrayElement, const D: usize>(
    tensor: NdArrayTensor<E, D>,
    dim: usize,
    cmp: CmpType,
) -> NdArrayTensor<i64, D> {
    let mut reshape = tensor.array.shape().to_vec();
    reshape[dim] = 1;

    let output = tensor.array.map_axis(Axis(dim), |arr| {
        // Find the min/max value in the array, and return its index.
        let (_e, idx) = arr.indexed_iter().fold((arr[0], 0usize), |acc, (idx, e)| {
            let cmp = match cmp {
                CmpType::Min => e < &acc.0,
                CmpType::Max => e > &acc.0,
            };

            if cmp {
                (*e, idx)
            } else {
                acc
            }
        });

        idx as i64
    });

    let output = output.into_shape(Dim(reshape.as_slice())).unwrap();

    NdArrayTensor {
        array: output.into_shared(),
    }
}
