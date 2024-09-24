use alloc::{vec, vec::Vec};
use burn_tensor::ElementConversion;
use burn_tensor::TensorData;
use core::fmt::Debug;
use core::{marker::PhantomData, ops::Range};
use ndarray::s;
use ndarray::Array2;
use ndarray::IntoDimension;
use ndarray::SliceInfo;
use ndarray::Zip;
use num_traits::Signed;

use burn_tensor::Shape;
use ndarray::Axis;
use ndarray::Dim;
use ndarray::IxDyn;
use ndarray::SliceInfoElem;

use crate::element::NdArrayElement;
use crate::ops::macros::{keepdim, mean_dim, prod_dim, sum_dim};
use crate::{reshape, tensor::NdArrayTensor};

pub struct NdArrayOps<E> {
    e: PhantomData<E>,
}

pub(crate) struct NdArrayMathOps<E> {
    e: PhantomData<E>,
}

impl<E> NdArrayOps<E>
where
    E: Copy + Debug,
{
    pub fn slice(tensor: NdArrayTensor<E>, ranges: &[Range<usize>]) -> NdArrayTensor<E> {
        let slices = Self::to_slice_args(ranges, tensor.shape().num_dims());
        let array = tensor.array.slice_move(slices.as_slice()).into_shared();

        NdArrayTensor { array }
    }

    pub fn slice_assign(
        tensor: NdArrayTensor<E>,
        ranges: &[Range<usize>],
        value: NdArrayTensor<E>,
    ) -> NdArrayTensor<E> {
        let slices = Self::to_slice_args(ranges, tensor.shape().num_dims());
        let mut array = tensor.array.into_owned();
        array.slice_mut(slices.as_slice()).assign(&value.array);
        let array = array.into_shared();

        NdArrayTensor { array }
    }

    pub fn reshape(tensor: NdArrayTensor<E>, shape: Shape) -> NdArrayTensor<E> {
        reshape!(
            ty E,
            shape shape,
            array tensor.array,
            d shape.num_dims()
        )
    }

    pub fn cat(tensors: Vec<NdArrayTensor<E>>, dim: usize) -> NdArrayTensor<E> {
        let arrays: Vec<ndarray::ArrayView<E, IxDyn>> =
            tensors.iter().map(|t| t.array.view()).collect();
        let array = ndarray::concatenate(Axis(dim), &arrays)
            .unwrap()
            .into_shared();

        // Transform column-major layout into row-major (standard) layout. (fix #1053)
        let array = NdArrayTensor { array };
        Self::reshape(array.clone(), array.shape())
    }

    fn to_slice_args(ranges: &[Range<usize>], ndims: usize) -> Vec<SliceInfoElem> {
        let mut slices = vec![SliceInfoElem::NewAxis; ndims];
        for i in 0..ndims {
            if i >= ranges.len() {
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

    pub fn swap_dims(tensor: NdArrayTensor<E>, dim1: usize, dim2: usize) -> NdArrayTensor<E> {
        let mut array = tensor.array;
        array.swap_axes(dim1, dim2);

        NdArrayTensor::new(array)
    }

    pub fn permute(tensor: NdArrayTensor<E>, axes: &[usize]) -> NdArrayTensor<E> {
        let array = tensor.array.permuted_axes(axes.into_dimension());

        NdArrayTensor::new(array)
    }

    /// Broadcasts the tensor to the given shape
    pub(crate) fn expand(tensor: NdArrayTensor<E>, shape: Shape) -> NdArrayTensor<E> {
        let array = tensor
            .array
            .broadcast(shape.dims.into_dimension())
            .expect("The shapes should be broadcastable")
            // need to convert view to owned array because NdArrayTensor expects owned array
            // and try_into_owned_nocopy() panics for broadcasted arrays (zero strides)
            .into_owned()
            .into_shared();
        NdArrayTensor { array }
    }

    pub fn flip(tensor: NdArrayTensor<E>, axes: &[usize]) -> NdArrayTensor<E> {
        let slice_items: Vec<_> = (0..tensor.shape().num_dims())
            .map(|i| {
                if axes.contains(&i) {
                    SliceInfoElem::Slice {
                        start: 0,
                        end: None,
                        step: -1,
                    }
                } else {
                    SliceInfoElem::Slice {
                        start: 0,
                        end: None,
                        step: 1,
                    }
                }
            })
            .collect();
        let slice_info =
            SliceInfo::<Vec<SliceInfoElem>, IxDyn, IxDyn>::try_from(slice_items).unwrap();
        let array = tensor.array.slice(slice_info).into_owned().into_shared();

        NdArrayTensor::new(array)
    }
}

impl<E> NdArrayMathOps<E>
where
    E: Copy + NdArrayElement,
{
    pub fn add(lhs: NdArrayTensor<E>, rhs: NdArrayTensor<E>) -> NdArrayTensor<E> {
        let array = &lhs.array + &rhs.array;
        let array = array.into_shared();

        NdArrayTensor { array }
    }

    pub fn add_scalar(lhs: NdArrayTensor<E>, rhs: E) -> NdArrayTensor<E> {
        let array = lhs.array + rhs;
        let array = array.into_shared();

        NdArrayTensor { array }
    }

    pub fn sub(lhs: NdArrayTensor<E>, rhs: NdArrayTensor<E>) -> NdArrayTensor<E> {
        let array = lhs.array - rhs.array;
        let array = array.into_shared();

        NdArrayTensor { array }
    }

    pub fn sub_scalar(lhs: NdArrayTensor<E>, rhs: E) -> NdArrayTensor<E> {
        let array = lhs.array - rhs;
        let array = array.into_shared();

        NdArrayTensor { array }
    }

    pub fn mul(lhs: NdArrayTensor<E>, rhs: NdArrayTensor<E>) -> NdArrayTensor<E> {
        let array = lhs.array * rhs.array;
        let array = array.into_shared();

        NdArrayTensor { array }
    }

    pub fn mul_scalar(lhs: NdArrayTensor<E>, rhs: E) -> NdArrayTensor<E> {
        let array = lhs.array * rhs;
        let array = array.into_shared();

        NdArrayTensor { array }
    }

    pub fn div(lhs: NdArrayTensor<E>, rhs: NdArrayTensor<E>) -> NdArrayTensor<E> {
        let array = lhs.array / rhs.array;
        let array = array.into_shared();

        NdArrayTensor { array }
    }

    pub fn div_scalar(lhs: NdArrayTensor<E>, rhs: E) -> NdArrayTensor<E> {
        let array = lhs.array / rhs;
        let array = array.into_shared();

        NdArrayTensor { array }
    }

    pub fn remainder_scalar(lhs: NdArrayTensor<E>, rhs: E) -> NdArrayTensor<E>
    where
        E: core::ops::Rem<Output = E>,
    {
        let array = lhs.array.mapv(|x| ((x % rhs) + rhs) % rhs);
        let array = array.into_shared();

        NdArrayTensor { array }
    }

    pub fn recip(tensor: NdArrayTensor<E>) -> NdArrayTensor<E> {
        let array = tensor.array.map(|x| 1.elem::<E>() / *x);
        let array = array.into_shared();

        NdArrayTensor { array }
    }

    pub fn mean(tensor: NdArrayTensor<E>) -> NdArrayTensor<E> {
        let data = TensorData::from([tensor.array.mean().unwrap()]);
        NdArrayTensor::from_data(data)
    }

    pub fn sum(tensor: NdArrayTensor<E>) -> NdArrayTensor<E> {
        let data = TensorData::from([tensor.array.sum()]);
        NdArrayTensor::from_data(data)
    }

    pub fn prod(tensor: NdArrayTensor<E>) -> NdArrayTensor<E> {
        let data = TensorData::from([tensor.array.product()]);
        NdArrayTensor::from_data(data)
    }

    pub fn mean_dim(tensor: NdArrayTensor<E>, dim: usize) -> NdArrayTensor<E> {
        let ndims = tensor.shape().num_dims();
        match ndims {
            d if (1..=6).contains(&d) => keepdim!(dim, tensor, mean),
            _ => panic!("Dim not supported {ndims}"),
        }
    }

    pub fn sum_dim(tensor: NdArrayTensor<E>, dim: usize) -> NdArrayTensor<E> {
        let ndims = tensor.shape().num_dims();
        match ndims {
            d if (1..=6).contains(&d) => keepdim!(dim, tensor, sum),
            _ => panic!("Dim not supported {ndims}"),
        }
    }

    pub fn prod_dim(tensor: NdArrayTensor<E>, dim: usize) -> NdArrayTensor<E> {
        let ndims = tensor.shape().num_dims();
        match ndims {
            d if (1..=6).contains(&d) => keepdim!(dim, tensor, prod),
            _ => panic!("Dim not supported {ndims}"),
        }
    }

    pub fn gather(
        dim: usize,
        mut tensor: NdArrayTensor<E>,
        mut indices: NdArrayTensor<i64>,
    ) -> NdArrayTensor<E> {
        let ndims = tensor.shape().num_dims();
        if dim != ndims - 1 {
            tensor.array.swap_axes(ndims - 1, dim);
            indices.array.swap_axes(ndims - 1, dim);
        }
        let (shape_tensor, shape_indices) = (tensor.shape(), indices.shape());
        let (size_tensor, size_index) =
            (shape_tensor.dims[ndims - 1], shape_indices.dims[ndims - 1]);
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
            NdArrayTensor::<E>::new(output.into_shared().into_dyn()),
            shape_indices,
        );

        if dim != ndims - 1 {
            output.array.swap_axes(ndims - 1, dim);
        }

        output
    }

    pub fn scatter(
        dim: usize,
        mut tensor: NdArrayTensor<E>,
        mut indices: NdArrayTensor<i64>,
        mut value: NdArrayTensor<E>,
    ) -> NdArrayTensor<E> {
        let ndims = tensor.shape().num_dims();
        if dim != ndims - 1 {
            tensor.array.swap_axes(ndims - 1, dim);
            indices.array.swap_axes(ndims - 1, dim);
            value.array.swap_axes(ndims - 1, dim);
        }

        let (shape_tensor, shape_indices, shape_value) =
            (tensor.shape(), indices.shape(), value.shape());
        let (size_tensor, size_index, size_value) = (
            shape_tensor.dims[ndims - 1],
            shape_indices.dims[ndims - 1],
            shape_value.dims[ndims - 1],
        );
        let batch_size = Self::gather_batch_size(&shape_tensor, &shape_indices);

        if shape_value != shape_indices {
            panic!(
                "Invalid dimension: the shape of the index tensor should be the same as the value \
                 tensor: Index {:?} value {:?}",
                shape_indices.dims, shape_value.dims
            );
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
            NdArrayTensor::<E>::new(tensor.into_shared().into_dyn()),
            shape_tensor,
        );
        if dim != ndims - 1 {
            output.array.swap_axes(ndims - 1, dim);
        }
        output
    }

    pub fn mask_where(
        tensor: NdArrayTensor<E>,
        mask: NdArrayTensor<bool>,
        source: NdArrayTensor<E>,
    ) -> NdArrayTensor<E> {
        let tensor = tensor.array.broadcast(mask.array.dim()).unwrap();
        let source = source.array.broadcast(mask.array.dim()).unwrap();
        let output = Zip::from(&tensor)
            .and(&mask.array)
            .and(&source)
            .map_collect(|&x, &mask_val, &y| if mask_val { y } else { x })
            .into_shared();
        NdArrayTensor::new(output)
    }

    pub fn mask_fill(
        tensor: NdArrayTensor<E>,
        mask: NdArrayTensor<bool>,
        value: E,
    ) -> NdArrayTensor<E> {
        let mut output = tensor.array.clone();
        let broadcast_mask = mask.array.broadcast(output.dim()).unwrap();
        Zip::from(&mut output)
            .and(&broadcast_mask)
            .for_each(|out, &mask_val| {
                if mask_val {
                    *out = value;
                }
            });
        NdArrayTensor::new(output.into_shared())
    }

    fn gather_batch_size(shape_tensor: &Shape, shape_indices: &Shape) -> usize {
        let ndims = shape_tensor.num_dims();
        let mut batch_size = 1;

        for i in 0..ndims - 1 {
            if shape_tensor.dims[i] != shape_indices.dims[i] {
                panic!(
                    "Unsupported dimension, only the last dimension can differ: Tensor {:?} Index \
                     {:?}",
                    shape_tensor.dims, shape_indices.dims
                );
            }
            batch_size *= shape_indices.dims[i];
        }

        batch_size
    }

    pub fn select(
        tensor: NdArrayTensor<E>,
        dim: usize,
        indices: NdArrayTensor<i64>,
    ) -> NdArrayTensor<E> {
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

    pub fn select_assign(
        tensor: NdArrayTensor<E>,
        dim: usize,
        indices: NdArrayTensor<i64>,
        value: NdArrayTensor<E>,
    ) -> NdArrayTensor<E> {
        let mut output_array = tensor.array.into_owned();

        for (index_value, index) in indices.array.into_iter().enumerate() {
            let mut view = output_array.index_axis_mut(Axis(dim), index as usize);
            let value = value.array.index_axis(Axis(dim), index_value);

            view.zip_mut_with(&value, |a, b| *a += *b);
        }

        NdArrayTensor::new(output_array.into_shared())
    }
    pub fn argmax(tensor: NdArrayTensor<E>, dim: usize) -> NdArrayTensor<i64> {
        arg(tensor, dim, CmpType::Max)
    }

    pub fn argmin(tensor: NdArrayTensor<E>, dim: usize) -> NdArrayTensor<i64> {
        arg(tensor, dim, CmpType::Min)
    }

    pub fn clamp_min(mut tensor: NdArrayTensor<E>, min: E) -> NdArrayTensor<E> {
        tensor.array.mapv_inplace(|x| match x < min {
            true => min,
            false => x,
        });

        tensor
    }

    pub fn clamp_max(mut tensor: NdArrayTensor<E>, max: E) -> NdArrayTensor<E> {
        tensor.array.mapv_inplace(|x| match x > max {
            true => max,
            false => x,
        });

        tensor
    }

    pub fn clamp(mut tensor: NdArrayTensor<E>, min: E, max: E) -> NdArrayTensor<E> {
        tensor.array.mapv_inplace(|x| match x < min {
            true => min,
            false => match x > max {
                true => max,
                false => x,
            },
        });

        tensor
    }

    pub(crate) fn elementwise_op<OtherE>(
        lhs: NdArrayTensor<E>,
        rhs: NdArrayTensor<OtherE>,
        var_name: impl FnMut(&E, &OtherE) -> E,
    ) -> NdArrayTensor<E> {
        NdArrayTensor::new(
            Zip::from(lhs.array.view())
                .and(rhs.array.view())
                .map_collect(var_name)
                .into_shared(),
        )
    }

    pub(crate) fn elementwise_op_scalar(
        lhs: NdArrayTensor<E>,
        var_name: impl FnMut(E) -> E,
    ) -> NdArrayTensor<E> {
        NdArrayTensor::new(lhs.array.mapv(var_name).into_shared())
    }

    pub(crate) fn sign_op(tensor: NdArrayTensor<E>) -> NdArrayTensor<E>
    where
        E: Signed,
    {
        let zero = 0.elem();
        let one = 1.elem::<E>();
        NdArrayTensor::new(
            tensor
                .array
                .mapv(|x| {
                    if x > zero {
                        one
                    } else if x < zero {
                        -one
                    } else {
                        zero
                    }
                })
                .into_shared(),
        )
    }
}

enum CmpType {
    Min,
    Max,
}

fn arg<E: NdArrayElement>(
    tensor: NdArrayTensor<E>,
    dim: usize,
    cmp: CmpType,
) -> NdArrayTensor<i64> {
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

    let output = output.to_shape(Dim(reshape.as_slice())).unwrap();

    NdArrayTensor {
        array: output.into_shared(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_generate_row_major_layout_for_cat() {
        let expected_shape: &[usize] = &[4, 6, 2];
        let expected_strides: &[isize] = &[12, 2, 1];
        let expected_array: NdArrayTensor<i32> = NdArrayTensor::from_data(TensorData::from([
            [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0]],
            [[7, 0], [8, 0], [9, 0], [10, 0], [11, 0], [12, 0]],
            [[13, 0], [14, 0], [15, 0], [16, 0], [17, 0], [18, 0]],
            [[19, 0], [20, 0], [21, 0], [22, 0], [23, 0], [24, 0]],
        ]));

        // unsqueeze dim on the outermost axis
        let array = NdArrayOps::reshape(
            NdArrayTensor::<i32>::from_data(TensorData::from([
                [1, 2, 3, 4, 5, 6],
                [7, 8, 9, 10, 11, 12],
                [13, 14, 15, 16, 17, 18],
                [19, 20, 21, 22, 23, 24],
            ])),
            Shape::from([4, 6, 1]),
        );
        let zeros = NdArrayTensor::<i32>::from_data(TensorData::zeros::<i32, _>([4, 6, 1]));
        // make `ndarray` concatenates array on the outermost axis
        let array = NdArrayOps::cat([array, zeros].to_vec(), 2);

        assert!(array.array.is_standard_layout());
        assert_eq!(array.array.shape(), expected_shape);
        assert_eq!(array.array.strides(), expected_strides);
        assert_eq!(
            array.array.into_iter().collect::<Vec<_>>(),
            expected_array.array.into_iter().collect::<Vec<_>>(),
        );
    }
}
