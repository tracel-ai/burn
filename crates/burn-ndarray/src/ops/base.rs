use alloc::{vec, vec::Vec};
use burn_tensor::ElementConversion;
use burn_tensor::TensorData;
use burn_tensor::TensorMetadata;
#[cfg(feature = "simd")]
use burn_tensor::{DType, quantization::QuantizationType};
use core::fmt::Debug;
use core::{marker::PhantomData, ops::Range};
use ndarray::Array2;
use ndarray::IntoDimension;
use ndarray::SliceInfo;
use ndarray::Zip;
use ndarray::s;
use num_traits::Signed;
#[cfg(feature = "simd")]
use paste::paste;

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use burn_tensor::Shape;
use ndarray::Axis;
use ndarray::Dim;
use ndarray::IxDyn;
use ndarray::SliceInfoElem;

use crate::element::NdArrayElement;
#[cfg(feature = "simd")]
use crate::ops::simd::{
    binary::try_binary_simd,
    binary_elemwise::{
        VecAdd, VecBitAnd, VecBitOr, VecBitXor, VecClamp, VecDiv, VecMax, VecMin, VecMul, VecSub,
        try_binary_scalar_simd,
    },
    cmp::{
        VecEquals, VecGreater, VecGreaterEq, VecLower, VecLowerEq, try_cmp_scalar_simd,
        try_cmp_simd,
    },
    unary::{RecipVec, VecAbs, VecBitNot, try_unary_simd},
};
use crate::{
    IntNdArrayElement,
    ops::macros::{keepdim, mean_dim, prod_dim, sum_dim},
};
use crate::{reshape, tensor::NdArrayTensor};

pub struct NdArrayOps<E> {
    e: PhantomData<E>,
}

pub(crate) struct NdArrayMathOps<E> {
    e: PhantomData<E>,
}

impl<E> NdArrayOps<E>
where
    E: Copy + Debug + burn_tensor::Element,
{
    pub fn into_data(tensor: NdArrayTensor<E>) -> TensorData {
        tensor.into_data()
    }

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

    pub(crate) fn concatenate(
        arrays: &[ndarray::ArrayView<E, IxDyn>],
        dim: usize,
    ) -> NdArrayTensor<E> {
        let array = ndarray::concatenate(Axis(dim), arrays)
            .unwrap()
            .into_shared();

        // Transform column-major layout into row-major (standard) layout. (fix #1053)
        let array = NdArrayTensor { array };
        Self::reshape(array.clone(), array.shape())
    }

    pub fn cat(tensors: Vec<NdArrayTensor<E>>, dim: usize) -> NdArrayTensor<E> {
        let arrays: Vec<_> = tensors.iter().map(|t| t.array.view()).collect();
        Self::concatenate(&arrays, dim)
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

#[cfg(feature = "simd")]
macro_rules! dispatch_binary_simd {
    (noq, $elem: ty, $op: ty, $lhs: expr, $rhs: expr, $($ty: ty),*) => {{
        paste! {
            let simd = match $elem::dtype() {
                $(DType::[<$ty:upper>] => try_binary_simd::<$elem, $elem, $ty, $ty, $op>($lhs, $rhs),)*
                _ => Err(($lhs, $rhs)),
            };
            match simd {
                Ok(out) => return out,
                Err(args) => args,
            }
        }
    }};
    ($elem: ty, $op: ty, $lhs: expr, $rhs: expr, $($ty: ty),*) => {{
        paste! {
            let simd = match $elem::dtype() {
                $(DType::[<$ty:upper>] => try_binary_simd::<$elem, $elem, $ty, $ty, $op>($lhs, $rhs),)*
                DType::QFloat(strategy) => match strategy.q_type() {
                    QuantizationType::QInt8 => try_binary_simd::<$elem, $elem, i8, i8, $op>($lhs, $rhs),
                },
                _ => Err(($lhs, $rhs)),
            };
            match simd {
                Ok(out) => return out,
                Err(args) => args,
            }
        }
    }};
}

#[cfg(not(feature = "simd"))]
macro_rules! dispatch_binary_simd {
    (noq, $elem: ty, $op: ty, $lhs: expr, $rhs: expr, $($ty: ty),*) => {{ ($lhs, $rhs) }};
    ($elem: ty, $op: ty, $lhs: expr, $rhs: expr, $($ty: ty),*) => {{ ($lhs, $rhs) }};
}

#[cfg(feature = "simd")]
macro_rules! dispatch_binary_scalar_simd {
    (noq, $elem: ty, $op: ty, $lhs: expr, $rhs: expr, $($ty: ty),*) => {{
        paste! {
            let simd = match $elem::dtype() {
                $(DType::[<$ty:upper>] => try_binary_scalar_simd::<$elem, $elem, $ty, $ty, $op>($lhs, $rhs),)*
                _ => Err($lhs),
            };
            match simd {
                Ok(out) => return out,
                Err(args) => args,
            }
        }
    }};
    ($elem: ty, $op: ty, $lhs: expr, $rhs: expr, $($ty: ty),*) => {{
        paste! {
            let simd = match $elem::dtype() {
                $(DType::[<$ty:upper>] => try_binary_scalar_simd::<$elem, $elem, $ty, $ty, $op>($lhs, $rhs),)*
                DType::QFloat(strategy) => match strategy.q_type() {
                    QuantizationType::QInt8 => try_binary_scalar_simd::<$elem, $elem, i8, i8, $op>($lhs, $rhs),
                },
                _ => Err($lhs),
            };
            match simd {
                Ok(out) => return out,
                Err(args) => args,
            }
        }
    }};
}

#[cfg(not(feature = "simd"))]
macro_rules! dispatch_binary_scalar_simd {
    (noq, $elem: ty, $op: ty, $lhs: expr, $rhs: expr, $($ty: ty),*) => {{ $lhs }};
    ($elem: ty, $op: ty, $lhs: expr, $rhs: expr, $($ty: ty),*) => {{ $lhs }};
}

#[cfg(feature = "simd")]
macro_rules! dispatch_cmp_simd {
    ($elem: ty, $op: ty, $lhs: expr, $rhs: expr, $($ty: ty),*) => {{
        paste! {
            let simd = match $elem::dtype() {
                $(DType::[<$ty:upper>] => try_cmp_simd::<$elem, $ty, $op>($lhs, $rhs),)*
                DType::QFloat(strategy) => match strategy.q_type() {
                    QuantizationType::QInt8 => try_cmp_simd::<$elem, i8, $op>($lhs, $rhs),
                },
                _ => Err(($lhs, $rhs)),
            };
            match simd {
                Ok(out) => return out,
                Err(args) => args,
            }
        }
    }};
}

#[cfg(not(feature = "simd"))]
macro_rules! dispatch_cmp_simd {
    ($elem: ty, $op: ty, $lhs: expr, $rhs: expr, $($ty: ty),*) => {{ ($lhs, $rhs) }};
}

#[cfg(feature = "simd")]
macro_rules! dispatch_cmp_scalar_simd {
    ($elem: ty, $op: ty, $lhs: expr, $rhs: expr, $($ty: ty),*) => {{
        paste! {
            let simd = match $elem::dtype() {
                $(DType::[<$ty:upper>] => try_cmp_scalar_simd::<$elem, $ty, $op>($lhs, $rhs),)*
                DType::QFloat(strategy) => match strategy.q_type() {
                    QuantizationType::QInt8 => try_cmp_scalar_simd::<$elem, i8, $op>($lhs, $rhs),
                },
                _ => Err($lhs),
            };
            match simd {
                Ok(out) => return out,
                Err(args) => args,
            }
        }
    }};
}

#[cfg(not(feature = "simd"))]
macro_rules! dispatch_cmp_scalar_simd {
    ($elem: ty, $op: ty, $lhs: expr, $rhs: expr, $($ty: ty),*) => {{ $lhs }};
}

#[cfg(feature = "simd")]
macro_rules! dispatch_unary_simd {
    ($elem: ty, $op: ty, $lhs: expr, $($ty: ty),*) => {{
        paste! {
            let simd = match $elem::dtype() {
                $(DType::[<$ty:upper>] => try_unary_simd::<$elem, $elem, $ty, $ty, $op>($lhs),)*
                _ => Err($lhs),
            };
            match simd {
                Ok(out) => return out,
                Err(args) => args,
            }
        }
    }};
}

#[cfg(not(feature = "simd"))]
macro_rules! dispatch_unary_simd {
    ($elem: ty, $op: ty, $lhs: expr, $($ty: ty),*) => {{ $lhs }};
}

impl<E> NdArrayMathOps<E>
where
    E: Copy + NdArrayElement,
{
    pub fn add(lhs: NdArrayTensor<E>, rhs: NdArrayTensor<E>) -> NdArrayTensor<E> {
        let (lhs, rhs) = dispatch_binary_simd!(
            E, VecAdd, lhs, rhs, u8, i8, u16, i16, u32, i32, f32, u64, i64, f64
        );

        let array = &lhs.array + &rhs.array;
        let array = array.into_shared();

        NdArrayTensor { array }
    }

    pub fn add_scalar(lhs: NdArrayTensor<E>, rhs: E) -> NdArrayTensor<E> {
        let lhs = dispatch_binary_scalar_simd!(
            E,
            VecAdd,
            lhs,
            rhs.elem(),
            u8,
            i8,
            u16,
            i16,
            u32,
            i32,
            f32,
            u64,
            i64,
            f64
        );

        let array = lhs.array + rhs;
        let array = array.into_shared();

        NdArrayTensor { array }
    }

    pub fn sub(lhs: NdArrayTensor<E>, rhs: NdArrayTensor<E>) -> NdArrayTensor<E> {
        let (lhs, rhs) = dispatch_binary_simd!(
            E, VecSub, lhs, rhs, u8, i8, u16, i16, u32, i32, f32, u64, i64, f64
        );

        let array = lhs.array - rhs.array;
        let array = array.into_shared();

        NdArrayTensor { array }
    }

    pub fn sub_scalar(lhs: NdArrayTensor<E>, rhs: E) -> NdArrayTensor<E> {
        let lhs = dispatch_binary_scalar_simd!(
            E,
            VecSub,
            lhs,
            rhs.elem(),
            u8,
            i8,
            u16,
            i16,
            u32,
            i32,
            f32,
            u64,
            i64,
            f64
        );

        let array = lhs.array - rhs;
        let array = array.into_shared();

        NdArrayTensor { array }
    }

    pub fn mul(lhs: NdArrayTensor<E>, rhs: NdArrayTensor<E>) -> NdArrayTensor<E> {
        let (lhs, rhs) =
            dispatch_binary_simd!(noq, E, VecMul, lhs, rhs, u16, i16, u32, i32, f32, f64);

        let array = lhs.array * rhs.array;
        let array = array.into_shared();

        NdArrayTensor { array }
    }

    pub fn mul_scalar(lhs: NdArrayTensor<E>, rhs: E) -> NdArrayTensor<E> {
        let lhs = dispatch_binary_scalar_simd!(
            noq,
            E,
            VecMul,
            lhs,
            rhs.elem(),
            u16,
            i16,
            u32,
            i32,
            f32,
            f64
        );

        let array = lhs.array * rhs;
        let array = array.into_shared();

        NdArrayTensor { array }
    }

    pub fn div(lhs: NdArrayTensor<E>, rhs: NdArrayTensor<E>) -> NdArrayTensor<E> {
        let (lhs, rhs) = dispatch_binary_simd!(noq, E, VecDiv, lhs, rhs, f32, f64);

        let array = lhs.array / rhs.array;
        let array = array.into_shared();

        NdArrayTensor { array }
    }

    pub fn div_scalar(lhs: NdArrayTensor<E>, rhs: E) -> NdArrayTensor<E> {
        let lhs = dispatch_binary_scalar_simd!(noq, E, VecDiv, lhs, rhs.elem(), f32, f64);

        let array = lhs.array / rhs;
        let array = array.into_shared();

        NdArrayTensor { array }
    }

    pub fn remainder(lhs: NdArrayTensor<E>, rhs: NdArrayTensor<E>) -> NdArrayTensor<E> {
        let array = lhs.array.clone()
            - (lhs.array / rhs.array.clone()).mapv_into(|a| (a.to_f64()).floor().elem())
                * rhs.array;
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
        let tensor = dispatch_unary_simd!(E, RecipVec, tensor, f32);

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

    pub fn gather<I: NdArrayElement>(
        dim: usize,
        mut tensor: NdArrayTensor<E>,
        mut indices: NdArrayTensor<I>,
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
                output[[b, i]] = tensor[[b, index.elem::<i64>() as usize]];
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

    pub fn scatter<I: NdArrayElement>(
        dim: usize,
        mut tensor: NdArrayTensor<E>,
        mut indices: NdArrayTensor<I>,
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
                let index = index.elem::<i64>() as usize;
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

    pub fn select<I: NdArrayElement>(
        tensor: NdArrayTensor<E>,
        dim: usize,
        indices: NdArrayTensor<I>,
    ) -> NdArrayTensor<E> {
        let array = tensor.array.select(
            Axis(dim),
            &indices
                .array
                .into_iter()
                .map(|i| i.elem::<i64>() as usize)
                .collect::<Vec<_>>(),
        );

        NdArrayTensor::new(array.into_shared())
    }

    pub fn select_assign<I: NdArrayElement>(
        tensor: NdArrayTensor<E>,
        dim: usize,
        indices: NdArrayTensor<I>,
        value: NdArrayTensor<E>,
    ) -> NdArrayTensor<E> {
        let mut output_array = tensor.array.into_owned();

        for (index_value, index) in indices.array.into_iter().enumerate() {
            let mut view = output_array.index_axis_mut(Axis(dim), index.elem::<i64>() as usize);
            let value = value.array.index_axis(Axis(dim), index_value);

            view.zip_mut_with(&value, |a, b| *a += *b);
        }

        NdArrayTensor::new(output_array.into_shared())
    }
    pub fn argmax<I: NdArrayElement>(tensor: NdArrayTensor<E>, dim: usize) -> NdArrayTensor<I> {
        arg(tensor, dim, CmpType::Max)
    }

    pub fn argmin<I: NdArrayElement>(tensor: NdArrayTensor<E>, dim: usize) -> NdArrayTensor<I> {
        arg(tensor, dim, CmpType::Min)
    }

    pub fn clamp_min(tensor: NdArrayTensor<E>, min: E) -> NdArrayTensor<E> {
        let mut tensor = dispatch_binary_scalar_simd!(
            E,
            VecMax,
            tensor,
            min.elem(),
            u8,
            i8,
            u16,
            i16,
            u32,
            i32,
            f32,
            u64,
            i64,
            f64
        );

        tensor.array.mapv_inplace(|x| match x < min {
            true => min,
            false => x,
        });

        tensor
    }

    pub fn clamp_max(tensor: NdArrayTensor<E>, max: E) -> NdArrayTensor<E> {
        let mut tensor = dispatch_binary_scalar_simd!(
            E,
            VecMin,
            tensor,
            max.elem(),
            u8,
            i8,
            u16,
            i16,
            u32,
            i32,
            f32,
            u64,
            i64,
            f64
        );

        tensor.array.mapv_inplace(|x| match x > max {
            true => max,
            false => x,
        });

        tensor
    }

    pub fn clamp(tensor: NdArrayTensor<E>, min: E, max: E) -> NdArrayTensor<E> {
        let mut tensor = dispatch_binary_scalar_simd!(
            E,
            VecClamp,
            tensor,
            (min.elem(), max.elem()),
            u8,
            i8,
            u16,
            i16,
            u32,
            i32,
            f32,
            u64,
            i64,
            f64
        );

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
        let lhs = lhs
            .array
            .broadcast(rhs.array.dim())
            .unwrap_or(lhs.array.view());
        let rhs = rhs.array.broadcast(lhs.dim()).unwrap_or(rhs.array.view());

        NdArrayTensor::new(Zip::from(lhs).and(rhs).map_collect(var_name).into_shared())
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

    pub(crate) fn abs(tensor: NdArrayTensor<E>) -> NdArrayTensor<E> {
        let tensor = dispatch_unary_simd!(E, VecAbs, tensor, i8, i16, i32, f32, f64);

        let array = tensor.array.mapv_into(|a| a.abs_elem()).into_shared();

        NdArrayTensor::new(array)
    }

    pub(crate) fn equal(lhs: NdArrayTensor<E>, rhs: NdArrayTensor<E>) -> NdArrayTensor<bool> {
        let (lhs, rhs) = dispatch_cmp_simd!(
            E, VecEquals, lhs, rhs, u8, i8, u16, i16, u32, f32, i32, u64, i64, f64
        );

        let output = Zip::from(&lhs.array)
            .and(&rhs.array)
            .map_collect(|&lhs_val, &rhs_val| (lhs_val == rhs_val))
            .into_shared();
        NdArrayTensor::new(output)
    }

    pub(crate) fn equal_elem(lhs: NdArrayTensor<E>, rhs: E) -> NdArrayTensor<bool> {
        let lhs = dispatch_cmp_scalar_simd!(
            E,
            VecEquals,
            lhs,
            rhs.elem(),
            u8,
            i8,
            u16,
            i16,
            u32,
            f32,
            i32,
            u64,
            i64,
            f64
        );

        let array = lhs.array.mapv(|a| a == rhs).into_shared();
        NdArrayTensor { array }
    }

    pub(crate) fn greater(lhs: NdArrayTensor<E>, rhs: NdArrayTensor<E>) -> NdArrayTensor<bool> {
        let (lhs, rhs) = dispatch_cmp_simd!(
            E, VecGreater, lhs, rhs, u8, i8, u16, i16, u32, f32, i32, u64, i64, f64
        );

        let lhs = lhs
            .array
            .broadcast(rhs.array.dim())
            .unwrap_or(lhs.array.view());
        let rhs = rhs.array.broadcast(lhs.dim()).unwrap_or(rhs.array.view());

        NdArrayTensor::new(
            Zip::from(lhs)
                .and(rhs)
                .map_collect(|lhs, rhs| lhs > rhs)
                .into_shared(),
        )
    }

    pub(crate) fn greater_elem(lhs: NdArrayTensor<E>, rhs: E) -> NdArrayTensor<bool> {
        let lhs = dispatch_cmp_scalar_simd!(
            E,
            VecGreater,
            lhs,
            rhs.elem(),
            u8,
            i8,
            u16,
            i16,
            u32,
            f32,
            i32,
            u64,
            i64,
            f64
        );

        let array = lhs.array.mapv(|a| a > rhs).into_shared();
        NdArrayTensor { array }
    }

    pub(crate) fn greater_equal(
        lhs: NdArrayTensor<E>,
        rhs: NdArrayTensor<E>,
    ) -> NdArrayTensor<bool> {
        let (lhs, rhs) = dispatch_cmp_simd!(
            E,
            VecGreaterEq,
            lhs,
            rhs,
            u8,
            i8,
            u16,
            i16,
            u32,
            f32,
            i32,
            u64,
            i64,
            f64
        );

        let lhs = lhs
            .array
            .broadcast(rhs.array.dim())
            .unwrap_or(lhs.array.view());
        let rhs = rhs.array.broadcast(lhs.dim()).unwrap_or(rhs.array.view());

        NdArrayTensor::new(
            Zip::from(lhs)
                .and(rhs)
                .map_collect(|lhs, rhs| lhs >= rhs)
                .into_shared(),
        )
    }

    pub(crate) fn greater_equal_elem(lhs: NdArrayTensor<E>, rhs: E) -> NdArrayTensor<bool> {
        let lhs = dispatch_cmp_scalar_simd!(
            E,
            VecGreaterEq,
            lhs,
            rhs.elem(),
            u8,
            i8,
            u16,
            i16,
            u32,
            f32,
            i32,
            u64,
            i64,
            f64
        );

        let array = lhs.array.mapv(|a| a >= rhs).into_shared();
        NdArrayTensor { array }
    }

    pub(crate) fn lower_equal(lhs: NdArrayTensor<E>, rhs: NdArrayTensor<E>) -> NdArrayTensor<bool> {
        let (lhs, rhs) = dispatch_cmp_simd!(
            E, VecLowerEq, lhs, rhs, u8, i8, u16, i16, u32, f32, i32, u64, i64, f64
        );

        let lhs = lhs
            .array
            .broadcast(rhs.array.dim())
            .unwrap_or(lhs.array.view());
        let rhs = rhs.array.broadcast(lhs.dim()).unwrap_or(rhs.array.view());

        NdArrayTensor::new(
            Zip::from(lhs)
                .and(rhs)
                .map_collect(|lhs, rhs| lhs <= rhs)
                .into_shared(),
        )
    }

    pub(crate) fn lower_equal_elem(lhs: NdArrayTensor<E>, rhs: E) -> NdArrayTensor<bool> {
        let lhs = dispatch_cmp_scalar_simd!(
            E,
            VecLowerEq,
            lhs,
            rhs.elem(),
            u8,
            i8,
            u16,
            i16,
            u32,
            f32,
            i32,
            u64,
            i64,
            f64
        );

        let array = lhs.array.mapv(|a| a <= rhs).into_shared();
        NdArrayTensor { array }
    }

    pub(crate) fn lower(lhs: NdArrayTensor<E>, rhs: NdArrayTensor<E>) -> NdArrayTensor<bool> {
        let (lhs, rhs) = dispatch_cmp_simd!(
            E, VecLower, lhs, rhs, u8, i8, u16, i16, u32, f32, i32, u64, i64, f64
        );

        let lhs = lhs
            .array
            .broadcast(rhs.array.dim())
            .unwrap_or(lhs.array.view());
        let rhs = rhs.array.broadcast(lhs.dim()).unwrap_or(rhs.array.view());

        NdArrayTensor::new(
            Zip::from(lhs)
                .and(rhs)
                .map_collect(|lhs, rhs| lhs < rhs)
                .into_shared(),
        )
    }

    pub(crate) fn lower_elem(lhs: NdArrayTensor<E>, rhs: E) -> NdArrayTensor<bool> {
        let lhs = dispatch_cmp_scalar_simd!(
            E,
            VecLower,
            lhs,
            rhs.elem(),
            u8,
            i8,
            u16,
            i16,
            u32,
            f32,
            i32,
            u64,
            i64,
            f64
        );

        let array = lhs.array.mapv(|a| a < rhs).into_shared();
        NdArrayTensor { array }
    }
}

pub struct NdArrayBitOps<I: IntNdArrayElement>(PhantomData<I>);

impl<I: IntNdArrayElement> NdArrayBitOps<I> {
    pub(crate) fn bitand(lhs: NdArrayTensor<I>, rhs: NdArrayTensor<I>) -> NdArrayTensor<I> {
        let (lhs, rhs) =
            dispatch_binary_simd!(I, VecBitAnd, lhs, rhs, i8, u8, i16, u16, i32, u32, i64, u64);

        NdArrayMathOps::elementwise_op(lhs, rhs, |a: &I, b: &I| {
            (a.elem::<i64>() & (b.elem::<i64>())).elem()
        })
    }

    pub(crate) fn bitand_scalar(lhs: NdArrayTensor<I>, rhs: I) -> NdArrayTensor<I> {
        let lhs = dispatch_binary_scalar_simd!(
            I,
            VecBitAnd,
            lhs,
            rhs.elem(),
            i8,
            u8,
            i16,
            u16,
            i32,
            u32,
            i64,
            u64
        );

        NdArrayMathOps::elementwise_op_scalar(lhs, |a: I| {
            (a.elem::<i64>() & rhs.elem::<i64>()).elem()
        })
    }

    pub(crate) fn bitor(lhs: NdArrayTensor<I>, rhs: NdArrayTensor<I>) -> NdArrayTensor<I> {
        let (lhs, rhs) =
            dispatch_binary_simd!(I, VecBitOr, lhs, rhs, i8, u8, i16, u16, i32, u32, i64, u64);

        NdArrayMathOps::elementwise_op(lhs, rhs, |a: &I, b: &I| {
            (a.elem::<i64>() | (b.elem::<i64>())).elem()
        })
    }

    pub(crate) fn bitor_scalar(lhs: NdArrayTensor<I>, rhs: I) -> NdArrayTensor<I> {
        let lhs = dispatch_binary_scalar_simd!(
            I,
            VecBitOr,
            lhs,
            rhs.elem(),
            i8,
            u8,
            i16,
            u16,
            i32,
            u32,
            i64,
            u64
        );

        NdArrayMathOps::elementwise_op_scalar(lhs, |a: I| {
            (a.elem::<i64>() | rhs.elem::<i64>()).elem()
        })
    }

    pub(crate) fn bitxor(lhs: NdArrayTensor<I>, rhs: NdArrayTensor<I>) -> NdArrayTensor<I> {
        let (lhs, rhs) =
            dispatch_binary_simd!(I, VecBitXor, lhs, rhs, i8, u8, i16, u16, i32, u32, i64, u64);

        NdArrayMathOps::elementwise_op(lhs, rhs, |a: &I, b: &I| {
            (a.elem::<i64>() ^ (b.elem::<i64>())).elem()
        })
    }

    pub(crate) fn bitxor_scalar(lhs: NdArrayTensor<I>, rhs: I) -> NdArrayTensor<I> {
        let lhs = dispatch_binary_scalar_simd!(
            I,
            VecBitXor,
            lhs,
            rhs.elem(),
            i8,
            u8,
            i16,
            u16,
            i32,
            u32,
            i64,
            u64
        );

        NdArrayMathOps::elementwise_op_scalar(lhs, |a: I| {
            (a.elem::<i64>() ^ rhs.elem::<i64>()).elem()
        })
    }

    pub(crate) fn bitnot(tensor: NdArrayTensor<I>) -> NdArrayTensor<I> {
        let tensor =
            dispatch_unary_simd!(I, VecBitNot, tensor, i8, u8, i16, u16, i32, u32, i64, u64);

        NdArrayMathOps::elementwise_op_scalar(tensor, |a: I| (!a.elem::<i64>()).elem())
    }
}

pub struct NdArrayBoolOps;

// Rust booleans are either `00000000` or `00000001`, so bitwise and/or is fine, but bitwise not would
// produce invalid values.
impl NdArrayBoolOps {
    pub(crate) fn equal(lhs: NdArrayTensor<bool>, rhs: NdArrayTensor<bool>) -> NdArrayTensor<bool> {
        #[cfg(feature = "simd")]
        let (lhs, rhs) = match try_cmp_simd::<bool, u8, VecEquals>(lhs, rhs) {
            Ok(out) => return out,
            Err(args) => args,
        };

        let output = Zip::from(&lhs.array)
            .and(&rhs.array)
            .map_collect(|&lhs_val, &rhs_val| (lhs_val == rhs_val))
            .into_shared();
        NdArrayTensor::new(output)
    }

    pub(crate) fn and(lhs: NdArrayTensor<bool>, rhs: NdArrayTensor<bool>) -> NdArrayTensor<bool> {
        #[cfg(feature = "simd")]
        let (lhs, rhs) = match try_binary_simd::<bool, bool, u8, u8, VecBitAnd>(lhs, rhs) {
            Ok(out) => return out,
            Err(args) => args,
        };

        let output = Zip::from(&lhs.array)
            .and(&rhs.array)
            .map_collect(|&lhs_val, &rhs_val| (lhs_val && rhs_val))
            .into_shared();
        NdArrayTensor::new(output)
    }

    pub(crate) fn or(lhs: NdArrayTensor<bool>, rhs: NdArrayTensor<bool>) -> NdArrayTensor<bool> {
        #[cfg(feature = "simd")]
        let (lhs, rhs) = match try_binary_simd::<bool, bool, u8, u8, VecBitOr>(lhs, rhs) {
            Ok(out) => return out,
            Err(args) => args,
        };

        let output = Zip::from(&lhs.array)
            .and(&rhs.array)
            .map_collect(|&lhs_val, &rhs_val| (lhs_val || rhs_val))
            .into_shared();
        NdArrayTensor::new(output)
    }
}

enum CmpType {
    Min,
    Max,
}

fn arg<E: NdArrayElement, I: NdArrayElement>(
    tensor: NdArrayTensor<E>,
    dim: usize,
    cmp: CmpType,
) -> NdArrayTensor<I> {
    let mut reshape = tensor.array.shape().to_vec();
    reshape[dim] = 1;

    let output = tensor.array.map_axis(Axis(dim), |arr| {
        // Find the min/max value in the array, and return its index.
        let (_e, idx) = arr.indexed_iter().fold((arr[0], 0usize), |acc, (idx, e)| {
            let cmp = match cmp {
                CmpType::Min => e < &acc.0,
                CmpType::Max => e > &acc.0,
            };

            if cmp { (*e, idx) } else { acc }
        });

        (idx as i64).elem()
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
