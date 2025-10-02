use alloc::{vec, vec::Vec};
#[cfg(feature = "simd")]
use burn_tensor::{DType, quantization::QuantValue};
use burn_tensor::{ElementConversion, Slice};
use core::fmt::Debug;
use core::marker::PhantomData;
use ndarray::IntoDimension;
use ndarray::SliceInfo;
use ndarray::Zip;
use ndarray::s;
use ndarray::{Array2, ArrayD};
use num_traits::Signed;
#[cfg(feature = "simd")]
use paste::paste;

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

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
use crate::reshape;
use crate::{
    IntNdArrayElement, ShapeOps,
    ops::macros::{cumsum_dim, cumprod_dim, cummin_dim, cummax_dim, keepdim, mean_dim, prod_dim, sum_dim},
};
use crate::{SharedArray, element::NdArrayElement};
use burn_tensor::Shape;
use burn_tensor::ops::unfold::calculate_unfold_shape;
use ndarray::Axis;
use ndarray::Dim;
use ndarray::IxDyn;
use ndarray::SliceInfoElem;

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
    pub fn slice(tensor: SharedArray<E>, slices: &[burn_tensor::Slice]) -> SharedArray<E> {
        let slices = Self::to_slice_args_with_steps(slices, tensor.shape().num_dims());
        tensor.slice_move(slices.as_slice()).into_shared()
    }

    pub fn slice_assign(
        tensor: SharedArray<E>,
        slices: &[burn_tensor::Slice],
        value: SharedArray<E>,
    ) -> SharedArray<E> {
        let slices = Self::to_slice_args_with_steps(slices, tensor.shape().num_dims());
        let mut array = tensor.into_owned();
        array.slice_mut(slices.as_slice()).assign(&value);
        array.into_shared()
    }

    pub fn reshape(tensor: SharedArray<E>, shape: Shape) -> SharedArray<E> {
        reshape!(
            ty E,
            shape shape,
            array tensor,
            d shape.num_dims()
        )
    }

    pub(crate) fn concatenate(
        arrays: &[ndarray::ArrayView<E, IxDyn>],
        dim: usize,
    ) -> SharedArray<E> {
        let array = ndarray::concatenate(Axis(dim), arrays)
            .unwrap()
            .into_shared();

        // Transform column-major layout into row-major (standard) layout. (fix #1053)
        Self::reshape(array.clone(), array.shape().into_shape())
    }

    pub fn cat(tensors: Vec<SharedArray<E>>, dim: usize) -> SharedArray<E> {
        let arrays: Vec<_> = tensors.iter().map(|t| t.view()).collect();
        Self::concatenate(&arrays, dim)
    }

    #[allow(clippy::wrong_self_convention)]
    fn to_slice_args_with_steps(
        burn_slices: &[burn_tensor::Slice],
        ndims: usize,
    ) -> Vec<SliceInfoElem> {
        let mut slices = vec![SliceInfoElem::NewAxis; ndims];

        for i in 0..ndims {
            slices[i] = if i < burn_slices.len() {
                let slice = &burn_slices[i];

                // Check for empty range (would result in no elements)
                if let Some(end) = slice.end
                    && slice.start == end
                {
                    SliceInfoElem::Slice {
                        start: 0,
                        end: Some(0),
                        step: 1,
                    }
                } else {
                    // Pass slice parameters directly to ndarray
                    // ndarray handles both positive and negative steps correctly:
                    // - Positive step: iterates forward from start
                    // - Negative step: iterates backward from the last element in range
                    SliceInfoElem::Slice {
                        start: slice.start,
                        end: slice.end,
                        step: slice.step,
                    }
                }
            } else {
                // Dimension not specified in slices - use full range
                SliceInfoElem::Slice {
                    start: 0,
                    end: None,
                    step: 1,
                }
            }
        }

        slices
    }

    pub fn swap_dims(mut tensor: SharedArray<E>, dim1: usize, dim2: usize) -> SharedArray<E> {
        tensor.swap_axes(dim1, dim2);

        tensor
    }

    pub fn permute(tensor: SharedArray<E>, axes: &[usize]) -> SharedArray<E> {
        tensor.permuted_axes(axes.into_dimension())
    }

    /// Broadcasts the tensor to the given shape
    pub(crate) fn expand(tensor: SharedArray<E>, shape: Shape) -> SharedArray<E> {
        tensor
            .broadcast(shape.dims.into_dimension())
            .expect("The shapes should be broadcastable")
            // need to convert view to owned array because NdArrayTensor expects owned array
            // and try_into_owned_nocopy() panics for broadcasted arrays (zero strides)
            .into_owned()
            .into_shared()
    }

    pub fn flip(tensor: SharedArray<E>, axes: &[usize]) -> SharedArray<E> {
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
        tensor.slice(slice_info).into_owned().into_shared()
    }

    /// Unfold windows along a dimension.
    ///
    /// # Warning
    ///
    /// This is a copy impl; `ndarray` doesn't expose the layout machinery
    /// necessary to build the stride view.
    ///
    /// Returns a copy of the tensor with all complete windows of size `size` in dimension `dim`;
    /// where windows are advanced by `step` at each index.
    ///
    /// The number of windows is `max(0, (shape[dim] - size).ceil_div(step))`.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor to unfold; of shape ``[pre=..., dim shape, post=...]``
    /// * `dim` - the dimension to unfold.
    /// * `size` - the size of each unfolded window.
    /// * `step` - the step between each window.
    ///
    /// # Returns
    ///
    /// A tensor view with shape ``[pre=..., windows, post=..., size]``.
    #[allow(unused)]
    pub(crate) fn unfold(
        tensor: SharedArray<E>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> SharedArray<E> {
        let result_shape = calculate_unfold_shape(tensor.shape(), dim, size, step);
        let windows = result_shape[dim];

        let mut slices = vec![Slice::new(0, None, 1); tensor.shape().len()];
        let new_axis = slices.len();

        let mut stack = Vec::with_capacity(windows);
        for widx in 0..windows {
            let start = widx * step;
            let end = start + size;
            slices[dim] = Slice::new(start as isize, Some(end as isize), 1);

            let mut window_slice =
                tensor.slice(Self::to_slice_args_with_steps(&slices, slices.len()).as_slice());
            window_slice.insert_axis_inplace(Axis(new_axis));
            window_slice.swap_axes(dim, new_axis);

            stack.push(window_slice);
        }
        Self::concatenate(&stack, dim)
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
                DType::QFloat(strategy) => match strategy.value {
                    QuantValue::Q8F | QuantValue::Q8S => try_binary_simd::<$elem, $elem, i8, i8, $op>($lhs, $rhs),
                    _ => Err(($lhs, $rhs)),
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
                DType::QFloat(strategy) => match strategy.value {
                    QuantValue::Q8F | QuantValue::Q8S => try_binary_scalar_simd::<$elem, $elem, i8, i8, $op>($lhs, $rhs),
                    QuantValue::Q4F | QuantValue::Q4S | QuantValue::Q2F | QuantValue::Q2S => Err($lhs)
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
                DType::QFloat(strategy) => match strategy.value {
                    QuantValue::Q8F | QuantValue::Q8S => try_cmp_simd::<$elem, i8, $op>($lhs, $rhs),
                    QuantValue::Q4F | QuantValue::Q4S | QuantValue::Q2F | QuantValue::Q2S => Err(($lhs, $rhs))
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
                DType::QFloat(strategy) => match strategy.value {
                    QuantValue::Q8F | QuantValue::Q8S => try_cmp_scalar_simd::<$elem, i8, $op>($lhs, $rhs),
                    QuantValue::Q4F | QuantValue::Q4S | QuantValue::Q2F | QuantValue::Q2S => Err($lhs)
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

// Helper function to broadcast two tensors to a common shape for comparison operations
// Returns broadcasted views that can be safely zipped
fn broadcast_for_comparison<'a, E: Copy, S1, S2>(
    lhs: &'a ndarray::ArrayBase<S1, ndarray::IxDyn>,
    rhs: &'a ndarray::ArrayBase<S2, ndarray::IxDyn>,
) -> (
    ndarray::ArrayView<'a, E, ndarray::IxDyn>,
    ndarray::ArrayView<'a, E, ndarray::IxDyn>,
)
where
    S1: ndarray::Data<Elem = E>,
    S2: ndarray::Data<Elem = E>,
{
    // Get shapes
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();

    // Compute broadcast shape using ndarray's broadcast compatibility rules
    let ndims = lhs_shape.len().max(rhs_shape.len());
    let mut broadcast_shape = vec![1; ndims];

    for i in 0..ndims {
        let lhs_dim = if i < lhs_shape.len() {
            lhs_shape[lhs_shape.len() - 1 - i]
        } else {
            1
        };
        let rhs_dim = if i < rhs_shape.len() {
            rhs_shape[rhs_shape.len() - 1 - i]
        } else {
            1
        };

        if lhs_dim == rhs_dim {
            broadcast_shape[ndims - 1 - i] = lhs_dim;
        } else if lhs_dim == 1 {
            broadcast_shape[ndims - 1 - i] = rhs_dim;
        } else if rhs_dim == 1 {
            broadcast_shape[ndims - 1 - i] = lhs_dim;
        } else {
            panic!(
                "Incompatible shapes for broadcasting: {:?} and {:?}",
                lhs_shape, rhs_shape
            );
        }
    }

    // Create IxDyn from broadcast shape
    let broadcast_dim = ndarray::IxDyn(&broadcast_shape);

    // Broadcast both arrays
    let lhs_broadcast = lhs
        .broadcast(broadcast_dim.clone())
        .expect("Failed to broadcast lhs");
    let rhs_broadcast = rhs
        .broadcast(broadcast_dim)
        .expect("Failed to broadcast rhs");

    (lhs_broadcast, rhs_broadcast)
}

impl<E> NdArrayMathOps<E>
where
    E: Copy + NdArrayElement,
{
    pub fn add(lhs: SharedArray<E>, rhs: SharedArray<E>) -> SharedArray<E> {
        let (lhs, rhs) = dispatch_binary_simd!(
            E, VecAdd, lhs, rhs, u8, i8, u16, i16, u32, i32, f32, u64, i64, f64
        );

        let array = &lhs + &rhs;
        array.into_shared()
    }

    pub fn add_scalar(lhs: SharedArray<E>, rhs: E) -> SharedArray<E> {
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

        let array = lhs + rhs;
        array.into_shared()
    }

    pub fn sub(lhs: SharedArray<E>, rhs: SharedArray<E>) -> SharedArray<E> {
        let (lhs, rhs) = dispatch_binary_simd!(
            E, VecSub, lhs, rhs, u8, i8, u16, i16, u32, i32, f32, u64, i64, f64
        );

        let array = lhs - rhs;
        array.into_shared()
    }

    pub fn sub_scalar(lhs: SharedArray<E>, rhs: E) -> SharedArray<E> {
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

        let array = lhs - rhs;
        array.into_shared()
    }

    pub fn mul(lhs: SharedArray<E>, rhs: SharedArray<E>) -> SharedArray<E> {
        let (lhs, rhs) =
            dispatch_binary_simd!(noq, E, VecMul, lhs, rhs, u16, i16, u32, i32, f32, f64);

        let array = lhs * rhs;
        array.into_shared()
    }

    pub fn mul_scalar(lhs: SharedArray<E>, rhs: E) -> SharedArray<E> {
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

        let array = lhs * rhs;
        array.into_shared()
    }

    pub fn div(lhs: SharedArray<E>, rhs: SharedArray<E>) -> SharedArray<E> {
        let (lhs, rhs) = dispatch_binary_simd!(noq, E, VecDiv, lhs, rhs, f32, f64);

        let array = lhs / rhs;
        array.into_shared()
    }

    pub fn div_scalar(lhs: SharedArray<E>, rhs: E) -> SharedArray<E> {
        let lhs = dispatch_binary_scalar_simd!(noq, E, VecDiv, lhs, rhs.elem(), f32, f64);

        let array = lhs / rhs;
        array.into_shared()
    }

    pub fn remainder(lhs: SharedArray<E>, rhs: SharedArray<E>) -> SharedArray<E> {
        let array =
            lhs.clone() - (lhs / rhs.clone()).mapv_into(|a| (a.to_f64()).floor().elem()) * rhs;
        array.into_shared()
    }

    pub fn remainder_scalar(lhs: SharedArray<E>, rhs: E) -> SharedArray<E>
    where
        E: core::ops::Rem<Output = E>,
    {
        let array = lhs.mapv(|x| ((x % rhs) + rhs) % rhs);
        array.into_shared()
    }

    pub fn recip(tensor: SharedArray<E>) -> SharedArray<E> {
        let tensor = dispatch_unary_simd!(E, RecipVec, tensor, f32);

        let array = tensor.map(|x| 1.elem::<E>() / *x);
        array.into_shared()
    }

    pub fn mean(tensor: SharedArray<E>) -> SharedArray<E> {
        let mean = tensor.mean().unwrap();
        ArrayD::from_elem(IxDyn(&[1]), mean).into_shared()
    }

    pub fn sum(tensor: SharedArray<E>) -> SharedArray<E> {
        let sum = tensor.sum();
        ArrayD::from_elem(IxDyn(&[1]), sum).into_shared()
    }

    pub fn prod(tensor: SharedArray<E>) -> SharedArray<E> {
        let prod = tensor.product();
        ArrayD::from_elem(IxDyn(&[1]), prod).into_shared()
    }

    pub fn mean_dim(tensor: SharedArray<E>, dim: usize) -> SharedArray<E> {
        let ndims = tensor.shape().num_dims();
        match ndims {
            d if (1..=6).contains(&d) => keepdim!(dim, tensor, mean),
            _ => panic!("Dim not supported {ndims}"),
        }
    }

    pub fn sum_dim(tensor: SharedArray<E>, dim: usize) -> SharedArray<E> {
        let ndims = tensor.shape().num_dims();
        match ndims {
            d if (1..=6).contains(&d) => keepdim!(dim, tensor, sum),
            _ => panic!("Dim not supported {ndims}"),
        }
    }

    pub fn prod_dim(tensor: SharedArray<E>, dim: usize) -> SharedArray<E> {
        let ndims = tensor.shape().num_dims();
        match ndims {
            d if (1..=6).contains(&d) => keepdim!(dim, tensor, prod),
            _ => panic!("Dim not supported {ndims}"),
        }
    }

    pub fn cumsum(tensor: SharedArray<E>, dim: usize) -> SharedArray<E> {
        cumsum_dim(tensor, dim)
    }

    pub fn cumprod(tensor: SharedArray<E>, dim: usize) -> SharedArray<E> {
        cumprod_dim(tensor, dim)
    }

    pub fn cummin(tensor: SharedArray<E>, dim: usize) -> SharedArray<E> {
        cummin_dim(tensor, dim)
    }

    pub fn cummax(tensor: SharedArray<E>, dim: usize) -> SharedArray<E> {
        cummax_dim(tensor, dim)
    }

    pub fn gather<I: NdArrayElement>(
        dim: usize,
        mut tensor: SharedArray<E>,
        mut indices: SharedArray<I>,
    ) -> SharedArray<E> {
        let ndims = tensor.shape().num_dims();
        if dim != ndims - 1 {
            tensor.swap_axes(ndims - 1, dim);
            indices.swap_axes(ndims - 1, dim);
        }
        let (shape_tensor, shape_indices) = (tensor.shape(), indices.shape().into_shape());
        let (size_tensor, size_index) = (shape_tensor[ndims - 1], shape_indices.dims[ndims - 1]);
        let batch_size = Self::gather_batch_size(shape_tensor, &shape_indices.dims);

        let indices = NdArrayOps::reshape(indices, Shape::new([batch_size, size_index]));
        let tensor = NdArrayOps::reshape(tensor, Shape::new([batch_size, size_tensor]));
        let mut output = Array2::zeros((batch_size, size_index));

        for b in 0..batch_size {
            let indices = indices.slice(s!(b, ..));
            for (i, index) in indices.iter().enumerate() {
                output[[b, i]] = tensor[[b, index.elem::<i64>() as usize]];
            }
        }

        let mut output = NdArrayOps::reshape(output.into_shared().into_dyn(), shape_indices);

        if dim != ndims - 1 {
            output.swap_axes(ndims - 1, dim);
        }

        output
    }

    pub fn scatter<I: NdArrayElement>(
        dim: usize,
        mut tensor: SharedArray<E>,
        mut indices: SharedArray<I>,
        mut value: SharedArray<E>,
    ) -> SharedArray<E> {
        let ndims = tensor.shape().num_dims();
        if dim != ndims - 1 {
            tensor.swap_axes(ndims - 1, dim);
            indices.swap_axes(ndims - 1, dim);
            value.swap_axes(ndims - 1, dim);
        }

        let (shape_tensor, shape_indices, shape_value) =
            (tensor.shape().into_shape(), indices.shape(), value.shape());
        let (size_tensor, size_index, size_value) = (
            shape_tensor.dims[ndims - 1],
            shape_indices[ndims - 1],
            shape_value[ndims - 1],
        );
        let batch_size = Self::gather_batch_size(&shape_tensor.dims, shape_indices);

        if shape_value != shape_indices {
            panic!(
                "Invalid dimension: the shape of the index tensor should be the same as the value \
                 tensor: Index {:?} value {:?}",
                shape_indices, shape_value
            );
        }

        let indices = NdArrayOps::reshape(indices, Shape::new([batch_size, size_index]));
        let value = NdArrayOps::reshape(value, Shape::new([batch_size, size_value]));
        let mut tensor = NdArrayOps::reshape(tensor, Shape::new([batch_size, size_tensor]));

        for b in 0..batch_size {
            let indices = indices.slice(s!(b, ..));

            for (i, index) in indices.iter().enumerate() {
                let index = index.elem::<i64>() as usize;
                tensor[[b, index]] += value[[b, i]];
            }
        }

        let mut output = NdArrayOps::reshape(tensor.into_shared().into_dyn(), shape_tensor);
        if dim != ndims - 1 {
            output.swap_axes(ndims - 1, dim);
        }
        output
    }

    pub fn mask_where(
        tensor: SharedArray<E>,
        mask: SharedArray<bool>,
        source: SharedArray<E>,
    ) -> SharedArray<E> {
        let tensor = tensor.broadcast(mask.dim()).unwrap();
        let source = source.broadcast(mask.dim()).unwrap();
        Zip::from(&tensor)
            .and(&mask)
            .and(&source)
            .map_collect(|&x, &mask_val, &y| if mask_val { y } else { x })
            .into_shared()
    }

    pub fn mask_fill(tensor: SharedArray<E>, mask: SharedArray<bool>, value: E) -> SharedArray<E> {
        let mut output = tensor.clone();
        let broadcast_mask = mask.broadcast(output.dim()).unwrap();
        Zip::from(&mut output)
            .and(&broadcast_mask)
            .for_each(|out, &mask_val| {
                if mask_val {
                    *out = value;
                }
            });
        output.into_shared()
    }

    fn gather_batch_size(shape_tensor: &[usize], shape_indices: &[usize]) -> usize {
        let ndims = shape_tensor.num_dims();
        let mut batch_size = 1;

        for i in 0..ndims - 1 {
            if shape_tensor[i] != shape_indices[i] {
                panic!(
                    "Unsupported dimension, only the last dimension can differ: Tensor {:?} Index \
                     {:?}",
                    shape_tensor, shape_indices
                );
            }
            batch_size *= shape_indices[i];
        }

        batch_size
    }

    pub fn select<I: NdArrayElement>(
        tensor: SharedArray<E>,
        dim: usize,
        indices: SharedArray<I>,
    ) -> SharedArray<E> {
        let array = tensor.select(
            Axis(dim),
            &indices
                .into_iter()
                .map(|i| i.elem::<i64>() as usize)
                .collect::<Vec<_>>(),
        );

        array.into_shared()
    }

    pub fn select_assign<I: NdArrayElement>(
        tensor: SharedArray<E>,
        dim: usize,
        indices: SharedArray<I>,
        value: SharedArray<E>,
    ) -> SharedArray<E> {
        let mut output_array = tensor.into_owned();

        for (index_value, index) in indices.into_iter().enumerate() {
            let mut view = output_array.index_axis_mut(Axis(dim), index.elem::<i64>() as usize);
            let value = value.index_axis(Axis(dim), index_value);

            view.zip_mut_with(&value, |a, b| *a += *b);
        }

        output_array.into_shared()
    }
    pub fn argmax<I: NdArrayElement>(tensor: SharedArray<E>, dim: usize) -> SharedArray<I> {
        arg(tensor, dim, CmpType::Max)
    }

    pub fn argmin<I: NdArrayElement>(tensor: SharedArray<E>, dim: usize) -> SharedArray<I> {
        arg(tensor, dim, CmpType::Min)
    }

    pub fn clamp_min(tensor: SharedArray<E>, min: E) -> SharedArray<E> {
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

        tensor.mapv_inplace(|x| match x < min {
            true => min,
            false => x,
        });

        tensor
    }

    pub fn clamp_max(tensor: SharedArray<E>, max: E) -> SharedArray<E> {
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

        tensor.mapv_inplace(|x| match x > max {
            true => max,
            false => x,
        });

        tensor
    }

    pub fn clamp(tensor: SharedArray<E>, min: E, max: E) -> SharedArray<E> {
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

        tensor.mapv_inplace(|x| match x < min {
            true => min,
            false => match x > max {
                true => max,
                false => x,
            },
        });

        tensor
    }

    pub(crate) fn elementwise_op<OtherE>(
        lhs: SharedArray<E>,
        rhs: SharedArray<OtherE>,
        var_name: impl FnMut(&E, &OtherE) -> E,
    ) -> SharedArray<E> {
        let lhs = lhs.broadcast(rhs.dim()).unwrap_or(lhs.view());
        let rhs = rhs.broadcast(lhs.dim()).unwrap_or(rhs.view());

        Zip::from(lhs).and(rhs).map_collect(var_name).into_shared()
    }

    pub(crate) fn elementwise_op_scalar(
        lhs: SharedArray<E>,
        var_name: impl FnMut(E) -> E,
    ) -> SharedArray<E> {
        lhs.mapv(var_name).into_shared()
    }

    pub(crate) fn sign_op(tensor: SharedArray<E>) -> SharedArray<E>
    where
        E: Signed,
    {
        let zero = 0.elem();
        let one = 1.elem::<E>();

        tensor
            .mapv(|x| {
                if x > zero {
                    one
                } else if x < zero {
                    -one
                } else {
                    zero
                }
            })
            .into_shared()
    }

    pub(crate) fn abs(tensor: SharedArray<E>) -> SharedArray<E> {
        let tensor = dispatch_unary_simd!(E, VecAbs, tensor, i8, i16, i32, f32, f64);

        tensor.mapv_into(|a| a.abs_elem()).into_shared()
    }

    pub(crate) fn equal(lhs: SharedArray<E>, rhs: SharedArray<E>) -> SharedArray<bool> {
        let (lhs, rhs) = dispatch_cmp_simd!(
            E, VecEquals, lhs, rhs, u8, i8, u16, i16, u32, f32, i32, u64, i64, f64
        );

        // Use the helper to broadcast both arrays to a common shape
        let (lhs_broadcast, rhs_broadcast) = broadcast_for_comparison(&lhs, &rhs);
        // Now we can safely zip and compare
        Zip::from(&lhs_broadcast)
            .and(&rhs_broadcast)
            .map_collect(|&lhs, &rhs| lhs == rhs)
            .into_shared()
    }

    pub(crate) fn equal_elem(lhs: SharedArray<E>, rhs: E) -> SharedArray<bool> {
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

        lhs.mapv(|a| a == rhs).into_shared()
    }

    pub(crate) fn greater(lhs: SharedArray<E>, rhs: SharedArray<E>) -> SharedArray<bool> {
        let (lhs, rhs) = dispatch_cmp_simd!(
            E, VecGreater, lhs, rhs, u8, i8, u16, i16, u32, f32, i32, u64, i64, f64
        );

        // Use the helper to broadcast both arrays to a common shape
        let (lhs_broadcast, rhs_broadcast) = broadcast_for_comparison(&lhs, &rhs);
        // Now we can safely zip and compare
        Zip::from(&lhs_broadcast)
            .and(&rhs_broadcast)
            .map_collect(|&lhs, &rhs| lhs > rhs)
            .into_shared()
    }

    pub(crate) fn greater_elem(lhs: SharedArray<E>, rhs: E) -> SharedArray<bool> {
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

        lhs.mapv(|a| a > rhs).into_shared()
    }

    pub(crate) fn greater_equal(lhs: SharedArray<E>, rhs: SharedArray<E>) -> SharedArray<bool> {
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

        // Use the helper to broadcast both arrays to a common shape
        let (lhs_broadcast, rhs_broadcast) = broadcast_for_comparison(&lhs, &rhs);
        // Now we can safely zip and compare
        Zip::from(&lhs_broadcast)
            .and(&rhs_broadcast)
            .map_collect(|&lhs, &rhs| lhs >= rhs)
            .into_shared()
    }

    pub(crate) fn greater_equal_elem(lhs: SharedArray<E>, rhs: E) -> SharedArray<bool> {
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

        lhs.mapv(|a| a >= rhs).into_shared()
    }

    pub(crate) fn lower_equal(lhs: SharedArray<E>, rhs: SharedArray<E>) -> SharedArray<bool> {
        let (lhs, rhs) = dispatch_cmp_simd!(
            E, VecLowerEq, lhs, rhs, u8, i8, u16, i16, u32, f32, i32, u64, i64, f64
        );

        // Use the helper to broadcast both arrays to a common shape
        let (lhs_broadcast, rhs_broadcast) = broadcast_for_comparison(&lhs, &rhs);
        // Now we can safely zip and compare
        Zip::from(&lhs_broadcast)
            .and(&rhs_broadcast)
            .map_collect(|&lhs, &rhs| lhs <= rhs)
            .into_shared()
    }

    pub(crate) fn lower_equal_elem(lhs: SharedArray<E>, rhs: E) -> SharedArray<bool> {
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

        lhs.mapv(|a| a <= rhs).into_shared()
    }

    pub(crate) fn lower(lhs: SharedArray<E>, rhs: SharedArray<E>) -> SharedArray<bool> {
        let (lhs, rhs) = dispatch_cmp_simd!(
            E, VecLower, lhs, rhs, u8, i8, u16, i16, u32, f32, i32, u64, i64, f64
        );

        // Use the helper to broadcast both arrays to a common shape
        let (lhs_broadcast, rhs_broadcast) = broadcast_for_comparison(&lhs, &rhs);

        // Now we can safely zip and compare
        Zip::from(&lhs_broadcast)
            .and(&rhs_broadcast)
            .map_collect(|&lhs, &rhs| lhs < rhs)
            .into_shared()
    }

    pub(crate) fn lower_elem(lhs: SharedArray<E>, rhs: E) -> SharedArray<bool> {
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

        lhs.mapv(|a| a < rhs).into_shared()
    }
}

pub struct NdArrayBitOps<I: IntNdArrayElement>(PhantomData<I>);

impl<I: IntNdArrayElement> NdArrayBitOps<I> {
    pub(crate) fn bitand(lhs: SharedArray<I>, rhs: SharedArray<I>) -> SharedArray<I> {
        let (lhs, rhs) =
            dispatch_binary_simd!(I, VecBitAnd, lhs, rhs, i8, u8, i16, u16, i32, u32, i64, u64);

        NdArrayMathOps::elementwise_op(lhs, rhs, |a: &I, b: &I| {
            (a.elem::<i64>() & (b.elem::<i64>())).elem()
        })
    }

    pub(crate) fn bitand_scalar(lhs: SharedArray<I>, rhs: I) -> SharedArray<I> {
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

    pub(crate) fn bitor(lhs: SharedArray<I>, rhs: SharedArray<I>) -> SharedArray<I> {
        let (lhs, rhs) =
            dispatch_binary_simd!(I, VecBitOr, lhs, rhs, i8, u8, i16, u16, i32, u32, i64, u64);

        NdArrayMathOps::elementwise_op(lhs, rhs, |a: &I, b: &I| {
            (a.elem::<i64>() | (b.elem::<i64>())).elem()
        })
    }

    pub(crate) fn bitor_scalar(lhs: SharedArray<I>, rhs: I) -> SharedArray<I> {
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

    pub(crate) fn bitxor(lhs: SharedArray<I>, rhs: SharedArray<I>) -> SharedArray<I> {
        let (lhs, rhs) =
            dispatch_binary_simd!(I, VecBitXor, lhs, rhs, i8, u8, i16, u16, i32, u32, i64, u64);

        NdArrayMathOps::elementwise_op(lhs, rhs, |a: &I, b: &I| {
            (a.elem::<i64>() ^ (b.elem::<i64>())).elem()
        })
    }

    pub(crate) fn bitxor_scalar(lhs: SharedArray<I>, rhs: I) -> SharedArray<I> {
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

    pub(crate) fn bitnot(tensor: SharedArray<I>) -> SharedArray<I> {
        let tensor =
            dispatch_unary_simd!(I, VecBitNot, tensor, i8, u8, i16, u16, i32, u32, i64, u64);

        NdArrayMathOps::elementwise_op_scalar(tensor, |a: I| (!a.elem::<i64>()).elem())
    }
}

pub struct NdArrayBoolOps;

// Rust booleans are either `00000000` or `00000001`, so bitwise and/or is fine, but bitwise not would
// produce invalid values.
impl NdArrayBoolOps {
    pub(crate) fn equal(lhs: SharedArray<bool>, rhs: SharedArray<bool>) -> SharedArray<bool> {
        #[cfg(feature = "simd")]
        let (lhs, rhs) = match try_cmp_simd::<bool, u8, VecEquals>(lhs, rhs) {
            Ok(out) => return out,
            Err(args) => args,
        };

        // Use the helper to broadcast both arrays to a common shape
        let (lhs_broadcast, rhs_broadcast) = broadcast_for_comparison(&lhs, &rhs);
        // Now we can safely zip and compare
        Zip::from(&lhs_broadcast)
            .and(&rhs_broadcast)
            .map_collect(|&lhs, &rhs| lhs == rhs)
            .into_shared()
    }

    pub(crate) fn and(lhs: SharedArray<bool>, rhs: SharedArray<bool>) -> SharedArray<bool> {
        #[cfg(feature = "simd")]
        let (lhs, rhs) = match try_binary_simd::<bool, bool, u8, u8, VecBitAnd>(lhs, rhs) {
            Ok(out) => return out,
            Err(args) => args,
        };

        // Use the helper to broadcast both arrays to a common shape
        let (lhs_broadcast, rhs_broadcast) = broadcast_for_comparison(&lhs, &rhs);
        // Now we can safely zip and compare
        Zip::from(&lhs_broadcast)
            .and(&rhs_broadcast)
            .map_collect(|&lhs, &rhs| lhs && rhs)
            .into_shared()
    }

    pub(crate) fn or(lhs: SharedArray<bool>, rhs: SharedArray<bool>) -> SharedArray<bool> {
        #[cfg(feature = "simd")]
        let (lhs, rhs) = match try_binary_simd::<bool, bool, u8, u8, VecBitOr>(lhs, rhs) {
            Ok(out) => return out,
            Err(args) => args,
        };

        // Use the helper to broadcast both arrays to a common shape
        let (lhs_broadcast, rhs_broadcast) = broadcast_for_comparison(&lhs, &rhs);
        // Now we can safely zip and compare
        Zip::from(&lhs_broadcast)
            .and(&rhs_broadcast)
            .map_collect(|&lhs, &rhs| lhs || rhs)
            .into_shared()
    }
}

enum CmpType {
    Min,
    Max,
}

fn arg<E: NdArrayElement, I: NdArrayElement>(
    tensor: SharedArray<E>,
    dim: usize,
    cmp: CmpType,
) -> SharedArray<I> {
    let mut reshape = tensor.shape().to_vec();
    reshape[dim] = 1;

    let output = tensor.map_axis(Axis(dim), |arr| {
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

    output.into_shared()
}

#[cfg(test)]
mod tests {
    use burn_tensor::TensorData;

    use crate::NdArrayTensor;

    use super::*;

    #[test]
    fn should_generate_row_major_layout_for_cat() {
        let expected_shape: &[usize] = &[4, 6, 2];
        let expected_strides: &[isize] = &[12, 2, 1];
        let NdArrayTensor::I32(expected_array) = NdArrayTensor::from_data(TensorData::from([
            [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0]],
            [[7, 0], [8, 0], [9, 0], [10, 0], [11, 0], [12, 0]],
            [[13, 0], [14, 0], [15, 0], [16, 0], [17, 0], [18, 0]],
            [[19, 0], [20, 0], [21, 0], [22, 0], [23, 0], [24, 0]],
        ])) else {
            panic!()
        };

        let NdArrayTensor::I32(tensor) = NdArrayTensor::from_data(TensorData::from([
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18],
            [19, 20, 21, 22, 23, 24],
        ])) else {
            panic!()
        };

        // unsqueeze dim on the outermost axis
        let array = NdArrayOps::reshape(tensor, Shape::from([4, 6, 1]));
        let NdArrayTensor::I32(zeros) =
            NdArrayTensor::from_data(TensorData::zeros::<i32, _>([4, 6, 1]))
        else {
            panic!()
        };
        // make `ndarray` concatenates array on the outermost axis
        let array = NdArrayOps::cat([array, zeros].to_vec(), 2);

        assert!(array.is_standard_layout());
        assert_eq!(array.shape(), expected_shape);
        assert_eq!(array.strides(), expected_strides);
        assert_eq!(
            array.into_iter().collect::<Vec<_>>(),
            expected_array.into_iter().collect::<Vec<_>>(),
        );
    }
}
