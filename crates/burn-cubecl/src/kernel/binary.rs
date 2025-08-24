use std::marker::PhantomData;

use crate::{
    CubeRuntime,
    element::CubeElement,
    kernel::utils::{linear_tensor, linear_tensor_alias},
    ops::{into_data_sync, max_line_size, numeric::empty_device},
    tensor::CubeTensor,
};
use burn_tensor::Shape;
use cubecl::{
    calculate_cube_count_elemwise,
    prelude::*,
    std::tensor::{
        index_offset_with_layout, layout::linear::LinearTensorView, r#virtual::ReadWrite,
    },
    tensor_line_size_parallel,
};

pub(crate) trait BinaryOpFamily: Send + Sync + 'static {
    type BinaryOp<C: Numeric>: BinaryOp<C>;
}

#[cube]
pub(crate) trait BinaryOp<C: Numeric>: 'static + Send + Sync {
    /// Execute a binary operation.
    fn execute(lhs: Line<C>, rhs: Line<C>) -> Line<C>;
}

pub(crate) struct AddOp;
pub(crate) struct SubOp;
pub(crate) struct MulOp;
pub(crate) struct DivOp;
pub(crate) struct RemainderOp;
pub(crate) struct AndOp;
pub(crate) struct OrOp;

/// Since Powf only works on float, but we still want to implement the numeric binary op family, we
/// set another precision in the family type to cast, when necessary, the input value to a valid
/// float.
///
/// Because of this we won't benefit from the cubecl rust compilation speed improvement from using
/// the family pattern for [PowOp], but at least we don't duplicate code.
pub(crate) struct PowOp<F: Float> {
    _f: PhantomData<F>,
}

impl BinaryOpFamily for AddOp {
    type BinaryOp<C: Numeric> = Self;
}

impl BinaryOpFamily for SubOp {
    type BinaryOp<C: Numeric> = Self;
}

impl BinaryOpFamily for MulOp {
    type BinaryOp<C: Numeric> = Self;
}

impl BinaryOpFamily for DivOp {
    type BinaryOp<C: Numeric> = Self;
}

impl BinaryOpFamily for RemainderOp {
    type BinaryOp<C: Numeric> = Self;
}

impl<F: Float> BinaryOpFamily for PowOp<F> {
    type BinaryOp<C: Numeric> = Self;
}

impl BinaryOpFamily for AndOp {
    type BinaryOp<C: Numeric> = Self;
}

impl BinaryOpFamily for OrOp {
    type BinaryOp<C: Numeric> = Self;
}

#[cube]
impl<N: Numeric> BinaryOp<N> for AddOp {
    fn execute(lhs: Line<N>, rhs: Line<N>) -> Line<N> {
        lhs + rhs
    }
}

#[cube]
impl<N: Numeric> BinaryOp<N> for SubOp {
    fn execute(lhs: Line<N>, rhs: Line<N>) -> Line<N> {
        lhs - rhs
    }
}

#[cube]
impl<N: Numeric> BinaryOp<N> for MulOp {
    fn execute(lhs: Line<N>, rhs: Line<N>) -> Line<N> {
        lhs * rhs
    }
}

#[cube]
impl<N: Numeric> BinaryOp<N> for DivOp {
    fn execute(lhs: Line<N>, rhs: Line<N>) -> Line<N> {
        lhs / rhs
    }
}

#[cube]
impl<N: Numeric> BinaryOp<N> for RemainderOp {
    fn execute(lhs: Line<N>, rhs: Line<N>) -> Line<N> {
        Line::rem(lhs, rhs)
    }
}

#[cube]
impl<N: Numeric, F: Float> BinaryOp<N> for PowOp<F> {
    fn execute(lhs: Line<N>, rhs: Line<N>) -> Line<N> {
        let lhs = Line::<F>::cast_from(lhs);
        let rhs = Line::<F>::cast_from(rhs);
        let out = Line::powf(lhs, rhs);

        Line::cast_from(out)
    }
}

#[cube]
impl<N: Numeric> BinaryOp<N> for AndOp {
    fn execute(lhs: Line<N>, rhs: Line<N>) -> Line<N> {
        Line::cast_from(Line::<bool>::cast_from(lhs).and(Line::<bool>::cast_from(rhs)))
    }
}

#[cube]
impl<N: Numeric> BinaryOp<N> for OrOp {
    fn execute(lhs: Line<N>, rhs: Line<N>) -> Line<N> {
        Line::cast_from(Line::<bool>::cast_from(lhs).or(Line::<bool>::cast_from(rhs)))
    }
}

#[cube(launch_unchecked)]
pub(crate) fn kernel_scalar_binop<C: Numeric, O: BinaryOpFamily>(
    input: &LinearTensorView<C>,
    scalar: C,
    output: &mut LinearTensorView<C, ReadWrite>,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    output[ABSOLUTE_POS] = O::BinaryOp::<C>::execute(input[ABSOLUTE_POS], Line::new(scalar));
}

#[cube(launch_unchecked)]
pub(crate) fn kernel_binop<C: Numeric, O: BinaryOpFamily>(
    lhs: &Tensor<Line<C>>,
    rhs: &Tensor<Line<C>>,
    out: &mut Tensor<Line<C>>,
    #[comptime] rank: Option<u32>,
    #[comptime] to_contiguous_lhs: bool,
    #[comptime] to_contiguous_rhs: bool,
) {
    let offset_out = ABSOLUTE_POS;
    let mut offset_lhs = ABSOLUTE_POS;
    let mut offset_rhs = ABSOLUTE_POS;

    if offset_out >= out.len() {
        terminate!();
    }

    if to_contiguous_lhs {
        offset_lhs = index_offset_with_layout::<C, C>(
            lhs,
            out,
            offset_out,
            0,
            rank.unwrap_or_else(|| out.rank()),
            rank.is_some(),
        );
    }

    if to_contiguous_rhs {
        offset_rhs = index_offset_with_layout::<C, C>(
            rhs,
            out,
            offset_out,
            0,
            rank.unwrap_or_else(|| out.rank()),
            rank.is_some(),
        );
    }

    out[offset_out] = O::BinaryOp::<C>::execute(lhs[offset_lhs], rhs[offset_rhs]);
}

pub(crate) fn launch_binop<R: CubeRuntime, E: CubeElement, O: BinaryOpFamily>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
) -> CubeTensor<R> {
    let ndims = lhs.shape.num_dims();
    let line_size_lhs = tensor_line_size_parallel(
        R::line_size_elem(&E::as_elem_native_unchecked()),
        &lhs.shape.dims,
        &lhs.strides,
        ndims - 1,
    );
    let line_size_rhs = tensor_line_size_parallel(
        R::line_size_elem(&E::as_elem_native_unchecked()),
        &rhs.shape.dims,
        &rhs.strides,
        ndims - 1,
    );
    let line_size = Ord::min(line_size_lhs, line_size_rhs);

    let mut shape_out = vec![0; ndims];
    lhs.shape
        .dims
        .iter()
        .zip(rhs.shape.dims.iter())
        .enumerate()
        .for_each(|(index, (dim_lhs, dim_rhs))| {
            shape_out[index] = usize::max(*dim_lhs, *dim_rhs);
        });

    let shape_out = Shape::from(shape_out);
    let client = lhs.client.clone();
    let num_elems = shape_out.num_elements();

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems / line_size as usize, cube_dim);

    unsafe {
        // Only re-used lhs/rhs if contiguous for now, strided indices are not correctly accounted for
        // and optimizations in fusion do not have this issue anyway
        if lhs.can_mut_broadcast(&rhs) && lhs.is_contiguous() {
            kernel_binop::launch_unchecked::<E, O, R>(
                &client,
                cube_count,
                cube_dim,
                lhs.as_tensor_arg::<E>(line_size),
                rhs.as_tensor_arg::<E>(line_size),
                TensorArg::alias(0),
                None,
                false,
                rhs.strides != lhs.strides || rhs.shape != lhs.shape,
            );

            lhs
        } else if rhs.can_mut_broadcast(&lhs) && rhs.is_contiguous() {
            kernel_binop::launch_unchecked::<E, O, R>(
                &client,
                cube_count,
                cube_dim,
                lhs.as_tensor_arg::<E>(line_size),
                rhs.as_tensor_arg::<E>(line_size),
                TensorArg::alias(1),
                None,
                rhs.strides != lhs.strides || rhs.shape != lhs.shape,
                false,
            );

            rhs
        } else {
            let output = empty_device::<R, E>(lhs.client.clone(), lhs.device.clone(), shape_out);
            let to_contiguous_lhs = lhs.strides != output.strides || lhs.shape != output.shape;
            let to_contiguous_rhs = rhs.strides != output.strides || rhs.shape != output.shape;

            kernel_binop::launch_unchecked::<E, O, R>(
                &client,
                cube_count,
                cube_dim,
                lhs.as_tensor_arg::<E>(line_size),
                rhs.as_tensor_arg::<E>(line_size),
                output.as_tensor_arg::<E>(line_size),
                None,
                to_contiguous_lhs,
                to_contiguous_rhs,
            );

            output
        }
    }
}

pub(crate) fn launch_scalar_binop<R: CubeRuntime, E: CubeElement, O: BinaryOpFamily>(
    tensor: CubeTensor<R>,
    scalar: E,
) -> CubeTensor<R> {
    // Vectorization is only enabled when the last dimension is contiguous.
    let line_size = max_line_size(&tensor);
    let client = tensor.client.clone();
    let num_elems = tensor.shape.num_elements();

    println!(
        "scalar binop input: {}",
        into_data_sync::<R, E>(tensor.clone())
    );
    println!("scalar binop scalar: {}", scalar);

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems / line_size as usize, cube_dim);

    let output = unsafe {
        if tensor.can_mut() {
            kernel_scalar_binop::launch_unchecked::<E, O, R>(
                &client,
                cube_count,
                cube_dim,
                linear_tensor(&tensor, &line_size),
                ScalarArg::new(scalar),
                linear_tensor_alias(&tensor, &line_size, 0),
            );

            tensor
        } else {
            let output = empty_device::<R, E>(
                tensor.client.clone(),
                tensor.device.clone(),
                tensor.shape.clone(),
            );

            kernel_scalar_binop::launch_unchecked::<E, O, R>(
                &client,
                cube_count,
                CubeDim::default(),
                linear_tensor(&tensor, &line_size),
                ScalarArg::new(scalar),
                linear_tensor(&output, &line_size),
            );

            output
        }
    };
    println!(
        "scalar binop output: {}",
        into_data_sync::<R, E>(output.clone())
    );
    output
}
