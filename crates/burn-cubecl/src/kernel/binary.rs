use std::marker::PhantomData;

use crate::{
    CubeRuntime,
    element::CubeElement,
    kernel::utils::{broadcast_shape, linear_view, linear_view_alias, linear_view_ref},
    ops::{max_line_size, numeric::empty_device},
    tensor::CubeTensor,
};
use cubecl::{calculate_cube_count_elemwise, prelude::*, std::tensor::layout::linear::LinearView};

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
    input: &LinearView<Line<C>>,
    scalar: C,
    output: &mut LinearView<Line<C>, ReadWrite>,
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    output[ABSOLUTE_POS] = O::BinaryOp::<C>::execute(input[ABSOLUTE_POS], Line::new(scalar));
}

#[cube(launch_unchecked)]
pub(crate) fn kernel_binop<C: Numeric, O: BinaryOpFamily>(
    lhs: &LinearView<Line<C>>,
    rhs: &LinearView<Line<C>>,
    out: &mut LinearView<Line<C>, ReadWrite>,
) {
    if !out.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    out[ABSOLUTE_POS] = O::BinaryOp::<C>::execute(lhs[ABSOLUTE_POS], rhs[ABSOLUTE_POS]);
}

pub(crate) fn launch_binop<R: CubeRuntime, E: CubeElement, O: BinaryOpFamily>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
) -> CubeTensor<R> {
    let line_size_lhs = max_line_size(&lhs);
    let line_size_rhs = max_line_size(&rhs);
    let line_size = Ord::min(line_size_lhs, line_size_rhs);

    let shape_out = broadcast_shape(&[&lhs, &rhs]);

    let client = lhs.client.clone();
    let num_elems = shape_out.num_elements();

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems / line_size as usize, cube_dim);

    unsafe {
        if lhs.can_mut_broadcast(&rhs) {
            kernel_binop::launch_unchecked::<E, O, R>(
                &client,
                cube_count,
                cube_dim,
                linear_view(&lhs, line_size),
                linear_view_ref(&rhs, &lhs, line_size),
                linear_view_alias(&lhs, line_size, 0),
            );

            lhs
        } else if rhs.can_mut_broadcast(&lhs) {
            kernel_binop::launch_unchecked::<E, O, R>(
                &client,
                cube_count,
                cube_dim,
                linear_view_ref(&lhs, &rhs, line_size),
                linear_view(&rhs, line_size),
                linear_view_alias(&rhs, line_size, 1),
            );

            rhs
        } else {
            let output = empty_device::<R, E>(lhs.client.clone(), lhs.device.clone(), shape_out);

            kernel_binop::launch_unchecked::<E, O, R>(
                &client,
                cube_count,
                cube_dim,
                linear_view_ref(&lhs, &output, line_size),
                linear_view_ref(&rhs, &output, line_size),
                linear_view(&output, line_size),
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

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems / line_size as usize, cube_dim);

    unsafe {
        if tensor.can_mut() && tensor.is_contiguous_buffer() {
            kernel_scalar_binop::launch_unchecked::<E, O, R>(
                &client,
                cube_count,
                cube_dim,
                linear_view(&tensor, line_size),
                ScalarArg::new(scalar),
                linear_view_alias(&tensor, line_size, 0),
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
                linear_view(&tensor, line_size),
                ScalarArg::new(scalar),
                linear_view(&output, line_size),
            );

            output
        }
    }
}
