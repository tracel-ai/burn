use crate::{
    CubeRuntime,
    kernel::utils::{
        address_type, broadcast_shape, linear_view, linear_view_alias, linear_view_ref,
    },
    ops::{max_line_size, numeric::empty_device_dtype},
    tensor::CubeTensor,
};
use cubecl::{calculate_cube_count_elemwise, prelude::*, std::tensor::layout::linear::LinearView};

pub(crate) trait BinaryOpFloatFamily: Send + Sync + 'static {
    type BinaryOp<C: Float>: BinaryOpFloat<C>;
}

#[cube]
pub(crate) trait BinaryOpFloat<C: Float>: 'static + Send + Sync {
    /// Execute a binary operation.
    fn execute(lhs: Line<C>, rhs: Line<C>) -> Line<C>;
}

pub(crate) struct ArcTan2Op;

impl BinaryOpFloatFamily for ArcTan2Op {
    type BinaryOp<C: Float> = Self;
}

#[cube]
impl<N: Float> BinaryOpFloat<N> for ArcTan2Op {
    fn execute(lhs: Line<N>, rhs: Line<N>) -> Line<N> {
        Line::atan2(lhs, rhs)
    }
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub(crate) fn kernel_binop<C: Float, O: BinaryOpFloatFamily>(
    lhs: &LinearView<Line<C>>,
    rhs: &LinearView<Line<C>>,
    out: &mut LinearView<Line<C>, ReadWrite>,
    #[define(C)] _dtype: StorageType,
) {
    if !out.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    out[ABSOLUTE_POS] = O::BinaryOp::<C>::execute(lhs[ABSOLUTE_POS], rhs[ABSOLUTE_POS]);
}

pub(crate) fn launch_binop_float<R: CubeRuntime, O: BinaryOpFloatFamily>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
) -> CubeTensor<R> {
    let line_size_lhs = max_line_size(&lhs);
    let line_size_rhs = max_line_size(&rhs);
    let line_size = Ord::min(line_size_lhs, line_size_rhs);

    let shape_out = broadcast_shape(&[&lhs, &rhs]);
    let dtype = lhs.dtype;

    let client = lhs.client.clone();
    let num_elems = shape_out.num_elements();
    let working_units = num_elems / line_size as usize;

    let cube_dim = CubeDim::new(&lhs.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&lhs.client, working_units, cube_dim);

    unsafe {
        if lhs.can_mut_broadcast(&rhs) {
            kernel_binop::launch_unchecked::<O, R>(
                &client,
                cube_count,
                cube_dim,
                address_type!(lhs, rhs),
                linear_view(&lhs, line_size),
                linear_view_ref(&rhs, &lhs, line_size),
                linear_view_alias(&lhs, line_size, 0),
                dtype.into(),
            )
            .expect("Kernel to never fail");

            lhs
        } else if rhs.can_mut_broadcast(&lhs) {
            kernel_binop::launch_unchecked::<O, R>(
                &client,
                cube_count,
                cube_dim,
                address_type!(lhs, rhs),
                linear_view_ref(&lhs, &rhs, line_size),
                linear_view(&rhs, line_size),
                linear_view_alias(&rhs, line_size, 1),
                dtype.into(),
            )
            .expect("Kernel to never fail");

            rhs
        } else {
            let output =
                empty_device_dtype(lhs.client.clone(), lhs.device.clone(), shape_out, dtype);

            kernel_binop::launch_unchecked::<O, R>(
                &client,
                cube_count,
                cube_dim,
                address_type!(lhs, rhs, output),
                linear_view_ref(&lhs, &output, line_size),
                linear_view_ref(&rhs, &output, line_size),
                linear_view(&output, line_size),
                dtype.into(),
            )
            .expect("Kernel to never fail");

            output
        }
    }
}

/// Calculate the four-quadrant inverse tangent of `lhs / rhs`.
pub fn atan2<R: CubeRuntime>(lhs: CubeTensor<R>, rhs: CubeTensor<R>) -> CubeTensor<R> {
    launch_binop_float::<R, ArcTan2Op>(lhs, rhs)
}
