use crate::{
    CubeRuntime,
    kernel::utils::{
        address_type, broadcast_shape, linear_view, linear_view_alias, linear_view_ref,
    },
    ops::{max_line_size, numeric::empty_device_dtype},
    tensor::CubeTensor,
};
use burn_backend::TensorMetadata;
use cubecl::{calculate_cube_count_elemwise, prelude::*, std::tensor::layout::linear::LinearView};

pub(crate) trait BinaryOpIntFamily: Send + Sync + 'static {
    type BinaryOp<C: Int>: BinaryOpInt<C>;
}

#[cube]
pub(crate) trait BinaryOpInt<C: Int>: 'static + Send + Sync {
    /// Execute a binary operation.
    fn execute(lhs: Line<C>, rhs: Line<C>) -> Line<C>;
}

pub(crate) struct BitwiseAndOp;
pub(crate) struct BitwiseOrOp;
pub(crate) struct BitwiseXorOp;
pub(crate) struct BitwiseShrOp;
pub(crate) struct BitwiseShlOp;

impl BinaryOpIntFamily for BitwiseAndOp {
    type BinaryOp<C: Int> = Self;
}

impl BinaryOpIntFamily for BitwiseOrOp {
    type BinaryOp<C: Int> = Self;
}

impl BinaryOpIntFamily for BitwiseXorOp {
    type BinaryOp<C: Int> = Self;
}

impl BinaryOpIntFamily for BitwiseShrOp {
    type BinaryOp<C: Int> = Self;
}

impl BinaryOpIntFamily for BitwiseShlOp {
    type BinaryOp<C: Int> = Self;
}

#[cube]
impl<N: Int> BinaryOpInt<N> for BitwiseAndOp {
    fn execute(lhs: Line<N>, rhs: Line<N>) -> Line<N> {
        lhs & rhs
    }
}

#[cube]
impl<N: Int> BinaryOpInt<N> for BitwiseOrOp {
    fn execute(lhs: Line<N>, rhs: Line<N>) -> Line<N> {
        lhs | rhs
    }
}

#[cube]
impl<N: Int> BinaryOpInt<N> for BitwiseXorOp {
    fn execute(lhs: Line<N>, rhs: Line<N>) -> Line<N> {
        lhs ^ rhs
    }
}

#[cube]
impl<N: Int> BinaryOpInt<N> for BitwiseShrOp {
    fn execute(lhs: Line<N>, rhs: Line<N>) -> Line<N> {
        lhs >> rhs
    }
}

#[cube]
impl<N: Int> BinaryOpInt<N> for BitwiseShlOp {
    fn execute(lhs: Line<N>, rhs: Line<N>) -> Line<N> {
        lhs << rhs
    }
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub(crate) fn kernel_scalar_binop_int<C: Int, O: BinaryOpIntFamily>(
    input: &LinearView<Line<C>>,
    scalar: InputScalar,
    output: &mut LinearView<Line<C>, ReadWrite>,
    #[define(C)] _dtype: StorageType,
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    output[ABSOLUTE_POS] =
        O::BinaryOp::<C>::execute(input[ABSOLUTE_POS], Line::new(scalar.get::<C>()));
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub(crate) fn kernel_binop_int<C: Int, O: BinaryOpIntFamily>(
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

pub(crate) fn launch_binop_int<R: CubeRuntime, O: BinaryOpIntFamily>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
) -> CubeTensor<R> {
    let line_size_lhs = max_line_size(&lhs);
    let line_size_rhs = max_line_size(&rhs);
    let line_size = Ord::min(line_size_lhs, line_size_rhs);

    let shape_out = broadcast_shape(&[&lhs, &rhs]);

    let client = lhs.client.clone();
    let num_elems = shape_out.num_elements();

    let working_units = num_elems / line_size as usize;
    let cube_dim = CubeDim::new(&lhs.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&lhs.client, working_units, cube_dim);

    unsafe {
        if lhs.can_mut_broadcast(&rhs) {
            kernel_binop_int::launch_unchecked::<O, R>(
                &client,
                cube_count,
                cube_dim,
                address_type!(lhs, rhs),
                linear_view(&lhs, line_size),
                linear_view_ref(&rhs, &lhs, line_size),
                linear_view_alias(&lhs, line_size, 0),
                lhs.dtype.into(),
            );

            lhs
        } else if rhs.can_mut_broadcast(&lhs) {
            kernel_binop_int::launch_unchecked::<O, R>(
                &client,
                cube_count,
                cube_dim,
                address_type!(lhs, rhs),
                linear_view_ref(&lhs, &rhs, line_size),
                linear_view(&rhs, line_size),
                linear_view_alias(&rhs, line_size, 1),
                lhs.dtype.into(),
            );

            rhs
        } else {
            let output =
                empty_device_dtype(lhs.client.clone(), lhs.device.clone(), shape_out, lhs.dtype);

            kernel_binop_int::launch_unchecked::<O, R>(
                &client,
                cube_count,
                cube_dim,
                address_type!(lhs, rhs, output),
                linear_view_ref(&lhs, &output, line_size),
                linear_view_ref(&rhs, &output, line_size),
                linear_view(&output, line_size),
                lhs.dtype.into(),
            );

            output
        }
    }
}

pub(crate) fn launch_scalar_binop_int<R: CubeRuntime, O: BinaryOpIntFamily>(
    tensor: CubeTensor<R>,
    scalar: InputScalar,
) -> CubeTensor<R> {
    let line_size = max_line_size(&tensor);
    let client = tensor.client.clone();
    let num_elems = tensor.meta.shape.num_elements();

    let working_units = num_elems / line_size as usize;
    let cube_dim = CubeDim::new(&tensor.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&tensor.client, working_units, cube_dim);

    unsafe {
        if tensor.can_mut() && tensor.is_nonoverlapping() {
            kernel_scalar_binop_int::launch_unchecked::<O, R>(
                &client,
                cube_count,
                cube_dim,
                address_type!(tensor),
                linear_view(&tensor, line_size),
                scalar,
                linear_view_alias(&tensor, line_size, 0),
                tensor.dtype.into(),
            );

            tensor
        } else {
            let output = empty_device_dtype(
                tensor.client.clone(),
                tensor.device.clone(),
                tensor.shape(),
                tensor.dtype,
            );

            kernel_scalar_binop_int::launch_unchecked::<O, R>(
                &client,
                cube_count,
                cube_dim,
                address_type!(tensor, output),
                linear_view(&tensor, line_size),
                scalar,
                linear_view(&output, line_size),
                tensor.dtype.into(),
            );

            output
        }
    }
}
