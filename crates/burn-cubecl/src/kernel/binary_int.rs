use crate::{
    CubeRuntime, IntElement,
    kernel::utils::{linear_view, linear_view_alias},
    ops::numeric::empty_device,
    tensor::CubeTensor,
};
use burn_tensor::Shape;
use cubecl::{
    calculate_cube_count_elemwise,
    prelude::*,
    std::tensor::{index_offset_with_layout, layout::linear::LinearView, r#virtual::ReadWrite},
    tensor_line_size_parallel,
};

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

#[cube(launch_unchecked)]
pub(crate) fn kernel_scalar_binop_int<C: Int, O: BinaryOpIntFamily>(
    input: &LinearView<C>,
    scalar: C,
    output: &mut LinearView<C, ReadWrite>,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    output[ABSOLUTE_POS] = O::BinaryOp::<C>::execute(input[ABSOLUTE_POS], Line::new(scalar));
}

#[cube(launch_unchecked)]
pub(crate) fn kernel_binop_int<C: Int, O: BinaryOpIntFamily>(
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

pub(crate) fn launch_binop_int<R: CubeRuntime, E: IntElement, O: BinaryOpIntFamily>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
) -> CubeTensor<R> {
    let ndims = lhs.shape.num_dims();
    let line_size_lhs = tensor_line_size_parallel(
        R::line_size_type(&E::as_type_native_unchecked()),
        &lhs.shape.dims,
        &lhs.strides,
        ndims - 1,
    );
    let line_size_rhs = tensor_line_size_parallel(
        R::line_size_type(&E::as_type_native_unchecked()),
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
            kernel_binop_int::launch_unchecked::<E, O, R>(
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
            kernel_binop_int::launch_unchecked::<E, O, R>(
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

            kernel_binop_int::launch_unchecked::<E, O, R>(
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

pub(crate) fn launch_scalar_binop_int<R: CubeRuntime, E: IntElement, O: BinaryOpIntFamily>(
    tensor: CubeTensor<R>,
    scalar: E,
) -> CubeTensor<R> {
    // Vectorization is only enabled when the last dimension is contiguous.
    let ndims = tensor.shape.num_dims();
    let line_size = tensor_line_size_parallel(
        R::line_size_type(&E::as_type_native_unchecked()),
        &tensor.shape.dims,
        &tensor.strides,
        ndims - 1,
    );
    let client = tensor.client.clone();
    let num_elems = tensor.shape.num_elements();

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems / line_size as usize, cube_dim);

    unsafe {
        if tensor.can_mut() && tensor.is_contiguous_buffer() {
            kernel_scalar_binop_int::launch_unchecked::<E, O, R>(
                &client,
                cube_count,
                cube_dim,
                linear_view(&tensor, &line_size),
                ScalarArg::new(scalar),
                linear_view_alias(&tensor, &line_size, 0),
            );

            tensor
        } else {
            let output = empty_device::<R, E>(
                tensor.client.clone(),
                tensor.device.clone(),
                tensor.shape.clone(),
            );

            kernel_scalar_binop_int::launch_unchecked::<E, O, R>(
                &client,
                cube_count,
                CubeDim::default(),
                linear_view(&tensor, &line_size),
                ScalarArg::new(scalar),
                linear_view(&output, &line_size),
            );

            output
        }
    }
}
