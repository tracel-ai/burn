use crate::{
    CubeRuntime,
    kernel::utils::{
        address_type, broadcast_shape, linear_view, linear_view_alias, linear_view_ref,
    },
    ops::{max_vector_size, numeric::empty_device_dtype},
    tensor::CubeTensor,
};
use burn_backend::TensorMetadata;
use cubecl::{calculate_cube_count_elemwise, prelude::*, std::tensor::layout::linear::LinearView};

pub(crate) trait BinaryOpIntFamily: Send + Sync + 'static {
    type BinaryOp<C: Int, N: Size>: BinaryOpInt<C, N>;
}

#[cube]
pub(crate) trait BinaryOpInt<C: Int, N: Size>: 'static + Send + Sync {
    /// Execute a binary operation.
    fn execute(lhs: Vector<C, N>, rhs: Vector<C, N>) -> Vector<C, N>;
}

pub(crate) struct BitwiseAndOp;
pub(crate) struct BitwiseOrOp;
pub(crate) struct BitwiseXorOp;
pub(crate) struct BitwiseShrOp;
pub(crate) struct BitwiseShlOp;

impl BinaryOpIntFamily for BitwiseAndOp {
    type BinaryOp<C: Int, N: Size> = Self;
}

impl BinaryOpIntFamily for BitwiseOrOp {
    type BinaryOp<C: Int, N: Size> = Self;
}

impl BinaryOpIntFamily for BitwiseXorOp {
    type BinaryOp<C: Int, N: Size> = Self;
}

impl BinaryOpIntFamily for BitwiseShrOp {
    type BinaryOp<C: Int, N: Size> = Self;
}

impl BinaryOpIntFamily for BitwiseShlOp {
    type BinaryOp<C: Int, N: Size> = Self;
}

#[cube]
impl<T: Int, N: Size> BinaryOpInt<T, N> for BitwiseAndOp {
    fn execute(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N> {
        lhs & rhs
    }
}

#[cube]
impl<T: Int, N: Size> BinaryOpInt<T, N> for BitwiseOrOp {
    fn execute(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N> {
        lhs | rhs
    }
}

#[cube]
impl<T: Int, N: Size> BinaryOpInt<T, N> for BitwiseXorOp {
    fn execute(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N> {
        lhs ^ rhs
    }
}

#[cube]
impl<T: Int, N: Size> BinaryOpInt<T, N> for BitwiseShrOp {
    fn execute(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N> {
        lhs >> rhs
    }
}

#[cube]
impl<T: Int, N: Size> BinaryOpInt<T, N> for BitwiseShlOp {
    fn execute(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N> {
        lhs << rhs
    }
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub(crate) fn kernel_scalar_binop_int<C: Int, N: Size, O: BinaryOpIntFamily>(
    input: &LinearView<Vector<C, N>>,
    scalar: InputScalar,
    output: &mut LinearView<Vector<C, N>, ReadWrite>,
    #[define(C)] _dtype: StorageType,
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    output[ABSOLUTE_POS] =
        O::BinaryOp::<C, N>::execute(input[ABSOLUTE_POS], Vector::new(scalar.get::<C>()));
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub(crate) fn kernel_binop_int<C: Int, N: Size, O: BinaryOpIntFamily>(
    lhs: &LinearView<Vector<C, N>>,
    rhs: &LinearView<Vector<C, N>>,
    out: &mut LinearView<Vector<C, N>, ReadWrite>,
    #[define(C)] _dtype: StorageType,
) {
    if !out.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    out[ABSOLUTE_POS] = O::BinaryOp::<C, N>::execute(lhs[ABSOLUTE_POS], rhs[ABSOLUTE_POS]);
}

pub(crate) fn launch_binop_int<R: CubeRuntime, O: BinaryOpIntFamily>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
) -> CubeTensor<R> {
    let vector_size_lhs = max_vector_size(&lhs);
    let vector_size_rhs = max_vector_size(&rhs);
    let vector_size = Ord::min(vector_size_lhs, vector_size_rhs);

    let shape_out = broadcast_shape(&[&lhs, &rhs]);

    let client = lhs.client.clone();
    let num_elems = shape_out.num_elements();

    let working_units = num_elems / vector_size as usize;
    let cube_dim = CubeDim::new(&lhs.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&lhs.client, working_units, cube_dim);
    let dtype = lhs.dtype;

    unsafe {
        if lhs.can_mut_broadcast(&rhs) {
            kernel_binop_int::launch_unchecked::<O, R>(
                &client,
                cube_count,
                cube_dim,
                address_type!(lhs, rhs),
                vector_size,
                linear_view(lhs.clone(), vector_size),
                linear_view_ref(rhs, &lhs, vector_size),
                linear_view_alias(&lhs, vector_size, 0),
                dtype.into(),
            );

            lhs
        } else if rhs.can_mut_broadcast(&lhs) {
            kernel_binop_int::launch_unchecked::<O, R>(
                &client,
                cube_count,
                cube_dim,
                address_type!(lhs, rhs),
                vector_size,
                linear_view_ref(lhs, &rhs, vector_size),
                linear_view(rhs.clone(), vector_size),
                linear_view_alias(&rhs, vector_size, 1),
                dtype.into(),
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
                vector_size,
                linear_view_ref(lhs, &output, vector_size),
                linear_view_ref(rhs, &output, vector_size),
                linear_view(output.clone(), vector_size),
                dtype.into(),
            );

            output
        }
    }
}

pub(crate) fn launch_scalar_binop_int<R: CubeRuntime, O: BinaryOpIntFamily>(
    tensor: CubeTensor<R>,
    scalar: InputScalar,
) -> CubeTensor<R> {
    let vector_size = max_vector_size(&tensor);
    let client = tensor.client.clone();
    let num_elems = tensor.meta.shape.num_elements();

    let working_units = num_elems / vector_size as usize;
    let cube_dim = CubeDim::new(&tensor.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&tensor.client, working_units, cube_dim);

    unsafe {
        if tensor.can_mut() && tensor.is_nonoverlapping() {
            kernel_scalar_binop_int::launch_unchecked::<O, R>(
                &client,
                cube_count,
                cube_dim,
                address_type!(tensor),
                vector_size,
                linear_view(tensor.clone(), vector_size),
                scalar,
                linear_view_alias(&tensor, vector_size, 0),
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
                vector_size,
                linear_view(tensor, vector_size),
                scalar,
                linear_view(output.clone(), vector_size),
                output.dtype.into(),
            );

            output
        }
    }
}
