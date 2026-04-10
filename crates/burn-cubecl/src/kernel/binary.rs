use crate::{
    CubeRuntime,
    kernel::utils::{address_type, broadcast_shape},
    ops::{max_vector_size, numeric::empty_device_dtype},
    tensor::CubeTensor,
};
use burn_backend::{TensorMetadata, bf16, f16};
use cubecl::{
    calculate_cube_count_elemwise, intrinsic, prelude::*, std::tensor::layout::linear::LinearView,
};

pub(crate) trait BinaryOpFamily: Send + Sync + 'static {
    type BinaryOp<C: Numeric, N: Size>: BinaryOp<C, N>;
}

#[cube]
pub(crate) trait BinaryOp<C: Numeric, N: Size>: 'static + Send + Sync {
    /// Execute a binary operation.
    fn execute(lhs: Vector<C, N>, rhs: Vector<C, N>) -> Vector<C, N>;
}

pub(crate) struct AddOp;
pub(crate) struct SubOp;
pub(crate) struct MulOp;
pub(crate) struct DivOp;
pub(crate) struct RemainderOp;
pub(crate) struct AndOp;
pub(crate) struct OrOp;
pub(crate) struct PowOp;
pub(crate) struct AssignOp;
pub(crate) struct BinaryMinOp;
pub(crate) struct BinaryMaxOp;

impl BinaryOpFamily for AddOp {
    type BinaryOp<C: Numeric, N: Size> = Self;
}

impl BinaryOpFamily for SubOp {
    type BinaryOp<C: Numeric, N: Size> = Self;
}

impl BinaryOpFamily for MulOp {
    type BinaryOp<C: Numeric, N: Size> = Self;
}

impl BinaryOpFamily for DivOp {
    type BinaryOp<C: Numeric, N: Size> = Self;
}

impl BinaryOpFamily for RemainderOp {
    type BinaryOp<C: Numeric, N: Size> = Self;
}

impl BinaryOpFamily for PowOp {
    type BinaryOp<C: Numeric, N: Size> = Self;
}

impl BinaryOpFamily for AndOp {
    type BinaryOp<C: Numeric, N: Size> = Self;
}

impl BinaryOpFamily for OrOp {
    type BinaryOp<C: Numeric, N: Size> = Self;
}

impl BinaryOpFamily for AssignOp {
    type BinaryOp<C: Numeric, N: Size> = Self;
}

impl BinaryOpFamily for BinaryMinOp {
    type BinaryOp<C: Numeric, N: Size> = Self;
}

impl BinaryOpFamily for BinaryMaxOp {
    type BinaryOp<C: Numeric, N: Size> = Self;
}

#[cube]
impl<T: Numeric, N: Size> BinaryOp<T, N> for AddOp {
    fn execute(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N> {
        lhs + rhs
    }
}

#[cube]
impl<T: Numeric, N: Size> BinaryOp<T, N> for SubOp {
    fn execute(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N> {
        lhs - rhs
    }
}

#[cube]
impl<T: Numeric, N: Size> BinaryOp<T, N> for MulOp {
    fn execute(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N> {
        lhs * rhs
    }
}

#[cube]
impl<T: Numeric, N: Size> BinaryOp<T, N> for DivOp {
    fn execute(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N> {
        lhs / rhs
    }
}

#[cube]
impl<T: Numeric, N: Size> BinaryOp<T, N> for RemainderOp {
    fn execute(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N> {
        Vector::rem(lhs, rhs)
    }
}

#[cube]
impl<T: Numeric, N: Size> BinaryOp<T, N> for PowOp {
    #[allow(unused)]
    fn execute(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N> {
        intrinsic!(|scope| {
            let elem = T::as_type(scope).elem_type();

            if let cubecl::ir::ElemType::Float(kind) = elem {
                match kind {
                    cubecl::ir::FloatKind::F16 => {
                        let lhs = <Vector<f16, N> as Cast>::__expand_cast_from(scope, lhs);
                        let rhs = <Vector<f16, N> as Cast>::__expand_cast_from(scope, rhs);
                        let out = Vector::__expand_powf(scope, lhs, rhs);
                        return <Vector<T, N> as Cast>::__expand_cast_from(scope, out);
                    }
                    cubecl::ir::FloatKind::BF16 => {
                        let lhs = <Vector<bf16, N> as Cast>::__expand_cast_from(scope, lhs);
                        let rhs = <Vector<bf16, N> as Cast>::__expand_cast_from(scope, rhs);
                        let out = Vector::__expand_powf(scope, lhs, rhs);
                        return <Vector<T, N> as Cast>::__expand_cast_from(scope, out);
                    }
                    cubecl::ir::FloatKind::F64 => {
                        let lhs = <Vector<f64, N> as Cast>::__expand_cast_from(scope, lhs);
                        let rhs = <Vector<f64, N> as Cast>::__expand_cast_from(scope, rhs);
                        let out = Vector::__expand_powf(scope, lhs, rhs);
                        return <Vector<T, N> as Cast>::__expand_cast_from(scope, out);
                    }
                    _ => {}
                }
            };

            let lhs = <Vector<f32, N> as Cast>::__expand_cast_from(scope, lhs);
            let rhs = <Vector<f32, N> as Cast>::__expand_cast_from(scope, rhs);
            let out = Vector::__expand_powf(scope, lhs, rhs);
            return <Vector<T, N> as Cast>::__expand_cast_from(scope, out);
        })
    }
}

#[cube]
impl<T: Numeric, N: Size> BinaryOp<T, N> for AndOp {
    fn execute(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N> {
        Vector::cast_from(Vector::<bool, N>::cast_from(lhs).and(Vector::<bool, N>::cast_from(rhs)))
    }
}

#[cube]
impl<T: Numeric, N: Size> BinaryOp<T, N> for OrOp {
    fn execute(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N> {
        Vector::cast_from(Vector::<bool, N>::cast_from(lhs).or(Vector::<bool, N>::cast_from(rhs)))
    }
}

#[cube]
impl<T: Numeric, N: Size> BinaryOp<T, N> for AssignOp {
    fn execute(_lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N> {
        rhs
    }
}

#[cube]
impl<T: Numeric, N: Size> BinaryOp<T, N> for BinaryMinOp {
    fn execute(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N> {
        clamp_max(lhs, rhs)
    }
}

#[cube]
impl<T: Numeric, N: Size> BinaryOp<T, N> for BinaryMaxOp {
    fn execute(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N> {
        clamp_min(lhs, rhs)
    }
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub(crate) fn kernel_scalar_binop<C: Numeric, N: Size, O: BinaryOpFamily>(
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
pub(crate) fn kernel_binop<C: Numeric, N: Size, O: BinaryOpFamily>(
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

pub(crate) fn launch_binop<R: CubeRuntime, O: BinaryOpFamily>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
) -> CubeTensor<R> {
    let vector_size_lhs = max_vector_size(&lhs);
    let vector_size_rhs = max_vector_size(&rhs);
    let vector_size = Ord::min(vector_size_lhs, vector_size_rhs);

    let shape_out = broadcast_shape(&[&lhs, &rhs]);
    let dtype = lhs.dtype;

    let client = lhs.client.clone();
    let num_elems = shape_out.num_elements();
    let working_units = num_elems / vector_size as usize;

    let cube_dim = CubeDim::new(&lhs.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&lhs.client, working_units, cube_dim);

    unsafe {
        if lhs.can_mut_broadcast(&rhs) {
            kernel_binop::launch_unchecked::<O, R>(
                &client,
                cube_count,
                cube_dim,
                address_type!(lhs, rhs),
                vector_size,
                lhs.clone().into_linear_view(),
                rhs.into_linear_view_like(&lhs),
                lhs.as_linear_view_alias(0),
                dtype.into(),
            );

            lhs
        } else if rhs.can_mut_broadcast(&lhs) {
            kernel_binop::launch_unchecked::<O, R>(
                &client,
                cube_count,
                cube_dim,
                address_type!(lhs, rhs),
                vector_size,
                lhs.into_linear_view_like(&rhs),
                rhs.clone().into_linear_view(),
                rhs.as_linear_view_alias(1),
                dtype.into(),
            );

            rhs
        } else {
            let output =
                empty_device_dtype(lhs.client.clone(), lhs.device.clone(), shape_out, dtype);

            kernel_binop::launch_unchecked::<O, R>(
                &client,
                cube_count,
                cube_dim,
                address_type!(lhs, rhs, output),
                vector_size,
                lhs.into_linear_view_like(&output),
                rhs.into_linear_view_like(&output),
                output.clone().into_linear_view(),
                dtype.into(),
            );

            output
        }
    }
}

pub(crate) fn launch_scalar_binop<R: CubeRuntime, O: BinaryOpFamily>(
    tensor: CubeTensor<R>,
    scalar: InputScalar,
) -> CubeTensor<R> {
    // Vectorization is only enabled when the last dimension is contiguous.
    let vector_size = max_vector_size(&tensor);
    let client = tensor.client.clone();
    let num_elems = tensor.meta.num_elements();
    let dtype = tensor.dtype;

    let working_units = num_elems / vector_size as usize;
    let cube_dim = CubeDim::new(&tensor.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&tensor.client, working_units, cube_dim);

    unsafe {
        if tensor.can_mut() && tensor.is_nonoverlapping() {
            kernel_scalar_binop::launch_unchecked::<O, R>(
                &client,
                cube_count,
                cube_dim,
                address_type!(tensor),
                vector_size,
                tensor.clone().into_linear_view(),
                scalar,
                tensor.as_linear_view_alias(0),
                dtype.into(),
            );

            tensor
        } else {
            let output = empty_device_dtype(
                tensor.client.clone(),
                tensor.device.clone(),
                tensor.shape(),
                dtype,
            );

            kernel_scalar_binop::launch_unchecked::<O, R>(
                &client,
                cube_count,
                cube_dim,
                address_type!(tensor, output),
                vector_size,
                tensor.into_linear_view(),
                scalar,
                output.clone().into_linear_view(),
                dtype.into(),
            );

            output
        }
    }
}
