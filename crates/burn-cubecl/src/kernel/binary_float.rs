use crate::kernel::memory_order::in_memory_order;
use crate::kernel::{FloatUnaryOp, FloatUnaryOpFamily, launch_unary_float};
use crate::{
    kernel::utils::{address_type, broadcast_shape},
    ops::{max_vector_size, numeric::empty_device_dtype},
    tensor::CubeTensor,
};
use burn_backend::cubecl::dtype_to_storage_type;
use cubecl::{
    calculate_cube_count_elemwise,
    prelude::*,
    std::tensor::layout::linear::{LinearView, LinearViewMut},
};

pub(crate) trait BinaryOpFloatFamily: Send + Sync + 'static {
    type BinaryOp<C: Float, N: Size>: BinaryOpFloat<C, N>;
}

#[cube]
pub(crate) trait BinaryOpFloat<C: Float, N: Size>: 'static + Send + Sync {
    /// Execute a binary operation.
    fn execute(lhs: Vector<C, N>, rhs: Vector<C, N>) -> Vector<C, N>;
}

pub(crate) struct ArcTan2Op;
pub(crate) struct FmodOp;

impl BinaryOpFloatFamily for ArcTan2Op {
    type BinaryOp<C: Float, N: Size> = Self;
}

impl BinaryOpFloatFamily for FmodOp {
    type BinaryOp<C: Float, N: Size> = Self;
}

#[cube]
impl<T: Float, N: Size> BinaryOpFloat<T, N> for ArcTan2Op {
    fn execute(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N> {
        Vector::atan2(lhs, rhs)
    }
}

#[cube]
impl<T: Float, N: Size> BinaryOpFloat<T, N> for FmodOp {
    fn execute(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N> {
        fmod_vector(lhs, rhs)
    }
}

impl FloatUnaryOpFamily for FmodOp {
    type Options = InputScalar;
    type Unary<F: Float, N: Size> = Self;
}

#[cube]
impl<F: Float, N: Size> FloatUnaryOp<F, N> for FmodOp {
    type Options = InputScalar;

    fn execute(input: Vector<F, N>, options: &Self::Options) -> Vector<F, N> {
        fmod_vector(input, Vector::new(options.get::<F>()))
    }
}

/// `fmod(x, ±inf)` is `x` for a finite `x` and NaN for an infinite one. Targets disagree on `%`
/// there: WGSL and Vulkan allow `x - y * trunc(x / y)`, which is NaN for a finite `x`, and NVPTX
/// returns `x` even when `x` is infinite. So that case is handled explicitly.
#[cube]
fn fmod_vector<F: Float, N: Size>(lhs: Vector<F, N>, rhs: Vector<F, N>) -> Vector<F, N> {
    let nan = Vector::new(F::new(f64::NAN));
    let inf_divisor = select_many(Vector::is_inf(lhs), nan, lhs);
    select_many(Vector::is_inf(rhs), inf_divisor, lhs % rhs)
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub(crate) fn kernel_binop<C: Float, N: Size, O: BinaryOpFloatFamily>(
    lhs: LinearView<'_, Vector<C, N>>,
    rhs: LinearView<'_, Vector<C, N>>,
    mut out: LinearViewMut<'_, Vector<C, N>>,
    #[define(C)] _dtype: ElemType,
) {
    if !out.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let res = O::BinaryOp::<C, N>::execute(lhs.read(ABSOLUTE_POS), rhs.read(ABSOLUTE_POS));

    out.write(ABSOLUTE_POS, res)
}

pub(crate) fn launch_binop_float<O: BinaryOpFloatFamily>(
    lhs: CubeTensor,
    rhs: CubeTensor,
) -> CubeTensor {
    let output_shape = broadcast_shape(&[&lhs, &rhs]);
    in_memory_order([lhs, rhs], output_shape, |[lhs, rhs], shape_out| {
        let vector_size_lhs = max_vector_size(&lhs);
        let vector_size_rhs = max_vector_size(&rhs);
        let vector_size = Ord::min(vector_size_lhs, vector_size_rhs);

        let dtype = lhs.dtype;

        // A zero-sized broadcast output has no elements to compute, and the in-place/kernel paths
        // below assume a non-empty output. Return the empty output directly.
        if shape_out.num_elements() == 0 {
            return empty_device_dtype(lhs.client.clone(), lhs.device.clone(), shape_out, dtype);
        }

        let client = lhs.client.clone();
        let num_elems = shape_out.num_elements();
        let working_units = num_elems / vector_size as usize;

        let cube_dim = CubeDim::new(&lhs.client, working_units);
        let cube_count = calculate_cube_count_elemwise(&lhs.client, working_units, cube_dim);

        unsafe {
            if lhs.can_mut_broadcast(&rhs) {
                kernel_binop::launch_unchecked::<O>(
                    &client,
                    cube_count,
                    cube_dim,
                    address_type!(lhs, rhs),
                    vector_size,
                    lhs.clone().into_linear_view(),
                    rhs.clone().into_linear_view_like(&lhs),
                    lhs.as_linear_view_alias(0),
                    dtype_to_storage_type(dtype),
                );

                lhs
            } else if rhs.can_mut_broadcast(&lhs) {
                kernel_binop::launch_unchecked::<O>(
                    &client,
                    cube_count,
                    cube_dim,
                    address_type!(lhs, rhs),
                    vector_size,
                    lhs.into_linear_view_like(&rhs),
                    rhs.clone().into_linear_view(),
                    rhs.as_linear_view_alias(1),
                    dtype_to_storage_type(dtype),
                );

                rhs
            } else {
                let output =
                    empty_device_dtype(lhs.client.clone(), lhs.device.clone(), shape_out, dtype);

                kernel_binop::launch_unchecked::<O>(
                    &client,
                    cube_count,
                    cube_dim,
                    address_type!(lhs, rhs, output),
                    vector_size,
                    lhs.into_linear_view_like(&output),
                    rhs.into_linear_view_like(&output),
                    output.clone().into_linear_view(),
                    dtype_to_storage_type(dtype),
                );

                output
            }
        }
    })
}

/// Calculate the four-quadrant inverse tangent of `lhs / rhs`.
pub fn atan2(lhs: CubeTensor, rhs: CubeTensor) -> CubeTensor {
    launch_binop_float::<ArcTan2Op>(lhs, rhs)
}

/// Calculate the floating-point remainder of `lhs / rhs`, with the sign of `lhs`.
pub fn fmod(lhs: CubeTensor, rhs: CubeTensor) -> CubeTensor {
    launch_binop_float::<FmodOp>(lhs, rhs)
}

/// Calculate the floating-point remainder of `lhs / rhs` for a scalar `rhs`, with the sign of
/// `lhs`.
pub fn fmod_scalar(lhs: CubeTensor, rhs: InputScalar) -> CubeTensor {
    launch_unary_float::<FmodOp, _>(lhs, |_| rhs)
}
