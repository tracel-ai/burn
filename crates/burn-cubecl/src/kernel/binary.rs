use crate::{
    CubeRuntime,
    kernel::utils::{broadcast_shape, linear_view, linear_view_alias, linear_view_ref},
    ops::{max_line_size, numeric::empty_device_dtype},
    tensor::CubeTensor,
};
use burn_backend::{bf16, f16};
use cubecl::{
    calculate_cube_count_elemwise, intrinsic, prelude::*, std::tensor::layout::linear::LinearView,
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
pub(crate) struct PowOp;

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

impl BinaryOpFamily for PowOp {
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
impl<N: Numeric> BinaryOp<N> for PowOp {
    #[allow(unused)]
    fn execute(lhs: Line<N>, rhs: Line<N>) -> Line<N> {
        intrinsic!(|scope| {
            let elem = N::as_type(scope).elem_type();

            if let cubecl::ir::ElemType::Float(kind) = elem {
                match kind {
                    cubecl::ir::FloatKind::F16 => {
                        let lhs = <Line<f16> as Cast>::__expand_cast_from(scope, lhs);
                        let rhs = <Line<f16> as Cast>::__expand_cast_from(scope, rhs);
                        let out = Line::__expand_powf(scope, lhs, rhs);
                        return <Line<N> as Cast>::__expand_cast_from(scope, out);
                    }
                    cubecl::ir::FloatKind::BF16 => {
                        let lhs = <Line<bf16> as Cast>::__expand_cast_from(scope, lhs);
                        let rhs = <Line<bf16> as Cast>::__expand_cast_from(scope, rhs);
                        let out = Line::__expand_powf(scope, lhs, rhs);
                        return <Line<N> as Cast>::__expand_cast_from(scope, out);
                    }
                    cubecl::ir::FloatKind::F64 => {
                        let lhs = <Line<f64> as Cast>::__expand_cast_from(scope, lhs);
                        let rhs = <Line<f64> as Cast>::__expand_cast_from(scope, rhs);
                        let out = Line::__expand_powf(scope, lhs, rhs);
                        return <Line<N> as Cast>::__expand_cast_from(scope, out);
                    }
                    _ => {}
                }
            };

            let lhs = <Line<f32> as Cast>::__expand_cast_from(scope, lhs);
            let rhs = <Line<f32> as Cast>::__expand_cast_from(scope, rhs);
            let out = Line::__expand_powf(scope, lhs, rhs);
            return <Line<N> as Cast>::__expand_cast_from(scope, out);
        })
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

#[cube(launch_unchecked)]
pub(crate) fn kernel_binop<C: Numeric, O: BinaryOpFamily>(
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

pub(crate) fn launch_binop<R: CubeRuntime, O: BinaryOpFamily>(
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
                linear_view_ref(&lhs, &rhs, line_size),
                linear_view(&rhs, line_size),
                linear_view_alias(&rhs, line_size, 1),
                dtype.into(),
            )
            .expect("Kernel to never fail");

            rhs
        } else {
            let output = empty_device_dtype(
                lhs.client.clone(),
                lhs.device.clone(),
                shape_out,
                dtype,
            );

            kernel_binop::launch_unchecked::<O, R>(
                &client,
                cube_count,
                cube_dim,
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

pub(crate) fn launch_scalar_binop<R: CubeRuntime, O: BinaryOpFamily>(
    tensor: CubeTensor<R>,
    scalar: InputScalar,
) -> CubeTensor<R> {
    // Vectorization is only enabled when the last dimension is contiguous.
    let line_size = max_line_size(&tensor);
    let client = tensor.client.clone();
    let num_elems = tensor.shape.num_elements();
    let dtype = tensor.dtype;

    let working_units = num_elems / line_size as usize;
    let cube_dim = CubeDim::new(&tensor.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&tensor.client, working_units, cube_dim);

    unsafe {
        if tensor.can_mut() && tensor.is_nonoverlapping() {
            kernel_scalar_binop::launch_unchecked::<O, R>(
                &client,
                cube_count,
                cube_dim,
                linear_view(&tensor, line_size),
                scalar,
                linear_view_alias(&tensor, line_size, 0),
                dtype.into(),
            )
            .expect("Kernel to never fail");

            tensor
        } else {
            let output = empty_device_dtype(
                tensor.client.clone(),
                tensor.device.clone(),
                tensor.shape.clone(),
                dtype,
            );

            kernel_scalar_binop::launch_unchecked::<O, R>(
                &client,
                cube_count,
                cube_dim,
                linear_view(&tensor, line_size),
                scalar,
                linear_view(&output, line_size),
                dtype.into(),
            )
            .expect("Kernel to never fail");

            output
        }
    }
}
