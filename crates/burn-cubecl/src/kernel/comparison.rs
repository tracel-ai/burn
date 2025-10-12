use std::marker::PhantomData;

use crate::{
    BoolElement, CubeRuntime,
    element::CubeElement,
    kernel::utils::{broadcast_shape, linear_view, linear_view_alias, linear_view_ref},
    ops::{max_line_size, numeric::empty_device},
    tensor::CubeTensor,
};
use cubecl::{calculate_cube_count_elemwise, prelude::*, std::tensor::layout::linear::LinearView};

#[cube]
pub(crate) trait ComparisonOp<C: Numeric>: 'static + Send + Sync {
    /// Execute a comparison operation.
    fn execute(lhs: Line<C>, rhs: Line<C>) -> bool;
}

struct EqualOp;
struct GreaterEqualOp;
struct LowerEqualOp;
struct GreaterOp;
struct LowerOp;

#[cube]
impl<N: Numeric> ComparisonOp<N> for EqualOp {
    fn execute(lhs: Line<N>, rhs: Line<N>) -> bool {
        lhs == rhs
    }
}

#[cube]
impl<N: Numeric> ComparisonOp<N> for GreaterEqualOp {
    fn execute(lhs: Line<N>, rhs: Line<N>) -> bool {
        lhs >= rhs
    }
}

#[cube]
impl<N: Numeric> ComparisonOp<N> for LowerEqualOp {
    fn execute(lhs: Line<N>, rhs: Line<N>) -> bool {
        lhs <= rhs
    }
}

#[cube]
impl<N: Numeric> ComparisonOp<N> for GreaterOp {
    fn execute(lhs: Line<N>, rhs: Line<N>) -> bool {
        lhs > rhs
    }
}

#[cube]
impl<N: Numeric> ComparisonOp<N> for LowerOp {
    fn execute(lhs: Line<N>, rhs: Line<N>) -> bool {
        lhs < rhs
    }
}

pub(crate) trait ScalarOpSpec: Send + Sync + 'static {
    type C: Numeric;
    type B: Numeric;
}

pub(crate) struct Spec<C, B> {
    _c: PhantomData<C>,
    _b: PhantomData<B>,
}

impl<C: Numeric, B: Numeric> ScalarOpSpec for Spec<C, B> {
    type C = C;
    type B = B;
}

#[cube(launch_unchecked)]
pub(crate) fn kernel_scalar_cmp<SS: ScalarOpSpec, O: ComparisonOp<SS::C>>(
    input: &LinearView<Line<SS::C>>,
    scalar: SS::C,
    output: &mut LinearView<Line<SS::B>, ReadWrite>,
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    output[ABSOLUTE_POS] = Line::cast_from(O::execute(input[ABSOLUTE_POS], Line::new(scalar)));
}

#[cube(launch)]
pub(crate) fn kernel_cmp<SS: ScalarOpSpec, O: ComparisonOp<SS::C>>(
    lhs: &LinearView<Line<SS::C>>,
    rhs: &LinearView<Line<SS::C>>,
    out: &mut LinearView<Line<SS::B>, ReadWrite>,
) {
    if !out.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    out[ABSOLUTE_POS] = Line::cast_from(O::execute(lhs[ABSOLUTE_POS], rhs[ABSOLUTE_POS]));
}

pub(crate) fn launch_cmp<R: CubeRuntime, E: CubeElement, BT: BoolElement, O: ComparisonOp<E>>(
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

    let same_tensor_type = core::any::TypeId::of::<E>() == core::any::TypeId::of::<BT>();
    if same_tensor_type && lhs.can_mut_broadcast(&rhs) {
        kernel_cmp::launch::<Spec<E, BT>, O, R>(
            &client,
            cube_count,
            cube_dim,
            linear_view(&lhs, line_size),
            linear_view_ref(&rhs, &lhs, line_size),
            linear_view_alias(&lhs, line_size, 0),
        );

        CubeTensor::new(
            lhs.client,
            lhs.handle,
            lhs.shape,
            lhs.device,
            lhs.strides,
            BT::dtype(),
        )
    } else if same_tensor_type && rhs.can_mut_broadcast(&lhs) {
        kernel_cmp::launch::<Spec<E, BT>, O, R>(
            &client,
            cube_count,
            CubeDim::default(),
            linear_view_ref(&lhs, &rhs, line_size),
            linear_view(&rhs, line_size),
            linear_view_alias(&rhs, line_size, 1),
        );

        CubeTensor::new(
            rhs.client,
            rhs.handle,
            rhs.shape,
            rhs.device,
            rhs.strides,
            BT::dtype(),
        )
    } else {
        let output = empty_device::<R, BT>(lhs.client.clone(), lhs.device.clone(), shape_out);

        kernel_cmp::launch::<Spec<E, BT>, O, R>(
            &client,
            cube_count,
            CubeDim::default(),
            linear_view_ref(&lhs, &output, line_size),
            linear_view_ref(&rhs, &output, line_size),
            linear_view(&output, line_size),
        );

        output
    }
}

pub(crate) fn launch_scalar_cmp<
    R: CubeRuntime,
    E: CubeElement,
    BT: BoolElement,
    O: ComparisonOp<E>,
>(
    tensor: CubeTensor<R>,
    scalar: E,
) -> CubeTensor<R> {
    let line_size = max_line_size(&tensor);
    let client = tensor.client.clone();
    let num_elems = tensor.shape.num_elements();

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems / line_size as usize, cube_dim);

    let same_tensor_type = core::any::TypeId::of::<E>() == core::any::TypeId::of::<BT>();
    if same_tensor_type && tensor.can_mut() {
        unsafe {
            kernel_scalar_cmp::launch_unchecked::<Spec<E, BT>, O, R>(
                &client,
                cube_count,
                cube_dim,
                linear_view(&tensor, line_size),
                ScalarArg::new(scalar),
                linear_view_alias(&tensor, line_size, 0),
            );
        }

        CubeTensor::new(
            tensor.client,
            tensor.handle,
            tensor.shape,
            tensor.device,
            tensor.strides,
            BT::dtype(),
        )
    } else {
        let output = empty_device::<R, BT>(
            tensor.client.clone(),
            tensor.device.clone(),
            tensor.shape.clone(),
        );

        unsafe {
            kernel_scalar_cmp::launch_unchecked::<Spec<E, BT>, O, R>(
                &client,
                cube_count,
                CubeDim::default(),
                linear_view(&tensor, line_size),
                ScalarArg::new(scalar),
                linear_view(&output, line_size),
            );
        }

        output
    }
}

pub fn equal<R: CubeRuntime, E: CubeElement, BT: BoolElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
) -> CubeTensor<R> {
    launch_cmp::<R, E, BT, EqualOp>(lhs, rhs)
}

pub fn greater<R: CubeRuntime, E: CubeElement, BT: BoolElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
) -> CubeTensor<R> {
    launch_cmp::<R, E, BT, GreaterOp>(lhs, rhs)
}

pub fn greater_equal<R: CubeRuntime, E: CubeElement, BT: BoolElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
) -> CubeTensor<R> {
    launch_cmp::<R, E, BT, GreaterEqualOp>(lhs, rhs)
}

pub fn lower<R: CubeRuntime, E: CubeElement, BT: BoolElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
) -> CubeTensor<R> {
    launch_cmp::<R, E, BT, LowerOp>(lhs, rhs)
}

pub fn lower_equal<R: CubeRuntime, E: CubeElement, BT: BoolElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
) -> CubeTensor<R> {
    launch_cmp::<R, E, BT, LowerEqualOp>(lhs, rhs)
}

pub fn equal_elem<R: CubeRuntime, E: CubeElement, BT: BoolElement>(
    lhs: CubeTensor<R>,
    rhs: E,
) -> CubeTensor<R> {
    launch_scalar_cmp::<R, E, BT, EqualOp>(lhs, rhs)
}

pub fn greater_elem<R: CubeRuntime, E: CubeElement, BT: BoolElement>(
    lhs: CubeTensor<R>,
    rhs: E,
) -> CubeTensor<R> {
    launch_scalar_cmp::<R, E, BT, GreaterOp>(lhs, rhs)
}

pub fn lower_elem<R: CubeRuntime, E: CubeElement, BT: BoolElement>(
    lhs: CubeTensor<R>,
    rhs: E,
) -> CubeTensor<R> {
    launch_scalar_cmp::<R, E, BT, LowerOp>(lhs, rhs)
}

pub fn greater_equal_elem<R: CubeRuntime, E: CubeElement, BT: BoolElement>(
    lhs: CubeTensor<R>,
    rhs: E,
) -> CubeTensor<R> {
    launch_scalar_cmp::<R, E, BT, GreaterEqualOp>(lhs, rhs)
}

pub fn lower_equal_elem<R: CubeRuntime, E: CubeElement, BT: BoolElement>(
    lhs: CubeTensor<R>,
    rhs: E,
) -> CubeTensor<R> {
    launch_scalar_cmp::<R, E, BT, LowerEqualOp>(lhs, rhs)
}

// Unary comparison / predicate / relational ops

#[cube]
pub(crate) trait PredicateOp<F: Float>: 'static + Send + Sync {
    /// Execute a predicate operation.
    fn execute(input: Line<F>) -> bool;
}

struct IsNanOp;
struct IsInfOp;

#[cube]
impl<F: Float> PredicateOp<F> for IsNanOp {
    fn execute(input: Line<F>) -> bool {
        Line::is_nan(input)
    }
}

#[cube]
impl<F: Float> PredicateOp<F> for IsInfOp {
    fn execute(input: Line<F>) -> bool {
        Line::is_inf(input)
    }
}

// Defines the input/output types for a predicate
pub(crate) trait PredicateOpSpec: Send + Sync + 'static {
    type F: Float;
    type B: Numeric;
}

impl<F: Float, B: Numeric> PredicateOpSpec for Spec<F, B> {
    type F = F;
    type B = B;
}

#[cube(launch_unchecked)]
pub(crate) fn kernel_predicate<S: PredicateOpSpec, O: PredicateOp<S::F>>(
    input: &LinearView<Line<S::F>>,
    output: &mut LinearView<Line<S::B>, ReadWrite>,
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    output[ABSOLUTE_POS] = Line::cast_from(O::execute(input[ABSOLUTE_POS]));
}

pub(crate) fn launch_predicate<
    R: CubeRuntime,
    E: CubeElement + Float,
    BT: BoolElement,
    O: PredicateOp<E>,
>(
    tensor: CubeTensor<R>,
) -> CubeTensor<R> {
    let line_size = max_line_size(&tensor);

    let client = tensor.client.clone();
    let num_elems = tensor.shape.num_elements();

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems / line_size as usize, cube_dim);

    let output = empty_device::<R, BT>(
        tensor.client.clone(),
        tensor.device.clone(),
        tensor.shape.clone(),
    );

    unsafe {
        kernel_predicate::launch_unchecked::<Spec<E, BT>, O, R>(
            &client,
            cube_count,
            CubeDim::default(),
            linear_view_ref(&tensor, &output, line_size),
            linear_view(&output, line_size),
        );
    }

    output
}

pub fn is_nan<R: CubeRuntime, E: CubeElement + Float, BT: BoolElement>(
    tensor: CubeTensor<R>,
) -> CubeTensor<R> {
    launch_predicate::<R, E, BT, IsNanOp>(tensor)
}

pub fn is_inf<R: CubeRuntime, E: CubeElement + Float, BT: BoolElement>(
    tensor: CubeTensor<R>,
) -> CubeTensor<R> {
    launch_predicate::<R, E, BT, IsInfOp>(tensor)
}
