use crate::{
    CubeRuntime,
    kernel::utils::{broadcast_shape, linear_view, linear_view_alias, linear_view_ref},
    ops::{max_line_size, numeric::empty_device_dtype},
    tensor::CubeTensor,
};
use burn_backend::DType;
use cubecl::{
    calculate_cube_count_elemwise,
    prelude::*,
    std::{scalar::InputScalar, tensor::layout::linear::LinearView},
};

#[cube]
pub(crate) trait ComparisonOpFamily: 'static + Send + Sync {
    type Operation<N: Numeric>: ComparisonOp<N>;
}

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

impl ComparisonOpFamily for EqualOp {
    type Operation<N: Numeric> = Self;
}

#[cube]
impl<N: Numeric> ComparisonOp<N> for EqualOp {
    fn execute(lhs: Line<N>, rhs: Line<N>) -> bool {
        lhs == rhs
    }
}

impl ComparisonOpFamily for GreaterEqualOp {
    type Operation<N: Numeric> = Self;
}

#[cube]
impl<N: Numeric> ComparisonOp<N> for GreaterEqualOp {
    fn execute(lhs: Line<N>, rhs: Line<N>) -> bool {
        lhs >= rhs
    }
}

impl ComparisonOpFamily for LowerEqualOp {
    type Operation<N: Numeric> = Self;
}

#[cube]
impl<N: Numeric> ComparisonOp<N> for LowerEqualOp {
    fn execute(lhs: Line<N>, rhs: Line<N>) -> bool {
        lhs <= rhs
    }
}

impl ComparisonOpFamily for GreaterOp {
    type Operation<N: Numeric> = Self;
}

#[cube]
impl<N: Numeric> ComparisonOp<N> for GreaterOp {
    fn execute(lhs: Line<N>, rhs: Line<N>) -> bool {
        lhs > rhs
    }
}

impl ComparisonOpFamily for LowerOp {
    type Operation<N: Numeric> = Self;
}

#[cube]
impl<N: Numeric> ComparisonOp<N> for LowerOp {
    fn execute(lhs: Line<N>, rhs: Line<N>) -> bool {
        lhs < rhs
    }
}

#[cube(launch_unchecked)]
pub(crate) fn kernel_scalar_cmp<N: Numeric, Bool: Numeric, O: ComparisonOpFamily>(
    input: &LinearView<Line<N>>,
    scalar: InputScalar,
    output: &mut LinearView<Line<Bool>, ReadWrite>,
    #[define(N, Bool)] _dtypes: [StorageType; 2],
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    output[ABSOLUTE_POS] = Line::cast_from(O::Operation::<N>::execute(
        input[ABSOLUTE_POS],
        Line::new(scalar.get::<N>()),
    ));
}

#[cube(launch_unchecked)]
pub(crate) fn kernel_cmp<N: Numeric, Bool: Numeric, O: ComparisonOpFamily>(
    lhs: &LinearView<Line<N>>,
    rhs: &LinearView<Line<N>>,
    out: &mut LinearView<Line<Bool>, ReadWrite>,
    #[define(N, Bool)] _dtype: [StorageType; 2],
) {
    if !out.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    out[ABSOLUTE_POS] = Line::cast_from(O::Operation::<N>::execute(
        lhs[ABSOLUTE_POS],
        rhs[ABSOLUTE_POS],
    ));
}

pub(crate) fn launch_cmp<R: CubeRuntime, O: ComparisonOpFamily>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    dtype_bool: DType,
) -> CubeTensor<R> {
    let line_size_lhs = max_line_size(&lhs);
    let line_size_rhs = max_line_size(&rhs);

    let line_size = Ord::min(line_size_lhs, line_size_rhs);

    let shape_out = broadcast_shape(&[&lhs, &rhs]);
    let client = lhs.client.clone();
    let num_elems = shape_out.num_elements();

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems / line_size as usize, cube_dim);

    let dtypes = [lhs.dtype.into(), dtype_bool.into()];
    let same_tensor_type = dtypes[0] == dtypes[1];
    if same_tensor_type && lhs.can_mut_broadcast(&rhs) {
        unsafe {
            kernel_cmp::launch_unchecked::<O, R>(
                &client,
                cube_count,
                cube_dim,
                linear_view(&lhs, line_size),
                linear_view_ref(&rhs, &lhs, line_size),
                linear_view_alias(&lhs, line_size, 0),
                dtypes,
            )
            .expect("Kernel to never fail");
        }

        CubeTensor::new(
            lhs.client,
            lhs.handle,
            lhs.shape,
            lhs.device,
            lhs.strides,
            dtype_bool,
        )
    } else if same_tensor_type && rhs.can_mut_broadcast(&lhs) {
        unsafe {
            kernel_cmp::launch_unchecked::<O, R>(
                &client,
                cube_count,
                CubeDim::default(),
                linear_view_ref(&lhs, &rhs, line_size),
                linear_view(&rhs, line_size),
                linear_view_alias(&rhs, line_size, 1),
                dtypes,
            )
            .expect("Kernel to never fail");
        };

        CubeTensor::new(
            rhs.client,
            rhs.handle,
            rhs.shape,
            rhs.device,
            rhs.strides,
            dtype_bool,
        )
    } else {
        let output = empty_device_dtype(
            lhs.client.clone(),
            lhs.device.clone(),
            shape_out,
            dtype_bool,
        );

        unsafe {
            kernel_cmp::launch_unchecked::<O, R>(
                &client,
                cube_count,
                CubeDim::default(),
                linear_view_ref(&lhs, &output, line_size),
                linear_view_ref(&rhs, &output, line_size),
                linear_view(&output, line_size),
                dtypes,
            )
            .expect("Kernel to never fail");
        };

        output
    }
}

pub(crate) fn launch_scalar_cmp<R: CubeRuntime, O: ComparisonOpFamily>(
    tensor: CubeTensor<R>,
    scalar: InputScalar,
    dtype_bool: DType,
) -> CubeTensor<R> {
    let line_size = max_line_size(&tensor);
    let client = tensor.client.clone();
    let num_elems = tensor.shape.num_elements();

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems / line_size as usize, cube_dim);

    let dtypes = [tensor.dtype.into(), dtype_bool.into()];
    let same_tensor_type = dtypes[0] == dtypes[1];

    if same_tensor_type && tensor.can_mut() {
        unsafe {
            kernel_scalar_cmp::launch_unchecked::<O, R>(
                &client,
                cube_count,
                cube_dim,
                linear_view(&tensor, line_size),
                scalar,
                linear_view_alias(&tensor, line_size, 0),
                dtypes,
            )
            .expect("Kernel to never fail");
        }

        CubeTensor::new(
            tensor.client,
            tensor.handle,
            tensor.shape,
            tensor.device,
            tensor.strides,
            dtype_bool,
        )
    } else {
        let output = empty_device_dtype(
            tensor.client.clone(),
            tensor.device.clone(),
            tensor.shape.clone(),
            dtype_bool,
        );

        unsafe {
            kernel_scalar_cmp::launch_unchecked::<O, R>(
                &client,
                cube_count,
                CubeDim::default(),
                linear_view(&tensor, line_size),
                scalar,
                linear_view(&output, line_size),
                dtypes,
            )
            .expect("Kernel to never fail");
        }

        output
    }
}

pub fn equal<R: CubeRuntime>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    dtype_bool: DType,
) -> CubeTensor<R> {
    launch_cmp::<R, EqualOp>(lhs, rhs, dtype_bool)
}

pub fn greater<R: CubeRuntime>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    dtype_bool: DType,
) -> CubeTensor<R> {
    launch_cmp::<R, GreaterOp>(lhs, rhs, dtype_bool)
}

pub fn greater_equal<R: CubeRuntime>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    dtype_bool: DType,
) -> CubeTensor<R> {
    launch_cmp::<R, GreaterEqualOp>(lhs, rhs, dtype_bool)
}

pub fn lower<R: CubeRuntime>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    dtype_bool: DType,
) -> CubeTensor<R> {
    launch_cmp::<R, LowerOp>(lhs, rhs, dtype_bool)
}

pub fn lower_equal<R: CubeRuntime>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    dtype_bool: DType,
) -> CubeTensor<R> {
    launch_cmp::<R, LowerEqualOp>(lhs, rhs, dtype_bool)
}

pub fn equal_elem<R: CubeRuntime>(
    lhs: CubeTensor<R>,
    rhs: InputScalar,
    dtype_bool: DType,
) -> CubeTensor<R> {
    launch_scalar_cmp::<R, EqualOp>(lhs, rhs, dtype_bool)
}

pub fn greater_elem<R: CubeRuntime>(
    lhs: CubeTensor<R>,
    rhs: InputScalar,
    dtype_bool: DType,
) -> CubeTensor<R> {
    launch_scalar_cmp::<R, GreaterOp>(lhs, rhs, dtype_bool)
}

pub fn lower_elem<R: CubeRuntime>(
    lhs: CubeTensor<R>,
    rhs: InputScalar,
    dtype_bool: DType,
) -> CubeTensor<R> {
    launch_scalar_cmp::<R, LowerOp>(lhs, rhs, dtype_bool)
}

pub fn greater_equal_elem<R: CubeRuntime>(
    lhs: CubeTensor<R>,
    rhs: InputScalar,
    dtype_bool: DType,
) -> CubeTensor<R> {
    launch_scalar_cmp::<R, GreaterEqualOp>(lhs, rhs, dtype_bool)
}

pub fn lower_equal_elem<R: CubeRuntime>(
    lhs: CubeTensor<R>,
    rhs: InputScalar,
    dtype_bool: DType,
) -> CubeTensor<R> {
    launch_scalar_cmp::<R, LowerEqualOp>(lhs, rhs, dtype_bool)
}

// Unary comparison / predicate / relational ops

#[cube]
pub(crate) trait PredicateOp<F: Float>: 'static + Send + Sync {
    /// Execute a predicate operation.
    fn execute(input: Line<F>) -> bool;
}

pub(crate) trait PredicateOpFamily: 'static + Send + Sync {
    type Operation<F: Float>: PredicateOp<F>;
}

struct IsNanOp;
struct IsInfOp;

impl PredicateOpFamily for IsNanOp {
    type Operation<F: Float> = Self;
}

#[cube]
impl<F: Float> PredicateOp<F> for IsNanOp {
    fn execute(input: Line<F>) -> bool {
        Line::is_nan(input)
    }
}

impl PredicateOpFamily for IsInfOp {
    type Operation<F: Float> = Self;
}
#[cube]
impl<F: Float> PredicateOp<F> for IsInfOp {
    fn execute(input: Line<F>) -> bool {
        Line::is_inf(input)
    }
}

#[cube(launch_unchecked)]
pub(crate) fn kernel_predicate<F: Float, Bool: Numeric, O: PredicateOpFamily>(
    input: &LinearView<Line<F>>,
    output: &mut LinearView<Line<Bool>, ReadWrite>,
    #[define(F, Bool)] _dtypes: [StorageType; 2],
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    output[ABSOLUTE_POS] = Line::cast_from(O::Operation::<F>::execute(input[ABSOLUTE_POS]));
}

pub(crate) fn launch_predicate<R: CubeRuntime, O: PredicateOpFamily>(
    tensor: CubeTensor<R>,
    dtype_bool: DType,
) -> CubeTensor<R> {
    let line_size = max_line_size(&tensor);

    let client = tensor.client.clone();
    let num_elems = tensor.shape.num_elements();

    let dtypes = [tensor.dtype.into(), dtype_bool.into()];
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems / line_size as usize, cube_dim);

    let output = empty_device_dtype(
        tensor.client.clone(),
        tensor.device.clone(),
        tensor.shape.clone(),
        dtype_bool,
    );

    unsafe {
        kernel_predicate::launch_unchecked::<O, R>(
            &client,
            cube_count,
            CubeDim::default(),
            linear_view_ref(&tensor, &output, line_size),
            linear_view(&output, line_size),
            dtypes,
        )
        .expect("Kernel to never fail");
    }

    output
}

pub fn is_nan<R: CubeRuntime>(tensor: CubeTensor<R>, dtype_bool: DType) -> CubeTensor<R> {
    launch_predicate::<R, IsNanOp>(tensor, dtype_bool)
}

pub fn is_inf<R: CubeRuntime>(tensor: CubeTensor<R>, dtype_bool: DType) -> CubeTensor<R> {
    launch_predicate::<R, IsInfOp>(tensor, dtype_bool)
}
