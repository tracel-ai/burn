use crate::{
    CubeRuntime,
    kernel::utils::{address_type, broadcast_shape},
    ops::{max_vector_size, numeric::empty_device_dtype},
    tensor::CubeTensor,
};
use burn_backend::{DType, TensorMetadata};
use cubecl::{calculate_cube_count_elemwise, prelude::*, std::tensor::layout::linear::LinearView};

#[cube]
pub(crate) trait ComparisonOpFamily: 'static + Send + Sync {
    type Operation<T: Numeric, N: Size>: ComparisonOp<T, N>;
}

#[cube]
pub(crate) trait ComparisonOp<C: Numeric, N: Size>: 'static + Send + Sync {
    /// Execute a comparison operation.
    fn execute(lhs: Vector<C, N>, rhs: Vector<C, N>) -> bool;
}

struct EqualOp;
struct GreaterEqualOp;
struct LowerEqualOp;
struct GreaterOp;
struct LowerOp;

impl ComparisonOpFamily for EqualOp {
    type Operation<T: Numeric, N: Size> = Self;
}

#[cube]
impl<T: Numeric, N: Size> ComparisonOp<T, N> for EqualOp {
    fn execute(lhs: Vector<T, N>, rhs: Vector<T, N>) -> bool {
        lhs == rhs
    }
}

impl ComparisonOpFamily for GreaterEqualOp {
    type Operation<T: Numeric, N: Size> = Self;
}

#[cube]
impl<T: Numeric, N: Size> ComparisonOp<T, N> for GreaterEqualOp {
    fn execute(lhs: Vector<T, N>, rhs: Vector<T, N>) -> bool {
        lhs >= rhs
    }
}

impl ComparisonOpFamily for LowerEqualOp {
    type Operation<T: Numeric, N: Size> = Self;
}

#[cube]
impl<T: Numeric, N: Size> ComparisonOp<T, N> for LowerEqualOp {
    fn execute(lhs: Vector<T, N>, rhs: Vector<T, N>) -> bool {
        lhs <= rhs
    }
}

impl ComparisonOpFamily for GreaterOp {
    type Operation<T: Numeric, N: Size> = Self;
}

#[cube]
impl<T: Numeric, N: Size> ComparisonOp<T, N> for GreaterOp {
    fn execute(lhs: Vector<T, N>, rhs: Vector<T, N>) -> bool {
        lhs > rhs
    }
}

impl ComparisonOpFamily for LowerOp {
    type Operation<T: Numeric, N: Size> = Self;
}

#[cube]
impl<T: Numeric, N: Size> ComparisonOp<T, N> for LowerOp {
    fn execute(lhs: Vector<T, N>, rhs: Vector<T, N>) -> bool {
        lhs < rhs
    }
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub(crate) fn kernel_scalar_cmp<T: Numeric, Bool: Numeric, N: Size, O: ComparisonOpFamily>(
    input: &LinearView<Vector<T, N>>,
    scalar: InputScalar,
    output: &mut LinearView<Vector<Bool, N>, ReadWrite>,
    #[define(T, Bool)] _dtypes: [StorageType; 2],
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    output[ABSOLUTE_POS] = Vector::cast_from(O::Operation::<T, N>::execute(
        input[ABSOLUTE_POS],
        Vector::new(scalar.get::<T>()),
    ));
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub(crate) fn kernel_cmp<T: Numeric, Bool: Numeric, N: Size, O: ComparisonOpFamily>(
    lhs: &LinearView<Vector<T, N>>,
    rhs: &LinearView<Vector<T, N>>,
    out: &mut LinearView<Vector<Bool, N>, ReadWrite>,
    #[define(T, Bool)] _dtype: [StorageType; 2],
) {
    if !out.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    out[ABSOLUTE_POS] = Vector::cast_from(O::Operation::<T, N>::execute(
        lhs[ABSOLUTE_POS],
        rhs[ABSOLUTE_POS],
    ));
}

pub(crate) fn launch_cmp<R: CubeRuntime, O: ComparisonOpFamily>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    dtype_bool: DType,
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

    let dtypes = [lhs.dtype.into(), dtype_bool.into()];
    let same_tensor_type = dtypes[0] == dtypes[1];
    if same_tensor_type && lhs.can_mut_broadcast(&rhs) {
        unsafe {
            kernel_cmp::launch_unchecked::<O, R>(
                &client,
                cube_count,
                cube_dim,
                address_type!(lhs, rhs),
                vector_size,
                lhs.clone().into_linear_view(),
                rhs.into_linear_view_like(&lhs),
                lhs.as_linear_view_alias(0),
                dtypes,
            );
        }

        CubeTensor::new(
            lhs.client.clone(),
            lhs.handle.clone(),
            *lhs.meta.clone(),
            lhs.device.clone(),
            dtype_bool,
        )
    } else if same_tensor_type && rhs.can_mut_broadcast(&lhs) {
        unsafe {
            kernel_cmp::launch_unchecked::<O, R>(
                &client,
                cube_count,
                cube_dim,
                address_type!(lhs, rhs),
                vector_size,
                lhs.into_linear_view_like(&rhs),
                rhs.clone().into_linear_view(),
                rhs.as_linear_view_alias(1),
                dtypes,
            );
        };

        CubeTensor::new(
            rhs.client.clone(),
            rhs.handle.clone(),
            *rhs.meta.clone(),
            rhs.device.clone(),
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
                cube_dim,
                address_type!(lhs, rhs, output),
                vector_size,
                lhs.into_linear_view_like(&output),
                rhs.into_linear_view_like(&output),
                output.clone().into_linear_view(),
                dtypes,
            );
        };

        output
    }
}

pub(crate) fn launch_scalar_cmp<R: CubeRuntime, O: ComparisonOpFamily>(
    tensor: CubeTensor<R>,
    scalar: InputScalar,
    dtype_bool: DType,
) -> CubeTensor<R> {
    let vector_size = max_vector_size(&tensor);
    let client = tensor.client.clone();
    let num_elems = tensor.meta.num_elements();

    let working_units = num_elems / vector_size as usize;
    let cube_dim = CubeDim::new(&tensor.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&tensor.client, working_units, cube_dim);

    let dtypes = [tensor.dtype.into(), dtype_bool.into()];
    let same_tensor_type = dtypes[0] == dtypes[1];

    if same_tensor_type && tensor.can_mut() && tensor.is_nonoverlapping() {
        unsafe {
            kernel_scalar_cmp::launch_unchecked::<O, R>(
                &client,
                cube_count,
                cube_dim,
                address_type!(tensor),
                vector_size,
                tensor.clone().into_linear_view(),
                scalar,
                tensor.as_linear_view_alias(0),
                dtypes,
            );
        }

        CubeTensor::new(
            tensor.client.clone(),
            tensor.handle.clone(),
            *tensor.meta.clone(),
            tensor.device.clone(),
            dtype_bool,
        )
    } else {
        let output = empty_device_dtype(
            tensor.client.clone(),
            tensor.device.clone(),
            tensor.shape(),
            dtype_bool,
        );

        unsafe {
            kernel_scalar_cmp::launch_unchecked::<O, R>(
                &client,
                cube_count,
                cube_dim,
                address_type!(tensor, output),
                vector_size,
                tensor.into_linear_view(),
                scalar,
                output.clone().into_linear_view(),
                dtypes,
            );
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
pub(crate) trait PredicateOp<F: Float, N: Size>: 'static + Send + Sync {
    /// Execute a predicate operation.
    fn execute(input: Vector<F, N>) -> Vector<bool, N>;
}

pub(crate) trait PredicateOpFamily: 'static + Send + Sync {
    type Operation<F: Float, N: Size>: PredicateOp<F, N>;
}

struct IsNanOp;
struct IsInfOp;

impl PredicateOpFamily for IsNanOp {
    type Operation<F: Float, N: Size> = Self;
}

#[cube]
impl<F: Float, N: Size> PredicateOp<F, N> for IsNanOp {
    fn execute(input: Vector<F, N>) -> Vector<bool, N> {
        Vector::is_nan(input)
    }
}

impl PredicateOpFamily for IsInfOp {
    type Operation<F: Float, N: Size> = Self;
}
#[cube]
impl<F: Float, N: Size> PredicateOp<F, N> for IsInfOp {
    fn execute(input: Vector<F, N>) -> Vector<bool, N> {
        Vector::is_inf(input)
    }
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub(crate) fn kernel_predicate<F: Float, Bool: Numeric, N: Size, O: PredicateOpFamily>(
    input: &LinearView<Vector<F, N>>,
    output: &mut LinearView<Vector<Bool, N>, ReadWrite>,
    #[define(F, Bool)] _dtypes: [StorageType; 2],
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    output[ABSOLUTE_POS] = Vector::cast_from(O::Operation::<F, N>::execute(input[ABSOLUTE_POS]));
}

pub(crate) fn launch_predicate<R: CubeRuntime, O: PredicateOpFamily>(
    tensor: CubeTensor<R>,
    dtype_bool: DType,
) -> CubeTensor<R> {
    let vector_size = max_vector_size(&tensor);

    let client = tensor.client.clone();
    let num_elems = tensor.meta.num_elements();

    let dtypes = [tensor.dtype.into(), dtype_bool.into()];
    let working_units = num_elems / vector_size as usize;
    let cube_dim = CubeDim::new(&tensor.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&tensor.client, working_units, cube_dim);

    let output = empty_device_dtype(
        tensor.client.clone(),
        tensor.device.clone(),
        tensor.shape(),
        dtype_bool,
    );

    unsafe {
        kernel_predicate::launch_unchecked::<O, R>(
            &client,
            cube_count,
            cube_dim,
            address_type!(tensor, output),
            vector_size,
            tensor.into_linear_view_like(&output),
            output.clone().into_linear_view(),
            dtypes,
        );
    }

    output
}

pub fn is_nan<R: CubeRuntime>(tensor: CubeTensor<R>, dtype_bool: DType) -> CubeTensor<R> {
    launch_predicate::<R, IsNanOp>(tensor, dtype_bool)
}

pub fn is_inf<R: CubeRuntime>(tensor: CubeTensor<R>, dtype_bool: DType) -> CubeTensor<R> {
    launch_predicate::<R, IsInfOp>(tensor, dtype_bool)
}
