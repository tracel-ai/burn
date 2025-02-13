use std::marker::PhantomData;

use crate::{
    element::CubeElement, ops::numeric::empty_device, tensor::CubeTensor, BoolElement, CubeRuntime,
};
use burn_tensor::Shape;
use cubecl::{
    calculate_cube_count_elemwise, linalg::tensor::index_offset_with_layout, prelude::*,
    tensor_vectorization_factor,
};

use super::into_contiguous;

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

#[cube(launch)]
pub(crate) fn kernel_scalar_cmp<SS: ScalarOpSpec, O: ComparisonOp<SS::C>>(
    input: &Tensor<Line<SS::C>>,
    scalar: SS::C,
    output: &mut Tensor<Line<SS::B>>,
) {
    let offset_output = ABSOLUTE_POS;

    if offset_output >= output.len() {
        terminate!();
    }

    output[ABSOLUTE_POS] = Line::cast_from(O::execute(input[ABSOLUTE_POS], Line::new(scalar)));
}

#[cube(launch)]
pub(crate) fn kernel_cmp<SS: ScalarOpSpec, O: ComparisonOp<SS::C>>(
    lhs: &Tensor<Line<SS::C>>,
    rhs: &Tensor<Line<SS::C>>,
    out: &mut Tensor<Line<SS::B>>,
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
        offset_lhs = index_offset_with_layout::<SS::C, SS::B>(
            lhs,
            out,
            offset_out,
            0,
            rank.unwrap_or_else(|| out.rank()),
            rank.is_some(),
        );
    }

    if to_contiguous_rhs {
        offset_rhs = index_offset_with_layout::<SS::C, SS::B>(
            rhs,
            out,
            offset_out,
            0,
            rank.unwrap_or_else(|| out.rank()),
            rank.is_some(),
        );
    }

    out[offset_out] = Line::cast_from(O::execute(lhs[offset_lhs], rhs[offset_rhs]));
}

pub(crate) fn launch_cmp<R: CubeRuntime, E: CubeElement, BT: BoolElement, O: ComparisonOp<E>>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
) -> CubeTensor<R> {
    let ndims = lhs.shape.num_dims();
    let vectorization_factor_lhs =
        tensor_vectorization_factor(&[4, 2], &lhs.shape.dims, &lhs.strides, ndims - 1);
    let vectorization_factor_rhs =
        tensor_vectorization_factor(&[4, 2], &rhs.shape.dims, &rhs.strides, ndims - 1);

    let vectorization_factor = Ord::min(vectorization_factor_lhs, vectorization_factor_rhs);

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
    let cube_count =
        calculate_cube_count_elemwise(num_elems / vectorization_factor as usize, cube_dim);

    let same_tensor_type = core::any::TypeId::of::<E>() == core::any::TypeId::of::<BT>();
    if same_tensor_type && lhs.can_mut_broadcast(&rhs) {
        kernel_cmp::launch::<Spec<E, BT>, O, R>(
            &client,
            cube_count,
            cube_dim,
            lhs.as_tensor_arg::<E>(vectorization_factor),
            rhs.as_tensor_arg::<E>(vectorization_factor),
            TensorArg::alias(0),
            None,
            false,
            rhs.strides != lhs.strides || rhs.shape != lhs.shape,
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
            lhs.as_tensor_arg::<E>(vectorization_factor),
            rhs.as_tensor_arg::<E>(vectorization_factor),
            TensorArg::alias(1),
            None,
            rhs.strides != lhs.strides || rhs.shape != lhs.shape,
            false,
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
        let to_contiguous_lhs = lhs.strides != output.strides || lhs.shape != output.shape;
        let to_contiguous_rhs = rhs.strides != output.strides || rhs.shape != output.shape;

        kernel_cmp::launch::<Spec<E, BT>, O, R>(
            &client,
            cube_count,
            CubeDim::default(),
            lhs.as_tensor_arg::<E>(vectorization_factor),
            rhs.as_tensor_arg::<E>(vectorization_factor),
            output.as_tensor_arg::<BT>(vectorization_factor),
            None,
            to_contiguous_lhs,
            to_contiguous_rhs,
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
    mut tensor: CubeTensor<R>,
    scalar: E,
) -> CubeTensor<R> {
    if !tensor.is_contiguous_buffer() {
        tensor = into_contiguous(tensor);
    }

    let ndims = tensor.shape.num_dims();
    // Vectorization is only enabled when the last dimension is contiguous.
    let vectorization_factor =
        tensor_vectorization_factor(&[4, 2], &tensor.shape.dims, &tensor.strides, ndims - 1);
    let client = tensor.client.clone();
    let num_elems = tensor.shape.num_elements();

    let cube_dim = CubeDim::default();
    let cube_count =
        calculate_cube_count_elemwise(num_elems / vectorization_factor as usize, cube_dim);

    let same_tensor_type = core::any::TypeId::of::<E>() == core::any::TypeId::of::<BT>();
    if same_tensor_type && tensor.can_mut() {
        kernel_scalar_cmp::launch::<Spec<E, BT>, O, R>(
            &client,
            cube_count,
            cube_dim,
            tensor.as_tensor_arg::<E>(vectorization_factor),
            ScalarArg::new(scalar),
            TensorArg::alias(0),
        );

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

        kernel_scalar_cmp::launch::<Spec<E, BT>, O, R>(
            &client,
            cube_count,
            CubeDim::default(),
            tensor.as_tensor_arg::<E>(vectorization_factor),
            ScalarArg::new(scalar),
            output.as_tensor_arg::<BT>(vectorization_factor),
        );

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
