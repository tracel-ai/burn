use std::marker::PhantomData;

use crate::{
    BoolElement, CubeRuntime,
    element::CubeElement,
    kernel::utils::{linear_tensor, linear_tensor_alias},
    ops::{into_data_sync, numeric::empty_device},
    tensor::CubeTensor,
};
use burn_tensor::Shape;
use cubecl::{
    calculate_cube_count_elemwise,
    prelude::*,
    std::tensor::{
        index_offset_with_layout, layout::linear::LinearTensorView, r#virtual::ReadWrite,
    },
    tensor_vectorization_factor,
};

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
    input: &Tensor<Line<SS::C>>,
    scalar: SS::C,
    output: &mut Tensor<Line<SS::B>>,
) {
    let offset_output = ABSOLUTE_POS;

    if offset_output >= output.len() {
        terminate!();
    }

    let index_input =
        index_offset_with_layout(input, output, offset_output, 0, output.rank(), false);

    output[offset_output] = Line::cast_from(O::execute(input[index_input], Line::new(scalar)));
}

#[cube(launch)]
pub(crate) fn kernel_cmp<SS: ScalarOpSpec, O: ComparisonOp<SS::C>>(
    lhs: &LinearTensorView<SS::C>,
    rhs: &LinearTensorView<SS::C>,
    out: &mut LinearTensorView<SS::B, ReadWrite>,
) {
    if ABSOLUTE_POS >= out.len() {
        terminate!();
    }

    out[ABSOLUTE_POS] = Line::cast_from(O::execute(lhs[ABSOLUTE_POS], rhs[ABSOLUTE_POS]));
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

    let line_size = Ord::min(vectorization_factor_lhs, vectorization_factor_rhs);

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

    println!("cmp lhs: {}", into_data_sync::<R, E>(lhs.clone()));
    println!("cmp rhs: {}", into_data_sync::<R, E>(rhs.clone()));

    let same_tensor_type = core::any::TypeId::of::<E>() == core::any::TypeId::of::<BT>();
    let out = if same_tensor_type && lhs.can_mut_broadcast(&rhs) {
        kernel_cmp::launch::<Spec<E, BT>, O, R>(
            &client,
            cube_count,
            cube_dim,
            linear_tensor(&lhs, &line_size),
            linear_tensor(&rhs, &line_size),
            linear_tensor_alias(&lhs, &line_size, 0),
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
            linear_tensor(&lhs, &line_size),
            linear_tensor(&rhs, &line_size),
            linear_tensor_alias(&lhs, &line_size, 1),
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
            linear_tensor(&lhs, &line_size),
            linear_tensor(&rhs, &line_size),
            linear_tensor(&output, &line_size),
        );

        output
    };
    println!("cmp out: {}", into_data_sync::<R, BT>(out.clone()));
    out
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
        unsafe {
            kernel_scalar_cmp::launch_unchecked::<Spec<E, BT>, O, R>(
                &client,
                cube_count,
                cube_dim,
                tensor.as_tensor_arg::<E>(vectorization_factor),
                ScalarArg::new(scalar),
                TensorArg::alias(0),
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
                tensor.as_tensor_arg::<E>(vectorization_factor),
                ScalarArg::new(scalar),
                output.as_tensor_arg::<BT>(vectorization_factor),
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
