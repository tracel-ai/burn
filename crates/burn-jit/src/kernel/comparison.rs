use super::{index_offset_with_layout, index_offset_with_layout_expand, Kernel};
use crate::{element::JitElement, tensor::JitTensor, JitRuntime};
use burn_cube::{
    calculate_cube_count_elemwise, prelude::*, tensor_vectorization_factor, unexpanded, Runtime,
    SUBCUBE_DIM_APPROX,
};
use burn_tensor::Shape;

pub(crate) trait ComparisonOp<C: Numeric>: 'static + Send + Sync {
    /// Execute a comparison operation.
    #[allow(unused_variables)]
    fn execute(lhs: C, rhs: C) -> bool {
        unexpanded!();
    }
    fn execute_expand(
        context: &mut CubeContext,
        lhs: C::ExpandType,
        rhs: C::ExpandType,
    ) -> <bool as CubeType>::ExpandType;
}

#[cube(launch)]
pub(crate) fn kernel_scalar_cmp<C: Numeric, O: ComparisonOp<C>>(
    input: &Tensor<C>,
    scalar: C,
    output: &mut Tensor<UInt>,
) {
    let offset_output = ABSOLUTE_POS;

    if offset_output >= output.len() {
        return;
    }

    output[ABSOLUTE_POS] = UInt::cast_from(O::execute(input[ABSOLUTE_POS], scalar));
}

#[cube(launch)]
pub(crate) fn kernel_cmp<C: Numeric, O: ComparisonOp<C>>(
    lhs: &Tensor<C>,
    rhs: &Tensor<C>,
    out: &mut Tensor<UInt>,
    rank: Comptime<Option<UInt>>,
    to_contiguous_lhs: Comptime<bool>,
    to_contiguous_rhs: Comptime<bool>,
) {
    let offset_out = ABSOLUTE_POS;
    let mut offset_lhs = ABSOLUTE_POS;
    let mut offset_rhs = ABSOLUTE_POS;

    if offset_out >= out.len() {
        return;
    }

    if Comptime::get(to_contiguous_lhs) {
        offset_lhs = index_offset_with_layout::<C, UInt>(
            lhs,
            out,
            offset_out,
            UInt::new(0),
            Comptime::unwrap_or_else(rank, || out.rank()),
            Comptime::is_some(rank),
        );
    }

    if Comptime::get(to_contiguous_rhs) {
        offset_rhs = index_offset_with_layout::<C, UInt>(
            rhs,
            out,
            offset_out,
            UInt::new(0),
            Comptime::unwrap_or_else(rank, || out.rank()),
            Comptime::is_some(rank),
        );
    }

    out[offset_out] = UInt::cast_from(O::execute(lhs[offset_lhs], rhs[offset_rhs]));
}

pub(crate) fn launch_cmp<
    const D: usize,
    R: JitRuntime,
    E: JitElement,
    O: ComparisonOp<E::Primitive>,
>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
) -> JitTensor<R, u32, D> {
    let vectorization_factor_lhs =
        tensor_vectorization_factor(&[4, 2], &lhs.shape.dims, &lhs.strides);
    let vectorization_factor_rhs =
        tensor_vectorization_factor(&[4, 2], &lhs.shape.dims, &lhs.strides);

    let vectorization_factor = u8::min(vectorization_factor_lhs, vectorization_factor_rhs);

    let mut shape_out = [0; D];
    lhs.shape
        .dims
        .iter()
        .zip(rhs.shape.dims.iter())
        .enumerate()
        .for_each(|(index, (dim_lhs, dim_rhs))| {
            shape_out[index] = usize::max(*dim_lhs, *dim_rhs);
        });

    let shape_out = Shape::new(shape_out);
    let client = lhs.client.clone();
    let num_elems = shape_out.num_elements();
    let cube_count = calculate_cube_count_elemwise(
        num_elems / vectorization_factor as usize,
        SUBCUBE_DIM_APPROX,
    );

    let same_tensor_type = core::any::TypeId::of::<E>() == core::any::TypeId::of::<UInt>();
    if same_tensor_type && lhs.can_mut_broadcast(&rhs) {
        kernel_cmp_launch::<E::Primitive, O, R>(
            client,
            cube_count,
            CubeDim::default(),
            TensorArg::vectorized(
                vectorization_factor,
                &lhs.handle,
                &lhs.strides,
                &lhs.shape.dims,
            ),
            TensorArg::vectorized(
                vectorization_factor,
                &rhs.handle,
                &rhs.strides,
                &rhs.shape.dims,
            ),
            TensorArg::alias(0),
            Some(UInt::new(D as u32)),
            false,
            !rhs.is_contiguous(),
        );

        JitTensor::new(lhs.client, lhs.handle, lhs.shape, lhs.device, lhs.strides)
    } else if same_tensor_type && rhs.can_mut_broadcast(&lhs) {
        kernel_cmp_launch::<E::Primitive, O, R>(
            client,
            cube_count,
            CubeDim::default(),
            TensorArg::vectorized(
                vectorization_factor,
                &lhs.handle,
                &lhs.strides,
                &lhs.shape.dims,
            ),
            TensorArg::vectorized(
                vectorization_factor,
                &rhs.handle,
                &rhs.strides,
                &rhs.shape.dims,
            ),
            TensorArg::alias(1),
            Some(UInt::new(D as u32)),
            !lhs.is_contiguous(),
            false,
        );

        JitTensor::new(rhs.client, rhs.handle, rhs.shape, rhs.device, rhs.strides)
    } else {
        let buffer = lhs.client.empty(num_elems * core::mem::size_of::<E>());
        let to_contiguous_lhs = !lhs.is_contiguous();
        let to_contiguous_rhs = !rhs.is_contiguous();
        let output = JitTensor::new_contiguous(lhs.client.clone(), lhs.device, shape_out, buffer);

        kernel_cmp_launch::<E::Primitive, O, R>(
            client,
            cube_count,
            CubeDim::default(),
            TensorArg::vectorized(
                vectorization_factor,
                &lhs.handle,
                &lhs.strides,
                &lhs.shape.dims,
            ),
            TensorArg::vectorized(
                vectorization_factor,
                &rhs.handle,
                &rhs.strides,
                &rhs.shape.dims,
            ),
            TensorArg::vectorized(
                vectorization_factor,
                &output.handle,
                &output.strides,
                &output.shape.dims,
            ),
            Some(UInt::new(D as u32)),
            to_contiguous_lhs,
            to_contiguous_rhs,
        );

        output
    }
}

pub(crate) fn launch_scalar_cmp<
    const D: usize,
    R: JitRuntime,
    E: JitElement,
    O: ComparisonOp<E::Primitive>,
>(
    tensor: JitTensor<R, E, D>,
    scalar: E,
) -> JitTensor<R, u32, D> {
    // Vectorization is only enabled when the last dimension is contiguous.
    let vectorization_factor =
        tensor_vectorization_factor(&[4, 2], &tensor.shape.dims, &tensor.strides);
    let client = tensor.client.clone();
    let num_elems = tensor.shape.num_elements();
    let cube_count = calculate_cube_count_elemwise(
        num_elems / vectorization_factor as usize,
        SUBCUBE_DIM_APPROX,
    );

    let same_tensor_type = core::any::TypeId::of::<E>() == core::any::TypeId::of::<UInt>();
    if same_tensor_type && tensor.can_mut() {
        kernel_scalar_cmp_launch::<E::Primitive, O, R>(
            client,
            cube_count,
            CubeDim::default(),
            TensorArg::vectorized(
                vectorization_factor,
                &tensor.handle,
                &tensor.strides,
                &tensor.shape.dims,
            ),
            ScalarArg::new(scalar),
            TensorArg::alias(0),
        );

        JitTensor::new(
            tensor.client,
            tensor.handle,
            tensor.shape,
            tensor.device,
            tensor.strides,
        )
    } else {
        let buffer = tensor.client.empty(num_elems * core::mem::size_of::<E>());
        let output = JitTensor::new(
            tensor.client.clone(),
            buffer,
            tensor.shape.clone(),
            tensor.device,
            tensor.strides,
        );

        kernel_scalar_cmp_launch::<E::Primitive, O, R>(
            client,
            cube_count,
            CubeDim::default(),
            TensorArg::vectorized(
                vectorization_factor,
                &tensor.handle,
                &tensor.strides,
                &tensor.shape.dims,
            ),
            ScalarArg::new(scalar),
            TensorArg::vectorized(
                vectorization_factor,
                &output.handle,
                &output.strides,
                &output.shape.dims,
            ),
        );

        output
    }
}

struct EqualOp;
struct GreaterEqualOp;
struct LowerEqualOp;
struct GreaterOp;
struct LowerOp;

impl<N: Numeric> ComparisonOp<N> for EqualOp {
    fn execute_expand(
        context: &mut CubeContext,
        lhs: <N>::ExpandType,
        rhs: <N>::ExpandType,
    ) -> <bool as CubeType>::ExpandType {
        #[cube]
        fn cmp<N: Numeric>(lhs: N, rhs: N) -> bool {
            lhs == rhs
        }
        cmp_expand::<N>(context, lhs, rhs)
    }
}

impl<N: Numeric> ComparisonOp<N> for GreaterEqualOp {
    fn execute_expand(
        context: &mut CubeContext,
        lhs: <N>::ExpandType,
        rhs: <N>::ExpandType,
    ) -> <bool as CubeType>::ExpandType {
        #[cube]
        fn cmp<N: Numeric>(lhs: N, rhs: N) -> bool {
            lhs >= rhs
        }
        cmp_expand::<N>(context, lhs, rhs)
    }
}

impl<N: Numeric> ComparisonOp<N> for LowerEqualOp {
    fn execute_expand(
        context: &mut CubeContext,
        lhs: <N>::ExpandType,
        rhs: <N>::ExpandType,
    ) -> <bool as CubeType>::ExpandType {
        #[cube]
        fn cmp<N: Numeric>(lhs: N, rhs: N) -> bool {
            lhs <= rhs
        }
        cmp_expand::<N>(context, lhs, rhs)
    }
}

impl<N: Numeric> ComparisonOp<N> for GreaterOp {
    fn execute_expand(
        context: &mut CubeContext,
        lhs: <N>::ExpandType,
        rhs: <N>::ExpandType,
    ) -> <bool as CubeType>::ExpandType {
        #[cube]
        fn cmp<N: Numeric>(lhs: N, rhs: N) -> bool {
            lhs > rhs
        }
        cmp_expand::<N>(context, lhs, rhs)
    }
}

impl<N: Numeric> ComparisonOp<N> for LowerOp {
    fn execute_expand(
        context: &mut CubeContext,
        lhs: <N>::ExpandType,
        rhs: <N>::ExpandType,
    ) -> <bool as CubeType>::ExpandType {
        #[cube]
        fn cmp<N: Numeric>(lhs: N, rhs: N) -> bool {
            lhs < rhs
        }
        cmp_expand::<N>(context, lhs, rhs)
    }
}

pub fn equal<R: JitRuntime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
) -> JitTensor<R, u32, D> {
    launch_cmp::<D, R, E, EqualOp>(lhs, rhs)
}

pub fn greater<R: JitRuntime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
) -> JitTensor<R, u32, D> {
    launch_cmp::<D, R, E, GreaterOp>(lhs, rhs)
}

pub fn greater_equal<R: JitRuntime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
) -> JitTensor<R, u32, D> {
    launch_cmp::<D, R, E, GreaterEqualOp>(lhs, rhs)
}

pub fn lower<R: JitRuntime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
) -> JitTensor<R, u32, D> {
    launch_cmp::<D, R, E, LowerOp>(lhs, rhs)
}

pub fn lower_equal<R: JitRuntime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
) -> JitTensor<R, u32, D> {
    launch_cmp::<D, R, E, LowerEqualOp>(lhs, rhs)
}

pub fn equal_elem<R: JitRuntime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: E,
) -> JitTensor<R, u32, D> {
    launch_scalar_cmp::<D, R, E, EqualOp>(lhs, rhs)
}

pub fn greater_elem<R: JitRuntime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: E,
) -> JitTensor<R, u32, D> {
    launch_scalar_cmp::<D, R, E, GreaterOp>(lhs, rhs)
}

pub fn lower_elem<R: JitRuntime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: E,
) -> JitTensor<R, u32, D> {
    launch_scalar_cmp::<D, R, E, LowerOp>(lhs, rhs)
}

pub fn greater_equal_elem<R: JitRuntime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: E,
) -> JitTensor<R, u32, D> {
    launch_scalar_cmp::<D, R, E, GreaterEqualOp>(lhs, rhs)
}

pub fn lower_equal_elem<R: JitRuntime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: E,
) -> JitTensor<R, u32, D> {
    launch_scalar_cmp::<D, R, E, LowerEqualOp>(lhs, rhs)
}
