use crate::{element::JitElement, tensor::JitTensor, JitRuntime};
use cubecl::{
    calculate_cube_count_elemwise, linalg::tensor::index_offset_with_layout, prelude::*,
    tensor_vectorization_factor, unexpanded,
};

use super::Kernel;

pub(crate) trait UnaryOp<C: CubePrimitive>: 'static + Send + Sync {
    type Options: LaunchArg;

    /// Execute a unary operation.
    fn execute(_input: C, _options: &Self::Options) -> C {
        unexpanded!();
    }
    fn __expand_execute(
        context: &mut CubeContext,
        input: C::ExpandType,
        options: <Self::Options as CubeType>::ExpandType,
    ) -> C::ExpandType;
}

#[cube(launch)]
pub(crate) fn unary_kernel<C: CubePrimitive, O: UnaryOp<C>>(
    input: &Tensor<C>,
    output: &mut Tensor<C>,
    options: &O::Options,
    rank: Comptime<Option<UInt>>,
    to_contiguous: Comptime<bool>,
) {
    let offset_output = ABSOLUTE_POS;

    if offset_output >= output.len() {
        return;
    }

    if Comptime::get(to_contiguous) {
        let offset_input = index_offset_with_layout::<C, C>(
            input,
            output,
            offset_output,
            UInt::new(0),
            Comptime::unwrap_or_else(rank, || output.rank()),
            Comptime::is_some(rank),
        );

        output[offset_output] = O::execute(input[offset_input], options);
    } else {
        output[offset_output] = O::execute(input[offset_output], options);
    }
}

pub(crate) fn launch_unary<
    const D: usize,
    R: JitRuntime,
    E: JitElement,
    O: UnaryOp<E::Primitive>,
    F,
>(
    tensor: JitTensor<R, E, D>,
    options: F,
) -> JitTensor<R, E, D>
where
    // Magic fix for lifetime, the closure is supposed to capture everything required to create the
    // argument.
    for<'a> F: FnOnce(&'a ()) -> RuntimeArg<'a, O::Options, R>,
{
    // Vectorization is only enabled when the last dimension is contiguous.
    let vectorization_factor =
        tensor_vectorization_factor(&[4, 2], &tensor.shape.dims, &tensor.strides, D - 1);

    let client = tensor.client.clone();
    let num_elems = tensor.shape.num_elements();

    let cube_dim = CubeDim::default();
    let cube_count =
        calculate_cube_count_elemwise(num_elems / vectorization_factor as usize, cube_dim);
    let is_contiguous = tensor.is_contiguous();

    if tensor.can_mut() && is_contiguous {
        unary_kernel::launch::<E::Primitive, O, R>(
            &client,
            cube_count,
            cube_dim,
            TensorArg::vectorized(
                vectorization_factor,
                &tensor.handle,
                &tensor.strides,
                &tensor.shape.dims,
            ),
            TensorArg::alias(0),
            options(&()),
            None,
            false,
        );

        tensor
    } else {
        let buffer = tensor.client.empty(num_elems * core::mem::size_of::<E>());
        let output = JitTensor::new_contiguous(
            tensor.client.clone(),
            tensor.device,
            tensor.shape.clone(),
            buffer,
        );

        unary_kernel::launch::<E::Primitive, O, R>(
            &client,
            cube_count,
            CubeDim::default(),
            TensorArg::vectorized(
                vectorization_factor,
                &tensor.handle,
                &tensor.strides,
                &tensor.shape.dims,
            ),
            TensorArg::vectorized(
                vectorization_factor,
                &output.handle,
                &output.strides,
                &output.shape.dims,
            ),
            options(&()),
            Some(UInt::new(D as u32)),
            !is_contiguous,
        );
        output
    }
}

macro_rules! unary_op {
    ($name:ident, $elem:ident, $expand:expr) => {
        struct $name;

        impl<C: $elem> UnaryOp<C> for $name {
            type Options = ();

            #[allow(clippy::redundant_closure_call)]
            fn __expand_execute(
                context: &mut CubeContext,
                input: C::ExpandType,
                _options: <Self::Options as CubeType>::ExpandType,
            ) -> C::ExpandType {
                $expand(context, input)
            }
        }
    };
    (scalar $name:ident, $elem:ident, $expand:expr) => {
        struct $name;

        impl<C: $elem> UnaryOp<C> for $name {
            type Options = C;

            #[allow(clippy::redundant_closure_call)]
            fn __expand_execute(
                context: &mut CubeContext,
                input: C::ExpandType,
                scalar: C::ExpandType,
            ) -> C::ExpandType {
                $expand(context, input, scalar)
            }
        }
    };
    (float($tensor:expr) => $exp:expr) => {{
        unary_op!(Op, Float, $exp);
        launch_unary::<D, R, F, Op, _>($tensor, |_| ())
    }};
    (int($tensor:expr) => $exp:expr) => {{
        unary_op!(Op, Numeric, $exp);
        launch_unary::<D, R, I, Op, _>($tensor, |_| ())
    }};
    (numeric($tensor:expr) => $exp:expr) => {{
        unary_op!(Op, Numeric, $exp);
        launch_unary::<D, R, E, Op, _>($tensor, |_| ())
    }};
    (numeric($tensor:expr, $scalar:expr) => $exp:expr) => {{
        unary_op!(scalar Op, Numeric, $exp);
        launch_unary::<D, R, E, Op, _>($tensor, |_| ScalarArg::new($scalar))
    }};
    (float($tensor:expr, $scalar:expr) => $exp:expr) => {{
        unary_op!(scalar Op, Float, $exp);
        launch_unary::<D, R, F, Op, _>($tensor, |_| ScalarArg::new($scalar))
    }};
}

pub(crate) use unary_op;
