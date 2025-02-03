use crate::{element::JitElement, ops::numeric::empty_device, tensor::JitTensor, JitRuntime};
use cubecl::{
    calculate_cube_count_elemwise, linalg::tensor::index_offset_with_layout, prelude::*,
    tensor_line_size_parallel,
};

pub(crate) trait NumericUnaryOpFamily: 'static + Send + Sync {
    type Options<N: Numeric>: LaunchArg;
    type Unary<N: Numeric>: NumericUnaryOp<N, Options = Self::Options<N>>;
}

#[cube]
pub(crate) trait NumericUnaryOp<N: CubePrimitive>: 'static + Send + Sync {
    type Options: LaunchArg;

    fn execute(input: Line<N>, options: &Self::Options) -> Line<N>;
}

#[cube(launch_unchecked)]
pub(crate) fn unary_numeric<N: Numeric, O: NumericUnaryOpFamily>(
    input: &Tensor<Line<N>>,
    output: &mut Tensor<Line<N>>,
    options: &O::Options<N>,
    #[comptime] rank: Option<u32>,
    #[comptime] to_contiguous: bool,
) {
    let offset_output = ABSOLUTE_POS;

    if offset_output >= output.len() {
        terminate!();
    }

    if comptime![to_contiguous] {
        let offset_input = index_offset_with_layout::<N, N>(
            input,
            output,
            offset_output,
            0,
            rank.unwrap_or_else(|| output.rank()),
            rank.is_some(),
        );

        output[offset_output] = O::Unary::<N>::execute(input[offset_input], options);
    } else {
        output[offset_output] = O::Unary::<N>::execute(input[offset_output], options);
    }
}

pub(crate) fn launch_unary_numeric<R, E, O, Args>(tensor: JitTensor<R>, args: Args) -> JitTensor<R>
where
    // Magic fix for lifetime, the closure is supposed to capture everything required to create the
    // argument.
    for<'a> Args: FnOnce(&'a ()) -> RuntimeArg<'a, O::Options<E>, R>,
    R: JitRuntime,
    E: JitElement + Numeric,
    O: NumericUnaryOpFamily,
{
    let ndims = tensor.shape.num_dims();
    let line_size = tensor_line_size_parallel(
        R::line_size_elem(&E::as_elem_native_unchecked()),
        &tensor.shape.dims,
        &tensor.strides,
        ndims - 1,
    );
    let client = tensor.client.clone();
    let num_elems = tensor.shape.num_elements();

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems / line_size as usize, cube_dim);
    let is_contiguous = tensor.is_contiguous();

    unsafe {
        if tensor.can_mut() && tensor.is_contiguous_buffer() {
            unary_numeric::launch_unchecked::<E, O, R>(
                &client,
                cube_count,
                cube_dim,
                tensor.as_tensor_arg::<E>(line_size),
                TensorArg::alias(0),
                args(&()),
                None,
                false,
            );

            tensor
        } else {
            let output = empty_device::<R, E>(
                tensor.client.clone(),
                tensor.device.clone(),
                tensor.shape.clone(),
            );

            unary_numeric::launch_unchecked::<E, O, R>(
                &client,
                cube_count,
                CubeDim::default(),
                tensor.as_tensor_arg::<E>(line_size),
                output.as_tensor_arg::<E>(line_size),
                args(&()),
                Some(ndims as u32),
                !is_contiguous,
            );
            output
        }
    }
}
