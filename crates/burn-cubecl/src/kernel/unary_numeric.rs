use crate::{
    CubeRuntime,
    kernel::utils::{linear_view, linear_view_alias},
    ops::{max_line_size, numeric::empty_device_dtype},
    tensor::CubeTensor,
};
use cubecl::{calculate_cube_count_elemwise, prelude::*, std::tensor::layout::linear::LinearView};

pub(crate) trait NumericUnaryOpFamily: 'static + Send + Sync {
    type Options: LaunchArg;
    type Unary<N: Numeric>: NumericUnaryOp<N, Options = Self::Options>;
}

#[cube]
pub(crate) trait NumericUnaryOp<N: CubePrimitive>: 'static + Send + Sync {
    type Options: LaunchArg;

    fn execute(input: Line<N>, options: &Self::Options) -> Line<N>;
}

#[cube(launch_unchecked)]
pub(crate) fn unary_numeric<N: Numeric, O: NumericUnaryOpFamily>(
    input: &LinearView<Line<N>>,
    output: &mut LinearView<Line<N>, ReadWrite>,
    options: &O::Options,
    #[define(N)] _dtype: StorageType,
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    output[ABSOLUTE_POS] = O::Unary::<N>::execute(input[ABSOLUTE_POS], options);
}

pub(crate) fn launch_unary_numeric<R, O, Args>(tensor: CubeTensor<R>, args: Args) -> CubeTensor<R>
where
    // Magic fix for lifetime, the closure is supposed to capture everything required to create the
    // argument.
    for<'a> Args: FnOnce(&'a ()) -> RuntimeArg<'a, O::Options, R>,
    R: CubeRuntime,
    O: NumericUnaryOpFamily,
{
    let line_size = max_line_size(&tensor);
    let client = tensor.client.clone();
    let num_elems = tensor.shape.num_elements();

    let working_units = num_elems / line_size as usize;
    let cube_dim = CubeDim::new(&tensor.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&tensor.client, working_units, cube_dim);

    unsafe {
        if tensor.can_mut() && tensor.is_contiguous_buffer() {
            unary_numeric::launch_unchecked::<O, R>(
                &client,
                cube_count,
                cube_dim,
                linear_view(&tensor, line_size),
                linear_view_alias(&tensor, line_size, 0),
                args(&()),
                tensor.dtype.into(),
            )
            .expect("Kernel to never fail");

            tensor
        } else {
            let output = empty_device_dtype(
                tensor.client.clone(),
                tensor.device.clone(),
                tensor.shape.clone(),
                tensor.dtype,
            );

            unary_numeric::launch_unchecked::<O, R>(
                &client,
                cube_count,
                cube_dim,
                linear_view(&tensor, line_size),
                linear_view(&output, line_size),
                args(&()),
                tensor.dtype.into(),
            )
            .expect("Kernel to never fail");

            output
        }
    }
}
