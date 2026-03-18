use crate::{
    CubeRuntime,
    kernel::utils::address_type,
    ops::{max_vector_size, numeric::empty_device_dtype},
    tensor::CubeTensor,
};
use burn_backend::TensorMetadata;
use cubecl::{calculate_cube_count_elemwise, prelude::*, std::tensor::layout::linear::LinearView};

pub(crate) trait NumericUnaryOpFamily: 'static + Send + Sync {
    type Options: LaunchArg;
    type Unary<T: Numeric, N: Size>: NumericUnaryOp<T, N, Options = Self::Options>;
}

#[cube]
pub(crate) trait NumericUnaryOp<T: Scalar, N: Size>: 'static + Send + Sync {
    type Options: LaunchArg;

    fn execute(input: Vector<T, N>, options: &Self::Options) -> Vector<T, N>;
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub(crate) fn unary_numeric<T: Numeric, N: Size, O: NumericUnaryOpFamily>(
    input: &LinearView<Vector<T, N>>,
    output: &mut LinearView<Vector<T, N>, ReadWrite>,
    options: &O::Options,
    #[define(T)] _dtype: StorageType,
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    output[ABSOLUTE_POS] = O::Unary::<T, N>::execute(input[ABSOLUTE_POS], options);
}

pub(crate) fn launch_unary_numeric<R, O, Args>(tensor: CubeTensor<R>, args: Args) -> CubeTensor<R>
where
    // Magic fix for lifetime, the closure is supposed to capture everything required to create the
    // argument.
    for<'a> Args: FnOnce(&'a ()) -> RuntimeArg<O::Options, R>,
    R: CubeRuntime,
    O: NumericUnaryOpFamily,
{
    let vector_size = max_vector_size(&tensor);
    let client = tensor.client.clone();
    let num_elems = tensor.meta.num_elements();

    let working_units = num_elems / vector_size as usize;
    let cube_dim = CubeDim::new(&tensor.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&tensor.client, working_units, cube_dim);
    let dtype = tensor.dtype;

    unsafe {
        if tensor.can_mut() && tensor.is_nonoverlapping() {
            unary_numeric::launch_unchecked::<O, R>(
                &client,
                cube_count,
                cube_dim,
                address_type!(tensor),
                vector_size,
                tensor.clone().into_linear_view(),
                tensor.as_linear_view_alias(0),
                args(&()),
                dtype.into(),
            );

            tensor
        } else {
            let output = empty_device_dtype(
                tensor.client.clone(),
                tensor.device.clone(),
                tensor.shape(),
                tensor.dtype,
            );

            unary_numeric::launch_unchecked::<O, R>(
                &client,
                cube_count,
                cube_dim,
                address_type!(tensor, output),
                vector_size,
                tensor.into_linear_view(),
                output.clone().into_linear_view(),
                args(&()),
                dtype.into(),
            );

            output
        }
    }
}
