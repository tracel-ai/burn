use burn_backend::{cubecl::dtype_to_storage_type, ops::ConvOptions};
use burn_std::Shape;
use cubek::{
    convolution::{AcceleratedTileKind, ConvolutionArgs, components::ConvSetupError},
    matmul::definition::{MatmulElems, MatmulGlobalElems},
    std::InputBinding,
};

use crate::{ops::numeric::empty_device_dtype, tensor::CubeTensor};

/// Computes weights `[out_channels, ...kernel, in_channels]` from NHWC tensors of rank `N + 2`.
/// CubeK splits the contraction and combines F32 partial sums before casting to the output dtype.
pub(crate) fn wgrad_split_implicit<const N: usize, const SPATIAL_LAST: bool>(
    input: CubeTensor,
    out_grad: CubeTensor,
    weight_shape: Shape,
    options: ConvOptions<N>,
) -> Result<CubeTensor, ConvSetupError> {
    if options.groups != 1 {
        return Err(ConvSetupError::Groups(options.groups));
    }

    let input = crate::kernel::untile(input);
    let out_grad = crate::kernel::untile(out_grad);
    let output = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        weight_shape,
        out_grad.dtype,
    );
    let dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: dtype_to_storage_type(out_grad.dtype),
        rhs: dtype_to_storage_type(input.dtype),
        out: dtype_to_storage_type(output.dtype),
    });
    let client = input.client.clone();
    let input_dtype = input.dtype;
    let out_grad_dtype = out_grad.dtype;
    cubek::convolution::backward_weight::split::launch_accelerated::<N, SPATIAL_LAST>(
        &client,
        InputBinding::new(input.binding(), dtype_to_storage_type(input_dtype)),
        InputBinding::new(out_grad.binding(), dtype_to_storage_type(out_grad_dtype)),
        output.clone().binding(),
        ConvolutionArgs {
            stride: options.stride,
            padding: options.padding_begin(),
            dilation: options.dilation,
        },
        AcceleratedTileKind::Mma,
        dtypes,
    )?;
    Ok(output)
}
