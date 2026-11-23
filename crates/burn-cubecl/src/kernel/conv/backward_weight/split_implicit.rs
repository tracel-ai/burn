use burn_backend::{DType, cubecl::dtype_to_storage_type, ops::ConvOptions};
use burn_std::Shape;
use cubek::{
    convolution::{
        AcceleratedTileKind, ConvolutionArgs,
        backward_weight::split::{AcceleratedSplitStrategy, PartialColumnOrder},
        components::ConvSetupError,
    },
    matmul::definition::{MatmulElems, MatmulGlobalElems},
    std::InputBinding,
};

use crate::{ops::numeric::empty_device_dtype, tensor::CubeTensor};

use super::fallback::conv_weight_backward_fallback;

/// Computes weights `[out_channels, ...kernel, in_channels]` from NHWC tensors of rank `N + 2`.
/// CubeK splits the contraction and combines F32 partial sums before casting to the output dtype.
pub(crate) fn wgrad_split_implicit<const N: usize>(
    input: CubeTensor,
    out_grad: CubeTensor,
    weight_shape: Shape,
    options: ConvOptions<N>,
    column_order: PartialColumnOrder,
) -> Result<CubeTensor, ConvSetupError> {
    if options.groups != 1 {
        return Err(ConvSetupError::Groups(options.groups));
    }
    if ![input.dtype, out_grad.dtype]
        .into_iter()
        .all(|dtype| matches!(dtype, DType::F16 | DType::BF16 | DType::F32))
    {
        return Err(ConvSetupError::Unknown);
    }

    let input = crate::kernel::untile(input);
    let out_grad = crate::kernel::untile(out_grad);
    let output = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        weight_shape.clone(),
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
    let result = cubek::convolution::backward_weight::split::launch_accelerated::<N>(
        &client,
        InputBinding::new(input.clone().binding(), dtype_to_storage_type(input_dtype)),
        InputBinding::new(
            out_grad.clone().binding(),
            dtype_to_storage_type(out_grad_dtype),
        ),
        output.clone().binding(),
        ConvolutionArgs {
            stride: options.stride,
            padding: options.padding_begin(),
            dilation: options.dilation,
        },
        AcceleratedSplitStrategy {
            tile_kind: AcceleratedTileKind::Mma,
            column_order,
        },
        dtypes,
    );
    match result {
        Ok(()) => Ok(output),
        // CubeK reports workspace and size limits as Unknown for these dtypes.
        // A cached candidate must still serve shapes sharing an anchored key.
        Err(ConvSetupError::Unknown) => {
            drop(output);
            conv_weight_backward_fallback(input, out_grad, weight_shape, options)
        }
        Err(error) => Err(error),
    }
}
