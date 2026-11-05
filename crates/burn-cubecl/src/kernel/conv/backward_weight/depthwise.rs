use burn_backend::{DType, cubecl::dtype_to_storage_type, ops::ConvOptions};
use burn_std::Shape;
use cubek::convolution::{
    ConvolutionArgs, DepthwiseWeightBackwardTensors, components::ConvSetupError,
    launch_depthwise_weight_backward,
};

use crate::{ops::numeric::empty_device_dtype, tensor::CubeTensor};

/// Spatially split F32 depthwise weight gradients. Unsupported problems stay
/// on the existing fallback or accelerated candidates.
pub fn wgrad_depthwise<const N: usize>(
    input: CubeTensor,
    out_grad: CubeTensor,
    weight_shape: Shape,
    options: ConvOptions<N>,
) -> Result<CubeTensor, ConvSetupError> {
    if N != 2
        || input.dtype != DType::F32
        || options.groups != input.meta.shape()[N + 1]
        || weight_shape[N + 1] != 1
    {
        return Err(ConvSetupError::Unknown);
    }
    let weight_grad = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        weight_shape,
        input.dtype,
    );
    let padding = options.padding_begin();
    let client = input.client.clone();
    let dtype = input.dtype;
    launch_depthwise_weight_backward(
        &client,
        DepthwiseWeightBackwardTensors {
            input: input.binding(),
            out_grad: out_grad.binding(),
            weight_grad: weight_grad.clone().binding(),
        },
        ConvolutionArgs {
            stride: [options.stride[0], options.stride[1]],
            padding: [padding[0], padding[1]],
            dilation: [options.dilation[0], options.dilation[1]],
        },
        options.groups,
        dtype_to_storage_type(dtype),
    )?;
    Ok(weight_grad)
}
